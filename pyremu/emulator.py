#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 17:00:47
# Last modified at 2026/06/09 星期二

"""
多核 RISC-V 模拟器顶层 — 管理 Bus、CLINT、外设、多个 Hart 的执行循环。

上电后 hart 从可配置的复位向量 (reset vector) 开始执行。
裸金属固件应被直接加载到复位向量对应的物理地址。

平台拓扑通过 PlatformConfig 描述, 外设按配置自动创建并注册。
可选生成 FDT blob (通过 pyfdt) 供固件以 a1 寄存器接收。

Usage:
    from pyremu.platform import PlatformConfig

    cfg = PlatformConfig.sifive_u54()
    emu = Emulator(cfg)
    emu.load_code(addr=0x80000000, code=my_firmware)
    emu.step()        # 所有 hart 各执行一条指令
    emu.run(1000)     # 执行 1000 个周期
"""

import ctypes
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any

from pyremu._native import (
    ClintInfo,
    DevInfo,
    icount_flush,
    native_available,
    PmpInfo,
    run_parallel,
    UartInfo,
    VirtIOInfo,
)
from pyremu.core.decoder import Hart
from pyremu.core.hart import (
    BatchResult,
    EXIT_BREAKPOINT,
    EXIT_ECALL,
    EXIT_MMIO,
    EXIT_TRAP,
    HartState,
    marshal_hart,
    unmarshal_hart,
)
from pyremu.core.mem_check_aux import (
    MemoryAccessFault, check_instruction_fetch, inject_memory_backend,
)
from pyremu.core.registers import gpr_alias, gpr_name
from pyremu.core.trap_def import TrapType
from pyremu.core.trap_handler import check_pending_interrupts, deliver_trap, try_wfi_wakeup
from pyremu.core.watchdog import HartStallWatchdog
from pyremu.interrupt.clint import CLINT
from pyremu.interrupt.plic import PLIC
from pyremu.memory.bus import Bus
from pyremu.memory.l2cache import L2Cache
from pyremu.peripheral import GPIO, I2C, SPI, TerminalIO, UART, VirtIOBlock
from pyremu.peripheral.virtio_blk import VIRTIO_BLK_IRQ
from pyremu.peripheral.uart import UART_IRQ
from pyremu.peripheral.watchdog import HartWatchdog
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils import dtb
from pyremu.utils.parse_bin import FirmwareImage
from pyremu.utils.tick import yield_cpu


def _wrap_phy_write_for_uart(
    hart_id: int,
    uart,  # UART | None
    orig_write,  # (pa: int, data: bytes) -> None
):
    """包装物理写回调: hart 写入 UART MMIO 范围时自动标记写者身份.

    UART 据此将 TXDATA 字节归入 hart 日志文件 (set_hart_log_dir).
    不做行缓冲 — 仅用于多 hart 输出分流归档.
    """
    if uart is None:
        return orig_write
    uart_base = uart.base_addr
    uart_end = uart.base_addr + uart.size

    def _write(pa: int, data: bytes) -> None:
        if uart_base <= pa < uart_end:
            uart._current_writer = hart_id
        return orig_write(pa, data)

    return _write



# virtio 单次 _process_queue 最多处理描述符数,
# 拆批以确保 Ctrl+C 可在批次间响应 (每批 ~1ms).
_VIRTIO_PROCESS_BATCH: int = 16

# 阈值判断: ≤2 MiB 的段走 L2 缓存, 更大的段 (如 Linux 内核 Image)
# 绕过 L2 直写 RAM, 避免 cache-line 级逐片处理膨胀到数十秒.
_FAST_LOAD_THRESHOLD = 2 * 1024 * 1024  # 2 MiB


class Emulator:
    """多核 RISC-V 模拟器.

    管理 N 个 hart, 共享总线、中断控制器和外设.
    提供 step / run 执行循环及状态检查辅助方法.
    """

    @staticmethod
    def _build_default_config(**kwargs: int) -> PlatformConfig:
        """从旧式 API 的 **kwargs 构建 PlatformConfig; 无参数时用 qemu_virt 预设."""
        if not kwargs:
            return PlatformConfig.qemu_virt()
        periph_kw = {"clint_base": 0x0200_0000}
        plat_kw: dict[str, Any] = {"periph": PeripheralConfig(**periph_kw)}
        for field_name in PlatformConfig.__dataclass_fields__:
            if field_name in kwargs:
                plat_kw[field_name] = kwargs[field_name]
        return PlatformConfig(**plat_kw)

    def __init__(
        self,
        config: PlatformConfig | None = None,
        bootargs: str | None = None,
        **kwargs: int,
    ) -> None:
        """根据 *config* 初始化模拟器.

        兼容旧式 API: ``Emulator(num_harts=4, ram_base=0, ...)`` 等价于
        用对应字段构建 Minimal PlatformConfig.
        *config* 传入时忽略 **kwargs.
        """
        if config is None:
            config = self._build_default_config(**kwargs)
        self._cfg = config
        # 默认 bootargs: 内核控制台输出 + 信任 bootloader 随机数种子.
        # 用户可通过 bootargs=None 显式禁用或传入自定义 bootargs 覆盖.
        # 调试用途 (日后需要详细内核日志时取消下面注释即可):
        #   "printk.devkmsg=on loglevel=8 debug ignore_loglevel "
        #   "dyndbg=\"file *sbi* +p; file *smp* +p; file *tlb* +p; file *cpu* +p\""
        self._dtb_addr: int | None = None  # 最近一次 load_dtb 的地址
        self._bootargs = (
            bootargs
            if bootargs is not None
            else (
                "earlycon=sbi console=ttySIF0 "
                "random.trust_bootloader=on "
                "ro init=/bin/zsh TERM=vt100"
            )
        )
        self._cycle = 0
        self._total_instrs = 0

        # initramfs 物理范围 (load_initrd 设置; None=未挂载)。
        # build_dtb 据此写入 /chosen/linux,initrd-start/end。
        self._initrd: dtb.Initrd | None = None

        # Terminal I/O — 后台线程管理 stdin/stdout, 由 _init_native_batch 在 UART 就绪后创建
        self._termio = None

        # 单次 native 批次的指令上限 (可变成员; 调试器按需覆盖)。
        self._native_max_instrs: int = 100000
        # 共享停止标志 — 由 debugger Ctrl+C 信号处理器写入, Rust batch engine 每指令检查
        self._native_stop_flag = ctypes.c_uint8(0)

        # WFI 唤醒事件 — 全部 hart 等待时用于阻塞而非轮询
        self._wake_event = threading.Event()

        ram_size = config.ram_size
        ram_base = config.ram_base

        # L2 缓存 — 始终启用
        l2 = L2Cache(size=config.l2_size)

        # 共享总线
        self.bus = Bus(ram_size=ram_size, ram_base=ram_base, l2_cache=l2)

        # CLINT 设备
        self.clint = CLINT(num_harts=config.num_harts)
        self.clint.base_addr = config.periph.clint_base
        self.bus.add_device(self.clint.base_addr, self.clint)

        # PLIC 设备 — 标准双 context/hart: context 2h=hart h 的 M 模式,
        # 2h+1=S 模式。Linux 用 S-context, M 模式可信程序未来可用 M-context。
        self.plic = PLIC(
            base_addr=config.periph.plic_base,
            num_sources=128,
            num_contexts=2 * config.num_harts,
        )
        self.bus.add_device(self.plic.base_addr, self.plic)

        # 外设 — 按配置创建, base=0 时跳过
        self.uart: UART | None = None
        self.spi: SPI | None = None
        self.i2c: I2C | None = None
        self.gpio: GPIO | None = None
        self.virtio_blk: VirtIOBlock | None = None
        self._peripherals: dict[str, object] = {}

        p = config.periph
        if p.uart_base:
            self.uart = UART(
                base=p.uart_base,
                plic=self.plic,
                irq=UART_IRQ,
            )
            self.bus.add_device(p.uart_base, self.uart)
            self._peripherals["uart"] = self.uart
        if p.spi_base:
            self.spi = SPI(base=p.spi_base)
            self.bus.add_device(p.spi_base, self.spi)
            self._peripherals["spi"] = self.spi
        if p.i2c_base:
            self.i2c = I2C(base=p.i2c_base)
            self.bus.add_device(p.i2c_base, self.i2c)
            self._peripherals["i2c"] = self.i2c
        if p.gpio_base:
            self.gpio = GPIO(base=p.gpio_base)
            self.bus.add_device(p.gpio_base, self.gpio)
            self._peripherals["gpio"] = self.gpio
        if p.virtio_blk_base and config.disk_image:
            self.virtio_blk = VirtIOBlock(
                image_path=config.disk_image,
                mem_read=self.bus.read,
                mem_write=self.bus.write,
                plic=self.plic,
                irq=VIRTIO_BLK_IRQ,
            )
            self.bus.add_device(p.virtio_blk_base, self.virtio_blk)
            self._peripherals["virtio_blk"] = self.virtio_blk
        else:
            self.virtio_blk = None

        # 看门狗设备 — 多 hart 停滞检测, DTB 可见
        _wdog_base = p.watchdog_base if p.watchdog_base else 0x1000_4000
        self.watchdog = HartWatchdog(
            self, base=_wdog_base, num_harts=config.num_harts,
        )
        self.bus.add_device(_wdog_base, self.watchdog)
        self._peripherals["watchdog"] = self.watchdog

        # PC-stall watchdog — 检测 M-mode hart 停滞 + S-mode WFI
        # (不依赖 all_exec_cnt==0, 每批次均检查)
        self._stall_watchdog = HartStallWatchdog(self, threshold=3)

        # 创建 harts, 注入后端
        self.harts: list[Hart] = []
        for i in range(config.num_harts):
            h = Hart(id=i, pmp_entries=config.pmp_entries)
            h.pc = config.prog_cnt
            inject_memory_backend(
                h, self.bus.read,
                _wrap_phy_write_for_uart(h.id, self.uart, self.bus.write),
            )
            h.bus = self.bus
            h.interrupt_ctrl = self.clint
            h.plic = self.plic
            self.harts.append(h)

        # 互引用: 每个 hart 持有全 hart 列表, 供 mfence.did 等广播操作
        for h in self.harts:
            h.all_harts = self.harts

        # ---- Phase A: native batch execution state ----
        # Set PYREMU_NATIVE_BATCH=0 to force pure-Python step for differential testing.
        self._native_batch: bool = False
        self._native_states: Any = None  # ctypes HartState array
        self._native_result: Any = None  # ctypes BatchResult
        self._native_ram_buf: Any = None  # ctypes array from_buffer(bytearray)
        self._bp_addrs: list[int] = []  # breakpoint PCs -> Rust batch checks inline
        self._warned_infinite_virtio = False  # 防止 virtio 异常循环时重复告警
        # WFI 空闲轮询回调: debugger 设为其 stdin 转发函数, 确保用户输入
        # 能在内核 WFI 等待期间被及时 preload 到 UART RX 并触发中断唤醒。
        self._idle_poll_cb: Any = None  # () -> bool
        self._init_native_batch()

    def _init_native_batch(self) -> None:
        """Initialise the native batch acceleration infrastructure.

        Called once during ``__init__``.  Sets up ctypes arrays mirroring the
        Rust ``HartState`` / ``BatchResult`` structs and wraps the bus RAM
        ``bytearray`` as a ctypes pointer the FFI layer can read/write.

        Set ``PYREMU_NATIVE_BATCH=0`` to force pure-Python step for
        differential testing or tests that depend on per-instruction
        ``step()`` semantics.
        """
        if os.environ.get("PYREMU_NATIVE_BATCH") == "0":
            return
        if not native_available():
            return

        self._native_batch = True
        self._native_result = BatchResult()

        num_harts = len(self.harts)
        self._native_states = (HartState * num_harts)()
        # TLB generation counter — persists across batches so SFENCE.VMA
        # broadcast semantics (gen increment) survive FFI call boundaries.
        self._tlb_gen = ctypes.c_uint64(0)
        self._tlb_gen_per_hart = (ctypes.c_uint64 * num_harts)()

        # Per-hart MSIP edge counters — Python CLINT writes only set the
        # level bit; we track 0->1 transitions and encode them as edge
        # increments so Rust's sync_msip can detect cross-batch MSIP.
        self._clint_msip_edge: list[int] = [0] * num_harts
        self._clint_msip_prev: list[int] = [0] * num_harts

        # Wrap the bus RAM bytearray so Rust can read/write it directly
        ram = self.bus._ram  # bytearray
        self._native_ram_buf = (ctypes.c_uint8 * len(ram)).from_buffer(ram)  # type: ignore[attr-defined]

        # Terminal I/O 管理器 — 封装 Rust termio 后台线程 (或 Python 回退)
        # 独立于 CPU 批次循环, 持续转发 stdin->UART RX 和 UART TX->stdout.
        # 创建时机: _init_native_batch (仅在 native batch 可用时).
        # UART 此时已构造完毕 (见 __init__ 顺序), 可安全注入。
        if self.uart is not None and self._termio is None:
            self._termio = TerminalIO(
                uart=self.uart,
                wake_event=self._wake_event,
                stdin_fd=sys.stdin.fileno(),
                stdout_fd=sys.stdout.fileno(),
            )
            self.uart.termio = self._termio
            # 启动 TX 归档 daemon — 异步将 ring buffer 写入 hart 日志,
            # 与批次循环完全解耦, 不依赖 _native_flush_uart.
            self._termio.start_tx_archive_thread()

        # 默认 WFI 空闲轮询回调: 用于无调试器直连 run() 的场景 (如
        # make emu-linux-jump)。调试器 _enter_run_mode 会按需覆盖此回调。

        # if self._termio is not None and self._idle_poll_cb is None:
        #     self._idle_poll_cb = self._default_idle_poll

        # _step_native marshal/unmarshal 之间传递的临时缓冲 (每批次重建)。
        self._pmp_num: int = 0
        self._pmp_cfg_buf = (ctypes.c_uint8 * 0)()
        self._pmp_addr_buf = (ctypes.c_uint64 * 0)()
        self._clint_mtimecmp_arr = (ctypes.c_uint64 * 0)()
        self._clint_msip_arr = (ctypes.c_uint8 * 0)()

    # ----------------------------------------------------------
    #  平台
    # ----------------------------------------------------------

    @property
    def config(self) -> PlatformConfig:
        return self._cfg

    @property
    def peripherals(self) -> dict[str, object]:
        return self._peripherals

    # ----------------------------------------------------------
    #  设备树 (FDT/DTB)
    # ----------------------------------------------------------

    def build_dtb(
        self,
    ) -> bytes:
        """基于当前平台配置, 通过 libfdt 构建 DTB blob.

        固件可将此 blob 加载到 RAM 并通过 a1 寄存器接收其地址.
        """
        return dtb.build_dtb(
            self._cfg,
            uart=self.uart,
            spi=self.spi,
            i2c_gen=self.i2c,
            gpio=self.gpio,
            virtio_blk=self.virtio_blk,
            watchdog=self.watchdog,
            plic=self.plic,
            bootargs=self._bootargs,
            initrd=self._initrd,
        )

    def load_dtb(
        self,
        addr: int,
    ) -> None:
        """生成 DTB, 写入 RAM, 并将地址写入所有 hart 的 a1 (x11).

        遵循 RISC-V 引导约定: firmware 入口时 a1 指向设备树 blob.
        """
        dtb = self.build_dtb()
        self.load_dtb_blob(addr, dtb)

    def load_dtb_file(
        self,
        path: str,
        addr: int | None = None,
    ) -> None:
        """加载预编译的 DTB 文件到 RAM, 并将地址写入所有 hart 的 a1.

        *addr* 为 None 时自动放在 RAM 顶端 − 64 KiB.
        """
        if addr is None:
            addr = self._cfg.ram_base + self._cfg.ram_size - 0x10000
        dtb = Path(path).read_bytes()
        self.load_dtb_blob(addr, dtb)

    def load_dtb_blob(
        self,
        addr: int,
        dtb: bytes,
    ) -> None:
        """将 *dtb* 写入 RAM 的 *addr*, 并设所有 hart 的 a1."""
        self._dtb_addr = addr
        self.load_code(addr, dtb)
        for hart in self.harts:
            hart.write_gpr(11, addr)

    # ----------------------------------------------------------
    #  代码加载
    # ----------------------------------------------------------

    def load_initrd(
        self,
        path: str,
        addr: int,
    ) -> dtb.Initrd:
        """将 initramfs (cpio[.gz]) 加载到 RAM 的 *addr*, 记录物理范围.

        返回 Initrd(start, end); 随后 build_dtb 会把该范围写入
        /chosen/linux,initrd-start / -end 供内核解包为 rootfs。
        必须在生成 DTB 之前调用 (范围需进入设备树)。
        """
        data = Path(path).read_bytes()
        self.load_code(addr, data)
        self._initrd = dtb.Initrd(start=addr, end=addr + len(data))
        return self._initrd

    def load_code(
        self,
        addr: int,
        code: bytes,
    ) -> None:
        """将机器码写入物理 RAM 的指定地址.

        裸金属程序应加载到复位向量对应的地址.

        优先写入 bytearray (Rust batch 读取路径); 仅对小段 (< 2 MiB)
        预热 L2 缓存 (Python 路径受益), 避免大段逐 cache-line 拷贝膨胀到数十秒.
        """
        self.bus.write_ram_direct(addr, code)
        if len(code) <= _FAST_LOAD_THRESHOLD:
            self.bus.write(addr, code)

    def _write_segment(self, seg, vaddr_offset: int, threshold: int) -> None:
        """Write a single firmware segment (data + BSS zero-fill) to RAM.

        始终写入 bytearray (Rust native batch 的直接读取源);
        对小段同时写入 L2 以预热缓存 (Python 路径受益).
        """
        vaddr = seg.vaddr + vaddr_offset

        # 主写入: bytearray (Rust batch 读取路径的权威数据源)
        self.bus.write_ram_direct(vaddr, seg.data)
        # 小段预热 L2 (Python 路径)
        if len(seg.data) <= threshold:
            self.bus.write(vaddr, seg.data)

        if seg.memsz > len(seg.data):
            zero_pad = seg.memsz - len(seg.data)
            zp_addr = vaddr + len(seg.data)
            self.bus.write_ram_direct(zp_addr, b"\x00" * zero_pad)
            if zero_pad <= threshold:
                self.bus.write(zp_addr, b"\x00" * zero_pad)

    def load_firmware(
        self,
        image: FirmwareImage | None,
        load_offset: int = 0,
    ) -> None:
        """加载由 parse_firmware() 解析得到的固件镜像.

        将镜像的所有内存段写入物理 RAM, 并将所有 hart
        的 PC 设置为镜像的入口地址 (均加 *load_offset*).

        *load_offset* 用于 PIE 固件搬迁: 当 ELF 段链接地址为 0x0
        但 RAM 从 0x80000000 开始时, 传入 load_offset=ram_base
        即可将各段上移.
        """
        if image is None:
            raise ValueError("载入了无效的镜像")


        for seg in image.segments:
            self._write_segment(seg, load_offset, _FAST_LOAD_THRESHOLD)

        # 若 load_offset ≠ 0, 开放 VMA 影子映射:
        # 固件链接在低地址 (VMA), 但 RAM 在高地址.
        # S-mode mret 用 VMA 作为目标地址, 需要 VMA 也能访问同一 RAM.
        if load_offset != 0 and image.segments:
            for seg in image.segments:
                self._write_segment(seg, 0, _FAST_LOAD_THRESHOLD)
            vma_min = min(seg.vaddr for seg in image.segments)
            vma_max = max(seg.vaddr + seg.memsz for seg in image.segments)
            self.bus.set_ram_shadow(vma_min, vma_max - vma_min)

        # 将所有 hart 的 PC 设置为入口地址
        for hart in self.harts:
            hart.pc = image.entry_point + load_offset

    # ----------------------------------------------------------
    #  执行
    # ----------------------------------------------------------

    _TRAP_LOOP_THRESHOLD = 3  # 连续 trap 超过此次数视为不可恢复

    @staticmethod
    def _handle_native_sys_exit(hart: Hart, instr: int) -> int:
        """Execute one instruction deferred by the Rust batch engine.

        Returns 1 if an instruction was executed, 0 if it was a no-op
        (instr == 0 or WFI wake).
        """
        if instr == 0:
            return 0
        saved_pc = hart.pc
        try:
            advance = hart.exec_instr(instr)
        except (NotImplementedError, MemoryAccessFault):
            deliver_trap(hart, TrapType.IllInstr, tval=instr, is_interrupt=False)
            advance = 0
        if advance != 0 and hart.pc == saved_pc:
            hart.pc = (hart.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF
            hart._consecutive_traps = 0
        elif advance == 0 and hart.pc == saved_pc:
            hart._consecutive_traps += 1
        if hart._consecutive_traps >= Emulator._TRAP_LOOP_THRESHOLD:
            hart._halted = True
        if not hart._halted:
            check_pending_interrupts(hart)
        return 0 if hart._wfi_woken else 1

    # ----------------------------------------------------------
    #  Native batch fast path (Phase A)
    # ----------------------------------------------------------

    def request_native_stop(self) -> None:
        """由调试器信号处理器调用, 请求 Rust batch engine 尽快退出.
        同时设置 _wake_event 打断任何 idle sleep (WFI 轮询)."""
        self._native_stop_flag.value = 1
        self._wake_event.set()

    def _step_native(self, active: list[Hart]) -> int:
        # 重置共享停止标志 (上一轮可能已被 Ctrl+C 设置)
        self._native_stop_flag.value = 0

        # 将 L2 脏行回写到 bytearray, 确保 Rust batch 从 bytearray
        # 直接读取指令/数据时能看到 Python 侧的全部写入.
        self.bus.flush_l2()

        # Marshal ALL harts (including halted): Rust needs every state
        # for total_instrs summation and active_hart_num counting.
        # 每轮批次前转发 stdin (不只在 WFI idle 时) — 消除 1 字符输入延迟:
        # TermIO reader 线程异步写入 ring buffer, 此处同步 preload 到 UART.
        if self._termio is not None:
            self._termio.drain_rx()
        # 先把 PLIC 外部中断 (MEIP/SEIP) 同步进各 hart 的 mip —— native 引擎内部
        # 只同步 CLINT (MSIP/MTIP), 不感知 PLIC, 否则 virtio 等外设中断永远到不了 hart。
        self._native_sync_plic_mip()
        for i, hart in enumerate(self.harts):
            marshal_hart(hart, self._native_states[i])

        # Save pre-batch CLINT MSIP levels to distinguish between MSIPs
        # that were already pending before the batch (processed by Rust,
        # safe to clear) and MSIPs that arrived during the batch (must
        # preserve so _update_hw_mip picks them up on next WFI check).
        _pre_batch_msip = [self.clint._msip[hid] & 1 for hid in range(len(self.harts))]

        dev_info = self._native_marshal_dev()
        pmp_info = self._native_marshal_pmp(active)
        clint_info = self._native_marshal_clint()
        uart_info = self._native_marshal_uart()
        virtio_info = self._native_marshal_virtio()

        # 统一使用 run_parallel (thread-per-hart 并发引擎)。串行 run_batch
        # 已弃用: round-robin 切片会在跨 hart IPI / TLB-shootdown 协议上死锁。
        virtio_ffi = run_parallel(
            self._native_states,
            len(self.harts),
            self._native_ram_buf,
            self.bus.ram_size,
            self.bus.ram_base,
            self.bus._shadow_base or 0,
            self.bus._shadow_size or 0,
            self._native_max_instrs,
            self._native_result,
            pmp=pmp_info,
            clint=clint_info,
            dev=dev_info,
            uart=uart_info,
            virtio=virtio_info,
            bp_addrs=self._bp_addrs if self._bp_addrs else None,
            stop_flag=self._native_stop_flag,
            tlb_gen=self._tlb_gen,
            tlb_gen_per_hart=self._tlb_gen_per_hart,
        )

        result = self._native_result
        for hid in range(len(self.harts)):
            unmarshal_hart(self._native_states[hid], self.harts[hid])

        # Rust 批量执行期间内核可能修改页表并执行 SFENCE.VMA,
        # 但 Python TLB 不参与 marshal/unmarshal (hart.py:904-907,
        # "Python TLB is ground truth") — 旧条目不被写回, 但旧条目的
        # 失效也不会传播到 Python 侧。下一轮 Python 模式的内存访问
        # 可能命中已在 Rust 侧被刷掉的陈旧 VA->PA 映射, 导致:
        #   1. 读到错误数据 (如把 7 当作指针) -> 计算错误地址
        #   2. SIGSEGV at nonsense VA (如 0x33d = NULL+0x33d)
        # 解决方案: 每轮 batch 后无条件刷掉全部 hart 的 itlb/dtlb。
        # 过度失效总是安全的 (至多多几次页表遍历), 热 TLB 会在下轮
        # Python 执行中快速重建。
        for hart in self.harts:
            hart.itlb.flush_all()
            hart.dtlb.flush_all()

        # Rust batch 可能直接修改了 bytearray; 使 L2 全部失效,
        # 强制 Python 侧后续读取从 bytearray 重新加载.
        self.bus.invalidate_l2()

        self._native_unmarshal_pmp()
        # TX ring buffer 归档由 _tx_archive daemon 异步处理,
        # 不在此处同步调用 — UART I/O 与批次循环完全解耦.
        self._native_unmarshal_clint(clint_info)

        # Rust 引擎投递 MSIP 后清除 state.mip.MSIP 以匹配 Python 侧
        # _trap_deliver_mmode 的自清零行为 (见 trap_handler.py line 285-300),
        # 但未回写清除 CLINT._msip。此处同步: 若 hart 的 mip.MSIP 已被 Rust
        # 清零, 且 CLINT MSIP 在 batch 前就已是 1 (旧 MSIP 已投递), 则同步
        # 清除 Python CLINT._msip。若 MSIP 在 batch 期间新到达 (_pre_batch_msip=0),
        # 则保留 CLINT._msip 不清理 — 否则在 batch 内由 hart A 写入、hart B
        # 尚未来得及由 sync_msip 检测到的跨核 IPI 会被永久丢弃 ->TLB-shootdown 死锁。
        for hid in range(len(self.harts)):
            if (
                (self.harts[hid].mip_val & (1 << 3)) == 0
                and self.clint._msip[hid]
                and _pre_batch_msip[hid]
            ):
                self.clint._msip[hid] = 0
                self.clint._notify_state_change()

        self._native_unmarshal_virtio(virtio_ffi)

        all_exec_cnt = result.total_instrs
        all_exec_cnt = self._native_handle_exit(result, all_exec_cnt)
        return self._native_finalize(all_exec_cnt, result, clint_info)

    # ---- _step_native 分解: marshal-in / unmarshal-out / 收尾 ----

    def _native_sync_plic_mip(self) -> None:
        """将各 hart 的 PLIC 挂起外部中断合并进其 mip (MEIP bit11 / SEIP bit9)。

        native 引擎内部只 `sync_msip`/`sync_mtip` (CLINT), 不感知 PLIC。设备 MMIO
        (含 virtio QueueNotify 与 PLIC claim/complete/ACK) 已 exit 到 Python 处理,
        故每次进入 native 批次前在此从 PLIC 刷新 MEIP/SEIP —— 否则 virtio 完成中断
        永远到不了 hart。MEIP/SEIP 纯由 PLIC 驱动, 直接替换这两位 (保留其余软件位)。
        """
        ext_mask = (1 << 9) | (1 << 11)
        for hart in self.harts:
            plic = hart._plic
            if plic is None:
                continue
            hart.mip_val = (hart.mip_val & ~ext_mask) | plic.get_pending_mip(hart.id)

    def _native_marshal_dev(self) -> DevInfo:
        """设备 MMIO 地址范围 -> DevInfo (供 Rust 判定 MMIO 直通)."""
        devices = self.bus.devices
        if devices:
            dev_bases = (ctypes.c_uint64 * len(devices))()
            dev_ends = (ctypes.c_uint64 * len(devices))()
            for j, (base, dev) in enumerate(sorted(devices.items())):
                dev_bases[j] = base
                dev_ends[j] = base + dev.size
        else:
            dev_bases = (ctypes.c_uint64 * 0)()
            dev_ends = (ctypes.c_uint64 * 0)()
        return DevInfo(bases=dev_bases, ends=dev_ends)

    def _native_marshal_pmp(self, active: list[Hart]) -> PmpInfo:
        """构建 per-hart PMP 缓冲 (num_harts * 64 项连续数组)。

        每个 hart 拥有独立 PMP (硬件语义), hart h 使用 [h*64, h*64+64) 切片;
        native 引擎按 hart_id 偏移取各自切片。

        切勿跨 hart 共享同一 PMP 缓冲: SMP 启动时各 hart 的 OpenSBI warm-boot
        会并发 (run_parallel 线程) 重写自己的 PMP, 共享缓冲会造成数据竞争与
        瞬时执行权限丢失 -> 内核取指访问故障 (cause=1, 常见于 handle_exception
        入口)。参见 CHANGELOG 2026-07-14。
        """
        total = len(self.harts)
        ref_pmp = self.harts[0]._pmp
        self._pmp_num = ref_pmp._num_entries if ref_pmp is not None else 0
        self._pmp_cfg_buf = (ctypes.c_uint8 * 0)()
        self._pmp_addr_buf = (ctypes.c_uint64 * 0)()
        if self._pmp_num > 0:
            self._pmp_cfg_buf = (ctypes.c_uint8 * (64 * total))()
            self._pmp_addr_buf = (ctypes.c_uint64 * (64 * total))()
            for hid, h in enumerate(self.harts):
                hp = h._pmp
                if hp is None or hp._num_entries == 0:
                    continue
                if hp._cache_dirty:
                    hp._rebuild_cache()
                base = hid * 64
                fc = hp._flat_cfg
                fa = hp._flat_addr
                for j in range(64):
                    self._pmp_cfg_buf[base + j] = fc[j]
                    self._pmp_addr_buf[base + j] = fa[j]
        return PmpInfo(
            cfg=self._pmp_cfg_buf,
            addr=self._pmp_addr_buf,
            pmpsplit=active[0].pmpsplit_val if active else 0,
            hart_num=total,
        )

    def _native_unmarshal_pmp(self) -> None:
        """将各 hart 的 PMP 切片回写到其独立 Pmp; 仅在实际改写时同步 CSR.

        每个 hart 的 OpenSBI 会把 pmpcfg/pmpaddr 写入自己的 64 项切片。
        不再镜像到其它 hart。sync_from_flat (回写 CSR) 开销较大, 绝大多数
        批次 PMP 无变化, 故仅在切片确实被 Rust 改写时才同步。
        """
        if self._pmp_num <= 0:
            return
        for hid, h in enumerate(self.harts):
            hp = h._pmp
            if hp is None or hp._num_entries == 0:
                continue
            base = hid * 64
            fc = hp._flat_cfg
            fa = hp._flat_addr
            changed = False
            for j in range(64):
                cv = self._pmp_cfg_buf[base + j]
                av = self._pmp_addr_buf[base + j]
                if cv != fc[j] or av != fa[j]:
                    fc[j] = cv
                    fa[j] = av
                    changed = True
            if changed:
                hp._sync_from_flat()

    def _native_marshal_clint(self) -> ClintInfo:
        """构建 ClintInfo (可变数组, Rust 可 inline 更新 MSIP/MTIMECMP)。

        数组按全部 hart (非仅 active) 分配: 某 hart 可能写 halted hart 的
        MSIP/MTIMECMP, 需为每个 hart 预留槽位。

        Python CLINT MSIP 仅维护 level bit (0/1); Rust sync_msip 需要
        edge-counter (bits 7:1) 来检测跨 batch 的 MSIP 变化。此处检测
        0->1 跳变并递增 edge counter, 编码为 ``level | (edge << 1)``.
        """
        clint = self.clint
        total = len(self.harts)
        self._clint_mtimecmp_arr = (ctypes.c_uint64 * total)()
        self._clint_msip_arr = (ctypes.c_uint8 * total)()
        if clint is not None:
            # Sync stimecmp -> CLINT mtimecmp BEFORE marshaling.
            # Python-side CSR writes (e.g. during MMIO/ECALL handling) update
            # the hart's stimecmp but NOT CLINT._mtimecmp.  If we marshal
            # the stale CLINT value, Rust's sync_mtip / wfi_check_all_idle
            # see an expired or wrong deadline -> timer-interrupt storm that
            # saturates the batch budget without forward progress.
            for hart in self.harts:
                st_val = hart._csr_read_raw("stimecmp")
                if st_val > 0 and hart.id < len(clint._mtimecmp):
                    clint._mtimecmp[hart.id] = st_val
            for hid in range(total):
                if hid < len(clint._mtimecmp):
                    self._clint_mtimecmp_arr[hid] = clint._mtimecmp[hid]
                if hid < len(clint._msip):
                    level = clint._msip[hid] & 1
                    prev = self._clint_msip_prev[hid]
                    if level == 1 and prev == 0:
                        self._clint_msip_edge[hid] = (
                            (self._clint_msip_edge[hid] + 1) & 0x7F
                        )
                    self._clint_msip_prev[hid] = level
                    self._clint_msip_arr[hid] = level | (
                        self._clint_msip_edge[hid] << 1
                    )
        return ClintInfo(
            mtime=clint.get_mtime() if clint is not None else 0,
            mtimecmp=self._clint_mtimecmp_arr,
            msip=self._clint_msip_arr,
            base=clint.base_addr if clint is not None else 0,
        )

    def _native_unmarshal_clint(self, clint_info: ClintInfo) -> None:
        """从 Rust 改写的数组同步 CLINT 状态回 Python; 并对齐 SSTC stimecmp.

        内核通过 Sstc (csrw stimecmp) 设定时器, stimecmp 即权威定时器值;
        始终复制到 mtimecmp 供 SBI_SET_TIMER 与 WFI 空闲优化一致使用。

        MSIP: 从 Rust 返回的编码值中提取 edge counter 供下一轮 marshal;
        仅 level bit 回写到 Python CLINT._msip.
        """
        clint = self.clint
        if clint is None:
            return
        for hid in range(len(self.harts)):
            if hid < len(clint._mtimecmp):
                clint._mtimecmp[hid] = self._clint_mtimecmp_arr[hid]
            if hid < len(clint._msip):
                raw = self._clint_msip_arr[hid]
                self._clint_msip_edge[hid] = (raw >> 1) & 0x7F
                level = raw & 1
                clint._msip[hid] = level
                self._clint_msip_prev[hid] = level  # avoid double-count on next marshal
        clint._mtime = clint_info.mtime
        for hart in self.harts:
            st_val = hart._csr_read_raw("stimecmp")
            if st_val > 0 and hart.id < len(clint._mtimecmp):
                clint._mtimecmp[hart.id] = st_val

    def _native_marshal_uart(self) -> UartInfo:
        """UART info — 让 Rust inline 处理 TXDATA + IE/IP/TXCTRL 寄存器访问.

        TX 环形缓冲和写索引由 TerminalIO 提供。tx_notify_fd 是 TX 通知管道
        的写端, Rust 每写一个 TXDATA 字节后写 1 字节到此 fd, 唤醒 Python
        TX drain daemon -> 零轮询事件驱动 TX 排空。
        """
        if self._termio is None:
            return UartInfo(base=0)
        uart = self.uart
        return UartInfo(
            base=uart.base_addr if uart is not None else 0,
            tx_buf=self._termio.tx_buf,
            tx_wr=self._termio.tx_wr,
            ie=uart._ie if uart is not None else 0,
            txctrl=uart._txctrl if uart is not None else 0,
            rxctrl=uart._rxctrl if uart is not None else 0,
            rx_fifo_len=len(uart._rx_fifo) if uart is not None else 0,
            tx_notify_fd=self._termio.tx_notify_w,
        )

    def _native_flush_uart(self) -> None:
        """TX ring buffer -> hart 日志归档.

        Rust libc::write 已即时输出, 此处仅消费 ring buffer 防止溢出,
        不触发 _tx_callback->stdout (避免双重输出).
        """
        if self.uart is None or self._termio is None:
            return
        self._termio.drain_tx_logs_archive_only()

    # ---- virtio-blk inline marshal / unmarshal ----

    def _native_marshal_virtio(self) -> VirtIOInfo:
        """构建 VirtIOInfo, 携带完整运行时 MMIO 寄存器状态。

        Rust 批次内 inline 处理全部 virtio MMIO 读写; 批次结束后
        _native_unmarshal_virtio 将 Rust 修改过的字段同步回 Python vblk。
        本方法必须在每轮批次前将当前 Python 状态完整传递给 Rust,
        否则动态寄存器 (queue 描述符、中断状态等) 会在跨批次时归零,
        客机看到的设备状态回退为未初始化。"""
        vblk = self.virtio_blk
        if vblk is None:
            return VirtIOInfo(base=0)
        return VirtIOInfo(
            base=vblk.base_addr,
            capacity=vblk._disk_size // 512,
            queue_num_max=vblk._queue_num_max,
            device_features_sel=vblk._device_features_sel,
            driver_features_sel=vblk._driver_features_sel,
            driver_features=vblk._driver_features,
            queue_sel=vblk._queue_sel,
            queue_num=vblk._queue_num,
            queue_ready=vblk._queue_ready,
            queue_desc=vblk._queue_desc,
            queue_driver=vblk._queue_driver,
            queue_device=vblk._queue_device,
            status=vblk._status,
            interrupt_status=vblk._interrupt_status,
        )

    def _native_unmarshal_virtio(self, virtio_ffi) -> None:
        """将 Rust 修改过的 virtio 状态同步回 Python VirtIOBlock.

        Rust 批量执行期间 inline 处理了 MMIO 寄存器读写,
        批次结束后同步 changed fields:
        - QueueNotify pending ->调用 _process_queue
        - irq_maybe_lower ->调用 _lower_irq_if_idle
        - 寄存器状态 ->同步回 VirtIOBlock
        """
        vblk = self.virtio_blk
        if vblk is None or virtio_ffi is None:
            return

        # 同步 inline 修改的寄存器状态
        vblk._device_features_sel = virtio_ffi.device_features_sel
        vblk._driver_features_sel = virtio_ffi.driver_features_sel
        vblk._driver_features = virtio_ffi.driver_features
        vblk._queue_sel = virtio_ffi.queue_sel
        vblk._queue_num = virtio_ffi.queue_num
        vblk._queue_ready = bool(virtio_ffi.queue_ready)
        vblk._queue_desc = virtio_ffi.queue_desc
        vblk._queue_driver = virtio_ffi.queue_driver
        vblk._queue_device = virtio_ffi.queue_device
        vblk._status = virtio_ffi.status
        vblk._interrupt_status = virtio_ffi.interrupt_status

        # InterruptACK 清空所有中断位 ->拉低 PLIC IRQ.
        # 必须在 notify_pending 之前处理: 若两者在同一批次内触发,
        # 先降低 IRQ 电平再处理新队列, 避免 _lower_irq_if_idle 看到
        # _process_queue 刚写入的 _interrupt_status 而跳过降电平,
        # 导致 _do_complete 时 level 仍为高 ->无限 re-level.
        if virtio_ffi.irq_maybe_lower:
            virtio_ffi.irq_maybe_lower = 0
            vblk._lower_irq_if_idle()

        # QueueNotify: Rust 设置 notify_pending=1 ->Python 处理 virtqueue.
        # 分批处理 (每批最多 _VIRTIO_PROCESS_BATCH 个描述符), 循环直到全部完成.
        # 每批之间 Python 可响应 ctrl+C; _max_batches 防止异常情况下的死循环.
        if virtio_ffi.notify_pending:
            virtio_ffi.notify_pending = 0
            _batch_guard = 0
            _max_batches = 256  # 256 × 16 = 4096 描述符, 远超正常 ext4 mount 所需
            while vblk._process_queue(max_descriptors=_VIRTIO_PROCESS_BATCH):
                _batch_guard += 1
                if _batch_guard >= _max_batches:
                    if not self._warned_infinite_virtio:
                        self._warned_infinite_virtio = True
                    break
            virtio_ffi.interrupt_status = vblk._interrupt_status

    def _native_handle_exit(self, result: BatchResult, all_exec_cnt: int) -> int:
        """处理 native 批次退出原因 (ECALL/MMIO/TRAP/BREAKPOINT)."""
        total_harts = len(self.harts)
        if result.exit_hart_id >= total_harts:
            return all_exec_cnt
        exit_hart = self.harts[result.exit_hart_id]
        if result.exit_reason in (EXIT_ECALL, EXIT_MMIO):
            all_exec_cnt += self._handle_native_sys_exit(exit_hart, result.exit_instr)
        elif result.exit_reason == EXIT_TRAP:
            if exit_hart._consecutive_traps >= self._TRAP_LOOP_THRESHOLD:
                exit_hart._halted = True
            all_exec_cnt += 1
        elif result.exit_reason == EXIT_BREAKPOINT:
            # Breakpoint hit in Rust — PC 已核对 bp_addrs, 匹配指令已执行并计数;
            # debugger 的 _check_multi_hart_bp 会据 h.pc == 断点地址上报命中。
            # Dump accumulated instruction counters once for the user to inspect.
            # icount_flush()  # temporarily disabled — focus on per-instruction diagnostics
            pass
        return all_exec_cnt

    def _native_finalize(
        self,
        all_exec_cnt: int,
        result: BatchResult,
        clint_info: ClintInfo,
    ) -> int:
        """推进计数器 -> WFI 唤醒 -> 空闲休眠 -> 看门狗/PC-stall 检测."""
        clint = self.clint
        # Advance mtime / CSR counters BEFORE WFI wakeup checks so that timer
        # interrupts (MTIP/STIP) that matured during the batch are visible to
        # try_wfi_wakeup.  mtime 已由 Rust 在批次中推进; 仅补推 Python 侧
        # (ECALL/MMIO 处理) 执行的指令差值。
        self.sync_counters(all_exec_cnt, advance_mtime=False)
        if clint is not None:
            py_delta = all_exec_cnt - result.total_instrs
            if py_delta > 0:
                clint.tick(py_delta)
                clint_info.mtime = clint._mtime

        wfi_waiting = 0
        for hart in self.harts:
            if not hart._waiting or hart._halted:
                continue
            try_wfi_wakeup(hart)
            if hart._waiting:
                wfi_waiting += 1

        # 当有 hart 处于 WFI 且本批次无指令执行时, mtime 可能停滞不动
        # (Rust 未执行任何指令, py_delta 亦为 0)。主动推进 mtime 到最近定时器
        # 截止值, 确保定时器中断能在下次 try_wfi_wakeup 时被检测到。
        if wfi_waiting > 0 and all_exec_cnt == 0 and clint is not None:
            remaining = self._wfi_ticks_until_wake(self.harts)
            if remaining is not None:
                clint.tick(remaining)
                clint_info.mtime = clint._mtime
                for hart in self.harts:
                    if not hart._waiting or hart._halted:
                        continue
                    try_wfi_wakeup(hart)
                    if not hart._waiting:
                        wfi_waiting -= 1
            # 始终落到 _wfi_sleep_if_idle + PLIC 重同步;
            # 不在此处 early return — stdin 可能在 sleep 期间到达,
            # 需由下方的 PLIC mip 同步 + WFI 重唤醒处理.

        self._wfi_sleep_if_idle(self.harts, wfi_waiting)

        # 上文 _wfi_sleep_if_idle 可能在 WFI 轮询期间通过 _idle_poll_cb
        # 把 stdin 字节 preload 到了 UART RX 并置位了 PLIC 中断, 但此时
        # hart.mip 尚未同步 (SEIP/MEIP 仍为 0)。立即同步 PLIC 并重试
        # WFI 唤醒, 避免等待到下轮 _step_native() ->_native_sync_plic_mip(),
        # 匹配 QEMU 的事件驱动模型: stdin ->UART RX ->中断 ->即时唤醒。
        self._native_sync_plic_mip()
        _recheck = 0
        for hart in self.harts:
            if not hart._waiting or hart._halted:
                continue
            if try_wfi_wakeup(hart):
                _recheck += 1
        if _recheck:
            wfi_waiting = max(0, wfi_waiting - _recheck)
            self._wake_event.clear()

        # 看门狗: 本轮有指令执行则踢狗 (重置), 否则递减; 归零时向 WFI hart 注入 MSIP.
        if all_exec_cnt > 0:
            self.watchdog.kick_all()
        else:
            self.watchdog.tick()

        # PC-stall 检测: M-mode hart 停滞 + S-mode WFI 时触发 MSIP 重注入,
        # 打破 tlb_sync 死锁 (发端 M-mode 自旋等 sync=0, 收端 S-mode WFI 已清
        # CLINT MSIP 无法再唤醒, 且自旋不重发 MSIP)。
        if not self._stall_watchdog.check():
            return all_exec_cnt

        # 看门狗已设置 CLINT MSIP; 重新唤醒 WFI hart
        for hart in self.harts:
            if not hart._waiting or hart._halted:
                continue
            try_wfi_wakeup(hart)
            if not hart._waiting:
                wfi_waiting = max(0, wfi_waiting - 1)
        return all_exec_cnt

    @staticmethod
    def _hex(v: int) -> str:
        return f"0x{v:016x}"

    def _dump_hart_state(self, hart: Hart) -> str:
        """构建 hart 状态转储字符串 (GPR + CSR) — 不直接输出."""
        sep = "=" * 70
        lines = [
            sep,
            f"  Hart {hart.id} 进入不可恢复陷态 — 状态转储",
            sep,
            f"  PC       = {self._hex(hart.pc)}",
            f"  Mode     = {hart.mode.name}",
            f"  mstatus  = {self._hex(hart.mstatus_val)}",
            f"  mtvec    = {self._hex(hart.mtvec_val)}",
            f"  mepc     = {self._hex(hart.mepc_val)}",
            f"  mcause   = {self._hex(hart.mcause_val)}",
            f"  mtval    = {self._hex(hart.mtval_val)}",
            f"  satp     = {self._hex(hart.satp_val)}",
            sep + " GPRs",
        ]
        for i in range(16):
            lo_val = hart.gprs[i]
            hi_val = hart.gprs[i + 16]
            lo_name = gpr_name(i)
            lo_alias = gpr_alias(i)
            hi_name = gpr_name(i + 16)
            hi_alias = gpr_alias(i + 16)
            lines.append(
                f"  {lo_name:<3} {lo_alias:<5} = {self._hex(lo_val)}  "
                f"{hi_name:<3} {hi_alias:<5} = {self._hex(hi_val)}"
            )
        lines.append(sep)
        return "\n".join(lines)

    # ----------------------------------------------------------
    #  默认 WFI 空闲轮询 (无调试器直连路径)
    # ----------------------------------------------------------

    def _default_idle_poll(self) -> bool:
        """默认 WFI 空闲轮询回调.

        TX ring buffer 归档由 _tx_archive daemon 异步处理,
        stdin 转发由 TermIO reader 线程异步处理.

        Returns:
            False — 当前不引入新输入, 唤醒由其他路径触发.
        """
        return False

    # ----------------------------------------------------------
    #  WFI 等待优化
    # ----------------------------------------------------------

    _WFI_MAX_SLEEP = 0.2  # 单次最大睡眠 200ms, 降低 CPU 占用
    _WFI_TICK_US = 1.0  # 1 tick ≈ 1 µs (1 MHz 等效)

    def _wfi_ticks_until_wake(self, active_harts: list) -> int | None:
        """返回最早定时器中断剩余的 tick 数; 无活跃定时器时返回 None.

        检查 CLINT mtimecmp 和 SSTC stimecmp CSR 两个定时器源,
        取最早到期者.  内核可能使用任一机制设定时器.
        """
        now = self.clint._mtime
        best = None
        for h in active_harts:
            # CLINT mtimecmp (通过 MMIO 或 SBI ecall 设置)
            cmp = self.clint._mtimecmp[h.id]
            if cmp > 0 and cmp > now:
                rem = cmp - now
                if best is None or rem < best:
                    best = rem
            # SSTC stimecmp (S 模式直接 CSR 写入, 无需 ecall)
            sstc = h._csr_read_raw("stimecmp")
            if sstc > 0 and sstc > now:
                rem = sstc - now
                if best is None or rem < best:
                    best = rem
        return best

    def _wfi_sleep_if_idle(self, active: list, waiting_count: int) -> None:
        """当全部 hart 处于 WFI 时阻塞等待, 避免 CPU 100% 轮询.

        计算最近定时器到期时间并睡眠对应时长; 无定时器时睡眠固定短间隔.
        ``_wake_event`` 可被外部中断源 (IPI / debugger) 显式触发.

        若设置了 ``_idle_poll_cb`` (debugger 挂载), 则以短间隔轮询,
        确保 stdin 输入在内核 WFI 等待期间被及时转发到 UART RX 并触发
        中断唤醒, 而非等到 sleep 超时后才处理。
        """
        if waiting_count < len(active) or waiting_count == 0:
            return

        remaining = self._wfi_ticks_until_wake(active)
        if remaining is not None and remaining > 0:
            sleep_sec = min(remaining * self._WFI_TICK_US * 1e-6, self._WFI_MAX_SLEEP)
        elif remaining is not None:
            # remaining == 0: timer just expired, no need to sleep
            sleep_sec = 0.0
        else:
            # 无定时器 — WFI 唤醒由看门狗保证: 空闲时 watchdog.tick()
            # 递减计数器, 归零时注入 MSIP 唤醒全部 WFI hart.
            if self._idle_poll_cb is not None:
                self._wfi_poll_stdin_loop(active, 0.05)
            return

        if sleep_sec <= 0.0:
            return

        # 若有 idle poll callback (debugger stdin 转发), 以短间隔轮询,
        # 确保用户输入能在内核 WFI 等待期间被及时处理.
        if self._idle_poll_cb is not None:
            self._wfi_poll_stdin_loop(active, sleep_sec)
            return

        # 阻塞等待唤醒或超时; Ctrl+C (SIGINT) 通过 CPython 信号机制中断 wait
        self._wake_event.wait(timeout=sleep_sec)
        self._wake_event.clear()

        # 推进 mtime 以反映睡眠期间经过的 tick 数
        elapsed = max(1, int(sleep_sec / (self._WFI_TICK_US * 1e-6)))
        if self.clint is not None:
            self.clint.tick(elapsed - 1)  # -1 因为 step() 末尾还会 tick(1)

    def _wfi_poll_stdin_loop(self, active: list, sleep_sec: float) -> None:
        """以短间隔轮询 idle-poll 回调, 推进 mtime 以反映经过的墙钟时间.

        Rust termio 线程异步写 RX ring buffer, Python 侧不感知.
        用独立短间隔 (2ms) 检查 ring buffer, 避免被长定时器 (如 zsh
        的 100ms 轮询) 拉伸导致的输入延迟 — 消除"回车需额外按键触发"。
        """
        # 固定 2ms 间隔检查 RX ring buffer — Rust termio 异步写,
        # 不依赖定时器周期, 消除"回车需额外按键触发"的延迟
        _rx_check_interval = 0.002
        start = time.monotonic()
        deadline = start + sleep_sec
        while time.monotonic() < deadline:
            if self._idle_poll_cb():
                break
            remain = min(deadline - time.monotonic(), _rx_check_interval)
            if remain > 0:
                if self._wake_event.wait(timeout=remain):
                    self._wake_event.clear()
                    break
        # 推进 mtime 以反映实际经过的时间
        elapsed_real = max(1, int((time.monotonic() - start) / (self._WFI_TICK_US * 1e-6)))
        if self.clint is not None:
            self.clint.tick(elapsed_real - 1)  # -1 因为 step() 末尾还会 tick(1)
        # 看门狗按墙钟时间推进 (1 tick/ms, 上限防大循环阻塞 SIGINT)
        _wdog_ticks = min(
            max(1, int((time.monotonic() - start) * 1000)),
            200,
        )
        for _ in range(_wdog_ticks):
            self.watchdog.tick()
            if self._native_stop_flag.value != 0:
                break

    # ----------------------------------------------------------
    #  执行
    # ----------------------------------------------------------

    def sync_counters(self, instr_delta: int = 0, *, advance_mtime: bool = True) -> None:
        """推进 CLINT mtime 并同步硬件计数器 CSR (cycle/time/instret).

        每条指令或每个仿真周期调用一次。 RISC-V 规范:
        mcycle/minstret 为 M 模式读写, cycle/instret 为 U 模式只读影子;
        time (0xC01/0xB01) 为 CLINT mtime 的只读影子.

        *advance_mtime* 为 False 时跳过 ``clint.tick()``, 用于 native batch 路径
        (Rust 已在 batch 执行期间通过 ``*mut mtime`` 推进了 mtime).
        """
        self._total_instrs += instr_delta
        ticks = max(1, instr_delta)
        self._cycle += ticks
        if advance_mtime:
            self.clint.tick(ticks)
        cycle_val = self._cycle & 0xFFFF_FFFF_FFFF_FFFF
        time_val = self.clint.get_mtime() & 0xFFFF_FFFF_FFFF_FFFF
        for h in self.harts:
            h._csr_write_raw("mcycle", cycle_val)
            h._csr_write_raw("cycle", cycle_val)
            # Per-hart instruction count — harts in WFI waiting state
            # don't execute, so their count doesn't advance.
            h._csr_write_raw("minstret", h._total_instrs & 0xFFFF_FFFF_FFFF_FFFF)
            h._csr_write_raw("instret", h._total_instrs & 0xFFFF_FFFF_FFFF_FFFF)
            h._csr_write_raw("time", time_val)

    def step(self) -> int:
        """
        轮询方式让所有 hart 各执行一条指令, 每条指令后检查中断,
        每个周期推进 CLINT 时钟.

        若 hart 进入不可恢复的陷态 (连续 trap 超过阈值),
        则转储全部寄存器状态并暂停该 hart.

        Returns:
            本轮执行的指令数.
        """
        # stdin 转发与部分行刷新由调用方负责 (_run_loop 在每批前后
        # 各调一次 _feed_uart_stdin + _flush_uart_if_present),
        # step() 不自行 I/O 以免 stdin 数据被多次碎片化读取, 导致客机
        # 收到乱序输入。
        active = [h for h in self.harts if not h._halted]

        # ---- Native concurrent batch ----
        # Use the concurrent engine (run_parallel) for multi-hart
        # correctness.  The serial engine (run_batch) uses round-robin
        # slices within a single thread, which deadlocks on cross-hart
        # IPI / TLB-shootdown protocols (sender spins waiting for a
        # receiver that can't execute until the sender's slice ends).
        # Concurrent thread-per-hart avoids this fundamental problem.
        if self._native_batch and active:
            return self._step_native(active)

        # ---- Pure-Python path (fallback / .so not loaded) ----
        # 每轮 step 前转发 stdin — 与 _step_native 行为一致
        if self._termio is not None:
            self._termio.drain_rx()
        if len(active) > 1:
            random.shuffle(active)

        all_exec_cnt = 0
        wfi_waiting = 0
        for hart in active:
            # 设置 L2 缓存的当前域标记, 分配/命中行时自动打上 hart 的 mdid
            if self.bus._l2 is not None:
                self.bus._l2.current_mdid = hart.mdid_val

            # WFI 等待状态: 不取指/执行, 但仍检查中断唤醒
            if hart._waiting:
                # 1. WFI 唤醒: 仅需 mip & mie != 0 (源级使能).
                #    不要求 mstatus.MIE=1 (全局使能是中断投递条件).
                if try_wfi_wakeup(hart):
                    # 2. 唤醒后尝试投递中断 (如 MIE=1 则立即投递).
                    check_pending_interrupts(hart)
                if hart._waiting:
                    wfi_waiting += 1
                continue

            pc_before = hart.pc

            # 取指校验: VA->PA (itlb 翻译) + PMP execute check
            ok, fetch_pa = check_instruction_fetch(hart, hart.pc)
            if not ok:
                # trap 已在 check_instruction_fetch 内部投递
                continue

            # 跨页取指: 当 PC 在页末 2 字节内 (offset >= 0xFFE),
            # 4 字节取指跨越到下个虚拟页. 虚拟页之间的物理页不保证连续,
            # 需分别通过 MMU 翻译后从各物理地址读取各自字节.
            _page_off = hart.pc & 0xFFF
            if _page_off >= 0xFFE:
                ok2, fetch_pa2 = check_instruction_fetch(
                    hart, (hart.pc + 2) & 0xFFFF_FFFF_FFFF_FFFF
                )
                if not ok2:
                    continue
                lo = self.bus.read(fetch_pa, 2)
                hi = self.bus.read(fetch_pa2, 2)
                instr_bytes = lo + hi
            else:
                instr_bytes = self.bus.read(fetch_pa, 4)
            instr = int.from_bytes(instr_bytes, "little", signed=False)

            try:
                advance = hart.exec_instr(instr)
            except (NotImplementedError, MemoryAccessFault):
                deliver_trap(hart, TrapType.IllInstr, tval=instr, is_interrupt=False)
                advance = 0

            # 正常顺序执行 -> 清零连续 trap 计数
            if advance != 0 and hart.pc == pc_before:
                hart.pc = (hart.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF
                hart._consecutive_traps = 0
            elif advance == 0 and hart.pc == pc_before:
                hart._consecutive_traps += 1

            # 连续 trap 超过阈值 -> 标记为不可恢复
            if hart._consecutive_traps >= self._TRAP_LOOP_THRESHOLD:
                hart._halted = True
                continue

            # 指令边界 — 检查中断
            check_pending_interrupts(hart)

            # WFI 唤醒路径 (中断 handler -> mret -> 回到正常流) 不计入指令数,
            # 以保证 _total_instrs 反映固件实际执行的非中断上下文指令.
            if not hart._wfi_woken:
                hart._total_instrs += 1
                all_exec_cnt += 1

        self.sync_counters(all_exec_cnt)

        # WFI 优化: 全部未 halted 的 hart 处于 WFI 等待时, 阻塞而非轮询
        self._wfi_sleep_if_idle(active, wfi_waiting)

        return all_exec_cnt

    class TimeoutError(RuntimeError):
        """执行超时, 可能是死循环."""

        def __init__(
            self,
            timeout_sec: float,
            cycles: int,
            pc: int | None = None,
        ) -> None:
            self.timeout_sec = timeout_sec
            self.cycles = cycles
            self.pc = pc
            msg = f"模拟器超时 ({timeout_sec:.0f}s), 已执行 {cycles} 周期"
            if pc is not None:
                msg += f", 最后 PC={pc:#018x}"
            super().__init__(msg)

    def run(
        self,
        max_cycles: int,
        *,
        timeout: float = 1800.0,
        yield_every: int = 10000,
        yield_interval: float = 0.001,
    ) -> int:
        """执行 *max_cycles* 个周期.

        每 *yield_every* 个周期后通过 ``select.select()`` 让出 CPU
        *yield_interval* 秒, 降低宿主机 CPU 占用.
        设为 0 禁用节流 (100% CPU).

        使用 ``select`` 而非 ``time.sleep``:
        I/O 多路复用原语同时检查 stdin 可读性并等待, 任一条件满足即返回,
        比纯 sleep 更早响应外部输入.

        Args:
            max_cycles: 最大周期数.
            timeout: 墙钟超时秒数, 默认 1800 (30 分钟).
                     设为 0 禁用超时检查.
            yield_every: 每隔多少个周期让出 CPU, 默认 10000.
                         0 = 不限速 (跑满 CPU).
            yield_interval: 每次让出等待秒数, 默认 0.001 (1ms).

        Returns:
            实际执行的周期数.

        Raises:
            TimeoutError: 当 *timeout* 秒数被超过.
        """
        deadline = time.monotonic() + timeout if timeout > 0 else None
        do_yield = yield_every > 0
        i = 0
        while i < max_cycles:
            # Use concurrent native engine for batch execution when available.
            # The concurrent engine runs all active harts until a stop
            # condition (ECALL, MMIO, WFI all-idle, BREAKPOINT, or TRAP).
            # Each batch can execute many cycles; we count them and continue.
            if self._native_batch:
                active = [h for h in self.harts if not h._halted]
                if active:
                    # 每批次前转发 stdin + 刷新部分行 (对照 debugger _run_loop)
                    if self._idle_poll_cb is not None:
                        self._idle_poll_cb()
                    self._step_native(active)
                    i += 1
                    # 批量后让出 CPU + 检查超时
                    if do_yield and i % yield_every == 0:
                        yield_cpu(yield_interval)
                    if deadline is not None and time.monotonic() >= deadline:
                        raise self.TimeoutError(timeout, i, self.harts[0].pc if self.harts else None)
                    continue
            # Pure-Python fallback
            chunk = min(yield_every, max_cycles - i) if do_yield else max_cycles - i
            for _ in range(chunk):
                self.step()
            i += chunk
            # 批量后让出 CPU + 检查超时
            if do_yield:
                yield_cpu(yield_interval)
            if deadline is not None and time.monotonic() >= deadline:
                raise self.TimeoutError(timeout, i, self.harts[0].pc if self.harts else None)
        return self._cycle

    # ----------------------------------------------------------
    #  状态检查辅助 (调试用)
    # ----------------------------------------------------------

    def dump_hart_regs(self, hart_id: int = 0) -> dict:
        """导出指定 hart 的关键寄存器状态."""
        h = self.harts[hart_id]
        return {
            "hart_id": h.id,
            "pc": h.pc,
            "mode": h.mode.name,
            "mstatus": hex(h.mstatus_val),
            "mepc": hex(h.mepc_val),
            "mcause": hex(h.mcause_val),
            "mtval": hex(h.mtval_val),
            "mie": h.mie,
            "gprs": {f"x{i}": hex(h.gprs[i]) for i in range(32)},
        }

    def dump_memory(self, addr: int, size: int) -> bytes:
        """读取物理内存的 *size* 字节."""
        return self.bus.read(addr, size)

    @staticmethod
    def _fmt_hexdump(addr: int, data: bytes) -> str:
        """将字节数据格式化为 hexdump 字符串 (供调试器复用).

        三段分列着色: 地址列亮蓝, hex 列默认前景 (白/亮),
        ASCII 列 dim 灰.  避免单一大区段 [dim] 包裹导致 hex 区
        的数据字节意外被 Rich 解析为 markup 标签 (产生白/蓝混色).
        """
        lines = []
        for offset in range(0, len(data), 16):
            chunk = data[offset : offset + 16]
            hex_bytes = " ".join(f"{b:02x}" for b in chunk)
            # 转义 '[' 以防 Rich 误解析为 markup tag.
            hex_bytes = hex_bytes.replace("[", "[[")
            ascii_chars = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            ascii_chars = ascii_chars.replace("[", "[[")
            lines.append(
                f"  [bright_blue]{addr + offset:016x}[/]  "
                f"{hex_bytes:<48s}  "
                f"[dim]|{ascii_chars}|[/]"
            )
        return "\n".join(lines)

    def mem_hexdump(self, addr: int, size: int) -> str:
        """返回物理内存的十六进制 dump 字符串; 读取失败返回提示文本."""
        data = self.bus.try_read(addr, size)
        if data is None:
            return "(无法读取该地址)"
        return self._fmt_hexdump(addr, data)

    # ----------------------------------------------------------
    #  属性
    # ----------------------------------------------------------

    @property
    def num_harts(self) -> int:
        return self._cfg.num_harts

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def total_instructions(self) -> int:
        return self._total_instrs

    @property
    def prog_cnt(self) -> int:
        return self._cfg.prog_cnt
