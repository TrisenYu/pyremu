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
import io
import os
import random
import select
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pyremu._native import native_available, run_batch
from pyremu.core.decoder import Hart
from pyremu.core.hart import (
    EXIT_SYS,
    EXIT_TRAP,
    BatchResult,
    HartState,
    marshal_hart,
    unmarshal_hart,
)
from pyremu.core.mem_check_aux import check_instruction_fetch, inject_memory_backend
from pyremu.core.registers import gpr_alias, gpr_name
from pyremu.core.trap import TrapType
from pyremu.core.trap_handler import check_pending_interrupts, deliver_trap
from pyremu.interrupt.clint import CLINT
from pyremu.interrupt.plic import PLIC
from pyremu.memory.bus import Bus
from pyremu.memory.l2cache import L2Cache
from pyremu.peripheral import GPIO, I2C, SPI, UART, VirtIOBlock
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils import dtb
from pyremu.utils.parse_bin import FirmwareImage


def _yield_cpu(interval: float = 0.001) -> None:
    """通过 ``select`` 在 stdin 上等待 *interval* 秒以让出 CPU.

    比 ``time.sleep`` 响应更快: 若 stdin 上有待读取数据
    (如管道输入 / 外部事件), ``select`` 立即返回而非傻等.

    当 stdin 不是真实文件描述符时 (如 CI / pytest 重定向),
    自动降级为 ``time.sleep``.
    """
    try:
        select.select([sys.stdin], [], [], interval)
    except (io.UnsupportedOperation, TypeError, OSError):
        time.sleep(interval)


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
        # 默认 bootargs: 确保内核输出始终可见.
        #   earlycon=sbi       — 启用 SBI 早期控制台 (bootconsole)
        #   console=ttySIF0    — 首选 SiFive UART 控制台 (驱动 probe 后自动切换)
        #   keep_bootcon       — 阻止 tty0 注册时关闭 earlycon (需内核 ≥5.15)
        # 用户可通过 bootargs=None 显式禁用或传入自定义 bootargs 覆盖.
        self._bootargs = (
            bootargs
            if bootargs is not None
            else "earlycon=sbi console=ttySIF0 keep_bootcon"
        )
        self._cycle = 0
        self._total_instrs = 0

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

        # PLIC 设备
        self.plic = PLIC(
            base_addr=config.periph.plic_base,
            num_sources=128,
            num_contexts=config.num_harts,
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
                tx_callback=sys.stdout.write,
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
            )
            self.bus.add_device(p.virtio_blk_base, self.virtio_blk)
            self._peripherals["virtio_blk"] = self.virtio_blk
        else:
            self.virtio_blk = None

        # 创建 harts, 注入后端
        self.harts: list[Hart] = []
        for i in range(config.num_harts):
            h = Hart(id=i, pmp_entries=config.pmp_entries)
            h.pc = config.prog_cnt
            inject_memory_backend(h, self.bus.read, self.bus.write)
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
        self._native_states: Any = None   # ctypes HartState array
        self._native_result: Any = None   # ctypes BatchResult
        self._native_ram_buf: Any = None  # ctypes array from_buffer(bytearray)
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

        # Wrap the bus RAM bytearray so Rust can read/write it directly
        ram = self.bus._ram  # bytearray
        self._native_ram_buf = (ctypes.c_uint8 * len(ram)).from_buffer(ram)  # type: ignore[attr-defined]

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
            plic=self.plic,
            bootargs=self._bootargs,
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
        """将 *dtb* 写入 RAM 的 *addr*, 并设所有 hart 的 a1.

        供 load_dtb / load_dtb_file 共用.
        """
        self.load_code(addr, dtb)
        for hart in self.harts:
            hart.write_gpr(11, addr)

    # ----------------------------------------------------------
    #  代码加载
    # ----------------------------------------------------------

    def load_code(
        self,
        addr: int,
        code: bytes,
    ) -> None:
        """将机器码写入物理 RAM 的指定地址.

        裸金属程序应加载到复位向量对应的地址.

        双写: L2 缓存 (Python 侧读路径) + bytearray (Rust batch 读路径).
        """
        self.bus.write(addr, code)
        self.bus.write_ram_direct(addr, code)

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
            raise ValueError("载入了无效的内存")

        # 阈值判断: ≤2 MiB 的段走 L2 缓存, 更大的段 (如 Linux 内核 Image)
        # 绕过 L2 直写 RAM, 避免 cache-line 级逐片处理膨胀到数十秒.
        _FAST_LOAD_THRESHOLD = 2 * 1024 * 1024  # 2 MiB

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
    _NATIVE_MAX_INSTRS = 100000

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
        except NotImplementedError:
            deliver_trap(hart, TrapType.IllInstr, tval=instr, is_interrupt=False)
            advance = 0
        if advance != 0 and hart.pc == saved_pc:
            hart.pc = (hart.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF
            hart._consecutive_traps = 0
        if hart._consecutive_traps >= Emulator._TRAP_LOOP_THRESHOLD:
            hart._halted = True
        if not hart._halted:
            check_pending_interrupts(hart)
        return 0 if hart._wfi_woken else 1

    # ----------------------------------------------------------
    #  Native batch fast path (Phase A)
    # ----------------------------------------------------------

    def _step_native(self, active: list[Hart]) -> int:
        # 将 L2 脏行回写到 bytearray, 确保 Rust batch 从 bytearray
        # 直接读取指令/数据时能看到 Python 侧的全部写入.
        self.bus.flush_l2()

        for i, hart in enumerate(active):
            marshal_hart(hart, self._native_states[i])

        # Build device MMIO ranges from the bus
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

        # Build PMP state from the first active hart (all harts share the same PMP in pyremu)
        pmp = active[0]._pmp
        pmp_num = 0
        if pmp is not None and pmp._num_entries > 0:
            pmp_num = pmp._num_entries
            pmp_cfg_raw = bytes(pmp._flat_cfg)
            pmp_addr_raw = pmp._flat_addr
        else:
            pmp_cfg_raw = b""
            pmp_addr_raw = []

        # CLINT state
        clint = self.clint
        mtime = clint.get_mtime() if clint is not None else 0
        num_harts = max(len(active), 1)
        mtimecmp_arr = (ctypes.c_uint64 * num_harts)()
        msip_arr = (ctypes.c_uint8 * num_harts)()
        if clint is not None:
            for j in range(min(num_harts, len(clint._mtimecmp))):
                mtimecmp_arr[j] = clint._mtimecmp[j]
            for j in range(min(num_harts, len(clint._msip))):
                msip_arr[j] = clint._msip[j]

        # Shadow range from bus
        shadow_base = self.bus._shadow_base or 0
        shadow_size = self.bus._shadow_size or 0

        run_batch(
            self._native_states, len(active), self._native_ram_buf,
            self.bus.ram_size, self.bus.ram_base,
            shadow_base, shadow_size,
            self._NATIVE_MAX_INSTRS,
            self._native_result,
            pmp_cfg_raw, pmp_addr_raw,
            active[0].pmpsplit_val if active else 0,
            mtime, mtimecmp_arr, msip_arr,
            dev_bases, dev_ends,
        )
        result = self._native_result
        all_exec_cnt = result.total_instrs
        for i, hart in enumerate(active):
            unmarshal_hart(self._native_states[i], hart)

        # Rust batch 可能直接修改了 bytearray; 使 L2 全部失效,
        # 强制 Python 侧后续读取从 bytearray 重新加载.
        self.bus.invalidate_l2()

        if result.exit_hart_id < len(active):
            if result.exit_reason == EXIT_SYS:
                all_exec_cnt += self._handle_native_sys_exit(
                    active[result.exit_hart_id], result.exit_instr
                )
            elif result.exit_reason == EXIT_TRAP:
                exit_hart = active[result.exit_hart_id]
                # Trap was already delivered by Rust (mcause/mepc/mtval/mode set)
                if exit_hart._consecutive_traps >= self._TRAP_LOOP_THRESHOLD:
                    exit_hart._halted = True
                all_exec_cnt += 1

        wfi_waiting = 0
        for hart in active:
            if hart._waiting and not hart._halted:
                check_pending_interrupts(hart)
                if hart._waiting:
                    wfi_waiting += 1
        self.sync_counters(all_exec_cnt)
        self._wfi_sleep_if_idle(active, wfi_waiting)
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
    #  WFI 等待优化
    # ----------------------------------------------------------

    _WFI_MAX_SLEEP = 0.2  # 单次最大睡眠 200ms, 降低 CPU 占用
    _WFI_TICK_US = 1.0  # 1 tick ≈ 1 µs (1 MHz 等效)

    def _wfi_ticks_until_wake(self, active_harts: list) -> int | None:
        """返回最早定时器中断剩余的 tick 数; 无活跃定时器时返回 None."""
        now = self.clint._mtime
        best = None
        for h in active_harts:
            cmp = self.clint._mtimecmp[h.id]
            if not (cmp > 0 and cmp > now):
                continue
            rem = cmp - now
            if best is None or rem < best:
                best = rem
        return best

    def _wfi_sleep_if_idle(self, active: list, waiting_count: int) -> None:
        """当全部 hart 处于 WFI 时阻塞等待, 避免 CPU 100% 轮询.

        计算最近定时器到期时间并睡眠对应时长; 无定时器时睡眠固定短间隔.
        ``_wake_event`` 可被外部中断源 (IPI / debugger) 显式触发.
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
            # No timer set for any hart — nothing will wake them.
            # Skip the sleep to avoid blocking tests that call step()
            # after firmware completion.
            return

        # 阻塞等待唤醒或超时; Ctrl+C (SIGINT) 通过 CPython 信号机制中断 wait
        if sleep_sec > 0.0:
            self._wake_event.wait(timeout=sleep_sec)
            self._wake_event.clear()

        # 推进 mtime 以反映睡眠期间经过的 tick 数
        elapsed = max(1, int(sleep_sec / (self._WFI_TICK_US * 1e-6)))
        self.clint.tick(elapsed - 1)  # -1 因为 step() 末尾还会 tick(1)

    # ----------------------------------------------------------
    #  执行
    # ----------------------------------------------------------

    def sync_counters(self, instr_delta: int = 0) -> None:
        """推进 CLINT mtime 并同步硬件计数器 CSR (cycle/time/instret).

        每条指令或每个仿真周期调用一次。 RISC-V 规范:
        mcycle/minstret 为 M 模式读写, cycle/instret 为 U 模式只读影子;
        time (0xC01/0xB01) 为 CLINT mtime 的只读影子.
        """
        self._total_instrs += instr_delta
        self._cycle += 1
        self.clint.tick(1)
        cycle_val = self._cycle & 0xFFFF_FFFF_FFFF_FFFF
        instret_val = self._total_instrs & 0xFFFF_FFFF_FFFF_FFFF
        time_val = self.clint.get_mtime() & 0xFFFF_FFFF_FFFF_FFFF
        for h in self.harts:
            h._csr_write_raw("mcycle", cycle_val)
            h._csr_write_raw("cycle", cycle_val)
            h._csr_write_raw("minstret", instret_val)
            h._csr_write_raw("instret", instret_val)
            h._csr_write_raw("time", time_val)

    def step(self) -> int:
        """
        非常理想的假设：所有指令和多核之间的中断调度只需要一个时钟周期来完成
        此处凭借轮询方式，让所有 hart 各执行一条指令，然后作为当前一个周期内发生的事情

        每条指令执行后检查中断, 每个周期推进 CLINT 时钟.
        若 hart 进入不可恢复的陷态 (连续 trap 超过阈值),
        则转储全部寄存器状态并暂停该 hart.

        Returns:
            本轮执行的指令数.
        """
        # 多 hart 时随机打乱执行顺序, 确保彩票锁等场景机会均等
        active = [h for h in self.harts if not h._halted]

        # ---- Native batch fast path (Phase A) ----
        if self._native_batch and active:
            if len(active) > 1:
                random.shuffle(active)
            return self._step_native(active)

        # ---- Pure-Python path (fallback / .so not loaded) ----
        if len(active) > 1:
            random.shuffle(active)

        all_exec_cnt = 0
        wfi_waiting = 0
        for hart in active:
            # 声明当前 UART 写者 hart (多 hart 输出不交错)
            if self.uart is not None:
                self.uart.set_writer(hart.id)

            # 设置 L2 缓存的当前域标记, 分配/命中行时自动打上 hart 的 mdid
            if self.bus._l2 is not None:
                self.bus._l2.current_mdid = hart.mdid_val

            # WFI 等待状态: 不取指/执行, 但仍检查中断唤醒
            if hart._waiting:
                check_pending_interrupts(hart)
                # deliver_trap 会清除 _waiting; 若仍为 True 说明无待处理中断
                if hart._waiting:
                    wfi_waiting += 1
                continue

            pc_before = hart.pc

            # 取指校验: VA->PA (itlb 翻译) + PMP execute check
            ok, fetch_pa = check_instruction_fetch(hart, hart.pc)
            if not ok:
                # trap 已在 check_instruction_fetch 内部投递
                continue
            instr_bytes = self.bus.read(fetch_pa, 4)
            instr = int.from_bytes(instr_bytes, "little", signed=False)

            try:
                advance = hart.exec_instr(instr)
            except NotImplementedError:
                deliver_trap(hart, TrapType.IllInstr, tval=instr, is_interrupt=False)
                advance = 0

            # 正常顺序执行 -> 清零连续 trap 计数
            if advance != 0 and hart.pc == pc_before:
                hart.pc = (hart.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF
                hart._consecutive_traps = 0

            # 连续 trap 超过阈值 -> 标记为不可恢复
            if hart._consecutive_traps >= self._TRAP_LOOP_THRESHOLD:
                hart._halted = True
                continue

            # 指令边界 — 检查中断
            check_pending_interrupts(hart)

            # WFI 唤醒路径 (中断 handler -> mret -> 回到正常流) 不计入指令数,
            # 以保证 _total_instrs 反映固件实际执行的非中断上下文指令.
            if not hart._wfi_woken:
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
            msg = (
                f"模拟器超时 ({timeout_sec:.0f}s), "
                f"已执行 {cycles} 周期"
            )
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
        # 超时检查合并到 yield 检查点, 避免每周期 syscall
        i = 0
        while i < max_cycles:
            chunk = min(yield_every, max_cycles - i) if do_yield else max_cycles - i
            for _ in range(chunk):
                self.step()
            i += chunk
            # 批量后让出 CPU + 检查超时
            if do_yield:
                _yield_cpu(yield_interval)
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

        使用 Rich markup 确保地址列 / hex 值 / ASCII 区颜色一致:
        地址 dim 灰, hex 值默认亮色, ASCII 区 dim 灰.
        """
        lines = []
        for offset in range(0, len(data), 16):
            chunk = data[offset : offset + 16]
            hex_bytes = " ".join(f"{b:02x}" for b in chunk)
            ascii_chars = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(
                f"  [dim]{addr + offset:016x}[/]  "
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
