#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 17:00:47
# Last modified at 2026/06/09 星期二

"""
多核 RISC-V 模拟器 — 管理 Bus、CLINT、外设、多个 Hart 的执行流程。

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
    emu.run()         # 连续执行, 直至固件执行停机序列 (semihosting SYS_EXIT)
"""

import ctypes
from enum import IntEnum
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Callable

from pyremu._native import (
    ClintInfo,
    DevInfo,
    FfiExtIrqCtx,
    native_available,
    PmpInfo,
    run_parallel,
    UartInfo,
    VirtIOInfo,
)
from pyremu.configs_gen import (
    CPU_FREQ_HZ,
    WFI_WATCHDOG_MS,
)
from pyremu.core.decoder import Hart
from pyremu.core.hart import (
    EXIT_BREAKPOINT,
    EXIT_EBREAK,
    EXIT_ECALL,
    EXIT_MMIO,
    EXIT_TIMEOUT,
    EXIT_TRAP,
    HartState,
    InstrToBeExec,
    marshal_hart,
    unmarshal_hart,
)
from pyremu.core.mem_check_aux import (
    check_instruction_fetch,
    inject_memory_backend,
    MemoryAccessFault,
)
from pyremu.core.trap_def import TrapType
from pyremu.core.trap_handler import check_pending_interrupts, deliver_trap, try_wfi_wakeup
from pyremu.core.watchdog import HartStallWatchdog
from pyremu.interrupt.aplic import APLIC
from pyremu.interrupt.clint import CLINT
from pyremu.interrupt.imsic import IMSIC
from pyremu.interrupt.plic import PLIC
from pyremu.memory.bus import Bus
from pyremu.memory.l2cache import L2Cache
from pyremu.peripheral import GPIO, I2C, SPI, TerminalIO, UART, VirtIOBlock
from pyremu.peripheral.uart import UART_IRQ
from pyremu.peripheral.virtio_blk import VIRTIO_BLK_IRQ
from pyremu.peripheral.watchdog import HartWatchdog
from pyremu.platform import InterruptMode, PeripheralConfig, PlatformConfig
from pyremu.utils import dtb
from pyremu.utils.mask import mask64
from pyremu.utils.parse_bin import FirmwareImage
from pyremu.utils.tick import yield_cpu

_WFI_MAX_SLEEP = 0.2  # 单次最大睡眠 200ms, 降低 CPU 占用
# virtio 单次 _process_queue 最多处理描述符数 — 拆小批避免单次处理
# 长时间占用主线程 (执行引擎逐指令检查 stop_flag, Ctrl+Q 始终即时响应).
_VIRTIO_PROCESS_BATCH: int = 16

# 阈值判断: ≤2 MiB 的段走 L2 缓存, 更大的段 (如 Linux 内核 Image)
# 绕过 L2 直写 RAM, 避免 cache-line 级逐片处理膨胀到数十秒.
_FAST_LOAD_THRESHOLD = 2 * 1024 * 1024  # 2 MiB

def _wrap_phy_write_for_uart(
    hart_id: int,
    uart,  # UART | None
    orig_write: Callable[[int, bytes], None],
) -> Callable[[int, bytes], None]:
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


class RunStopReason(IntEnum):
    """run() 停止原因 — 由边界检查或引擎退出映射写入 ``Emulator._run_stop_reason``.

    设备暂停事件 (stop_flag 由外部置位) 不属于本枚举的停止:
    标志保留给调用方消费, 保持 NONE.
    """

    NONE = 0  # 设备暂停事件 / 全部 halted / WFI 空闲 / 正常结束
    TIMEOUT = 1  # 时钟源超时 (单轮加速执行边界复核)
    BREAKPOINT = 2  # 断点命中 (引擎 EXIT_BREAKPOINT)
    EBREAK = 3  # 固件执行停机序列 (semihosting SYS_EXIT, 引擎 EXIT_EBREAK)


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
        msg = f"模拟器超时 ({timeout_sec:g}s), 已执行 {cycles} 周期"
        if pc is not None:
            msg += f", 最后 PC={pc:#018x}"
        super().__init__(msg)


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
        # 编译期 AIA 开关 (PYREMU_AIA=1) 已在 PlatformConfig.__post_init__
        # 中统一解析, 此处直接消费解析后的 interrupt_mode.
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

        # 全局性参考时钟周期数
        self._cycle = 0
        self._total_instrs = 0
        # mtime 按指令计数推进 (1 指令 = 1 ns = 1 GHz 内核), mcycle 按时钟源
        # (time.monotonic()) 推进 (CPU_FREQ_HZ).  mtime 与时钟源彻底解耦 — 见
        # _advance_mtime_instr 与 _advance_mcycle.  mtime 速率 = cfg.timebase_freq
        # (与 DTB timebase-frequency 同源), 换算比 = timebase_freq / CPU_FREQ_HZ
        # (10 MHz / 1 GHz = 100 指令/tick).
        self._last_clock_sync: float = time.monotonic()  # mcycle 的时钟源基准
        self._mtime_instr_frac: int = 0  # 指令计数推进 mtime 的亚 tick 余数
        self._prev_all_exec_cnt: int = 0  # 上一轮加速执行 Rust 累积 total_instrs
        # initramfs 物理范围 (load_initrd 设置; None=未挂载)。
        # build_dtb 据此写入 /chosen/linux,initrd-start/end。
        self._initrd: dtb.Initrd | None = None

        # Terminal I/O — 后台线程管理 stdin/stdout, 由 _init_termio 在 UART 就绪后创建
        self._termio = None

        # 共享停止标志 — 由外部实体经 notify_processor() 写入 (debugger
        # Ctrl+Q daemon 等), Rust 引擎逐指令检查. 连续执行无指令配额:
        # Ctrl+Q 是唯一的暂停机制, 暂停时全部 hart 的执行期间状态保留在
        # live ctypes 数组 (_speedup_hart_states) 中.
        self._native_stop_flag = ctypes.c_uint8(0)
        # run() 停止原因 — 由单轮加速执行边界检查 / 引擎退出映射写入
        self._run_stop_reason: RunStopReason = RunStopReason.NONE
        # 时钟源超时 deadline (time.monotonic() 秒), 0 = 禁用
        self._timeout_deadline: float = 0.0

        # 外部中断共享上下文 — stdin daemon 注入 UART 数据后置 pending=1,
        # Rust 循环检测到后退出以便 Python 同步 PLIC 中断到 mip.
        # ctypes.Structure 实例构造时字段自动零初始化 (pending/sources/max_priority=0).
        self._native_ext_irq = FfiExtIrqCtx()

        # WFI 唤醒事件 — 全部 hart 等待时用于阻塞而非轮询
        self._wake_event = threading.Event()

        # 共享总线
        self.bus = Bus(
            ram_size=config.ram_size,
            ram_base=config.ram_base,
            l2_cache=L2Cache(size=config.l2_size)
        )
        # CLINT 设备
        self.clint = CLINT(num_harts=config.num_harts)
        self.clint.base_addr = config.periph.clint_base
        self.bus.add_device(self.clint.base_addr, self.clint)
        # 统一声明外部中断设备为 None, 由 _setup_interrupt_controllers 按配置填充.
        self.plic: PLIC | None = None
        self.imsic: IMSIC | None = None
        self.aplic: APLIC | None = None
        # 外设 + 看门狗 — 外设按配置创建 (base=0 跳过, 保持 None);
        # 看门狗恒创建 (非 None), 二者均由 _setup_peripherals 填充.
        self.uart: UART | None = None
        self.spi: SPI | None = None
        self.i2c: I2C | None = None
        self.gpio: GPIO | None = None
        self.virtio_blk: VirtIOBlock | None = None
        self.watchdog: HartWatchdog  # 恒创建, 类型非 Optional
        self._peripherals: dict[str, object] = {}

        self._setup_interrupt_controllers(config)
        self._setup_peripherals(config)

        # PC-stall watchdog — 检测 M-mode hart 停滞 + S-mode WFI
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
            h.imsic = self.imsic
            self.harts.append(h)

        # 互引用: 每个 hart 持有全 hart 列表, 供 mfence.did 等广播操作
        for h in self.harts:
            h.all_harts = self.harts

        # ---- Phase A: speedup execution state ----
        self._speedup_hart_states: Any = None  # ctypes HartState array
        self._speedup_instr_group: Any = None  # ctypes InstrToBeExec
        self._native_ram_buf: Any = None  # ctypes array from_buffer(bytearray)
        self._bp_addrs: list[int] = []  # breakpoint PCs
        self._warned_infinite_virtio = False  # 防止 virtio 异常循环时重复告警
        # WFI 空闲轮询回调: debugger 设为其 stdin 转发函数, 确保用户输入
        # 能在内核 WFI 等待期间被及时 preload 到 UART RX 并触发中断唤醒。
        self._stdin_forward_callback: Any = None  # () -> bool
        self._init_termio()
        self._init_for_speedup_lib()

    def _setup_interrupt_controllers(self, config: PlatformConfig) -> None:
        """按 interrupt_mode 创建并注册中断控制器.

        前置: ``self.plic`` / ``self.imsic`` / ``self.aplic`` 已在 ``__init__``
        中统一声明为 None, 本函数仅填充当前模式对应的字段.

        - legacy: PLIC (外部中断 + 外设有线路由)
        - AIA:    IMSIC (MSI 中断) + APLIC (有线→MSI 桥)
        """
        if config.interrupt_mode == InterruptMode.AIA:
            p = config.periph
            self.imsic = IMSIC(
                num_harts=config.num_harts,
                m_base_addr=p.imsic_m_base,
                s_base_addr=p.imsic_s_base,
                ipi_target=self.clint,  # 路由 IPI 到 CLINT MSIP
            )
            self.bus.add_device(self.imsic.m_base_addr, self.imsic)

            self.aplic = APLIC(
                imsic=self.imsic,
                base_addr=p.aplic_base,
                num_sources=128,
            )
            self.bus.add_device(self.aplic.base_addr, self.aplic)
            return
        self.plic = PLIC(
            base_addr=config.periph.plic_base,
            num_sources=128,
            num_contexts=2 * config.num_harts,
        )
        self.bus.add_device(self.plic.base_addr, self.plic)

    def _setup_peripherals(self, config: PlatformConfig) -> None:
        """按配置创建并注册外设 (base=0 时跳过) + 看门狗设备.

        前置: ``self.uart`` / ``self.spi`` / ``self.i2c`` / ``self.gpio`` /
        ``self.virtio_blk`` / ``self.watchdog`` 已在 ``__init__`` 中统一声明
        为 None, 本函数仅填充配置启用的字段.
        """
        p = config.periph
        # 中断路由: AIA 模式 → APLIC, legacy 模式 → PLIC
        int_ctrl = self.plic if self.aplic is None else self.aplic
        if p.uart_base:
            self.uart = UART(
                base=p.uart_base,
                plic=int_ctrl,
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
                plic=int_ctrl,
                irq=VIRTIO_BLK_IRQ,
            )
            self.bus.add_device(p.virtio_blk_base, self.virtio_blk)
            self._peripherals["virtio_blk"] = self.virtio_blk

        # 看门狗设备 — 多 hart 停滞检测, DTB 可见
        _wdog_base = p.watchdog_base or 0x1000_4000
        self.watchdog = HartWatchdog(
            self, base=_wdog_base, num_harts=config.num_harts,
        )
        self.bus.add_device(_wdog_base, self.watchdog)
        self._peripherals["watchdog"] = self.watchdog

    def _init_termio(self) -> None:
        """初始化终端 I/O 管理器 (Terminal I/O).

        封装 Rust termio 后台线程
        (或 Python 回退), 持续转发 stdin->UART RX 和 UART TX->stdout。
        UART 此时已构造完毕 (见 __init__ 顺序), 可安全注入。
        """
        if self.uart is None or self._termio is not None:
            return
        try:
            stdin_fd = sys.stdin.fileno()
            stdout_fd = sys.stdout.fileno()
        except (OSError, ValueError):
            return
        self._termio = TerminalIO(
            uart=self.uart,
            wake_event=self._wake_event,
            stdin_fd=stdin_fd,
            stdout_fd=stdout_fd,
            ext_irq=self._native_ext_irq,
        )
        self.uart.termio = self._termio

    def _init_for_speedup_lib(self) -> None:
        """Initialise the acceleration infrastructure.

        Called once during ``__init__``.  Sets up ctypes arrays mirroring the
        Rust ``HartState`` / ``InstrToBeExec`` structs and wraps the bus RAM
        ``bytearray`` as a ctypes pointer the FFI layer can read/write.

        """
        # 隐式禁用: 环境变量未显式设置 且 非交互式 (无 TTY termio)
        # -> 避免后台线程泄漏 + 内存膨胀.
        if self._termio is None or not native_available():
            return

        self._speedup_instr_group = InstrToBeExec()

        num_harts = len(self.harts)
        self._speedup_hart_states = (HartState * num_harts)()
        self._tlb_gen = ctypes.c_uint64(0)
        self._tlb_gen_per_hart = (ctypes.c_uint64 * num_harts)()
        self._tlb_gen_before: int = 0

        self._clint_msip_edge: list[int] = [0] * num_harts
        self._clint_msip_prev: list[int] = [0] * num_harts

        ram = self.bus._ram  # bytearray
        self._native_ram_buf = (ctypes.c_uint8 * len(ram)).from_buffer(ram)  # type: ignore[attr-defined]

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

    def build_dtb(self) -> bytes:
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
            imsic=self.imsic,
            aplic=self.aplic,
            bootargs=self._bootargs,
            initrd=self._initrd,
            reserved_ranges=self._cfg.reserved_memory_ranges,
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

        优先写入 bytearray (Rust读取路径); 仅对小段 (< 2 MiB)
        预热 L2 缓存 (Python 路径受益), 避免大段逐 cache-line 拷贝膨胀到数十秒.
        """
        self.bus.write_ram_direct(addr, code)
        if len(code) <= _FAST_LOAD_THRESHOLD:
            self.bus.write(addr, code)

    def _write_segment(self, seg, vaddr_offset: int, threshold: int) -> None:
        """Write a single firmware segment (data + BSS zero-fill) to RAM.

        始终写入 bytearray (加速动态库的直接读取源);
        对小段同时写入 L2 以预热缓存 (Python 路径受益).
        """
        vaddr = seg.vaddr + vaddr_offset

        # 加速动态库读取路径的数据源
        self.bus.write_ram_direct(vaddr, seg.data)
        # 小段预热 L2
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

    @staticmethod
    def _handle_native_sys_exit(hart: Hart, instr: int) -> int:
        """Execute one instruction deferred by the Rust speedup execution engine.

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
            hart.pc = mask64(hart.pc + advance)
        if not hart._halted:
            check_pending_interrupts(hart)
        return 0 if hart._wfi_woken else 1

    # ----------------------------------------------------------
    #  acceleration fast path (Phase A)
    # ----------------------------------------------------------

    def notify_processor(self) -> None:
        """处理器通知接口 — 外部实体请求处理器暂停执行.

        暴露给所有外部实体 (看门狗设备/磁盘/串口/调试器 Ctrl+Q daemon):
        置 stop_flag (Rust 引擎逐指令检查, 单轮加速执行内即可退出) 并唤醒 WFI 阻塞.
        模拟器自身不内置看门狗线程 — 需要定时/条件暂停的外部设备通过
        本接口通知处理器, 这是打断连续执行的唯一通道.
        """
        self._native_stop_flag.value = 1
        self._wake_event.set()

    def raise_device_irq(self, source: int, pending: bool = True) -> None:
        """设备中断注入接口 — 外部设备 (磁盘/串口/看门狗) 修改中断线电平.

        等价于 QEMU 设备模型的 ``qemu_irq_raise``: 写 PLIC/APLIC 挂起状态,
        并置 ext_irq 通知位, 使引擎在单轮加速执行内即时拉起 MEIP/SEIP、唤醒 WFI
        hart; 单轮加速执行边界 _native_sync_plic_mip 保证 mip 与控制器状态最终一致.
        """
        if self.aplic is not None:
            self.aplic.set_irq(source, pending)
            self._native_ext_irq.pending = 1
            self._wake_event.set()
            return
        if self.plic is not None:
            self.plic.set_irq(source, pending)
            self._native_ext_irq.pending = 1
            self._wake_event.set()

    def _speedup_for_cmd_step(self, active: list[Hart]) -> int:
        # 将 L2 脏行回写到 bytearray, 确保使用动态链接库加速模拟处理器的计算/访存状态期间，
        # 从 bytearray 直接读取指令/数据时能看到 Python 侧的全部写入.
        self.bus.flush_l2()

        # Marshal ALL harts (including halted): Rust needs every state
        # for total_instrs summation and active_hart_num counting.
        # RX daemon 线程已在后台持续 drain_rx() -> UART FIFO, 不消费 _rx_notify.
        # _rx_notify 由 Rust speedup execution engine 检测 -> 快速单轮加速执行退出 -> idle poll 清零.
        # 先把 PLIC 外部中断 (MEIP/SEIP) 同步进各 hart 的 mip —— native 引擎内部
        # 只同步 CLINT (MSIP/MTIP), 不感知 PLIC, 否则 virtio 等外设中断永远到不了 hart。
        self._native_sync_plic_mip()
        for i, hart in enumerate(self.harts):
            marshal_hart(hart, self._speedup_hart_states[i])

        # 保存加速执行前的CLINT MSIP的电平状况，用于辨识MSIP是否在加速前已经陷入
        # 暂停态并且在加速执行期间有MSIP到达
        _pre_speedup_msip = [self.clint._msip[hid] & 1 for hid in range(len(self.harts))]

        dev_info = self._native_marshal_dev()
        pmp_info = self._native_marshal_pmp(active)
        clint_info = self._native_marshal_clint()
        uart_info = self._native_marshal_uart()
        virtio_info = self._native_marshal_virtio()

        # 统一使用 run_parallel (thread-per-hart 并发引擎)。
        # 已弃用: round-robin 切片会在跨 hart IPI / TLB-shootdown 协议上死锁。
        virtio_ffi = run_parallel(
            self._speedup_hart_states,
            len(self.harts),
            self._native_ram_buf,
            self.bus.ram_size,
            self.bus.ram_base,
            self.bus._shadow_base or 0,
            self.bus._shadow_size or 0,
            self._speedup_instr_group,
            pmp=pmp_info,
            clint=clint_info,
            dev=dev_info,
            uart=uart_info,
            virtio=virtio_info,
            watchdog_timeout_ns=self._watchdog_timeout_ns(),
            bp_addrs=self._bp_addrs if self._bp_addrs else None,
            stop_flag=self._native_stop_flag,
            ext_irq=self._native_ext_irq,
            tlb_gen=self._tlb_gen,
            tlb_gen_per_hart=self._tlb_gen_per_hart,
        )

        result = self._speedup_instr_group
        # Level-triggered ext_irq: only clear when ring buffer + UART FIFO
        # are both empty.  Otherwise new data that arrived during the acceleration
        # (between drain_rx and Rust's ext_irq check) would be missed.
        ring_empty = True
        if self._termio is not None:
            rd = self._termio._rx_rd.value
            wr = self._termio._rx_wr.value
            ring_empty = (rd == wr)
        fifo_empty = (self.uart is not None and len(self.uart._rx_fifo) == 0)
        if ring_empty and fifo_empty:
            self._native_ext_irq.pending = 0
            # _rx_notify 对应已完全消费的数据 (ring buffer + UART FIFO 皆空),
            # 安全清零避免下轮的虚假快速退出.
            if self._termio is not None:
                self._termio._rx_notify.value = 0
        for hid in range(len(self.harts)):
            unmarshal_hart(self._speedup_hart_states[hid], self.harts[hid])

        # Rust 批量执行期间内核可能修改页表并执行 SFENCE.VMA,
        # tlb_gen 会递增. 仅在 gen 实际变化时才刷新 Python TLB —
        # 若单轮加速执行内无 SFENCE.VMA, 保留已有 TLB 条目 (QEMU-style).
        tlb_gen_after = self._tlb_gen.value
        if tlb_gen_after != self._tlb_gen_before:
            for hart in self.harts:
                hart.itlb.flush_all()
                hart.dtlb.flush_all()
        self._tlb_gen_before = tlb_gen_after

        # Rust加速执行期间可能直接修改了 bytearray; 使 L2 全部失效,
        # 强制 Python 侧后续读取从 bytearray 重新加载.
        self.bus.invalidate_l2()

        self._native_unmarshal_pmp()
        # TX ring buffer 归档由 _tx_archive daemon 异步处理,
        # 不在此处同步调用 — UART I/O 与单轮加速执行循环完全解耦.
        self._native_unmarshal_clint(clint_info)

        # mtime 已由 Rust 引擎执行期间按指令计数推进 (advance_clock_source),
        # 此处不再重复推进; 仅推进 mcycle (_cycle, 按时钟源).
        self._advance_mcycle()

        # Rust 引擎投递 MSIP 后清除 state.mip.MSIP 以匹配 Python 侧
        # _trap_deliver_mmode 的自清零行为 (见 trap_handler.py line 285-300),
        # 但未回写清除 CLINT._msip。此处同步: 若 hart 的 mip.MSIP 已被 Rust
        # 清零, 且 CLINT MSIP 在投递到加速执行引擎执行前就已是 1 (旧 MSIP 已投递), 则同步
        # 清除 Python CLINT._msip。若 MSIP 在加速执行期间新到达,
        # 则保留 CLINT._msip 不清理 — 否则会在加速执行期间由 hart A 写入、hart B
        # 尚未来得及由 sync_msip 检测到的跨核 IPI 会被永久丢弃 ->TLB-shootdown 死锁。
        for hid in range(len(self.harts)):
            if (
                (self.harts[hid].mip_val & (1 << 3)) == 0
                and self.clint._msip[hid]
                and _pre_speedup_msip[hid]
            ):
                self.clint._msip[hid] = 0
                self.clint._notify_state_change()

        self._native_unmarshal_virtio(virtio_ffi)

        all_exec_cnt = result.total_instrs
        all_exec_cnt = self._native_handle_exit(result, all_exec_cnt)
        return self._native_finalize(all_exec_cnt)

    # ---- _speedup_for_cmd_step 分解: marshal-in / unmarshal-out / 收尾 ----

    def _native_sync_plic_mip(self) -> None:
        """将各 hart 的 PLIC 挂起外部中断合并进其 mip (MEIP bit11 / SEIP bit9)。

        native 引擎内部只 `sync_msip`/`sync_mtip` (CLINT), 不感知 PLIC。设备 MMIO
        (含 virtio QueueNotify 与 PLIC claim/complete/ACK) 已 exit 到 Python 处理,
        故每次进入 native 单轮加速执行前在此从 PLIC 刷新 MEIP/SEIP —— 否则 virtio 完成中断
        永远到不了 hart。MEIP/SEIP 纯由 PLIC 驱动, 直接替换这两位 (保留其余软件位)。
        """
        ext_mask = (1 << 9) | (1 << 11)
        for hart in self.harts:
            if hart._imsic is None and hart._plic is None:
                continue
            ext_mip = 0
            if hart._imsic is not None:
                ext_mip = hart._imsic.get_pending_mip(hart.id)
            if ext_mip == 0 and hart._plic is not None:
                ext_mip = hart._plic.get_pending_mip(hart.id)
            hart.mip_val = (hart.mip_val & ~ext_mask) | ext_mip

    def _native_marshal_dev(self) -> DevInfo:
        """设备 MMIO 地址范围 -> DevInfo (供 Rust 判定 MMIO 直通)."""
        devices = self.bus.devices
        dev_bases = (ctypes.c_uint64 * 0)()
        dev_ends = (ctypes.c_uint64 * 0)()
        if devices:
            dev_bases = (ctypes.c_uint64 * len(devices))()
            dev_ends = (ctypes.c_uint64 * len(devices))()
            for j, (base, dev) in enumerate(sorted(devices.items())):
                dev_bases[j] = base
                dev_ends[j] = base + dev.size
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
        单轮加速执行 PMP 无变化, 故仅在切片确实被 Rust 改写时才同步。
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
        edge-counter (bits 7:1) 来检测使用动态链接库加速执行期间的 MSIP 变化。此处检测
        0->1 跳变并递增 edge counter, 编码为 ``level | (edge << 1)``.
        """
        clint = self.clint
        total = len(self.harts)
        self._clint_mtimecmp_arr = (ctypes.c_uint64 * total)()
        self._clint_msip_arr = (ctypes.c_uint8 * total)()
        if clint is None:
            return ClintInfo(
                mtime=0,
                mtimecmp=self._clint_mtimecmp_arr,
                msip=self._clint_msip_arr,
                base=0,
                timebase_hz=0,
            )

        # mtime 按指令计数推进: native 路径由 Rust advance_clock_source 在执行
        # 期间推进, Python 侧不在此补足流逝时间 (mtime 与时钟源解耦).

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
            mtime=clint.get_mtime(),
            mtimecmp=self._clint_mtimecmp_arr,
            msip=self._clint_msip_arr,
            base=clint.base_addr,
            timebase_hz=self._cfg.timebase_freq,
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
            rx_notify=self._termio._rx_notify,
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

        Rust 单轮加速执行内 inline 处理全部 virtio MMIO 读写; 单轮加速执行结束后
        _native_unmarshal_virtio 将 Rust 修改过的字段同步回 Python vblk。
        本方法必须在每轮单轮加速执行前将当前 Python 状态完整传递给 Rust,
        否则动态寄存器 (queue 描述符、中断状态等) 会在跨单轮加速执行时归零,
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
        单轮加速执行结束后同步 changed fields:
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
        # 必须在 notify_pending 之前处理: 若两者在同一单轮加速执行内触发,
        # 先降低 IRQ 电平再处理新队列, 避免 _lower_irq_if_idle 看到
        # _process_queue 刚写入的 _interrupt_status 而跳过降电平,
        # 导致 _do_complete 时 level 仍为高 ->无限 re-level.
        if virtio_ffi.irq_maybe_lower:
            virtio_ffi.irq_maybe_lower = 0
            vblk._lower_irq_if_idle()

        # QueueNotify: Rust 设置 notify_pending=1 ->Python 处理 virtqueue.
        # 分批处理 (每批最多 _VIRTIO_PROCESS_BATCH 个描述符), 循环直到全部完成.
        # 每批之间 Python 可响应 ctrl+Q; _max_batches 防止异常情况下的死循环.
        if not virtio_ffi.notify_pending:
            return

        virtio_ffi.notify_pending = 0
        _batch_guard = 0
        _max_batches = 256  # 256 x 16 = 4096 描述符, 远超正常 ext4 mount 所需
        while vblk._process_queue(max_descriptors=_VIRTIO_PROCESS_BATCH):
            _batch_guard += 1
            if _batch_guard >= _max_batches:
                if not self._warned_infinite_virtio:
                    self._warned_infinite_virtio = True
                break
        virtio_ffi.interrupt_status = vblk._interrupt_status

    def _native_handle_exit(self, result: InstrToBeExec, all_exec_cnt: int) -> int:
        """处理 native 单轮加速执行退出原因 (ECALL/MMIO/TRAP/BREAKPOINT)."""
        total_harts = len(self.harts)
        if result.exit_hart_id >= total_harts:
            return all_exec_cnt
        exit_hart = self.harts[result.exit_hart_id]
        if result.exit_reason in (EXIT_ECALL, EXIT_MMIO):
            all_exec_cnt += self._handle_native_sys_exit(exit_hart, result.exit_instr)
        elif result.exit_reason == EXIT_TRAP:
            all_exec_cnt += 1
        elif result.exit_reason == EXIT_BREAKPOINT:
            # 断点命中 — PC 已核对 bp_addrs; 记录停止原因供 run()/调试器
            # 区分于设备暂停事件. 若不清醒退出, 重入单轮加速执行会立即再命中, 空转.
            self._run_stop_reason = RunStopReason.BREAKPOINT
        elif result.exit_reason == EXIT_EBREAK:
            # 固件停机序列 (semihosting SYS_EXIT) — 全部 hart 停止.
            self._run_stop_reason = RunStopReason.EBREAK
        elif result.exit_reason == EXIT_TIMEOUT:
            # 时钟源超时 — 看门狗线程在单轮加速执行内推进 mtime 越过 deadline 时
            # 置 stop; 记录原因供 run() 抛出 TimeoutError.
            self._run_stop_reason = RunStopReason.TIMEOUT
        return all_exec_cnt

    def _native_finalize(self, all_exec_cnt: int) -> int:
        """同步 CSR -> WFI 唤醒 -> 看门狗/PC-stall 检测.

        mtime 由加速执行引擎按指令计数推进 (advance_clock_source), 边界处不再
        重复推进; 此处仅同步 mcycle/minstret/time 等 CSR 镜像.
        """
        instr_delta = all_exec_cnt - self._prev_all_exec_cnt
        self._prev_all_exec_cnt = all_exec_cnt

        self.sync_counters(max(0, instr_delta))

        wfi_waiting = 0
        for hart in self.harts:
            if not hart._waiting or hart._halted:
                continue
            try_wfi_wakeup(hart)
            if hart._waiting:
                wfi_waiting += 1

        self._wfi_sleep_if_idle(self.harts, wfi_waiting)

        # 上文 _wfi_sleep_if_idle 可能在 WFI 轮询期间通过 _stdin_forward_callback
        # 把 stdin 字节 preload 到了 UART RX 并置位了 PLIC 中断, 但此时
        # hart.mip 尚未同步 (SEIP/MEIP 仍为 0)。立即同步 PLIC 并重试
        # WFI 唤醒, 避免等待到下轮 _speedup_for_cmd_step() ->_native_sync_plic_mip(),
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

        # 看门狗: 本轮有指令执行则重置, 否则递减; 归零时向 WFI hart 注入 MSIP.
        # 使用 instr_delta (增量) 而非 all_exec_cnt (累计) — 后者永远 > 0,
        # 导致看门狗被每轮重置从不触发, 全 hart WFI 时死锁无法打破.
        if instr_delta > 0:
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


    def _advance_mcycle(self) -> None:
        """mcycle 按 CPU_FREQ_HZ (~1 GHz 内核) 推进, 按时钟源流逝为准."""
        now = time.monotonic()
        elapsed = now - self._last_clock_sync
        self._last_clock_sync = now
        self._cycle += int(elapsed * CPU_FREQ_HZ)

    def _advance_mtime_instr(self, instr_delta: int) -> None:
        """按指令计数推进 mtime (1 指令 = 1 ns = 1 GHz 内核).

        仅纯 Python 路径 (step) 使用: native 路径由 Rust advance_clock_source 按
        每 hart 的 total_instrs 增量推进.  tick 换算 = instr_delta * timebase_freq
        / CPU_FREQ_HZ, 与时钟源无关.  累加亚 tick 余数, 避免单步路径 (每步 1
        指令) 因整除截断丢 tick (100 指令才满 1 tick).
        """
        if self.clint is None or instr_delta <= 0:
            return
        total = instr_delta * self._cfg.timebase_freq + self._mtime_instr_frac
        ticks, self._mtime_instr_frac = divmod(total, CPU_FREQ_HZ)
        if ticks:
            self.clint.tick(ticks)

    def _watchdog_timeout_ns(self) -> int:
        """剩余时钟源超时时间 (纳秒), 注入 FfiWatchdogCtx. 0 = 禁用."""
        if self._timeout_deadline <= 0.0:
            return 0
        remaining = self._timeout_deadline - time.monotonic()
        if remaining <= 0.0:
            return 0
        return int(remaining * 1_000_000_000)


    # ----------------------------------------------------------
    #  默认 WFI 空闲轮询 (无调试器直连路径)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    #  WFI 等待优化
    # ----------------------------------------------------------


    def _wfi_ticks_until_wake(self, active_harts: list) -> int | None:
        """返回最早定时器中断剩余的 tick 数; 无活跃定时器时返回 None.

        检查 CLINT mtimecmp 和 SSTC stimecmp CSR 两个定时器源,
        取最早到期者.
        """
        now = self.clint._mtime
        best = None
        for h in active_harts:
            cmp = self.clint._mtimecmp[h.id]
            if cmp > 0 and cmp >= now:
                rem = max(0, cmp - now)
                if best is None or rem < best:
                    best = rem
            sstc = h._csr_read_raw("stimecmp")
            if sstc > 0 and sstc >= now:
                rem = max(0, sstc - now)
                if best is None or rem < best:
                    best = rem
        return best

    def _wfi_sleep_if_idle(self, active: list, waiting_count: int) -> None:
        """全部 hart WFI 时阻塞等待, 避免 CPU 100% 空转.

        事件驱动: 键盘输入由独立 daemon 线程经 select(stdin) 分发 — 先判断
        Ctrl+Q 是否暂停 (调试器), 其余转发到 UART RX 并置外部中断
        (legacy: PLIC; AIA: APLIC→IMSIC。两者经 UART 的 set_irq 统一接口路由),
        然后 set(_wake_event) 唤醒本线程; native termio 路径亦经 drain_rx ->
        _wake_event.set() 唤醒。故此处仅等待 _wake_event, 无需 select(stdin)
        (stdin 由 daemon 独占, 避免与主线程竞争)。
        mtime 由 Rust post_instr_checks 内联推进 (1 指令 = 1 tick).
        """
        if waiting_count < len(active) or waiting_count == 0:
            return

        remaining = self._wfi_ticks_until_wake(active)
        if remaining is not None and remaining > 0:
            sleep_sec = min(remaining / 10_000_000, _WFI_MAX_SLEEP)
        elif remaining is not None:
            return
        else:
            sleep_sec = max(1, int(WFI_WATCHDOG_MS)) * 0.001

        if sleep_sec <= 0.0:
            return

        self._wake_event.wait(timeout=sleep_sec)

    # ----------------------------------------------------------
    #  执行
    # ----------------------------------------------------------

    def sync_counters(self, instr_delta: int = 0) -> None:
        """同步硬件计数器 CSR (cycle/time/instret).

        mtime 按指令计数、mcycle 按时钟源推进 (见 _advance_mtime_instr /
        _advance_mcycle); 本函数仅把当前值写入各 hart 的 CSR 镜像.
        """
        self._total_instrs += instr_delta
        cycle_val = mask64(self._cycle)
        time_val = mask64(self.clint.get_mtime())
        for h in self.harts:
            h._csr_write_raw("mcycle", cycle_val)
            h._csr_write_raw("cycle", cycle_val)
            # Per-hart instruction count — harts in WFI waiting state
            # don't execute, so their count doesn't advance.
            h._csr_write_raw("minstret", mask64(h._total_instrs))
            h._csr_write_raw("instret", mask64(h._total_instrs))
            h._csr_write_raw("time", time_val)

    def step(self) -> int:
        """
        QEMU-style per-instruction execution: 每个周期让所有 hart 各
        执行一条指令, 每条指令前转发 stdin->UART RX、每条指令后检查中断,
        CLINT 时钟按指令计数推进 (与时钟源解耦).

        指令单轮加速执行设计已去除 — step() 恒为精确单指令 (纯 Python 路径);
        高速连续执行由 run() (native 引擎) 承担.

        Returns:
            本轮执行的指令数.
        """
        active = [h for h in self.harts if not h._halted]

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
                    hart, mask64(hart.pc + 2)
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

            # 正常顺序执行 -> 推进 PC
            if advance != 0 and hart.pc == pc_before:
                hart.pc = mask64(hart.pc + advance)

            # 指令边界 — 检查中断
            check_pending_interrupts(hart)

            # WFI 唤醒路径 (中断 handler -> mret -> 回到正常流) 不计入指令数,
            # 以保证 _total_instrs 反映固件实际执行的非中断上下文指令.
            if not hart._wfi_woken:
                hart._total_instrs += 1
                all_exec_cnt += 1

        self._advance_mcycle()
        self._advance_mtime_instr(all_exec_cnt)
        self.sync_counters(all_exec_cnt)

        # WFI 优化: 全部未 halted 的 hart 处于 WFI 等待时, 阻塞而非轮询
        self._wfi_sleep_if_idle(active, wfi_waiting)

        return all_exec_cnt

    def _check_bp_hit_py(self) -> bool:
        """纯 Python 连续执行下的地址断点判定 (native 在 Rust 内联比对).

        ``_bp_addrs`` 由调试器在 ``run()`` 前注入 (含符号物理地址对应的 VA
        副本); 任一活跃 hart 的 PC 命中即返回 True, 由 ``run()`` 置
        ``BREAKPOINT`` 停止 — 与 native 引擎 pre-execution 断点语义一致.
        """
        if not self._bp_addrs:
            return False
        for h in self.harts:
            if not h._halted and h.pc in self._bp_addrs:
                return True
        return False

    def run(
        self,
        *,
        timeout: float = 1800.0,
        yield_every: int = 10000,
        yield_interval: float = 0.001,
    ) -> int:
        """连续执行, 直至停止条件满足.

        引擎无指令配额, 持续执行. 停止条件:
        - 固件执行停机序列 (semihosting SYS_EXIT) — 全部 hart 停止
        - 断点命中 (EXIT_BREAKPOINT)
        - 全部 hart halted
        - 超时 (*timeout*)

        全部 hart WFI 空闲时 Python 侧阻塞等待唤醒 (stdin 转发在此进行).
        执行中暂停由外部设备经 :meth:`notify_processor` 置 stop_flag:
        引擎逐指令检查、即时退出, 状态保留在 live 数组中, 标志留给
        调用方消费 — 不属于上述停止条件.

        Args:
            timeout: 超时秒数, 默认 1800 (30 分钟). 设为 0 禁用.
            yield_every: 每执行 *yield_every* 条指令后让出 CPU,
                         默认 10000.
                         0 = 不限速 (跑满 CPU).
            yield_interval: 每次让出等待秒数, 默认 0.001 (1ms).

        Returns:
            实际执行的 clock-source 周期数 (self._cycle).

        Raises:
            TimeoutError: 当 *timeout* 秒数被超过.
        """
        # 消费上次未处理的设备暂停标志; 执行期间新置位的标志留给调用方.
        self._native_stop_flag.value = 0
        self._run_stop_reason = RunStopReason.NONE

        # 超时基于时钟源 (time.monotonic()); mtime 按指令计数推进, 不能作超时基准.
        if timeout > 0:
            self._timeout_deadline = time.monotonic() + timeout
        else:
            self._timeout_deadline = 0.0
        do_yield = yield_every > 0
        instr_start = self._total_instrs
        use_native = self._speedup_hart_states is not None
        last_yield = instr_start

        while True:
            active = [h for h in self.harts if not h._halted]
            if not active:
                break
            # 时钟源超时检查
            if self._timeout_deadline and time.monotonic() >= self._timeout_deadline:
                self._run_stop_reason = RunStopReason.TIMEOUT
                raise TimeoutError(
                    timeout,
                    self._total_instrs - instr_start,
                    self.harts[0].pc if self.harts else None,
                )
            if self._stdin_forward_callback is not None:
                self._stdin_forward_callback()

            if use_native:
                self._speedup_for_cmd_step(active)
                # 时钟源超时 — 引擎已置 TIMEOUT, 抛出 (死循环/连续执行兜底).
                if self._run_stop_reason == RunStopReason.TIMEOUT:
                    raise TimeoutError(
                        timeout,
                        self._total_instrs - instr_start,
                        self.harts[0].pc if self.harts else None,
                    )
                # 设备暂停事件 — 标志保留给调用方消费; 断点/停机 — 重入
                # 单轮加速执行会立即再命中, 必须在此返回.
                if self._native_stop_flag.value != 0 or \
                self._run_stop_reason in (
                    RunStopReason.BREAKPOINT,
                    RunStopReason.EBREAK,
                ):
                    break
            else:
                # 纯 Python 路径 (native 不可用) — 逐指令, 无看门狗线程
                if self._check_bp_hit_py():
                    self._run_stop_reason = RunStopReason.BREAKPOINT
                    break
                self.step()

            if do_yield and self._total_instrs - last_yield >= yield_every:
                yield_cpu(yield_interval)
                last_yield = self._total_instrs

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
