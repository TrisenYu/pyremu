#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 17:00:47
# Last modified at 2026/10/24 星期六

"""
多核 RISC-V 模拟器 — 管理 Bus、CLINT、外设、多个 Hart 的执行流程.

上电后 hart 从可配置的复位向量开始执行. 裸金属固件应被直接加载到复位向量
对应的物理地址.

平台拓扑通过 PlatformConfig 描述, 外设按配置自动创建并注册. 可选生成设备树
二进制数据 (DTB) 供固件以 a1 寄存器接收.

装配、装载、执行循环与加速执行引擎的状态同步分别实现在 pyremu/emu 的四个
模块中, 本模块只保留构造流程与对外的公开接口.

Usage:
    from pyremu.platform import PlatformConfig

    cfg = PlatformConfig.sifive_u54()
    emu = Emulator(cfg)
    emu.load_code(addr=0x80000000, code=my_firmware)
    emu.step()        # 所有 hart 各执行一条指令
    emu.run()         # 连续执行, 直至固件执行停机序列 (semihosting SYS_EXIT)
"""

import ctypes
import threading
import time
from typing import Any

from pyremu._native import FfiExtIrqCtx
from pyremu.core.decoder import Hart
from pyremu.emu.load import LoadOps
from pyremu.emu.native import NativeOps
from pyremu.emu.run import RunOps, RunStopReason, TimeoutError
from pyremu.emu.setup import SetupOps
from pyremu.interrupt.aplic import APLIC
from pyremu.interrupt.clint import CLINT
from pyremu.interrupt.imsic import IMSIC
from pyremu.interrupt.irq_sink import DeviceIrqSink
from pyremu.interrupt.plic import PLIC
from pyremu.peripheral import (
    CRNG,
    GPIO,
    I2C,
    SPI,
    UART,
    VirtIOBlock,
    VirtIONet,
)
from pyremu.peripheral.watchdog import HartWatchdog
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils import dtb
from pyremu.utils.str_aux import fmt_hexdump
from pyremu.utils.wrapper import silent_on_err

__all__ = ["Emulator", "RunStopReason", "TimeoutError"]


class Emulator(LoadOps, NativeOps, RunOps, SetupOps):
    """多核 RISC-V 模拟器.

    管理 N 个 hart, 共享总线、中断控制器和外设.
    提供 step / run 执行循环及状态检查辅助方法.
    """

    @staticmethod
    def _build_default_config(**kwargs: int) -> PlatformConfig:
        """从 kwargs 设置平台配置; 缺少时使用 qemu_virt 作为默认配置."""
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

        兼容 kwargs 传参方式: ``Emulator(num_harts=4, ram_base=0, ...)`` 等价于
        用对应字段构建 Minimal PlatformConfig.
        *config* 传入时忽略 **kwargs.
        """
        if config is None:
            config = self._build_default_config(**kwargs)
        # 环境变量 PYREMU_AIA 已在 PlatformConfig.__post_init__ 中解析, 此处直接
        # 使用解析后的 interrupt_mode.
        self._cfg = config
        # bootargs 为 None 时取默认值: 内核控制台输出与信任 bootloader 随机数
        # 种子; 传入字符串时以传入值为准.
        self._dtb_addr: int | None = None  # 最近一次 load_dtb 的地址
        self._dtb_size: int = 0  # 最近一次 load_dtb 写入的字节数
        self._bootargs = (
            bootargs
            if bootargs is not None
            else (
                "earlycon=sbi console=ttySIF0 "
                "random.trust_bootloader=on "
                "ro init=/bin/zsh TERM=vt100"
            )
        )

        # 全局性参考时钟周期数: 按真实流逝记录, 供 run() 返回值与 cycle 属性
        # 使用, 不写入 mcycle.
        self._cycle = 0
        self._total_instrs = 0
        # mtime 在加速执行路径按各 hart 的指令增量推进, 见 Rust 的
        # advance_clock_source, 每条指令计 NS_PER_INSTR 纳秒; 纯 Python 单步路径
        # 同样按指令计数推进, 见 _advance_mtime_instr. 两条路径都不按真实流逝的
        # 时间推进: 低速模拟下时钟若跟随真实时间, 内核 HZ=250 的 tick 得不到
        # 足够的指令预算, 会长期停留于等待. mcycle 与 minstret 按各 hart 的指令
        # 数折算, 见 sync_counters. mtime 的速率取 cfg.timebase_freq, 与设备树的
        # timebase-frequency 同源, 换算比为 timebase_freq * NS_PER_INSTR / 1e9,
        # 10 MHz 时每个 tick 对应 5 条指令.
        self._last_clock_sync: float = time.monotonic()  # _cycle 的时钟源基准
        self._mtime_instr_frac: int = 0  # 指令计数推进 mtime 的亚 tick 余数
        self._prev_all_exec_cnt: int = 0  # 上一轮加速执行累计的指令数
        # initramfs 的物理范围, 由 load_initrd 设置, 未加载时为 None;
        # build_dtb 据此写入 /chosen/linux,initrd-start 与 linux,initrd-end.
        self._initrd: dtb.Initrd | None = None

        # 终端 I/O 管理器, 由 _init_termio 在 UART 就绪后创建; 后台线程管理标准
        # 输入与标准输出.
        self._termio = None
        # 资源是否已释放, 供 close() 幂等使用. 生命周期结束时须显式关闭 termio
        # 的管道与 UART 日志文件, 否则每个实例泄漏 6 个文件描述符; 进程内的
        # 描述符编号达到 1024 之后 select() 无法监视.
        self._closed = False

        # 共享停止标志: 外部实体经 notify_processor() 写入, 例如调试器的 Ctrl+Q
        # 处理, 加速执行引擎逐指令检查. 连续执行不设指令配额, 该标志是暂停执行
        # 的通道; 暂停时各 hart 的执行状态保留在跨轮保留的 ctypes 数组
        # _speedup_hart_states 中.
        self._native_stop_flag = ctypes.c_uint8(0)
        # run() 的停止原因, 由单轮加速执行的边界检查与引擎退出映射写入
        self._run_stop_reason: RunStopReason = RunStopReason.NONE
        # 时钟源超时的截止时刻, 单位为 time.monotonic() 的秒数, 取 0 表示禁用
        self._timeout_deadline: float = 0.0

        # 外部中断共享上下文: 接收守护线程写入 UART 数据后置 pending 为 1,
        # Rust 循环检出后退出本轮, 以便 Python 把 PLIC 中断同步到 mip.
        # ctypes.Structure 实例构造时字段自动清零.
        self._native_ext_irq = FfiExtIrqCtx()

        # WFI 唤醒事件 — 全部 hart 等待时用于阻塞而非轮询
        self._wake_event = threading.Event()

        # 共享总线
        self.bus = self._create_bus(config)
        # CLINT 设备
        self.clint = CLINT(num_harts=config.num_harts)
        self.clint.base_addr = config.periph.clint_base
        self.bus.add_device(self.clint.base_addr, self.clint)
        # 统一声明外部中断设备为 None, 由 _setup_interrupt_controllers 按配置填充.
        self.plic: PLIC | None = None
        self.imsic: IMSIC | None = None
        self.aplic: APLIC | None = None
        # 设备中断投递目标: 设备经 on_irq 持有本对象, 不持有 Emulator 的绑定
        # 方法, 理由见 DeviceIrqSink. 中断源编号与中断控制器的对应关系由
        # _setup_interrupt_controllers 经 set_fallback 与 bind 登记.
        self._irq_sink = DeviceIrqSink(self._native_ext_irq, self._wake_event)
        # 外设与看门狗: 外设按配置创建, 基址取 0 时保持 None; 看门狗恒创建.
        # 二者均由 _setup_peripherals 填充.
        self.uart: UART | None = None
        self.spi: SPI | None = None
        self.i2c: I2C | None = None
        self.gpio: GPIO | None = None
        self.virtio_blk: VirtIOBlock | None = None
        self.virtio_net_s: VirtIONet | None = None
        self.virtio_net_m: VirtIONet | None = None
        self.watchdog: HartWatchdog  # 恒创建, 类型非 Optional
        self.crng: CRNG | None = None
        self._peripherals: dict[str, object] = {}
        # 全部 hart, 由 _create_harts 原地填充. 本列表先于 hart 创建而存在, 使
        # 早于 hart 创建的外设可直接持有它.
        self.harts: list[Hart] = []

        self._setup_interrupt_controllers(config)
        self._setup_peripherals(config)
        self._create_harts(config)

        # 加速执行的状态
        self._speedup_hart_states: Any = None  # ctypes HartState 数组
        self._speedup_instr_group: Any = None  # ctypes InstrToBeExec
        self._native_ram_ptr: int = 0  # RAM 缓冲区首地址
        self._bp_addrs: list[int] = []  # 断点地址表
        self._warned_infinite_virtio = False  # 防止 virtio 异常循环时重复告警
        # WFI 空闲轮询回调: 调试器设为其标准输入转发函数, 使内核处于 WFI 等待
        # 期间用户输入仍能及时写入 UART 接收路径并触发中断唤醒.
        self._stdin_forward_callback: Any = None  # () -> bool
        self._init_termio()
        self._init_for_speedup_lib()

    # ----------------------------------------------------------
    #  公开属性
    # ----------------------------------------------------------

    @property
    def config(self) -> PlatformConfig:
        return self._cfg

    @property
    def peripherals(self) -> dict[str, object]:
        return self._peripherals

    @property
    def dtb_addr(self) -> int | None:
        """最近一次写入设备树的物理地址; 未写入时为 None."""
        return self._dtb_addr

    @property
    def dtb_size(self) -> int:
        """最近一次写入的设备树字节数; 未写入时为 0."""
        return self._dtb_size

    @property
    def initrd(self) -> dtb.Initrd | None:
        """initramfs 的物理范围; 未加载时为 None."""
        return self._initrd

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

    @property
    def irq_route(self) -> dict[int, APLIC | PLIC]:
        """中断源编号与中断控制器的对应关系, 由 _setup_interrupt_controllers 填充."""
        return self._irq_sink.irq_route

    # ----------------------------------------------------------
    #  生命周期与外部接口
    # ----------------------------------------------------------

    def close(self) -> None:
        """释放本 Emulator 占用的全部输入输出资源; 重复调用不产生效果.

        调试器退出、测试清理与对象回收时调用. TerminalIO.stop() 停止后台线程
        并关闭 3 个管道, 共 6 个文件描述符; 随后关闭 UART 的 hart 日志文件.
        不释放时每个实例占用 6 个文件描述符, 进程内的描述符编号达到 1024 之后
        select() 无法监视.
        """
        if self._closed:
            return
        self._closed = True
        if self._termio is not None:
            self._termio.stop()
        if self.uart is not None:
            self.uart.close_logs()

    @silent_on_err
    def __del__(self) -> None:
        """对象回收时释放资源: 未显式调用 close() 的实例在此释放文件描述符."""
        self.close()

    def notify_processor(self) -> None:
        """通知处理器暂停执行.

        供看门狗设备、磁盘、串口与调试器的 Ctrl+Q 处理等外部实体调用: 置
        stop_flag, 加速执行引擎逐指令检查该标志, 在本轮加速执行内退出, 并唤醒
        处于 WFI 等待的 hart. 模拟器自身不设定时停机的线程, 需要按时间或条件
        暂停的外部设备都经本接口通知处理器; 这是打断连续执行的唯一通道.
        """
        self._native_stop_flag.value = 1
        self._wake_event.set()

    def add_breakpoint(self, addr: int) -> None:
        """登记连续执行的停止地址.

        地址命中时 ``run()`` 以 ``BREAKPOINT`` 停止. 地址表与调试器注入的断点相同:
        纯 Python 循环在每步之前比对, 加速执行引擎在每条指令执行之前比对.
        """
        if addr not in self._bp_addrs:
            self._bp_addrs.append(addr)

    def raise_device_irq(self, source: int, pending: bool = True) -> None:
        """修改中断线的电平, 供磁盘、串口、网卡与看门狗等外部调用方使用.

        转交设备中断投递目标处理: 后者按中断源编号选中处理该中断源的控制器,
        写入其挂起状态并置 ext_irq 通知位. 设备自身持有的是 ``self._irq_sink``,
        理由见 DeviceIrqSink; 本方法供 Emulator 之外的调用方使用.
        """
        self._irq_sink(source, pending)

    # ----------------------------------------------------------
    #  状态导出
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

    def mem_hexdump(self, addr: int, size: int) -> str:
        """返回物理内存的十六进制转储字符串; 读取失败返回提示文本."""
        data = self.bus.try_read(addr, size)
        if data is None:
            return "(无法读取该地址)"
        return fmt_hexdump(data, addr)
