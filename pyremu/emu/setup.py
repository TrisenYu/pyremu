#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""SetupOps — 共享总线、中断控制器、外设、hart 与加速执行引擎的装配.

Emulator 的构造流程先声明共享字段, 再依次调用这里各步骤.
"""

import ctypes
import sys
from typing import Any, Callable

from pyremu._native import (
    FfiUartCtx,
    native_available,
    termio_set_emu_stop_flag,
)
from pyremu.core.decoder import Hart
from pyremu.core.hart import HartState, InstrToBeExec
from pyremu.core.mem_check_aux import inject_memory_backend
from pyremu.interrupt.aplic import APLIC
from pyremu.interrupt.imsic import IID_EXT_MIN, IMSIC
from pyremu.interrupt.plic import PLIC
from pyremu.memory.bus import Bus
from pyremu.memory.l2cache import L2Cache
from pyremu.peripheral import (
    CRNG,
    GPIO,
    I2C,
    SPI,
    TerminalIO,
    UART,
    VirtIOBlock,
    VirtIONet,
)
from pyremu.peripheral.uart import UART_IRQ
from pyremu.peripheral.virtio_blk import VIRTIO_BLK_IRQ
from pyremu.peripheral.virtio_net import (
    DEFAULT_VIRTIO_NET_M_MAC,
    DEFAULT_VIRTIO_NET_S_MAC,
    VIRTIO_NET_M_IRQ,
    VIRTIO_NET_S_IRQ,
)
from pyremu.peripheral.watchdog import HartWatchdog
from pyremu.platform import InterruptMode, PlatformConfig


def _wrap_phy_write_for_uart(
    hart_id: int,
    uart,  # UART | None
    orig_write: Callable[[int, bytes], None],
) -> Callable[[int, bytes], None]:
    """
    hart 写入 UART MMIO 范围时标记写者身份.

    UART 据此把 TXDATA 字节归入该 hart 的日志文件
    本函数不做行缓冲, 只用于多 hart 输出分流归档.
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


class SetupOps:
    """初始化步骤: 共享总线、中断控制器、外设、hart 与加速执行引擎的初始化状态."""

    @staticmethod
    def _create_bus(config: PlatformConfig) -> Bus:
        """创建共享总线, 并按配置注册引导只读存储器."""
        bus = Bus(
            ram_size=config.ram_size,
            ram_base=config.ram_base,
            l2_cache=L2Cache(size=config.l2_size),
        )
        # 基址取 0 时不注册引导只读存储器.
        if config.rom_base:
            bus.add_boot_rom(config.rom_base, config.rom_size)
        return bus

    def _setup_interrupt_controllers(self, config: PlatformConfig) -> None:
        """按 interrupt_mode 创建并注册中断控制器.

        ``self.plic`` / ``self.imsic`` / ``self.aplic`` 由 Emulator 的构造流程声明,
        本函数仅填充当前模式对应的字段.

        - legacy: PLIC, 承担外部中断与外设有线路由
        - AIA:    IMSIC 承担 MSI 中断, APLIC S 域与 APLIC M 域把有线中断转成 MSI;
                  M 域实例登记在设备中断投递目标的中断源表下
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
                base_addr=p.aplic_s_base,
                num_sources=128,
            )
            self.bus.add_device(self.aplic.base_addr, self.aplic)
            # 未登记的中断源交给 S 域 APLIC.
            self._irq_sink.set_fallback(self.aplic)

            if p.aplic_m_base:
                aplic_m = APLIC(
                    imsic=self.imsic,
                    base_addr=p.aplic_m_base,
                    num_sources=128,
                    is_mmode=True,
                )
                self.bus.add_device(aplic_m.base_addr, aplic_m)
                # 该域没有软件写 MMIO 配置中断源路由: 中断源停留在 inactive 时
                # set_irq 直接丢弃, 故在此预置网卡中断源的投递目标与使能.
                # 投递 identity 取外部中断起始值, 不取中断源编号本身: 源号 3 与
                # M 文件上的软件中断 identity 相同, 该 identity 的 pending 位由
                # IPI 快路径不经使能位直接投递, 网卡置位会被当作 IPI 认领.
                aplic_m.bind_source(VIRTIO_NET_M_IRQ, 0, IID_EXT_MIN)
                self._irq_sink.bind(VIRTIO_NET_M_IRQ, aplic_m)
            return
        self.plic = PLIC(
            base_addr=config.periph.plic_base,
            num_sources=128,
            num_contexts=2 * config.num_harts,
        )
        self.bus.add_device(self.plic.base_addr, self.plic)
        # legacy 模式下全部中断源交给 PLIC.
        self._irq_sink.set_fallback(self.plic)

    def _setup_peripherals(self, config: PlatformConfig) -> None:
        """按配置创建并注册外设与看门狗设备; 基址取 0 的设备跳过.

        ``self.uart`` / ``self.spi`` / ``self.i2c`` / ``self.gpio`` /
        ``self.virtio_blk`` / ``self.virtio_net_s`` / ``self.virtio_net_m`` /
        ``self.watchdog`` 由 Emulator 的构造流程声明, 本函数仅填充配置启用的
        字段.
        """
        p = config.periph
        # 中断路由目标: AIA 模式为 APLIC, legacy 模式为 PLIC
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
                # 主存访问路径取自 MainMem 而不取 Bus: Bus 持有本设备, 设备再持有 Bus
                # 的绑定方法会使二者之间形成环形引用, 见 Bus.main_mem.
                mem_read=self.bus.main_mem.read,
                mem_write=self.bus.main_mem.write,
                # 直接调 plic.set_irq 会略过通知加速执行引擎与唤醒 WFI 等待的
                # hart 这两步. 此处不得改传 raise_device_irq: 绑定方法持有
                # Emulator, 会与 Emulator 对设备的持有构成环形引用
                irq_sink=(VIRTIO_BLK_IRQ, self._irq_sink),
            )
            self.bus.add_device(p.virtio_blk_base, self.virtio_blk)
            self._peripherals["virtio_blk"] = self.virtio_blk
        if p.virtio_net_s_base:
            self.virtio_net_s = self._create_virtio_net(
                p.virtio_net_s_base,
                VIRTIO_NET_S_IRQ,
                DEFAULT_VIRTIO_NET_S_MAC,
            )
            self._peripherals["virtio_net_s"] = self.virtio_net_s
        if p.virtio_net_m_base:
            self.virtio_net_m = self._create_virtio_net(
                p.virtio_net_m_base,
                VIRTIO_NET_M_IRQ,
                DEFAULT_VIRTIO_NET_M_MAC,
            )
            self._peripherals["virtio_net_m"] = self.virtio_net_m

        # 看门狗设备: 多 hart 停滞检测, DTB 可见. 经 clint 与 wake_event 两个对象
        # 构造, 不传 Emulator, 也不传 hart 列表: 本设备持有其中任何一个都会与
        # Emulator 对设备的持有构成环形引用. hart 列表在每次 tick 时随参数传入.
        _wdog_base = p.watchdog_base or 0x1000_4000
        self.watchdog = HartWatchdog(
            clint=self.clint,
            wake_event=self._wake_event,
            base=_wdog_base,
            num_harts=config.num_harts,
        )
        self.bus.add_device(_wdog_base, self.watchdog)
        self._peripherals["watchdog"] = self.watchdog

        # 随机数生成器: 运行期熵源, DTB 可见, 设备树节点名为 pyremu,crng.
        if p.crng_base:
            self.crng = CRNG(base=p.crng_base)
            self.bus.add_device(p.crng_base, self.crng)
            self._peripherals["crng"] = self.crng

    def _create_virtio_net(
        self,
        base: int,
        irq: int,
        mac: bytes,
    ) -> VirtIONet:
        """按 MMIO 基址、中断源编号与 MAC 创建一个 virtio-net 实例并注册到总线.

        中断经设备中断投递目标投递: 该目标除置位中断控制器外还通知加速执行引擎并
        唤醒 WFI 阻塞的 hart, 直接调控制器 set_irq 会略过这两步.

        主存访问路径取自 MainMem 而不取 Bus: Bus 持有本设备, 设备再持有 Bus 的绑定
        方法会使二者之间形成环形引用, 见 Bus.main_mem.
        """
        net = VirtIONet(
            mem_read=self.bus.main_mem.read,
            mem_write=self.bus.main_mem.write,
            mac=mac,
            irq_sink=(irq, self._irq_sink),
        )
        self.bus.add_device(base, net)
        return net

    def _create_harts(self, config: PlatformConfig) -> None:
        """按配置创建全部 hart, 注入内存后端与中断控制器, 并原地填入 ``self.harts``.

        调用本步时 ``self.bus`` / ``self.clint`` / ``self.uart`` 与
        ``_setup_interrupt_controllers`` 填充的控制器字段均已就绪;
        ``self.harts`` 由构造流程置为空列表, 本函数只向其追加, 不另建列表.
        """
        for i in range(config.num_harts):
            h = Hart(id=i, pmp_entries=config.pmp_entries)
            h.pc = config.prog_cnt
            inject_memory_backend(
                h,
                self.bus.read,
                _wrap_phy_write_for_uart(h.id, self.uart, self.bus.write),
            )
            h.bus = self.bus
            h.interrupt_ctrl = self.clint
            h.plic = self.plic
            h.imsic = self.imsic
            self.harts.append(h)

        # 互引用: 每个 hart 持有全 hart 列表, 供 mfence.did 等广播操作使用.
        for h in self.harts:
            h.all_harts = self.harts

    def _init_termio(self) -> None:
        """初始化终端 I/O 管理器.

        管理器封装 Rust termio 后台线程或 Python 回退实现, 持续把宿主标准输入的
        字节写入 UART 接收路径, 并把 UART 发送的字节输出到宿主标准输出. 调用本步
        时 UART 已构造完毕, 可安全注入.
        """
        if self.uart is None or self._termio is not None:
            return
        try:
            stdin_fd = sys.stdin.fileno()
            stdout_fd = sys.stdout.fileno()
        except OSError, ValueError:
            return
        self._termio = TerminalIO(
            uart=self.uart,
            wake_event=self._wake_event,
            stdio_fds=(stdin_fd, stdout_fd),
            ext_irq=self._native_ext_irq,
            # RX 排空路径经设备中断投递目标同步 PLIC: 设备线程必须把中断写入
            # Rust 正在读取的同一批数组, 否则单轮加速执行内 Rust 处理 PLIC 状态
            # 时看不到新到达的 UART 数据, plic_recompute_mip 覆盖 SEIP, UART
            # 中断丢失, 输入停滞.
            on_irq=self._irq_sink,
        )
        self.uart.termio = self._termio
        # 把 CPU 引擎停止标志的地址注入 termio 线程: Ctrl+Q 时由 termio 直接
        # 置位, 不经 SIGINT 交给 Python 信号处理器再调用 notify_processor. 主线程
        # 阻塞于 run_parallel 时信号处理器不执行, 直接置位使加速执行引擎的看门狗
        # 仍能即时感知停止请求.
        termio_set_emu_stop_flag(ctypes.addressof(self._native_stop_flag))

    def _init_for_speedup_lib(self) -> None:
        """初始化加速执行所需的状态.

        建立与 Rust ``HartState`` / ``InstrToBeExec`` 结构体对应的 ctypes 数组,
        并把总线 RAM 的 bytearray 表示为 FFI 层可读写的 ctypes 指针. Emulator 的
        构造流程在 termio 与 hart 就绪后调用一次.
        """
        # 环境变量未显式设置且没有 TTY termio 时不启用加速执行, 避免后台线程
        # 泄漏与内存膨胀.
        self._native_uart_ffi: Any = None
        if self._termio is None or not native_available():
            return

        self._speedup_instr_group = InstrToBeExec()

        num_harts = len(self.harts)
        self._speedup_hart_states = (HartState * num_harts)()
        # IMSIC 的中断写入目标取引擎正在读取的这份 hart 状态数组, 与 PLIC 的
        # _ffi_pending 与 _ffi_level 同理: 设备线程经 set_ip_number 注入的 eip
        # 必须落在引擎正在读的这份数组上, 否则 AIA 模式下外部中断要等到轮边界
        # _native_sync_plic_mip 才可见, 输入回显延迟以百毫秒计.
        if self.imsic is not None:
            self.imsic.bind_ffi_states(self._speedup_hart_states)
        # TLB 刷新计数的两处存放位置: 全局计数由总线持有, 各 hart 上次见到的取值
        # 由本数组持有. 两者跨 FFI 调用保留, 引擎据此在两次调用之间沿用既有的翻译
        # 条目; 使条目失效的动作无论发生在哪一侧都计入该计数.
        self._tlb_flush_cnt_per_hart = (ctypes.c_uint64 * num_harts)()
        self._tlb_flush_cnt_before: int = self.bus.tlb_flush_cnt.value

        self._clint_msip_edge: list[int] = [0] * num_harts
        self._clint_msip_prev: list[int] = [0] * num_harts
        self._clint_mtimecmp_arr = (ctypes.c_uint64 * 0)()
        self._clint_msip_arr = (ctypes.c_uint8 * 0)()

        ram = self.bus._ram  # bytearray
        # 首地址用 ctypes.addressof 取得; 取地址不用 ctypes.cast, 后者会与所转换的缓冲区
        # 构成引用环.
        self._native_ram_ptr = ctypes.addressof(ctypes.c_uint8.from_buffer(ram))

        self._pmp_num: int = 0
        self._pmp_cfg_buf = (ctypes.c_uint8 * 0)()
        self._pmp_addr_buf = (ctypes.c_uint64 * 0)()

        # PLIC 的 ctypes 数组跨轮保留, 由 Rust 直接读写. 与 CLINT 每轮重新分配
        # 不同, 这几个数组必须跨轮保持同一份, 否则设备线程写入的数组与
        # run_parallel 读取的不是同一份, 单轮内新到达的设备中断对 Rust 不可见.
        # pending 与 level 由 PLIC 对象持有, 即 _ffi_pending 与 _ffi_level,
        # set_irq 时直接写入; 本处仅保留 priority、enable、threshold 与
        # claimed, 由 _native_marshal_plic 在尺寸变化时才重新分配.
        self._plic_base: int = 0
        self._plic_num_sources: int = 0
        self._plic_num_contexts: int = 0
        self._plic_priority_arr = (ctypes.c_uint8 * 0)()
        self._plic_enable_arr = (ctypes.c_uint32 * 0)()
        self._plic_threshold_arr = (ctypes.c_uint8 * 0)()
        self._plic_claimed_arr = (ctypes.c_uint32 * 0)()

        # UART 的 ctypes 上下文跨轮保持同一份, 与 PLIC 的数组同理: Rust 处理受
        # 调试程序对 IP 寄存器的读取时, 数据源必须是设备侧持续更新的那个对象,
        # 而非每轮重新构造的副本, 否则单轮内新到的输入对受调试程序不可见.
        if self.uart is not None:
            self._native_uart_ffi = FfiUartCtx()
            self.uart._ffi_ctx = self._native_uart_ffi
