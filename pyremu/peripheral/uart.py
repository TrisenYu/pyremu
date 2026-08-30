#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
SiFive 风格 UART (NS16550A 兼容子集).

寄存器布局 (每寄存器 4 字节宽, 共 0x1C 字节):
  偏移    名称     读写    描述
  ──────────────────────────────────────
  0x00    TXDATA   W      发送数据 (写入后存入 TX buffer)
  0x04    RXDATA   R      接收数据 (从 RX buffer 读取)
  0x08    TXCTRL   R/W    发送控制: bit0=TX 使能, bit16:18=nstop
  0x0C    RXCTRL   R/W    接收控制: bit0=RX 使能
  0x10    IE       R/W    中断使能: bit0=TX 完成, bit1=RX 可用
  0x14    IP       R      中断挂起: bit0=TX 完成, bit1=RX 可用
  0x18    DIV      R/W    波特率除数 (baud = coreclk / DIV)

TX: 固件写入的字节存入 _tx_buf 列表 (调试时可检查);
    若 TXCTRL.txen=1 则 IP.txwm 置位.
RX: 可通过 preload() 预填入数据; RXDATA 读取返回队首字节;
    RX buffer 为空时返回 UART_RXFIFO_EMPTY (bit31=1), 与 SiFive 硬件一致.

中断线: 当前版本不生成硬件中断 (待接入 interrupt controller).
"""

from __future__ import annotations

import os
import threading
from typing import IO

from pyremu.memory.bus import Device

# PLIC 中断源编号 (对齐 QEMU virt 平台, 与 DTB 中 serial@.../interrupts-extended 一致)
UART_IRQ = 10
# 寄存器偏移
REG_TXDATA = 0x00
REG_RXDATA = 0x04
REG_TXCTRL = 0x08
REG_RXCTRL = 0x0C
REG_IE = 0x10
REG_IP = 0x14
REG_DIV = 0x18

# IP 位
IP_TXWM = 1 << 0  # TX FIFO 占用 < txcnt (对照 QEMU SIFIVE_UART_IP_TXWM)
IP_RXWM = 1 << 1  # RX FIFO 占用 > rxcnt (对照 QEMU SIFIVE_UART_IP_RXWM)

# RX FIFO 固定大小 (QEMU SIFIVE_UART_RX_FIFO_SIZE = 8, 但 8 字节
# 不足以容纳完整 vt100 转义序列如 \x1b[1;5D (6 bytes) +
# 连续键入时的队列缓冲. 32 字节确保转义序列始终原子交付.)
RX_FIFO_SIZE = 32

# RXDATA 状态位 (SiFive 硬件兼容)
UART_RXFIFO_EMPTY = 1 << 31  # RX FIFO 空标志 (bit31=1 表示无数据)


class UART(Device):
    """SiFive 风格 UART 外设 — 纯被动 MMIO 设备.

    寄存器行为对照 QEMU ``hw/char/sifive_uart.c`` 实现:
    - RX FIFO 固定 8 字节 (``SIFIVE_UART_RX_FIFO_SIZE``)
    - preload() 受 ``can_rx()`` 控制: FIFO 满时拒绝新数据
    - RXDATA 读取后调用 ``_accept_input()`` 通知 termio 可继续接收
    - IP 按 FIFO 实际占用 vs 水位 (rxcnt) 计算, 非简单非空判断

    TXDATA 写入即自动输出 (无 hart 耦合): 每字节经 _tx_callback (若设置)
    即时输出, 不依赖外部 set_writer / flush_all.
    """


    def __init__(
        self,
        base: int = 0x1000_0000,
        size: int = 0x1000,
        tx_callback=None,  # (str) -> None: TXDATA 每字节回调 (默认直接 os.write stdout)
        plic=None,  # PLIC: 用于 RX 中断通知 (None = 无中断)
        irq: int = 0,  # PLIC 中断源编号
    ) -> None:
        self.base_addr = base
        self.size = size
        self._tx_callback = tx_callback
        self._plic = plic
        self._irq = irq

        # 寄存器状态
        self._txctrl: int = 0  # bit0 = txen
        self._rxctrl: int = 0  # bit0 = rxen
        self._ie: int = 0  # bit0=txwm, bit1=rxwm
        self._div: int = 0

        # 数据 buffer
        self._tx_buf: list[int] = []  # 已发送字节 (调试用)
        # RX FIFO — 固定 8 字节, 对照 QEMU s->rx_fifo[8] + s->rx_fifo_len
        self._rx_fifo: list[int] = []  # 待接收字节 (len ≤ RX_FIFO_SIZE)
        self._rx_lock = threading.Lock()  # daemon reader 线程与主线程共享 _rx_fifo

        # 当前 MMIO 写者 hart ID — 由 _wrap_phy_write_for_uart 在 hart
        # 写 UART MMIO 地址时自动标记, 仅用于 hart 日志文件分流归档.
        self._current_writer: int | None = None

        # Terminal I/O 反向引用 — 由 Emulator._init_for_speedup_lib 注入,
        # 用于调试器/模拟器直连路径管理终端所有权。
        self.termio: object | None = None

        # 控制台回显开关 — 单一输出 owner 原则 (QEMU chardev 模型):
        # native termio 线程运行期间为 False (Rust 已直写 stdout),
        # 其余时刻为 True (_tx_callback 负责回显)。
        self._console_echo: bool = True

        # 每 hart 日志文件 (可选): set_hart_log_dir 后各 hart 输出另存 hart<N>.log
        self._hart_log_dir: str | None = None
        self._hart_log_files: dict[int, IO] = {}

    # ---- 控制台回显 ----

    def set_console_echo(self, enabled: bool) -> None:
        """控制台回显开关.

        TerminalIO 在 Rust termio 线程接管终端时关闭 (线程直写 stdout,
        避免 _tx_callback 重复输出), 线程停止后恢复。
        关闭期间 hart 日志文件仍正常写入。
        """
        self._console_echo = enabled

    def set_hart_log_dir(self, log_dir: str | None) -> None:
        """启用每 hart 日志文件: 各 hart 的输出另存 <log_dir>/hart<N>.log (原始文本)。

        None 关闭该功能。目录不存在则创建。文件在首次写入时惰性打开 (行缓冲)。
        """
        self.close_logs()
        self._hart_log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

    def close_logs(self) -> None:
        """关闭所有已打开的 hart 日志文件."""
        for f in self._hart_log_files.values():
            try:
                f.close()
            except OSError:
                pass
        self._hart_log_files.clear()

    def _hart_log_file(self, hart_id: int) -> IO | None:
        """返回 hart_id 的日志文件句柄 (惰性打开); 未启用日志目录时返回 None."""
        if self._hart_log_dir is None:
            return None
        f = self._hart_log_files.get(hart_id)
        if f is None:
            path = os.path.join(self._hart_log_dir, f"hart{hart_id}.log")
            f = open(path, "w", buffering=1, encoding="utf-8")  # 每次运行覆写
            self._hart_log_files[hart_id] = f
        return f

    # ---- QEMU 风格容量控制接口 (对照 sifive_uart_can_rx / sifive_uart_rx) ----

    def can_rx(self) -> bool:
        """RX FIFO 是否有空闲槽位 (对照 QEMU ``sifive_uart_can_rx``).

        termio 的 ``drain_rx()`` 在 preload 前调用此方法; FIFO 满时不读
        stdin, 数据滞留内核 tty 缓冲 (对照 QEMU ``fd_chr_read_poll`` 流控).
        """
        with self._rx_lock:
            return len(self._rx_fifo) < RX_FIFO_SIZE

    def _accept_input(self) -> None:
        """RXDATA 读取后调用 — 通知 termio 可继续接收数据.

        对照 QEMU ``sifive_uart_read`` 中的 ``qemu_chr_fe_accept_input()``:
        读走一个字节后 FIFO 多出一个空位, 上层可继续 preload.
        当前实现: 读取动作本身释放槽位, 无需额外操作;
        此钩子保留供未来 termio 唤醒优化 (如读后立即触发一次 stdin poll).
        """


    def _ip_value(self) -> int:
        """动态计算 IP 寄存器值 (对照 QEMU ``sifive_uart_ip``).

        - **txwm**: TX FIFO 占用 < txctrl.txcnt (bits[18:16])。本模型 TX 即时
          排空 (FIFO 恒空, 占用=0), 故 txcnt>0 时条件恒成立。
        - **rxwm**: RX FIFO 占用 > rxctrl.rxcnt (bits[18:16], 与 txcnt 同偏移)。
          默认 rxcnt=0, 故 FIFO 非空时置位。

        TXDATA 写入可能由加速执行动态链接库于内部处理 (绕过 _write_reg),
        故 IP 不能依赖写入路径锁存, 必须在读取时按状态计算。
        """
        ip = 0
        if ((self._txctrl >> 16) & 0x7) > 0:
            ip |= IP_TXWM
        rxcnt = (self._rxctrl >> 16) & 0x7
        with self._rx_lock:
            if len(self._rx_fifo) > rxcnt:
                ip |= IP_RXWM
        return ip

    def irq_asserted(self) -> bool:
        """返回当前 UART 中断线电平 (IP & IE != 0).

        供外部 (termio RX 排空路径) 在注入数据后查询真实电平, 用于
        按电平语义同步 PLIC 挂起位 — 而非无条件拉高, 避免 IE 关闭时
        产生虚假挂起。
        """
        return bool(self._ip_value() & self._ie)

    def _update_plic_irq(self) -> None:
        """根据 IP 水位状态与 IE 使能同步 PLIC 中断线 (电平语义).

        pending = (IP & IE) != 0 — TX 与 RX 任一满足即拉高; RXDATA 读空 /
        IE 关闭 / txcnt 清零时拉低。PLIC claim 后重新拉高由各读写路径
        调用本方法恢复 (level-triggered)。
        """
        if self._plic is None or self._irq <= 0:
            return
        self._plic.set_irq(self._irq, self.irq_asserted())

    # ---- 公开方法 ----
    def preload(self, data: bytes) -> int:
        """向 RX FIFO 预填入数据, 受 ``can_rx()`` 控制 (对照 QEMU ``sifive_uart_rx``).

        Returns:
            实际接受的字节数 (FIFO 满时可能少于 ``len(data)``).
            调用方应检查返回值以决定是否保留剩余数据在 ring buffer 中。
        """
        accepted = 0
        with self._rx_lock:
            for b in data:
                if len(self._rx_fifo) >= RX_FIFO_SIZE:
                    # TODO: 问题在于有可能是队列事先已经满了，而不是从空变到满
                    break
                self._rx_fifo.append(b)
                accepted += 1
        if accepted > 0:
            self._update_plic_irq()
        return accepted

    def clear_rx(self) -> None:
        """清空 RX FIFO 并复位 RX 中断标志, 丢弃所有待接收数据.

        用于固件启动完成后、shell 接管终端前丢弃用户误敲入的字符,
        避免预启动输入污染 shell 的 termios 初始化。
        """
        with self._rx_lock:
            self._rx_fifo.clear()
        self._update_plic_irq()

    def tx_data(self) -> bytes:
        """返回已发送的全部字节 (调试用)."""
        return bytes(self._tx_buf)

    def tx_clear(self) -> None:
        """清空 TX buffer."""
        self._tx_buf.clear()

    def record_tx_byte(self, byte: int) -> None:
        """记录 TX 字节到行缓冲 (native Rust 直写 stdout 路径的归档回填).

        与 ``_write_reg`` 的 TXDATA 路径不同, 此方法仅追加到 ``_tx_buf``,
        不触发 ``_tx_callback`` / ``os.write`` — native 模式下 Rust 引擎已
        即时输出到 stdout, 再输出即双重写控制台. 供 termio 在排空 TX ring
        buffer 时同步回填, 使 ``tx_data()`` 在 native 与纯 Python 两条路径
        下返回一致的完整输出.
        """
        self._tx_buf.append(byte)

    # ---- Device 接口 ----

    def read(
        self,
        offset: int,
        size: int,
    ) -> bytes:
        val = self._read_reg(offset)
        mask = (1 << (size * 8)) - 1
        return (val & mask).to_bytes(size, "little", signed=False)

    def write(
        self,
        offset: int,
        data: bytes,
    ) -> None:
        val = int.from_bytes(data, "little", signed=False)
        self._write_reg(offset, val)

    def _read_reg(self, offset: int) -> int:
        if offset == REG_RXDATA:
            with self._rx_lock:
                if not self._rx_fifo:
                    return UART_RXFIFO_EMPTY
                b = self._rx_fifo.pop(0)
            self._accept_input()
            self._update_plic_irq()
            return b
        if offset == REG_TXDATA:
            return 0  # TXDATA 只写; full 位 (bit31) 恒 0 — FIFO 即时排空
        if offset == REG_TXCTRL:
            return self._txctrl
        if offset == REG_RXCTRL:
            return self._rxctrl
        if offset == REG_IE:
            return self._ie
        if offset == REG_IP:
            # 电平语义: 按 FIFO 水位状态动态计算, 不依赖写入路径锁存
            # (TXDATA 写可能被 Rust 引擎 inline 处理, 绕过 Python).
            return self._ip_value()
        if offset == REG_DIV:
            return self._div
        return 0

    def _write_reg(self, offset: int, val: int) -> None:
        if offset == REG_TXDATA:
            val &= 0xFF
            self._tx_buf.append(val)
            # 自动即时输出: _tx_callback 受 _console_echo 约束 (避免 Rust+Python
            # 双写 stdout), os.write 不受约束 — UART 始终自动输出.
            if self._tx_callback is not None and self._console_echo:
                self._tx_callback(chr(val))
            if self._tx_callback is None or not self._console_echo:
                os.write(1, bytes([val]))
            # Hart 日志分流: 若 _current_writer 已标记 (由 _wrap_phy_write_for_uart
            # 在 MMIO 写时自动设置), 同步写入该 hart 的日志文件.
            if self._current_writer is not None:
                log_f = self._hart_log_file(self._current_writer)
                if log_f is not None:
                    log_f.write(chr(val))
            return
        if offset == REG_TXCTRL:
            self._txctrl = val
            # txcnt (bits[18:16]) 变化影响 txwm 水位条件 — 若 IE.txwm 已使能,
            # 此处需立即拉高/拉低 PLIC (驱动 probe 先写 txcnt 后开中断,
            # 但顺序不可假设).
            self._update_plic_irq()
            return
        if offset == REG_RXCTRL:
            self._rxctrl = val
            # rxcnt (bits[18:16]) 是 IP.rxwm 的比较阈值, 其变化可直接改变
            # 中断触发条件 — 与 TXCTRL 路径同理, 写后必须同步 PLIC 中断线.
            self._update_plic_irq()
            return
        if offset == REG_IE:
            self._ie = val & 3
            self._update_plic_irq()
            return
        if offset == REG_IP:
            # SiFive spec: IP 只读 (水位条件电平语义), 写入忽略.
            return
        if offset == REG_DIV:
            self._div = val & 0xFFFF
            return
        # 忽略未定义偏移的写入

