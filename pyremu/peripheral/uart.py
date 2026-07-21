#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
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
from typing import IO

from pyremu.memory.bus import Device

# PLIC 中断源编号 (对齐 QEMU virt 平台, 与 DTB 中 serial@.../interrupts-extended 一致)
UART_IRQ = 10

class UART(Device):
    """SiFive 风格 UART 外设 — 支持多 hart 行缓冲输出.

    寄存器行为对照 QEMU ``hw/char/sifive_uart.c`` 实现:
    - RX FIFO 固定 8 字节 (``SIFIVE_UART_RX_FIFO_SIZE``)
    - preload() 受 ``can_rx()`` 背压: FIFO 满时拒绝新数据
    - RXDATA 读取后调用 ``_accept_input()`` 通知 termio 可继续接收
    - IP 按 FIFO 实际占用 vs 水位 (rxcnt) 计算, 非简单非空判断
    """

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

    # RX FIFO 固定大小 (对照 QEMU SIFIVE_UART_RX_FIFO_SIZE = 8)
    RX_FIFO_SIZE = 8

    # RXDATA 状态位 (SiFive 硬件兼容)
    UART_RXFIFO_EMPTY = 1 << 31  # RX FIFO 空标志 (bit31=1 表示无数据)

    def __init__(
        self,
        base: int = 0x1000_0000,
        size: int = 0x1000,
        tx_callback=None,  # (str) -> None: 每输出一行时调用 (含换行)
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

        # 多 hart 行缓冲: {hart_id: [bytes]}
        self._line_bufs: dict[int, list[int]] = {}
        self._current_writer: int | None = None

        # Terminal I/O 反向引用 — 由 Emulator._init_native_batch 注入,
        # 用于调试器/模拟器直连路径管理终端所有权。
        self._termio: object | None = None

        # 控制台回显开关 — 单一输出 owner 原则 (QEMU chardev 模型):
        # native termio 线程运行期间为 False (Rust 已直写 stdout, 行缓冲
        # 仅归档日志), 其余时刻为 True (行缓冲 flush 经 _tx_callback 回显)。
        self._console_echo: bool = True

        # 每 hart 日志文件 (可选): set_hart_log_dir 后各 hart 输出另存 hart<N>.log
        self._hart_log_dir: str | None = None
        self._hart_log_files: dict[int, IO] = {}

    # ---- 多 hart 行缓冲 ----

    def set_console_echo(self, enabled: bool) -> None:
        """控制台回显开关.

        TerminalIO 在 Rust termio 线程接管终端时关闭 (线程直写 stdout,
        避免行缓冲 flush 时经 _tx_callback 重复输出), 线程停止后恢复。
        关闭期间 hart 日志文件仍正常写入。
        """
        self._console_echo = enabled

    def set_writer(self, hart_id: int) -> None:
        """声明当前写者 hart (不刷新, 仅切换缓冲区)."""
        self._current_writer = hart_id
        if hart_id not in self._line_bufs:
            self._line_bufs[hart_id] = []

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

    def flush_all(self) -> None:
        """刷新所有 hart 的未完成 (不以 \\n 结尾的) 行缓冲."""
        for hid in list(self._line_bufs.keys()):
            self._flush_hart(hid)

    def _flush_hart(self, hart_id: int) -> None:
        """将 hart_id 的缓冲字节拼接为文本, 回调输出并写入日志文件.

        控制台输出为原始字节 (不加 ``[hart N]`` 前缀, 与 QEMU ``-nographic``
        行为一致)。多 hart 调试信息通过 ``set_hart_log_dir()`` 提供的每 hart
        日志文件获取。
        """
        buf = self._line_bufs.get(hart_id, [])
        if not buf:
            return
        text = bytes(buf).decode("utf-8", errors="replace")
        self._line_bufs[hart_id] = []
        if not text:
            return
        # 每 hart 日志文件: 写原始文本 (无 ANSI 颜色/标签)。
        log_f = self._hart_log_file(hart_id)
        if log_f is not None:
            log_f.write(text)
        # 控制台: 直接输出原始字节, 不加前缀。
        # _console_echo=False (native termio 线程运行中) 时跳过 — Rust 侧
        # 已直写 stdout, 此处再回调会双写控制台。
        if self._tx_callback and self._console_echo:
            self._tx_callback(text)

    # ---- QEMU 风格背压接口 (对照 sifive_uart_can_rx / sifive_uart_rx) ----

    def can_rx(self) -> bool:
        """RX FIFO 是否有空闲槽位 (对照 QEMU ``sifive_uart_can_rx``).

        termio 的 ``drain_rx()`` 在 preload 前调用此方法; FIFO 满时不读
        stdin, 数据滞留内核 tty 缓冲 (对照 QEMU ``fd_chr_read_poll`` 流控).
        """
        return len(self._rx_fifo) < self.RX_FIFO_SIZE

    def _accept_input(self) -> None:
        """RXDATA 读取后调用 — 通知 termio 可继续接收数据.

        对照 QEMU ``sifive_uart_read`` 中的 ``qemu_chr_fe_accept_input()``:
        读走一个字节后 FIFO 多出一个空位, 上层可继续 preload.
        当前实现: 读取动作本身释放槽位, 无需额外操作;
        此钩子保留供未来 termio 唤醒优化 (如读后立即触发一次 stdin poll).
        """

    # ---- 公开方法 ----

    def _ip_value(self) -> int:
        """动态计算 IP 寄存器值 (对照 QEMU ``sifive_uart_ip``).

        - **txwm**: TX FIFO 占用 < txctrl.txcnt (bits[18:16])。本模型 TX 即时
          排空 (FIFO 恒空, 占用=0), 故 txcnt>0 时条件恒成立。
        - **rxwm**: RX FIFO 占用 > rxctrl.rxcnt (bits[2:0])。
          默认 rxcnt=0, 故 FIFO 非空时置位。

        TXDATA 写入可能由 Rust 批量引擎 inline 处理 (绕过 _write_reg),
        故 IP 不能依赖写入路径锁存, 必须在读取时按状态计算。
        """
        ip = 0
        if ((self._txctrl >> 16) & 0x7) > 0:
            ip |= self.IP_TXWM
        rxcnt = self._rxctrl & 0x7
        if len(self._rx_fifo) > rxcnt:
            ip |= self.IP_RXWM
        return ip

    def _update_plic_irq(self) -> None:
        """根据 IP 水位状态与 IE 使能同步 PLIC 中断线 (电平语义).

        pending = (IP & IE) != 0 — TX 与 RX 任一满足即拉高; RXDATA 读空 /
        IE 关闭 / txcnt 清零时拉低。PLIC claim 后重新拉高由各读写路径
        调用本方法恢复 (level-triggered)。
        """
        if self._plic is None or self._irq <= 0:
            return
        self._plic.set_irq(self._irq, bool(self._ip_value() & self._ie))

    def preload(self, data: bytes) -> int:
        """向 RX FIFO 预填入数据, 受 ``can_rx()`` 背压 (对照 QEMU ``sifive_uart_rx``).

        Returns:
            实际接受的字节数 (FIFO 满时可能少于 ``len(data)``).
            调用方应检查返回值以决定是否保留剩余数据在 ring buffer 中。
        """
        accepted = 0
        for b in data:
            if len(self._rx_fifo) >= self.RX_FIFO_SIZE:
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
        self._rx_fifo.clear()
        self._update_plic_irq()

    def tx_data(self) -> bytes:
        """返回已发送的全部字节 (调试用)."""
        return bytes(self._tx_buf)

    def tx_clear(self) -> None:
        """清空 TX buffer."""
        self._tx_buf.clear()

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
        if offset == self.REG_RXDATA:
            if not self._rx_fifo:
                return self.UART_RXFIFO_EMPTY
            b = self._rx_fifo.pop(0)
            # 读走一个字节后通知 termio 可继续接收 (对照 QEMU
            # sifive_uart_read -> qemu_chr_fe_accept_input)
            self._accept_input()
            # 读后总是同步 PLIC: FIFO 空时拉低中断线;
            # FIFO 仍有数据时检查水位是否仍需挂起 (level-triggered)
            self._update_plic_irq()
            return b
        if offset == self.REG_TXDATA:
            return 0  # TXDATA 只写; full 位 (bit31) 恒 0 — FIFO 即时排空
        if offset == self.REG_TXCTRL:
            return self._txctrl
        if offset == self.REG_RXCTRL:
            return self._rxctrl
        if offset == self.REG_IE:
            return self._ie
        if offset == self.REG_IP:
            # 电平语义: 按 FIFO 水位状态动态计算, 不依赖写入路径锁存
            # (TXDATA 写可能被 Rust 引擎 inline 处理, 绕过 Python).
            return self._ip_value()
        if offset == self.REG_DIV:
            return self._div
        return 0

    def _write_reg(self, offset: int, val: int) -> None:
        if offset == self.REG_TXDATA:
            val &= 0xFF
            self._tx_buf.append(val)
            # 多 hart 行缓冲: 按当前写者 hart 累积, 遇换行则刷新.
            # 控制台输出不加 [hart N] 前缀 (与 QEMU -nographic 一致);
            # 多 hart 调试信息通过 set_hart_log_dir() 提供的日志文件获取.
            if self._current_writer is not None:
                hid = self._current_writer
                if hid not in self._line_bufs:
                    self._line_bufs[hid] = []
                self._line_bufs[hid].append(val)
                if val == 0x0A:  # '\n'
                    self._flush_hart(hid)
            elif self._tx_callback and self._console_echo:
                self._tx_callback(chr(val))
            return
        if offset == self.REG_TXCTRL:
            self._txctrl = val
            # txcnt (bits[18:16]) 变化影响 txwm 水位条件 — 若 IE.txwm 已使能,
            # 此处需立即拉高/拉低 PLIC (驱动 probe 先写 txcnt 后开中断,
            # 但顺序不可假设).
            self._update_plic_irq()
            return
        if offset == self.REG_RXCTRL:
            self._rxctrl = val
            return
        if offset == self.REG_IE:
            self._ie = val & 3
            self._update_plic_irq()
            return
        if offset == self.REG_IP:
            # SiFive spec: IP 只读 (水位条件电平语义), 写入忽略.
            return
        if offset == self.REG_DIV:
            self._div = val & 0xFFFF
            return
        # 忽略未定义偏移的写入

