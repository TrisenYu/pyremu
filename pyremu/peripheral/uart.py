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
    RX buffer 为空时返回 0.

中断线: 当前版本不生成硬件中断 (待接入 interrupt controller).
"""

from pyremu.memory.bus import Device


class UART(Device):
    """SiFive 风格 UART 外设."""

    # 寄存器偏移
    REG_TXDATA = 0x00
    REG_RXDATA = 0x04
    REG_TXCTRL = 0x08
    REG_RXCTRL = 0x0C
    REG_IE = 0x10
    REG_IP = 0x14
    REG_DIV = 0x18

    # IP 位
    IP_TXWM = 1 << 0  # TX 完成 (写 TXDATA 后自动置位)
    IP_RXWM = 1 << 1  # RX 可用 (preload 后自动置位)

    def __init__(
        self,
        base: int = 0x1000_0000,
        size: int = 0x1000,
        tx_callback=None,  # (int) -> None: 每发送一个字节时调用
    ) -> None:
        self.base_addr = base
        self.size = size
        self._tx_callback = tx_callback

        # 寄存器状态
        self._txctrl: int = 0  # bit0 = txen
        self._rxctrl: int = 0  # bit0 = rxen
        self._ie: int = 0  # bit0=txwm, bit1=rxwm
        self._ip: int = 0  # 中断挂起
        self._div: int = 0

        # 数据 buffer
        self._tx_buf: list[int] = []  # 已发送字节 (调试用)
        self._rx_buf: list[int] = []  # 待接收字节

    # ---- 公开方法 ----

    def preload(self, data: bytes) -> None:
        """向 RX buffer 预填入数据 (模拟外部输入)."""
        self._rx_buf.extend(data)
        self._ip |= self.IP_RXWM

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
        return val.to_bytes(size, "little", signed=False)

    def write(
        self,
        offset: int,
        data: bytes,
    ) -> None:
        val = int.from_bytes(data, "little", signed=False)
        self._write_reg(offset, val)

    # ---- 寄存器读写逻辑 ----

    def _read_reg(self, offset: int) -> int:
        if offset == self.REG_RXDATA:
            if self._rx_buf:
                b = self._rx_buf.pop(0)
                if not self._rx_buf:
                    self._ip &= ~self.IP_RXWM
                return b
            return 0
        if offset == self.REG_TXDATA:
            return 0  # TXDATA 只写
        if offset == self.REG_TXCTRL:
            return self._txctrl
        if offset == self.REG_RXCTRL:
            return self._rxctrl
        if offset == self.REG_IE:
            return self._ie
        if offset == self.REG_IP:
            return self._ip
        if offset == self.REG_DIV:
            return self._div
        return 0

    def _write_reg(self, offset: int, val: int) -> None:
        if offset == self.REG_TXDATA:
            val &= 0xFF
            self._tx_buf.append(val)
            if self._tx_callback:
                self._tx_callback(val)
            if self._txctrl & 1:  # TX 使能
                self._ip |= self.IP_TXWM
            return
        if offset == self.REG_TXCTRL:
            self._txctrl = val
            return
        if offset == self.REG_RXCTRL:
            self._rxctrl = val
            return
        if offset == self.REG_IE:
            self._ie = val & 3
            return
        if offset == self.REG_IP:
            # 写 1 清零 (W1C)
            self._ip &= ~val
            return
        if offset == self.REG_DIV:
            self._div = val & 0xFFFF
            return
        # 忽略未定义偏移的写入
