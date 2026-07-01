#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
SPI 主模式控制器.

寄存器布局 (每寄存器 4 字节宽):
  偏移    名称     读写    描述
  ──────────────────────────────────────
  0x00    CTRL     R/W    控制: bit0=EN, bit1=CPOL, bit2=CPHA, bit8:11=frame_len(0→8bit)
  0x04    STATUS   R      状态: bit0=busy, bit1=rx_avail
  0x08    TXDATA   W      发送数据 (写入后启动传输, 同时捕获到 RXDATA)
  0x0C    RXDATA   R      接收数据 (上次传输的结果)
  0x10    DIV      R/W    时钟除数 (sclk = coreclk / (2*(DIV+1)))

传输模型: 瞬时完成 — 写 TXDATA 立即将数据镜像到 RXDATA,
模拟全双工 shift 寄存器操作. 实际固件使用时轮询 STATUS.busy=0
后写 TXDATA, 再轮询 STATUS.busy=0 后读 RXDATA.
"""

from pyremu.memory.bus import Device


class SPI(Device):
    """SPI 主模式控制器."""

    REG_CTRL = 0x00
    REG_STATUS = 0x04
    REG_TXDATA = 0x08
    REG_RXDATA = 0x0C
    REG_DIV = 0x10

    CTRL_EN = 1 << 0
    CTRL_CPOL = 1 << 1
    CTRL_CPHA = 1 << 2

    STATUS_BUSY = 1 << 0
    STATUS_RX_AVAIL = 1 << 1

    def __init__(
        self,
        base: int = 0x1000_1000,
        size: int = 0x1000,
    ) -> None:
        self.base_addr = base
        self.size = size

        self._ctrl: int = 0
        self._div: int = 0
        self._rxdata: int = 0
        self._busy: bool = False

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
        if offset == self.REG_CTRL:
            return self._ctrl
        if offset == self.REG_STATUS:
            st = 0
            if self._busy:
                st |= self.STATUS_BUSY
            if (self._ctrl & self.CTRL_EN) and not self._busy:
                st |= self.STATUS_RX_AVAIL
            return st
        if offset == self.REG_RXDATA:
            return self._rxdata & 0xFF_FFFF_FFFF_FFFF
        if offset == self.REG_DIV:
            return self._div
        return 0

    def _write_reg(self, offset: int, val: int) -> None:
        if offset == self.REG_CTRL:
            self._ctrl = val
            return
        if offset == self.REG_TXDATA:
            # 全双工: 同时写 TX 和捕获 RX (瞬时完成)
            self._rxdata = val
            self._busy = False  # 瞬时传输完毕
            return
        if offset == self.REG_DIV:
            self._div = val & 0xFF
            return
