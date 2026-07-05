#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
I2C 主模式控制器.

寄存器布局 (每寄存器 4 字节宽):
  偏移    名称       读写    描述
  ────────────────────────────────────────
  0x00    CTRL       R/W    控制: bit0=EN, bit1=START, bit2=STOP, bit3=ACK
  0x04    STATUS     R      状态: bit0=busy, bit1=rx_avail, bit2=tx_done, bit3=nack
  0x08    DATA       R/W    数据寄存器 (写=TX, 读=RX)
  0x0C    ADDR       R/W    从设备地址 (bit0:7=addr, bit8=rw)
  0x10    PRESCALE   R/W    时钟预分频器

传输模型: 写 ADDR + CTRL.START 后, 写 DATA 为 TX, 读 DATA 为 RX.
CTRL.STOP 结束传输. 简化为即时完成模型 (总线不实际仿真时序).
"""

from pyremu.memory.bus import Device


class I2C(Device):
    """I2C 主模式控制器."""

    REG_CTRL = 0x00
    REG_STATUS = 0x04
    REG_DATA = 0x08
    REG_ADDR = 0x0C
    REG_PRESCALE = 0x10

    CTRL_EN = 1 << 0
    CTRL_START = 1 << 1
    CTRL_STOP = 1 << 2
    CTRL_ACK = 1 << 3

    STATUS_BUSY = 1 << 0
    STATUS_RX_AVAIL = 1 << 1
    STATUS_TX_DONE = 1 << 2
    STATUS_NACK = 1 << 3

    def __init__(
        self,
        base: int = 0x1000_2000,
        size: int = 0x1000,
    ) -> None:
        self.base_addr = base
        self.size = size

        self._ctrl: int = 0
        self._data: int = 0
        self._addr: int = 0
        self._prescale: int = 0
        self._busy: bool = False
        self._nack: bool = False

        # 模拟从设备寄存器 (简化: 单一目标地址)
        self._slave_data: dict[int, int] = {}  # reg_addr -> value

    # ---- 公开方法 ----

    def set_slave_data(self, reg: int, val: int) -> None:
        """设置从设备寄存器的模拟值."""
        self._slave_data[reg] = val & 0xFF

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
        if offset == self.REG_CTRL:
            return self._ctrl
        if offset == self.REG_STATUS:
            st = self.STATUS_TX_DONE  # TX 始终就绪
            if not self._busy:
                st |= self.STATUS_RX_AVAIL
            if self._busy:
                st |= self.STATUS_BUSY
            if self._nack:
                st |= self.STATUS_NACK
            return st
        if offset == self.REG_DATA:
            # 读取从设备数据 (模拟)
            self._busy = False
            if self._addr & 0x100:  # rw=1: read
                reg = self._addr & 0xFF
                return self._slave_data.get(reg, 0)
            return self._data
        if offset == self.REG_ADDR:
            return self._addr
        if offset == self.REG_PRESCALE:
            return self._prescale
        return 0

    def _write_reg(self, offset: int, val: int) -> None:
        if offset == self.REG_CTRL:
            prev = self._ctrl
            self._ctrl = val
            # START 置位
            if (val & self.CTRL_START) and not (prev & self.CTRL_START):
                self._busy = True
                self._nack = False
            # STOP 置位
            if val & self.CTRL_STOP:
                self._busy = False
            return
        if offset == self.REG_DATA:
            self._data = val & 0xFF
            self._busy = False  # 瞬时完成
            return
        if offset == self.REG_ADDR:
            self._addr = val
            # 检查地址是否有对应的从设备数据
            self._nack = (val & 0xFF) not in self._slave_data
            return
        if offset == self.REG_PRESCALE:
            self._prescale = val & 0xFFFF
            return
