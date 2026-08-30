#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
GPIO 控制器.

寄存器布局 (每寄存器 4 字节宽):
  偏移    名称       读写    描述
  ──────────────────────────────────────
  0x00    INPUT_VAL  R      输入引脚值 (外部电平)
  0x04    INPUT_EN   R/W    输入使能 (1=输入模式, 需 OUTPUT_EN=0)
  0x08    OUTPUT_EN  R/W    输出使能 (1=输出模式)
  0x0C    OUTPUT_VAL R/W    输出引脚值 (OUTPUT_EN=1 时驱动外部引脚)

每个 GPIO 位独立控制方向; INPUT_EN 和 OUTPUT_EN 不应同时为 1.
输出引脚值可通过 set_pin() / get_pin() 方法在外部操作.
"""

from pyremu.memory.bus import Device


class GPIO(Device):
    """通用 I/O 控制器."""

    REG_INPUT_VAL = 0x00
    REG_INPUT_EN = 0x04
    REG_OUTPUT_EN = 0x08
    REG_OUTPUT_VAL = 0x0C

    def __init__(
        self,
        base: int = 0x1000_3000,
        size: int = 0x1000,
        pin_count: int = 32,
    ) -> None:
        self.base_addr = base
        self.size = size
        self._pin_count = pin_count
        self._pin_mask = (1 << pin_count) - 1

        # 寄存器
        self._input_en: int = 0
        self._output_en: int = 0
        self._output_val: int = 0
        self._input_val: int = 0  # 外部驱动值

    # ---- 公开方法 (模拟外部引脚) ----

    def set_pin(self, pin: int, high: bool) -> None:
        """设置外部引脚电平 (模拟外部设备驱动)."""
        if not (0 <= pin < self._pin_count):
            return
        if high:
            self._input_val |= 1 << pin
        else:
            self._input_val &= ~(1 << pin)

    def get_pin_output(self, pin: int) -> bool:
        """读取指定引脚的输出值."""
        return bool(self._output_val & (1 << pin))

    def set_port(self, val: int) -> None:
        """批量设置外部输入引脚值."""
        self._input_val = val & self._pin_mask

    @property
    def input_val(self) -> int:
        return self._input_val

    @property
    def output_val(self) -> int:
        return self._output_val

    @property
    def pin_count(self) -> int:
        return self._pin_count

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
        if offset == self.REG_INPUT_VAL:
            # 返回: (外部电平 & INPUT_EN) | (OUTPUT_VAL & OUTPUT_EN)
            in_part = self._input_val & self._input_en
            out_part = self._output_val & self._output_en
            return (in_part | out_part) & self._pin_mask
        if offset == self.REG_INPUT_EN:
            return self._input_en
        if offset == self.REG_OUTPUT_EN:
            return self._output_en
        if offset == self.REG_OUTPUT_VAL:
            return self._output_val
        return 0

    def _write_reg(self, offset: int, val: int) -> None:
        if offset == self.REG_INPUT_EN:
            self._input_en = val & self._pin_mask
            return
        if offset == self.REG_OUTPUT_EN:
            self._output_en = val & self._pin_mask
            return
        if offset == self.REG_OUTPUT_VAL:
            self._output_val = val & self._pin_mask
            return
