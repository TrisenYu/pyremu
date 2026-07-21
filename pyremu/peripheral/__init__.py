#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
内存映射外设 (MMIO Peripherals).

每个外设实现 memory.bus.Device 接口, 通过 Emulator 注册到 Bus
并在设备树中描述, 供固件通过 FDT 发现和配置.

设备:
- UART  : SiFive 风格 NS16550A 串口
- SPI   : 主模式 SPI 控制器
- I2C   : 主模式 I2C 控制器
- GPIO  : 双向通用 I/O 控制器
"""

from pyremu.peripheral.gpio import GPIO
from pyremu.peripheral.i2c import I2C
from pyremu.peripheral.spi import SPI
from pyremu.peripheral.termio import TerminalIO
from pyremu.peripheral.uart import UART
from pyremu.peripheral.virtio_blk import VirtIOBlock

__all__ = ["TerminalIO", "UART", "SPI", "I2C", "GPIO", "VirtIOBlock"]
