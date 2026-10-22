#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""GPIO 控制器测试.

覆盖: 寄存器读写与位宽截断 / 输入与输出方向对 INPUT_VAL 的影响 / 外部引脚注入 /
未定义偏移 / Bus 集成 / 平台配置与 DTB 节点生成.
"""

import struct

import libfdt
import pytest

from pyremu import configs_gen
from pyremu.emulator import Emulator
from pyremu.memory.bus import Bus
from pyremu.peripheral.gpio import GPIO
from pyremu.platform import PeripheralConfig, PlatformConfig

INPUT_VAL = GPIO.REG_INPUT_VAL
INPUT_EN = GPIO.REG_INPUT_EN
OUTPUT_EN = GPIO.REG_OUTPUT_EN
OUTPUT_VAL = GPIO.REG_OUTPUT_VAL

GPIO_BASE = 0x1000_3000


def _u32(dev: GPIO, offset: int) -> int:
    """按 32 位宽读取一个寄存器."""
    return int.from_bytes(dev.read(offset, 4), "little", signed=False)


class TestGPIORegisters:
    """寄存器的读写语义."""

    @pytest.fixture
    def gpio(self) -> GPIO:
        return GPIO()

    def test_default_pin_count(self, gpio):
        """默认引脚数为 32."""
        assert gpio.pin_count == 32

    def test_direction_register_roundtrip(self, gpio):
        """方向的三个寄存器写入后原样读出."""
        gpio.write(INPUT_EN, struct.pack("<I", 0x0F0F))
        gpio.write(OUTPUT_EN, struct.pack("<I", 0xF0F0))
        gpio.write(OUTPUT_VAL, struct.pack("<I", 0x1234))
        assert _u32(gpio, INPUT_EN) == 0x0F0F
        assert _u32(gpio, OUTPUT_EN) == 0xF0F0
        assert _u32(gpio, OUTPUT_VAL) == 0x1234

    def test_registers_masked_to_pin_count(self):
        """引脚数少于 32 时写入值截断到引脚位宽."""
        gpio = GPIO(pin_count=8)
        for offset in (INPUT_EN, OUTPUT_EN, OUTPUT_VAL):
            gpio.write(offset, struct.pack("<I", 0x1FF))
            assert _u32(gpio, offset) == 0xFF

    def test_input_val_offset_is_read_only(self, gpio):
        """写 INPUT_VAL 偏移无副作用: 该寄存器的值由外部电平与输出寄存器算出."""
        gpio.set_port(0x3)
        gpio.write(INPUT_VAL, b"\xff\xff\xff\xff")
        gpio.write(INPUT_EN, b"\xff\xff\xff\xff")
        assert _u32(gpio, INPUT_VAL) == 0x3

    def test_unmapped_offset_reads_zero(self, gpio):
        """未定义偏移读返回全 0."""
        assert gpio.read(0x10, 4) == b"\x00" * 4
        assert gpio.read(0x20, 8) == b"\x00" * 8

    def test_unmapped_offset_write_is_ignored(self, gpio):
        """未定义偏移写无副作用."""
        gpio.write(0x10, b"\xff\xff\xff\xff")
        assert gpio.read(0x10, 4) == b"\x00" * 4

    def test_read_size_truncates(self, gpio):
        """读取按请求宽度截断, 低位在前."""
        gpio.write(OUTPUT_VAL, struct.pack("<I", 0x1234))
        assert gpio.read(OUTPUT_VAL, 1) == b"\x34"
        assert gpio.read(OUTPUT_VAL, 2) == b"\x34\x12"


class TestGPIOInputVal:
    """INPUT_VAL 由输入方向与输出方向两部分合成."""

    @pytest.fixture
    def gpio(self) -> GPIO:
        return GPIO()

    def test_input_part_requires_input_enable(self, gpio):
        """外部电平只在对应的输入使能位置位时出现在 INPUT_VAL 中."""
        gpio.set_pin(1, True)
        assert _u32(gpio, INPUT_VAL) == 0
        gpio.write(INPUT_EN, struct.pack("<I", 1 << 1))
        assert _u32(gpio, INPUT_VAL) == 1 << 1

    def test_output_part_requires_output_enable(self, gpio):
        """输出寄存器只在对应的输出使能位置位时出现在 INPUT_VAL 中."""
        gpio.write(OUTPUT_VAL, struct.pack("<I", 1 << 5))
        assert _u32(gpio, INPUT_VAL) == 0
        gpio.write(OUTPUT_EN, struct.pack("<I", 1 << 5))
        assert _u32(gpio, INPUT_VAL) == 1 << 5

    def test_two_parts_are_combined(self, gpio):
        """同一引脚不得同时属于两部分时, 两部分各自按位合成."""
        gpio.set_pin(0, True)
        gpio.write(INPUT_EN, struct.pack("<I", 1 << 0))
        gpio.write(OUTPUT_VAL, struct.pack("<I", 1 << 8))
        gpio.write(OUTPUT_EN, struct.pack("<I", 1 << 8))
        assert _u32(gpio, INPUT_VAL) == (1 << 0) | (1 << 8)

    def test_external_value_masked_by_input_enable(self, gpio):
        """未被输入使能覆盖的外部电平不出现在 INPUT_VAL 中."""
        gpio.set_port(0xF0)
        gpio.write(INPUT_EN, struct.pack("<I", 0x0F))
        assert _u32(gpio, INPUT_VAL) == 0


class TestGPIOPinAccess:
    """外部引脚注入与读出."""

    @pytest.fixture
    def gpio(self) -> GPIO:
        return GPIO()

    def test_set_pin_high_and_low(self, gpio):
        """置高与置低分别设置与清除对应位."""
        gpio.set_pin(3, True)
        assert gpio.input_val & (1 << 3)
        gpio.set_pin(3, False)
        assert not gpio.input_val & (1 << 3)

    def test_set_pin_out_of_range_is_ignored(self, gpio):
        """引脚号越界时电平不变."""
        gpio.set_pin(-1, True)
        gpio.set_pin(32, True)
        assert gpio.input_val == 0

    def test_get_pin_output(self, gpio):
        """读回指定引脚的输出值."""
        gpio.write(OUTPUT_VAL, struct.pack("<I", 1 << 7))
        assert gpio.get_pin_output(7)
        assert not gpio.get_pin_output(6)

    def test_set_port_masked(self):
        """批量设置外部电平按引脚数截断."""
        gpio = GPIO(pin_count=8)
        gpio.set_port(0x1FF)
        assert gpio.input_val == 0xFF

    def test_output_val_property(self, gpio):
        """output_val 属性与寄存器读出的一致."""
        gpio.write(OUTPUT_VAL, struct.pack("<I", 0xA5A5))
        assert gpio.output_val == 0xA5A5


class TestGPIOBus:
    """Bus 集成."""

    def test_bus_routes_read_and_write(self):
        """经 Bus 读写 GPIO 地址应路由到设备."""
        gpio = GPIO(base=GPIO_BASE)
        bus = Bus(ram_size=1024 * 1024, ram_base=0)
        bus.add_device(GPIO_BASE, gpio)
        assert bus.is_device_addr(GPIO_BASE)
        bus.write(GPIO_BASE + OUTPUT_VAL, struct.pack("<I", 1 << 4))
        assert bus.read(GPIO_BASE + OUTPUT_VAL, 4) == struct.pack("<I", 1 << 4)


class TestGPIOConfig:
    """平台配置与 DTB 节点生成."""

    @staticmethod
    def _cfg() -> PlatformConfig:
        return PlatformConfig(
            num_harts=1,
            ram_base=0x8000_0000,
            prog_cnt=0x8000_0000,
            periph=PeripheralConfig(gpio_base=GPIO_BASE),
        )

    def test_default_gpio_base_follows_build_config(self):
        """默认 GPIO 基址取自 emu-configs.mk 生成的常量."""
        assert PeripheralConfig().gpio_base == configs_gen.GPIO_BASE

    def test_emulator_registers_device(self):
        """基址非零时 Emulator 创建 GPIO 并挂到 Bus 上."""
        emu = Emulator(self._cfg(), bootargs="console=ttySIF0")
        assert emu.gpio is not None
        assert emu.bus.is_device_addr(GPIO_BASE)

    def test_dtb_contains_gpio_node(self):
        """DTB 含 /soc/gpio@... 节点, compatible 与 reg 正确."""
        emu = Emulator(self._cfg(), bootargs="console=ttySIF0")
        fdt = libfdt.Fdt(emu.build_dtb())
        node = fdt.path_offset(f"/soc/gpio@{GPIO_BASE:x}")
        assert node >= 0
        assert bytes(fdt.getprop(node, "compatible")) == b"pyremu,gpio0\x00"
        base_hi, base_lo, size_hi, size_lo = struct.unpack(
            ">IIII", bytes(fdt.getprop(node, "reg"))
        )
        assert (base_hi << 32) | base_lo == GPIO_BASE
        assert (size_hi << 32) | size_lo == 0x1000

    def test_zero_base_disables_device(self):
        """基址为 0 时不注册设备, DTB 中也没有对应节点."""
        cfg = self._cfg()
        cfg.periph.gpio_base = 0
        emu = Emulator(cfg, bootargs="console=ttySIF0")
        assert emu.gpio is None
        assert not emu.bus.is_device_addr(GPIO_BASE)
        fdt = libfdt.Fdt(emu.build_dtb())
        with pytest.raises(libfdt.FdtException):
            fdt.path_offset("/soc/gpio@0")
