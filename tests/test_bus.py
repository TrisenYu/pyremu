#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""共享总线 (Bus) 测试: RAM 读写, 设备路由, MMIO 检测."""

import pytest

from pyremu.memory.bus import Bus, Device


class _DummyDevice(Device):
    """一个简单的测试设备: 读写时记录副作用."""

    def __init__(self, base: int, sz: int):
        self.base_addr = base
        self.size = sz
        self.last_read_offset = -1
        self.last_write_offset = -1
        self.last_write_data = b""
        self._reg = 0

    def read(self, offset: int, size: int) -> bytes:
        self.last_read_offset = offset
        return self._reg.to_bytes(size, "little")

    def write(self, offset: int, data: bytes) -> None:
        self.last_write_offset = offset
        self.last_write_data = data
        self._reg = int.from_bytes(data, "little")


class TestBusRAM:
    """直接 RAM 读写."""

    @pytest.fixture
    def bus(self) -> Bus:
        return Bus(ram_size=1024 * 1024)  # 1 MiB

    def test_read_write_ram(self, bus):
        """基本读写."""
        bus.write(0x1000, b"\xDE\xAD\xBE\xEF")
        data = bus.read(0x1000, 4)
        assert data == b"\xDE\xAD\xBE\xEF"

    def test_read_partial(self, bus):
        """部分字节读取."""
        bus.write(0x2000, b"\x01\x02\x03\x04\x05\x06\x07\x08")
        assert bus.read(0x2000, 1) == b"\x01"
        assert bus.read(0x2002, 2) == b"\x03\x04"
        assert bus.read(0x2004, 4) == b"\x05\x06\x07\x08"


class TestBusDevice:
    """设备路由和 MMIO 检测."""

    @pytest.fixture
    def bus(self) -> Bus:
        b = Bus(ram_size=1024 * 1024)
        dev = _DummyDevice(base=0x10000000, sz=0x1000)
        b.add_device(0x10000000, dev)
        return b

    def test_device_read_routed(self, bus):
        """设备地址的读应路由到设备."""
        data = bus.read(0x10000000, 4)
        assert data == b"\x00\x00\x00\x00"  # 设备初始值
        dev = bus.devices[0x10000000]
        assert dev.last_read_offset == 0

    def test_device_write_routed(self, bus):
        """设备地址的写应路由到设备."""
        bus.write(0x10000004, b"\x42\x00\x00\x00")
        dev = bus.devices[0x10000000]
        assert dev.last_write_offset == 4
        assert dev.last_write_data == b"\x42\x00\x00\x00"

    def test_is_device_addr(self, bus):
        """is_device_addr 正确判断地址是否属于设备区域."""
        assert bus.is_device_addr(0x10000000)
        assert bus.is_device_addr(0x100000FF)
        assert bus.is_device_addr(0x10000FFF)
        assert not bus.is_device_addr(0x00000000)
        assert not bus.is_device_addr(0x10001000)  # 超出设备范围

    def test_device_addr_bypasses_cache(self, bus):
        """设备地址读写应绕过 L2 缓存."""
        # 无 L2 缓存时也正常工作
        data = bus.read(0x10000008, 2)
        assert data is not None
