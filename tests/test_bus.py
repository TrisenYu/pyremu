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
    """直接 RAM 读写.

    使用 ram_base=0 以简化地址计算, 脱离平台约束.
    默认 ram_base (0x8000_0000) 的测试由 test_emulator 间接覆盖.
    """

    @pytest.fixture
    def bus(self) -> Bus:
        return Bus(ram_size=1024 * 1024, ram_base=0)  # 1 MiB, 基址为 0 简化测试

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
        b = Bus(ram_size=1024 * 1024, ram_base=0)
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


class TestBusPMA:
    """PMA (Physical Memory Attributes) 和地址空间布局.

    RISC-V 特权架构未规定统一的全局物理内存映射, 但主流平台
    (SiFive / QEMU virt) 将 DRAM 置于 0x8000_0000 起始,
    低地址留给 MMIO 设备。Bus 通过 *ram_base* 参数适配不同布局。
    """

    # 模拟 SiFive Freedom Ux00 平台: 128 MiB DRAM @ 0x8000_0000
    RAM_BASE = 0x8000_0000

    @pytest.fixture
    def bus(self) -> Bus:
        return Bus(ram_size=16 * 1024 * 1024, ram_base=self.RAM_BASE)  # 16 MiB

    # -------- RAM 范围 --------

    def test_ram_read_write_in_range(self, bus):
        """RAM 范围内的读写正常工作."""
        addr = self.RAM_BASE + 0x1000
        bus.write(addr, b"\xDE\xAD\xC0\xDE")
        assert bus.read(addr, 4) == b"\xDE\xAD\xC0\xDE"

    def test_ram_read_write_at_base(self, bus):
        """RAM 基址 (ram_base) 本身可读写."""
        bus.write(self.RAM_BASE, b"\x01\x02\x03\x04")
        assert bus.read(self.RAM_BASE, 4) == b"\x01\x02\x03\x04"

    def test_ram_read_out_of_range_returns_zero(self, bus):
        """超出 RAM 范围的读返回全 0 (模拟未映射地址)."""
        # 低于 RAM 基址
        assert bus.read(self.RAM_BASE - 4, 4) == b"\x00\x00\x00\x00"
        # 高于 RAM 末尾
        assert bus.read(self.RAM_BASE + 0x1000_0000, 4) == b"\x00\x00\x00\x00"

    # -------- PMA 查询 --------

    def test_is_ram_addr(self, bus):
        """is_ram_addr 正确判断地址是否在 RAM 范围内."""
        assert bus.is_ram_addr(self.RAM_BASE)
        assert bus.is_ram_addr(self.RAM_BASE + 0xFFF)
        assert not bus.is_ram_addr(self.RAM_BASE - 1)  # 基址前一字节
        assert not bus.is_ram_addr(0x0)

    def test_is_valid_addr(self, bus):
        """is_valid_addr 对 RAM 和设备地址返回 True, 空洞返回 False."""
        # RAM 地址
        assert bus.is_valid_addr(self.RAM_BASE)
        # 设备地址
        dev = _DummyDevice(base=0x0200_0000, sz=0x10000)
        bus.add_device(0x0200_0000, dev)
        assert bus.is_valid_addr(0x0200_0000)
        # 空洞地址: 设备区与 RAM 区之间
        assert not bus.is_valid_addr(0x4000_0000)
        # 地址 0
        assert not bus.is_valid_addr(0x0)

    def test_is_valid_addr_device_above_ram(self, bus):
        """设备地址可高于 RAM 基址, 不影响判断."""
        dev = _DummyDevice(base=0xC000_0000, sz=0x1000)
        bus.add_device(0xC000_0000, dev)
        assert bus.is_valid_addr(0xC000_0000)

    # -------- ram_base 可配置 --------

    def test_ram_base_zero(self):
        """ram_base=0 的配置: RAM 覆盖 [0, ram_size), 兼容嵌入式布局."""
        bus = Bus(ram_size=64 * 1024, ram_base=0)
        bus.write(0, b"\xAA")
        assert bus.read(0, 1) == b"\xAA"
        bus.write(0xFFFF, b"\xBB")
        assert bus.read(0xFFFF, 1) == b"\xBB"
        # 超出范围
        assert bus.read(0x10000, 1) == b"\x00"

    def test_ram_base_mid_space(self):
        """ram_base 可在地址空间任意位置, 不受低地址设备影响."""
        bus = Bus(ram_size=4096, ram_base=0x4000_0000)
        dev = _DummyDevice(base=0x1000_0000, sz=0x1000)
        bus.add_device(0x1000_0000, dev)
        # 设备访问正常
        bus.write(0x1000_0000, b"\x42")
        assert bus.read(0x1000_0000, 1) == b"\x42"
        # RAM 访问正常
        bus.write(0x4000_0000, b"\x77")
        assert bus.read(0x4000_0000, 1) == b"\x77"
        # 空洞
        assert not bus.is_valid_addr(0x2000_0000)
