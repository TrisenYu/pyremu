#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""I2C 主模式控制器测试.

覆盖: 寄存器读写 / STATUS 的位语义 / START 与 STOP 的边沿 / 读方向位对 DATA 的影响 /
从设备应答判定 / Bus 集成 / 平台配置与 DTB 节点生成.
"""

import struct

import libfdt
import pytest

from pyremu.emulator import Emulator
from pyremu.memory.bus import Bus
from pyremu.peripheral.i2c import I2C
from pyremu.platform import PeripheralConfig, PlatformConfig

CTRL = I2C.REG_CTRL
STATUS = I2C.REG_STATUS
DATA = I2C.REG_DATA
ADDR = I2C.REG_ADDR
PRESCALE = I2C.REG_PRESCALE

I2C_BASE = 0x1000_2000
RW_BIT = 0x100  # ADDR 的 bit8: 1 表示读方向


def _u32(dev: I2C, offset: int) -> int:
    """按 32 位宽读取一个寄存器."""
    return int.from_bytes(dev.read(offset, 4), "little", signed=False)


class TestI2CRegisters:
    """寄存器的读写语义."""

    @pytest.fixture
    def i2c(self) -> I2C:
        return I2C()

    def test_ctrl_roundtrip(self, i2c):
        """CTRL 的四位置位后原样读出."""
        i2c.write(CTRL, bytes([0b1111]))
        assert _u32(i2c, CTRL) == 0b1111

    def test_prescale_masked_to_halfword(self, i2c):
        """PRESCALE 只保留低 16 位."""
        i2c.write(PRESCALE, struct.pack("<I", 0x1_FFFF))
        assert _u32(i2c, PRESCALE) == 0xFFFF

    def test_data_write_masked_to_byte(self, i2c):
        """写入 DATA 的值截断到 8 位, 读回同一值."""
        i2c.write(DATA, struct.pack("<I", 0x1FF))
        assert _u32(i2c, DATA) == 0xFF

    def test_addr_roundtrip_keeps_direction_bit(self, i2c):
        """ADDR 原样保存, 含读方向位."""
        i2c.write(ADDR, struct.pack("<I", RW_BIT | 0x42))
        assert _u32(i2c, ADDR) == RW_BIT | 0x42

    def test_unmapped_offset_reads_zero(self, i2c):
        """未定义偏移读返回全 0."""
        assert i2c.read(0x14, 4) == b"\x00" * 4
        assert i2c.read(0x20, 8) == b"\x00" * 8

    def test_unmapped_offset_write_is_ignored(self, i2c):
        """未定义偏移写无副作用."""
        i2c.write(0x14, b"\xff\xff\xff\xff")
        assert i2c.read(0x14, 4) == b"\x00" * 4
        assert _u32(i2c, CTRL) == 0

    def test_read_size_truncates(self, i2c):
        """读取按请求宽度截断, 低位在前."""
        i2c.write(PRESCALE, struct.pack("<I", 0x1234))
        assert i2c.read(PRESCALE, 1) == b"\x34"
        assert i2c.read(PRESCALE, 2) == b"\x34\x12"


class TestI2CStatus:
    """STATUS 的位语义."""

    @pytest.fixture
    def i2c(self) -> I2C:
        return I2C()

    def test_tx_done_always_set(self, i2c):
        """TX_DONE 恒置位: 发送始终就绪."""
        assert _u32(i2c, STATUS) & I2C.STATUS_TX_DONE

    def test_idle_reports_rx_avail(self, i2c):
        """空闲时置 RX_AVAIL."""
        assert _u32(i2c, STATUS) & I2C.STATUS_RX_AVAIL

    def test_start_sets_busy(self, i2c):
        """CTRL.START 由 0 置 1 时进入忙状态."""
        i2c.write(CTRL, bytes([I2C.CTRL_EN | I2C.CTRL_START]))
        status = _u32(i2c, STATUS)
        assert status & I2C.STATUS_BUSY
        assert not status & I2C.STATUS_RX_AVAIL

    def test_start_held_does_not_reassert_busy(self, i2c):
        """START 位保持为 1 期间再次写 CTRL 不重新进入忙状态."""
        i2c.write(CTRL, bytes([I2C.CTRL_START]))
        i2c.read(DATA, 4)  # 读 DATA 结束本次传输
        i2c.write(CTRL, bytes([I2C.CTRL_START]))
        assert not _u32(i2c, STATUS) & I2C.STATUS_BUSY

    def test_stop_clears_busy(self, i2c):
        """CTRL.STOP 置位结束传输."""
        i2c.write(CTRL, bytes([I2C.CTRL_START]))
        i2c.write(CTRL, bytes([I2C.CTRL_START | I2C.CTRL_STOP]))
        assert not _u32(i2c, STATUS) & I2C.STATUS_BUSY

    def test_data_access_clears_busy(self, i2c):
        """读写 DATA 都结束本次传输."""
        i2c.write(CTRL, bytes([I2C.CTRL_START]))
        i2c.write(DATA, b"\x11")
        assert not _u32(i2c, STATUS) & I2C.STATUS_BUSY


class TestI2CSlaveResponse:
    """从设备应答与数据方向."""

    @pytest.fixture
    def i2c(self) -> I2C:
        return I2C()

    def test_nack_when_slave_absent(self, i2c):
        """地址无对应从设备数据时置 NACK."""
        i2c.write(ADDR, struct.pack("<I", 0x50))
        assert _u32(i2c, STATUS) & I2C.STATUS_NACK

    def test_ack_when_slave_present(self, i2c):
        """地址已登记从设备数据时不置 NACK."""
        i2c.set_slave_data(0x50, 0x7E)
        i2c.write(ADDR, struct.pack("<I", 0x50))
        assert not _u32(i2c, STATUS) & I2C.STATUS_NACK

    def test_slave_data_masked_to_byte(self, i2c):
        """从设备数据截断到 8 位."""
        i2c.set_slave_data(0x30, 0x1AB)
        i2c.write(ADDR, struct.pack("<I", RW_BIT | 0x30))
        assert _u32(i2c, DATA) == 0xAB

    def test_read_direction_returns_slave_data(self, i2c):
        """ADDR 的读方向位置位时, 读 DATA 取得从设备寄存器的值."""
        i2c.set_slave_data(0x30, 0x5A)
        i2c.write(DATA, b"\x99")  # 先写入本地数据, 验证读方向位优先
        i2c.write(ADDR, struct.pack("<I", RW_BIT | 0x30))
        assert _u32(i2c, DATA) == 0x5A

    def test_write_direction_returns_local_data(self, i2c):
        """读方向位为 0 时, 读 DATA 取得本设备写入的值."""
        i2c.set_slave_data(0x30, 0x5A)
        i2c.write(ADDR, struct.pack("<I", 0x30))
        i2c.write(DATA, b"\x99")
        assert _u32(i2c, DATA) == 0x99

    def test_read_direction_without_slave_data_returns_zero(self, i2c):
        """读方向位指向未登记的从设备寄存器时读回 0."""
        i2c.write(ADDR, struct.pack("<I", RW_BIT | 0x3F))
        assert _u32(i2c, DATA) == 0

    def test_start_clears_nack(self, i2c):
        """CTRL.START 由 0 置 1 时清除上一次的 NACK."""
        i2c.write(ADDR, struct.pack("<I", 0x50))
        assert _u32(i2c, STATUS) & I2C.STATUS_NACK
        i2c.write(CTRL, bytes([I2C.CTRL_START]))
        assert not _u32(i2c, STATUS) & I2C.STATUS_NACK


class TestI2CBus:
    """Bus 集成."""

    def test_bus_routes_read_and_write(self):
        """经 Bus 读写 I2C 地址应路由到设备."""
        i2c = I2C(base=I2C_BASE)
        bus = Bus(ram_size=1024 * 1024, ram_base=0)
        bus.add_device(I2C_BASE, i2c)
        assert bus.is_device_addr(I2C_BASE)
        bus.write(I2C_BASE + PRESCALE, struct.pack("<I", 0x1234))
        assert bus.read(I2C_BASE + PRESCALE, 4) == struct.pack("<I", 0x1234)


class TestI2CConfig:
    """平台配置与 DTB 节点生成."""

    @staticmethod
    def _cfg() -> PlatformConfig:
        return PlatformConfig(
            num_harts=1,
            ram_base=0x8000_0000,
            prog_cnt=0x8000_0000,
            periph=PeripheralConfig(),
        )

    def test_default_i2c_base(self):
        """默认 PeripheralConfig 启用 I2C."""
        assert PeripheralConfig().i2c_base == I2C_BASE

    def test_emulator_registers_device(self):
        """基址非零时 Emulator 创建 I2C 并挂到 Bus 上."""
        emu = Emulator(self._cfg(), bootargs="console=ttySIF0")
        assert emu.i2c is not None
        assert emu.bus.is_device_addr(I2C_BASE)

    def test_dtb_contains_i2c_node(self):
        """DTB 含 /soc/i2c@... 节点, compatible 与 reg 正确."""
        emu = Emulator(self._cfg(), bootargs="console=ttySIF0")
        fdt = libfdt.Fdt(emu.build_dtb())
        node = fdt.path_offset(f"/soc/i2c@{I2C_BASE:x}")
        assert node >= 0
        assert bytes(fdt.getprop(node, "compatible")) == b"pyremu,i2c0\x00"
        base_hi, base_lo, size_hi, size_lo = struct.unpack(
            ">IIII", bytes(fdt.getprop(node, "reg"))
        )
        assert (base_hi << 32) | base_lo == I2C_BASE
        assert (size_hi << 32) | size_lo == 0x1000

    def test_zero_base_disables_device(self):
        """基址为 0 时不注册设备, DTB 中也没有对应节点."""
        cfg = self._cfg()
        cfg.periph.i2c_base = 0
        emu = Emulator(cfg, bootargs="console=ttySIF0")
        assert emu.i2c is None
        assert not emu.bus.is_device_addr(I2C_BASE)
        fdt = libfdt.Fdt(emu.build_dtb())
        with pytest.raises(libfdt.FdtException):
            fdt.path_offset("/soc/i2c@0")
