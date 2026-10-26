#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""SPI 主模式控制器测试.

覆盖: 寄存器读写 / STATUS 的位语义 / 全双工镜像 / DIV 掩码 / 未定义偏移 / Bus 集成 /
平台配置与 DTB 节点生成.
"""

import struct

import libfdt
import pytest

from pyremu import configs_gen
from pyremu.emulator import Emulator
from pyremu.memory.bus import Bus
from pyremu.peripheral.spi import SPI
from pyremu.platform import PeripheralConfig, PlatformConfig

CTRL = SPI.REG_CTRL
STATUS = SPI.REG_STATUS
TXDATA = SPI.REG_TXDATA
RXDATA = SPI.REG_RXDATA
DIV = SPI.REG_DIV

SPI_BASE = 0x1000_1000


def _u32(dev: SPI, offset: int) -> int:
    """按 32 位宽读取一个寄存器."""
    return int.from_bytes(dev.read(offset, 4), "little", signed=False)


class TestSPIRegisters:
    """寄存器的读写语义."""

    @pytest.fixture
    def spi(self) -> SPI:
        return SPI()

    def test_ctrl_roundtrip(self, spi):
        """CTRL 写入的三位置位后原样读出."""
        spi.write(CTRL, bytes([SPI.CTRL_EN | SPI.CTRL_CPOL | SPI.CTRL_CPHA]))
        assert _u32(spi, CTRL) == 0b111

    def test_ctrl_is_not_masked(self, spi):
        """CTRL 不做位宽截断: 全 1 写入后读回全 1."""
        spi.write(CTRL, b"\xff\xff\xff\xff")
        assert _u32(spi, CTRL) == 0xFFFF_FFFF

    def test_div_masked_to_byte(self, spi):
        """DIV 只保留低 8 位."""
        spi.write(DIV, struct.pack("<I", 0x1FF))
        assert _u32(spi, DIV) == 0xFF

    def test_read_size_truncates(self, spi):
        """读取按请求宽度截断, 低位在前."""
        spi.write(CTRL, struct.pack("<I", 0x1234_5678))
        assert spi.read(CTRL, 1) == b"\x78"
        assert spi.read(CTRL, 2) == b"\x78\x56"

    def test_txdata_mirrors_to_rxdata(self, spi):
        """写 TXDATA 立即镜像到 RXDATA."""
        spi.write(TXDATA, b"\xa5")
        assert _u32(spi, RXDATA) == 0xA5

    def test_rxdata_keeps_all_64_bits(self, spi):
        """RXDATA 保留 8 字节传输的全部 8 个字节.

        改前掩码写作 0xFF_FFFF_FFFF_FFFF (56 位), 最高字节被清零.
        """
        value = 0xFF00_FF00_FF00_FF00
        spi.write(TXDATA, struct.pack("<Q", value))
        assert spi.read(RXDATA, 8) == struct.pack("<Q", value)
        # 56 位掩码在 bit 63:56 上产生的取值, 修复后不得再出现.
        buggy = value & 0x00FF_FFFF_FFFF_FFFF
        assert spi.read(RXDATA, 8) != struct.pack("<Q", buggy)

    def test_unmapped_offset_reads_zero(self, spi):
        """未定义偏移读返回全 0."""
        assert spi.read(0x14, 4) == b"\x00" * 4
        assert spi.read(0x20, 8) == b"\x00" * 8

    def test_unmapped_offset_write_is_ignored(self, spi):
        """未定义偏移写无副作用."""
        spi.write(0x14, b"\xff\xff\xff\xff")
        assert spi.read(0x14, 4) == b"\x00" * 4
        assert _u32(spi, CTRL) == 0


class TestSPIStatus:
    """STATUS 的位语义."""

    @pytest.fixture
    def spi(self) -> SPI:
        return SPI()

    def test_idle_without_enable(self, spi):
        """未使能时 STATUS 为 0: 既不忙也不报告接收数据可用."""
        assert _u32(spi, STATUS) == 0

    def test_rx_avail_when_enabled_and_idle(self, spi):
        """使能且空闲时置 RX_AVAIL."""
        spi.write(CTRL, bytes([SPI.CTRL_EN]))
        assert _u32(spi, STATUS) == SPI.STATUS_RX_AVAIL

    def test_busy_excludes_rx_avail(self, spi):
        """传输进行中置 BUSY, 且不置 RX_AVAIL."""
        spi.write(CTRL, bytes([SPI.CTRL_EN]))
        spi._busy = True
        status = _u32(spi, STATUS)
        assert status & SPI.STATUS_BUSY
        assert not status & SPI.STATUS_RX_AVAIL

    def test_txdata_clears_busy(self, spi):
        """写 TXDATA 即完成传输, BUSY 随之清除."""
        spi.write(CTRL, bytes([SPI.CTRL_EN]))
        spi._busy = True
        spi.write(TXDATA, b"\x01")
        assert not _u32(spi, STATUS) & SPI.STATUS_BUSY


class TestSPIBus:
    """Bus 集成."""

    def test_bus_routes_read_and_write(self):
        """经 Bus 读写 SPI 地址应路由到设备."""
        spi = SPI(base=SPI_BASE)
        bus = Bus(ram_size=1024 * 1024, ram_base=0)
        bus.add_device(SPI_BASE, spi)
        assert bus.is_device_addr(SPI_BASE)
        bus.write(SPI_BASE + TXDATA, b"\x5a")
        assert bus.read(SPI_BASE + RXDATA, 4) == b"\x5a\x00\x00\x00"


class TestSPIConfig:
    """平台配置与 DTB 节点生成."""

    @staticmethod
    def _cfg() -> PlatformConfig:
        return PlatformConfig(
            num_harts=1,
            ram_base=0x8000_0000,
            prog_cnt=0x8000_0000,
            periph=PeripheralConfig(spi_base=SPI_BASE),
        )

    def test_default_spi_base_follows_build_config(self):
        assert PeripheralConfig().spi_base == configs_gen.SPI_BASE

    def test_emulator_registers_device(self):
        """基址非零时 Emulator 创建 SPI 并挂到 Bus 上."""
        emu = Emulator(self._cfg(), bootargs="console=ttySIF0")
        assert emu.spi is not None
        assert emu.bus.is_device_addr(SPI_BASE)

    def test_dtb_contains_spi_node(self):
        """DTB 含 /soc/spi@... 节点, compatible 与 reg 正确."""
        emu = Emulator(self._cfg(), bootargs="console=ttySIF0")
        fdt = libfdt.Fdt(emu.build_dtb())
        node = fdt.path_offset(f"/soc/spi@{SPI_BASE:x}")
        assert node >= 0
        assert bytes(fdt.getprop(node, "compatible")) == b"pyremu,spi0\x00"
        base_hi, base_lo, size_hi, size_lo = struct.unpack(
            ">IIII", bytes(fdt.getprop(node, "reg"))
        )
        assert (base_hi << 32) | base_lo == SPI_BASE
        assert (size_hi << 32) | size_lo == 0x1000

    def test_zero_base_disables_device(self):
        """基址为 0 时不注册设备, DTB 中也没有对应节点."""
        cfg = self._cfg()
        cfg.periph.spi_base = 0
        emu = Emulator(cfg, bootargs="console=ttySIF0")
        assert emu.spi is None
        assert not emu.bus.is_device_addr(SPI_BASE)
        fdt = libfdt.Fdt(emu.build_dtb())
        with pytest.raises(libfdt.FdtException):
            fdt.path_offset("/soc/spi@0")
