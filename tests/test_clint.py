#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""CLINT 设备测试: MSIP (IPI), MTIMECMP (定时器), MTIME."""

import pytest

from pyremu.interrupt.clint import (
    CLINT,
    CLINT_BASE,
    MSIP_OFFSET,
    MTIME_OFFSET,
    MTIMECMP_OFFSET,
)
from pyremu.interrupt.controller import INT_SOURCE_MIP_MASK, IntSource
from pyremu.memory.bus import Bus


class TestCLINTRegisters:
    """CLINT 寄存器读写."""

    @pytest.fixture
    def clint(self) -> CLINT:
        return CLINT(num_harts=4)

    def test_read_msip_initial_zero(self, clint):
        """初始 MSIP 应为 0."""
        data = clint.read(MSIP_OFFSET, 4)  # hart 0
        assert data == b"\x00\x00\x00\x00"

    def test_write_msip_triggers_ipi(self, clint):
        """写 MSIP 应设置软件中断挂起."""
        clint.write(MSIP_OFFSET + 4, b"\x01\x00\x00\x00")  # hart 1 msip = 1
        has_pending, mip, src = clint.check_interrupt(1)
        assert has_pending
        assert mip & INT_SOURCE_MIP_MASK[IntSource.MSI]
        assert src == IntSource.MSI

    def test_write_msip_zero_clears_ipi(self, clint):
        """写 0 到 MSIP 应清除中断."""
        clint.write(MSIP_OFFSET, b"\x01\x00\x00\x00")  # hart 0 msip = 1
        has_pending, _, _ = clint.check_interrupt(0)
        assert has_pending
        clint.write(MSIP_OFFSET, b"\x00\x00\x00\x00")  # hart 0 msip = 0
        has_pending, _, _ = clint.check_interrupt(0)
        assert not has_pending

    def test_send_ipi(self, clint):
        """send_ipi 辅助方法."""
        clint.send_ipi(2)  # 向 hart 2 发送 IPI
        has_pending, _, _ = clint.check_interrupt(2)
        assert has_pending

    def test_clear_ipi(self, clint):
        """clear_ipi 辅助方法."""
        clint.send_ipi(3)
        clint.clear_ipi(3)
        has_pending, _, _ = clint.check_interrupt(3)
        assert not has_pending

    def test_mtimecmp_timer_interrupt(self, clint):
        """mtime >= mtimecmp 时应触发定时器中断."""
        clint.tick(100)  # mtime = 100
        # 设置 hart 0 的 mtimecmp = 50 (已过期)
        clint.write(MTIMECMP_OFFSET, (50).to_bytes(8, "little"))
        has_pending, mip, src = clint.check_interrupt(0)
        assert has_pending
        assert mip & INT_SOURCE_MIP_MASK[IntSource.MTI]

    def test_mtimecmp_not_reached(self, clint):
        """mtime < mtimecmp 时不应触发定时器中断."""
        clint.tick(10)  # mtime = 10
        clint.write(MTIMECMP_OFFSET, (100).to_bytes(8, "little"))
        has_pending, _, _ = clint.check_interrupt(0)
        assert not has_pending

    def test_mtime_read(self, clint):
        """MTIME 寄存器读."""
        clint.tick(42)
        data = clint.read(MTIME_OFFSET, 8)
        val = int.from_bytes(data, "little")
        assert val == 42

    def test_mtime_write(self, clint):
        """MTIME 寄存器写."""
        clint.write(MTIME_OFFSET, (0xDEAD).to_bytes(8, "little"))
        assert clint.get_mtime() == 0xDEAD

    def test_interrupt_priority(self, clint):
        """中断优先级: MSI > MTI."""
        clint.send_ipi(0)  # MSI
        clint.tick(100)
        clint.write(MTIMECMP_OFFSET, (50).to_bytes(8, "little"))  # MTI
        _, _, src = clint.check_interrupt(0)
        assert src == IntSource.MSI, "MSI 优先级应高于 MTI"


class TestCLINTBusIntegration:
    """通过 Bus 访问 CLINT."""

    def test_bus_write_to_msip(self):
        """通过总线写 MSIP → IPI 触发."""
        clint = CLINT(num_harts=4)
        bus = Bus(ram_size=1024 * 1024)
        bus.add_device(CLINT_BASE, clint)

        # hart 0 写 hart 1 的 msip
        bus.write(CLINT_BASE + MSIP_OFFSET + 4, b"\x01\x00\x00\x00")
        has_pending, _, _ = clint.check_interrupt(1)
        assert has_pending

    def test_bus_read_device_addr_detected(self):
        """Bus.is_device_addr 应检测 CLINT 地址."""
        clint = CLINT(num_harts=2)
        bus = Bus(ram_size=1024 * 1024)
        bus.add_device(CLINT_BASE, clint)
        assert bus.is_device_addr(CLINT_BASE)
        assert bus.is_device_addr(CLINT_BASE + MSIP_OFFSET)
        assert bus.is_device_addr(CLINT_BASE + MTIME_OFFSET)
