#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""多 hart 看门狗测试.

覆盖: 每 hart 寄存器块的读写 / 计数递减与超时恢复 / 使能位的作用 / 踢狗 /
越界 hart 与未定义偏移 / Bus 集成 / 平台配置与 DTB 节点生成.
"""

import struct
import threading

import libfdt
import pytest

from pyremu.emulator import Emulator
from pyremu.interrupt.clint import CLINT
from pyremu.memory.bus import Bus
from pyremu.peripheral.watchdog import HartWatchdog
from pyremu.platform import PeripheralConfig, PlatformConfig

WDOG_CTRL = 0x00
WDOG_TIMEOUT = 0x04
WDOG_COUNT = 0x08
WDOG_KICK = 0x0C
WDOG_STRIDE = 0x10
CTRL_ENABLE = 1 << 0

WDOG_BASE = 0x1000_4000
MSIP_STRIDE = 4


def _u32(dev: HartWatchdog, offset: int) -> int:
    """按 32 位宽读取一个寄存器."""
    return int.from_bytes(dev.read(offset, 4), "little", signed=False)


def _msip_pending(clint: CLINT, hart_id: int) -> bool:
    """读 CLINT 的 msip 寄存器, 判定该 hart 的软件中断是否挂起."""
    return bool(int.from_bytes(clint.read(hart_id * MSIP_STRIDE, 4), "little") & 1)


class HartStub:
    """恢复操作只读取等待标志、停机标志与 hart 编号."""

    def __init__(self, hart_id: int, is_waiting: bool, is_halted: bool = False) -> None:
        self.id = hart_id
        self._waiting = is_waiting
        self._halted = is_halted


@pytest.fixture
def clint() -> CLINT:
    return CLINT(num_harts=4)


@pytest.fixture
def wake_event() -> threading.Event:
    return threading.Event()


@pytest.fixture
def wdog(clint, wake_event) -> HartWatchdog:
    return HartWatchdog(clint=clint, wake_event=wake_event, num_harts=2, timeout=500)


class TestWatchdogRegisters:
    """每 hart 寄存器块的读写."""

    def test_default_state_is_enabled_and_loaded(self, wdog):
        """构造后每个 hart 使能, 计数等于超时周期."""
        assert _u32(wdog, WDOG_CTRL) == CTRL_ENABLE
        assert _u32(wdog, WDOG_TIMEOUT) == 500
        assert _u32(wdog, WDOG_COUNT) == 500

    def test_size_covers_all_harts(self, wdog):
        """设备大小为 hart 数乘以寄存器块间距."""
        assert wdog.size == 2 * WDOG_STRIDE

    def test_register_block_per_hart(self, wdog):
        """第 1 个 hart 的寄存器块从偏移 0x10 起, 与第 0 个互不影响."""
        wdog.write(WDOG_STRIDE + WDOG_TIMEOUT, struct.pack("<I", 77))
        wdog.write(WDOG_STRIDE + WDOG_CTRL, struct.pack("<I", 0))
        assert _u32(wdog, WDOG_STRIDE + WDOG_TIMEOUT) == 77
        assert _u32(wdog, WDOG_STRIDE + WDOG_CTRL) == 0
        assert _u32(wdog, WDOG_TIMEOUT) == 500
        assert _u32(wdog, WDOG_CTRL) == CTRL_ENABLE

    def test_kick_register_reads_zero(self, wdog):
        """踢狗寄存器只写, 读出为 0."""
        assert wdog.read(WDOG_KICK, 4) == b"\x00" * 4
        assert wdog.read(WDOG_KICK, 8) == b"\x00" * 8

    def test_offset_outside_register_boundary_reads_zero(self, wdog):
        """偏移未落在任何 4 字节寄存器边界上时读出为 0."""
        assert wdog.read(0x01, 4) == b"\x00" * 4

    def test_hart_beyond_count_reads_zero(self, wdog):
        """hart 编号超出设备覆盖范围时读出为所请求长度的 0 字节."""
        assert wdog.read(2 * WDOG_STRIDE, 4) == b"\x00" * 4
        assert wdog.read(3 * WDOG_STRIDE + WDOG_COUNT, 8) == b"\x00" * 8

    def test_write_beyond_hart_count_is_ignored(self, wdog):
        """写超出覆盖范围的 hart 无副作用."""
        wdog.write(2 * WDOG_STRIDE + WDOG_TIMEOUT, struct.pack("<I", 9))
        assert wdog.read(2 * WDOG_STRIDE + WDOG_TIMEOUT, 4) == b"\x00" * 4

    def test_count_register_is_read_only(self, wdog):
        """写计数寄存器被忽略."""
        wdog.write(WDOG_COUNT, struct.pack("<I", 1))
        assert _u32(wdog, WDOG_COUNT) == 500

    def test_kick_resets_count_to_timeout(self, wdog):
        """写踢狗寄存器把计数重置为该 hart 的超时周期."""
        wdog.write(WDOG_TIMEOUT, struct.pack("<I", 7))
        wdog.write(WDOG_KICK, struct.pack("<I", 1))
        assert _u32(wdog, WDOG_COUNT) == 7

    def test_count_decrements_on_tick(self, wdog):
        """每推进一次计数器减一."""
        wdog.tick([HartStub(0, is_waiting=False)])
        assert _u32(wdog, WDOG_COUNT) == 499
        assert _u32(wdog, WDOG_STRIDE + WDOG_COUNT) == 499


class TestWatchdogEnabledBit:
    """使能位对计数与超时恢复的作用."""

    def test_disabled_hart_is_not_counted_down(self, clint, wake_event):
        """未使能的 hart 计数不推进, 也不触发恢复操作."""
        wdog = HartWatchdog(clint=clint, wake_event=wake_event, num_harts=1, timeout=1)
        wdog.write(WDOG_CTRL, struct.pack("<I", 0))
        assert wdog.tick([HartStub(0, is_waiting=True)]) is False
        assert _u32(wdog, WDOG_COUNT) == 1
        assert not wake_event.is_set()

    def test_only_enabled_hart_fires(self, clint, wake_event):
        """某个 hart 未使能时, 其它 hart 到期不影响它的计数."""
        wdog = HartWatchdog(clint=clint, wake_event=wake_event, num_harts=2, timeout=2)
        wdog.write(WDOG_STRIDE + WDOG_CTRL, struct.pack("<I", 0))
        assert wdog.tick([]) is False
        assert wdog.tick([]) is True
        assert _u32(wdog, WDOG_COUNT) == 2
        assert _u32(wdog, WDOG_STRIDE + WDOG_COUNT) == 2

    def test_kick_all_skips_disabled_hart(self, clint, wake_event):
        """kick_all 只重置使能的 hart 的计数."""
        wdog = HartWatchdog(clint=clint, wake_event=wake_event, num_harts=2, timeout=9)
        wdog.tick([])
        wdog.tick([])
        wdog.write(WDOG_STRIDE + WDOG_CTRL, struct.pack("<I", 0))
        wdog.kick_all()
        assert _u32(wdog, WDOG_COUNT) == 9
        assert _u32(wdog, WDOG_STRIDE + WDOG_COUNT) == 7

    def test_kick_single_hart(self, clint, wake_event):
        """kick 只作用于指定的 hart."""
        wdog = HartWatchdog(clint=clint, wake_event=wake_event, num_harts=2, timeout=9)
        wdog.tick([])
        wdog.kick(1)
        assert _u32(wdog, WDOG_COUNT) == 8
        assert _u32(wdog, WDOG_STRIDE + WDOG_COUNT) == 9

    def test_kick_disabled_or_out_of_range_hart_is_noop(self, clint, wake_event):
        """kick 对未使能的 hart 与越界的 hart 编号都不改动计数."""
        wdog = HartWatchdog(clint=clint, wake_event=wake_event, num_harts=2, timeout=9)
        wdog.tick([])
        wdog.tick([])
        wdog.write(WDOG_CTRL, struct.pack("<I", 0))
        wdog.kick(0)
        wdog.kick(9)
        assert _u32(wdog, WDOG_COUNT) == 7
        assert _u32(wdog, WDOG_STRIDE + WDOG_COUNT) == 7


class TestWatchdogRecover:
    """计数归零后的恢复操作."""

    def test_fire_resets_count_and_sets_event(self, clint, wake_event):
        """归零时执行一次恢复操作, 计数回到超时周期."""
        wdog = HartWatchdog(clint=clint, wake_event=wake_event, num_harts=1, timeout=1)
        assert wdog.tick([]) is True
        assert _u32(wdog, WDOG_COUNT) == 1
        assert wake_event.is_set()

    def test_recover_injects_msip_to_waiting_harts_only(self, clint, wake_event):
        """只有处于等待且未停机的 hart 收到核间中断."""
        wdog = HartWatchdog(clint=clint, wake_event=wake_event, num_harts=1, timeout=1)
        harts = [
            HartStub(0, is_waiting=True),
            HartStub(1, is_waiting=False),
            HartStub(2, is_waiting=True, is_halted=True),
        ]
        assert wdog.tick(harts) is True
        assert _msip_pending(clint, 0)
        assert not _msip_pending(clint, 1)
        assert not _msip_pending(clint, 2)

    def test_recover_does_not_inject_when_no_hart_waits(self, clint, wake_event):
        """没有等待中的 hart 时恢复操作不产生核间中断."""
        wdog = HartWatchdog(clint=clint, wake_event=wake_event, num_harts=1, timeout=1)
        assert wdog.tick([HartStub(0, is_waiting=False)]) is True
        assert not _msip_pending(clint, 0)


class TestWatchdogBus:
    """Bus 集成."""

    def test_bus_routes_read_and_write(self, clint, wake_event):
        """经 Bus 读写看门狗地址应路由到设备."""
        wdog = HartWatchdog(
            clint=clint, wake_event=wake_event, base=WDOG_BASE, num_harts=1, timeout=5
        )
        bus = Bus(ram_size=1024 * 1024, ram_base=0)
        bus.add_device(WDOG_BASE, wdog)
        assert bus.is_device_addr(WDOG_BASE)
        bus.write(WDOG_BASE + WDOG_TIMEOUT, struct.pack("<I", 12))
        assert bus.read(WDOG_BASE + WDOG_TIMEOUT, 4) == struct.pack("<I", 12)


class TestWatchdogConfig:
    """平台配置与 DTB 节点生成."""

    @staticmethod
    def _cfg() -> PlatformConfig:
        return PlatformConfig(
            num_harts=1,
            ram_base=0x8000_0000,
            prog_cnt=0x8000_0000,
            periph=PeripheralConfig(),
        )

    def test_default_watchdog_base(self):
        """默认 PeripheralConfig 启用看门狗."""
        assert PeripheralConfig().watchdog_base == WDOG_BASE

    def test_emulator_registers_device(self):
        """Emulator 恒创建看门狗并挂到 Bus 上."""
        emu = Emulator(self._cfg(), bootargs="console=ttySIF0")
        assert emu.watchdog is not None
        assert emu.bus.is_device_addr(WDOG_BASE)

    def test_dtb_contains_watchdog_node(self):
        """DTB 含 /soc/watchdog@... 节点, compatible 与 reg 正确."""
        emu = Emulator(self._cfg(), bootargs="console=ttySIF0")
        fdt = libfdt.Fdt(emu.build_dtb())
        node = fdt.path_offset(f"/soc/watchdog@{WDOG_BASE:x}")
        assert node >= 0
        assert bytes(fdt.getprop(node, "compatible")) == b"pyremu,hart-watchdog-1.0\x00"
        base_hi, base_lo, size_hi, size_lo = struct.unpack(
            ">IIII", bytes(fdt.getprop(node, "reg"))
        )
        assert (base_hi << 32) | base_lo == WDOG_BASE
        assert (size_hi << 32) | size_lo == 0x100
