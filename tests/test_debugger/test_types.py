#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""测试 pyremu.debug.types."""

from unittest.mock import MagicMock

from pyremu.debug.types import (
    Breakpoint,
    HartSnapshot,
    MemoryChange,
    MemWriteTracker,
    StackFrame,
)


class TestHartSnapshot:
    """Hart 状态快照."""

    def test_create(self):
        snap = HartSnapshot(
            pc=0x80000000,
            gpr_vals=[0] * 32,
            csr_vals={"mstatus": 0},
            mode=3,
            reservation_valid=False,
            reservation_addr=0,
        )
        assert snap.pc == 0x80000000
        assert len(snap.gpr_vals) == 32
        assert snap.mode == 3
        assert not snap.reservation_valid

    def test_with_reservation(self):
        snap = HartSnapshot(
            pc=0x1000,
            gpr_vals=[i for i in range(32)],
            csr_vals={},
            mode=3,
            reservation_valid=True,
            reservation_addr=0x2000,
        )
        assert snap.reservation_valid
        assert snap.reservation_addr == 0x2000


class TestMemoryChange:
    """内存变更记录."""

    def test_create(self):
        mc = MemoryChange(addr=0x1000, old=b"\x00\x00\x00\x00")
        assert mc.addr == 0x1000
        assert mc.old == b"\x00\x00\x00\x00"


class TestStackFrame:
    """栈帧."""

    def test_create(self):
        sf = StackFrame(
            idx=0, fp=0x80001000, sp=0x80000FF0,
            ra=0x80000100, pc=0x800000FC,
        )
        assert sf.idx == 0
        assert sf.fp == 0x80001000
        assert sf.sp == 0x80000FF0
        assert sf.ra == 0x80000100
        assert sf.pc == 0x800000FC

    def test_with_note(self):
        sf = StackFrame(
            idx=1, fp=0, sp=0, ra=0, pc=0x80000000,
            note="trap 入口", mode="S",
        )
        assert sf.note == "trap 入口"
        assert sf.mode == "S"


class TestMemWriteTracker:
    """内存写入追踪器."""

    def test_tracks_write_and_records_old(self):
        bus = MagicMock()
        bus.try_read.return_value = b"\x00\x00\x00\x00"
        orig = MagicMock()
        tracker = MemWriteTracker(bus, orig)
        tracker(0x1000, b"\xEF\xBE\xAD\xDE")
        assert len(tracker.changes) == 1
        assert tracker.changes[0].addr == 0x1000
        assert tracker.changes[0].old == b"\x00\x00\x00\x00"
        orig.assert_called_once_with(0x1000, b"\xEF\xBE\xAD\xDE")

    def test_multiple_writes_accumulate(self):
        bus = MagicMock()
        bus.try_read.return_value = b"\x00" * 8
        orig = MagicMock()
        tracker = MemWriteTracker(bus, orig)
        tracker(0x1000, b"\x01" * 8)
        tracker(0x2000, b"\x02" * 8)
        assert len(tracker.changes) == 2


class TestBreakpoint:
    """断点数据模型."""

    def test_addr_breakpoint(self):
        bp = Breakpoint(kind="addr", value=0x80000000, desc="entry")
        assert bp.kind == "addr"
        assert bp.value == 0x80000000
        assert bp.cond_type == ""

    def test_cond_breakpoint(self):
        bp = Breakpoint(
            kind="cond", value=0, desc="test",
            cond_type="csr", cond_reg="mcause",
            cond_op="==", cond_val=7,
        )
        assert bp.kind == "cond"
        assert bp.cond_reg == "mcause"
        assert bp.cond_op == "=="
        assert bp.cond_val == 7
