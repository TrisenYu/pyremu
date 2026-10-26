#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""trap cause code 的编码与名称映射."""

import pytest

from pyremu.core.trap_def import (
    trap_cause_code,
    trap_cause_name,
    trap_is_interrupt,
    TrapType,
)


class TestTrapCauseCode:
    """验证每个 TrapType 对应的 mcause 编码是否正确."""

    def test_exception_codes(self):
        """异常: bit 63 = 0, 编码 = 规范定义的编号."""
        cases = [
            (TrapType.InstrAddrMisaligned, 0),
            (TrapType.InstrAccessFault, 1),
            (TrapType.IllInstr, 2),
            (TrapType.Breakpoint, 3),
            (TrapType.LdAddrMisaligned, 4),
            (TrapType.LdAccessFault, 5),
            (TrapType.StAddrMisaligned, 6),
            (TrapType.StAccessFault, 7),
            (TrapType.EcallFromUmode, 8),
            (TrapType.EcallFromSmode, 9),
            (TrapType.EcallFromMmode, 11),
            (TrapType.InstrPageFault, 12),
            (TrapType.LdPageFault, 13),
            (TrapType.StPageFault, 15),
        ]
        for trap, expected in cases:
            code = trap_cause_code(trap)
            assert code == expected, f"{trap.name}: expected mcause={expected}, got {code}"
            assert not trap_is_interrupt(trap), f"{trap.name} should NOT be an interrupt"

    def test_interrupt_codes(self):
        """中断: bit 63 = 1, 低 63 位 = 中断编号."""
        cases = [
            (TrapType.UmodeSoftInterrupt, 0),
            (TrapType.SmodeSoftInterrupt, 1),
            (TrapType.MmodeSoftInterrupt, 3),
            (TrapType.UmodeTimerInterrupt, 4),
            (TrapType.SmodeTimerInterrupt, 5),
            (TrapType.MmodeTimerInterrupt, 7),
            (TrapType.UmodeExternInterrupt, 8),
            (TrapType.SmodeExternInterrupt, 9),
            (TrapType.MmodeExternInterrupt, 11),
        ]
        for trap, exc_code in cases:
            code = trap_cause_code(trap)
            expected = exc_code | (1 << 63)
            assert code == expected, (
                f"{trap.name}: expected mcause=0x{expected:016x}, got 0x{code:016x}"
            )
            assert trap_is_interrupt(trap), f"{trap.name} SHOULD be an interrupt"

    def test_all_trap_types_have_code(self):
        """确保所有 TrapType 成员都有对应的编码."""
        for trap in TrapType:
            code = trap_cause_code(trap)
            assert code != 0 or trap in (
                TrapType.InstrAddrMisaligned,
                TrapType.UmodeSoftInterrupt,
            ), f"{trap.name} returned unexpected zero code"


class TestCauseName:
    """mcause 寄存器值到可读名称的翻译."""

    @pytest.mark.parametrize(
        "mcause_val, expect",
        [
            (2, "IllInstr"),
            (13, "LdPageFault"),
            ((1 << 63) | 7, "MmodeTimerInterrupt"),
            ((1 << 63) | 11, "MmodeExternInterrupt"),
            # 未定义的编号按 "异常#N" 或 "中断#N" 给出
            (14, "异常#14"),
            ((1 << 63) | 12, "中断#12"),
        ],
    )
    def test_cause_name_translates_code(self, mcause_val, expect):
        assert trap_cause_name(mcause_val) == expect
