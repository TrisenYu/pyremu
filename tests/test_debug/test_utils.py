#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""测试 pyremu.debug.utils."""

import pytest

from pyremu.debug.utils import (
    EXC_NAMES,
    IRQ_NAMES,
    MAX_INSTR_COUNT,
    check_rv64_addr,
    fmt_instr_count,
    fmt_size,
    group_order,
    hex_addr,
    ip_bits,
    trap_cause_name,
)


class TestHexAddr:
    def test_basic(self):
        assert hex_addr(0) == "0x0000000000000000"
        assert hex_addr(0x80000000) == "0x0000000080000000"

    def test_mask_upper_bits(self):
        """Python int 无位宽限制, 超出 64-bit 的部分应被掩码清除."""
        assert hex_addr(0x1_0000_0000_0000_0001) == "0x0000000000000001"

    def test_negative_value(self):
        """负值在补码下转换为 64-bit 无符号."""
        assert hex_addr(-1) == "0xffffffffffffffff"


class TestFmtSize:
    def test_bytes(self):
        assert fmt_size(0) == "0 B"
        assert fmt_size(512) == "512 B"

    def test_kb(self):
        assert fmt_size(2048) == "2.0 KB"

    def test_mb(self):
        assert fmt_size(2 * 1024 * 1024) == "2.0 MB"

    def test_gb(self):
        assert fmt_size(3 * 1024**3) == "3.0 GB"


class TestFmtInstrCount:
    def test_small(self):
        assert fmt_instr_count(0) == "0"
        assert fmt_instr_count(999) == "999"

    def test_k(self):
        assert fmt_instr_count(1000) == "1.000K"
        assert fmt_instr_count(999999) == "999.999K"

    def test_m(self):
        assert fmt_instr_count(1_000_000) == "1.000000M"

    def test_b(self):
        assert fmt_instr_count(1_000_000_000) == "1.0000000B"


class TestTrapCauseName:
    def test_known(self):
        name = trap_cause_name(0x8000_0000_0000_0003)  # MSIP
        assert len(name) > 0

    def test_unknown(self):
        name = trap_cause_name(0xDEAD)
        # 未知 cause code 格式: "异常#57005" (中文"异常" + 编号)
        assert "#" in name


class TestCheckRv64Addr:
    def test_valid(self):
        assert check_rv64_addr(0) is True
        assert check_rv64_addr(0xFFFFFFFFFFFFFFFF) is True
        assert check_rv64_addr(0x80000000) is True

    def test_invalid(self):
        assert check_rv64_addr(-1) is False
        assert check_rv64_addr(1 << 64) is False

    @pytest.mark.parametrize("v", [2**64, -(2**63)])
    def test_boundary_invalid(self, v):
        assert check_rv64_addr(v) is False


class TestIpBits:
    def test_returns_list(self):
        bits = ip_bits()
        assert len(bits) == 9
        assert bits[0] == ("USIP", 0, "U 模式软件中断挂起")
        assert bits[-1] == ("MEIP", 11, "M 模式外部中断挂起")


class TestExcIrqNames:
    def test_exc_names_coverage(self):
        assert EXC_NAMES[2] == "IllInstr"
        assert EXC_NAMES[12] == "InstrPageFault"

    def test_irq_names_coverage(self):
        assert IRQ_NAMES[5] == "STIP (S 模式定时器中断)"


class TestGroupOrder:
    def test_alpha_first(self):
        assert group_order("a") < group_order("1")
        assert group_order("z") < group_order("0")

    def test_digit_before_underscore(self):
        assert group_order("0") < group_order("_")

    def test_underscore_before_dot(self):
        assert group_order("_") < group_order(".")


class TestMaxInstrCount:
    def test_value(self):
        assert MAX_INSTR_COUNT == 100_000
