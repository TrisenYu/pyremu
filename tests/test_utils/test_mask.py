#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""位宽掩码与符号扩展测试.

覆盖: 各位宽掩码常量 / mask* 的截断语义 / sext 系列把有符号取值规范到
``[0, 2**64)`` 区间.
"""

from pyremu.utils.mask import (
    MASK04,
    MASK08,
    MASK12,
    MASK16,
    MASK32,
    MASK64,
    mask04,
    mask08,
    mask12,
    mask16,
    mask32,
    mask64,
    sext,
    sext8,
    sext12,
    sext16,
    sext32,
)

UINT64_MAX = 0xFFFF_FFFF_FFFF_FFFF


class TestMaskConstants:
    """掩码常量由窄到宽逐级合成."""

    def test_constant_values(self):
        """各掩码常量的取值等于对应位宽的全 1."""
        assert MASK04 == 0xF
        assert MASK08 == 0xFF
        assert MASK12 == 0xFFF
        assert MASK16 == 0xFFFF
        assert MASK32 == 0xFFFF_FFFF
        assert MASK64 == UINT64_MAX

    def test_wide_constants_are_built_from_narrow(self):
        """宽掩码按窄掩码平移后补齐低位, 不出现某一段为零的取值."""
        assert MASK16 == (MASK12 << 4) | MASK04
        assert MASK32 == (MASK16 << 16) | MASK16
        assert MASK64 == (MASK32 << 32) | MASK32


class TestMaskTruncation:
    """mask* 只保留低位."""

    def test_value_within_width_is_kept(self):
        """取值未超出位宽时原样返回."""
        assert mask04(0xA) == 0xA
        assert mask08(0xA5) == 0xA5
        assert mask12(0xABC) == 0xABC
        assert mask16(0xABCD) == 0xABCD
        assert mask32(0xABCD_EF01) == 0xABCD_EF01
        assert mask64(0x0123_4567_89AB_CDEF) == 0x0123_4567_89AB_CDEF

    def test_value_beyond_width_is_truncated(self):
        """取值超出位宽时只保留低位."""
        assert mask04(0x1F) == 0xF
        assert mask12(0x1FFF) == 0xFFF
        assert mask04(0xFFFF_FFFF) == 0xF
        assert mask32(0x1_2345_6789) == 0x2345_6789

    def test_negative_value_is_taken_by_bit_pattern(self):
        """负取值按二进制补码取低位, 不保留符号."""
        assert mask04(-1) == 0xF
        assert mask12(-1) == 0xFFF
        assert mask64(-1) == UINT64_MAX


class TestSignExtend:
    """sext 系列把有符号取值规范到 ``[0, 2**64)``."""

    def test_sext8(self):
        """8 位取值扩展后: 最高位为 0 保留原值, 为 1 得到 2**64 减去其模."""
        assert sext8(0x7F) == 0x7F
        assert sext8(0x80) == UINT64_MAX - 0x7F
        assert sext8(0xFF) == UINT64_MAX

    def test_sext16(self):
        """16 位取值扩展后按同一条规则规范."""
        assert sext16(0x7FFF) == 0x7FFF
        assert sext16(0x8000) == UINT64_MAX - 0x7FFF
        assert sext16(0xFFFF) == UINT64_MAX

    def test_sext12_and_sext32_agree_with_the_general_form(self):
        """固定宽度版本与 sext 的一般形式结果一致."""
        for val in (0x0, 0x7FF, 0x800, 0xFFF):
            assert sext12(val) == sext(val, 12)
        for val in (0x0, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF):
            assert sext32(val) == sext(val, 32)

    def test_sext_result_stays_unsigned(self):
        """符号扩展的结果非负, 不出现负的 Python 整数."""
        for val in (0x80, 0xFF, 0x8000, 0xFFFF, 0x8000_0000, 0xFFFF_FFFF):
            assert sext(val, 8 if val < 0x100 else 16 if val < 0x1_0000 else 32) >= 0

    def test_sext64_keeps_bit_pattern(self):
        """64 位的 sext 与 mask64 结果一致: 该位宽下不再有更宽的高位."""
        assert sext(UINT64_MAX, 64) == UINT64_MAX
        assert sext(0x8000_0000_0000_0000, 64) == 0x8000_0000_0000_0000
