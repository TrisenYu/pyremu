#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""验证 pyremu._native 各纯 Python 回退分支的取值符合 RISC-V 语义。

回退分支以模块级 ``_lib`` 为门控, 该模块没有公开的开关, 故用 ``mock.patch`` 把它
置为 None。期望值一律取自 RISC-V 规范与原生实现, 不取自回退自身的当前输出, 使每个
用例在缺少动态库时同样能判定回退是否正确。
"""

from __future__ import annotations

from unittest import mock

import pytest

from pyremu import _native
from pyremu._native import native_available

MASK64 = 0xFFFF_FFFF_FFFF_FFFF
PTE_U = 1 << 4


@pytest.fixture
def pure_python():
    """使 pyremu._native 的各个入口走纯 Python 回退分支。"""
    with mock.patch("pyremu._native._lib", None):
        yield


class TestDivisionPrecision:
    """商的绝对值达到 2**53 之后, 商与余数不得因浮点舍入而改变。"""

    def test_div_beyond_53_bit_significand(self, pure_python):
        """被除数取 i64 上界、除数为 3: 真商 3074457345618258602。

        改前用 ``int(a / b)``, 双精度把商舍入到 3074457345618258432。
        """
        got = _native.exec_alu_op(0b100, 1, 0x7FFF_FFFF_FFFF_FFFF, 3).value
        assert got == 3074457345618258602 and got != 3074457345618258432

    def test_rem_beyond_53_bit_significand(self, pure_python):
        """被除数取 2**60、除数为 7: 真余数 1。

        改前余数由被舍入的商算出: 商少 9, 误差乘 7 使余数成为 64。
        """
        got = _native.exec_alu_op(0b110, 1, 1 << 60, 7).value
        assert got == 1 and got != 64

    def test_quotient_truncates_toward_zero(self, pure_python):
        """商的取整方向是向零: DIV(-7, 2) 为 -3, REM(-7, 2) 为 -1。"""
        assert _native.exec_alu_op(0b100, 1, MASK64 - 6, 2).value == MASK64 - 2
        assert _native.exec_alu_op(0b110, 1, MASK64 - 6, 2).value == MASK64

    def test_div_zero_returns_all_ones(self, pure_python):
        """除数为零: DIV 与 DIVU 返回全 1, 不抛出异常。"""
        assert _native.exec_alu_op(0b100, 1, 5, 0).value == MASK64
        assert _native.exec_alu_op(0b101, 1, 5, 0).value == MASK64

    def test_rem_zero_returns_dividend(self, pure_python):
        """除数为零: REM 与 REMU 返回被除数。"""
        assert _native.exec_alu_op(0b110, 1, 5, 0).value == 5
        assert _native.exec_alu_op(0b111, 1, 5, 0).value == 5

    def test_division_overflow_wraps(self, pure_python):
        """i64 最小值除以 -1: 商为 i64 最小值, 余数为 0。"""
        assert _native.exec_alu_op(0b100, 1, 1 << 63, MASK64).value == 1 << 63
        assert _native.exec_alu_op(0b110, 1, 1 << 63, MASK64).value == 0


class TestWordResultSignExtension:
    """32 位运算的结果按第 31 位符号扩展到 64 位。"""

    def test_addw_sign_extends(self, pure_python):
        """ADDW(0x7FFF_FFFF, 1) 为 0xFFFF_FFFF_8000_0000 (改前为 0x8000_0000)。"""
        assert _native.exec_op32(0b000, 0, 0x7FFF_FFFF, 1).value == 0xFFFF_FFFF_8000_0000

    def test_addw_positive_result_unchanged(self, pure_python):
        """结果为非负时不扩展出高位: ADDW(1, 1) 为 2。"""
        assert _native.exec_op32(0b000, 0, 1, 1).value == 2

    def test_subw_zero_minus_one(self, pure_python):
        """SUBW(0, 1) 为 -1 的 64 位表示。"""
        assert _native.exec_op32(0b000, 0x20, 0, 1).value == MASK64

    def test_sllw_sign_extends(self, pure_python):
        """SLLW(1, 31) 的 32 位结果为 0x8000_0000, 扩展后为 0xFFFF_FFFF_8000_0000。"""
        assert _native.exec_op32(0b001, 0, 1, 31).value == 0xFFFF_FFFF_8000_0000

    def test_srlw_does_not_introduce_high_bits(self, pure_python):
        """SRLW 是逻辑右移: 结果为 0x0800_0000, 第 31 位为 0。"""
        assert _native.exec_op32(0b101, 0, 0x8000_0000, 4).value == 0x0800_0000

    def test_sraw_shifts_negative_value(self, pure_python):
        """SRAW(0x8000_0000, 4) 为 0xFFFF_FFFF_F800_0000 (改前按无符号移位得 0x0800_0000)。"""
        assert _native.exec_op32(0b101, 0x20, 0x8000_0000, 4).value == 0xFFFF_FFFF_F800_0000

    def test_divw_negative_dividend(self, pure_python):
        """DIVW(0xFFFF_FFFE, 2) 为 -1 (改前把被除数当无符号数, 得 0x7FFF_FFFF)。"""
        assert _native.exec_op32(0b100, 1, 0xFFFF_FFFE, 2).value == MASK64

    def test_divw_overflow_wraps(self, pure_python):
        """i32 最小值除以 -1: 商为 i32 最小值, 扩展后为 0xFFFF_FFFF_8000_0000。"""
        got = _native.exec_op32(0b100, 1, 0x8000_0000, MASK64).value
        assert got == 0xFFFF_FFFF_8000_0000

    def test_divuw_reinterprets_quotient(self, pure_python):
        """DIVUW 的商按有符号数扩展: DIVUW(0xFFFF_FFFF, 1) 为全 1。"""
        assert _native.exec_op32(0b101, 1, 0xFFFF_FFFF, 1).value == MASK64

    def test_remw_values(self, pure_python):
        """REMW 按有符号数取余, 除数为零时返回符号扩展后的被除数。"""
        assert _native.exec_op32(0b110, 1, 0xFFFF_FFFE, 2).value == 0
        assert _native.exec_op32(0b110, 1, 5, 0).value == 5

    def test_remuw_values(self, pure_python):
        """REMUW 按无符号数取余, 除数为零时返回符号扩展后的被除数。"""
        assert _native.exec_op32(0b111, 1, 0xFFFF_FFFE, 2).value == 0
        assert _native.exec_op32(0b111, 1, 5, 0).value == 5

    def test_addiw_sign_extends(self, pure_python):
        """ADDIW(0x7FFF_FFFF, 1) 为 0xFFFF_FFFF_8000_0000。"""
        assert _native.exec_op_imm32(0b000, 0, 0x7FFF_FFFF, 1).value == 0xFFFF_FFFF_8000_0000

    def test_sraiw_shifts_negative_value(self, pure_python):
        """SRAIW(0x8000_0000, 4) 为 0xFFFF_FFFF_F800_0000。"""
        got = _native.exec_op_imm32(0b101, 0x20, 0x8000_0000, 4).value
        assert got == 0xFFFF_FFFF_F800_0000

    def test_srliw_does_not_introduce_high_bits(self, pure_python):
        """SRLIW(0x8000_0000, 4) 为 0x0800_0000。"""
        assert _native.exec_op_imm32(0b101, 0, 0x8000_0000, 4).value == 0x0800_0000


class TestSltiImmediate:
    """SLTI 把立即数按有符号数比较, SLTIU 按无符号数比较。"""

    def test_slti_with_negative_immediate(self, pure_python):
        """SLTI(0, -1) 为 0: 0 不小于 -1 (改前把立即数当无符号数, 取值为 1)。"""
        assert _native.exec_op_imm(0b010, 0, 0, MASK64).value == 0

    def test_slti_with_positive_immediate(self, pure_python):
        """SLTI 的正立即数不受影响: SLTI(5, 7) 为 1, SLTI(5, 3) 为 0。"""
        assert _native.exec_op_imm(0b010, 0, 5, 7).value == 1
        assert _native.exec_op_imm(0b010, 0, 5, 3).value == 0

    def test_slti_with_negative_operand(self, pure_python):
        """SLTI(-1, 0) 为 1: 被除数按有符号数解释。"""
        assert _native.exec_op_imm(0b010, 0, MASK64, 0).value == 1

    def test_sltiu_compares_unsigned(self, pure_python):
        """SLTIU 的对照组: SLTIU(0, -1) 为 1, 因为 0 小于 2**64-1。"""
        assert _native.exec_op_imm(0b011, 0, 0, MASK64).value == 1


class TestPtePermBits:
    """PTE 权限字段取 bits[4:1], 即 R|W|X|U。"""

    def test_u_bit_is_reported(self, pure_python):
        """raw 置 U 位时 perm 必须含 U 位 (改前掩码为 0xF, U 位恒缺)。"""
        perm = _native.pte_parse(0x1F).perm
        assert perm == 0x1E
        assert perm & PTE_U

    def test_perm_excludes_valid_bit(self, pure_python):
        """raw=0xCF 的 perm 为 0x0E: V 位不进 perm (改前为 0x0F)。"""
        assert _native.pte_parse(0xCF).perm == 0x0E

    def test_rwx_bits_are_preserved(self, pure_python):
        """R/W/X 三位逐位保留, 与 raw 的 bits[3:1] 相同。"""
        for raw in (0x0F, 0x1B, 0x1E, 0xCF, 0xDF, 0xEF):
            assert _native.pte_parse(raw).perm & 0b1110 == raw & 0b1110


class TestCompressedDecode:
    """压缩指令的立即数位段与 RISC-V 编码一致。"""

    def test_lwsp_offset_bits(self, pure_python):
        """C.LWSP 的 uimm[7:6] 取 bits[3:2]、uimm[4:2] 取 bits[6:4]。

        0x4522 的偏移为 8 (改前取 bits[5:4] 与 bits[4:2], 得 64)。
        """
        assert _native.decode_compressed(0x4522).imm == 8
        assert _native.decode_compressed(0x5522).imm == 40

    def test_ldsp_and_fldsp_share_uimm(self, pure_python):
        """C.FLDSP 与 C.LDSP 共用 uimm[5|4:3|8:6]: 0x2006 与 0x6006 均为 64。"""
        assert _native.decode_compressed(0x2006).imm == 64
        assert _native.decode_compressed(0x6006).imm == 64

    def test_c0_floating_load_store_uimm(self, pure_python):
        """C.FLD 与 C.FSD 与 C.LD / C.SD 共用 uimm[5:3|6|7]。

        0x2020 (funct3=1) 与 0xA020 (funct3=5) 均为 64 (改前返回 0)。
        """
        assert _native.decode_compressed(0x2020).imm == 64
        assert _native.decode_compressed(0xA020).imm == 64

    def test_fsdsp_uimm2(self, pure_python):
        """C.FSDSP 与 C.SDSP 共用 uimm[5:3|8:6]: 0xA082 为 64 (改前返回 0)。"""
        assert _native.decode_compressed(0xA082).imm2 == 64
        assert _native.decode_compressed(0xE082).imm2 == 64


@pytest.mark.skipif(not native_available(), reason="需要加速执行库作为对照")
class TestFallbackMatchesNative:
    """回退分支与加速执行库在整段语料上逐位一致。"""

    def test_compressed_decode_over_all_halfwords(self):
        """全部 65536 个半字的字段逐一比对。"""
        native = {h: _native.decode_compressed(h) for h in range(0x10000)}
        with mock.patch("pyremu._native._lib", None):
            fallback = {h: _native.decode_compressed(h) for h in range(0x10000)}
        fields = [name for name, _ in _native.CompressedFields._fields_]
        diff = [
            h
            for h in range(0x10000)
            if [getattr(native[h], k) for k in fields]
            != [getattr(fallback[h], k) for k in fields]
        ]
        assert diff == []

    def test_word_alu_over_edge_operands(self):
        """32 位与 32 位立即数两个入口在边界操作数上逐一比对。"""
        values = (0, 1, 2, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF, 1 << 53, MASK64 - 1, MASK64)
        for funct3 in range(8):
            for funct7 in (0, 1, 0x20):
                for v1 in values:
                    for v2 in values:
                        native = _native.exec_op32(funct3, funct7, v1, v2).value
                        with mock.patch("pyremu._native._lib", None):
                            fallback = _native.exec_op32(funct3, funct7, v1, v2).value
                        assert native == fallback, (funct3, funct7, v1, v2)

    def test_div_rem_over_edge_operands(self):
        """除法与取余在边界操作数上逐一比对。"""
        dividends = (0, 1, 7, 1 << 53, (1 << 53) + 1, (1 << 63), MASK64 - 1, MASK64)
        divisors = (0, 1, 2, 3, 7, 0x8000_0000, MASK64 - 1, MASK64)
        for funct3 in (0b100, 0b101, 0b110, 0b111):
            for funct7 in (1, 0x20):
                for a in dividends:
                    for b in divisors:
                        native = _native.exec_alu_op(funct3, funct7, a, b).value
                        with mock.patch("pyremu._native._lib", None):
                            fallback = _native.exec_alu_op(funct3, funct7, a, b).value
                        assert native == fallback, (funct3, funct7, a, b)

    def test_pte_parse_over_perm_combinations(self):
        """页表项的权限组合逐一比对, 含 U 位与 V 位的全部组合。"""
        for low in range(0x100):
            raw = ((0x80200 << 10) | low) & MASK64
            native = _native.pte_parse(raw)
            with mock.patch("pyremu._native._lib", None):
                fallback = _native.pte_parse(raw)
            fields = [name for name, _ in _native.PteFields._fields_]
            assert [getattr(native, k) for k in fields] == [
                getattr(fallback, k) for k in fields
            ], hex(raw)
