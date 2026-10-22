#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""验证无 Rust 动态库时纯 Python 路径仍然正确。"""

from __future__ import annotations

from unittest import mock

import pytest

from pyremu._native import native_available
from pyremu.core.decoder import Hart
from pyremu.core.mem_check_aux import inject_memory_backend
from pyremu.utils.disassem import Opc
from tests.instr_encode import (
    b_type,
    i_type,
    j_type,
    pack_funct,
    pack_src_regs,
    r_type,
    s_type,
    u_type,
)
from tests.test_calc.test_compressed_diff import c1, make_ram


@pytest.fixture
def no_native():
    """关闭加速执行, 使指令执行走纯 Python 路径.

    ``handle_alu`` / ``handle_op_imm`` / ``handle_op32`` / ``handle_op_imm32``
    以公开门控 ``native_available()`` 判断是否调用加速执行库, 故替换该函数返回
    False 即退回纯 Python 实现, 无须触及动态库的装载状态。
    """
    with mock.patch("pyremu.core.decoder.native_available", return_value=False) as gate:
        yield gate


def test_fallback_correctness(no_native):
    """无加速执行库时 32-bit / 压缩指令均正确执行 (复用 diff test 编码辅助)。"""
    ram, rf, wf = make_ram()
    h = Hart(id=0)
    inject_memory_backend(h, rf, wf)

    # ---- 32-bit ----
    h.gprs[5] = 10
    h.exec_instr(i_type(Opc.op_imm.value, 5, 0, 5, 7))  # ADDI x5, 7
    assert h.gprs[5] == 17

    h.pc = 0x80000000
    h.exec_instr(j_type(Opc.jal.value, 1, 40))  # JAL x1, +40
    assert h.gprs[1] == 0x80000004
    assert h.pc == 0x80000028

    h.pc = 0x80000000
    h.exec_instr(j_type(Opc.jal.value, 0, -2))  # JAL 负偏移 (sext21)
    assert h.pc == 0x7FFFFFFE

    h.exec_instr(u_type(Opc.lui.value, 10, 0x12345))  # LUI
    assert h.gprs[10] == 0x12345000

    h.gprs[10] = 0
    h.gprs[11] = 0
    h.pc = 0x80000000
    h.exec_instr(b_type(Opc.br.value, 0, 10, 11, 24))  # BEQ (taken)
    assert h.pc == 0x80000018

    # ---- 压缩 ----
    h.gprs[10] = 5
    h.exec_instr(0x050D)  # C.ADDI x10, 3
    assert h.gprs[10] == 8

    h.pc = 0x80000000
    h.gprs[8] = 0
    c_beqz = c1(
        6,
        1,
        int(  # rs1'=x9, offset=+8
            ((8 >> 8) & 0x1) << 12
            | ((8 >> 3) & 0x3) << 10
            | ((8 >> 7) & 0x3) << 5
            | ((8 >> 1) & 0x3) << 3
            | ((8 >> 5) & 0x1) << 2
        ),
    )
    h.exec_instr(c_beqz)  # C.BEQZ x9, +8 (taken)
    assert h.pc == 0x80000008

    h.gprs[2] = 0x80000
    h.exec_instr(0x717D)  # C.ADDI16SP -16
    assert h.gprs[2] == 0x7FFF0

    # ADDI 与 LUI 经纯 Python 算术路径, 门控必须确实被查询过
    assert no_native.called, "纯 Python 路径未被走到: native_available 门控未被查询"


def test_native_available_true():
    """测试环境中加速执行库应已装载。"""
    assert native_available() is True


# =====================================================================
#  纯 Python 算术路径 — 逐分支覆盖
# =====================================================================

OP_ALU = 0b0110011
OP_IMM = 0b0010011
OP_IMM32 = 0b0011011
OP32 = 0b0111011
OP_LOAD = 0b0000011
OP_STORE = 0b0100011
OP_BRANCH = 0b1100011
OP_SYS = 0b1110011
OP_FENCE = 0b0001111
OP_AMO = 0b0101111
OP_FP_LOAD = 0b0000111
OP_FP_STORE = 0b0100111
OP_FP_OP = 0b1010011
OP_FMADD = 0b1000011
OP_FMSUB = 0b1000111

MASK64 = 0xFFFF_FFFF_FFFF_FFFF
FS_ON = 0b11 << 13


@pytest.fixture
def hart():
    """带物理内存后端的 hart."""
    _, rf, wf = make_ram()
    h = Hart(id=0)
    inject_memory_backend(h, rf, wf)
    return h


def _assert_ill_instr(h: Hart, instr: int) -> None:
    """断言 *instr* 触发非法指令陷态."""
    h.csrs["mtvec"].val = 0x1000
    h.pc = 0x2000
    assert h.exec_instr(instr) == 0
    assert h.pc == 0x1000
    assert h.csrs["mcause"].val == 2


class TestPurePythonAlu:
    """handle_alu 的纯 Python 分支: 全部 funct3 与 funct7 组合."""

    @pytest.mark.parametrize(
        "case",
        [
            (5, 7, 0b000, 0, 12),
            (6, 7, 0b000, 1, 42),
            (10, 3, 0b000, 0x20, 7),
            (1, 4, 0b001, 0, 16),
            (MASK64, 1, 0b001, 1, MASK64),
            (0x8000_0000_0000_0001, 1, 0b001, 0x30, 3),
            (MASK64, 1, 0b010, 0, 1),
            (MASK64, 1, 0b010, 1, MASK64),
            (1, MASK64, 0b011, 0, 1),
            (1 << 63, 2, 0b011, 1, 1),
            (0xF0, 0x0F, 0b100, 0, 0xFF),
            (7, 2, 0b100, 1, 3),
            (0x10, 2, 0b101, 0, 4),
            (7, 2, 0b101, 1, 3),
            (MASK64 - 7, 1, 0b101, 0x20, MASK64 - 3),
            (1, 1, 0b101, 0x30, 1 << 63),
            (0x0F, 0xF0, 0b110, 0, 0xFF),
            (7, 2, 0b110, 1, 1),
            (0xFF, 0x0F, 0b111, 0, 0x0F),
            (7, 2, 0b111, 1, 1),
            # Zbb 子指令: 按位同或, 有符号最小与最大, 或非, 与非, 无符号最大
            (0xF0, 0x0F, 0b100, 0x20, MASK64 ^ 0xFF),
            (MASK64, 1, 0b100, 0x05, MASK64),
            (0xF0, 0x0F, 0b110, 0x20, MASK64 ^ 0x0F),
            (0, 7, 0b110, 0x05, 7),
            (MASK64, 1, 0b111, 0x20, MASK64 - 1),
            (1, MASK64, 0b111, 0x05, MASK64),
        ],
    )
    def test_alu_branch(self, no_native, hart, case):
        v1, v2, funct3, funct7, expect = case
        hart.gprs[1] = v1
        hart.gprs[2] = v2
        hart.exec_instr(r_type(OP_ALU, 3, pack_funct(funct7, funct3), pack_src_regs(1, 2)))
        assert hart.gprs[3] == expect
        assert no_native.called

    @pytest.mark.parametrize(
        "case",
        [
            (7, 0b100, 1, MASK64),  # DIV / 0
            (7, 0b101, 1, MASK64),  # DIVU / 0
            (7, 0b110, 1, 7),  # REM / 0
            (7, 0b111, 1, 7),  # REMU / 0
        ],
    )
    def test_alu_divide_by_zero(self, no_native, hart, case):
        v1, funct3, funct7, expect = case
        hart.gprs[1] = v1
        hart.gprs[2] = 0
        hart.exec_instr(r_type(OP_ALU, 3, pack_funct(funct7, funct3), pack_src_regs(1, 2)))
        assert hart.gprs[3] == expect

    @pytest.mark.parametrize("funct3", [0b000, 0b001, 0b011, 0b101])
    def test_alu_invalid_funct7_traps(self, no_native, hart, funct3):
        _assert_ill_instr(
            hart, r_type(OP_ALU, 3, pack_funct(0x10, funct3), pack_src_regs(1, 2))
        )

    def test_alu_invalid_zbb_traps(self, no_native, hart):
        """funct7 属于 Zbb 但 funct3 不对应任何子指令."""
        _assert_ill_instr(
            hart, r_type(OP_ALU, 3, pack_funct(0x05, 0b010), pack_src_regs(1, 2))
        )


class TestPurePyOpImm:
    """handle_op_imm 的纯 Python 分支."""

    @pytest.mark.parametrize(
        "case",
        [
            (5, 0b000, 7, 12),
            (1, 0b001, 4, 16),
            (MASK64 - 1, 0b010, 0xFFF, 1),  # SLTI, imm=-1, v1=-2
            (0, 0b011, 1, 1),
            (0xF0, 0b100, 0x0F, 0xFF),
            (0x10, 0b101, 4, 1),
            (MASK64 - 7, 0b101, (0x20 << 5) | 1, MASK64 - 3),  # SRAI
            (0x0F, 0b110, 0xF0, 0xFF),
            (0xFF, 0b111, 0x0F, 0x0F),
        ],
    )
    def test_op_imm_branch(self, no_native, hart, case):
        v1, funct3, imm12, expect = case
        hart.gprs[1] = v1
        hart.exec_instr(i_type(OP_IMM, 3, funct3, 1, imm12))
        assert hart.gprs[3] == expect

    @pytest.mark.parametrize(
        "case",
        [
            (0, 0b001, 0x600, 64),  # clz
            (0x100, 0b001, 0x601, 8),  # ctz
            (0xFF, 0b001, 0x602, 8),  # cpop
            (0x80, 0b001, 0x604, MASK64 - 0x7F),  # sext.b
            (0x8000, 0b001, 0x605, MASK64 - 0x7FFF),  # sext.h
            (1, 0b101, 0x601, 1 << 63),  # rori
            (0x0100, 0b101, 0x287, 0xFF00),  # orc.b
            (0x0123_4567_89AB_CDEF, 0b101, 0x6B8, 0xEFCD_AB89_6745_2301),  # rev8
        ],
    )
    def test_op_imm_zbb(self, no_native, hart, case):
        v1, funct3, imm12, expect = case
        hart.gprs[1] = v1
        hart.exec_instr(i_type(OP_IMM, 3, funct3, 1, imm12))
        assert hart.gprs[3] == expect

    def test_op_imm_invalid_slli_funct6_traps(self, no_native, hart):
        _assert_ill_instr(hart, i_type(OP_IMM, 3, 0b001, 1, 0x200))

    def test_op_imm_invalid_srli_funct6_traps(self, no_native, hart):
        _assert_ill_instr(hart, i_type(OP_IMM, 3, 0b101, 1, 0x220))

    def test_op_imm_invalid_zbb_shamt_traps(self, no_native, hart):
        _assert_ill_instr(hart, i_type(OP_IMM, 3, 0b001, 1, 0x603))


class TestPurePythonOp32:
    """handle_op32 的纯 Python 分支."""

    @pytest.mark.parametrize(
        "case",
        [
            (0x7FFF_FFFF, 1, 0b000, 0, 0xFFFF_FFFF_8000_0000),
            (2, 3, 0b000, 1, 6),
            (5, 7, 0b000, 0x20, 0xFFFF_FFFF_FFFF_FFFE),
            (1, 4, 0b001, 0, 16),
            (0x8000_0000, 2, 0b001, 1, MASK64),
            (0x8000_0001, 1, 0b001, 0x30, 3),
            (MASK64, 1, 0b010, 0, 1),
            (MASK64, 1, 0b010, 1, MASK64),
            (1, 0xFFFF_FFFF, 0b011, 0, 1),
            (0x8000_0000, 2, 0b011, 1, 1),
            (0xFF, 0x0F, 0b100, 0, 0xF0),
            (7, 2, 0b100, 1, 3),
            (0x8000_0000, 4, 0b101, 0, 0x0800_0000),
            (7, 2, 0b101, 1, 3),
            (0x8000_0000, 1, 0b101, 0x20, 0xFFFF_FFFF_C000_0000),
            (1, 1, 0b101, 0x30, 0xFFFF_FFFF_8000_0000),
            (0x0F, 0xF0, 0b110, 0, 0xFF),
            (7, 2, 0b110, 1, 1),
            (0xFF, 0x0F, 0b111, 0, 0x0F),
            (7, 2, 0b111, 1, 1),
        ],
    )
    def test_op32_branch(self, no_native, hart, case):
        v1, v2, funct3, funct7, expect = case
        hart.gprs[1] = v1
        hart.gprs[2] = v2
        hart.exec_instr(r_type(OP32, 3, pack_funct(funct7, funct3), pack_src_regs(1, 2)))
        assert hart.gprs[3] == expect

    @pytest.mark.parametrize(
        "funct3", [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111]
    )
    def test_op32_invalid_funct7_traps(self, no_native, hart, funct3):
        _assert_ill_instr(hart, r_type(OP32, 3, pack_funct(0x10, funct3), pack_src_regs(1, 2)))


class TestPurePyOpImm32:
    """handle_op_imm32 的纯 Python 分支."""

    @pytest.mark.parametrize(
        "case",
        [
            (1, 0b000, 1, 2),
            (1, 0b001, 4, 16),
            (0, 0b001, 0x600, 32),  # clzw
            (0x100, 0b001, 0x601, 8),  # ctzw
            (0xFF, 0b001, 0x602, 8),  # cpopw
            (0x10, 0b101, 2, 4),
            (0x8000_0000, 0b101, (0x20 << 5) | 1, 0xFFFF_FFFF_C000_0000),
            (0x1234_5678, 0b101, 0x604, 0xFFFF_FFFF_8123_4567),  # roriw
        ],
    )
    def test_op_imm32_branch(self, no_native, hart, case):
        v1, funct3, imm12, expect = case
        hart.gprs[1] = v1
        hart.exec_instr(i_type(OP_IMM32, 3, funct3, 1, imm12))
        assert hart.gprs[3] == expect

    def test_op_imm32_invalid_funct7_traps(self, no_native, hart):
        _assert_ill_instr(hart, i_type(OP_IMM32, 3, 0b001, 1, 0x200))

    def test_op_imm32_invalid_srliw_funct7_traps(self, no_native, hart):
        """funct3=101 的立即数形式只接受 funct7 为 0x00 / 0x20 / 0x30."""
        _assert_ill_instr(hart, i_type(OP_IMM32, 3, 0b101, 1, 0x200))

    def test_op_imm32_invalid_funct3_traps(self, no_native, hart):
        _assert_ill_instr(hart, i_type(OP_IMM32, 3, 0b010, 1, 0))


class TestNativePathEncodingTraps:
    """加速执行库返回无效编码后, 算术处理函数自身抛出的非法指令."""

    def test_alu_invalid_funct7_traps(self, hart):
        """非 Zbb 的无效 funct7 不经 _zbb_alu, 直接抛 ValueError."""
        _assert_ill_instr(
            hart, r_type(OP_ALU, 3, pack_funct(0x10, 0b000), pack_src_regs(1, 2))
        )

    def test_alu_invalid_zbb_funct3_traps(self, hart):
        """funct7 属于 Zbb 但 funct3 不对应任何子指令."""
        _assert_ill_instr(
            hart, r_type(OP_ALU, 3, pack_funct(0x05, 0b010), pack_src_regs(1, 2))
        )

    @pytest.mark.parametrize(
        "funct3, funct7",
        [
            (0b000, 0x10),  # funct7 不属于任何 32 位字运算
            (0b010, 0x30),  # funct7=0x30 只适用于 001 / 101
        ],
    )
    def test_op32_invalid_encoding_traps(self, hart, funct3, funct7):
        _assert_ill_instr(
            hart, r_type(OP32, 3, pack_funct(funct7, funct3), pack_src_regs(1, 2))
        )

    def test_op_imm32_invalid_funct7_traps(self, hart):
        _assert_ill_instr(hart, i_type(OP_IMM32, 3, 0b001, 1, 0x200))

    def test_op_imm32_invalid_zbb_funct3_traps(self, hart):
        """funct7=0x30 且 funct3 不对应任何子指令时 _zbb_imm32 抛出非法指令."""
        _assert_ill_instr(hart, i_type(OP_IMM32, 3, 0b010, 1, 0x600))


class TestBranchAndMemoryEncodings:
    """分支与访存的边界编码."""

    @pytest.mark.parametrize(
        "funct3, v1, v2, taken",
        [
            (0b110, 1, 2, True),  # BLTU
            (0b110, 2, 1, False),
            (0b111, 2, 1, True),  # BGEU
            (0b111, 1, 2, False),
        ],
    )
    def test_unsigned_branches(self, hart, funct3, v1, v2, taken):
        hart.gprs[1] = v1
        hart.gprs[2] = v2
        hart.pc = 0x2000
        hart.exec_instr(b_type(OP_BRANCH, funct3, 1, 2, 8))
        assert hart.pc == (0x2008 if taken else 0x2000)

    def test_branch_invalid_funct3_advances(self, hart):
        hart.pc = 0x2000
        assert hart.exec_instr(b_type(OP_BRANCH, 0b010, 1, 2, 8)) == 4
        assert hart.pc == 0x2000

    @pytest.mark.parametrize(
        "funct3, raw, expect",
        [
            (0b000, bytes([0x80, 0, 0, 0, 0, 0, 0, 0]), MASK64 - 0x7F),  # LB
            (0b100, bytes([0x80, 0, 0, 0, 0, 0, 0, 0]), 0x80),  # LBU
            (0b001, bytes([0x00, 0x80, 0, 0, 0, 0, 0, 0]), MASK64 - 0x7FFF),  # LH
            (0b101, bytes([0x00, 0x80, 0, 0, 0, 0, 0, 0]), 0x8000),  # LHU
        ],
    )
    def test_load_sign_extension(self, hart, funct3, raw, expect):
        hart._mem_write_phy(0x100, raw)
        hart.gprs[1] = 0x100
        hart.exec_instr(i_type(OP_LOAD, 3, funct3, 1, 0))
        assert hart.gprs[3] == expect

    def test_load_invalid_funct3_traps(self, hart):
        _assert_ill_instr(hart, i_type(OP_LOAD, 3, 0b111, 1, 0))

    @pytest.mark.parametrize(
        "funct3, val, expect",
        [
            (0b001, 0x1234, bytes([0x34, 0x12])),
            (0b010, 0x1234_5678, bytes([0x78, 0x56, 0x34, 0x12])),
            (
                0b011,
                0x0123_4567_89AB_CDEF,
                bytes([0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01]),
            ),
        ],
    )
    def test_store_widths(self, hart, funct3, val, expect):
        hart.gprs[1] = 0x100
        hart.gprs[2] = val
        hart.exec_instr(s_type(OP_STORE, funct3, 1, 2, 0))
        assert hart._mem_read_phy(0x100, len(expect)) == expect

    def test_store_invalid_funct3_traps(self, hart):
        _assert_ill_instr(hart, s_type(OP_STORE, 0b100, 1, 2, 0))

    def test_fence_and_fence_i(self, hart):
        hart.pc = 0x2000
        assert hart.exec_instr(0b0001111) == 4  # FENCE
        assert hart.exec_instr(0b0001111 | (1 << 12)) == 4  # FENCE.I

    def test_fence_invalid_funct3_traps(self, hart):
        _assert_ill_instr(hart, 0b0001111 | (0b010 << 12))

    def test_amo_invalid_encoding_traps(self, hart):
        _assert_ill_instr(hart, r_type(OP_AMO, 3, pack_funct(0, 0b100), pack_src_regs(1, 2)))

    def test_sys_invalid_funct3_traps(self, hart):
        _assert_ill_instr(hart, r_type(OP_SYS, 3, pack_funct(0, 0b100), pack_src_regs(1, 2)))


class TestCsrImmediateForms:
    """CSRRWI / CSRRSI / CSRRCI."""

    MSCRATCH = 0x340

    def _csr_instr(self, funct3: int, rd: int, csr: int, uimm: int) -> int:
        return r_type(OP_SYS, rd, pack_funct(csr >> 5, funct3), pack_src_regs(uimm, 0)) | (
            (csr & 0x1F) << 20
        )

    def test_csrrwi(self, hart):
        hart.csrs["mscratch"].val = 7
        hart.exec_instr(self._csr_instr(0b101, 3, self.MSCRATCH, 5))
        assert hart.gprs[3] == 7
        assert hart.csrs["mscratch"].val == 5

    def test_csrrsi_with_zero_uimm_reads_only(self, hart):
        hart.csrs["mscratch"].val = 7
        hart.exec_instr(self._csr_instr(0b110, 3, self.MSCRATCH, 0))
        assert hart.gprs[3] == 7
        assert hart.csrs["mscratch"].val == 7

    def test_csrrsi_or_immediate(self, hart):
        hart.csrs["mscratch"].val = 0b100
        hart.exec_instr(self._csr_instr(0b110, 3, self.MSCRATCH, 0b011))
        assert hart.csrs["mscratch"].val == 0b111

    def test_csrrci_clear_immediate(self, hart):
        hart.csrs["mscratch"].val = 0b111
        hart.exec_instr(self._csr_instr(0b111, 3, self.MSCRATCH, 0b011))
        assert hart.csrs["mscratch"].val == 0b100


class TestFpDisabled:
    """mstatus.FS=Off 时全部浮点指令触发非法指令陷态."""

    def test_fp_load_disabled(self, hart):
        _assert_ill_instr(hart, i_type(OP_FP_LOAD, 3, 0b010, 1, 0))

    def test_fp_load_invalid_funct3(self, hart):
        hart.mstatus_val |= FS_ON
        _assert_ill_instr(hart, i_type(OP_FP_LOAD, 3, 0b100, 1, 0))

    def test_fp_store_disabled(self, hart):
        _assert_ill_instr(hart, s_type(OP_FP_STORE, 0b010, 1, 2, 0))

    def test_fp_store_invalid_funct3(self, hart):
        hart.mstatus_val |= FS_ON
        _assert_ill_instr(hart, s_type(OP_FP_STORE, 0b100, 1, 2, 0))

    def test_fp_op_disabled(self, hart):
        _assert_ill_instr(hart, r_type(OP_FP_OP, 3, pack_funct(0, 0b000), pack_src_regs(1, 2)))

    def test_fp_op_invalid_encoding(self, hart):
        hart.mstatus_val |= FS_ON
        _assert_ill_instr(
            hart, r_type(OP_FP_OP, 3, pack_funct(0x7C, 0b000), pack_src_regs(1, 2))
        )

    def test_fp_fma_disabled(self, hart):
        _assert_ill_instr(hart, r_type(OP_FMADD, 3, pack_funct(0, 0b000), pack_src_regs(1, 2)))

    def test_fp_fma_invalid_fmt(self, hart):
        hart.mstatus_val |= FS_ON
        _assert_ill_instr(
            hart, r_type(OP_FMSUB, 3, pack_funct(0b10, 0b000), pack_src_regs(1, 2))
        )
