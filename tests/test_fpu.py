#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""F/D 浮点扩展测试 — 经 decoder 路径 (native softfloat 计算).

覆盖 OP-FP (算术/转换/比较/符号/分类/移动)、FMA、FP load/store。
计算委托给 native softfloat-pure; 本测试验证 decoder 的寄存器/内存路由、
NaN-boxing、fcsr 累积与 mstatus.FS 门控。
"""

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.mem_check_aux import inject_memory_backend

# 单/双精度位模式常量
BOX = 0xFFFF_FFFF_0000_0000  # NaN-boxing 掩码
S1_0 = 0x3F80_0000  # 1.0f
S2_0 = 0x4000_0000  # 2.0f
S3_0 = 0x4040_0000  # 3.0f
S4_0 = 0x4080_0000  # 4.0f
S6_0 = 0x40C0_0000  # 6.0f
S7_0 = 0x40E0_0000  # 7.0f
D1_0 = 0x3FF0_0000_0000_0000  # 1.0
D2_0 = 0x4000_0000_0000_0000  # 2.0
D3_0 = 0x4008_0000_0000_0000  # 3.0

MSTATUS_FS_INIT = 1 << 13  # FS = Initial


def _make_ram():
    ram = bytearray(2 * 1024 * 1024)

    def read_fn(addr, size):
        return bytes(ram[addr : addr + size])

    def write_fn(addr, data):
        for i, b in enumerate(data):
            ram[addr + i] = b

    return ram, read_fn, write_fn


def _boxf(v32: int) -> int:
    return BOX | v32


def _op_fp(funct7: int, funct3: int, rs2: int, rs1: int, rd: int) -> int:
    """构造 OP-FP 指令字 (opcode 0x53)."""
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0b1010011


def _fma(opcode: int, fmt: int, rs3: int, rs2: int, rs1: int, rd: int, rm: int = 0) -> int:
    """构造 FMA 指令字."""
    return (
        (rs3 << 27) | (fmt << 25) | (rs2 << 20) | (rs1 << 15)
        | (rm << 12) | (rd << 7) | opcode
    )


@pytest.fixture
def hart() -> Hart:
    h = Hart(id=0)
    ram, rf, wf = _make_ram()
    inject_memory_backend(h, rf, wf)
    # 启用浮点单元 (FS = Initial)。
    h.mstatus_val = h.mstatus_val | MSTATUS_FS_INIT
    return h


# ============================================================
#  单精度算术
# ============================================================


class TestSinglePrecisionArith:
    def test_fadd_s(self, hart):
        hart._fpr_bits[1] = _boxf(S1_0)
        hart._fpr_bits[2] = _boxf(S2_0)
        hart.exec_instr(_op_fp(0x00, 0, 2, 1, 3))  # FADD.S f3, f1, f2
        assert hart._fpr_bits[3] == _boxf(S3_0)

    def test_fsub_s(self, hart):
        hart._fpr_bits[1] = _boxf(S3_0)
        hart._fpr_bits[2] = _boxf(S1_0)
        hart.exec_instr(_op_fp(0x04, 0, 2, 1, 3))  # FSUB.S
        assert hart._fpr_bits[3] == _boxf(S2_0)

    def test_fmul_s(self, hart):
        hart._fpr_bits[1] = _boxf(S2_0)
        hart._fpr_bits[2] = _boxf(S3_0)
        hart.exec_instr(_op_fp(0x08, 0, 2, 1, 3))  # FMUL.S
        assert hart._fpr_bits[3] == _boxf(S6_0)

    def test_fdiv_s(self, hart):
        hart._fpr_bits[1] = _boxf(S6_0)
        hart._fpr_bits[2] = _boxf(S2_0)
        hart.exec_instr(_op_fp(0x0C, 0, 2, 1, 3))  # FDIV.S
        assert hart._fpr_bits[3] == _boxf(S3_0)

    def test_fsqrt_s(self, hart):
        hart._fpr_bits[1] = _boxf(S4_0)
        hart.exec_instr(_op_fp(0x2C, 0, 0, 1, 3))  # FSQRT.S f3, f1
        assert hart._fpr_bits[3] == _boxf(S2_0)


# ============================================================
#  双精度算术
# ============================================================


class TestDoublePrecisionArith:
    def test_fadd_d(self, hart):
        hart._fpr_bits[1] = D1_0
        hart._fpr_bits[2] = D2_0
        hart.exec_instr(_op_fp(0x01, 0, 2, 1, 3))  # FADD.D
        assert hart._fpr_bits[3] == D3_0

    def test_fmul_d(self, hart):
        hart._fpr_bits[1] = D2_0
        hart._fpr_bits[2] = D3_0  # 2.0 * 3.0 = 6.0
        hart.exec_instr(_op_fp(0x09, 0, 2, 1, 3))  # FMUL.D
        assert hart._fpr_bits[3] == 0x4018_0000_0000_0000  # 6.0


# ============================================================
#  比较 / 转换 / 分类 / 移动 (跨 GPR)
# ============================================================


class TestCompareConvertMove:
    def test_feq_s_equal(self, hart):
        hart._fpr_bits[1] = _boxf(S1_0)
        hart._fpr_bits[2] = _boxf(S1_0)
        hart.exec_instr(_op_fp(0x50, 2, 2, 1, 5))  # FEQ.S x5, f1, f2
        assert hart.gprs[5] == 1

    def test_flt_s(self, hart):
        hart._fpr_bits[1] = _boxf(S1_0)
        hart._fpr_bits[2] = _boxf(S2_0)
        hart.exec_instr(_op_fp(0x50, 1, 2, 1, 5))  # FLT.S x5, f1, f2
        assert hart.gprs[5] == 1

    def test_fcvt_w_s_rtz(self, hart):
        hart._fpr_bits[1] = _boxf(0x4020_0000)  # 2.5f
        hart.exec_instr(_op_fp(0x60, 1, 0, 1, 5))  # FCVT.W.S x5, f1, RTZ
        assert hart.gprs[5] == 2

    def test_fcvt_s_w(self, hart):
        hart.gprs[5] = 3
        hart.exec_instr(_op_fp(0x68, 0, 0, 5, 1))  # FCVT.S.W f1, x5
        assert hart._fpr_bits[1] == _boxf(S3_0)

    def test_fclass_s_neg_inf(self, hart):
        hart._fpr_bits[1] = _boxf(0xFF80_0000)  # -inf
        hart.exec_instr(_op_fp(0x70, 1, 0, 1, 5))  # FCLASS.S x5, f1
        assert hart.gprs[5] == (1 << 0)

    def test_fmv_x_w(self, hart):
        hart._fpr_bits[1] = _boxf(S1_0)
        hart.exec_instr(_op_fp(0x70, 0, 0, 1, 5))  # FMV.X.W x5, f1
        assert hart.gprs[5] == S1_0

    def test_fmv_w_x(self, hart):
        hart.gprs[5] = S1_0
        hart.exec_instr(_op_fp(0x78, 0, 0, 5, 1))  # FMV.W.X f1, x5
        assert hart._fpr_bits[1] == _boxf(S1_0)

    def test_fsgnj_s(self, hart):
        neg1 = 0xBF80_0000  # -1.0f
        hart._fpr_bits[1] = _boxf(S1_0)
        hart._fpr_bits[2] = _boxf(neg1)
        hart.exec_instr(_op_fp(0x10, 0, 2, 1, 3))  # FSGNJ.S: mag(1.0) sign(-1.0)
        assert hart._fpr_bits[3] == _boxf(neg1)

    def test_fmin_s(self, hart):
        hart._fpr_bits[1] = _boxf(S1_0)
        hart._fpr_bits[2] = _boxf(S2_0)
        hart.exec_instr(_op_fp(0x14, 0, 2, 1, 3))  # FMIN.S
        assert hart._fpr_bits[3] == _boxf(S1_0)

    def test_fmax_s(self, hart):
        hart._fpr_bits[1] = _boxf(S1_0)
        hart._fpr_bits[2] = _boxf(S2_0)
        hart.exec_instr(_op_fp(0x14, 1, 2, 1, 3))  # FMAX.S
        assert hart._fpr_bits[3] == _boxf(S2_0)


# ============================================================
#  FMA
# ============================================================


class TestFusedMultiplyAdd:
    def test_fmadd_s(self, hart):
        # 2.0 * 3.0 + 1.0 = 7.0
        hart._fpr_bits[1] = _boxf(S2_0)
        hart._fpr_bits[2] = _boxf(S3_0)
        hart._fpr_bits[3] = _boxf(S1_0)
        hart.exec_instr(_fma(0b1000011, 0, 3, 2, 1, 4))  # FMADD.S f4
        assert hart._fpr_bits[4] == _boxf(S7_0)

    def test_fmsub_s(self, hart):
        # 2.0 * 3.0 - 1.0 = 5.0
        hart._fpr_bits[1] = _boxf(S2_0)
        hart._fpr_bits[2] = _boxf(S3_0)
        hart._fpr_bits[3] = _boxf(S1_0)
        hart.exec_instr(_fma(0b1000111, 0, 3, 2, 1, 4))  # FMSUB.S
        assert hart._fpr_bits[4] == _boxf(0x40A0_0000)  # 5.0


# ============================================================
#  FP load / store
# ============================================================


class TestFpLoadStore:
    def test_flw_nanboxes(self, hart):
        # 在 RAM 写入 1.0f, FLW 加载并 NaN-box。
        hart.gprs[2] = 0x1000
        hart._mem_write_phy(0x1000, S1_0.to_bytes(4, "little"))
        instr = (0 << 20) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0b0000111  # FLW f1, 0(x2)
        hart.exec_instr(instr)
        assert hart._fpr_bits[1] == _boxf(S1_0)

    def test_fld_full64(self, hart):
        hart.gprs[2] = 0x1000
        hart._mem_write_phy(0x1000, D2_0.to_bytes(8, "little"))
        instr = (0 << 20) | (2 << 15) | (0b011 << 12) | (1 << 7) | 0b0000111  # FLD f1, 0(x2)
        hart.exec_instr(instr)
        assert hart._fpr_bits[1] == D2_0

    def test_fsw_stores_low32(self, hart):
        hart.gprs[2] = 0x1000
        hart._fpr_bits[3] = _boxf(S3_0)
        instr = (3 << 20) | (2 << 15) | (0b010 << 12) | (0 << 7) | 0b0100111  # FSW f3, 0(x2)
        hart.exec_instr(instr)
        assert hart._mem_read_phy(0x1000, 4) == S3_0.to_bytes(4, "little")

    def test_fsd_stores64(self, hart):
        hart.gprs[2] = 0x1000
        hart._fpr_bits[3] = D3_0
        instr = (3 << 20) | (2 << 15) | (0b011 << 12) | (0 << 7) | 0b0100111  # FSD f3, 0(x2)
        hart.exec_instr(instr)
        assert hart._mem_read_phy(0x1000, 8) == D3_0.to_bytes(8, "little")


# ============================================================
#  mstatus.FS 门控 + fcsr
# ============================================================


class TestFsGatingAndFcsr:
    def test_fp_traps_when_fs_off(self, hart):
        # 关闭 FS → FADD 应触发 IllInstr 陷态 (PC 跳转到 mtvec)。
        hart.mstatus_val = hart.mstatus_val & ~(0b11 << 13)
        hart.csrs["mtvec"].val = 0x8000_0000
        hart._fpr_bits[1] = _boxf(S1_0)
        hart._fpr_bits[2] = _boxf(S2_0)
        hart.exec_instr(_op_fp(0x00, 0, 2, 1, 3))
        # IllInstr 委派/投递后 PC 应指向陷态向量。
        assert hart.pc == 0x8000_0000

    def test_fp_marks_fs_dirty(self, hart):
        hart._fpr_bits[1] = _boxf(S1_0)
        hart._fpr_bits[2] = _boxf(S2_0)
        hart.exec_instr(_op_fp(0x00, 0, 2, 1, 3))
        # FS 应被置为 Dirty (0b11)。
        assert (hart.mstatus_val >> 13) & 0b11 == 0b11

    def test_inexact_flag_accumulates(self, hart):
        # 1.0 / 3.0 不精确 → NX (fflags bit 0)。
        s0_333 = 0x3EAA_AAAB
        hart._fpr_bits[1] = _boxf(S1_0)
        hart._fpr_bits[2] = _boxf(S3_0)
        hart.exec_instr(_op_fp(0x0C, 0, 2, 1, 3))  # FDIV.S 1.0/3.0
        assert hart.csrs["fcsr"].val & 0x1F != 0  # NX 置位
