#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""Zbb 位操作扩展指令测试.

覆盖 Zbb 扩展全部指令:
- R-type: andn, orn, xnor, min, max, minu, maxu, rol, ror
- I-type: clz, ctz, cpop, sext.b, sext.h, rori
- OP-IMM-32: clzw, ctzw, cpopw, roriw
- OP-32: rolw, rorw

Native 路径对 Zbb 返回 trap, 自动 fallback 到 Python 实现.
"""

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.mem_check_aux import inject_memory_backend

# -- 指令编码辅助 --

def _r_type(  # noqa: PLR0913, PLR0917 — R-type 编码天然 6 字段
    funct7: int,
    rs2: int,
    rs1: int,
    funct3: int,
    rd: int,
    opcode: int,
) -> int:
    """构造 R-type 指令字."""
    return (
        (funct7 << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (rd << 7)
        | opcode
    )


def _i_type(imm12: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    """构造 I-type 指令字. imm12 为 12-bit 无符号."""
    return (
        ((imm12 & 0xFFF) << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (rd << 7)
        | opcode
    )


# Opcode 常量
OP_ALU = 0b0110011       # R-type ALU
OP_IMM = 0b0010011       # I-type immediate
OP_IMM32 = 0b0011011     # I-type 32-bit
OP32 = 0b0111011         # R-type 32-bit

# Zbb funct7 常量
ZBB_FUNCT7_ANDN_ORN_XNOR = 0x20
ZBB_FUNCT7_MIN_MAX = 0x05
ZBB_FUNCT7_ROL_ROR = 0x30


def _make_ram():
    ram = bytearray(2 * 1024 * 1024)

    def read_fn(addr, size):
        return bytes(ram[addr : addr + size])

    def write_fn(addr, data):
        for i, b in enumerate(data):
            ram[addr + i] = b

    return ram, read_fn, write_fn


@pytest.fixture
def hart() -> Hart:
    h = Hart(id=0)
    ram, rf, wf = _make_ram()
    inject_memory_backend(h, rf, wf)
    return h


# =====================================================================
#  Zbb R-type (OP = 0x33)
# =====================================================================



class TestZbbRtype:
    """Zbb R-type 指令: andn, orn, xnor, min, max, minu, maxu, rol, ror,
    及 OP-32 (opcode=0x3B) 的 rolw/rorw."""

    @pytest.mark.parametrize(
        "rs1, rs2, fn7, fn3, opcode, expect_rd, instr_name", [
            # ---- andn/orn/xnor (funct7=0x20): andn=rs1&~rs2, orn=rs1|~rs2 ----
            (
                0xFF00FF00FF00FF00, 0xF0F0F0F0F0F0F0F0,
                ZBB_FUNCT7_ANDN_ORN_XNOR, 0b111, OP_ALU,
                0x0F000F000F000F00, "andn"
            ), (
                0xFF00FF00FF00FF00, 0xF0F0F0F0F0F0F0F0,
                ZBB_FUNCT7_ANDN_ORN_XNOR, 0b110, OP_ALU,
                0xFF0FFF0FFF0FFF0F, "orn"
            ), (
                0xFF00FF00FF00FF00, 0xF0F0F0F0F0F0F0F0,
                ZBB_FUNCT7_ANDN_ORN_XNOR, 0b100, OP_ALU,
                0xF00FF00FF00FF00F, "xnor"
            ),
            # ---- min/max/minu/maxu (funct7=0x05) ----
            (
                0xFFFFFFFFFFFFFFF0, 0x0000000000000010,
                ZBB_FUNCT7_MIN_MAX, 0b100, OP_ALU,
                0xFFFFFFFFFFFFFFF0, "min"
            ), (
                0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFD,
                ZBB_FUNCT7_MIN_MAX, 0b100, OP_ALU,
                0xFFFFFFFFFFFFFFFD, "min"
            ), (
                0xFFFFFFFFFFFFFFF0, 0x0000000000000010,
                ZBB_FUNCT7_MIN_MAX, 0b110, OP_ALU,
                0x0000000000000010, "max"
            ), (
                0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFD,
                ZBB_FUNCT7_MIN_MAX, 0b110, OP_ALU,
                0xFFFFFFFFFFFFFFFE, "max"
            ), (
                0xFFFFFFFFFFFFFFF0, 0x0000000000000010,
                ZBB_FUNCT7_MIN_MAX, 0b101, OP_ALU,
                0x0000000000000010, "minu"
            ), (
                0xFFFFFFFFFFFFFFF0, 0x0000000000000010,
                ZBB_FUNCT7_MIN_MAX, 0b111, OP_ALU,
                0xFFFFFFFFFFFFFFF0, "maxu"
            ),
            # ---- rol/ror (funct7=0x30, OP=0x33) ----
            (
                0x8000000000000001, 1,
                ZBB_FUNCT7_ROL_ROR, 0b001, OP_ALU,
                0x0000000000000003, "rol"
            ), (
                0xDEADBEEFCAFEBABE, 0,
                ZBB_FUNCT7_ROL_ROR, 0b001, OP_ALU,
                0xDEADBEEFCAFEBABE, "rol"
            ), (
                0x8000000000000001, 1,
                ZBB_FUNCT7_ROL_ROR, 0b101, OP_ALU,
                0xC000000000000000, "ror"
            ), (
                0x123456789ABCDEF0, 4,
                ZBB_FUNCT7_ROL_ROR, 0b101, OP_ALU,
                0x0123456789ABCDEF, "ror"
            ),
            # ---- OP-32 (opcode=0x3B): rolw/rorw, 32 位旋转, 结果 sign-extend ----
            (
                0x80000001, 1,
                ZBB_FUNCT7_ROL_ROR, 0b001, OP32,
                0x0000000000000003, "rolw"
            ), (
                0x40000000, 1,
                ZBB_FUNCT7_ROL_ROR, 0b001, OP32,
                0xFFFFFFFF80000000, "rolw"
            ), (
                0x00000001, 1,
                ZBB_FUNCT7_ROL_ROR, 0b101, OP32,
                0xFFFFFFFF80000000, "rorw"
            ), (
                0x12345678, 4,
                ZBB_FUNCT7_ROL_ROR, 0b101, OP32,
                0xFFFFFFFF81234567, "rorw"
            ), (
                0x12345678, 0,
                ZBB_FUNCT7_ROL_ROR, 0b001, OP32,
                0x12345678, "rolw"
            ), (
                0x12345678, 0,
                ZBB_FUNCT7_ROL_ROR, 0b101, OP32,
                0x12345678, "rorw"
            ), (
                0x12345678, 0x104,  # rs2 低 5 位 = 4
                ZBB_FUNCT7_ROL_ROR, 0b101, OP32,
                0xFFFFFFFF81234567, "rorw"
            ),
        ]
    )
    def test_zbb_rtype(  # noqa: PLR0913, PLR0917 — 参数化测试, 每列一个形参
        self, hart,
        rs1: int, rs2: int,
        fn7: int, fn3: int, opcode:int,
        expect_rd: int,
        instr_name: str
    ):
        hart.gprs[1] = rs1
        hart.gprs[2] = rs2
        instr = _r_type(fn7, 2, 1, fn3, 3, opcode)
        hart.exec_instr(instr)
        assert hart.gprs[3] == expect_rd, \
        f'wrong at instr<{instr_name}>:fn7<{fn7}>:fn3<{fn3}>: ' + \
        f'gpr1<{rs1}>, gpr2<{rs2}>'

    def test_andn_rd_zero(self, hart):
        """rd=0 不写结果."""
        hart.gprs[1] = 0xFF
        hart.gprs[2] = 0xFF
        instr = _r_type(ZBB_FUNCT7_ANDN_ORN_XNOR, 2, 1, 0b111, 0, OP_ALU)
        hart.exec_instr(instr)
        assert hart.gprs[0] == 0


# =====================================================================
#  Zbb I-type (OP-IMM = 0x13, funct7=0x30)
# =====================================================================


class TestZbbItype:
    """Zbb I-type (OP-IMM=0x13) 与 OP-IMM-32 (0x1B) 指令:
    clz/ctz/cpop/sext/rori 及 32 位变体."""
    @pytest.mark.parametrize(
    "gpr1_val, imm, fn3, gpr3_val, instruction, opcode", [
        # ---- OP-IMM (64 位语义) ----
        # clz
        (0x00FFFFFFFFFFFFFF, 0x600, 0b001, 8, "clz", OP_IMM),
        (0xFFFFFFFFFFFFFF00, 0x600, 0b001, 0, "clz", OP_IMM),  # MSB=1 -> clz=0
        (0, 0x600, 0b001, 64, "clz", OP_IMM),
        (0xFFFFFFFFFFFFFFFF, 0x600, 0b001, 0, "clz", OP_IMM),
        (0x123456789ABCDEF1, 0x600, 0b001, 3, "clz", OP_IMM),  # 0x1xxx -> 前导 3 个 0
        # cpop
        (0xFF, 0x602, 0b001, 8, "cpop", OP_IMM),
        (0xFFFFFFFFFFFFFFFF, 0x602, 0b001, 64, "cpop", OP_IMM),
        (0, 0x602, 0b001, 0, "cpop", OP_IMM),
        # sext
        (0x7F, 0x604, 0b001, 0x7F, "sext.b", OP_IMM),
        (0x80, 0x604, 0b001, 0xFFFFFFFFFFFFFF80, "sext.b", OP_IMM),
        (0x7FFF, 0x605, 0b001, 0x7FFF, "sext.h", OP_IMM),
        (0x8000, 0x605, 0b001, 0xFFFFFFFFFFFF8000, "sext.h", OP_IMM),
        # rori
        (0x8000000000000001, 0x601, 0b101, 0xC000000000000000, "rori", OP_IMM),
        (0x123456789ABCDEF0, 0x604, 0b101, 0x0123456789ABCDEF, "rori", OP_IMM),
        (0xDEADBEEFCAFEBABE, 0x600, 0b101, 0xDEADBEEFCAFEBABE, "rori", OP_IMM),
        # ---- OP-IMM-32: 仅看低 32 位, 结果按 32 位 sign-extend ----
        # clzw
        (0x00FFFFFF, 0x600, 0b001, 8, "clzw", OP_IMM32),
        (0, 0x600, 0b001, 32, "clzw", OP_IMM32),
        (0xFFFFFFFF00000001, 0x600, 0b001, 31, "clzw", OP_IMM32),  # 高 32 位被忽略
        # ctzw
        (0xFFFFFF00, 0x601, 0b001, 8, "ctzw", OP_IMM32),
        (0, 0x601, 0b001, 32, "ctzw", OP_IMM32),
        # cpopw
        (0xFF, 0x602, 0b001, 8, "cpopw", OP_IMM32),
        (0xFFFFFFFF0000000F, 0x602, 0b001, 4, "cpopw", OP_IMM32),  # 高 32 位被忽略
        # roriw
        (0x12345678, 0x604, 0b101, 0xFFFFFFFF81234567, "roriw", OP_IMM32),
        (0x80000000, 0x600, 0b101, 0xFFFFFFFF80000000, "roriw", OP_IMM32),
    ])
    def test_zbb_itype( # noqa: PLR0913, PLR0917
        self,
        hart,
        gpr1_val: int,
        imm: int,
        fn3: int,
        gpr3_val: int,
        instruction: str,
        opcode: int
    ):
        hart.gprs[1] = gpr1_val
        instr = _i_type(imm, 1, fn3, 3, opcode)
        hart.exec_instr(instr)
        assert hart.gprs[3] == gpr3_val, \
        f'wrong at instr<{instruction}>:imm<{imm}>, ' + \
        f'gpr1: {gpr1_val}, gpr3: {gpr3_val}'
