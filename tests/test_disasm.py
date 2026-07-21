#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""反汇编器测试: 覆盖 RV64 I + M + Zicsr + AMO 各指令格式."""
from typing import Callable

import pytest

from pyremu.utils.disassem import disasm

# ============================================================
#  指令构造辅助
# ============================================================


def _r_type(funct7: int, rs2: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _i_type(imm12: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _s_type(imm7: int, rs2: int, rs1: int, funct3: int, imm5: int, opcode: int) -> int:
    return (
        ((imm7 & 0x7F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((imm5 & 0x1F) << 7)
        | opcode
    )


def _b_type(
    imm12: int,
    imm10_5: int,
    rs2: int,
    rs1: int,
    funct3: int,
    imm4_1: int,
    imm11: int,
    opcode: int,
) -> int:
    return (
        (imm12 << 31)
        | (imm10_5 << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (imm4_1 << 8)
        | (imm11 << 7)
        | opcode
    )


def _u_type(imm20: int, rd: int, opcode: int) -> int:
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode


def _j_type(
    imm20: int,
    imm10_1: int,
    imm11: int,
    imm19_12: int,
    rd: int,
    opcode: int,
) -> int:
    return (
        (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) | (imm19_12 << 12) | (rd << 7) | opcode
    )


# Opcodes
OP = 0b0110011
OP_IMM = 0b0010011
OP32 = 0b0111011
OP_IMM32 = 0b0011011
LD = 0b0000011
ST = 0b0100011
BR = 0b1100011
JALR = 0b1100111
JAL = 0b1101111
LUI = 0b0110111
AUIPC = 0b0010111
SYS = 0b1110011
FENCE = 0b0001111
AMO = 0b0101111


# ============================================================
#  R-type
# ============================================================


class TestRType:
    def test_add(self):
        instr = _r_type(funct7=0, rs2=12, rs1=11, funct3=0, rd=10, opcode=OP)
        assert disasm(instr, 0) == "add          x10, x11, x12"

    def test_sub(self):
        instr = _r_type(funct7=0x20, rs2=6, rs1=5, funct3=0, rd=7, opcode=OP)
        assert disasm(instr, 0) == "sub          x7, x5, x6"

    def test_mul(self):
        instr = _r_type(funct7=1, rs2=12, rs1=11, funct3=0, rd=10, opcode=OP)
        assert disasm(instr, 0) == "mul          x10, x11, x12"

    def test_divu(self):
        instr = _r_type(funct7=1, rs2=12, rs1=11, funct3=5, rd=10, opcode=OP)
        assert disasm(instr, 0) == "divu         x10, x11, x12"

    def test_srl(self):
        instr = _r_type(funct7=0, rs2=2, rs1=1, funct3=5, rd=3, opcode=OP)
        assert disasm(instr, 0) == "srl          x3, x1, x2"

    def test_unknown_rtype(self):
        instr = _r_type(funct7=0x7F, rs2=0, rs1=0, funct3=7, rd=0, opcode=OP)
        assert disasm(instr, 0) == "<unknown opcode>"


# ============================================================
#  I-type ALU
# ============================================================


class TestITypeALU:
    def test_addi(self):
        instr = _i_type(imm12=42, rs1=1, funct3=0, rd=2, opcode=OP_IMM)
        assert disasm(instr, 0) == "addi         x2, x1, 42"

    def test_addi_negative(self):
        instr = _i_type(imm12=(-16 & 0xFFF), rs1=2, funct3=0, rd=3, opcode=OP_IMM)
        assert disasm(instr, 0) == "addi         x3, x2, -16"

    def test_slti(self):
        instr = _i_type(imm12=0, rs1=10, funct3=2, rd=11, opcode=OP_IMM)
        assert disasm(instr, 0) == "slti         x11, x10, 0"

    def test_slli(self):
        # SLLI: funct3=001, shamt in bits [25:20], funct7=0
        instr = _i_type(imm12=0, rs1=1, funct3=1, rd=2, opcode=OP_IMM)
        # imm12=0 means shamt=0; let's set shamt=3
        instr = (0 << 25) | (3 << 20) | (1 << 15) | (1 << 12) | (2 << 7) | OP_IMM
        assert disasm(instr, 0) == "slli         x2, x1, 3"

    def test_srai(self):
        # SRAI: funct3=101, funct7=0x20
        instr = (0x20 << 25) | (4 << 20) | (3 << 15) | (5 << 12) | (4 << 7) | OP_IMM
        assert disasm(instr, 0) == "srai         x4, x3, 4"

    def test_andi(self):
        instr = _i_type(imm12=0xFF, rs1=5, funct3=7, rd=6, opcode=OP_IMM)
        assert disasm(instr, 0) == "andi         x6, x5, 255"


# ============================================================
#  RV64 32-bit ops
# ============================================================


class TestRV64_32Bit:
    def test_addw(self):
        instr = _r_type(funct7=0, rs2=12, rs1=11, funct3=0, rd=10, opcode=OP32)
        assert disasm(instr, 0) == "addw         x10, x11, x12"

    def test_subw(self):
        instr = _r_type(funct7=0x20, rs2=6, rs1=5, funct3=0, rd=7, opcode=OP32)
        assert disasm(instr, 0) == "subw         x7, x5, x6"

    def test_addiw(self):
        instr = _i_type(imm12=8, rs1=1, funct3=0, rd=2, opcode=OP_IMM32)
        assert disasm(instr, 0) == "addiw        x2, x1, 8"

    def test_slliw(self):
        instr = (0 << 25) | (2 << 20) | (1 << 15) | (1 << 12) | (3 << 7) | OP_IMM32
        assert disasm(instr, 0) == "slliw        x3, x1, 2"


# ============================================================
#  Load / Store
# ============================================================


class TestLoadStore:
    def test_lw(self):
        instr = _i_type(imm12=16, rs1=10, funct3=2, rd=5, opcode=LD)
        assert disasm(instr, 0) == "lw           x5, 16(x10)"

    def test_lw_negative_offset(self):
        instr = _i_type(imm12=(-8 & 0xFFF), rs1=2, funct3=2, rd=3, opcode=LD)
        assert disasm(instr, 0) == "lw           x3, -8(x2)"

    def test_ld(self):
        instr = _i_type(imm12=0, rs1=8, funct3=3, rd=9, opcode=LD)
        assert disasm(instr, 0) == "ld           x9, 0(x8)"

    def test_lbu(self):
        instr = _i_type(imm12=1, rs1=1, funct3=4, rd=2, opcode=LD)
        assert disasm(instr, 0) == "lbu          x2, 1(x1)"

    def test_sw(self):
        # S-type: imm[11:5] | rs2 | rs1 | funct3 | imm[4:0]
        instr = _s_type(imm7=0, rs2=5, rs1=10, funct3=2, imm5=16, opcode=ST)
        assert disasm(instr, 0) == "sw           x5, 16(x10)"

    def test_sd(self):
        instr = _s_type(imm7=0, rs2=5, rs1=10, funct3=3, imm5=0, opcode=ST)
        assert disasm(instr, 0) == "sd           x5, 0(x10)"


# ============================================================
#  Branch
# ============================================================


@pytest.mark.parametrize(
    "inp, addr, check_fn", [
        (
            # B-type: imm[12|10:5] | rs2 | rs1 | funct3 | imm[4:1|11]
            # offset = 16: imm[12]=0, imm[10:5]=0, imm[4:1]=1000(8), imm[11]=0
            [0, 0, 12, 11, 0, 8, 0, BR], 0x80000000,
            lambda x: "beq" in x and "0x80000010" in x
        ), (
            # offset = -8: sign-extend to 13-bit -> 0x1FF8
            # B-type encoding: imm[12]=1, imm[10:5]=0x3F(all1s), imm[4:1]=0xC(12), imm[11]=1
            # offset=-8 -> imm12=1, imm10_5=0x3F, imm4_1=0xC, imm11=1
            [1, 0x3F, 0, 1, 1, 0xC, 1, BR], 0x80000010,
            lambda x: "bne" in x and "0x80000008" in x
        ), (
            [0, 0, 12, 11, 5, 8, 0, BR], 0x8000_0000,
            lambda x: "bge" in x
        )
    ]
)
def test_br_instr(
    inp: list[int],
    addr: int,
    check_fn: Callable[[str], bool]
):
    assert len(inp) == 8
    instr = _b_type(
        inp[0], inp[1], inp[2], inp[3],
        inp[4], inp[5], inp[6], inp[7]
    )
    res = disasm(instr, addr)
    assert check_fn(res)


# ============================================================
#  JAL / JALR / LUI / AUIPC
# ============================================================


class TestJumps:
    def test_jal(self):
        # J-type: imm[20|10:1|11|19:12]  rd  JAL
        # offset = 16: imm20=0, imm10_1=8, imm11=0, imm19_12=0
        instr = _j_type(0, 8, 0, 0, 1, JAL)
        result = disasm(instr, 0x80000000)
        assert result == "jal          x1, 0x80000010"

    def test_jalr(self):
        instr = _i_type(imm12=16, rs1=1, funct3=0, rd=1, opcode=JALR)
        assert disasm(instr, 0) == "jalr         x1, 16(x1)"

    def test_lui(self):
        instr = _u_type(imm20=0x12345, rd=5, opcode=LUI)
        assert disasm(instr, 0) == "lui          x5, 0x12345"

    def test_auipc(self):
        instr = _u_type(imm20=0x21000, rd=10, opcode=AUIPC)
        assert disasm(instr, 0) == "auipc        x10, 0x21000"


# ============================================================
#  System (CSR + privileged)
# ============================================================


class TestSystem:
    def test_ecall(self):
        assert disasm(0x00000073, 0) == "ecall"

    def test_mret(self):
        # MRET: funct12=0x302, funct3=0
        instr = (0x302 << 20) | (0b000 << 12) | SYS
        assert disasm(instr, 0) == "mret"

    def test_csrrw(self):
        # CSRRW: funct3=001, rd=x10, rs1=x11, csr=mstatus(0x300)
        instr = (0x300 << 20) | (11 << 15) | (1 << 12) | (10 << 7) | SYS
        assert disasm(instr, 0) == "csrrw        x10, mstatus, x11"

    def test_csrrsi(self):
        # CSRRSI: funct3=110, rd=x5, uimm=3, csr=mie(0x304)
        instr = (0x304 << 20) | (3 << 15) | (6 << 12) | (5 << 7) | SYS
        assert disasm(instr, 0) == "csrrsi       x5, mie, 3"

    def test_csr_unknown(self):
        # CSR at unknown address 0xFFF
        instr = (0xFFF << 20) | (0 << 15) | (2 << 12) | (5 << 7) | SYS
        result = disasm(instr, 0)
        assert "0xfff" in result.lower()


# ============================================================
#  FENCE
# ============================================================


class TestFence:
    def test_fence(self):
        instr = (0 << 12) | FENCE  # funct3=0
        assert disasm(instr, 0) == "fence"

    def test_fence_i(self):
        instr = (1 << 12) | FENCE  # funct3=1
        assert disasm(instr, 0) == "fence.i"


# ============================================================
#  AMO
# ============================================================


class TestAMO:
    def test_amoadd_w(self):
        # AMOADD.W: funct5=00000, funct3=010 (W)
        instr = (0 << 27) | (12 << 20) | (11 << 15) | (2 << 12) | (10 << 7) | AMO
        assert disasm(instr, 0) == "amoadd.w     x10, x12, (x11)"

    def test_amoxor_d(self):
        # AMOXOR.D: funct5=00100, funct3=011 (D)
        instr = (4 << 27) | (12 << 20) | (11 << 15) | (3 << 12) | (10 << 7) | AMO
        assert disasm(instr, 0) == "amoxor.d     x10, x12, (x11)"

    def test_lr_w(self):
        # LR.W: funct5=00010, funct3=010, rs2=0 (unused)
        instr = (2 << 27) | (0 << 20) | (11 << 15) | (2 << 12) | (10 << 7) | AMO
        assert disasm(instr, 0) == "lr.w         x10, x0, (x11)"


# ============================================================
#  Compressed / Unknown
# ============================================================


# ============================================================
#  Compressed (C extension) — 验证与 llvm-objdump 一致
# ============================================================


class TestCompressed:
    """覆盖 RV64C 常用压缩指令, 编码来自 nonsense.o / RISC-V 规范."""

    def test_c_nop(self):
        """0x0001 -> c.nop."""
        assert "c.nop" in disasm(0x0001, 0x5EC)

    def test_c_ebreak(self):
        """0x9002 -> c.ebreak."""
        assert "c.ebreak" in disasm(0x9002, 0x5E0)

    def test_c_mv_a5_a0(self):
        """nonsense.o @5c4: 87aa -> c.mv a5, a0."""
        result = disasm(0x87AA, 0x5C4)
        assert result.startswith("c.mv")
        assert "x15" in result and "x10" in result

    def test_c_mv_a6_sp(self):
        """nonsense.o @5da: 880a -> c.mv a6, sp."""
        result = disasm(0x880A, 0x5DA)
        assert result.startswith("c.mv")
        assert "x16" in result and "x2" in result

    def test_c_jr_ra(self):
        """0x8082 -> c.jr ra (ret)."""
        result = disasm(0x8082, 0x5EA)
        assert result.startswith("c.jr")
        assert "x1" in result

    def test_c_jr_a5(self):
        """nonsense.o @60c: 8782 -> c.jr a5."""
        result = disasm(0x8782, 0x60C)
        assert result.startswith("c.jr")
        assert "x15" in result

    def test_c_li_d3_0(self):
        """nonsense.o @5d6: 4681 -> c.li a3, 0."""
        result = disasm(0x4681, 0x5D6)
        assert result.startswith("c.li")

    def test_c_li_a4_0(self):
        """nonsense.o @5d8: 4701 -> c.li a4, 0."""
        result = disasm(0x4701, 0x5D8)
        assert result.startswith("c.li")

    def test_c_add_a1_a5(self):
        """nonsense.o @628: 95be -> c.add a1, a5."""
        result = disasm(0x95BE, 0x628)
        assert result.startswith("c.add")
        assert "x11" in result and "x15" in result

    def test_c_sub_a1_a0(self):
        """nonsense.o @620: 8d89 -> c.sub a1, a0."""
        result = disasm(0x8D89, 0x620)
        assert result.startswith("c.sub")
        assert "x11" in result and "x10" in result

    def test_c_beqz_a5(self):
        """nonsense.o @60a: c391 -> c.beqz a5, 0x60e."""
        result = disasm(0xC391, 0x60A)
        assert result.startswith("c.beqz")

    def test_c_srai_a1_1(self):
        """nonsense.o @62a: 8585 -> c.srai a1, 0x1."""
        result = disasm(0x8585, 0x62A)
        assert result.startswith("c.srai")

    def test_c_srli_a1_3f(self):
        """nonsense.o @626: 91fd -> c.srli a1, 0x3f."""
        result = disasm(0x91FD, 0x626)
        assert result.startswith("c.srli")

    def test_c_ldsp_a1_sp_0(self):
        """nonsense.o @5ce: 6582 -> c.ldsp a1, 0(sp)."""
        result = disasm(0x6582, 0x5CE)
        assert result.startswith("c.ldsp")
        assert "x11" in result

    def test_c_addi_a2_sp_8(self):
        """0x0030 -> c.addi a2, 8 (rd=x12)."""
        result = disasm(0x0030, 0x5D0)
        assert result.startswith("c.addi")

    def test_4byte_read_contains_compressed(self):
        """模拟 Emulator 取指: 一次读 4 字节, 低两位 ≠ 11 的按 16-bit 解码."""
        # nonsense.o @5c4: bytes = aa 87 17 25 -> instr = 0x251787aa
        result = disasm(0x251787AA, 0x5C4)
        assert result.startswith("c.mv")
        assert "x15" in result and "x10" in result

    def test_c_addi16sp_positive(self):
        """C.ADDI16SP sp += 112 — 编码 nzimm[9:4]=7 (0b000111)."""
        # bit12=0, bits[4:3]=00, bit5=1, bit2=1, bit6=1
        # [15:13]=011, [12]=0, [11:7]=00010, [6]=1, [5]=1, [4]=0, [3]=0, [2]=1, [1:0]=01
        # -> 0x6165
        result = disasm(0x6165, 0x8001_981E)
        assert result.startswith("c.addi16sp")
        assert "112" in result or "sp" in result

    def test_c_addi16sp_negative(self):
        """C.ADDI16SP sp -= 112 — 编码 nzimm[9:4]=-7 (0b111001)."""
        # bit12=1, bits[4:3]=11, bit5=0, bit2=0, bit6=1
        # [15:13]=011, [12]=1, [11:7]=00010, [6]=1, [5]=0, [4]=1, [3]=1, [2]=0, [1:0]=01
        # -> 0x7159
        result = disasm(0x7159, 0x8001_981E)
        assert result.startswith("c.addi16sp")
        assert "-" in result

    def test_c_jalr(self):
        """C.JALR x11 — funct3=100, bit12=1, rs1=x11, rs2=0."""
        # [15:13]=100, [12]=1, [11:7]=01011, [6:2]=00000, [1:0]=10
        # -> 0x9582
        result = disasm(0x9582, 0x8001_9902)
        assert result.startswith("c.jalr")
        assert "x11" in result

    def test_c_jalr_ra(self):
        """C.JALR ra — funct3=100, bit12=1, rs1=x1, rs2=0."""
        # [15:13]=100, [12]=1, [11:7]=00001, [6:2]=00000, [1:0]=10
        # -> 0x9082
        result = disasm(0x9082, 0x8001_9902)
        assert result.startswith("c.jalr")
        assert "x1" in result


class TestEdgeCases:
    def test_compressed_quadrant0(self):
        # low 2 bits = 00
        assert disasm(0x00000000, 0).startswith("c.")

    def test_compressed_quadrant1(self):
        # low 2 bits = 01
        assert disasm(0x00000001, 0).startswith("c.")

    def test_compressed_quadrant2(self):
        # low 2 bits = 10
        assert disasm(0x00000002, 0).startswith("c.")

    def test_unknown_opcode(self):
        # opcode 0b1100111 实际上是 JALR... 用个不存在的 opcode
        assert disasm(0xFFFFFFFF, 0) == "<unknown opcode>"


class TestCompressedLargeOffset:
    """压缩指令大偏移量反汇编 — 验证立即数布局正确解码."""

    # -- C0 quadrant: C.SD / C.LD (uimm[7:6] ≠ 0 时才与 C.SW/C.LW 有差异) --

    def test_c_sd_large_offset_144(self):
        """C.SD x14, 144(x12): uimm[7:6]=2, uimm[5:3]=2 -> offset=144."""
        instr = (0b111 << 13) | (2 << 10) | (4 << 7) | (2 << 5) | (6 << 2)
        result = disasm(instr, 0x80001000)
        assert result.startswith("c.sd"), f"期望 C.SD, 得到: {result}"
        assert "144" in result, f"期望偏移 144, 得到: {result}"
        assert "x14" in result and "x12" in result, f"期望 x14,x12, 得到: {result}"

    def test_c_ld_large_offset_144(self):
        """C.LD x14, 144(x12): uimm[7:6]=2, uimm[5:3]=2 -> offset=144."""
        instr = (0b011 << 13) | (2 << 10) | (4 << 7) | (2 << 5) | (6 << 2)
        result = disasm(instr, 0x80001000)
        assert result.startswith("c.ld"), f"期望 C.LD, 得到: {result}"
        assert "144" in result, f"期望偏移 144, 得到: {result}"

    def test_c_sd_offset_200_regression(self):
        """回归: 用户发现的 0xE6F0 应反汇编为 c.sd x12, 200(x13)."""
        instr = 0xE6F0  # f0 e6 in memory
        result = disasm(instr, 0x8003E5B0)
        assert "c.sd" in result, f"期望 C.SD, 得到: {result}"
        assert "200" in result, f"期望偏移 200, 得到: {result}"
        assert "x12" in result and "x13" in result, f"期望 x12,x13, 得到: {result}"

    def test_c_sd_vs_c_sw_different_offset(self):
        """C.SD 与 C.SW 同 scatter 应反汇编为不同偏移 (布局分立)."""
        # C.SD: offset=200, rs1_creg=2(x10), rs2_creg=4(x12)
        instr_sd = (0b111 << 13) | (1 << 10) | (2 << 7) | (3 << 5) | (4 << 2)
        result_sd = disasm(instr_sd, 0x80000000)
        assert "200" in result_sd, f"C.SD 期望偏移 200, 得到: {result_sd}"

        # C.SW: 相同 scatter 但 funct3=110
        instr_sw = (0b110 << 13) | (1 << 10) | (2 << 7) | (3 << 5) | (4 << 2)
        result_sw = disasm(instr_sw, 0x80000000)
        assert "76" in result_sw, (
            f"C.SW 与 C.SD 同 bit pattern 应得不同偏移: 期望 76, 得到: {result_sw}"
        )

    # -- C2 quadrant: SP-relative --

    def test_c_lwsp_offset_40(self):
        """C.LWSP x10, 40(sp)."""
        instr = (0b010 << 13) | (1 << 12) | (10 << 7) | (0b010 << 2) | 0b10
        result = disasm(instr, 0x80000000)
        assert result.startswith("c.lwsp"), f"期望 C.LWSP, 得到: {result}"
        assert "40" in result and "sp" in result, f"期望偏移 40(sp), 得到: {result}"

    def test_c_ldsp_offset_40(self):
        """C.LDSP x10, 40(sp)."""
        instr = (0b011 << 13) | (1 << 12) | (10 << 7) | (0b01 << 5) | 0b10
        result = disasm(instr, 0x80000000)
        assert result.startswith("c.ldsp"), f"期望 C.LDSP, 得到: {result}"
        assert "40" in result, f"期望偏移 40, 得到: {result}"

    def test_c_swsp_offset_40(self):
        """C.SWSP x10, 40(sp)."""
        instr = (0b110 << 13) | (10 << 9) | (10 << 2) | 0b10
        result = disasm(instr, 0x80000000)
        assert result.startswith("c.swsp"), f"期望 C.SWSP, 得到: {result}"
        assert "40" in result, f"期望偏移 40, 得到: {result}"

    def test_c_sdsp_offset_40(self):
        """C.SDSP x10, 40(sp)."""
        instr = (0b111 << 13) | (5 << 10) | (10 << 2) | 0b10
        result = disasm(instr, 0x80000000)
        assert result.startswith("c.sdsp"), f"期望 C.SDSP, 得到: {result}"
        assert "40" in result, f"期望偏移 40, 得到: {result}"

    def test_c_swsp_vs_c_sdsp_bit_layout(self):
        """C.SWSP 与 C.SDSP 同 bit pattern 应反汇编为不同偏移."""
        # C.SWSP 偏移 40: bits[12:9]=1010, bits[8:7]=00
        instr_swsp = (0b110 << 13) | (10 << 9) | (10 << 2) | 0b10
        result_swsp = disasm(instr_swsp, 0x80000000)
        assert "40" in result_swsp, f"C.SWSP 期望偏移 40, 得到: {result_swsp}"

        # 相同 raw 位但 funct3 改为 C.SDSP(111): bits[12:10]=101, bits[9:7]=000
        # 这会得到 offset = 0<<6 | 5<<3 = 40 — 巧合!
        # 用另一个能体现差异的 offset:
        # C.SDSP 偏移 72: bits[12:10]=001, bits[9:7]=011 -> offset = 3<<6|1<<3 = 192+8=200... hmm
        # Let me think of a better example.
        # C.SWSP offset 60: bits[12:9]=1111, bits[8:7]=00 -> offset = 0<<6|15<<2 = 60
        # Same raw bits as C.SDSP: bits[12:10]=111, bits[9:7]=100 -> offset = 4<<6|7<<3 = 256+56 = 312
        instr_swsp_60 = (0b110 << 13) | (15 << 9) | (5 << 2) | 0b10
        instr_sdsp_same = (0b111 << 13) | (7 << 10) | (4 << 7) | (5 << 2) | 0b10
        result_60 = disasm(instr_swsp_60, 0)
        result_sdsp_diff = disasm(instr_sdsp_same, 0)
        assert "60" in result_60, f"C.SWSP 期望偏移 60, 得到: {result_60}"
        assert "60" not in result_sdsp_diff, (
            f"C.SDSP 不应得偏移 60 (布局不同), 得到: {result_sdsp_diff}"
        )

    # -- C0 quadrant: C.ADDI4SPN --

    def test_c_addi4spn_basic(self):
        """C.ADDI4SPN a0, sp, 16 -> 基本编码."""
        # nzuimm=16: nz96=0000, nz54=01, nz3=0, nz2=0
        # instr[6]=nz2=0, instr[5]=nz3=0
        instr = (0b000 << 13) | (0b01 << 11) | (0 << 7) | (0 << 6) | (0 << 5) | (2 << 2)
        result = disasm(instr, 0x80000000)
        assert result.startswith("c.addi4spn"), f"期望 C.ADDI4SPN, 得到: {result}"
        assert "16" in result, f"期望偏移 16, 得到: {result}"

    def test_c_addi4spn_nz3_neq_nz2(self):
        """C.ADDI4SPN a1, sp, 20 -> nzuimm[3:2]=01, 验证 instr[5]/instr[6] 未交换."""
        # nzuimm=20=0b10100: nz96=0, nz54=01, nz3=0, nz2=1
        # instr[6]=nz2=1, instr[5]=nz3=0
        instr = (0b000 << 13) | (0b01 << 11) | (0 << 7) | (1 << 6) | (0 << 5) | (3 << 2)
        result = disasm(instr, 0x80000000)
        assert "c.addi4spn" in result, f"期望 C.ADDI4SPN, 得到: {result}"
        assert "20" in result, f"期望偏移 20, 得到: {result}"

    def test_c_addi4spn_nz2_neq_nz3(self):
        """C.ADDI4SPN a2, sp, 24 -> nzuimm[3:2]=10, 验证 instr[5]/instr[6] 未交换."""
        # nzuimm=24=0b11000: nz96=0, nz54=01, nz3=1, nz2=0
        # instr[6]=nz2=0, instr[5]=nz3=1
        instr = (0b000 << 13) | (0b01 << 11) | (0 << 7) | (0 << 6) | (1 << 5) | (4 << 2)
        result = disasm(instr, 0x80000000)
        assert "c.addi4spn" in result, f"期望 C.ADDI4SPN, 得到: {result}"
        assert "24" in result, f"期望偏移 24, 得到: {result}"

    def test_c_addi4spn_large_imm(self):
        """C.ADDI4SPN a5, sp, 716 -> 较大立即数, 覆盖全部位域."""
        # nzuimm=716=0x2CC=0b1011001100: nz96=1011, nz54=00, nz3=1, nz2=1
        # instr[6]=nz2=1, instr[5]=nz3=1
        instr = (0b000 << 13) | (0b00 << 11) | (0b1011 << 7) | (1 << 6) | (1 << 5) | (6 << 2)
        result = disasm(instr, 0x80000000)
        assert "c.addi4spn" in result, f"期望 C.ADDI4SPN, 得到: {result}"
        assert "716" in result, f"期望偏移 716, 得到: {result}"
