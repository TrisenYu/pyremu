#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""压缩指令差分测试 — 每条 C 指令与其等效的非压缩指令在行为上必须一致.

对每条 RV64C 指令, 按 RISC-V 规范构造正确的 16-bit 压缩编码,
与等价的 32-bit 非压缩编码从相同初始状态执行, 比对全部 GPR (x0–x31) 和内存副作用.
"""

import pytest

from pyremu.core.decoder import Hart, Opc, _sext
from pyremu.core.mem_check_aux import inject_memory_backend

# ============================================================
#  32-bit 标准指令编码辅助
# ============================================================

def _r_type(opcode: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int) -> int:
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _i_type(opcode: int, rd: int, funct3: int, rs1: int, imm12: int) -> int:
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _i_type_load(opcode: int, rd: int, funct3: int, rs1: int, imm12: int) -> int:
    """Load: funct3=width, imm12=unsigned offset."""
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _s_type(opcode: int, funct3: int, rs1: int, rs2: int, imm12: int) -> int:
    return (
        ((imm12 & 0xFE0) << 20)    # imm[11:5] -> bits[31:25]
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((imm12 & 0x1F) << 7)    # imm[4:0]  -> bits[11:7]
        | opcode
    )


def _b_type(opcode: int, funct3: int, rs1: int, rs2: int, offset: int) -> int:
    return (
        ((offset & 0x1000) << 19)  # imm[12] -> bit[31]
        | ((offset & 0x7E0) << 20)  # imm[10:5] -> bits[30:25]
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((offset & 0x1E) << 7)   # imm[4:1] -> bits[11:8]
        | ((offset & 0x800) >> 4)  # imm[11] -> bit[7]
        | opcode
    )


def _u_type(opcode: int, rd: int, imm20: int) -> int:
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode


def _j_type(opcode: int, rd: int, offset: int) -> int:
    return (
        ((offset & 0x100000) << 11)  # imm[20]   -> bit[31]
        | ((offset & 0x7FE) << 20)   # imm[10:1] -> bits[30:21]
        | ((offset & 0x800) << 9)    # imm[11]   -> bit[20]
        | ((offset & 0xFF000) << 0)  # imm[19:12]-> bits[19:12]
        | (rd << 7)
        | opcode
    )


# ============================================================
#  C 寄存器映射: 3-bit creg -> 实际寄存器号 (x8–x15)
# ============================================================

def _c_reg(creg: int) -> int:
    """Map 3-bit C register encoding (0-7) to actual register x8-x15."""
    return creg + 8


# ============================================================
#  C0 象限指令编码 (bits[1:0] = 00)
#  C0 格式: funct3[15:13] | scatter[12:2] | rd'[4:2] | 00
# ============================================================

def _c0(funct3: int, rd_creg: int, scatter_bits_12_to_2: int) -> int:
    """C0: funct3[15:13] | scatter[12:2] | rd'[4:2] | 00."""
    return (
        ((funct3 & 0x7) << 13)
        | (scatter_bits_12_to_2 & 0x1FFC)
        | ((rd_creg & 0x7) << 2)
        | 0b00
    )


# ============================================================
#  C1 象限指令编码 (bits[1:0] = 01)
#  C1 格式: funct3[15:13] | scatter[12:2] | rd/rs1[11:7] | 01
# ============================================================

def _c1(funct3: int, rd_5bit: int, scatter_bits_12_to_2: int) -> int:
    return (
        ((funct3 & 0x7) << 13)
        | (scatter_bits_12_to_2 & 0x1FFF)
        | ((rd_5bit & 0x1F) << 7)
        | 0b01
    )


# ============================================================
#  C2 象限指令编码 (bits[1:0] = 10)
#  C2 格式: funct3[15:13] | scatter[12:2] | rd/rs1[11:7] | 10
# ============================================================

def _c2(funct3: int, rd_5bit: int, scatter_bits_12_to_2: int) -> int:
    """C2: funct3[15:13] | scatter[12:2] | rd/rs1[11:7] | 10."""
    return (
        ((funct3 & 0x7) << 13)
        | (scatter_bits_12_to_2 & 0x1FFF)
        | ((rd_5bit & 0x1F) << 7)
        | 0b10
    )


# ============================================================
#  测试基础设施
# ============================================================

def _make_ram():
    """构造模拟物理内存."""
    ram = bytearray(2 * 1024 * 1024)

    def read_fn(addr, size):
        return bytes(ram[addr : addr + size])

    def write_fn(addr, data):
        for i, b in enumerate(data):
            ram[addr + i] = b

    return ram, read_fn, write_fn


def _snapshot_gprs(h: Hart) -> dict[int, int]:
    """快照全部 32 个 GPR."""
    return {i: h.gprs[i] for i in range(32)}


def _assert_gprs_equal(a: dict[int, int], b: dict[int, int], x0_may_differ: bool = True):
    """断言两组 GPR 值一致. x0 永远为 0 (硬件), 允许跳过."""
    for i in range(32):
        if x0_may_differ and i == 0:
            continue
        assert a[i] == b[i], (
            f"GPR x{i} differs: compressed={a[i]:#018x}, uncompressed={b[i]:#018x}"
        )


# ============================================================
#  测试类
# ============================================================

class TestCompressedDifferential:
    """差分测试: 每条 C 指令 vs 等价 32-bit 指令."""

    # ===========================================================
    #  C.ADDI (rd≠0)  ←->  ADDI rd, rd, imm
    #  Encoding: funct3=000, rd[11:7], imm[5]=bit12, imm[4:0]=bits[6:2]
    # ===========================================================
    @pytest.mark.parametrize("rd,init_val,imm", [
        (5, 10, 3),        # small positive
        (7, 0, -5),        # negative (imm sign-extended)
        (12, 100, 31),     # max positive 6-bit
        (8, 0, -32),       # max negative 6-bit
        (15, 0xFFFF_FFFF_FFFF_FFFE, 1),  # overflow wrap
    ])
    def test_c_addi_vs_addi(self, rd, init_val, imm):
        ram, rf, wf = _make_ram()
        imm6 = imm & 0x3F
        # C.ADDI: bit[12]=imm[5], bits[6:2]=imm[4:0]
        c_instr = _c1(0b000, rd, ((imm6 & 0x20) << 7) | ((imm6 & 0x1F) << 2))
        imm12 = _sext(imm, 6) & 0xFFF
        nc_instr = _i_type(Opc.opImm.value, rd, 0b000, rd, imm12)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[rd] = init_val
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[rd] = init_val
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.ADDIW (rd≠0)  ←->  ADDIW rd, rd, imm
    #  Encoding: funct3=001, rd[11:7], imm[5]=bit12, imm[4:0]=bits[6:2]
    # ===========================================================
    @pytest.mark.parametrize("rd,init_val,imm", [
        (5, 10, 3),
        (7, 0, -5),
        (8, 0xFFFF_FFFF_8000_0000, 1),  # 32-bit overflow -> sign-extend
        (9, 100, -32),
    ])
    def test_c_addiw_vs_addiw(self, rd, init_val, imm):
        ram, rf, wf = _make_ram()
        imm6 = imm & 0x3F
        c_instr = _c1(0b001, rd, ((imm6 & 0x20) << 7) | ((imm6 & 0x1F) << 2))
        imm12 = _sext(imm, 6) & 0xFFF
        nc_instr = _i_type(Opc.opImm32.value, rd, 0b000, rd, imm12)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[rd] = init_val
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[rd] = init_val
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.LI (rd≠0)  ←->  ADDI rd, x0, imm
    #  Encoding: funct3=010, rd[11:7], imm[5]=bit12, imm[4:0]=bits[6:2]
    # ===========================================================
    @pytest.mark.parametrize("rd,imm", [
        (5, 0), (7, -1), (12, 31), (8, -32), (15, 1),
    ])
    def test_c_li_vs_addi_x0(self, rd, imm):
        ram, rf, wf = _make_ram()
        imm6 = imm & 0x3F
        c_instr = _c1(0b010, rd, ((imm6 & 0x20) << 7) | ((imm6 & 0x1F) << 2))
        imm12 = _sext(imm, 6) & 0xFFF
        nc_instr = _i_type(Opc.opImm.value, rd, 0b000, 0, imm12)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.exec_instr(c_instr)
        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.exec_instr(nc_instr)
        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.LUI (rd∉{0,2}, nzimm≠0)  ←->  LUI rd, imm20
    #  Encoding: funct3=011, rd[11:7],
    #    imm[17]=bit12, imm[16:12]=bits[6:2]
    # ===========================================================
    @pytest.mark.parametrize("rd,imm6", [
        (5, 1), (8, -1), (12, 31), (3, 16), (15, -32),
    ])
    def test_c_lui_vs_lui(self, rd, imm6):
        ram, rf, wf = _make_ram()
        val6 = imm6 & 0x3F
        c_instr = _c1(0b011, rd, ((val6 & 0x20) << 7) | ((val6 & 0x1F) << 2))
        imm20 = _sext(imm6, 6) & 0xFFFFF
        nc_instr = _u_type(Opc.lui.value, rd, imm20)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.exec_instr(c_instr)
        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.exec_instr(nc_instr)
        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.ADDI16SP  ←->  ADDI x2, x2, nzimm
    #  6-bit nzimm[9:4] -> sign-extend to 10-bit (nzimm[3:0]=0)
    #  nzimm[9]=bit12, nzimm[8:7]=bits[4:3], nzimm[6]=bit5,
    #  nzimm[5]=bit2, nzimm[4]=bit6
    # ===========================================================
    @pytest.mark.parametrize("init_sp,nzimm10", [
        (0x1000, 16), (0x1000, -16), (0x1000, 496), (0x1000, -512),
        (0xFFFF_FFFF_FFFF_FF00, 256),
    ])
    def test_c_addi16sp_vs_addi_sp(self, init_sp, nzimm10):
        ram, rf, wf = _make_ram()
        nz = nzimm10 >> 4  # 6-bit: nzimm[9:4]
        if nz == 0:
            pytest.skip("C.ADDI16SP: nzuimm==0 is illegal")
        c_instr = _c1(0b011, 2, (
            ((nz & 0x01) << 6)    # nzimm[4]  ← instr[6]
            | ((nz & 0x02) << 1)   # nzimm[5]  ← instr[2]
            | ((nz & 0x04) << 3)   # nzimm[6]  ← instr[5]
            | (nz & 0x18)          # nzimm[8:7]← instr[4:3]
            | ((nz & 0x20) << 7)   # nzimm[9]  ← instr[12]
        ))
        nc_instr = _i_type(Opc.opImm.value, 2, 0b000, 2, nzimm10 & 0xFFF)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[2] = init_sp
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[2] = init_sp
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.ADDI4SPN  ←->  ADDI rd, x2, nzuimm
    #  C0 quadrant: funct3=000, rd'[4:2]
    #  nzuimm[9:6]=bits[10:7], nzuimm[5]=bit12, nzuimm[4]=bit11,
    #  nzuimm[3]=bit5, nzuimm[2]=bit6
    # ===========================================================
    @pytest.mark.parametrize("rd_creg,init_sp,nzuimm", [
        (3, 0x1000, 4),        # minimal (4 * 1)
        (5, 0x1000, 1020),     # max (4 * 255)
        (0, 0xFFFF_FFFF_FFFF_FFF0, 16),
    ])
    def test_c_addi4spn_vs_addi_sp(self, rd_creg, init_sp, nzuimm):
        ram, rf, wf = _make_ram()
        rd = _c_reg(rd_creg)
        # C.ADDI4SPN: nzuimm[5:4]->instr[12:11], nzuimm[9:6]->instr[10:7],
        #   nzuimm[2]->instr[6], nzuimm[3]->instr[5]
        c_instr = _c0(0b000, rd_creg, (
            ((nzuimm >> 2) & 0x1) << 6    # nzuimm[2] -> instr[6]
            | ((nzuimm >> 3) & 0x1) << 5   # nzuimm[3] -> instr[5]
            | ((nzuimm >> 4) & 0x1) << 11  # nzuimm[4] -> instr[11]
            | ((nzuimm >> 5) & 0x1) << 12  # nzuimm[5] -> instr[12]
            | ((nzuimm >> 6) & 0xF) << 7   # nzuimm[9:6] -> instr[10:7]
        ))
        nc_instr = _i_type(Opc.opImm.value, rd, 0b000, 2, nzuimm & 0xFFF)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[2] = init_sp
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[2] = init_sp
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.SRLI  ←->  SRLI rd, rd, shamt
    #  RV64C: sf=00, shamt = {bit12, bits[6:2]} (1-63)
    # ===========================================================
    @pytest.mark.parametrize("rd_creg,init_val,shamt", [
        (3, 0xDEAD_BEEF_0000_0000, 4),    # rd=x11, small shift, shamt[5]=0
        (5, 0x8000_0000_0000_0000, 1),    # MSB shift, shamt[5]=0
        (2, 0xFFFF_FFFF_FFFF_FFFF, 31),   # max shamt[5]=0
        (0, 0, 31),                       # zero shifted, shamt[5]=0
        #  shamt >= 32 (bit12=1) — 之前错误地 reject 了这些编码
        (1, 0x8000_0000_0000_0000, 32),   # shamt[5]=1
        (3, 0xF000_0000_0000_0000, 40),   # mid-range large shift
        (5, 0xFFFF_FFFF_FFFF_FFFF, 63),   # max shamt, shamt[5]=1
    ])
    def test_c_srli_vs_srli(self, rd_creg, init_val, shamt):
        ram, rf, wf = _make_ram()
        rd = _c_reg(rd_creg)
        # C.SRLI: funct3=100, sf=00, shamt = {bit12, bits[6:2]}
        c_instr = _c1(0b100, rd_creg,
            (0b00 << 10)                       # sf=00 = SRLI
            | ((shamt >> 5) << 12)             # bit12 = shamt[5]
            | ((shamt & 0x1F) << 2)            # bits[6:2] = shamt[4:0]
        )
        nc_instr = _i_type(Opc.opImm.value, rd, 0b101, rd, shamt & 0x3F)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[rd] = init_val
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[rd] = init_val
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.SRAI  ←->  SRAI rd, rd, shamt
    #  RV64C: sf=01, shamt = {bit12, bits[6:2]} (1-63)
    # ===========================================================
    @pytest.mark.parametrize("rd_creg,init_val,shamt", [
        #  shamt < 32 (bit12=0) — 之前错误地 reject 了这些编码
        (1, 0x8000_0000_0000_0000, 4),    # small arithmetic shift
        (3, 0xFFFF_FFFF_FFFF_FFFF, 1),    # shift -1 by 1
        (5, 0x0F00_0000_0000_0000, 31),   # max shamt[5]=0
        (0, 0, 31),                       # zero shifted
        #  shamt >= 32 (bit12=1)
        (3, 0x8000_0000_0000_0000, 36),   # 32+4: arithmetic shift MSB
        (5, 0xFFFF_FFFF_0000_0000, 48),   # 32+16: large shift
        (2, 0x0F00_0000_0000_0000, 63),   # 32+31: max shift
        (4, 0, 32),                       # minimum bit12=1
    ])
    def test_c_srai_vs_srai(self, rd_creg, init_val, shamt):
        ram, rf, wf = _make_ram()
        rd = _c_reg(rd_creg)
        # C.SRAI: funct3=100, sf=01, shamt = {bit12, bits[6:2]}
        c_instr = _c1(0b100, rd_creg,
            (0b01 << 10)                   # sf=01 = SRAI
            | ((shamt >> 5) << 12)         # bit12 = shamt[5]
            | ((shamt & 0x1F) << 2)        # bits[6:2] = shamt[4:0]
        )
        nc_instr = _i_type(Opc.opImm.value, rd, 0b101, rd, 0x400 | (shamt & 0x3F))

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[rd] = init_val
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[rd] = init_val
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.ANDI  ←->  ANDI rd, rd, imm
    #  sf=10, bit12=imm[5], bits[6:2]=imm[4:0]
    # ===========================================================
    @pytest.mark.parametrize("rd_creg,init_val,imm6", [
        (3, 0xFFFF_FFFF_FFFF_FFFF, -1),   # all ones & -1 = all ones
        (5, 0xABCD_EF01_2345_6789, 0xFF),  # isolate low byte
        (2, 0, 31),
        (7, 0xFFFF_FFFF_FFFF_FFFF, 0),    # clear
    ])
    def test_c_andi_vs_andi(self, rd_creg, init_val, imm6):
        ram, rf, wf = _make_ram()
        rd = _c_reg(rd_creg)
        val6 = imm6 & 0x3F
        c_instr = _c1(0b100, rd_creg,
            (0b10 << 10)                    # sf=10 = C.ANDI
            | ((val6 & 0x20) << 7)          # imm[5] -> bit12
            | ((val6 & 0x1F) << 2)          # imm[4:0] -> bits[6:2]
        )
        imm12 = _sext(imm6, 6) & 0xFFF
        nc_instr = _i_type(Opc.opImm.value, rd, 0b111, rd, imm12)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[rd] = init_val
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[rd] = init_val
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.SUB / C.XOR / C.OR / C.AND  ←->  sub/xor/or/and
    #  RV64C: sf=01, bit[6:5]=op (00=SUB,01=XOR,10=OR,11=AND)
    # ===========================================================
    @pytest.mark.parametrize("rd_creg,rs2_creg,v1,v2,op_bits", [
        (3, 5, 100, 30, 0b00),              # SUB
        (3, 5, 0xFFFF, 0xAAAA, 0b01),       # XOR
        (3, 5, 0xF0F0, 0x0F0F, 0b10),       # OR
        (3, 5, 0xFFFF, 0x8000, 0b11),       # AND
        (0, 1, 0, 0, 0b00),                 # SUB: 0-0=0
        (2, 4, 5, 10, 0b00),                # SUB: negative result
    ])
    def test_c_alu_vs_alu(self, rd_creg, rs2_creg, v1, v2, op_bits):
        ram, rf, wf = _make_ram()
        rd = _c_reg(rd_creg)
        rs2 = _c_reg(rs2_creg)
        # LLVM encoding: sf=11, bit12=0 for C.SUB/C.XOR/C.OR/C.AND
        c_instr = _c1(0b100, rd_creg,
            (0b11 << 10)                      # sf=11
            | (0 << 12)                       # bit12=0
            | (op_bits << 5)                  # op -> bits[6:5]
            | (rs2_creg << 2)                 # rs2' -> bits[4:2]
        )

        funct3_map = {0b00: 0b000, 0b01: 0b100, 0b10: 0b110, 0b11: 0b111}
        funct7 = 0b0100000 if op_bits == 0b00 else 0b0000000
        nc_instr = _r_type(Opc.op.value, rd, funct3_map[op_bits], rd, rs2, funct7)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[rd] = v1
        hc.gprs[rs2] = v2
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[rd] = v1
        hn.gprs[rs2] = v2
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.SUBW / C.ADDW  ←->  subw / addw
    #  RV64C only: sf=11, bit[6:5]=00->SUBW, 01->ADDW
    # ===========================================================
    @pytest.mark.parametrize("rd_creg,rs2_creg,v1,v2,op_bit5", [
        (3, 5, 100, 30, 0),                  # SUBW
        (1, 7, 0xFFFF_FFFF_8000_0000, 1, 1), # ADDW: overflow wraps in 32-bit
        (0, 2, 5, 10, 0),                    # SUBW: negative result (sign-extended)
        (4, 6, 10, 10, 0),                   # SUBW: 0 result
        (3, 5, 0x1_0000_0001, 1, 1),        # ADDW: 32-bit wrap to 2 -> sext32
    ])
    def test_c_subw_addw_vs_32bit(self, rd_creg, rs2_creg, v1, v2, op_bit5):
        ram, rf, wf = _make_ram()
        rd = _c_reg(rd_creg)
        rs2 = _c_reg(rs2_creg)
        # LLVM encoding: sf=11, bit12=1 for C.SUBW/C.ADDW
        c_instr = _c1(0b100, rd_creg,
            (0b11 << 10)                      # sf=11
            | (1 << 12)                       # bit12=1 (RV64C)
            | (op_bit5 << 5)                  # 0=SUBW, 1=ADDW -> bit[5]
            | (0 << 6)                        # bit[6]=0
            | (rs2_creg << 2)                 # rs2' -> bits[4:2]
        )

        if op_bit5 == 0:
            # SUBW: funct3=000, funct7=0b0100000
            nc_instr = _r_type(Opc.op32.value, rd, 0b000, rd, rs2, 0b0100000)
        else:
            # ADDW: funct3=000, funct7=0b0000000
            nc_instr = _r_type(Opc.op32.value, rd, 0b000, rd, rs2, 0b0000000)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[rd] = v1
        hc.gprs[rs2] = v2
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[rd] = v1
        hn.gprs[rs2] = v2
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.LW  ←->  LW rd, uimm(rs1)
    #  C0: funct3=010, uimm[6]=bit6, uimm[5:3]=bits[12:10], uimm[2]=bit5
    # ===========================================================
    @pytest.mark.parametrize("rd_creg,rs1_creg,init_rs1,uimm,mem_val", [
        (3, 5, 0x1000, 0, 0xDEAD_BEEF),
        (0, 7, 0x1000, 4, 0x0102_0304),
        (7, 1, 0x0FFC, 0, 0xFFFF_FFFF),      # sign-extend boundary
        (3, 2, 0x2000, 0x7C, 0xCAFE_BABE),   # max offset: uimm[6]=1, uimm[5:3]=7, uimm[2]=0
    ])
    def test_c_lw_vs_lw(self, rd_creg, rs1_creg, init_rs1, uimm, mem_val):
        ram, rf, wf = _make_ram()
        rd = _c_reg(rd_creg)
        rs1 = _c_reg(rs1_creg)
        addr = init_rs1 + uimm
        ram[addr:addr+4] = mem_val.to_bytes(4, 'little')

        # C.LW: uimm[5:3]=bits[12:10], uimm[2]=bit6, uimm[6]=bit5
        c_instr = _c0(0b010, rd_creg, (
            (rs1_creg << 7)                   # rs1' in bits[9:7]
            | ((uimm >> 2) & 0x1) << 6        # uimm[2] -> bit[6]
            | ((uimm >> 3) & 0x7) << 10       # uimm[5:3] -> bits[12:10]
            | ((uimm >> 6) & 0x1) << 5        # uimm[6] -> bit[5]
        ))
        nc_instr = _i_type_load(Opc.ld.value, rd, 0b010, rs1, uimm & 0xFFF)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[rs1] = init_rs1
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[rs1] = init_rs1
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.SW  ←->  SW rs2, uimm(rs1)
    #  C0: funct3=110, uimm same layout as C.LW, rs2' at bits[4:2]
    # ===========================================================
    @pytest.mark.parametrize("rs2_creg,rs1_creg,init_rs1,init_rs2,uimm", [
        (3, 5, 0x1000, 0xCAFE_BABE, 0),
        (7, 1, 0x2000, 0x0102_0304, 4),
        (5, 3, 0x3000, 0xDEAD_BEEF, 0x7C),   # max offset
    ])
    def test_c_sw_vs_sw(self, rs2_creg, rs1_creg, init_rs1, init_rs2, uimm):
        ram, rf, wf = _make_ram()
        rs2 = _c_reg(rs2_creg)
        rs1 = _c_reg(rs1_creg)

        # C.SW: funct3=110, uimm same layout as C.LW
        c_instr = _c0(0b110, rs2_creg, (       # rd field = rs2' for C.SW
            (rs1_creg << 7)
            | ((uimm >> 2) & 0x1) << 6         # uimm[2] -> bit[6]
            | ((uimm >> 3) & 0x7) << 10        # uimm[5:3] -> bits[12:10]
            | ((uimm >> 6) & 0x1) << 5         # uimm[6] -> bit[5]
        ))
        nc_instr = _s_type(Opc.st.value, 0b010, rs1, rs2, uimm & 0xFFF)

        ram_c, rf_c, wf_c = _make_ram()
        hc = Hart(id=0)
        inject_memory_backend(hc, rf_c, wf_c)
        hc.gprs[rs1] = init_rs1
        hc.gprs[rs2] = init_rs2
        hc.exec_instr(c_instr)

        ram_n, rf_n, wf_n = _make_ram()
        hn = Hart(id=0)
        inject_memory_backend(hn, rf_n, wf_n)
        hn.gprs[rs1] = init_rs1
        hn.gprs[rs2] = init_rs2
        hn.exec_instr(nc_instr)

        addr = init_rs1 + uimm
        assert ram_c[addr:addr+4] == ram_n[addr:addr+4], (
            f"Memory differs at {addr:#x}: "
            f"C={ram_c[addr:addr+4].hex()}, NC={ram_n[addr:addr+4].hex()}"
        )
        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.LD  ←->  LD rd, uimm(rs1)
    #  C0: funct3=011, uimm[7:6]=bits[6:5], uimm[5:3]=bits[12:10]
    # ===========================================================
    @pytest.mark.parametrize("rd_creg,rs1_creg,init_rs1,uimm,mem_val", [
        (3, 5, 0x1000, 0, 0xDEAD_BEEF_CAFE_BABE),
        (0, 7, 0x1000, 8, 0x0102_0304_0506_0708),
        (2, 1, 0x2000, 0xF8, 0xAAAA_BBBB_CCCC_DDDD),  # max offset
    ])
    def test_c_ld_vs_ld(self, rd_creg, rs1_creg, init_rs1, uimm, mem_val):
        ram, rf, wf = _make_ram()
        rd = _c_reg(rd_creg)
        rs1 = _c_reg(rs1_creg)
        addr = init_rs1 + uimm
        ram[addr:addr+8] = mem_val.to_bytes(8, 'little')

        # C.LD: uimm[7:6]=bits[6:5], uimm[5:3]=bits[12:10]
        c_instr = _c0(0b011, rd_creg, (
            (rs1_creg << 7)
            | ((uimm >> 6) & 0x3) << 5         # uimm[7:6] -> bits[6:5]
            | ((uimm >> 3) & 0x7) << 10        # uimm[5:3] -> bits[12:10]
        ))
        nc_instr = _i_type_load(Opc.ld.value, rd, 0b011, rs1, uimm & 0xFFF)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[rs1] = init_rs1
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[rs1] = init_rs1
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.SD  ←->  SD rs2, uimm(rs1)
    #  C0: funct3=111, uimm same layout as C.LD, rs2' at bits[4:2]
    # ===========================================================
    @pytest.mark.parametrize("rs2_creg,rs1_creg,init_rs1,init_rs2,uimm", [
        (3, 5, 0x1000, 0xDEAD_BEEF_CAFE_BABE, 0),
        (7, 1, 0x2000, 0x0102_0304_0506_0708, 8),
        (4, 2, 0x3000, 0xBBBB_AAAA_DDDD_CCCC, 0xF8),  # max offset
    ])
    def test_c_sd_vs_sd(self, rs2_creg, rs1_creg, init_rs1, init_rs2, uimm):
        ram, rf, wf = _make_ram()
        rs2 = _c_reg(rs2_creg)
        rs1 = _c_reg(rs1_creg)

        # C.SD: funct3=111, uimm same layout as C.LD
        c_instr = _c0(0b111, rs2_creg, (
            (rs1_creg << 7)
            | ((uimm >> 6) & 0x3) << 5
            | ((uimm >> 3) & 0x7) << 10
        ))
        nc_instr = _s_type(Opc.st.value, 0b011, rs1, rs2, uimm & 0xFFF)

        ram_c, rf_c, wf_c = _make_ram()
        hc = Hart(id=0)
        inject_memory_backend(hc, rf_c, wf_c)
        hc.gprs[rs1] = init_rs1
        hc.gprs[rs2] = init_rs2
        hc.exec_instr(c_instr)

        ram_n, rf_n, wf_n = _make_ram()
        hn = Hart(id=0)
        inject_memory_backend(hn, rf_n, wf_n)
        hn.gprs[rs1] = init_rs1
        hn.gprs[rs2] = init_rs2
        hn.exec_instr(nc_instr)

        addr = init_rs1 + uimm
        assert ram_c[addr:addr+8] == ram_n[addr:addr+8], (
            f"Memory differs at {addr:#x}: "
            f"C={ram_c[addr:addr+8].hex()}, NC={ram_n[addr:addr+8].hex()}"
        )
        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.SLLI  ←->  SLLI rd, rd, shamt
    #  C2: funct3=000, rd[11:7], bit12=shamt[5], bits[6:2]=shamt[4:0]
    # ===========================================================
    @pytest.mark.parametrize("rd,init_val,shamt", [
        (5, 1, 0), (5, 1, 31), (8, 0x8000_0000_0000_0001, 1),
        (12, 0xFFFF_FFFF_FFFF_FFFF, 63),
    ])
    def test_c_slli_vs_slli(self, rd, init_val, shamt):
        ram, rf, wf = _make_ram()
        c_instr = _c2(0b000, rd,
            ((shamt & 0x20) << 7)               # shamt[5] -> bit12
            | ((shamt & 0x1F) << 2)             # shamt[4:0] -> bits[6:2]
        )
        nc_instr = _i_type(Opc.opImm.value, rd, 0b001, rd, shamt & 0x3F)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[rd] = init_val
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[rd] = init_val
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.MV (rd≠0, rs2≠0)  ←->  ADD rd, x0, rs2
    #  C2: funct3=100, bit12=0, rd[11:7], rs2[6:2]
    # ===========================================================
    @pytest.mark.parametrize("rd,rs2,init_rs2", [
        (5, 8, 100), (7, 12, 0), (15, 3, -1), (10, 10, 0xDEAD),
    ])
    def test_c_mv_vs_add_x0(self, rd, rs2, init_rs2):
        ram, rf, wf = _make_ram()
        c_instr = _c2(0b100, rd,
            (0 << 12)                           # bit12=0 -> C.MV
            | (rs2 << 2)                        # rs2 -> bits[6:2]
        )
        nc_instr = _r_type(Opc.op.value, rd, 0b000, 0, rs2, 0b0000000)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[rs2] = init_rs2
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[rs2] = init_rs2
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.JR (rs1≠0)  ←->  JALR x0, rs1, 0
    #  C2: funct3=100, bit12=0, rs2=0
    #  Note: C.JR uses rs1 in bits[11:7], not _creg
    # ===========================================================
    @pytest.mark.parametrize("rs1,target", [
        (5, 0x100), (8, 0), (12, 0xFFFF_FFFF_FFFF_FFFE),  # PC & ~1 clears LSB
    ])
    def test_c_jr_vs_jalr_x0(self, rs1, target):
        ram, rf, wf = _make_ram()
        c_instr = _c2(0b100, rs1,
            (0 << 12) | (0 << 2)                # bit12=0, rs2=0 -> C.JR
        )
        nc_instr = _i_type(Opc.jalr.value, 0, 0b000, rs1, 0)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.pc = 0x8000_0000
        hc.gprs[rs1] = target
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.pc = 0x8000_0000
        hn.gprs[rs1] = target
        hn.exec_instr(nc_instr)

        assert hc.pc == hn.pc, f"PC differs: C={hc.pc:#x}, NC={hn.pc:#x}"
        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.JALR (rs1≠0)  ←->  JALR ra, rs1, 0
    #  C2: funct3=100, bit12=1, rs2=0
    #  Note: C.JALR saves pc+2 to ra. JALR saves pc+4 to ra.
    #  These WILL differ — skip x1 in GPR comparison.
    # ===========================================================
    @pytest.mark.parametrize("rs1,target", [
        (5, 0x200), (8, 0),
    ])
    def test_c_jalr_vs_jalr_ra(self, rs1, target):
        ram, rf, wf = _make_ram()
        c_instr = _c2(0b100, rs1,
            (1 << 12) | (0 << 2)                # bit12=1 -> C.JALR, rs2=0
        )
        nc_instr = _i_type(Opc.jalr.value, 1, 0b000, rs1, 0)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.pc = 0x8000_0000
        hc.gprs[rs1] = target
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.pc = 0x8000_0000
        hn.gprs[rs1] = target
        hn.exec_instr(nc_instr)

        assert hc.pc == hn.pc, f"PC differs: C={hc.pc:#x}, NC={hn.pc:#x}"
        # C.JALR saves pc+2 to ra; JALR saves pc+4.
        # Compare all GPRs EXCEPT x1 (ra).
        for i in range(32):
            if i == 0 or i == 1:
                continue
            assert hc.gprs[i] == hn.gprs[i], (
                f"GPR x{i} differs: C={hc.gprs[i]:#018x}, NC={hn.gprs[i]:#018x}"
            )

    # ===========================================================
    #  C.LWSP (rd≠0)  ←->  LW rd, uimm(x2)
    #  C2: funct3=010, rd[11:7]
    #  uimm[7:6]=bits[6:5], uimm[5]=bit12, uimm[4:2]=bits[4:2]
    # ===========================================================
    @pytest.mark.parametrize("rd,init_sp,uimm,mem_val", [
        (5, 0x1000, 0, 0xCAFE_BABE),
        (8, 0x1000, 4, 0x0102_0304),
        (7, 0x0FFC, 0, 0xFFFF_FFFF),           # sign-extend
        (3, 0x2000, 0xFC, 0x1234_5678),        # large offset
    ])
    def test_c_lwsp_vs_lw_sp(self, rd, init_sp, uimm, mem_val):
        ram, rf, wf = _make_ram()
        addr = init_sp + uimm
        ram[addr:addr+4] = mem_val.to_bytes(4, 'little')

        # C.LWSP: uimm[7:6]=bits[6:5], uimm[5]=bit12, uimm[4:2]=bits[4:2]
        c_instr = _c2(0b010, rd, (
            ((uimm >> 2) & 0x7) << 2            # uimm[4:2] -> bits[4:2]
            | ((uimm >> 5) & 0x1) << 12         # uimm[5] -> bit12
            | ((uimm >> 6) & 0x3) << 5          # uimm[7:6] -> bits[6:5]
        ))
        nc_instr = _i_type_load(Opc.ld.value, rd, 0b010, 2, uimm & 0xFFF)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[2] = init_sp
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[2] = init_sp
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.LDSP  ←->  LD rd, uimm(x2)
    #  C2: funct3=011, rd[11:7]
    #  uimm[8:6]=bits[4:2], uimm[5]=bit12, uimm[4:3]=bits[6:5]
    # ===========================================================
    @pytest.mark.parametrize("rd,init_sp,uimm,mem_val", [
        (5, 0x1000, 0, 0xDEAD_BEEF_CAFE_BABE),
        (8, 0x1000, 8, 0x0102_0304_0506_0708),
        (3, 0x2000, 0x1F8, 0xAAAA_BBBB_CCCC_DDDD),  # large offset
    ])
    def test_c_ldsp_vs_ld_sp(self, rd, init_sp, uimm, mem_val):
        ram, rf, wf = _make_ram()
        addr = init_sp + uimm
        ram[addr:addr+8] = mem_val.to_bytes(8, 'little')

        # C.LDSP: uimm[8:6]=bits[4:2], uimm[5]=bit12, uimm[4:3]=bits[6:5]
        c_instr = _c2(0b011, rd, (
            ((uimm >> 6) & 0x7) << 2             # uimm[8:6] -> bits[4:2]
            | ((uimm >> 5) & 0x1) << 12          # uimm[5] -> bit12
            | ((uimm >> 3) & 0x3) << 5           # uimm[4:3] -> bits[6:5]
        ))
        nc_instr = _i_type_load(Opc.ld.value, rd, 0b011, 2, uimm & 0xFFF)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.gprs[2] = init_sp
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.gprs[2] = init_sp
        hn.exec_instr(nc_instr)

        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.SWSP  ←->  SW rs2, uimm(x2)
    #  C2: funct3=110, rs2[6:2]
    #  uimm[7:6]=bits[8:7], uimm[5:2]=bits[12:9]
    #  Note: rs2 is in bits[6:2], NOT bits[11:7]!
    # ===========================================================
    @pytest.mark.parametrize("rs2,init_sp,init_rs2,uimm", [
        (5, 0x1000, 0xCAFE_BABE, 0),
        (8, 0x2000, 0x0102_0304, 4),
        (3, 0x3000, 0xDEAD_BEEF, 0xFC),        # large offset
    ])
    def test_c_swsp_vs_sw_sp(self, rs2, init_sp, init_rs2, uimm):
        ram, rf, wf = _make_ram()

        # C.SWSP: uimm[7:6]=instr[8:7], uimm[5:2]=instr[12:9], rs2=instr[6:2]
        c_instr = (
            (0b110 << 13)                       # funct3 [15:13]
            | ((uimm >> 2) & 0xF) << 9          # uimm[5:2] -> bits[12:9]
            | ((uimm >> 6) & 0x3) << 7          # uimm[7:6] -> bits[8:7]
            | (rs2 & 0x1F) << 2                 # rs2 -> bits[6:2]
            | 0b10                              # quadrant C2
        )
        nc_instr = _s_type(Opc.st.value, 0b010, 2, rs2, uimm & 0xFFF)

        ram_c, rf_c, wf_c = _make_ram()
        hc = Hart(id=0)
        inject_memory_backend(hc, rf_c, wf_c)
        hc.gprs[2] = init_sp
        hc.gprs[rs2] = init_rs2
        hc.exec_instr(c_instr)

        ram_n, rf_n, wf_n = _make_ram()
        hn = Hart(id=0)
        inject_memory_backend(hn, rf_n, wf_n)
        hn.gprs[2] = init_sp
        hn.gprs[rs2] = init_rs2
        hn.exec_instr(nc_instr)

        addr = init_sp + uimm
        assert ram_c[addr:addr+4] == ram_n[addr:addr+4], (
            f"Memory differs at {addr:#x}: "
            f"C={ram_c[addr:addr+4].hex()}, NC={ram_n[addr:addr+4].hex()}"
        )
        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.SDSP  ←->  SD rs2, uimm(x2)
    #  C2: funct3=111, rs2[6:2]
    #  uimm[8:6]=bits[9:7], uimm[5:3]=bits[12:10]
    #  Note: rs2 is in bits[6:2], NOT bits[11:7]!
    # ===========================================================
    @pytest.mark.parametrize("rs2,init_sp,init_rs2,uimm", [
        (5, 0x1000, 0xDEAD_BEEF_CAFE_BABE, 0),
        (8, 0x2000, 0x0102_0304_0506_0708, 8),
        (3, 0x3000, 0xBBBB_AAAA_DDDD_CCCC, 0x1F8),  # large offset
    ])
    def test_c_sdsp_vs_sd_sp(self, rs2, init_sp, init_rs2, uimm):
        ram, rf, wf = _make_ram()

        # C.SDSP: uimm[8:6]=instr[9:7], uimm[5:3]=instr[12:10], rs2=instr[6:2]
        c_instr = (
            (0b111 << 13)                       # funct3 [15:13]
            | ((uimm >> 3) & 0x7) << 10         # uimm[5:3] -> bits[12:10]
            | ((uimm >> 6) & 0x7) << 7          # uimm[8:6] -> bits[9:7]
            | (rs2 & 0x1F) << 2                 # rs2 -> bits[6:2]
            | 0b10                              # quadrant C2
        )
        nc_instr = _s_type(Opc.st.value, 0b011, 2, rs2, uimm & 0xFFF)

        ram_c, rf_c, wf_c = _make_ram()
        hc = Hart(id=0)
        inject_memory_backend(hc, rf_c, wf_c)
        hc.gprs[2] = init_sp
        hc.gprs[rs2] = init_rs2
        hc.exec_instr(c_instr)

        ram_n, rf_n, wf_n = _make_ram()
        hn = Hart(id=0)
        inject_memory_backend(hn, rf_n, wf_n)
        hn.gprs[2] = init_sp
        hn.gprs[rs2] = init_rs2
        hn.exec_instr(nc_instr)

        addr = init_sp + uimm
        assert ram_c[addr:addr+8] == ram_n[addr:addr+8], (
            f"Memory differs at {addr:#x}: "
            f"C={ram_c[addr:addr+8].hex()}, NC={ram_n[addr:addr+8].hex()}"
        )
        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.J  ←->  JAL x0, offset
    #  C1: funct3=101, offset[11]=bit12, offset[10]=bit8,
    #      offset[9:8]=bits[10:9], offset[7]=bit6, offset[6]=bit7,
    #      offset[5]=bit2, offset[4]=bit11, offset[3:1]=bits[5:3]
    # ===========================================================
    @pytest.mark.parametrize("init_pc,offset", [
        (0x8000_0000, 4), (0x8000_0000, -2), (0x8000_0000, 2046),
        (0x8000_0000, -2048),
    ])
    def test_c_j_vs_jal_x0(self, init_pc, offset):
        ram, rf, wf = _make_ram()
        # Spec encoding: offset[11]=bit12, offset[10]=bit8,
        #   offset[9:8]=bits[10:9], offset[7]=bit6, offset[6]=bit7,
        #   offset[5]=bit2, offset[4]=bit11, offset[3:1]=bits[5:3]
        c_instr = _c1(0b101, 0, (
            ((offset >> 11) & 0x1) << 12         # offset[11] -> bit[12]
            | ((offset >> 4) & 0x1) << 11        # offset[4] -> bit[11]
            | ((offset >> 8) & 0x3) << 9         # offset[9:8] -> bits[10:9]
            | ((offset >> 10) & 0x1) << 8        # offset[10] -> bit[8]
            | ((offset >> 6) & 0x1) << 7         # offset[6] -> bit[7]
            | ((offset >> 7) & 0x1) << 6         # offset[7] -> bit[6]
            | ((offset >> 1) & 0x7) << 3         # offset[3:1] -> bits[5:3]
            | ((offset >> 5) & 0x1) << 2         # offset[5] -> bit[2]
        ))
        nc_instr = _j_type(Opc.jal.value, 0, offset)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.pc = init_pc
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.pc = init_pc
        hn.exec_instr(nc_instr)

        assert hc.pc == hn.pc, f"PC differs: C={hc.pc:#x}, NC={hn.pc:#x}"
        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.BEQZ  ←->  BEQ rs1, x0, offset
    #  C1: funct3=110, rs1' in bits[9:7] (C-register)
    #  offset[8]=bit12, offset[4:3]=bits[11:10], offset[7:6]=bits[6:5],
    #  offset[2:1]=bits[4:3], offset[5]=bit2
    # ===========================================================
    @pytest.mark.parametrize("rs1_creg,init_rs1,init_pc,offset", [
        (3, 0, 0x8000_0000, 4),       # rs1==0 -> taken, positive
        (5, 1, 0x8000_0000, 8),       # rs1≠0 -> not taken
        (7, 0, 0x8000_0000, -16),     # taken, negative offset
        (0, 0xFFFF_FFFF_FFFF_FFFF, 0x8000_0000, 128),  # not taken, rs1≠0
    ])
    def test_c_beqz_vs_beq_x0(self, rs1_creg, init_rs1, init_pc, offset):
        ram, rf, wf = _make_ram()
        rs1 = _c_reg(rs1_creg)
        # C.BEQZ encoding per spec:
        c_instr = _c1(0b110, rs1_creg, (          # rd field = rs1' for BEQZ
            ((offset >> 8) & 0x1) << 12            # offset[8] -> bit[12]
            | ((offset >> 3) & 0x3) << 10          # offset[4:3] -> bits[11:10]
            | ((offset >> 6) & 0x3) << 5           # offset[7:6] -> bits[6:5]
            | ((offset >> 1) & 0x3) << 3           # offset[2:1] -> bits[4:3]
            | ((offset >> 5) & 0x1) << 2           # offset[5] -> bit[2]
        ))
        nc_instr = _b_type(Opc.br.value, 0b000, rs1, 0, offset)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.pc = init_pc
        hc.gprs[rs1] = init_rs1
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.pc = init_pc
        hn.gprs[rs1] = init_rs1
        hn.exec_instr(nc_instr)

        assert hc.pc == hn.pc, f"PC differs: C={hc.pc:#x}, NC={hn.pc:#x}"
        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))

    # ===========================================================
    #  C.BNEZ  ←->  BNE rs1, x0, offset
    #  C1: funct3=111, same offset encoding as C.BEQZ
    # ===========================================================
    @pytest.mark.parametrize("rs1_creg,init_rs1,init_pc,offset", [
        (3, 5, 0x8000_0000, 8),       # rs1≠0 -> taken
        (5, 0, 0x8000_0000, 4),       # rs1==0 -> not taken
        (7, -1, 0x8000_0000, -32),    # taken, negative offset
    ])
    def test_c_bnez_vs_bne_x0(self, rs1_creg, init_rs1, init_pc, offset):
        ram, rf, wf = _make_ram()
        rs1 = _c_reg(rs1_creg)
        # C.BNEZ: same encoding as C.BEQZ, just funct3=111
        c_instr = _c1(0b111, rs1_creg, (
            ((offset >> 8) & 0x1) << 12
            | ((offset >> 3) & 0x3) << 10
            | ((offset >> 6) & 0x3) << 5
            | ((offset >> 1) & 0x3) << 3
            | ((offset >> 5) & 0x1) << 2
        ))
        nc_instr = _b_type(Opc.br.value, 0b001, rs1, 0, offset)

        hc = Hart(id=0)
        inject_memory_backend(hc, rf, wf)
        hc.pc = init_pc
        hc.gprs[rs1] = init_rs1
        hc.exec_instr(c_instr)

        hn = Hart(id=0)
        inject_memory_backend(hn, rf, wf)
        hn.pc = init_pc
        hn.gprs[rs1] = init_rs1
        hn.exec_instr(nc_instr)

        assert hc.pc == hn.pc, f"PC differs: C={hc.pc:#x}, NC={hn.pc:#x}"
        _assert_gprs_equal(_snapshot_gprs(hc), _snapshot_gprs(hn))
