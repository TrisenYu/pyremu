#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""压缩指令 (C-extension) 测试 — 使用程序化计算的 RV64C 指令编码."""

import pytest

from pyremu.core.decoder import Hart


def _make_ram():
    """构造模拟物理内存."""
    ram = bytearray(2 * 1024 * 1024)

    def read_fn(addr, size):
        return bytes(ram[addr : addr + size])

    def write_fn(addr, data):
        for i, b in enumerate(data):
            ram[addr + i] = b

    return ram, read_fn, write_fn


# ============================================================
#  编码辅助: C1/C2 共用 bits[11:7] 为 rd 或 rs1 字段
#  C0 使用 3-bit 压缩寄存器号 (rd'/rs1'/rs2' ∈ [0,7] → x8–x15)
# ============================================================

def _c0(funct3: int, rd_creg: int, rs1_creg: int, scatter: int) -> int:
    """构造 C0 象限指令 (低 2 位 = 00)."""
    return ((funct3 & 0x7) << 13) | (scatter & 0x3FFF) | ((rd_creg & 0x7) << 2) | 0b00


def _c1(funct3: int, rd: int, scatter: int) -> int:
    """构造 C1 象限指令 (低 2 位 = 01). rd 在 bits[11:7]."""
    return ((funct3 & 0x7) << 13) | (scatter & 0x1FFF) | ((rd & 0x1F) << 7) | 0b01


def _c2(funct3: int, rd_rs1: int, scatter: int) -> int:
    """构造 C2 象限指令 (低 2 位 = 10). rd/rs1 在 bits[11:7]."""
    return ((funct3 & 0x7) << 13) | (scatter & 0x1FFF) | ((rd_rs1 & 0x1F) << 7) | 0b10


# ============================================================
#  C1 象限测试 — 寄存器直接操作、立即数、分支、跳转
# ============================================================


class TestCompressedC1:
    """C1 象限 (低 2 位 = 01)."""

    def test_c_addi(self):
        """C.ADDI x5, 3 → x5 += 3."""
        h = Hart(id=0)
        h.gprs[5].val = 10
        # imm[4:0]=3 在 bits[6:2], imm[5]=0
        instr = _c1(0b000, 5, (0 << 5) | (3 << 2))
        h.exec_instr(instr)
        assert h.gprs[5].val == 13

    def test_c_addiw(self):
        """C.ADDIW x5, -2 → 对 32-bit 做加法再符号扩展 (RV64C)."""
        h = Hart(id=0)
        h.gprs[5].val = 10
        imm = 0x3E  # 6-bit -2: [5]=1, [4:0]=30
        instr = _c1(0b001, 5, ((imm & 0x20) << 7) | ((imm & 0x1F) << 2))
        h.exec_instr(instr)
        assert h.gprs[5].val == 8

    def test_c_li(self):
        """C.LI x5, -1 → x5 = -1."""
        h = Hart(id=0)
        # imm=-1: bit[12]=1, bits[6:2]=31
        instr = _c1(0b010, 5, (1 << 12) | (31 << 2))
        h.exec_instr(instr)
        assert h.gprs[5].val == 0xFFFF_FFFF_FFFF_FFFF

    def test_c_mv(self):
        """C.MV x10, x5 → x10 = x5."""
        h = Hart(id=0)
        h.gprs[5].val = 0xCAFE
        instr = _c2(0b100, 10, 5 << 2)
        h.exec_instr(instr)
        assert h.gprs[10].val == 0xCAFE

    def test_c_jr(self):
        """C.JR x5 → pc = x5."""
        h = Hart(id=0)
        h.gprs[5].val = 0x3000
        h.pc = 0x1000
        instr = _c2(0b100, 5, 0)
        adv = h.exec_instr(instr)
        assert adv == 0  # PC 已被修改
        assert h.pc == 0x3000

    def test_c_jalr(self):
        """C.JALR x5 → ra=pc+2, pc=x5."""
        h = Hart(id=0)
        h.gprs[5].val = 0x4000
        h.pc = 0x1000
        # bit12=1 表示 JALR
        instr = _c2(0b100, 5, 1 << 12)
        h.exec_instr(instr)
        assert h.gprs[1].val == 0x1002  # ra
        assert h.pc == 0x4000

    def test_c_ebreak(self):
        """C.EBREAK → Breakpoint trap."""
        h = Hart(id=0)
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        instr = _c2(0b100, 0, 0)
        h.exec_instr(instr)
        assert h.mcause_val == 3

    def test_c_beqz_taken(self):
        """C.BEQZ x8, +4 — x8==0, 跳转."""
        h = Hart(id=0)
        h.gprs[8].val = 0
        h.pc = 0x2000
        # offset[2:1]=2→4, rs1'=0→x8, funct3=110
        instr = _c1(0b110, 0, 2 << 3)
        h.exec_instr(instr)
        assert h.pc == 0x2004

    def test_c_bnez_not_taken(self):
        """C.BNEZ x8, +4 — x8==0, 不跳转."""
        h = Hart(id=0)
        h.gprs[8].val = 0
        h.pc = 0x2000
        instr = _c1(0b111, 0, 2 << 3)
        adv = h.exec_instr(instr)
        assert adv == 2

    def test_c_slli(self):
        """C.SLLI x10, 3 → x10 <<= 3."""
        h = Hart(id=0)
        h.gprs[10].val = 5
        instr = _c2(0b000, 10, 3 << 2)
        h.exec_instr(instr)
        assert h.gprs[10].val == 40


# ============================================================
#  C0 象限测试 — 栈相对 load/store (使用 3-bit 压缩寄存器)
# ============================================================


class TestCompressedC0:
    """C0 象限 (低 2 位 = 00)."""

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        ram, read_fn, write_fn = _make_ram()
        h.set_memory_backend(read_fn, write_fn)
        h.gprs[2].val = 0x8000  # sp
        return h

    def test_c_addi4spn(self, hart):
        """C.ADDI4SPN x8, sp, 16."""
        # nzuimm=16: [5:4]=01,[9:6]=0000,[3]=0,[2]=0
        # bits: [12:11]=01, [10:7]=0000, [6]=0, [5]=0, [4:2]=000(rd'=x8)
        scatter = (0b01 << 11) | (0b0000 << 7) | (0 << 6) | (0 << 5)
        instr = _c0(0b000, 0, 0, scatter)
        hart.exec_instr(instr)
        assert hart.gprs[8].val == 0x8010

    def test_c_lw(self, hart):
        """C.LW x8, 8(x10) → 加载 32-bit."""
        test_val = 0xDEAD
        hart._mem_write_phy(0x8008, test_val.to_bytes(4, "little"))
        hart.gprs[10].val = 0x8000
        # uimm=8: [6]=0,[5:3]=001,[2]=0
        # bits: [12:10]=001, [9:7]=010(rs1'=x10), [6]=0, [5]=0, [4:2]=000(rd'=x8)
        scatter = (1 << 10) | (2 << 7) | (0 << 6) | (0 << 5)
        instr = _c0(0b010, 0, 0, scatter)
        hart.exec_instr(instr)
        assert (hart.gprs[8].val & 0xFFFF) == 0xDEAD

    def test_c_ld(self, hart):
        """C.LD x8, 8(x10) → 加载 64-bit."""
        test_val = 0xFEED_CACE
        hart._mem_write_phy(0x8008, test_val.to_bytes(8, "little"))
        hart.gprs[10].val = 0x8000
        scatter = (1 << 10) | (2 << 7)
        instr = _c0(0b011, 0, 0, scatter)
        hart.exec_instr(instr)
        assert hart.gprs[8].val == test_val

    def test_c_sw(self, hart):
        """C.SW x9, 8(x10) → 存储 32-bit."""
        hart.gprs[10].val = 0x8000
        hart.gprs[9].val = 0xCAFE
        # rs2'=1→x9, rd' bits 是 rs2'
        scatter = (1 << 10) | (2 << 7)
        instr = _c0(0b110, 1, 0, scatter)
        hart.exec_instr(instr)
        val = int.from_bytes(hart._mem_read_phy(0x8008, 4), "little")
        assert val == 0xCAFE

    def test_c_sd(self, hart):
        """C.SD x9, 8(x10) → 存储 64-bit."""
        hart.gprs[10].val = 0x8000
        hart.gprs[9].val = 0xDEAD_BEEF
        scatter = (1 << 10) | (2 << 7)
        instr = _c0(0b111, 1, 0, scatter)
        hart.exec_instr(instr)
        val = int.from_bytes(hart._mem_read_phy(0x8008, 8), "little")
        assert val == 0xDEAD_BEEF


# ============================================================
#  C2 象限测试 — SP-relative load/store
# ============================================================


class TestCompressedC2:
    """C2 象限 (低 2 位 = 10)."""

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        ram, read_fn, write_fn = _make_ram()
        h.set_memory_backend(read_fn, write_fn)
        h.gprs[2].val = 0x8000
        return h

    def test_c_lwsp(self, hart):
        """C.LWSP x10, 8 → 从 sp+8 加载 32-bit."""
        test_val = 0x12345678
        hart._mem_write_phy(0x8008, test_val.to_bytes(4, "little"))
        # uimm=8: [5]=0, [4:2]=010, [7:6]=00
        scatter = (0 << 12) | (2 << 2) | (0 << 7)
        instr = _c2(0b010, 10, scatter)
        hart.exec_instr(instr)
        assert hart.gprs[10].val == test_val

    def test_c_ldsp(self, hart):
        """C.LDSP x10, 8 → 从 sp+8 加载 64-bit."""
        test_val = 0xFEED_FACE
        hart._mem_write_phy(0x8008, test_val.to_bytes(8, "little"))
        scatter = (0 << 12) | (2 << 2)
        instr = _c2(0b011, 10, scatter)
        hart.exec_instr(instr)
        assert hart.gprs[10].val == test_val

    def test_c_swsp(self, hart):
        """C.SWSP x5, 4 → 向 sp+4 存储 32-bit."""
        hart.gprs[5].val = 0xBEEF
        # uimm[5:2]=1→4, rs2=x5
        scatter = (1 << 9) | (5 << 2)
        instr = (0b110 << 13) | scatter | 0b10
        hart.exec_instr(instr)
        val = int.from_bytes(hart._mem_read_phy(0x8004, 4), "little")
        assert val == 0xBEEF

    def test_c_sdsp(self, hart):
        """C.SDSP x5, 8 → 向 sp+8 存储 64-bit."""
        hart.gprs[5].val = 0xDEAD_BEEF
        scatter = (2 << 9) | (5 << 2)  # uimm=8
        instr = (0b111 << 13) | scatter | 0b10
        hart.exec_instr(instr)
        val = int.from_bytes(hart._mem_read_phy(0x8008, 8), "little")
        assert val == 0xDEAD_BEEF
