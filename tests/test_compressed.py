#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""压缩指令 (C-extension) 测试 — 使用程序化计算的 RV64C 指令编码."""

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.mem_check_aux import inject_memory_backend


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

    def test_c_jalr_rd_eq_rs1_uses_old_value(self):
        """C.JALR ra (rd_rs1==x1): 跳转目标应为 ra 旧值, ra←pc+2."""
        h = Hart(id=0)
        old_ra = 0x80004000
        h.gprs[1].val = old_ra  # ra = 旧值
        h.pc = 0x1000
        # C.JALR ra: funct3=100, rd_rs1=1(x1/ra), bit12=1(JALR)
        instr = _c2(0b100, 1, 1 << 12)
        h.exec_instr(instr)
        assert h.gprs[1].val == 0x1002, f"ra 应为 pc+2, 实际 0x{h.gprs[1].val:x}"
        assert h.pc == old_ra, f"PC 应为 ra 旧值 0x{old_ra:x}, 实际 0x{h.pc:x}"

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
        inject_memory_backend(h, read_fn, write_fn)
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

    # -- 大偏移量测试 (uimm[7:6] ≠ 0, 验证 C.LD/C.SD 立即数布局) --

    def test_c_sd_large_offset(self, hart):
        """C.SD x14, 144(x12) — uimm[7:6]=2, 偏移 ≠ 8*n 时验证位布局."""
        hart.gprs[12].val = 0x8000
        hart.gprs[14].val = 0xDEAD_BEEF_CAFE
        # offset=144 → uimm_field=18 → uimm[5:3]=2, uimm[7:6]=2
        # C.SD: funct3=111, rs1_creg=4(x12), rs2_creg=6(x14)
        instr = (0b111 << 13) | (2 << 10) | (4 << 7) | (2 << 5) | (6 << 2)
        hart.exec_instr(instr)
        val = int.from_bytes(hart._mem_read_phy(0x8000 + 144, 8), "little")
        assert val == 0xDEAD_BEEF_CAFE, (
            f"偏移 144 的 C.SD: 期望 0xDEAD_BEEF_CAFE, 得到 0x{val:016x}"
        )

    def test_c_ld_large_offset(self, hart):
        """C.LD x14, 144(x12) — uimm[7:6]=2, 验证与 C.SD 相同位布局."""
        test_val = 0xFEED_CACE_BABE
        hart._mem_write_phy(0x8000 + 144, test_val.to_bytes(8, "little"))
        hart.gprs[12].val = 0x8000
        # C.LD: funct3=011, rs1_creg=4(x12), rd_creg=6(x14)
        instr = (0b011 << 13) | (2 << 10) | (4 << 7) | (2 << 5) | (6 << 2)
        hart.exec_instr(instr)
        assert hart.gprs[14].val == test_val, (
            f"偏移 144 的 C.LD: 期望 0x{test_val:016x}, 得到 0x{hart.gprs[14].val:016x}"
        )

    def test_c_sd_vs_c_sw_bit_layout(self, hart):
        """C.SD 与 C.SW 的同 bit pattern 应解码为不同偏移 (验证布局分立)."""
        hart.gprs[10].val = 0x8000
        # C.SD: funct3=111, offset=200, rs1_creg=2(x10), rs2_creg=4(x12)
        # uimm_field=25 → uimm[5:3]=1, uimm[7:6]=3
        hart.gprs[12].val = 0xAAAA
        instr_sd = (0b111 << 13) | (1 << 10) | (2 << 7) | (3 << 5) | (4 << 2)
        hart.exec_instr(instr_sd)
        val_at_200 = int.from_bytes(hart._mem_read_phy(0x8000 + 200, 8), "little")
        assert val_at_200 == 0xAAAA, f"C.SD 应在偏移 200, 实际写了偏移 {200 if val_at_200 == 0xAAAA else '?'}"

        # C.SW: 相同 scatter, funct3=110 — 应计算出不同偏移 (uimm[2] 非零)
        hart.gprs[12].val = 0xBBBB
        instr_sw = (0b110 << 13) | (1 << 10) | (2 << 7) | (3 << 5) | (4 << 2)
        hart.exec_instr(instr_sw)
        # C.SW: uimm = instr[6]<<6 | instr[12:10]<<3 | instr[5]<<2
        # = 1<<6 | 1<<3 | 1<<2 = 64 + 8 + 4 = 76
        val_at_76 = int.from_bytes(hart._mem_read_phy(0x8000 + 76, 4), "little") & 0xFFFF_FFFF
        assert val_at_76 == 0xBBBB, (
            "C.SW 与 C.SD 应有不同偏移: C.SW→76, 实际写了偏移 ?"
        )
        # 确认 C.SD 的偏移 200 处没有被 C.SW 覆写
        val_still = int.from_bytes(hart._mem_read_phy(0x8000 + 200, 8), "little")
        assert val_still == 0xAAAA, "C.SW 不应覆写 C.SD 的偏移 200 位置"


# ============================================================
#  C2 象限测试 — SP-relative load/store
# ============================================================


class TestCompressedC2:
    """C2 象限 (低 2 位 = 10)."""

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        ram, read_fn, write_fn = _make_ram()
        inject_memory_backend(h, read_fn, write_fn)
        h.gprs[2].val = 0x8000
        return h

    def test_c_lwsp(self, hart):
        """C.LWSP x10, 8 → 从 sp+8 加载 32-bit."""
        test_val = 0x12345678
        hart._mem_write_phy(0x8008, test_val.to_bytes(4, "little"))
        # C.LWSP: funct3=010, uimm=8, rd=10
        # bit[12]=0(uimm[5]), bits[6:5]=00(uimm[7:6]), bits[4:2]=010(uimm[4:2])
        instr = (0b010 << 13) | (0 << 12) | (10 << 7) | (0b010 << 2) | 0b10
        hart.exec_instr(instr)
        assert hart.gprs[10].val == test_val

    def test_c_ldsp(self, hart):
        """C.LDSP x10, 8 → 从 sp+8 加载 64-bit."""
        test_val = 0xFEED_FACE
        hart._mem_write_phy(0x8008, test_val.to_bytes(8, "little"))
        # C.LDSP: funct3=011, uimm=8, rd=10
        # uimm=8: imm[5]=0, imm[4:3]=01, imm[8:6]=000
        # bit[12]=0, bits[6:5]=01, bits[4:2]=000
        instr = (0b011 << 13) | (0 << 12) | (10 << 7) | (0b01 << 5) | (0b000 << 2) | 0b10
        hart.exec_instr(instr)
        assert hart.gprs[10].val == test_val

    def test_c_swsp(self, hart):
        """C.SWSP x5, 4 → 向 sp+4 存储 32-bit."""
        hart.gprs[5].val = 0xBEEF
        # C.SWSP: funct3=110, uimm=4, rs2=x5
        # bit[15:13]=110, bits[12:9]=0001(uimm[5:2]), bits[8:7]=00(uimm[7:6])
        instr = (0b110 << 13) | (1 << 9) | (5 << 2) | 0b10
        hart.exec_instr(instr)
        val = int.from_bytes(hart._mem_read_phy(0x8004, 4), "little")
        assert val == 0xBEEF

    def test_c_sdsp(self, hart):
        """C.SDSP x5, 8 → 向 sp+8 存储 64-bit."""
        hart.gprs[5].val = 0xDEAD_BEEF
        # C.SDSP: funct3=111, uimm=8, rs2=x5
        # uimm=8: imm[5:3]=001, imm[8:6]=000
        # bits[12:10]=001, bits[9:7]=000, bits[6:2]=00101(rs2=5)
        instr = (0b111 << 13) | (1 << 10) | (5 << 2) | 0b10
        hart.exec_instr(instr)
        val = int.from_bytes(hart._mem_read_phy(0x8008, 8), "little")
        assert val == 0xDEAD_BEEF

    # -- 大偏移量测试 (验证 C.LWSP/C.LDSP/C.SDSP 立即数布局分立) --

    def test_c_lwsp_large_offset(self, hart):
        """C.LWSP x10, 40(sp) — uimm[5]=1 验证 imm[5] 位位置."""
        test_val = 0x12345678
        hart._mem_write_phy(0x8000 + 40, test_val.to_bytes(4, "little"))
        # C.LWSP: funct3=010, uimm=40, rd=10
        # bit[15:13]=010, bit[12]=1(uimm[5]), bits[11:7]=01010,
        # bits[6:5]=00(uimm[7:6]), bits[4:2]=010(uimm[4:2])
        instr = (0b010 << 13) | (1 << 12) | (10 << 7) | (0b010 << 2) | 0b10
        hart.exec_instr(instr)
        assert hart.gprs[10].val == test_val, (
            f"偏移 40 的 C.LWSP: 期望 0x{test_val:08x}, 得到 0x{hart.gprs[10].val:08x}"
        )

    def test_c_ldsp_large_offset(self, hart):
        """C.LDSP x10, 40(sp) — uimm[5]=1, uimm[4:3]=01 验证布局."""
        test_val = 0xFEED_FACE_CAFE
        hart._mem_write_phy(0x8000 + 40, test_val.to_bytes(8, "little"))
        # C.LDSP: funct3=011, uimm=40, rd=10
        # bit[15:13]=011, bit[12]=1(uimm[5]), bits[11:7]=01010,
        # bits[6:5]=01(uimm[4:3]), bits[4:2]=000(uimm[8:6])
        instr = (0b011 << 13) | (1 << 12) | (10 << 7) | (0b01 << 5) | 0b10
        hart.exec_instr(instr)
        assert hart.gprs[10].val == test_val, (
            f"偏移 40 的 C.LDSP: 期望 0x{test_val:016x}, 得到 0x{hart.gprs[10].val:016x}"
        )

    def test_c_sdsp_large_offset(self, hart):
        """C.SDSP x10, 40(sp) — uimm[8:6]=000, uimm[5:3]=101 验证布局."""
        hart.gprs[10].val = 0xDEAD_BEEF_BABE
        # C.SDSP: funct3=111, uimm=40, rs2=10
        # bit[15:13]=111, bits[12:10]=101(uimm[5:3]),
        # bits[9:7]=000(uimm[8:6]), bits[6:2]=01010(rs2)
        instr = (0b111 << 13) | (5 << 10) | (10 << 2) | 0b10
        hart.exec_instr(instr)
        val = int.from_bytes(hart._mem_read_phy(0x8000 + 40, 8), "little")
        assert val == 0xDEAD_BEEF_BABE, (
            f"偏移 40 的 C.SDSP: 期望 0xDEAD_BEEF_BABE, 得到 0x{val:016x}"
        )

    def test_c_lwsp_vs_c_ldsp_bit_layout(self, hart):
        """C.LWSP 与 C.LDSP 同 bit pattern 应解码为不同偏移."""
        test_val_lw = 0x42
        hart._mem_write_phy(0x8000 + 40, test_val_lw.to_bytes(4, "little"))
        # C.LWSP: 偏移 40 的编码 (见 test_c_lwsp_large_offset)
        instr_lwsp = (0b010 << 13) | (1 << 12) | (10 << 7) | (0b010 << 2) | 0b10
        hart.exec_instr(instr_lwsp)
        assert hart.gprs[10].val == 0x42, "C.LWSP 偏移 40 应读取正确值"

        # 用 C.LDSP 的 bit pattern 构造与 C.LWSP 相同 raw 位但不同 funct3
        # C.LDSP funct3=011, 同 offset=40 的编码
        test_val_ld = 0xDEAD_BEEF
        hart._mem_write_phy(0x8000 + 40, test_val_ld.to_bytes(8, "little"))
        instr_ldsp = (0b011 << 13) | (1 << 12) | (10 << 7) | (0b01 << 5) | 0b10
        hart.exec_instr(instr_ldsp)
        assert hart.gprs[10].val == test_val_ld, (
            f"C.LDSP 偏移 40: 期望 0x{test_val_ld:x}, 得到 0x{hart.gprs[10].val:x}"
        )

    # -- C.ADD (回归: 曾被误当 C.MV 执行) --

    def test_c_add(self, hart):
        """C.ADD x5, x6 → x5 += x6 (非 x5 = x6)."""
        hart.gprs[5].val = 0x804E0
        hart.gprs[6].val = 0x80000000
        # C.ADD: C2 象限 (bit1:0=10), funct3=4, bit12=1, rd_rs1=x5, rs2=x6
        instr = (0b100 << 13) | (1 << 12) | (5 << 7) | (6 << 2) | 0b10
        hart.exec_instr(instr)
        expected = (0x804E0 + 0x80000000) & 0xFFFF_FFFF_FFFF_FFFF
        assert hart.gprs[5].val == expected, (
            f"C.ADD: {hart.gprs[5].val:#x} != {expected:#x} (expected x5 += x6)"
        )

    def test_c_add_preserves_upper_bits(self, hart):
        """C.ADD 结果应规范化到 64 位."""
        hart.gprs[5].val = 0xFFFFFFFFD0000000
        hart.gprs[6].val = 0xD0000
        instr = (0b100 << 13) | (1 << 12) | (5 << 7) | (6 << 2) | 0b10
        hart.exec_instr(instr)
        expected = (0xFFFFFFFFD0000000 + 0xD0000) & 0xFFFF_FFFF_FFFF_FFFF
        assert hart.gprs[5].val == expected

    # -- C.SUB (RV64C, 回归: sf=0b11 未实现) --

    def test_c_sub(self, hart):
        """C.SUB x15, x14 → x15 -= x14 (sf=3 variant)."""
        hart.gprs[15].val = 0x804E0
        hart.gprs[14].val = 0x3E8
        # C.SUB: C1, funct3=4, bit12=0, bits[11:10]=11, bits[6:5]=00
        # rd/rs1=x15 (3-bit=7), rs2=x14 (3-bit=6)
        instr = (0b100 << 13) | (0b11 << 10) | (7 << 7) | (0b00 << 5) | (6 << 2) | 0b01
        hart.exec_instr(instr)
        expected = (0x804E0 - 0x3E8) & 0xFFFF_FFFF_FFFF_FFFF
        assert hart.gprs[15].val == expected, (
            f"C.SUB: {hart.gprs[15].val:#x} != {expected:#x}"
        )

    def test_c_or(self, hart):
        """C.OR x15, x14 → x15 |= x14 (RV64C sf=3, bits[6:5]=10)."""
        hart.gprs[15].val = 0xF0
        hart.gprs[14].val = 0x0F
        # C.OR: C1, funct3=4, bit12=0, bits[11:10]=11, bits[6:5]=10
        instr = (0b100 << 13) | (0b11 << 10) | (7 << 7) | (0b10 << 5) | (6 << 2) | 0b01
        hart.exec_instr(instr)
        expected = (0xF0 | 0x0F) & 0xFFFF_FFFF_FFFF_FFFF
        assert hart.gprs[15].val == expected

    def test_c_srli_64bit_shift(self, hart):
        """C.SRLI x15, 7 — RV64 移位, shamt[5] 在 bit12 而非 bit7."""
        hart.gprs[15].val = 0x8000000000140000
        # C.SRLI: C1, funct3=4, sf=bits[11:10]=00, bit12=0
        # shamt[4:0] = bits[6:2] = 7, shamt[5] = bit12 = 0
        instr = (0b100 << 13) | (0b00 << 10) | (7 << 7) | (0b00111 << 2) | 0b01
        hart.exec_instr(instr)
        # 64-bit shift: 0x8000000000140000 >> 7 = 0x0100000000002800
        expected = 0x8000000000140000 >> 7
        assert hart.gprs[15].val == expected, (
            f"期望 0x{expected:016x}, 实际 0x{hart.gprs[15].val:016x}"
        )

    def test_c_srli_shamt_bit7_is_ignored(self, hart):
        """C.SRLI: bit7 不参与 shamt 计算, shamt[5] 由 bit12 提供."""
        hart.gprs[15].val = 0x8000000000140000
        # 构造一条 bit7=1 但 bit12=0 的 C.SRLI: shamt 应 = bits[6:2] = 7
        # (旧 bug: 使用 bit7 作为 shamt[5] 会得到 shamt=7|32=39)
        instr = (0b100 << 13) | (0b00 << 10) | (7 << 7) | (1 << 7) | (0b00111 << 2) | 0b01
        hart.exec_instr(instr)
        # 正确: 0x8000000000140000 >> 7 = 0x0100000000002800
        expected = 0x8000000000140000 >> 7
        assert hart.gprs[15].val == expected, (
            f"bit7 err: exp 0x{expected:016x}, got 0x{hart.gprs[15].val:016x}"
        )

    def test_c_andi_sf2_encoding(self, hart):
        """C.ANDI x15, 1 — sf=2 (bits[11:10]=10), 非 sf=1."""
        hart.gprs[15].val = 0x0100000000002800
        # C.ANDI: C1, funct3=4, bits[12:10]=010 (sf=2)
        # imm[4:0] = bits[6:2] = 1
        instr = (0b100 << 13) | (0b010 << 10) | (7 << 7) | (0b00001 << 2) | 0b01
        hart.exec_instr(instr)
        # 0x0100000000002800 & 1 = 0
        assert hart.gprs[15].val == 0, (
            f"C.ANDI sf=2 编码错误: 期望 0, 实际 0x{hart.gprs[15].val:x}"
        )

    def test_c_srli_then_c_andi_chain(self, hart):
        """组合 srli+andi 提取 misa H-bit: 完整模拟 _start_warm 的检测逻辑."""
        # 模拟 misa = 0x8000000000140000 (H-bit=0)
        hart.gprs[15].val = 0x8000000000140000
        # C.SRLI x15, 7
        instr_srli = (0b100 << 13) | (0b00 << 10) | (7 << 7) | (0b00111 << 2) | 0b01
        hart.exec_instr(instr_srli)
        # 验证移位结果
        assert hart.gprs[15].val == 0x0100000000002800, "C.SRLI 64-bit 移位错误"
        # C.ANDI x15, 1
        instr_andi = (0b100 << 13) | (0b010 << 10) | (7 << 7) | (0b00001 << 2) | 0b01
        hart.exec_instr(instr_andi)
        # H-bit=0 → 结果应为 0
        assert hart.gprs[15].val == 0, (
            f"misa H-bit 检测失败: 期望 0, 实际 0x{hart.gprs[15].val:x}"
        )

    def test_c_srli_then_c_andi_h_ext_present(self, hart):
        """misa H-bit=1 场景: srli+andi 正确提取置位的 H 位."""
        # 模拟 misa bit7=1 (H 扩展存在)
        hart.gprs[15].val = 0x8000000000140080  # bit7=1
        # C.SRLI x15, 7
        instr_srli = (0b100 << 13) | (0b00 << 10) | (7 << 7) | (0b00111 << 2) | 0b01
        hart.exec_instr(instr_srli)
        # C.ANDI x15, 1
        instr_andi = (0b100 << 13) | (0b010 << 10) | (7 << 7) | (0b00001 << 2) | 0b01
        hart.exec_instr(instr_andi)
        # H-bit=1 → 结果应为 1
        assert hart.gprs[15].val == 1, (
            f"H-bit=1 检测失败: 期望 1, 实际 {hart.gprs[15].val}"
        )
