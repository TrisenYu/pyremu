#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 20:50:45
# Last modified at 2026/06/09 星期二

"""
RISC-V RV64 反汇编器.

将 32-bit / 16-bit 指令字翻译为可读汇编字符串。
覆盖 RV64 I + M + Zicsr + AMO, 与 decoder.py 的 Hart 执行单元对齐。
"""

from pyremu.core.decoder import (
    AmoFunct5,
    AmoWidth,
    Opc,
    brFn3,
    parse_compressed,
    parse_func3,
    parse_func6,
    parse_func7,
    parse_imm12_se,
    parse_imm20_raw,
    parse_imm_b,
    parse_imm_j,
    parse_imm_s,
    parse_opcode,
    parse_rd,
    parse_rs1,
    parse_rs2,
    stFn3,
)
from pyremu.core.registers import check_csr, gpr_name

_UNKNOWN = "<unknown opcode>"


def _fmt_imm(val: int) -> str:
    """格式化指令立即数: 有符号十进制 (处理 64-bit 规范化值)."""
    # _sext 规范化后, 负立即数以 64-bit 无符号形式传入.
    # 若 bit 63 置位则还原为有符号显示 (例如 0xFF…F0 -> -16).
    if val >= (1 << 63):
        val = val - (1 << 64)
    return str(val)


def _fmt_addr(val: int) -> str:
    """格式化绝对地址: hex."""
    return f"0x{val & 0xFFFF_FFFF_FFFF_FFFF:x}"


# ============================================================
#  字段提取辅助
# ============================================================


def _rd(instr: int) -> str:
    return gpr_name(parse_rd(instr))


def _rs1(instr: int) -> str:
    return gpr_name(parse_rs1(instr))


def _rs2(instr: int) -> str:
    return gpr_name(parse_rs2(instr))


# ============================================================
#  R-type 编码表 — funct3 -> (func7 -> mnemonic)
# ============================================================

_R_FUNCT3_MAP: dict[int, dict[int, str]] = {
    0b000: {0x00: "add", 0x01: "mul", 0x20: "sub"},
    0b001: {0x00: "sll", 0x01: "mulh"},
    0b010: {0x00: "slt", 0x01: "mulhsu"},
    0b011: {0x00: "sltu", 0x01: "mulhu"},
    0b100: {0x00: "xor", 0x01: "div"},
    0b101: {0x00: "srl", 0x01: "divu", 0x20: "sra"},
    0b110: {0x00: "or", 0x01: "rem"},
    0b111: {0x00: "and", 0x01: "remu"},
}


def _dis_rtype(instr: int) -> str:
    """R-type: mnemonic rd, rs1, rs2."""
    f3 = parse_func3(instr)
    f7 = parse_func7(instr)
    mnemonic = _R_FUNCT3_MAP.get(f3, {}).get(f7)
    if mnemonic is None:
        return _UNKNOWN
    return f"{mnemonic:<7} {_rd(instr)}, {_rs1(instr)}, {_rs2(instr)}"


# ============================================================
#  I-type ALU (opImm)
# ============================================================

_IMM_FUNCT3_MAP: dict[int, str] = {
    0b000: "addi",
    0b010: "slti",
    0b011: "sltiu",
    0b100: "xori",
    0b110: "ori",
    0b111: "andi",
}


def _dis_itype(instr: int) -> str:
    """I-type ALU: mnemonic rd, rs1, imm."""
    f3 = parse_func3(instr)

    if f3 == 0b001:  # SLLI — funct6 (bits 31:26), bit 25 属于 shamt
        if parse_func6(instr) != 0:
            return _UNKNOWN
        shamt = (instr >> 20) & 0x3F
        return f"slli    {_rd(instr)}, {_rs1(instr)}, {shamt}"
    if f3 == 0b101:  # SRLI / SRAI
        shamt = (instr >> 20) & 0x3F
        f6 = parse_func6(instr)
        if f6 == 0x00:  # SRLI (funct6=0b000000)
            return f"srli    {_rd(instr)}, {_rs1(instr)}, {shamt}"
        if f6 == 0x10:  # SRAI (funct6=0b010000)
            return f"srai    {_rd(instr)}, {_rs1(instr)}, {shamt}"
        return _UNKNOWN

    parse_func7(instr)  # 非 shift I-type 才使用 funct7

    mnemonic = _IMM_FUNCT3_MAP.get(f3)
    if mnemonic is None:
        return _UNKNOWN
    imm = parse_imm12_se(instr)
    return f"{mnemonic:<7} {_rd(instr)}, {_rs1(instr)}, {_fmt_imm(imm)}"


# ============================================================
#  RV64 32-bit ops (op32)
# ============================================================

_OP32_FUNCT3_MAP: dict[int, dict[int, str]] = {
    0b000: {0x00: "addw", 0x01: "mulw", 0x20: "subw"},
    0b001: {0x00: "sllw", 0x01: "mulhw"},
    0b010: {0x00: "sltw", 0x01: "mulhsuw"},
    0b011: {0x00: "sltuw", 0x01: "mulhuw"},
    0b100: {0x00: "xorw", 0x01: "divw"},
    0b101: {0x00: "srlw", 0x01: "divuw", 0x20: "sraw"},
    0b110: {0x00: "orw", 0x01: "remw"},
    0b111: {0x00: "andw", 0x01: "remuw"},
}


def _dis_op32(instr: int) -> str:
    """RV64 32-bit R-type: mnemonic rd, rs1, rs2."""
    f3 = parse_func3(instr)
    f7 = parse_func7(instr)
    mnemonic = _OP32_FUNCT3_MAP.get(f3, {}).get(f7)
    if mnemonic is None:
        return _UNKNOWN
    return f"{mnemonic:<7} {_rd(instr)}, {_rs1(instr)}, {_rs2(instr)}"


def _dis_op_imm32(instr: int) -> str:
    """RV64 32-bit I-type: ADDIW / SLLIW / SRLIW / SRAIW."""
    f3 = parse_func3(instr)
    f7 = parse_func7(instr)
    if f3 == 0b000:
        imm = parse_imm12_se(instr)
        return f"addiw   {_rd(instr)}, {_rs1(instr)}, {_fmt_imm(imm)}"
    if f3 == 0b001:
        if f7 != 0:
            return _UNKNOWN
        shamt = (instr >> 20) & 0x1F
        return f"slliw   {_rd(instr)}, {_rs1(instr)}, {shamt}"
    if f3 == 0b101:
        shamt = (instr >> 20) & 0x1F
        if f7 == 0x00:
            return f"srliw   {_rd(instr)}, {_rs1(instr)}, {shamt}"
        if f7 == 0x20:
            return f"sraiw   {_rd(instr)}, {_rs1(instr)}, {shamt}"
    return _UNKNOWN


# ============================================================
#  Load / Store
# ============================================================

_LD_MNEMONIC: dict[int, str] = {
    0b000: "lb",
    0b001: "lh",
    0b010: "lw",
    0b011: "ld",
    0b100: "lbu",
    0b101: "lhu",
    0b110: "lwu",
}


def _dis_load(instr: int) -> str:
    """Load: mnemonic rd, offset(rs1)."""
    f3 = parse_func3(instr)
    mnemonic = _LD_MNEMONIC.get(f3)
    if mnemonic is None:
        return _UNKNOWN
    offset = parse_imm12_se(instr)
    return f"{mnemonic:<7} {_rd(instr)}, {_fmt_imm(offset)}({_rs1(instr)})"


def _dis_store(instr: int) -> str:
    """Store: mnemonic rs2, offset(rs1)."""
    f3 = parse_func3(instr)
    try:
        mnemonic = stFn3(f3).name
    except ValueError:
        return _UNKNOWN
    offset = parse_imm_s(instr)
    return f"{mnemonic:<7} {_rs2(instr)}, {_fmt_imm(offset)}({_rs1(instr)})"


# ============================================================
#  Branch
# ============================================================


def _dis_branch(instr: int, pc: int) -> str:
    """Branch: mnemonic rs1, rs2, target_addr."""
    f3 = parse_func3(instr)
    try:
        mnemonic = brFn3(f3).name
    except ValueError:
        return _UNKNOWN
    offset = parse_imm_b(instr)
    target = (pc + offset) & 0xFFFF_FFFF_FFFF_FFFF
    return f"{mnemonic:<7} {_rs1(instr)}, {_rs2(instr)}, {_fmt_addr(target)}"


# ============================================================
#  JAL / JALR
# ============================================================


def _dis_jal(instr: int, pc: int) -> str:
    """JAL: mnemonic rd, target_addr."""
    offset = parse_imm_j(instr)
    target = (pc + offset) & 0xFFFF_FFFF_FFFF_FFFF
    return f"jal     {_rd(instr)}, {_fmt_addr(target)}"


def _dis_jalr(instr: int) -> str:
    """JALR: mnemonic rd, offset(rs1)."""
    offset = parse_imm12_se(instr)
    return f"jalr    {_rd(instr)}, {_fmt_imm(offset)}({_rs1(instr)})"


# ============================================================
#  LUI / AUIPC
# ============================================================


def _dis_lui(instr: int) -> str:
    # LUI 的 20-bit 立即数为地址高位常量 (实际值 = imm << 12), 用 hex 更直观
    imm = parse_imm20_raw(instr)
    return f"lui     {_rd(instr)}, {_fmt_addr(imm)}"


def _dis_auipc(instr: int) -> str:
    # AUIPC 同上, upper immediate 用 hex 表示
    imm = parse_imm20_raw(instr)
    return f"auipc   {_rd(instr)}, {_fmt_addr(imm)}"


# ============================================================
#  System (CSR + privileged)
# ============================================================

_CSR_MNEMONIC: dict[int, str] = {
    0b001: "csrrw",
    0b010: "csrrs",
    0b011: "csrrc",
    0b101: "csrrwi",
    0b110: "csrrsi",
    0b111: "csrrci",
}


def _csr_name(addr: int) -> str:
    """CSR 地址 -> 名称, 未找到则返回地址 hex."""
    ok, name = check_csr(addr)
    return name if ok else f"0x{addr:03x}"


def _dis_csr(instr: int) -> str:
    """CSR 指令: mnemonic rd, csr, rs1/uimm."""
    f3 = parse_func3(instr)
    mnemonic = _CSR_MNEMONIC.get(f3)
    if mnemonic is None:
        return _UNKNOWN
    csr_addr = (instr >> 20) & 0xFFF
    csr = _csr_name(csr_addr)
    rd_name = _rd(instr)
    if f3 & 0b100:  # immediate 形式
        uimm = parse_rs1(instr)
        return f"{mnemonic:<7} {rd_name}, {csr}, {uimm}"
    return f"{mnemonic:<7} {rd_name}, {csr}, {_rs1(instr)}"


_PRIV_MNEMONIC: dict[int, str] = {
    0x000: "ecall",
    0x001: "ebreak",
    0x302: "mret",
    0x102: "sret",
    0x105: "wfi",
    0x120: "sfence.vma",
    0x5A0: "mfence.did",
}


def _dis_priv(instr: int) -> str:
    """特权指令 (funct3=000)."""
    funct12 = (instr >> 20) & 0xFFF
    mnemonic = _PRIV_MNEMONIC.get(funct12)
    if mnemonic is None:
        return _UNKNOWN
    return mnemonic


# ============================================================
#  FENCE
# ============================================================


def _dis_fence(instr: int) -> str:
    f3 = parse_func3(instr)
    if f3 == 0b000:
        return "fence"
    if f3 == 0b001:
        return "fence.i"
    return _UNKNOWN


# ============================================================
#  AMO
# ============================================================


def _dis_amo(instr: int) -> str:
    funct5_val = (instr >> 27) & 0x1F
    funct3_val = parse_func3(instr)
    try:
        op = AmoFunct5(funct5_val)
        width = AmoWidth(funct3_val)
    except ValueError:
        return _UNKNOWN

    suffix = ".d" if width == AmoWidth.D else ".w"
    base = op.name.lower()
    # LR / SC 不带 amo 前缀, 其余原子操作带
    if op in (AmoFunct5.LR, AmoFunct5.SC):
        mnemonic = base + suffix
    else:
        mnemonic = "amo" + base + suffix
    return f"{mnemonic:<7} {_rd(instr)}, {_rs2(instr)}, ({_rs1(instr)})"


# ============================================================
#  压缩指令 (C extension) 反汇编
# ============================================================
# 寄存器: x8–x15 使用 3-bit 缩写索引 rd′/rs1′/rs2′.


def _c_x8(idx3: int) -> str:
    """3-bit 压缩寄存器索引 -> ABI 名 (x8–x15)."""
    return gpr_name(8 + (idx3 & 0b111))


def _dis_compressed(
    c16: int,
    pc: int,
) -> str:
    """反汇编 16-bit RISC-V 压缩指令 (C extension).

    覆盖 RV64C 常用指令: 寄存器操作、立即数、Load/Store、分支、跳转。
    未实现的编码退回 raw hex 显示。
    """
    quad = c16 & 0b11
    funct3 = (c16 >> 13) & 0b111
    b12 = (c16 >> 12) & 1
    b11_10 = (c16 >> 10) & 0b11

    # --- Quadrant 0: CIW / CL / CS ---
    if quad == 0b00:
        rd_p = _c_x8(c16 >> 2)
        rs1_p = _c_x8(c16 >> 7)

        if funct3 == 0b000:  # C.ADDI4SPN
            # nzuimm[5:4] = instr[12:11], nzuimm[9:6] = instr[10:7],
            # nzuimm[2]    = instr[6],    nzuimm[3]    = instr[5]
            # 基址寄存器固定为 sp (x2), bits[9:7] 属于立即数而非 rs1' 字段.
            uimm = (
                ((c16 >> 7) & 0x0F) << 6   # instr[10:7] -> nzuimm[9:6]
                | ((c16 >> 11) & 0x03) << 4  # instr[12:11] -> nzuimm[5:4]
                | ((c16 >> 5) & 1) << 3      # instr[5] -> nzuimm[3]
                | ((c16 >> 6) & 1) << 2      # instr[6] -> nzuimm[2]
            )
            if uimm == 0:
                return f"c.?    0x{c16:04x}  # C.ADDI4SPN nzuimm=0 (reserved)"
            return f"c.addi4spn {rd_p}, sp, {uimm}"

        if funct3 == 0b010:  # C.LW
            uimm = ((c16 >> 5) & 1) << 6 | ((c16 >> 10) & 0b111) << 3 | ((c16 >> 6) & 1) << 2
            return f"c.lw    {rd_p}, {uimm}({rs1_p})"

        if funct3 == 0b011:  # C.LD (RV64)
            uimm = ((c16 >> 5) & 0b11) << 6 | ((c16 >> 10) & 0b111) << 3
            return f"c.ld    {rd_p}, {uimm}({rs1_p})"

        if funct3 == 0b110:  # C.SW
            rs2_p = _c_x8(c16 >> 2)
            uimm = ((c16 >> 5) & 1) << 6 | ((c16 >> 10) & 0b111) << 3 | ((c16 >> 6) & 1) << 2
            return f"c.sw    {rs2_p}, {uimm}({rs1_p})"

        if funct3 == 0b111:  # C.SD (RV64)
            rs2_p = _c_x8(c16 >> 2)
            uimm = ((c16 >> 5) & 0b11) << 6 | ((c16 >> 10) & 0b111) << 3
            return f"c.sd    {rs2_p}, {uimm}({rs1_p})"

        return f"c.?    0x{c16:04x}"

    # --- Quadrant 1: CI / CJ / CB ---
    if quad == 0b01:
        rd_name = gpr_name((c16 >> 7) & 0x1F)
        rs1_p = _c_x8(c16 >> 7)
        imm6 = ((c16 >> 12) & 1) << 5 | ((c16 >> 2) & 0x1F)
        imm6_se = (imm6 & 0x20) and (imm6 | ~0x3F) or imm6  # sext 6-bit

        if funct3 == 0b000:  # C.NOP / C.ADDI
            if imm6 == 0 and ((c16 >> 7) & 0x1F) == 0:  # rd=x0, imm=0 -> C.NOP
                return "c.nop"
            return f"c.addi  {rd_name}, {imm6_se}"

        if funct3 == 0b001:  # C.ADDIW (RV64)
            return f"c.addiw {rd_name}, {imm6_se}"

        if funct3 == 0b010:  # C.LI
            return f"c.li    {rd_name}, {imm6_se}"

        if funct3 == 0b011:  # C.LUI (rd≠{0,2}) / C.ADDI16SP (rd=2)
            rd_raw = (c16 >> 7) & 0x1F
            if rd_raw == 2:  # C.ADDI16SP
                # nzimm[9:4] encoding per RISC-V spec:
                #   nzimm[4]=bit6, nzimm[5]=bit2, nzimm[6]=bit5,
                #   nzimm[8:7]=bits[4:3], nzimm[9]=bit12
                nzimm = (
                    ((c16 >> 12) & 1) << 9       # nzimm[9]
                    | ((c16 >> 3) & 0b11) << 7   # nzimm[8:7]
                    | ((c16 >> 5) & 1) << 6      # nzimm[6]
                    | ((c16 >> 2) & 1) << 5      # nzimm[5]
                    | ((c16 >> 6) & 1) << 4      # nzimm[4]
                )
                if nzimm & (1 << 9):
                    nzimm |= ~((1 << 10) - 1)
                return f"c.addi16sp sp, {nzimm}"
            # C.LUI
            nzuimm = ((c16 >> 12) & 1) << 17 | ((c16 >> 2) & 0x1F) << 12
            # sext from bit 17
            if nzuimm & (1 << 17):
                nzuimm = nzuimm - (1 << 18)
            if nzuimm == 0:
                return f"c.?    0x{c16:04x}  # C.LUI nzuimm=0 (reserved)"
            # Show the actual value (after << 12), keep compact
            val = nzuimm  # already shifted
            return f"c.lui   {rd_name}, 0x{(val >> 12) & 0x3F:x}"

        if funct3 == 0b100:  # misc ALU: SRLI / SRAI / ANDI / SUB/XOR/OR/AND
            rd_p = _c_x8(c16 >> 7)
            shamt5 = ((c16 >> 12) & 1) << 5  # bit 12 -> shamt[5] for RV64C
            uimm = ((c16 >> 2) & 0x1F) | shamt5
            imm6_se_alt = ((c16 >> 12) & 1) << 5 | ((c16 >> 2) & 0x1F)
            imm6_se_alt = (imm6_se_alt & 0x20) and (imm6_se_alt | ~0x3F) or imm6_se_alt

            if b11_10 == 0b00:  # C.SRLI (RV64C: shamt[5] must be 0 for RV32)
                return f"c.srli  {rd_p}, {uimm}"
            if b11_10 == 0b01:  # C.SRAI
                return f"c.srai  {rd_p}, {uimm}"
            if b11_10 == 0b10:  # C.ANDI
                return f"c.andi  {rd_p}, {imm6_se_alt}"
            if b11_10 == 0b11:  # register ops
                b6_5 = (c16 >> 5) & 0b11
                rs2_p = _c_x8(c16 >> 2)
                mn = {0b00: "c.sub", 0b01: "c.xor", 0b10: "c.or", 0b11: "c.and"}[b6_5]
                return f"{mn:<7} {rd_p}, {rs2_p}"

            return f"c.?    0x{c16:04x}"

        if funct3 == 0b101:  # C.J
            # offset = {b12, b8, b10_9, b6, b7, b2, b11, b5_3, 0}
            offset = (
                ((c16 >> 12) & 1) << 11
                | ((c16 >> 8) & 1) << 10
                | ((c16 >> 9) & 0b11) << 8
                | ((c16 >> 6) & 1) << 7
                | ((c16 >> 7) & 1) << 6
                | ((c16 >> 2) & 1) << 5
                | ((c16 >> 11) & 1) << 4
                | ((c16 >> 3) & 0b11) << 1
            )
            # sign extend from bit 11
            if offset & (1 << 11):
                offset |= ~((1 << 12) - 1)
            target = (pc + offset) & 0xFFFF_FFFF_FFFF_FFFF
            return f"c.j     {_fmt_addr(target)}"

        if funct3 == 0b110:  # C.BEQZ
            offset = (
                ((c16 >> 12) & 1) << 8
                | ((c16 >> 10) & 0b11) << 6
                | ((c16 >> 5) & 0b11) << 4
                | ((c16 >> 3) & 0b11) << 1
            )
            if offset & (1 << 8):
                offset |= ~((1 << 9) - 1)
            target = (pc + offset) & 0xFFFF_FFFF_FFFF_FFFF
            return f"c.beqz  {rs1_p}, {_fmt_addr(target)}"

        if funct3 == 0b111:  # C.BNEZ
            offset = (
                ((c16 >> 12) & 1) << 8
                | ((c16 >> 10) & 0b11) << 6
                | ((c16 >> 5) & 0b11) << 4
                | ((c16 >> 3) & 0b11) << 1
            )
            if offset & (1 << 8):
                offset |= ~((1 << 9) - 1)
            target = (pc + offset) & 0xFFFF_FFFF_FFFF_FFFF
            return f"c.bnez  {rs1_p}, {_fmt_addr(target)}"

        return f"c.?    0x{c16:04x}"

    # --- Quadrant 2: CR / CSS ---
    if quad == 0b10:
        rd_name_q2 = gpr_name((c16 >> 7) & 0x1F)
        rs2_name = gpr_name((c16 >> 2) & 0x1F)
        rd_q2 = (c16 >> 7) & 0x1F
        rs2_q2 = (c16 >> 2) & 0x1F

        if funct3 == 0b000:  # C.SLLI (RV64C: shamt[5] = bit12)
            shamt = ((c16 >> 2) & 0x1F) | ((c16 >> 12) & 1) << 5
            if rd_q2 == 0:
                return f"c.?    0x{c16:04x}  # C.SLLI rd=0 (HINT)"
            return f"c.slli  {rd_name_q2}, {shamt}"

        if funct3 == 0b010:  # C.LWSP
            # offset = {inst[6:5], inst[12], inst[4:2], 00} (4 字节对齐)
            uimm = (
                ((c16 >> 5) & 0b11) << 6 | ((c16 >> 12) & 1) << 5 | ((c16 >> 2) & 0b111) << 2
            )
            if rd_q2 == 0:
                return f"c.?    0x{c16:04x}  # C.LWSP rd=0 (reserved)"
            return f"c.lwsp  {rd_name_q2}, {uimm}(sp)"

        if funct3 == 0b011:  # C.LDSP (RV64)
            # offset = {inst[4:2], inst[12], inst[6:5], 000} (8 字节对齐)
            uimm = (
                ((c16 >> 2) & 0b111) << 6 | ((c16 >> 12) & 1) << 5 | ((c16 >> 5) & 0b11) << 3
            )
            if rd_q2 == 0:
                return f"c.?    0x{c16:04x}  # C.LDSP rd=0 (reserved)"
            return f"c.ldsp  {rd_name_q2}, {uimm}(sp)"

        if funct3 == 0b100:  # C.JR / C.JALR / C.MV / C.EBREAK / C.ADD
            # C.EBREAK: rd=0, rs2=0 (both bit12=0 and bit12=1 are common)
            if rs2_q2 == 0 and rd_q2 == 0:
                return "c.ebreak"
            if b12 == 0:
                if rs2_q2 == 0 and rd_q2 != 0:
                    return f"c.jr    {rd_name_q2}"
                if rs2_q2 != 0 and rd_q2 != 0:
                    return f"c.mv    {rd_name_q2}, {rs2_name}"
                return f"c.?    0x{c16:04x}"
            # b12 == 1
            if rs2_q2 == 0 and rd_q2 != 0:
                return f"c.jalr  {rd_name_q2}"
            if rs2_q2 != 0 and rd_q2 != 0:
                return f"c.add   {rd_name_q2}, {rs2_name}"
            return f"c.?    0x{c16:04x}"

        if funct3 == 0b110:  # C.SWSP
            # offset = {inst[8:7], inst[12:9], 00}  (4-byte aligned)
            uimm = ((c16 >> 7) & 0b11) << 6 | ((c16 >> 9) & 0b1111) << 2
            return f"c.swsp  {rs2_name}, {uimm}(sp)"

        if funct3 == 0b111:  # C.SDSP (RV64)
            # offset = {inst[9:7], inst[12:10], 000}  (8-byte aligned)
            uimm = ((c16 >> 7) & 0b111) << 6 | ((c16 >> 10) & 0b111) << 3
            return f"c.sdsp  {rs2_name}, {uimm}(sp)"

        return f"c.?    0x{c16:04x}"

    return f"c.?    0x{c16:04x}"


# ============================================================
#  主入口
# ============================================================


def disasm(
    instr: int,
    pc: int,
) -> str:
    """反汇编一条 RISC-V 指令.

    Args:
        instr: 4 字节指令字 (小端).
        pc: 当前指令的虚拟地址 (用于计算分支/跳转目标).

    Returns:
        反汇编字符串 ("mnemonic     operands").
        未知指令返回 "<unknown opcode>".
        16-bit 压缩指令返回解码后的汇编字符串.
    """
    if parse_compressed(instr):
        return _dis_compressed(instr & 0xFFFF, pc)

    try:
        opc = Opc(parse_opcode(instr))
    except ValueError:
        # opImm32 (0b00110_11) 不在 Opc enum 中, 单独处理
        if parse_opcode(instr) == 0b00110_11:
            return _dis_op_imm32(instr)
        return _UNKNOWN

    if opc == Opc.op:
        return _dis_rtype(instr)
    if opc == Opc.opImm:
        return _dis_itype(instr)
    if opc == Opc.opImm32:
        return _dis_op_imm32(instr)
    if opc == Opc.op32:
        return _dis_op32(instr)
    if opc == Opc.ld:
        return _dis_load(instr)
    if opc == Opc.st:
        return _dis_store(instr)
    if opc == Opc.br:
        return _dis_branch(instr, pc)
    if opc == Opc.jalr:
        return _dis_jalr(instr)
    if opc == Opc.jal:
        return _dis_jal(instr, pc)
    if opc == Opc.lui:
        return _dis_lui(instr)
    if opc == Opc.auipc:
        return _dis_auipc(instr)
    if opc == Opc.sys:
        f3 = parse_func3(instr)
        if f3 == 0b000:
            return _dis_priv(instr)
        return _dis_csr(instr)
    if opc == Opc.fence:
        return _dis_fence(instr)
    if opc == Opc.amo:
        return _dis_amo(instr)

    return _UNKNOWN
