#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 20:50:45
# Last modified at 2026/06/09 星期二

"""
RISC-V RV64 反汇编器.

将 32-bit / 16-bit 指令字翻译为可读汇编字符串。
覆盖 RV64 I + M + Zicsr + AMO, 与 decoder.py 的 Hart 执行单元对齐。
"""

from collections.abc import Callable
from enum import Enum

from pyremu.utils.mask import mask64, sext, sext12
from pyremu.utils.regname import check_csr, gpr_name

_UNKNOWN = "<unk-op>"


class Opc(Enum):
    """
    如果是压缩指令，低两位不是11,而可能是00,01,10
    然后长度按照16位来解析
    """

    ld = 0b00000_11  # 载入
    ld_fp = 0b00001_11  # 浮点载入 (FLW/FLD) — LOAD-FP
    fence = 0b00011_11  # 内存/执行流屏障
    op_imm = 0b00100_11  # 立即数 ALU (32-bit)
    op_imm32 = 0b00110_11  # RV64 32-bit 立即数 ALU
    auipc = 0b00101_11  # 累加 pc
    st = 0b01000_11  # 写入
    stfp = 0b01001_11  # 浮点存储 (FSW/FSD) — STORE-FP
    amo = 0b01011_11  # 原子
    fmadd = 0b10000_11  # 合并乘加 FMADD
    fmsub = 0b10001_11  # 合并乘减 FMSUB
    fnmsub = 0b10010_11  # 负合并乘减 FNMSUB
    fnmadd = 0b10011_11  # 负合并乘加 FNMADD
    fp_op = 0b10100_11  # 浮点算术/转换/比较 OP-FP
    op = 0b01100_11  # ALU (R-type)
    lui = 0b01101_11  # 立即数载入
    op32 = 0b01110_11  # RV64 32-bit 操作
    br = 0b11000_11  # 有条件跳转
    jalr = 0b11001_11  # 无条件跳转 (寄存器)
    jal = 0b11011_11  # 无条件跳转
    sys = 0b11100_11  # ecall/ebreak/sfence.vma / CSR


# ============================================================
#  Funct3 / Funct7 enumerations
# ============================================================

AluOp = Enum(
    "AluOp",
    (
        "add",
        "sub",
        "mul",
        "mulh",
        "mulhu",
        "mulhsu",
        "div",
        "divu",
        "rem",
        "remu",
        "andi",
        "ori",
        "xor",
        "sll",
        "srl",
        "sra",
        "slti",
        "sltiu",
        "slt",
        "sltu",
    ),
)

SysOp = Enum(
    "SysOp",
    (
        "ecall",
        "ebreak",
        "csrrc",
        "csrrci",
        "csrrw",
        "csrrwi",
        "csrrs",
        "csrrsi",
        "mret",
        "sret",
        "wfi",
        "sfence_vma",
    ),
)


class BrFn3(Enum):
    beq = 0b000
    bne = 0b001
    blt = 0b100
    bge = 0b101
    bltu = 0b110
    bgeu = 0b111


class LdFn3(Enum):
    lb = 0b000
    lh = 0b001
    lw = 0b010
    ld = 0b011
    lbu = 0b100
    lhu = 0b101
    lwu = 0b110


class StFn3(Enum):
    sb = 0b000
    sh = 0b001
    sw = 0b010
    sd = 0b011


class SysFn12(Enum):
    ecall = 0
    ebreak = 1


# AMO funct5 编码 (bits 31:27), funct3 区分 32/64-bit
class AmoFunct5(Enum):
    LR = 0b00010
    SC = 0b00011
    SWAP = 0b00001
    ADD = 0b00000
    XOR = 0b00100
    AND = 0b01100
    OR = 0b01000
    MIN = 0b10000
    MAX = 0b10100
    MINU = 0b11000
    MAXU = 0b11100


class AmoWidth(Enum):
    W = 0b010  # 32-bit
    D = 0b011  # 64-bit


# 热路径优化: 预建 dict 查找表替代 Enum() 构造调用 (~0.10s 节省)
# Enum.__call__ 内部做线性搜索, dict.get 是 O(1) 哈希查找
BrFn3_MAP: dict[int, BrFn3] = {
    v.value: v
    for v in BrFn3  # type: ignore[var-annotated]
}
LdFn3_MAP: dict[int, LdFn3] = {
    v.value: v
    for v in LdFn3  # type: ignore[var-annotated]
}
StFn3_MAP: dict[int, StFn3] = {
    v.value: v
    for v in StFn3  # type: ignore[var-annotated]
}
AMOF5_MAP: dict[int, AmoFunct5] = {
    v.value: v
    for v in AmoFunct5  # type: ignore[var-annotated]
}
AMOW_MAP: dict[int, AmoWidth] = {
    v.value: v
    for v in AmoWidth  # type: ignore[var-annotated]
}

# ============================================================
#  Instruction-field extractors
# ============================================================


def parse_opcode(x: int) -> int:
    return x & 0b111_1111


def parse_rd(x: int) -> int:
    return (x >> 7) & 0b1_1111


def parse_func3(x: int) -> int:
    return (x >> 12) & 0b0111


def parse_rs1(x: int) -> int:
    return (x >> 15) & 0b1_1111


def parse_rs2(x: int) -> int:
    return (x >> 20) & 0b1_1111


def parse_func7(x: int) -> int:
    return (x >> 25) & 0b111_1111


def parse_func6(x: int) -> int:
    """for SLLI/SRLI/SRAI (I-type shifts)"""
    return (x >> 26) & 0x3F


def parse_func12(x: int) -> int:
    """I-type funct12 (bits[31:20]) — CSR / ECALL / EBREAK / etc."""
    return (x >> 20) & 0xFFF


# 立即数解析
def parse_imm12_raw(x: int) -> int:
    """I-type"""
    return (x >> 20) & 0xFFF


def parse_imm12_se(x: int) -> int:
    return sext12((x >> 20) & 0xFFF)


def parse_imm20_raw(x: int) -> int:
    """U-type"""
    return (x >> 12) & 0xF_FFFF


def parse_imm_s(instr: int) -> int:
    """S-type 12-bit immediate, sign-extended."""
    imm = ((instr >> 25) & 0x7F) << 5  # imm[11:5]
    imm |= (instr >> 7) & 0x1F  # imm[4:0]  -- from rd field
    return sext12(imm)


def parse_imm_b(instr: int) -> int:
    """B-type 13-bit immediate, sign-extended (byte-address diff)."""
    imm = ((instr >> 31) & 1) << 12  # imm[12]
    imm |= ((instr >> 25) & 0x3F) << 5  # imm[10:5]
    imm |= ((instr >> 8) & 0xF) << 1  # imm[4:1]
    imm |= ((instr >> 7) & 1) << 11  # imm[11]
    return sext(imm, 13)


def parse_imm_j(instr: int) -> int:
    """J-type 21-bit immediate, sign-extended (byte-address diff)."""
    imm = ((instr >> 31) & 1) << 20  # imm[20]
    imm |= ((instr >> 21) & 0x3FF) << 1  # imm[10:1]
    imm |= ((instr >> 20) & 1) << 11  # imm[11]
    imm |= ((instr >> 12) & 0xFF) << 12  # imm[19:12]
    return sext(imm, 21)


def parse_compressed(instr: int) -> bool:
    return (instr & 0x3) != 3


# 可能会有指令别名，不过那是反编译器关心的事情


def decode_c_sdsp(half: int) -> tuple[int, int] | None:
    """If *half* is ``c.sdsp rs2, uimm(sp)``, return ``(rs2, uimm)``; else None.

    C.SDSP encoding (C2 quadrant, RV64 only):
        bits[1:0]   = 10
        bits[4:2]   = rs2 (x8--x15 in standard, but we also accept x1)
        bits[6:5]   = uimm[5:3] lower bits
        bits[12:10] = uimm[5:3]
        bits[9:7]   = uimm[8:6]
        bits[15:13] = 111 (funct3)
    The uimm is assembled from {bits[9:7], bits[12:10]} << 3, i.e. 8-byte aligned.
    """
    if (half & 0xE003) != 0xE002:  # bits[15:13]=111, bits[1:0]=10
        return None
    rs2 = (half >> 2) & 0x1F
    uimm = ((half >> 7) & 0x7) << 6  # bits[9:7]  -> uimm[8:6]
    uimm |= ((half >> 10) & 0x7) << 3  # bits[12:10] -> uimm[5:3]
    return rs2, uimm


def decode_sd_sp(instr: int) -> tuple[int, int] | None:
    """If *instr* is ``sd rs2, imm(sp)``, return ``(rs2, imm)``; else None.

    S-type encoding:
        opcode (bits[6:0])   = 0b0100011 (STORE)
        funct3 (bits[14:12]) = 0b011     (SD / doubleword store)
        rs1    (bits[19:15]) = 2         (sp)
        rs2    (bits[24:20]) = source register
        imm[4:0]  at bits[11:7], imm[11:5] at bits[31:25]
    """
    if (instr & 0x7F) != 0b0100011:
        return None
    if ((instr >> 12) & 0x7) != 0b011:  # funct3 = SD
        return None
    if ((instr >> 15) & 0x1F) != 2:  # rs1 = sp
        return None
    rs2 = (instr >> 20) & 0x1F
    imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1F)
    if imm & 0x800:  # 按有符号立即数返回, 供调用方与 sp 相加
        imm -= 0x1000
    return rs2, imm


def _fmt_imm(val: int) -> str:
    """格式化指令立即数: 有符号十进制 (处理 64-bit 规范化值)."""
    # sext 规范化后, 负立即数以 64-bit 无符号形式传入.
    # 若 bit 63 置位则还原为有符号显示 (例如 0xFF…F0 -> -16).
    if val >= (1 << 63):
        val = val - (1 << 64)
    return str(val)


def _fmt_addr(val: int) -> str:
    """格式化绝对地址: hex."""
    return f"0x{mask64(val):x}"


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
    return f"{mnemonic:<12} {_rd(instr)}, {_rs1(instr)}, {_rs2(instr)}"


# ============================================================
#  I-type ALU (op_imm)
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
    if f3 in (0b001, 0b101):  # SLLI / SRLI / SRAI
        return _dis_shift_imm(instr, f3)

    mnemonic = _IMM_FUNCT3_MAP.get(f3)
    if mnemonic is None:
        return _UNKNOWN
    imm = parse_imm12_se(instr)
    return f"{mnemonic:<12} {_rd(instr)}, {_rs1(instr)}, {_fmt_imm(imm)}"


def _dis_shift_imm(instr: int, f3: int) -> str:
    """移位立即数: SLLI 用 funct6, SRLI 与 SRAI 由 funct6 区分.

    bit 25 属于 shamt, 故移位类指令不按 funct7 判别.
    """
    shamt = (instr >> 20) & 0x3F
    f6 = parse_func6(instr)
    if f3 == 0b001:
        if f6 != 0:
            return _UNKNOWN
        return f"slli         {_rd(instr)}, {_rs1(instr)}, {shamt}"
    if f6 == 0x00:  # SRLI (funct6=0b000000)
        return f"srli         {_rd(instr)}, {_rs1(instr)}, {shamt}"
    if f6 == 0x10:  # SRAI (funct6=0b010000)
        return f"srai         {_rd(instr)}, {_rs1(instr)}, {shamt}"
    return _UNKNOWN


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
    return f"{mnemonic:<12} {_rd(instr)}, {_rs1(instr)}, {_rs2(instr)}"


def _dis_op_imm32(instr: int) -> str:
    """RV64 32-bit I-type: ADDIW / SLLIW / SRLIW / SRAIW."""
    f3 = parse_func3(instr)
    f7 = parse_func7(instr)
    if f3 == 0b000:
        imm = parse_imm12_se(instr)
        return f"addiw        {_rd(instr)}, {_rs1(instr)}, {_fmt_imm(imm)}"
    if f3 == 0b001:
        if f7 != 0:
            return _UNKNOWN
        shamt = (instr >> 20) & 0x1F
        return f"slliw        {_rd(instr)}, {_rs1(instr)}, {shamt}"
    if f3 == 0b101:
        shamt = (instr >> 20) & 0x1F
        if f7 == 0x00:
            return f"srliw        {_rd(instr)}, {_rs1(instr)}, {shamt}"
        if f7 == 0x20:
            return f"sraiw        {_rd(instr)}, {_rs1(instr)}, {shamt}"
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
    return f"{mnemonic:<12} {_rd(instr)}, {_fmt_imm(offset)}({_rs1(instr)})"


def _dis_store(instr: int) -> str:
    """Store: mnemonic rs2, offset(rs1)."""
    f3 = parse_func3(instr)
    try:
        mnemonic = StFn3(f3).name
    except ValueError:
        return _UNKNOWN
    offset = parse_imm_s(instr)
    return f"{mnemonic:<12} {_rs2(instr)}, {_fmt_imm(offset)}({_rs1(instr)})"


# ============================================================
#  Branch
# ============================================================


def _dis_branch(instr: int, pc: int) -> str:
    """Branch: mnemonic rs1, rs2, target_addr."""
    f3 = parse_func3(instr)
    try:
        mnemonic = BrFn3(f3).name
    except ValueError:
        return _UNKNOWN
    offset = parse_imm_b(instr)
    target = mask64((pc + offset))
    return f"{mnemonic:<12} {_rs1(instr)}, {_rs2(instr)}, {_fmt_addr(target)}"


# ============================================================
#  JAL / JALR
# ============================================================


def _dis_jal(instr: int, pc: int) -> str:
    """JAL: mnemonic rd, target_addr."""
    offset = parse_imm_j(instr)
    target = mask64((pc + offset))
    return f"jal          {_rd(instr)}, {_fmt_addr(target)}"


def _dis_jalr(instr: int) -> str:
    """JALR: mnemonic rd, offset(rs1)."""
    offset = parse_imm12_se(instr)
    return f"jalr         {_rd(instr)}, {_fmt_imm(offset)}({_rs1(instr)})"


# ============================================================
#  LUI / AUIPC
# ============================================================


def _dis_lui(instr: int) -> str:
    # LUI 的 20-bit 立即数为地址高位常量 (实际值 = imm << 12), 用 hex 更直观
    imm = parse_imm20_raw(instr)
    return f"lui          {_rd(instr)}, {_fmt_addr(imm)}"


def _dis_auipc(instr: int) -> str:
    # AUIPC 同上, upper immediate 用 hex 表示
    imm = parse_imm20_raw(instr)
    return f"auipc        {_rd(instr)}, {_fmt_addr(imm)}"


_CSR_MNEMONIC: dict[int, str] = {
    0b001: "csrrw",
    0b010: "csrrs",
    0b011: "csrrc",
    0b101: "csrrwi",
    0b110: "csrrsi",
    0b111: "csrrci",
}


def _csr_name(addr: int) -> str:
    """由 CSR 地址得到名称, 未找到则返回地址的十六进制形式."""
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
        return f"{mnemonic:<12} {rd_name}, {csr}, {uimm}"
    return f"{mnemonic:<12} {rd_name}, {csr}, {_rs1(instr)}"


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
    return f"{mnemonic:<12} {_rd(instr)}, {_rs2(instr)}, ({_rs1(instr)})"


# ============================================================
#  压缩指令 (C extension) 反汇编
# ============================================================
# 寄存器: x8-x15 使用 3-bit 缩写索引 rd′/rs1′/rs2′.


def _c_x8(idx3: int) -> str:
    """由 3-bit 压缩寄存器索引得到 ABI 名, 即 x8 到 x15."""
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
    if quad == 0b00:
        return _dis_c0(c16)
    if quad == 0b01:
        return _dis_c1(c16, pc)
    if quad == 0b10:
        return _dis_c2(c16, pc)
    return _c_unknown(c16)


def _c_unknown(c16: int) -> str:
    return f"c.?    0x{c16:04x}"


# ---- 象限 0: CIW / CL / CS ----


def _c0_addi4spn(c16: int) -> str:
    rd_p = _c_x8(c16 >> 2)
    # 基址寄存器固定为 sp (x2), bits[9:7] 属于立即数而非 rs1' 字段.
    uimm = (
        ((c16 >> 7) & 0x0F) << 6  # instr[10:7] -> nzuimm[9:6]
        | ((c16 >> 11) & 0x03) << 4  # instr[12:11] -> nzuimm[5:4]
        | ((c16 >> 5) & 1) << 3  # instr[5] -> nzuimm[3]
        | ((c16 >> 6) & 1) << 2  # instr[6] -> nzuimm[2]
    )
    if uimm == 0:
        return f"c.?    0x{c16:04x}  # C.ADDI4SPN nzuimm=0 (reserved)"
    return f"c.addi4spn {rd_p}, sp, {uimm}"


def _c0_fld(c16: int) -> str:
    uimm = ((c16 >> 5) & 0b11) << 6 | ((c16 >> 10) & 0b111) << 3
    fpr_rd = 8 + ((c16 >> 2) & 0b111)
    return f"c.fld        f{fpr_rd}, {uimm}({_c_x8(c16 >> 7)})"


def _c0_lw(c16: int) -> str:
    uimm = ((c16 >> 5) & 1) << 6 | ((c16 >> 10) & 0b111) << 3 | ((c16 >> 6) & 1) << 2
    return f"c.lw         {_c_x8(c16 >> 2)}, {uimm}({_c_x8(c16 >> 7)})"


def _c0_ld(c16: int) -> str:
    uimm = ((c16 >> 5) & 0b11) << 6 | ((c16 >> 10) & 0b111) << 3
    return f"c.ld         {_c_x8(c16 >> 2)}, {uimm}({_c_x8(c16 >> 7)})"


def _c0_fsd(c16: int) -> str:
    uimm = ((c16 >> 5) & 0b11) << 6 | ((c16 >> 10) & 0b111) << 3
    fpr_rs2 = 8 + ((c16 >> 2) & 0b111)
    return f"c.fsd        f{fpr_rs2}, {uimm}({_c_x8(c16 >> 7)})"


def _c0_sw(c16: int) -> str:
    uimm = ((c16 >> 5) & 1) << 6 | ((c16 >> 10) & 0b111) << 3 | ((c16 >> 6) & 1) << 2
    return f"c.sw         {_c_x8(c16 >> 2)}, {uimm}({_c_x8(c16 >> 7)})"


def _c0_sd(c16: int) -> str:
    uimm = ((c16 >> 5) & 0b11) << 6 | ((c16 >> 10) & 0b111) << 3
    return f"c.sd         {_c_x8(c16 >> 2)}, {uimm}({_c_x8(c16 >> 7)})"


_C0_HANDLERS: dict[int, Callable[[int], str]] = {
    0b000: _c0_addi4spn,
    0b001: _c0_fld,  # C.FLD (RV64DC)
    0b010: _c0_lw,
    0b011: _c0_ld,  # RV64
    0b101: _c0_fsd,  # C.FSD (RV64DC)
    0b110: _c0_sw,
    0b111: _c0_sd,  # RV64
}


def _dis_c0(c16: int) -> str:
    """象限 0: C.ADDI4SPN / C.FLD / C.LW / C.LD / C.FSD / C.SW / C.SD."""
    handler = _C0_HANDLERS.get((c16 >> 13) & 0b111)
    if handler is None:
        return _c_unknown(c16)
    return handler(c16)


# ---- 象限 1: CI / CJ / CB ----


def _c1_imm6(c16: int) -> int:
    """bits[12] 与 bits[6:2] 组成的 6 位立即数, 按 bit 5 符号扩展."""
    imm6 = ((c16 >> 12) & 1) << 5 | ((c16 >> 2) & 0x1F)
    return (imm6 & 0x20) and (imm6 | ~0x3F) or imm6


def _c1_addi(c16: int, pc: int) -> str:
    rd_raw = (c16 >> 7) & 0x1F
    imm6_se = _c1_imm6(c16)
    if imm6_se == 0 and rd_raw == 0:  # rd=x0, imm=0 -> C.NOP
        return "c.nop"
    return f"c.addi       {gpr_name(rd_raw)}, {imm6_se}"


def _c1_addiw(c16: int, pc: int) -> str:
    return f"c.addiw {gpr_name((c16 >> 7) & 0x1F)}, {_c1_imm6(c16)}"


def _c1_li(c16: int, pc: int) -> str:
    return f"c.li         {gpr_name((c16 >> 7) & 0x1F)}, {_c1_imm6(c16)}"


def _c1_addi16sp(c16: int) -> str:
    # nzimm[9:4] encoding per RISC-V spec:
    nzimm = (
        ((c16 >> 12) & 1) << 9  # nzimm[9]
        | ((c16 >> 3) & 0b11) << 7  # nzimm[8:7]
        | ((c16 >> 5) & 1) << 6  # nzimm[6]
        | ((c16 >> 2) & 1) << 5  # nzimm[5]
        | ((c16 >> 6) & 1) << 4  # nzimm[4]
    )
    if nzimm & (1 << 9):
        nzimm |= ~((1 << 10) - 1)
    return f"c.addi16sp sp, {nzimm}"


def _c1_lui(c16: int, pc: int) -> str:
    """C.LUI (rd≠{0,2}) 与 C.ADDI16SP (rd=2) 共用 funct3=0b011."""
    rd_raw = (c16 >> 7) & 0x1F
    if rd_raw == 2:
        return _c1_addi16sp(c16)
    nzuimm = ((c16 >> 12) & 1) << 17 | ((c16 >> 2) & 0x1F) << 12
    # sext from bit 17
    if nzuimm & (1 << 17):
        nzuimm = nzuimm - (1 << 18)
    if nzuimm == 0:
        return f"c.?    0x{c16:04x}  # C.LUI nzuimm=0 (reserved)"
    return f"c.lui        {gpr_name(rd_raw)}, 0x{(nzuimm >> 12) & 0x3F:x}"


_C1_ALU_REG_MNEMONICS: dict[int, str] = {
    0b00: "c.sub",
    0b01: "c.xor",
    0b10: "c.or",
    0b11: "c.and",
}


def _c1_alu(c16: int, pc: int) -> str:
    """C.SRLI / C.SRAI / C.ANDI 与寄存器间的 C.SUB / C.XOR / C.OR / C.AND."""
    rd_p = _c_x8(c16 >> 7)
    b11_10 = (c16 >> 10) & 0b11
    if b11_10 == 0b00:  # C.SRLI
        uimm = ((c16 >> 2) & 0x1F) | ((c16 >> 12) & 1) << 5
        return f"c.srli       {rd_p}, {uimm}"
    if b11_10 == 0b01:  # C.SRAI
        uimm = ((c16 >> 2) & 0x1F) | ((c16 >> 12) & 1) << 5
        return f"c.srai       {rd_p}, {uimm}"
    if b11_10 == 0b10:  # C.ANDI
        return f"c.andi       {rd_p}, {_c1_imm6(c16)}"
    mn = _C1_ALU_REG_MNEMONICS[(c16 >> 5) & 0b11]
    return f"{mn:<12} {rd_p}, {_c_x8(c16 >> 2)}"


def _c1_jump(c16: int, pc: int) -> str:
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
    return f"c.j          {_fmt_addr(mask64(pc + offset))}"


def _c1_branch_offset(c16: int) -> int:
    """C.BEQZ 与 C.BNEZ 共用的 9 位分支偏移."""
    offset = (
        ((c16 >> 12) & 1) << 8
        | ((c16 >> 10) & 0b11) << 6
        | ((c16 >> 5) & 0b11) << 4
        | ((c16 >> 3) & 0b11) << 1
    )
    if offset & (1 << 8):
        offset |= ~((1 << 9) - 1)
    return offset


def _c1_beqz(c16: int, pc: int) -> str:
    target = mask64(pc + _c1_branch_offset(c16))
    return f"c.beqz       {_c_x8(c16 >> 7)}, {_fmt_addr(target)}"


def _c1_bnez(c16: int, pc: int) -> str:
    target = mask64(pc + _c1_branch_offset(c16))
    return f"c.bnez       {_c_x8(c16 >> 7)}, {_fmt_addr(target)}"


_C1_HANDLERS: dict[int, Callable[[int, int], str]] = {
    0b000: _c1_addi,  # C.NOP / C.ADDI
    0b001: _c1_addiw,  # RV64
    0b010: _c1_li,
    0b011: _c1_lui,
    0b100: _c1_alu,  # SRLI / SRAI / ANDI / SUB / XOR / OR / AND
    0b101: _c1_jump,
    0b110: _c1_beqz,
    0b111: _c1_bnez,
}


def _dis_c1(c16: int, pc: int) -> str:
    """象限 1: C.NOP / C.ADDI / C.ADDIW / C.LI / C.LUI / C.ADDI16SP / 移位与分支."""
    handler = _C1_HANDLERS.get((c16 >> 13) & 0b111)
    if handler is None:
        return _c_unknown(c16)
    return handler(c16, pc)


# ---- 象限 2: CR / CSS ----


def _c2_slli(c16: int, pc: int) -> str:
    # rd=x0 is a HINT (executed as NOP), but display apparent semantics
    shamt = ((c16 >> 2) & 0x1F) | ((c16 >> 12) & 1) << 5
    return f"c.slli       {gpr_name((c16 >> 7) & 0x1F)}, {shamt}"


def _c2_fldsp(c16: int, pc: int) -> str:
    uimm = ((c16 >> 2) & 0b111) << 6 | ((c16 >> 12) & 1) << 5 | ((c16 >> 5) & 0b11) << 3
    # rd=0 -> ft0, valid FP destination (unlike GPR loads where x0 is reserved)
    return f"c.fldsp      f{(c16 >> 7) & 0x1F}, {uimm}(sp)"


def _c2_lwsp(c16: int, pc: int) -> str:
    # offset = {inst[6:5], inst[12], inst[4:2], 00} (4 字节对齐)
    uimm = ((c16 >> 5) & 0b11) << 6 | ((c16 >> 12) & 1) << 5 | ((c16 >> 2) & 0b111) << 2
    rd_raw = (c16 >> 7) & 0x1F
    if rd_raw == 0:
        return f"c.?    0x{c16:04x}  # C.LWSP rd=0 (reserved)"
    return f"c.lwsp       {gpr_name(rd_raw)}, {uimm}(sp)"


def _c2_ldsp(c16: int, pc: int) -> str:
    # offset = {inst[4:2], inst[12], inst[6:5], 000} (8 字节对齐)
    uimm = ((c16 >> 2) & 0b111) << 6 | ((c16 >> 12) & 1) << 5 | ((c16 >> 5) & 0b11) << 3
    rd_raw = (c16 >> 7) & 0x1F
    if rd_raw == 0:
        return f"c.?    0x{c16:04x}  # C.LDSP rd=0 (reserved)"
    return f"c.ldsp       {gpr_name(rd_raw)}, {uimm}(sp)"


def _c2_misc(c16: int, pc: int) -> str:
    """C.JR / C.JALR / C.MV / C.ADD / C.EBREAK."""
    rd_raw = (c16 >> 7) & 0x1F
    rs2_raw = (c16 >> 2) & 0x1F
    if rd_raw == 0:
        return "c.ebreak" if rs2_raw == 0 else _c_unknown(c16)
    rd_name = gpr_name(rd_raw)
    if rs2_raw == 0:  # bit12 区分 C.JR 与 C.JALR
        return f"c.jalr       {rd_name}" if c16 & 0x1000 else f"c.jr         {rd_name}"
    rs2_name = gpr_name(rs2_raw)
    if c16 & 0x1000:  # bit12 区分 C.MV 与 C.ADD
        return f"c.add        {rd_name}, {rs2_name}"
    return f"c.mv         {rd_name}, {rs2_name}"


def _c2_fsdsp(c16: int, pc: int) -> str:
    uimm = ((c16 >> 7) & 0b111) << 6 | ((c16 >> 10) & 0b111) << 3
    return f"c.fsdsp      f{(c16 >> 2) & 0x1F}, {uimm}(sp)"


def _c2_swsp(c16: int, pc: int) -> str:
    # offset = {inst[8:7], inst[12:9], 00}  (4-byte aligned)
    uimm = ((c16 >> 7) & 0b11) << 6 | ((c16 >> 9) & 0b1111) << 2
    return f"c.swsp       {gpr_name((c16 >> 2) & 0x1F)}, {uimm}(sp)"


def _c2_sdsp(c16: int, pc: int) -> str:
    # offset = {inst[9:7], inst[12:10], 000}  (8-byte aligned)
    uimm = ((c16 >> 7) & 0b111) << 6 | ((c16 >> 10) & 0b111) << 3
    return f"c.sdsp       {gpr_name((c16 >> 2) & 0x1F)}, {uimm}(sp)"


_C2_HANDLERS: dict[int, Callable[[int, int], str]] = {
    0b000: _c2_slli,  # RV64C: shamt[5] = bit12
    0b001: _c2_fldsp,  # C.FLDSP (RV64DC)
    0b010: _c2_lwsp,
    0b011: _c2_ldsp,  # RV64
    0b100: _c2_misc,
    0b101: _c2_fsdsp,  # C.FSDSP (RV64DC)
    0b110: _c2_swsp,
    0b111: _c2_sdsp,  # RV64
}


def _dis_c2(c16: int, pc: int) -> str:
    """象限 2: C.SLLI 与 sp 相对的载入/存储, 以及 C.EBREAK 一族."""
    handler = _C2_HANDLERS.get((c16 >> 13) & 0b111)
    if handler is None:
        return _c_unknown(c16)
    return handler(c16, pc)


# ============================================================
#  F/D floating-point disassembly
# ============================================================

_FP_LOAD_MNEMONIC = {0b010: "flw", 0b011: "fld"}
_FP_STORE_MNEMONIC = {0b010: "fsw", 0b011: "fsd"}
# OP-FP funct7[6:2] ->助记符基名 (S/D 后缀由 fmt 决定)
_FP_ARITH = {0x00: "fadd", 0x01: "fsub", 0x02: "fmul", 0x03: "fdiv"}
_FP_SGNJ = {0: "fsgnj", 1: "fsgnjn", 2: "fsgnjx"}
_FP_MINMAX = {0: "fmin", 1: "fmax"}
_FP_CMP = {0: "fle", 1: "flt", 2: "feq"}
_FP_FMA_MNEMONIC = {
    0b10000_11: "fmadd",
    0b10001_11: "fmsub",
    0b10010_11: "fnmsub",
    0b10011_11: "fnmadd",
}


def _fpr_name(idx: int) -> str:
    return f"f{idx}"


def _fsuffix(fmt: int) -> str:
    return "d" if fmt == 1 else "s"


def _dis_fp_load(instr: int) -> str:
    """FLW / FLD: mnemonic frd, offset(rs1)."""
    m = _FP_LOAD_MNEMONIC.get(parse_func3(instr))
    if m is None:
        return _UNKNOWN
    off = parse_imm12_se(instr)
    return f"{m:<12} {_fpr_name(parse_rd(instr))}, {_fmt_imm(off)}({_rs1(instr)})"


def _dis_fp_store(instr: int) -> str:
    """FSW / FSD: mnemonic frs2, offset(rs1)."""
    m = _FP_STORE_MNEMONIC.get(parse_func3(instr))
    if m is None:
        return _UNKNOWN
    off = parse_imm_s(instr)
    return f"{m:<12} {_fpr_name(parse_rs2(instr))}, {_fmt_imm(off)}({_rs1(instr)})"


def _dis_fp_fma(instr: int) -> str:
    """FMADD/FMSUB/FNMSUB/FNMADD: mnemonic.fmt frd, frs1, frs2, frs3."""
    base = _FP_FMA_MNEMONIC.get(parse_opcode(instr))
    if base is None:
        return _UNKNOWN
    fmt = (instr >> 25) & 0x3
    rs3 = (instr >> 27) & 0x1F
    m = f"{base}.{_fsuffix(fmt)}"
    return (
        f"{m:<12} {_fpr_name(parse_rd(instr))}, {_fpr_name(parse_rs1(instr))}, "
        f"{_fpr_name(parse_rs2(instr))}, {_fpr_name(rs3)}"
    )


def _fp_operands(instr: int) -> tuple[str, str, str]:
    """OP-FP 的目的与两个源浮点寄存器名."""
    return (
        _fpr_name(parse_rd(instr)),
        _fpr_name(parse_rs1(instr)),
        _fpr_name(parse_rs2(instr)),
    )


def _fp_arith(instr: int) -> str:
    funct7 = parse_func7(instr)
    m = f"{_FP_ARITH[funct7 >> 2]}.{_fsuffix(funct7 & 0x3)}"
    frd, frs1, frs2 = _fp_operands(instr)
    return f"{m:<12} {frd}, {frs1}, {frs2}"


def _fp_sqrt(instr: int) -> str:
    funct7 = parse_func7(instr)
    frd, frs1, _ = _fp_operands(instr)
    return f"{'fsqrt.' + _fsuffix(funct7 & 0x3):<12} {frd}, {frs1}"


def _fp_sgnj(instr: int) -> str:
    fmt = parse_func7(instr) & 0x3
    m = _FP_SGNJ.get(parse_func3(instr), "fsgnj?")
    frd, frs1, frs2 = _fp_operands(instr)
    return f"{m + '.' + _fsuffix(fmt):<12} {frd}, {frs1}, {frs2}"


def _fp_minmax(instr: int) -> str:
    fmt = parse_func7(instr) & 0x3
    m = _FP_MINMAX.get(parse_func3(instr), "fmin?")
    frd, frs1, frs2 = _fp_operands(instr)
    return f"{m + '.' + _fsuffix(fmt):<12} {frd}, {frs1}, {frs2}"


def _fp_cmp(instr: int) -> str:
    """FCMP 类指令的结果写入整数寄存器."""
    funct7 = parse_func7(instr)
    m = _FP_CMP.get(parse_func3(instr), "fcmp?")
    _, frs1, frs2 = _fp_operands(instr)
    return f"{m + '.' + _fsuffix(funct7 & 0x3):<12} {_rd(instr)}, {frs1}, {frs2}"


def _fp_cvt_f2i(instr: int) -> str:
    """浮点到整数的转换, 结果写入整数寄存器."""
    funct7 = parse_func7(instr)
    w = {0: "w", 1: "wu", 2: "l", 3: "lu"}.get(parse_rs2(instr), "?")
    frs1 = _fpr_name(parse_rs1(instr))
    return f"{'fcvt.' + w + '.' + _fsuffix(funct7 & 0x3):<12} {_rd(instr)}, {frs1}"


def _fp_cvt_i2f(instr: int) -> str:
    """整数到浮点的转换, 源为整数寄存器."""
    funct7 = parse_func7(instr)
    w = {0: "w", 1: "wu", 2: "l", 3: "lu"}.get(parse_rs2(instr), "?")
    frd = _fpr_name(parse_rd(instr))
    return f"{'fcvt.' + _fsuffix(funct7 & 0x3) + '.' + w:<12} {frd}, {_rs1(instr)}"


def _fp_cvt_sd(instr: int) -> str:
    """FCVT.S.D 与 FCVT.D.S 之间的精度转换."""
    m = "fcvt.d.s" if parse_func7(instr) & 0x3 == 1 else "fcvt.s.d"
    frd, frs1, _ = _fp_operands(instr)
    return f"{m:<12} {frd}, {frs1}"


def _fp_class_mvx(instr: int) -> str:
    """FMV.X.W / FMV.X.D 与 FCLASS, 结果写入整数寄存器."""
    is_single = parse_func7(instr) & 0x3 == 0
    if parse_func3(instr) == 0:
        m = "fmv.x.w" if is_single else "fmv.x.d"
    else:
        m = "fclass.s" if is_single else "fclass.d"
    frs1 = _fpr_name(parse_rs1(instr))
    return f"{m:<12} {_rd(instr)}, {frs1}"


def _fp_mvfx(instr: int) -> str:
    """FMV.W.X / FMV.D.X, 源为整数寄存器."""
    m = "fmv.w.x" if parse_func7(instr) & 0x3 == 0 else "fmv.d.x"
    frd = _fpr_name(parse_rd(instr))
    return f"{m:<12} {frd}, {_rs1(instr)}"


_FP_OP_HANDLERS: dict[int, Callable[[int], str]] = {
    0x00: _fp_arith,
    0x01: _fp_arith,
    0x02: _fp_arith,
    0x03: _fp_arith,
    0x04: _fp_sgnj,  # FSGNJ*
    0x05: _fp_minmax,  # FMIN/FMAX
    0x08: _fp_cvt_sd,  # FCVT.S.D / FCVT.D.S
    0x0B: _fp_sqrt,  # FSQRT
    0x14: _fp_cmp,  # FCMP -> GPR rd
    0x18: _fp_cvt_f2i,  # float -> int
    0x1A: _fp_cvt_i2f,  # int -> float
    0x1C: _fp_class_mvx,  # FMV.X.* / FCLASS -> GPR rd
    0x1E: _fp_mvfx,  # FMV.*.X (GPR rs1 -> FPR)
}


def _dis_fp_op(instr: int) -> str:
    """OP-FP: 算术/转换/比较/符号/分类/移动."""
    handler = _FP_OP_HANDLERS.get(parse_func7(instr) >> 2)
    if handler is None:
        return _UNKNOWN
    return handler(instr)


# ============================================================
#  主入口
# ============================================================


_DISASM_HANDLERS: dict[Opc, Callable[[int], str]] = {
    Opc.op: _dis_rtype,
    Opc.op_imm: _dis_itype,
    Opc.op_imm32: _dis_op_imm32,
    Opc.op32: _dis_op32,
    Opc.ld: _dis_load,
    Opc.st: _dis_store,
    Opc.jalr: _dis_jalr,
    Opc.lui: _dis_lui,
    Opc.auipc: _dis_auipc,
    Opc.fence: _dis_fence,
    Opc.amo: _dis_amo,
    Opc.ld_fp: _dis_fp_load,
    Opc.stfp: _dis_fp_store,
    Opc.fp_op: _dis_fp_op,
    Opc.fmadd: _dis_fp_fma,
    Opc.fmsub: _dis_fp_fma,
    Opc.fnmsub: _dis_fp_fma,
    Opc.fnmadd: _dis_fp_fma,
}

# 分支与跳转的助记符需要 pc 计算目标地址, 故单独一张表.
_DISASM_PC_HANDLERS: dict[Opc, Callable[[int, int], str]] = {
    Opc.br: _dis_branch,
    Opc.jal: _dis_jal,
}


def _dis_unknown(instr: int) -> str:
    return _UNKNOWN


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
        未知指令返回 "<unk-op>".
        16-bit 压缩指令返回解码后的汇编字符串.
    """
    if parse_compressed(instr):
        return _dis_compressed(instr & 0xFFFF, pc)

    try:
        opc = Opc(parse_opcode(instr))
    except ValueError:
        # op_imm32 (0b00110_11) 不在 Opc enum 中, 单独处理
        return _dis_op_imm32(instr) if parse_opcode(instr) == 0b00110_11 else _UNKNOWN

    if opc == Opc.sys:
        return _dis_priv(instr) if parse_func3(instr) == 0b000 else _dis_csr(instr)
    pc_handler = _DISASM_PC_HANDLERS.get(opc)
    if pc_handler is not None:
        return pc_handler(instr, pc)
    return _DISASM_HANDLERS.get(opc, _dis_unknown)(instr)
