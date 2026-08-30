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
from enum import Enum

from pyremu.utils.mask import mask64, sext, sext12
from pyremu.utils.regname import check_csr, gpr_name

_UNKNOWN = "<unknown opcode>"


# ============================================================
#  Opcodes
# ============================================================


class Opc(Enum):
    """
    如果是压缩指令，低两位不是11,而可能是00,01,10
    然后长度按照16位来解析
    """

    ld = 0b00000_11  # 载入
    opfp = 0b00001_11  # 浮点载入 (FLW/FLD) — LOAD-FP
    fence = 0b00011_11  # 内存/执行流屏障
    opImm = 0b00100_11  # 立即数 ALU (32-bit)
    opImm32 = 0b00110_11  # RV64 32-bit 立即数 ALU
    auipc = 0b00101_11  # 累加 pc
    st = 0b01000_11  # 写入
    stfp = 0b01001_11  # 浮点存储 (FSW/FSD) — STORE-FP
    amo = 0b01011_11  # 原子
    fmadd = 0b10000_11  # 融合乘加 FMADD
    fmsub = 0b10001_11  # 融合乘减 FMSUB
    fnmsub = 0b10010_11  # 负融合乘减 FNMSUB
    fnmadd = 0b10011_11  # 负融合乘加 FNMADD
    opFp = 0b10100_11  # 浮点算术/转换/比较 OP-FP
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

aluOp = Enum(
    "aluOp",
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

sysOp = Enum(
    "sysOp",
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


class brFn3(Enum):
    beq = 0b000
    bne = 0b001
    blt = 0b100
    bge = 0b101
    bltu = 0b110
    bgeu = 0b111


class ldFn3(Enum):
    lb = 0b000
    lh = 0b001
    lw = 0b010
    ld = 0b011
    lbu = 0b100
    lhu = 0b101
    lwu = 0b110


class stFn3(Enum):
    sb = 0b000
    sh = 0b001
    sw = 0b010
    sd = 0b011


class sysFn12(Enum):
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
BRFN3_MAP: dict[int, brFn3] = {
    v.value: v for v in brFn3  # type: ignore[var-annotated]
}
LDFN3_MAP: dict[int, ldFn3] = {
    v.value: v for v in ldFn3  # type: ignore[var-annotated]
}
STFN3_MAP: dict[int, stFn3] = {
    v.value: v for v in stFn3  # type: ignore[var-annotated]
}
AMOF5_MAP: dict[int, AmoFunct5] = {
    v.value: v for v in AmoFunct5  # type: ignore[var-annotated]
}
AMOW_MAP: dict[int, AmoWidth] = {
    v.value: v for v in AmoWidth  # type: ignore[var-annotated]
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
    """ for SLLI/SRLI/SRAI (I-type shifts) """
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
    uimm = ((half >> 7) & 0x7) << 6   # bits[9:7]  -> uimm[8:6]
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
    if ((instr >> 15) & 0x1F) != 2:     # rs1 = sp
        return None
    rs2 = (instr >> 20) & 0x1F
    imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1F)
    imm = (imm << 52) >> 52  # sign-extend 12-bit
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
        return f"slli         {_rd(instr)}, {_rs1(instr)}, {shamt}"
    elif f3 == 0b101:  # SRLI / SRAI
        shamt = (instr >> 20) & 0x3F
        f6 = parse_func6(instr)
        if f6 == 0x00:  # SRLI (funct6=0b000000)
            return f"srli         {_rd(instr)}, {_rs1(instr)}, {shamt}"
        if f6 == 0x10:  # SRAI (funct6=0b010000)
            return f"srai         {_rd(instr)}, {_rs1(instr)}, {shamt}"
        return _UNKNOWN

    parse_func7(instr)  # 非 shift I-type 才使用 funct7

    mnemonic = _IMM_FUNCT3_MAP.get(f3)
    if mnemonic is None:
        return _UNKNOWN
    imm = parse_imm12_se(instr)
    return f"{mnemonic:<12} {_rd(instr)}, {_rs1(instr)}, {_fmt_imm(imm)}"


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
        mnemonic = stFn3(f3).name
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
        mnemonic = brFn3(f3).name
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
    """3-bit 压缩寄存器索引 -> ABI 名 (x8-x15)."""
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

        if funct3 == 0b001:  # C.FLD (RV64DC)
            uimm = ((c16 >> 5) & 0b11) << 6 | ((c16 >> 10) & 0b111) << 3
            fpr_rd = 8 + ((c16 >> 2) & 0b111)
            return f"c.fld        f{fpr_rd}, {uimm}({rs1_p})"

        if funct3 == 0b010:  # C.LW
            uimm = ((c16 >> 5) & 1) << 6 | ((c16 >> 10) & 0b111) << 3 | ((c16 >> 6) & 1) << 2
            return f"c.lw         {rd_p}, {uimm}({rs1_p})"

        if funct3 == 0b011:  # C.LD (RV64)
            uimm = ((c16 >> 5) & 0b11) << 6 | ((c16 >> 10) & 0b111) << 3
            return f"c.ld         {rd_p}, {uimm}({rs1_p})"

        if funct3 == 0b101:  # C.FSD (RV64DC)
            uimm = ((c16 >> 5) & 0b11) << 6 | ((c16 >> 10) & 0b111) << 3
            fpr_rs2 = 8 + ((c16 >> 2) & 0b111)
            return f"c.fsd        f{fpr_rs2}, {uimm}({rs1_p})"

        if funct3 == 0b110:  # C.SW
            rs2_p = _c_x8(c16 >> 2)
            uimm = ((c16 >> 5) & 1) << 6 | ((c16 >> 10) & 0b111) << 3 | ((c16 >> 6) & 1) << 2
            return f"c.sw         {rs2_p}, {uimm}({rs1_p})"

        if funct3 == 0b111:  # C.SD (RV64)
            rs2_p = _c_x8(c16 >> 2)
            uimm = ((c16 >> 5) & 0b11) << 6 | ((c16 >> 10) & 0b111) << 3
            return f"c.sd         {rs2_p}, {uimm}({rs1_p})"

        return f"c.?    0x{c16:04x}"

    # --- Quadrant 1: CI / CJ / CB ---
    elif quad == 0b01:
        rd_name = gpr_name((c16 >> 7) & 0x1F)
        rs1_p = _c_x8(c16 >> 7)
        imm6 = ((c16 >> 12) & 1) << 5 | ((c16 >> 2) & 0x1F)
        imm6_se = (imm6 & 0x20) and (imm6 | ~0x3F) or imm6  # sext 6-bit

        if funct3 == 0b000:  # C.NOP / C.ADDI
            if imm6 == 0 and ((c16 >> 7) & 0x1F) == 0:  # rd=x0, imm=0 -> C.NOP
                return "c.nop"
            return f"c.addi       {rd_name}, {imm6_se}"

        if funct3 == 0b001:  # C.ADDIW (RV64)
            return f"c.addiw {rd_name}, {imm6_se}"

        if funct3 == 0b010:  # C.LI
            return f"c.li         {rd_name}, {imm6_se}"

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
            return f"c.lui        {rd_name}, 0x{(val >> 12) & 0x3F:x}"

        if funct3 == 0b100:  # misc ALU: SRLI / SRAI / ANDI / SUB/XOR/OR/AND
            rd_p = _c_x8(c16 >> 7)
            shamt5 = ((c16 >> 12) & 1) << 5  # bit 12 -> shamt[5] for RV64C
            uimm = ((c16 >> 2) & 0x1F) | shamt5
            imm6_se_alt = ((c16 >> 12) & 1) << 5 | ((c16 >> 2) & 0x1F)
            imm6_se_alt = (imm6_se_alt & 0x20) and (imm6_se_alt | ~0x3F) or imm6_se_alt

            if b11_10 == 0b00:  # C.SRLI (RV64C: shamt[5] must be 0 for RV32)
                return f"c.srli       {rd_p}, {uimm}"
            if b11_10 == 0b01:  # C.SRAI
                return f"c.srai       {rd_p}, {uimm}"
            if b11_10 == 0b10:  # C.ANDI
                return f"c.andi       {rd_p}, {imm6_se_alt}"
            if b11_10 == 0b11:  # register ops
                b6_5 = (c16 >> 5) & 0b11
                rs2_p = _c_x8(c16 >> 2)
                mn = {0b00: "c.sub", 0b01: "c.xor", 0b10: "c.or", 0b11: "c.and"}[b6_5]
                return f"{mn:<12} {rd_p}, {rs2_p}"

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
            target = mask64((pc + offset))
            return f"c.j          {_fmt_addr(target)}"

        if funct3 == 0b110:  # C.BEQZ
            offset = (
                ((c16 >> 12) & 1) << 8
                | ((c16 >> 10) & 0b11) << 6
                | ((c16 >> 5) & 0b11) << 4
                | ((c16 >> 3) & 0b11) << 1
            )
            if offset & (1 << 8):
                offset |= ~((1 << 9) - 1)
            target = mask64((pc + offset))
            return f"c.beqz       {rs1_p}, {_fmt_addr(target)}"

        if funct3 == 0b111:  # C.BNEZ
            offset = (
                ((c16 >> 12) & 1) << 8
                | ((c16 >> 10) & 0b11) << 6
                | ((c16 >> 5) & 0b11) << 4
                | ((c16 >> 3) & 0b11) << 1
            )
            if offset & (1 << 8):
                offset |= ~((1 << 9) - 1)
            target = mask64((pc + offset))
            return f"c.bnez       {rs1_p}, {_fmt_addr(target)}"

        return f"c.?    0x{c16:04x}"

    # --- Quadrant 2: CR / CSS ---
    elif quad == 0b10:
        rd_name_q2 = gpr_name((c16 >> 7) & 0x1F)
        rs2_name = gpr_name((c16 >> 2) & 0x1F)
        rd_q2 = (c16 >> 7) & 0x1F
        rs2_q2 = (c16 >> 2) & 0x1F

        if funct3 == 0b000:  # C.SLLI (RV64C: shamt[5] = bit12)
            shamt = ((c16 >> 2) & 0x1F) | ((c16 >> 12) & 1) << 5
            # rd=x0 is a HINT (executed as NOP), but display apparent semantics
            return f"c.slli       {rd_name_q2}, {shamt}"

        if funct3 == 0b001:  # C.FLDSP (RV64DC)
            uimm = (
                ((c16 >> 2) & 0b111) << 6 | ((c16 >> 12) & 1) << 5 | ((c16 >> 5) & 0b11) << 3
            )
            # rd=0 -> ft0, valid FP destination (unlike GPR loads where x0 is reserved)
            return f"c.fldsp      f{rd_q2}, {uimm}(sp)"

        if funct3 == 0b010:  # C.LWSP
            # offset = {inst[6:5], inst[12], inst[4:2], 00} (4 字节对齐)
            uimm = (
                ((c16 >> 5) & 0b11) << 6 | ((c16 >> 12) & 1) << 5 | ((c16 >> 2) & 0b111) << 2
            )
            if rd_q2 == 0:
                return f"c.?    0x{c16:04x}  # C.LWSP rd=0 (reserved)"
            return f"c.lwsp       {rd_name_q2}, {uimm}(sp)"

        if funct3 == 0b011:  # C.LDSP (RV64)
            # offset = {inst[4:2], inst[12], inst[6:5], 000} (8 字节对齐)
            uimm = (
                ((c16 >> 2) & 0b111) << 6 | ((c16 >> 12) & 1) << 5 | ((c16 >> 5) & 0b11) << 3
            )
            if rd_q2 == 0:
                return f"c.?    0x{c16:04x}  # C.LDSP rd=0 (reserved)"
            return f"c.ldsp       {rd_name_q2}, {uimm}(sp)"

        if funct3 == 0b100:  # C.JR / C.JALR / C.MV / C.EBREAK / C.ADD
            # C.EBREAK: rd=0, rs2=0 (both bit12=0 and bit12=1 are common)
            if rs2_q2 == 0 and rd_q2 == 0:
                return "c.ebreak"
            if b12 == 0:
                if rs2_q2 == 0 and rd_q2 != 0:
                    return f"c.jr         {rd_name_q2}"
                if rs2_q2 != 0 and rd_q2 != 0:
                    return f"c.mv         {rd_name_q2}, {rs2_name}"
                return f"c.?    0x{c16:04x}"
            # b12 == 1
            if rs2_q2 == 0 and rd_q2 != 0:
                return f"c.jalr       {rd_name_q2}"
            if rs2_q2 != 0 and rd_q2 != 0:
                return f"c.add        {rd_name_q2}, {rs2_name}"
            return f"c.?    0x{c16:04x}"

        if funct3 == 0b101:  # C.FSDSP (RV64DC)
            uimm = ((c16 >> 7) & 0b111) << 6 | ((c16 >> 10) & 0b111) << 3
            return f"c.fsdsp      f{rs2_q2}, {uimm}(sp)"

        if funct3 == 0b110:  # C.SWSP
            # offset = {inst[8:7], inst[12:9], 00}  (4-byte aligned)
            uimm = ((c16 >> 7) & 0b11) << 6 | ((c16 >> 9) & 0b1111) << 2
            return f"c.swsp       {rs2_name}, {uimm}(sp)"

        if funct3 == 0b111:  # C.SDSP (RV64)
            # offset = {inst[9:7], inst[12:10], 000}  (8-byte aligned)
            uimm = ((c16 >> 7) & 0b111) << 6 | ((c16 >> 10) & 0b111) << 3
            return f"c.sdsp       {rs2_name}, {uimm}(sp)"

    return f"c.?    0x{c16:04x}"


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


def _dis_fp_op(instr: int) -> str:
    """OP-FP: 算术/转换/比较/符号/分类/移动."""
    funct7 = parse_func7(instr)
    funct3 = parse_func3(instr)
    rs2 = parse_rs2(instr)
    fmt = funct7 & 0x3
    op5 = funct7 >> 2
    sfx = _fsuffix(fmt)
    frd, frs1, frs2 = (
        _fpr_name(parse_rd(instr)),
        _fpr_name(parse_rs1(instr)),
        _fpr_name(rs2),
    )
    if op5 in _FP_ARITH:
        return f"{_FP_ARITH[op5] + '.' + sfx:<12} {frd}, {frs1}, {frs2}"
    if op5 == 0x0B:  # FSQRT
        return f"{'fsqrt.' + sfx:<12} {frd}, {frs1}"
    if op5 == 0x04:  # FSGNJ*
        m = _FP_SGNJ.get(funct3, "fsgnj?")
        return f"{m + '.' + sfx:<12} {frd}, {frs1}, {frs2}"
    if op5 == 0x05:  # FMIN/FMAX
        m = _FP_MINMAX.get(funct3, "fmin?")
        return f"{m + '.' + sfx:<12} {frd}, {frs1}, {frs2}"
    if op5 == 0x14:  # FCMP ->GPR rd
        m = _FP_CMP.get(funct3, "fcmp?")
        return f"{m + '.' + sfx:<12} {_rd(instr)}, {frs1}, {frs2}"
    if op5 == 0x18:  # FCVT float->int (GPR rd)
        w = {0: "w", 1: "wu", 2: "l", 3: "lu"}.get(rs2, "?")
        return f"{'fcvt.' + w + '.' + sfx:<12} {_rd(instr)}, {frs1}"
    if op5 == 0x1A:  # FCVT int->float (GPR rs1)
        w = {0: "w", 1: "wu", 2: "l", 3: "lu"}.get(rs2, "?")
        return f"{'fcvt.' + sfx + '.' + w:<12} {frd}, {_rs1(instr)}"
    if op5 == 0x08:  # FCVT.S.D / FCVT.D.S
        m = "fcvt.d.s" if fmt == 1 else "fcvt.s.d"
        return f"{m:<12} {frd}, {frs1}"
    if op5 == 0x1C:  # FMV.X.* / FCLASS ->GPR rd
        if funct3 == 0:
            m = "fmv.x.w" if fmt == 0 else "fmv.x.d"
        else:
            m = "fclass.s" if fmt == 0 else "fclass.d"
        return f"{m:<12} {_rd(instr)}, {frs1}"
    if op5 == 0x1E:  # FMV.*.X (GPR rs1 ->FPR)
        m = "fmv.w.x" if fmt == 0 else "fmv.d.x"
        return f"{m:<12} {frd}, {_rs1(instr)}"
    return _UNKNOWN


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
    if opc == Opc.opfp:
        return _dis_fp_load(instr)
    if opc == Opc.stfp:
        return _dis_fp_store(instr)
    if opc == Opc.opFp:
        return _dis_fp_op(instr)
    if opc in (Opc.fmadd, Opc.fmsub, Opc.fnmsub, Opc.fnmadd):
        return _dis_fp_fma(instr)

    return _UNKNOWN
