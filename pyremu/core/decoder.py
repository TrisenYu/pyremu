#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 20:50:31
# Last modified at 2026/06/08 星期一


# https://luplab.gitlab.io/rvcodecjs/

"""
        7     5   5    3    5  7
R 类型: func7 rs2 rs1 func3 rd opcode
I 类型:   imm12   rs1 func3 rd opcode
S 类型: imm7  rs2 rs1 func3 imm5 opcode
B 类型: imm[12|10:5] rs2 rs1 func3 imm[4:1|11] opcode
U 类型:      imm20          rd opcode
J 类型: imm[20|10:1|11|19:12] rd opcode

Mem排序指令:  fm4 PSxIORW rs1 func3 rd opcode
ecall/ebreak:    func12   rs1 func3 rd opcode
"""

import ctypes
from enum import Enum

from pyremu.core.hart import (
    HartWithRegs,
)
from pyremu.core.mem_check_aux import (
    mem_read,
    mem_write,
    validate_csr,
)
from pyremu.core.registers import CsrAccessError
from pyremu.core.trap import TrapType
from pyremu.core.trap_handler import (
    deliver_trap,
    handle_wfi,
    trap_ebreak,
    trap_ecall,
    trap_mret,
    trap_sret,
)

_sint64 = ctypes.c_int64
_uint64 = ctypes.c_uint64

# ============================================================
#  Bit-manipulation helpers for sign extension
# ============================================================


def _sext(val: int, bits: int) -> int:
    """Sign-extend *val* from *bits* width to a canonical 64-bit unsigned Python int.

    Python's arbitrary-precision integers behave differently from finite-width
    hardware in bitwise operations (|, &, ^, <<, >>) when values are negative.
    Canonicalizing to [0, 2^64) ensures consistent behaviour regardless of
    whether the value was built via sign-extend, zero-extend, or arithmetic.
    """
    sign_bit = 1 << (bits - 1)
    result = (val & (sign_bit - 1)) - (val & sign_bit)
    # Normalize to 64-bit unsigned representation for consistent bitwise semantics.
    if bits <= 64:
        result &= (1 << 64) - 1
    return result


# ============================================================
#  Opcodes
# ============================================================


class Opc(Enum):
    """
    如果是压缩指令，低两位不是11,而可能是00,01,10
    然后长度按照16位来解析
    """

    ld = 0b00000_11  # 载入
    opfp = 0b00001_11  # 浮点
    fence = 0b00011_11  # 内存/执行流屏障
    opImm = 0b00100_11  # 立即数 ALU (32-bit)
    opImm32 = 0b00110_11  # RV64 32-bit 立即数 ALU
    auipc = 0b00101_11  # 累加 pc
    st = 0b01000_11  # 写入
    amo = 0b01011_11  # 原子
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


# ============================================================
#  Instruction-field extractors
# ============================================================

parse_opcode = lambda x: x & 0b111_1111
parse_rd = lambda x: (x >> 7) & 0b1_1111
parse_func3 = lambda x: (x >> 12) & 0b0111
parse_rs1 = lambda x: (x >> 15) & 0b1_1111
parse_rs2 = lambda x: (x >> 20) & 0b1_1111
parse_func7 = lambda x: (x >> 25) & 0b111_1111
parse_func6 = lambda x: (x >> 26) & 0x3F  # for SLLI/SRLI/SRAI (I-type shifts)

# 立即数解析（未符号拓展）
parse_imm12_raw = lambda x: (x >> 20) & 0xFFF  # I-type
parse_imm12_se = lambda x: _sext((x >> 20) & 0xFFF, 12)
parse_imm20_raw = lambda x: (x >> 12) & 0xF_FFFF  # U-type


def parse_imm_s(instr: int) -> int:
    """S-type 12-bit immediate, sign-extended."""
    imm = ((instr >> 25) & 0x7F) << 5  # imm[11:5]
    imm |= (instr >> 7) & 0x1F          # imm[4:0]  -- from rd field
    return _sext(imm, 12)


def parse_imm_b(instr: int) -> int:
    """B-type 13-bit immediate, sign-extended (byte-address diff)."""
    imm = ((instr >> 31) & 1) << 12  # imm[12]
    imm |= ((instr >> 25) & 0x3F) << 5  # imm[10:5]
    imm |= ((instr >> 8) & 0xF) << 1  # imm[4:1]
    imm |= ((instr >> 7) & 1) << 11  # imm[11]
    return _sext(imm, 13)


def parse_imm_j(instr: int) -> int:
    """J-type 21-bit immediate, sign-extended (byte-address diff)."""
    imm = ((instr >> 31) & 1) << 20  # imm[20]
    imm |= ((instr >> 21) & 0x3FF) << 1  # imm[10:1]
    imm |= ((instr >> 20) & 1) << 11  # imm[11]
    imm |= ((instr >> 12) & 0xFF) << 12  # imm[19:12]
    return _sext(imm, 21)


parse_compressed = lambda x: (x & 0x3) != 3

# 可能会有指令别名，不过那是反编译器关心的事情

# https://lhtin.github.io/01world/app/riscv-isa/?xlen=32
# https://msyksphinz-self.github.io/riscv-isadoc/


# ============================================================
#  Hart (inherits register file & CSRs from HartWithRegs)
# ============================================================

# HartWithRegs 和 RiscvMode 已在文件头部导入


"""
mul      有符号 x 有符号    取 低 64 位
mulh     有符号 x 有符号    取 高 64 位
mulhsu   有符号 x 无符号    取 高 64 位
mulhu    无符号 x 无符号    取 高 64 位
"""


def _trunc_div(a: int, b: int) -> int:
    """Integer division truncating toward zero (C99 / RISC-V semantics)."""
    if b == 0:
        return -1  # RISC-V: all-bits-1 on division by zero
    q = abs(a) // abs(b)
    return -q if (a < 0) ^ (b < 0) else q


def _trunc_rem(a: int, b: int) -> int:
    """Integer remainder with same sign as dividend (RISC-V semantics)."""
    if b == 0:
        return a  # RISC-V: dividend on division by zero
    return a - _trunc_div(a, b) * b


class Hart(HartWithRegs):
    # ----------------------------------------------------------
    #  R-type ALU (opcode = Opc.op)
    # ----------------------------------------------------------
    def handle_alu(self, instr: int):
        """Execute an R-type ALU instruction on this hart."""
        part1 = parse_func3(instr)
        part2 = parse_func7(instr)
        rd = parse_rd(instr)
        rs1 = parse_rs1(instr)
        rs2 = parse_rs2(instr)

        v1 = self.gprs[rs1].val
        v2 = self.gprs[rs2].val

        if part1 == 0b000:  # ADD / MUL / SUB
            if part2 == 0:
                result = (v1 + v2) & 0xFFFF_FFFF_FFFF_FFFF
            elif part2 == 1:
                result = (v1 * v2) & 0xFFFF_FFFF_FFFF_FFFF
            elif part2 == 0x20:
                result = (v1 - v2) & 0xFFFF_FFFF_FFFF_FFFF
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=000")

        elif part1 == 0b001:  # SLL / MULH
            if part2 == 0:
                result = (v1 << (v2 & 0x3F)) & 0xFFFF_FFFF_FFFF_FFFF
            elif part2 == 1:
                s1 = _sint64(v1).value
                s2 = _sint64(v2).value
                result = _sint64((s1 * s2) >> 64).value & 0xFFFF_FFFF_FFFF_FFFF
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=001")

        elif part1 == 0b010:  # SLT / MULHSU
            if part2 == 0:
                result = 1 if _sint64(v1).value < _sint64(v2).value else 0
            elif part2 == 1:
                s1 = _sint64(v1).value
                u2 = _uint64(v2).value
                result = _sint64((s1 * u2) >> 64).value & 0xFFFF_FFFF_FFFF_FFFF
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=010")

        elif part1 == 0b011:  # SLTU / MULHU
            if part2 == 0:
                result = 1 if _uint64(v1).value < _uint64(v2).value else 0
            elif part2 == 1:
                u1 = _uint64(v1).value
                u2 = _uint64(v2).value
                result = ((u1 * u2) >> 64) & 0xFFFF_FFFF_FFFF_FFFF
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=011")

        elif part1 == 0b100:  # XOR / DIV
            if part2 == 0:
                result = v1 ^ v2
            elif part2 == 1:
                result = (
                    _trunc_div(
                        _sint64(v1).value,
                        _sint64(v2).value,
                    )
                    & 0xFFFF_FFFF_FFFF_FFFF
                )
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=100")

        elif part1 == 0b101:  # SRL / DIVU / SRA
            if part2 == 0:
                result = (_uint64(v1).value >> (v2 & 0x3F)) & 0xFFFF_FFFF_FFFF_FFFF
            elif part2 == 1:
                result = (
                    _trunc_div(
                        _uint64(v1).value,
                        _uint64(v2).value,
                    )
                    & 0xFFFF_FFFF_FFFF_FFFF
                )
            elif part2 == 0x20:
                result = (
                    _sint64(_sint64(v1).value >> (v2 & 0x3F)).value & 0xFFFF_FFFF_FFFF_FFFF
                )
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=101")

        elif part1 == 0b110:  # OR / REM
            if part2 == 0:
                result = v1 | v2
            elif part2 == 1:
                result = (
                    _trunc_rem(
                        _sint64(v1).value,
                        _sint64(v2).value,
                    )
                    & 0xFFFF_FFFF_FFFF_FFFF
                )
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=110")

        elif part1 == 0b111:  # AND / REMU
            if part2 == 0:
                result = v1 & v2
            elif part2 == 1:
                result = (
                    _trunc_rem(
                        _uint64(v1).value,
                        _uint64(v2).value,
                    )
                    & 0xFFFF_FFFF_FFFF_FFFF
                )
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=111")

        else:
            raise ValueError(f"invalid funct3={part1:#x}")

        if rd != 0:
            self.gprs[rd].val = result

    # ----------------------------------------------------------
    #  I-type ALU (opcode = Opc.opImm)
    # ----------------------------------------------------------
    def handle_op_imm(self, instr: int):
        """Execute an I-type immediate ALU instruction."""
        part1 = parse_func3(instr)
        part6 = parse_func6(instr)  # funct6 for SLLI/SRLI/SRAI
        rd = parse_rd(instr)
        rs1 = parse_rs1(instr)
        imm = parse_imm12_se(instr)  # 12-bit signed immediate
        shamt = (instr >> 20) & 0x3F  # 6-bit shift amount (RV64)

        v1 = self.gprs[rs1].val

        if part1 == 0b000:  # ADDI
            result = (v1 + imm) & 0xFFFF_FFFF_FFFF_FFFF

        elif part1 == 0b001:  # SLLI
            if part6 != 0:
                raise ValueError(f"invalid funct6={part6:#x} for SLLI")
            result = (v1 << shamt) & 0xFFFF_FFFF_FFFF_FFFF

        elif part1 == 0b010:  # SLTI
            result = 1 if _sint64(v1).value < imm else 0

        elif part1 == 0b011:  # SLTIU
            result = 1 if _uint64(v1).value < _uint64(imm).value else 0

        elif part1 == 0b100:  # XORI
            result = v1 ^ imm

        elif part1 == 0b101:  # SRLI / SRAI
            if part6 == 0:         # SRLI (funct6=0b000000)
                result = (_uint64(v1).value >> shamt) & 0xFFFF_FFFF_FFFF_FFFF
            elif part6 == 0x10:    # SRAI (funct6=0b010000)
                result = _sint64(_sint64(v1).value >> shamt).value & 0xFFFF_FFFF_FFFF_FFFF
            else:
                raise ValueError(f"invalid funct6={part6:#x} for SRLI/SRAI")

        elif part1 == 0b110:  # ORI
            result = v1 | imm

        elif part1 == 0b111:  # ANDI
            result = v1 & imm

        else:
            raise ValueError(f"invalid funct3={part1:#x} for opImm")

        if rd != 0:
            self.gprs[rd].val = result

    # ----------------------------------------------------------
    #  RV64 32-bit word operations (opcode = Opc.op32)
    # ----------------------------------------------------------
    def handle_op32(self, instr: int):
        """Execute an RV64 32-bit word operation (e.g. ADDW, SUBW, SLLW, etc.)."""
        part1 = parse_func3(instr)
        part2 = parse_func7(instr)
        rd = parse_rd(instr)
        rs1 = parse_rs1(instr)
        rs2 = parse_rs2(instr)

        # Operate on lower 32 bits
        v1 = self.gprs[rs1].val & 0xFFFF_FFFF
        v2 = self.gprs[rs2].val & 0xFFFF_FFFF

        if part1 == 0b000:  # ADDW / SUBW / MULW
            if part2 == 0:
                result = (v1 + v2) & 0xFFFF_FFFF
            elif part2 == 1:
                result = (v1 * v2) & 0xFFFF_FFFF
            elif part2 == 0x20:
                result = (v1 - v2) & 0xFFFF_FFFF
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=000")
            # Sign-extend 32-bit result to 64 bits
            result = _sext(result, 32)

        elif part1 == 0b001:  # SLLW / MULHW
            if part2 == 0:
                result = (v1 << (v2 & 0x1F)) & 0xFFFF_FFFF
                result = _sext(result, 32)
            elif part2 == 1:
                s1 = _sint64(_sext(v1, 32)).value
                s2 = _sint64(_sext(v2, 32)).value
                result = _sint64((s1 * s2) >> 32).value & 0xFFFF_FFFF_FFFF_FFFF
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=001")

        elif part1 == 0b010:  # SLTW / MULHSUW
            if part2 == 0:
                result = (
                    1 if _sint64(_sext(v1, 32)).value < _sint64(_sext(v2, 32)).value else 0
                )
            elif part2 == 1:
                s1 = _sint64(_sext(v1, 32)).value
                u2 = _uint64(v2).value  # zero-extended 32-bit
                result = _sint64((s1 * u2) >> 32).value & 0xFFFF_FFFF_FFFF_FFFF
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=010")

        elif part1 == 0b011:  # SLTUW / MULHUW
            if part2 == 0:
                result = 1 if v1 < v2 else 0  # both zero-extended already
            elif part2 == 1:
                result = ((v1 * v2) >> 32) & 0xFFFF_FFFF_FFFF_FFFF
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=011")

        elif part1 == 0b100:  # XORW / DIVW
            if part2 == 0:
                result = (v1 ^ v2) & 0xFFFF_FFFF
                result = _sext(result, 32)
            elif part2 == 1:
                result = _trunc_div(
                    _sint64(_sext(v1, 32)).value,
                    _sint64(_sext(v2, 32)).value,
                )
                result = _sext(result & 0xFFFF_FFFF, 32)
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=100")

        elif part1 == 0b101:  # SRLW / DIVUW / SRAW
            if part2 == 0:
                result = (v1 >> (v2 & 0x1F)) & 0xFFFF_FFFF
                result = _sext(result, 32)
            elif part2 == 1:
                result = _trunc_div(v1, v2)  # unsigned 32-bit
                result = _sext(result & 0xFFFF_FFFF, 32)
            elif part2 == 0x20:
                result = _sint64(_sext(v1, 32) >> (v2 & 0x1F)).value
                result = _sext(result & 0xFFFF_FFFF, 32)
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=101")

        elif part1 == 0b110:  # ORW / REMW
            if part2 == 0:
                result = (v1 | v2) & 0xFFFF_FFFF
                result = _sext(result, 32)
            elif part2 == 1:
                result = _trunc_rem(
                    _sint64(_sext(v1, 32)).value,
                    _sint64(_sext(v2, 32)).value,
                )
                result = _sext(result & 0xFFFF_FFFF, 32)
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=110")

        elif part1 == 0b111:  # ANDW / REMUW
            if part2 == 0:
                result = (v1 & v2) & 0xFFFF_FFFF
                result = _sext(result, 32)
            elif part2 == 1:
                result = _trunc_rem(v1, v2)  # unsigned 32-bit
                result = _sext(result & 0xFFFF_FFFF, 32)
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=111")

        else:
            raise ValueError(f"invalid funct3={part1:#x} for op32")

        if rd != 0:
            self.gprs[rd].val = result

    # ----------------------------------------------------------
    #  I-type immediate word operations (opcode = Opc.opImm32)
    # ----------------------------------------------------------
    def handle_op_imm32(self, instr: int):
        """Execute an I-type 32-bit word immediate ALU instruction (ADDIW, SLLIW, etc.)."""
        part1 = parse_func3(instr)
        part7 = parse_func7(instr)
        rd = parse_rd(instr)
        rs1 = parse_rs1(instr)
        imm = parse_imm12_se(instr)
        shamt = (instr >> 20) & 0x1F  # 5-bit shift amount (RV64 word)

        v1 = self.gprs[rs1].val

        if part1 == 0b000:  # ADDIW
            result = (v1 + imm) & 0xFFFF_FFFF
            result = _sext(result, 32)

        elif part1 == 0b001:  # SLLIW
            if part7 != 0:
                raise ValueError(f"invalid funct7={part7:#x} for SLLIW")
            result = ((v1 & 0xFFFF_FFFF) << shamt) & 0xFFFF_FFFF
            result = _sext(result, 32)

        elif part1 == 0b101:  # SRLIW / SRAIW
            if part7 == 0:
                result = ((v1 & 0xFFFF_FFFF) >> shamt) & 0xFFFF_FFFF
            elif part7 == 0x20:
                result = _sint64(_sext(v1 & 0xFFFF_FFFF, 32) >> shamt).value & 0xFFFF_FFFF
            else:
                raise ValueError(f"invalid funct7={part7:#x} for SRLIW/SRAIW")
            result = _sext(result, 32)

        else:
            raise ValueError(f"invalid funct3={part1:#x} for opImm32")

        if rd != 0:
            self.gprs[rd].val = result

    # ----------------------------------------------------------
    #  Branch (opcode = Opc.br)
    # ----------------------------------------------------------
    def handle_br(self, instr: int) -> bool:
        """Execute a conditional branch.  Returns True when the branch
        was taken (pc has been updated), False otherwise."""
        fn3 = parse_func3(instr)
        rs1 = parse_rs1(instr)
        rs2 = parse_rs2(instr)
        offset = parse_imm_b(instr)

        v1 = self.gprs[rs1].val
        v2 = self.gprs[rs2].val

        taken = False
        try:
            f = brFn3(fn3)
        except ValueError:
            return False

        if f == brFn3.beq:
            taken = v1 == v2
        elif f == brFn3.bne:
            taken = v1 != v2
        elif f == brFn3.blt:
            taken = _sint64(v1).value < _sint64(v2).value
        elif f == brFn3.bge:
            taken = _sint64(v1).value >= _sint64(v2).value
        elif f == brFn3.bltu:
            taken = _uint64(v1).value < _uint64(v2).value
        elif f == brFn3.bgeu:
            taken = _uint64(v1).value >= _uint64(v2).value

        if taken:
            self.pc = (self.pc + offset) & 0xFFFF_FFFF_FFFF_FFFF
        return taken

    # ----------------------------------------------------------
    #  Load (opcode = Opc.ld)
    # ----------------------------------------------------------
    def handle_ld(self, instr: int):
        """Execute a load instruction."""
        fn3 = parse_func3(instr)
        rd = parse_rd(instr)
        rs1 = parse_rs1(instr)
        offset = parse_imm12_se(instr)

        addr = (self.gprs[rs1].val + offset) & 0xFFFF_FFFF_FFFF_FFFF

        try:
            f = ldFn3(fn3)
        except ValueError:
            raise ValueError(f"invalid funct3={fn3:#x} for load")

        # 按实际操作宽度读取, 避免对 lb/lbu/lh/lhu/lw/lwu 产生虚假对齐故障
        if f in (ldFn3.lb, ldFn3.lbu):
            read_size = 1
        elif f in (ldFn3.lh, ldFn3.lhu):
            read_size = 2
        elif f in (ldFn3.lw, ldFn3.lwu):
            read_size = 4
        else:
            read_size = 8  # ld

        mem = mem_read(self,addr, read_size)

        if f == ldFn3.lb:
            val = _sext(mem[0], 8)
        elif f == ldFn3.lh:
            val = _sext(mem[0] | (mem[1] << 8), 16)
        elif f == ldFn3.lw:
            val = _sext(mem[0] | (mem[1] << 8) | (mem[2] << 16) | (mem[3] << 24), 32)
        elif f == ldFn3.ld:
            val = 0
            for i in range(8):
                val |= mem[i] << (8 * i)
        elif f == ldFn3.lbu:
            val = mem[0]
        elif f == ldFn3.lhu:
            val = mem[0] | (mem[1] << 8)
        elif f == ldFn3.lwu:
            val = mem[0] | (mem[1] << 8) | (mem[2] << 16) | (mem[3] << 24)
        else:
            raise ValueError(f"unhandled load funct3={fn3:#x}")

        if rd != 0:
            self.gprs[rd].val = val

    # ----------------------------------------------------------
    #  Store (opcode = Opc.st)
    # ----------------------------------------------------------
    def handle_st(self, instr: int):
        """Execute a store instruction.
        TODO: integrate MMU / TLB for address translation."""
        fn3 = parse_func3(instr)
        rs1 = parse_rs1(instr)
        rs2 = parse_rs2(instr)
        offset = parse_imm_s(instr)

        addr = (self.gprs[rs1].val + offset) & 0xFFFF_FFFF_FFFF_FFFF
        val = self.gprs[rs2].val

        try:
            f = stFn3(fn3)
        except ValueError:
            raise ValueError(f"invalid funct3={fn3:#x} for store")

        if f == stFn3.sb:
            data = bytes([val & 0xFF])
        elif f == stFn3.sh:
            data = bytes([val & 0xFF, (val >> 8) & 0xFF])
        elif f == stFn3.sw:
            data = bytes(
                [val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF]
            )
        elif f == stFn3.sd:
            data = bytes([(val >> (8 * i)) & 0xFF for i in range(8)])
        else:
            raise ValueError(f"unhandled store funct3={fn3:#x}")

        mem_write(self,addr, data)

    # ----------------------------------------------------------
    #  Atomic Memory Operations (opcode = Opc.amo)
    # ----------------------------------------------------------
    # AMO 编码: funct5[31:27] | aq[26] | rl[25] | rs2[24:20] |
    #           rs1[19:15] | funct3[14:12] | rd[11:7] | opcode=0101111
    # aq/rl 为内存排序提示, 顺序执行模型中可忽略.

    def handle_amo(
        self,
        instr: int,
    ) -> None:
        """执行原子内存操作 (LR/SC/AMOxxx)."""
        funct5_val = (instr >> 27) & 0x1F
        funct3_val = parse_func3(instr)
        rd = parse_rd(instr)
        rs1 = parse_rs1(instr)
        rs2 = parse_rs2(instr)

        try:
            op = AmoFunct5(funct5_val)
            width = AmoWidth(funct3_val)
        except ValueError:
            raise ValueError(
                f"invalid AMO encoding: funct5={funct5_val:#07b}, funct3={funct3_val:#05b}"
            )

        is_64bit = width == AmoWidth.D
        byte_len = 8 if is_64bit else 4
        mask = 0xFFFF_FFFF_FFFF_FFFF if is_64bit else 0xFFFF_FFFF

        addr = self.gprs[rs1].val & 0xFFFF_FFFF_FFFF_FFFF

        if op == AmoFunct5.LR:
            # Load-Reserved: 读取内存并设置预留
            data_bytes = mem_read(self,addr, byte_len)
            val = int.from_bytes(data_bytes, "little", signed=False) & mask
            if rd != 0:
                # LR.D: 64-bit 值不需要符号扩展; LR.W: 32→64 符号扩展
                self.gprs[rd].val = val if is_64bit else _sext(val, 32)
            self.set_reservation(addr)

        elif op == AmoFunct5.SC:
            # Store-Conditional: 仅预留有效时写入
            if self.reservation_valid and self.reservation_addr == addr:
                store_val = self.gprs[rs2].val & mask
                data = store_val.to_bytes(byte_len, "little", signed=False)
                mem_write(self,addr, data)
                if rd != 0:
                    self.gprs[rd].val = 0  # 成功 → rd ← 0
            elif rd != 0:
                self.gprs[rd].val = 1  # 失败 → rd ← 非零
            self.clear_reservation()

        else:
            # AMOxxx: 原子读-改-写
            data_bytes = mem_read(self,addr, byte_len)
            mem_val = int.from_bytes(data_bytes, "little", signed=False) & mask
            op_val = self.gprs[rs2].val & mask

            if op == AmoFunct5.SWAP:
                result = op_val
            elif op == AmoFunct5.ADD:
                result = (mem_val + op_val) & mask
            elif op == AmoFunct5.XOR:
                result = mem_val ^ op_val
            elif op == AmoFunct5.AND:
                result = mem_val & op_val
            elif op == AmoFunct5.OR:
                result = mem_val | op_val
            elif op == AmoFunct5.MIN:
                s_mem = _sint64(_sext(mem_val, 64 if is_64bit else 32)).value
                s_op = _sint64(_sext(op_val, 64 if is_64bit else 32)).value
                result = (op_val if s_op < s_mem else mem_val) & mask
            elif op == AmoFunct5.MAX:
                s_mem = _sint64(_sext(mem_val, 64 if is_64bit else 32)).value
                s_op = _sint64(_sext(op_val, 64 if is_64bit else 32)).value
                result = (op_val if s_op > s_mem else mem_val) & mask
            elif op == AmoFunct5.MINU:
                result = (op_val if op_val < mem_val else mem_val) & mask
            elif op == AmoFunct5.MAXU:
                result = (op_val if op_val > mem_val else mem_val) & mask
            else:
                raise ValueError(f"unhandled AMO op: {op}")

            data = result.to_bytes(byte_len, "little", signed=False)
            mem_write(self,addr, data)
            if rd != 0:
                self.gprs[rd].val = (
                    _sext(mem_val, 64) if is_64bit else _sext(mem_val, 32)
                )

    # ----------------------------------------------------------
    #  JALR (opcode = Opc.jalr)
    # ----------------------------------------------------------
    def handle_jalr(self, instr: int):
        """Execute JALR: rd = pc+4; pc = (rs1 + imm) & ~1"""
        rd = parse_rd(instr)
        rs1 = parse_rs1(instr)
        imm = parse_imm12_se(instr)
        next_pc = (self.pc + 4) & 0xFFFF_FFFF_FFFF_FFFF
        target = (self.gprs[rs1].val + imm) & 0xFFFF_FFFF_FFFF_FFFF
        target &= ~1  # clear LSB to align

        if rd != 0:
            self.gprs[rd].val = next_pc
        self.pc = target

    # ----------------------------------------------------------
    #  System (opcode = Opc.sys)  -- ecall / ebreak / CSR / mret / ...
    # ----------------------------------------------------------
    def handle_sys(self, instr: int):
        """Execute a system instruction (privileged or CSR access)."""
        fn3 = parse_func3(instr)
        rd = parse_rd(instr)
        rs1 = parse_rs1(instr)
        csr_addr = (instr >> 20) & 0xFFF  # CSR address [31:20]
        uimm = rs1  # 5-bit zero-extended imm for CSR

        if fn3 == 0b000:
            # Privileged instructions: funct12 determines the operation
            funct12 = (instr >> 20) & 0xFFF
            if funct12 == 0x000:  # ECALL
                trap_ecall(self)
            elif funct12 == 0x001:  # EBREAK
                trap_ebreak(self)
            elif funct12 == 0x302:  # MRET
                trap_mret(self)
            elif funct12 == 0x102:  # SRET
                trap_sret(self)
            elif funct12 == 0x105:  # WFI
                handle_wfi(self, instr)
            elif funct12 == 0x120:  # SFENCE.VMA (funct7=0b0001001, rs2=0)
                # 刷新所有 hart 的 TLB (当前为单 hart, 故只刷新自己)
                self.itlb.flush_all()
                self.dtlb.flush_all()
            elif funct12 == 0x5A0:  # MFENCE.DID — 按内存域刷新全部 hart TLB + L2
                # 读取当前 hart 的 mdid, 广播刷新所有 hart 中匹配的条目
                mdid_val = self.csrs["mdid"].val
                for h in (self._all_harts or [self]):
                    h.itlb.flush_by_mdid(mdid_val)
                    h.dtlb.flush_by_mdid(mdid_val)
                # 刷新共享 L2 缓存中匹配的条目
                if self._bus is not None:
                    l2 = getattr(self._bus, "_l2", None)
                    if l2 is not None:
                        l2.flush_by_mdid(mdid_val)
            else:
                raise ValueError(f"unknown privileged funct12={funct12:#05x}")

        elif fn3 == 0b001:  # CSRRW
            validate_csr(self,csr_addr, is_write=True)
            old_csr = self.read_csr(csr_addr)
            new_csr = self.gprs[rs1].val     # 先读 rs1 (可能在 rd==rs1 时被覆盖)
            if rd != 0:
                self.gprs[rd].val = old_csr
            self.write_csr(csr_addr, new_csr)

        elif fn3 == 0b010:  # CSRRS
            do_write = rs1 != 0
            validate_csr(self,csr_addr, is_write=do_write)
            old_csr = self.read_csr(csr_addr)
            rs1_val = self.gprs[rs1].val  # 先读 rs1 (可能在 rd==rs1 时被覆盖)
            if rd != 0:
                self.gprs[rd].val = old_csr
            if do_write:
                self.write_csr(csr_addr, old_csr | rs1_val)

        elif fn3 == 0b011:  # CSRRC
            do_write = rs1 != 0
            validate_csr(self,csr_addr, is_write=do_write)
            old_csr = self.read_csr(csr_addr)
            rs1_val = self.gprs[rs1].val  # 先读 rs1 (可能在 rd==rs1 时被覆盖)
            if rd != 0:
                self.gprs[rd].val = old_csr
            if do_write:
                self.write_csr(csr_addr, old_csr & ~rs1_val)

        elif fn3 == 0b101:  # CSRRWI
            validate_csr(self,csr_addr, is_write=True)
            old = self.read_csr(csr_addr)
            if rd != 0:
                self.gprs[rd].val = old
            self.write_csr(csr_addr, uimm)

        elif fn3 == 0b110:  # CSRRSI
            do_write = uimm != 0
            validate_csr(self,csr_addr, is_write=do_write)
            old = self.read_csr(csr_addr)
            if rd != 0:
                self.gprs[rd].val = old
            if do_write:
                self.write_csr(csr_addr, old | uimm)

        elif fn3 == 0b111:  # CSRRCI
            do_write = uimm != 0
            validate_csr(self,csr_addr, is_write=do_write)
            old = self.read_csr(csr_addr)
            if rd != 0:
                self.gprs[rd].val = old
            if do_write:
                self.write_csr(csr_addr, old & ~uimm)

        else:
            raise ValueError(f"invalid funct3={fn3:#x} for sys")

    # ----------------------------------------------------------
    #  FENCE (opcode = Opc.fence)
    # ----------------------------------------------------------
    def handle_fence(self, instr: int):
        """Execute a FENCE / FENCE.I instruction.
        In a single-hart, in-order emulator these are mostly no-ops."""
        fn3 = parse_func3(instr)
        if fn3 == 0b000:  # FENCE
            pass  # no-op: sequential execution is already ordered
        elif fn3 == 0b001:  # FENCE.I
            pass  # TODO: flush instruction cache / pipeline
        else:
            raise ValueError(f"invalid funct3={fn3:#x} for fence")
    @staticmethod
    def _creg(n: int) -> int:
        """3-bit 压缩寄存器号 → 完整寄存器号 (x8–x15)."""
        return (n & 0x7) + 8

    # -- C0: Quadrant 0 (低 2 位 = 00) --
    # C.ADDI4SPN / C.LW / C.LD / C.SW / C.SD (FLD/FSD 暂未实现)

    def _handle_compressed_c0(
        self,
        instr: int,
    ) -> int:
        funct3 = (instr >> 13) & 0x7
        rd = self._creg((instr >> 2) & 0x7)

        if funct3 == 0b000:
            # C.ADDI4SPN: nzuimm[5:4|9:6|2|3] — 各 bit 分散编码
            nzuimm = (
                ((instr >> 6) & 0x1) << 2    # nzuimm[2] ← bit[6]
                | ((instr >> 5) & 0x1) << 3  # nzuimm[3] ← bit[5]
                | ((instr >> 11) & 0x1) << 4 # nzuimm[4] ← bit[11]
                | ((instr >> 12) & 0x1) << 5 # nzuimm[5] ← bit[12]
                | ((instr >> 7) & 0xF) << 6  # nzuimm[9:6] ← bits[10:7]
            )
            if nzuimm == 0:
                deliver_trap(self, TrapType.IllInstr, tval=instr, is_interrupt=False)
                return 0
            self.gprs[rd].val = (self.gprs[2].val + nzuimm) & 0xFFFF_FFFF_FFFF_FFFF
            return 2

        rs1 = self._creg((instr >> 7) & 0x7)

        # C.LW / C.SW: uimm = {instr[6], instr[12:10], instr[5]} (4-byte aligned)
        if funct3 in (0b010, 0b110):
            uimm = (
                ((instr >> 5) & 0x1) << 2
                | ((instr >> 10) & 0x7) << 3
                | ((instr >> 6) & 0x1) << 6
            )
            addr = (self.gprs[rs1].val + uimm) & 0xFFFF_FFFF_FFFF_FFFF
            if funct3 == 0b010:  # C.LW
                mem = mem_read(self, addr, 4)
                val = mem[0] | (mem[1] << 8) | (mem[2] << 16) | (mem[3] << 24)
                self.gprs[rd].val = _sext(val, 32)
            else:  # C.SW
                rs2 = self._creg((instr >> 2) & 0x7)
                v = self.gprs[rs2].val & 0xFFFF_FFFF
                data = bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF])
                mem_write(self, addr, data)

        # C.LD / C.SD (RV64C): uimm = {instr[6:5], instr[12:10]} (8-byte aligned)
        elif funct3 in (0b011, 0b111):
            uimm = (
                ((instr >> 5) & 0b11) << 6
                | ((instr >> 10) & 0b111) << 3
            )
            addr = (self.gprs[rs1].val + uimm) & 0xFFFF_FFFF_FFFF_FFFF
            if funct3 == 0b011:  # C.LD
                mem = mem_read(self, addr, 8)
                val = sum(mem[i] << (8 * i) for i in range(8))
                self.gprs[rd].val = val
            else:  # C.SD
                rs2 = self._creg((instr >> 2) & 0x7)
                v = self.gprs[rs2].val
                data = bytes([(v >> (8 * i)) & 0xFF for i in range(8)])
                mem_write(self, addr, data)

        else:
            raise NotImplementedError(f"C0 funct3={funct3:#05b} (FLD/FSD/reserved)")
        return 2

    # -- C1: Quadrant 1 (低 2 位 = 01) --
    # C.ADDI / C.ADDIW / C.LI / C.LUI / C.ADDI16SP /
    # C.SRLI/C.SRAI/C.ANDI / C.SUB/C.XOR/C.OR/C.AND /
    # C.J / C.BEQZ / C.BNEZ

    def _handle_compressed_c1(
        self,
        instr: int,
    ) -> int:
        funct3 = (instr >> 13) & 0x7
        rd_raw = (instr >> 7) & 0x1F  # C1: rd/rs1 在 bits [11:7]

        # C.ADDI / C.ADDIW / C.LI — 共用 6-bit imm (rd≠0)
        if funct3 in (0b000, 0b001, 0b010):
            imm = _sext(((instr >> 2) & 0x1F) | ((instr >> 7) & 0x20), 6)
            if funct3 == 0b000:
                self.gprs[rd_raw].val = (self.gprs[rd_raw].val + imm) & 0xFFFF_FFFF_FFFF_FFFF
            elif funct3 == 0b001:
                r = (self.gprs[rd_raw].val + imm) & 0xFFFF_FFFF
                self.gprs[rd_raw].val = _sext(r, 32)
            else:
                self.gprs[rd_raw].val = imm & 0xFFFF_FFFF_FFFF_FFFF
            return 2

        # C.LUI (rd≠{0,2}) / C.ADDI16SP (rd=2) — nzuimm 非零
        if funct3 == 0b011:
            nz = _sext(
                ((instr >> 2) & 0x1) << 4
                | ((instr >> 3) & 0x3) << 6
                | ((instr >> 5) & 0x1) << 5
                | ((instr >> 6) & 0x1) << 7
                | ((instr >> 7) & 0x3) << 8
                | ((instr >> 12) & 0x1) << 9,
                10,
            )
            if nz == 0:
                raise ValueError("C.LUI/C.ADDI16SP: nzuimm must be non-zero")
            if rd_raw == 2:
                self.gprs[2].val = (self.gprs[2].val + nz) & 0xFFFF_FFFF_FFFF_FFFF
            else:
                self.gprs[rd_raw].val = (nz << 12) & 0xFFFF_FFFF_FFFF_FFFF
            return 2

        # C1 ALU ops
        if funct3 == 0b100:
            return self._handle_compressed_c1_alu(instr)

        # C.J
        if funct3 == 0b101:
            offset = _sext(
                ((instr >> 2) & 0x1) << 5
                | ((instr >> 3) & 0x7) << 1
                | ((instr >> 6) & 0x1) << 7
                | ((instr >> 7) & 0x1) << 6
                | ((instr >> 8) & 0x1) << 10
                | ((instr >> 9) & 0x3) << 8
                | ((instr >> 11) & 0x1) << 4
                | ((instr >> 12) & 0x1) << 11,
                12,
            )
            self.pc = (self.pc + offset) & 0xFFFF_FFFF_FFFF_FFFF
            return 0

        # C.BEQZ / C.BNEZ
        if funct3 in (0b110, 0b111):
            rs1 = self._creg((instr >> 7) & 0x7)
            offset = _sext(
                ((instr >> 2) & 0x1) << 5
                | ((instr >> 3) & 0x3) << 1
                | ((instr >> 5) & 0x3) << 6
                | ((instr >> 10) & 0x3) << 3
                | ((instr >> 12) & 0x1) << 8,
                9,
            )
            taken = self.gprs[rs1].val == 0
            if funct3 == 0b111:
                taken = not taken
            if taken:
                self.pc = (self.pc + offset) & 0xFFFF_FFFF_FFFF_FFFF
                return 0
            return 2

        raise NotImplementedError(f"C1 funct3={funct3:#05b}")

    def _handle_compressed_c1_alu(
        self,
        instr: int,
    ) -> int:
        """C1 ALU: C.SRLI/C.SRAI (sf=0), C.ANDI (sf=2), C.SUB/XOR/OR/AND (sf=3)."""
        sf = (instr >> 10) & 0x3
        rd_rs1 = self._creg((instr >> 7) & 0x7)
        # RV64: shamt[5] 在 bit 12; bits[6:2] = shamt[4:0]
        shamt = ((instr >> 2) & 0x1F) | (((instr >> 12) & 0x1) << 5)
        v1 = self.gprs[rd_rs1].val

        if sf == 0b00:
            # C.SRLI (bit12=0) / C.SRAI (bit12=1)
            if (instr >> 12) & 0x1:
                v1 = _sint64(_sint64(v1).value >> shamt).value & 0xFFFF_FFFF_FFFF_FFFF
            else:
                v1 = (_uint64(v1).value >> shamt) & 0xFFFF_FFFF_FFFF_FFFF
            self.gprs[rd_rs1].val = v1
        elif sf == 0b10:
            # C.ANDI — imm[5:0] = {bit12, bits[6:2]}
            imm = _sext(((instr >> 2) & 0x1F) | (((instr >> 12) & 0x1) << 5), 6)
            self.gprs[rd_rs1].val = (v1 & imm) & 0xFFFF_FFFF_FFFF_FFFF
        elif sf in (0b01, 0b11):
            # C.SUB / C.XOR / C.OR / C.AND  (sf=0b11 RV32 regs, sf=0b01 RV64 regs)
            #   bit_6_5: 00=SUB, 01=XOR, 10=OR, 11=AND
            rs2 = self._creg((instr >> 2) & 0x7)
            v2 = self.gprs[rs2].val
            bit_6_5 = (instr >> 5) & 0x3
            if bit_6_5 == 0b00:
                r = (v1 - v2) & 0xFFFF_FFFF_FFFF_FFFF
            elif bit_6_5 == 0b01:
                r = v1 ^ v2
            elif bit_6_5 == 0b10:
                r = v1 | v2
            else:
                r = v1 & v2
            self.gprs[rd_rs1].val = r
        else:
            raise NotImplementedError(f"C1 ALU sub_fn={sf:#03b}")
        return 2

    # -- C2: Quadrant 2 (低 2 位 = 10) --
    # C.SLLI / C.LWSP / C.LDSP / C.JR/C.MV/C.EBREAK/C.JALR / C.SWSP / C.SDSP

    def _handle_compressed_c2(
        self,
        instr: int,
    ) -> int:
        funct3 = (instr >> 13) & 0x7
        rd_rs1 = (instr >> 7) & 0x1F  # bits[11:7] — rs1(C.JR/C.JALR) 或 rd(C.MV/C.ADD)
        rs2 = (instr >> 2) & 0x1F     # bits[6:2] — rs2(C.MV/C.ADD) 或 0(C.JR/C.JALR)

        if funct3 == 0b000:
            shamt = ((instr >> 2) & 0x1F) | (((instr >> 7) & 0x1) << 5)
            self.gprs[rd_rs1].val = (
                self.gprs[rd_rs1].val << shamt
            ) & 0xFFFF_FFFF_FFFF_FFFF
            return 2

        # C.LWSP: uimm = {instr[6:5], instr[12], instr[4:2]} (4-byte aligned)
        # C.LDSP: uimm = {instr[4:2], instr[12], instr[6:5]} (8-byte aligned)
        if funct3 == 0b010:  # C.LWSP
            uimm = (
                ((instr >> 5) & 0b11) << 6
                | ((instr >> 12) & 0x1) << 5
                | ((instr >> 2) & 0b111) << 2
            )
            addr = (self.gprs[2].val + uimm) & 0xFFFF_FFFF_FFFF_FFFF
            mem = mem_read(self, addr, 4)
            val = mem[0] | (mem[1] << 8) | (mem[2] << 16) | (mem[3] << 24)
            self.gprs[rd_rs1].val = _sext(val, 32)
            return 2

        if funct3 == 0b011:  # C.LDSP (RV64C)
            uimm = (
                ((instr >> 2) & 0b111) << 6
                | ((instr >> 12) & 0x1) << 5
                | ((instr >> 5) & 0b11) << 3
            )
            addr = (self.gprs[2].val + uimm) & 0xFFFF_FFFF_FFFF_FFFF
            mem = mem_read(self, addr, 8)
            val = sum(mem[i] << (8 * i) for i in range(8))
            self.gprs[rd_rs1].val = val
            return 2

        if funct3 == 0b100:
            # C.JR (bit12=0, rs2==0, rd_rs1≠0) / C.JALR (bit12=1, rs2==0, rd_rs1≠0)
            # C.MV (rs2≠0, rd_rs1≠0) / C.EBREAK (rd_rs1==0, rs2==0)
            is_jalr = (instr >> 12) & 0x1
            if rd_rs1 == 0 and rs2 == 0:
                # C.EBREAK
                trap_ebreak(self)
                return 0
            elif rs2 == 0:
                # C.JR 或 C.JALR
                if is_jalr:
                    target = self.gprs[rd_rs1].val  # 先读跳转目标 (rd_rs1 可能 == 1)
                    self.gprs[1].val = (self.pc + 2) & 0xFFFF_FFFF_FFFF_FFFF  # ra
                else:
                    target = self.gprs[rd_rs1].val
                self.pc = target & ~1 & 0xFFFF_FFFF_FFFF_FFFF
                return 0
            else:
                if is_jalr:
                    # C.ADD (bit12=1, rs2≠0): rd += rs2
                    result = self.gprs[rd_rs1].val + self.gprs[rs2].val
                    self.gprs[rd_rs1].val = result & 0xFFFF_FFFF_FFFF_FFFF
                else:
                    # C.MV (bit12=0, rs2≠0): rd = rs2
                    self.gprs[rd_rs1].val = self.gprs[rs2].val
                return 2

        # C.SWSP: uimm = {instr[8:7], instr[12:9]} (4-byte aligned)
        # C.SDSP: uimm = {instr[9:7], instr[12:10]} (8-byte aligned)
        if funct3 == 0b110:  # C.SWSP
            uimm = ((instr >> 9) & 0xF) << 2 | ((instr >> 7) & 0x3) << 6
            addr = (self.gprs[2].val + uimm) & 0xFFFF_FFFF_FFFF_FFFF
            v = self.gprs[rs2].val & 0xFFFF_FFFF
            data = bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF])
            mem_write(self, addr, data)
            return 2

        if funct3 == 0b111:  # C.SDSP (RV64C)
            uimm = ((instr >> 7) & 0x7) << 6 | ((instr >> 10) & 0x7) << 3
            addr = (self.gprs[2].val + uimm) & 0xFFFF_FFFF_FFFF_FFFF
            v = self.gprs[rs2].val
            data = bytes([(v >> (8 * i)) & 0xFF for i in range(8)])
            mem_write(self, addr, data)
            return 2

        raise NotImplementedError(f"C2 funct3={funct3:#05b} (FLDSP/FSDSP)")

    # -- 压缩指令调度入口 --

    def handle_compressed(
        self,
        instr: int,
    ) -> int:
        """执行一条 16-bit 压缩指令, 按低 2 位分派到对应象限处理函数.

        Returns:
            2: 正常完成, 调用方做 PC += 2
            0: PC 已被该指令修改 (跳转/trap 等), 调用方不追加 PC
        """
        op = instr & 0x3
        if op == 0b00:
            return self._handle_compressed_c0(instr)
        elif op == 0b01:
            return self._handle_compressed_c1(instr)
        elif op == 0b10:
            return self._handle_compressed_c2(instr)
        else:
            raise ValueError(f"handle_compressed: not compressed {instr:#06x}")

    # ----------------------------------------------------------
    #  Top-level dispatch
    # ----------------------------------------------------------
    def exec_instr(
        self,
        instr: int,
    ) -> int:
        """解码并执行一条 RISC-V 指令 (32-bit 标准 / 16-bit 压缩).

        Returns:
            指令字节数, 供调用方推进 PC:
            - 2: 16-bit 压缩指令, PC 未修改
            - 4: 32-bit 标准指令, PC 未修改
            - 0: PC 已被该指令修改 (分支跳转/JAL/JALR/MRET/SRET/trap)
        """
        # 16-bit 压缩指令 — 非法编码触发 IllInstr 陷态
        if parse_compressed(instr):
            try:
                return self.handle_compressed(instr & 0xFFFF)
            except (ValueError, NotImplementedError, CsrAccessError):
                deliver_trap(self, 
                    TrapType.IllInstr, tval=instr, is_interrupt=False
                )
                return 0

        # 32-bit 标准指令 — 所有未识别的编码一律触发非法指令陷态
        pc_changed = False
        try:
            code = Opc(parse_opcode(instr))
        except ValueError:
            deliver_trap(self, TrapType.IllInstr, tval=instr, is_interrupt=False)
            return 0

        try:
            if code == Opc.op:
                self.handle_alu(instr)
            elif code == Opc.opImm:
                self.handle_op_imm(instr)
            elif code == Opc.opImm32:
                self.handle_op_imm32(instr)
            elif code == Opc.op32:
                self.handle_op32(instr)
            elif code == Opc.ld:
                self.handle_ld(instr)
            elif code == Opc.st:
                self.handle_st(instr)
            elif code == Opc.br:
                if self.handle_br(instr):
                    pc_changed = True
            elif code == Opc.jalr:
                self.handle_jalr(instr)
                pc_changed = True
            elif code == Opc.jal:
                rd = parse_rd(instr)
                imm = parse_imm_j(instr)
                if rd != 0:
                    self.gprs[rd].val = (self.pc + 4) & 0xFFFF_FFFF_FFFF_FFFF
                self.pc = (self.pc + imm) & 0xFFFF_FFFF_FFFF_FFFF
                pc_changed = True
            elif code == Opc.lui:
                # U-immediate → 32-bit value sign-extended to 64 bits (RV64)
                imm20 = _sext(parse_imm20_raw(instr) << 12, 32)
                rd = parse_rd(instr)
                if rd != 0:
                    self.gprs[rd].val = imm20 & 0xFFFF_FFFF_FFFF_FFFF
            elif code == Opc.auipc:
                # U-immediate → 32-bit offset sign-extended to 64 bits (RV64)
                imm20 = _sext(parse_imm20_raw(instr) << 12, 32)
                rd = parse_rd(instr)
                if rd != 0:
                    self.gprs[rd].val = (self.pc + imm20) & 0xFFFF_FFFF_FFFF_FFFF
            elif code == Opc.sys:
                saved_pc = self.pc
                self.handle_sys(instr)
                if self.pc != saved_pc:
                    pc_changed = True
            elif code == Opc.fence:
                self.handle_fence(instr)
            elif code == Opc.amo:
                self.handle_amo(instr)
            else:
                # 合法 opcode 但尚未实现 (如浮点 opfp)
                deliver_trap(self, 
                    TrapType.IllInstr, tval=instr, is_interrupt=False
                )
                return 0
        except (ValueError, NotImplementedError, CsrAccessError):
            # 操作码合法但编码字段无效 (如非法 funct3/funct12/nzuimm=0 等)
            deliver_trap(self, TrapType.IllInstr, tval=instr, is_interrupt=False)
            return 0

        return 0 if pc_changed else 4

"""
TODO: 也该考虑一下
执行状态暂存的事情了
"""
