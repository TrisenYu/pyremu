#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 20:50:31
# Last modified at 2026/06/08 星期一


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

from pyremu._native import (
    decode_compressed as _native_decode_compressed,
    decode_fields as _native_decode_fields,
    exec_alu_op as _native_alu_op,
    exec_op32 as _native_op32,
    exec_op_imm as _native_op_imm,
    exec_op_imm32 as _native_op_imm32,
    fp_exec_fma as _native_fp_fma,
    fp_exec_op as _native_fp_op,
    native_available,
)
from pyremu.core.hart import HartWithRegs
from pyremu.core.mem_check_aux import (
    AccessFault,
    AlignmentFault,
    MemoryAccessFault,
    PageFault,
    mem_read,
    mem_write,
    validate_csr,
)
from pyremu.core.registers import CsrAccessError
from pyremu.core.trap_def import TrapType
from pyremu.core.trap_handler import (
    deliver_trap,
    handle_wfi,
    trap_ebreak,
    trap_ecall,
    trap_mret,
    trap_sret,
)

from pyremu.utils.mask import (
    mask32,
    mask64
)
from pyremu.utils.disassem import (
    sext,
    sext8,
    sext16,
    sext32,
    brFn3, ldFn3, stFn3,
    AmoFunct5, AmoWidth,
    BRFN3_MAP, LDFN3_MAP, STFN3_MAP, AMOF5_MAP, AMOW_MAP,
)

_sint64 = ctypes.c_int64
_uint64 = ctypes.c_uint64

# 热路径优化: 预建 dict 查找表替代 Enum() 构造调用 (~0.10s 节省)
# Enum.__call__ 内部做线性搜索, dict.get 是 O(1) 哈希查找

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


def _flush_tlbs(origin, vpn: int | None, all_harts: list | None) -> None:
    """SFENCE.VMA helper: broadcast TLB flush across all harts.

    Args:
        origin: 发起 SFENCE.VMA 的 hart (本身也参与刷新).
        vpn: None = flush_all; 否则为 VA>>12 的 VPN.
        all_harts: 全部 hart 列表, 由调用方传入以避免每 hart 持有重复引用.
    """
    if vpn is None:
        origin.itlb.flush_all()
        origin.dtlb.flush_all()
        for h in (all_harts or []):
            if h is origin:
                continue
            h.itlb.flush_all()
            h.dtlb.flush_all()
        return
    origin.itlb.flush(vpn)
    origin.dtlb.flush(vpn)
    for h in (all_harts or []):
        if h is origin:
            continue
        h.itlb.flush(vpn)
        h.dtlb.flush(vpn)



class Hart(HartWithRegs):
    # 32-bit 指令分发表: opcode (int 0..127) -> handler 方法名 (O(1) dispatch)
    _DISPATCH: dict[int, str] = {
        0b01100_11: "handle_alu",       # Opc.op
        0b00100_11: "handle_op_imm",    # Opc.opImm
        0b00110_11: "handle_op_imm32",  # Opc.opImm32
        0b01110_11: "handle_op32",      # Opc.op32
        0b00000_11: "handle_ld",        # Opc.ld
        0b01000_11: "handle_st",        # Opc.st
        0b11000_11: "handle_br",        # Opc.br
        0b11001_11: "handle_jalr",      # Opc.jalr
        0b11011_11: "_handle_jal",      # Opc.jal
        0b01101_11: "_handle_lui",      # Opc.lui
        0b00101_11: "_handle_auipc",    # Opc.auipc
        0b11100_11: "handle_sys",       # Opc.sys
        0b00011_11: "handle_fence",     # Opc.fence
        0b01011_11: "handle_amo",       # Opc.amo
        0b00001_11: "handle_fp_load",   # Opc.opfp (FLW/FLD)
        0b01001_11: "handle_fp_store",  # Opc.stfp (FSW/FSD)
        0b10100_11: "handle_fp_op",     # Opc.opFp (OP-FP)
        0b10000_11: "handle_fp_fma",    # Opc.fmadd
        0b10001_11: "handle_fp_fma",    # Opc.fmsub
        0b10010_11: "handle_fp_fma",    # Opc.fnmsub
        0b10011_11: "handle_fp_fma",    # Opc.fnmadd
    }

    # ----------------------------------------------------------
    #  R-type ALU (opcode = Opc.op)
    # ----------------------------------------------------------
    def handle_alu(self, instr: int) -> int:
        """Execute an R-type ALU instruction on this hart."""
        f = self._f
        v1, v2 = self.gprs[f.rs1], self.gprs[f.rs2]

        if native_available():
            r = _native_alu_op(f.func3, f.func7, v1, v2)
            if r.trap:
                raise ValueError(
                    f"invalid funct3={f.func3:#b} funct7={f.func7:#x}"
                )
            self.gprs[f.rd] = r.value
            return 4

        # -- 纯 Python fallback (.so 缺失或调试时使用) --
        part1, part2 = f.func3, f.func7
        if part1 == 0b000:
            if part2 == 0:
                result = mask64((v1 + v2))
            elif part2 == 1:
                result = mask64((v1 * v2))
            elif part2 == 0x20:
                result = mask64((v1 - v2))
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=000")
        elif part1 == 0b001:
            if part2 == 0:
                result = mask64((v1 << (v2 & 0x3F)))
            elif part2 == 1:
                s1 = _sint64(v1).value
                s2 = _sint64(v2).value
                result = mask64(_sint64((s1 * s2) >> 64).value)
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=001")
        elif part1 == 0b010:
            if part2 == 0:
                result = 1 if _sint64(v1).value < _sint64(v2).value else 0
            elif part2 == 1:
                s1 = _sint64(v1).value
                u2 = _uint64(v2).value
                result = mask64(_sint64((s1 * u2) >> 64).value)
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=010")
        elif part1 == 0b011:
            if part2 == 0:
                result = 1 if _uint64(v1).value < _uint64(v2).value else 0
            elif part2 == 1:
                u1 = _uint64(v1).value
                u2 = _uint64(v2).value
                result = mask64(((u1 * u2) >> 64))
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=011")
        elif part1 == 0b100:
            if part2 not in {0, 1}:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=100")
            result = v1 ^ v2
            if part2 == 1:
                result = mask64(_trunc_div(_sint64(v1).value, _sint64(v2).value))
        elif part1 == 0b101:
            if part2 == 0:
                result = mask64((_uint64(v1).value >> (v2 & 0x3F)))
            elif part2 == 1:
                result = mask64(_trunc_div(_uint64(v1).value, _uint64(v2).value))
            elif part2 == 0x20:
                result = mask64(_sint64(_sint64(v1).value >> (v2 & 0x3F)).value)
            else:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=101")
        elif part1 == 0b110:
            if part2 not in {0, 1}:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=110")
            result = v1 | v2
            if part2 == 1:
                result = mask64(_trunc_rem(_sint64(v1).value, _sint64(v2).value))
        else:  # part1 == 0b111
            if part2 not in {0, 1}:
                raise ValueError(f"invalid funct7={part2:#x} for funct3=111")
            result = v1 & v2
            if part2 == 1:
                result = mask64(_trunc_rem(_uint64(v1).value, _uint64(v2).value))
        if f.rd != 0:
            self.gprs[f.rd] = result
        return 4

    # ----------------------------------------------------------
    #  I-type ALU (opcode = Opc.opImm)
    # ----------------------------------------------------------
    def handle_op_imm(self, instr: int) -> int:
        """Execute an I-type immediate ALU instruction."""
        f = self._f
        v1 = self.gprs[f.rs1]

        if native_available():
            r = _native_op_imm(f.func3, f.func7, v1, f.imm12_se)
            if r.trap:
                raise ValueError(
                    f"invalid funct3={f.func3:#b} funct7={f.func7:#x}"
                )
            self.gprs[f.rd] = r.value
            return 4

        # -- 纯 Python fallback --
        part1 = f.func3
        part6 = f.func7 >> 1
        imm = f.imm12_se
        shamt = f.imm12_se & 0x3F

        if part1 == 0b000:
            result = mask64((v1 + imm))
        elif part1 == 0b001:
            if part6 != 0:
                raise ValueError(f"invalid funct6={part6:#x} for SLLI")
            result = mask64((v1 << shamt))
        elif part1 == 0b010:
            result = 1 if _sint64(v1).value < imm else 0
        elif part1 == 0b011:
            result = 1 if _uint64(v1).value < _uint64(imm).value else 0
        elif part1 == 0b100:
            result = v1 ^ imm
        elif part1 == 0b101:
            if part6 == 0:
                result = mask64((_uint64(v1).value >> shamt))
            elif part6 == 0x10:
                result = mask64(_sint64(_sint64(v1).value >> shamt).value)
            else:
                raise ValueError(f"invalid funct6={part6:#x} for SRLI/SRAI")
        elif part1 == 0b110:
            result = v1 | imm
        else:
            result = v1 & imm
        if f.rd != 0:
            self.gprs[f.rd] = result
        return 4

    # ----------------------------------------------------------
    #  RV64 32-bit word operations (opcode = Opc.op32)
    # ----------------------------------------------------------
    def handle_op32(self, instr: int):
        """Execute an RV64 32-bit word operation (e.g. ADDW, SUBW, SLLW, etc.)."""
        f = self._f
        v1, v2 = self.gprs[f.rs1], self.gprs[f.rs2]

        if native_available():
            r = _native_op32(f.func3, f.func7, v1, v2)
            if r.trap:
                raise ValueError(
                    f"invalid funct3={f.func3:#b} funct7={f.func7:#x}"
                )
            self.gprs[f.rd] = r.value
            return 4

        # -- 纯 Python fallback --
        part1, part2 = f.func3, f.func7
        v1 = mask32(v1)
        v2 = mask32(v2)

        if part1 == 0b000:
            if part2 == 0:
                result = mask32((v1 + v2))
            elif part2 == 1:
                result = mask32((v1 * v2))
            elif part2 == 0x20:
                result = mask32((v1 - v2))
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=000")
            result = sext32(result)
        elif part1 == 0b001:
            if part2 == 0:
                result = mask32((v1 << (v2 & 0x1F)))
                result = sext32(result)
            elif part2 == 1:
                s1 = _sint64(sext32(v1)).value
                s2 = _sint64(sext32(v2)).value
                result = mask64(_sint64((s1 * s2) >> 32).value)
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=001")
        elif part1 == 0b010:
            if part2 == 0:
                result = (
                    1 if _sint64(sext32(v1)).value < _sint64(sext32(v2)).value else 0
                )
            elif part2 == 1:
                s1 = _sint64(sext32(v1)).value
                u2 = _uint64(v2).value
                result = mask64(_sint64((s1 * u2) >> 32).value)
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=010")
        elif part1 == 0b011:
            if part2 == 0:
                result = 1 if v1 < v2 else 0
            elif part2 == 1:
                result = mask64(((v1 * v2) >> 32))
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=011")
        elif part1 == 0b100:
            if part2 == 0:
                result = mask32((v1 ^ v2))
                result = sext32(result)
            elif part2 == 1:
                result = _trunc_div(
                    _sint64(sext32(v1)).value,
                    _sint64(sext32(v2)).value,
                )
                result = sext32(result & 0xFFFF_FFFF)
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=100")
        elif part1 == 0b101:
            if part2 == 0:
                result = mask32((v1 >> (v2 & 0x1F)))
                result = sext32(result)
            elif part2 == 1:
                result = _trunc_div(v1, v2)
                result = sext32(result & 0xFFFF_FFFF)
            elif part2 == 0x20:
                result = _sint64(sext32(v1) >> (v2 & 0x1F)).value
                result = sext32(result & 0xFFFF_FFFF)
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=101")
        elif part1 == 0b110:
            if part2 == 0:
                result = mask32((v1 | v2))
                result = sext32(result)
            elif part2 == 1:
                result = _trunc_rem(
                    _sint64(sext32(v1)).value,
                    _sint64(sext32(v2)).value,
                )
                result = sext32(result & 0xFFFF_FFFF)
            else:
                raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=110")
        elif part2 == 0:
            result = mask32((v1 & v2))
            result = sext32(result)
        elif part2 == 1:
            result = _trunc_rem(v1, v2)
            result = sext32(result & 0xFFFF_FFFF)
        else:
            raise ValueError(f"invalid funct7={part2:#x} for op32 funct3=111")
        if f.rd != 0:
            self.gprs[f.rd] = result
        return 4

    # ----------------------------------------------------------
    #  I-type immediate word operations (opcode = Opc.opImm32)
    # ----------------------------------------------------------
    def handle_op_imm32(self, instr: int):
        """Execute an I-type 32-bit word immediate ALU instruction (ADDIW, SLLIW, etc.)."""
        f = self._f
        v1 = self.gprs[f.rs1]

        if native_available():
            r = _native_op_imm32(f.func3, f.func7, v1, f.imm12_se)
            if r.trap:
                raise ValueError(
                    f"invalid funct3={f.func3:#b} funct7={f.func7:#x}"
                )
            self.gprs[f.rd] = r.value
            return 4

        # -- 纯 Python fallback --
        part1, part7 = f.func3, f.func7
        imm = f.imm12_se
        shamt = f.imm12_se & 0x1F

        if part1 == 0b000:
            result = mask32((v1 + imm))
            result = sext32(result)
        elif part1 == 0b001:
            if part7 != 0:
                raise ValueError(f"invalid funct7={part7:#x} for SLLIW")
            result = mask32(((v1 & 0xFFFF_FFFF) << shamt))
            result = sext32(result)
        elif part1 == 0b101:
            if part7 == 0:
                result = mask32(((v1 & 0xFFFF_FFFF) >> shamt))
            elif part7 == 0x20:
                result = mask32(_sint64(sext32(v1 & 0xFFFF_FFFF) >> shamt).value)
            else:
                raise ValueError(f"invalid funct7={part7:#x} for SRLIW/SRAIW")
            result = sext32(result)
        else:
            raise ValueError(f"invalid funct3={part1:#x} for opImm32")
        if f.rd != 0:
            self.gprs[f.rd] = result
        return 4

    # ----------------------------------------------------------
    #  Branch (opcode = Opc.br)
    # ----------------------------------------------------------
    def handle_br(self, instr: int) -> int:
        """Execute a conditional branch.  Returns 0 if taken (pc updated),
        4 if not taken (pc advances normally)."""
        f = self._f
        fn3 = f.func3
        rs1 = f.rs1
        rs2 = f.rs2
        offset = f.imm_b

        v1 = self.gprs[rs1]
        v2 = self.gprs[rs2]

        taken = False
        f = BRFN3_MAP.get(fn3)
        if f is None:
            return 4

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
            self.pc = mask64((self.pc + offset))
            return 0
        return 4

    # ----------------------------------------------------------
    #  Load (opcode = Opc.ld)
    # ----------------------------------------------------------
    def handle_ld(self, instr: int):
        """Execute a load instruction."""
        f = self._f
        fn3 = f.func3
        rd = f.rd
        rs1 = f.rs1
        offset = f.imm12_se

        addr = mask64((self.gprs[rs1] + offset))

        f = LDFN3_MAP.get(fn3)
        if f is None:
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

        saved_pc = self.pc
        mem = mem_read(self, addr, read_size)
        # mem_read may have delivered a trap (LdPageFault / LdAccessFault) and
        # redirected PC to stvec.  If so, do NOT overwrite rd — the kernel's
        # trap handler must see the original register state.
        if self.pc != saved_pc:
            return 0

        if f == ldFn3.lb:
            val = sext8(mem[0])
        elif f == ldFn3.lh:
            val = sext16(mem[0] | (mem[1] << 8))
        elif f == ldFn3.lw:
            val = sext32(mem[0] | (mem[1] << 8) | (mem[2] << 16) | (mem[3] << 24))
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
            self.gprs[rd] = val
        return 4

    # ----------------------------------------------------------
    #  Store (opcode = Opc.st)
    # ----------------------------------------------------------
    def handle_st(self, instr: int):
        """Execute a store instruction."""
        f = self._f
        fn3 = f.func3
        rs1 = f.rs1
        rs2 = f.rs2
        offset = f.imm_s

        addr = mask64((self.gprs[rs1] + offset))
        val = self.gprs[rs2]

        f = STFN3_MAP.get(fn3)
        if f is None:
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

        saved_pc = self.pc
        mem_write(self, addr, data)
        # mem_write may have delivered a trap (StPageFault / StAccessFault) and
        # redirected PC to stvec.  If so, return 0 so the caller doesn't advance PC.
        if self.pc != saved_pc:
            return 0
        return 4

    # ----------------------------------------------------------
    #  F/D floating point (opcode = opfp/stfp/opFp/fmadd…)
    #
    #  计算委托给 native softfloat (fp_exec_op/fp_exec_fma);
    #  Python 侧仅负责寄存器/内存路由与 NaN-boxing。无纯 Python FPU。
    # ----------------------------------------------------------

    _NANBOX_S = 0xFFFF_FFFF_0000_0000
    _MSTATUS_FS = 0b11 << 13
    _MSTATUS_SD = 1 << 63

    def _fp_enabled(self) -> bool:
        """mstatus.FS != Off。"""
        return (self.mstatus_val & self._MSTATUS_FS) != 0

    def _fp_mark_dirty(self) -> None:
        """执行浮点指令后置 FS=Dirty 与 SD。"""
        self.mstatus_val = self.mstatus_val | self._MSTATUS_FS | self._MSTATUS_SD

    def _fp_frm(self) -> int:
        """当前动态舍入模式 fcsr.frm。"""
        return (self.csrs["fcsr"].val >> 5) & 0x7

    def _fp_accumulate_flags(self, fflags: int) -> None:
        """将异常标志累积进 fcsr.fflags 并同步 fflags CSR。"""
        fcsr = (self.csrs["fcsr"].val | (fflags & 0x1F)) & 0xFF
        self.csrs["fcsr"].val = fcsr
        self.csrs["fflags"].val = fcsr & 0x1F

    def handle_fp_load(self, instr: int) -> int:
        """FLW / FLD — 从内存加载到 FPR。FLW 结果 NaN-boxed。"""
        if not self._fp_enabled():
            raise ValueError("FP disabled (mstatus.FS=Off)")
        f = self._f
        fn3 = f.func3
        if fn3 == 0b010:
            size = 4
        elif fn3 == 0b011:
            size = 8
        else:
            raise ValueError(f"invalid funct3={fn3:#x} for FP load")
        addr = mask64((self.gprs[f.rs1] + f.imm12_se))
        saved_pc = self.pc
        mem = mem_read(self, addr, size)
        if self.pc != saved_pc:
            return 0
        val = 0
        for i in range(size):
            val |= mem[i] << (8 * i)
        self._fpr_bits[f.rd] = (self._NANBOX_S | val) if size == 4 else val
        self._fp_mark_dirty()
        return 4

    def handle_fp_store(self, instr: int) -> int:
        """FSW / FSD — 将 FPR 存入内存。FSW 存低 32 位。"""
        if not self._fp_enabled():
            raise ValueError("FP disabled (mstatus.FS=Off)")
        f = self._f
        fn3 = f.func3
        if fn3 == 0b010:
            size = 4
        elif fn3 == 0b011:
            size = 8
        else:
            raise ValueError(f"invalid funct3={fn3:#x} for FP store")
        addr = mask64((self.gprs[f.rs1] + f.imm_s))
        val = self._fpr_bits[f.rs2]
        data = bytes([(val >> (8 * i)) & 0xFF for i in range(size)])
        saved_pc = self.pc
        mem_write(self, addr, data)
        if self.pc != saved_pc:
            return 0
        return 4

    def handle_fp_op(self, instr: int) -> int:
        """OP-FP — 算术/转换/比较/符号/分类/移动 (native 计算)。"""
        if not self._fp_enabled():
            raise ValueError("FP disabled (mstatus.FS=Off)")
        f = self._f
        op5 = f.func7 >> 2
        # int->float (0x1A) 与 FMV.*.X (0x1E) 的 rs1 源为 GPR。
        if op5 in (0x1A, 0x1E):
            rs1_bits = self.gprs[f.rs1]
        else:
            rs1_bits = self._fpr_bits[f.rs1]
        rs2_bits = self._fpr_bits[f.rs2]
        out = _native_fp_op(f.func7, f.func3, f.rs2, rs1_bits, rs2_bits, self._fp_frm())
        if out.trap:
            raise ValueError(f"invalid OP-FP funct7={f.func7:#x} funct3={f.func3:#x}")
        if out.to_gpr:
            if f.rd != 0:
                self.gprs[f.rd] = out.value
        else:
            self._fpr_bits[f.rd] = out.value
        self._fp_accumulate_flags(out.fflags)
        self._fp_mark_dirty()
        return 4

    def handle_fp_fma(self, instr: int) -> int:
        """FMADD/FMSUB/FNMSUB/FNMADD — 融合乘加 (native 计算)。"""
        if not self._fp_enabled():
            raise ValueError("FP disabled (mstatus.FS=Off)")
        f = self._f
        out = _native_fp_fma(
            f.opcode, f.func3, f.fmt,
            self._fpr_bits[f.rs1], self._fpr_bits[f.rs2], self._fpr_bits[f.rs3],
            self._fp_frm(),
        )
        if out.trap:
            raise ValueError(f"invalid FMA opcode={f.opcode:#x} fmt={f.fmt}")
        self._fpr_bits[f.rd] = out.value
        self._fp_accumulate_flags(out.fflags)
        self._fp_mark_dirty()
        return 4

    # ----------------------------------------------------------
    #  Atomic Memory Operations (opcode = Opc.amo)
    # ----------------------------------------------------------
    # AMO 编码: funct5[31:27] | aq[26] | rl[25] | rs2[24:20] |
    #           rs1[19:15] | funct3[14:12] | rd[11:7] | opcode=0101111
    # aq/rl 为内存排序提示, 顺序执行模型中可忽略.

    def handle_amo(
        self,
        instr: int,
    ) -> int:
        """执行原子内存操作 (LR/SC/AMOxxx)."""
        f = self._f
        funct5_val = f.func7 >> 2  # bits[31:27] = func7[6:2]
        funct3_val = f.func3
        rd = f.rd
        rs1 = f.rs1
        rs2 = f.rs2

        op = AMOF5_MAP.get(funct5_val)
        width = AMOW_MAP.get(funct3_val)
        if op is None or width is None:
            raise ValueError(
                f"invalid AMO encoding: funct5={funct5_val:#07b}, funct3={funct3_val:#05b}"
            )

        is_64bit = width == AmoWidth.D
        byte_len = 8 if is_64bit else 4
        mask = 0xFFFF_FFFF_FFFF_FFFF if is_64bit else 0xFFFF_FFFF

        addr = mask64(self.gprs[rs1])

        if op == AmoFunct5.LR:
            # Load-Reserved: 读取内存并设置预留
            saved_pc = self.pc
            data_bytes = mem_read(self, addr, byte_len)
            if self.pc != saved_pc:
                return 0
            val = int.from_bytes(data_bytes, "little", signed=False) & mask
            if rd != 0:
                # LR.D: 64-bit 值不需要符号扩展; LR.W: 32->64 符号扩展
                self.gprs[rd] = val if is_64bit else sext32(val)
            self.set_reservation(addr, val)
            return 4
        elif op == AmoFunct5.SC:
            # Store-Conditional: 仅预留有效时写入
            if self.reservation_valid and self.reservation_addr == addr:
                store_val = self.gprs[rs2] & mask
                data = store_val.to_bytes(byte_len, "little", signed=False)
                saved_pc = self.pc
                mem_write(self, addr, data)
                if self.pc != saved_pc:
                    return 0
                if rd != 0:
                    self.gprs[rd] = 0  # 成功 -> rd ← 0
            elif rd != 0:
                self.gprs[rd] = 1  # 失败 -> rd ← 非零
            self.clear_reservation()
            return 4
        # else: AMOxxx - 原子读-改-写
        saved_pc = self.pc
        data_bytes = mem_read(self, addr, byte_len)
        if self.pc != saved_pc:
            return 0
        mem_val = int.from_bytes(data_bytes, "little", signed=False) & mask
        op_val = self.gprs[rs2] & mask

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
            s_mem = _sint64(sext(mem_val, 64 if is_64bit else 32)).value
            s_op = _sint64(sext(op_val, 64 if is_64bit else 32)).value
            result = (op_val if s_op < s_mem else mem_val) & mask
        elif op == AmoFunct5.MAX:
            s_mem = _sint64(sext(mem_val, 64 if is_64bit else 32)).value
            s_op = _sint64(sext(op_val, 64 if is_64bit else 32)).value
            result = (op_val if s_op > s_mem else mem_val) & mask
        elif op == AmoFunct5.MINU:
            result = (op_val if op_val < mem_val else mem_val) & mask
        elif op == AmoFunct5.MAXU:
            result = (op_val if op_val > mem_val else mem_val) & mask
        else:
            raise ValueError(f"unhandled AMO op: {op}")

        data = result.to_bytes(byte_len, "little", signed=False)
        saved_pc = self.pc
        mem_write(self, addr, data)
        if self.pc != saved_pc:
            return 0
        if rd != 0:
            self.gprs[rd] = sext(mem_val, 64) if is_64bit else sext32(mem_val)
        return 4

    # ----------------------------------------------------------
    #  JALR (opcode = Opc.jalr)
    # ----------------------------------------------------------
    def handle_jalr(self, instr: int) -> int:
        """Execute JALR: rd = pc+4; pc = (rs1 + imm) & ~1"""
        f = self._f
        rd = f.rd
        rs1 = f.rs1
        imm = f.imm12_se
        next_pc = mask64((self.pc + 4))
        target = mask64((self.gprs[rs1] + imm))
        target &= ~1  # clear LSB to align

        if rd != 0:
            self.gprs[rd] = next_pc
        self.pc = target
        return 0

    # ----------------------------------------------------------
    #  System (opcode = Opc.sys)  -- ecall / ebreak / CSR / mret / ...
    # ----------------------------------------------------------
    def handle_sys(self, instr: int) -> int:
        """Execute a system instruction (privileged or CSR access).
        Returns 0 if PC was modified (trap/mret/sret), 4 otherwise."""
        saved_pc = self.pc
        f = self._f
        fn3 = f.func3
        rd = f.rd
        rs1 = f.rs1
        csr_addr = f.func12  # CSR address [31:20] = func12
        uimm = rs1  # 5-bit zero-extended imm for CSR

        if fn3 == 0b000:
            # Privileged instructions: funct12 determines the operation
            funct12 = f.func12
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
            elif funct12 == 0x120 or (0x121 <= funct12 <= 0x13F):
                # SFENCE.VMA: funct7=0b0001001, funct12 = 0x120 | rs2.
                # RISC-V spec: rs1=x0 时刷新全部 TLB; rs1≠x0 时仅刷新该 VA 对应条目.
                # 同步广播到全部 hart — 匹配 Rust 引擎 tlb_gen 语义, 确保多核 TLB 一致性.
                vpn = None if rs1 == 0 else (self.gprs[rs1] >> 12)
                _flush_tlbs(self, vpn, self._all_harts)
            elif funct12 == 0x5A0:  # MFENCE.DID — 按内存域刷新全部 hart TLB + L2
                # 读取当前 hart 的 mdid, 广播刷新所有 hart 中匹配的条目
                mdid_val = self.mdid_val
                for h in self._all_harts or [self]:
                    h.itlb.flush_by_mdid(mdid_val)
                    h.dtlb.flush_by_mdid(mdid_val)
                # 刷新共享 L2 缓存中匹配的条目
                if self._bus is not None:
                    l2 = getattr(self._bus, "_l2", None)
                    if l2 is not None:
                        l2.flush_by_mdid(mdid_val)
            else:
                raise ValueError(f"unknown privileged funct12={funct12:#05x}")
            return 0 if self.pc != saved_pc else 4

        elif fn3 == 0b001:  # CSRRW
            validate_csr(self, csr_addr, is_write=True)
            old_csr = self.read_csr(csr_addr)
            new_csr = self.gprs[rs1]  # 先读 rs1 (可能在 rd==rs1 时被覆盖)
            if rd != 0:
                self.gprs[rd] = old_csr
            self.write_csr(csr_addr, new_csr)
            return 0 if self.pc != saved_pc else 4

        elif fn3 == 0b010:  # CSRRS
            do_write = rs1 != 0
            validate_csr(self, csr_addr, is_write=do_write)
            old_csr = self.read_csr(csr_addr)
            rs1_val = self.gprs[rs1]  # 先读 rs1 (可能在 rd==rs1 时被覆盖)
            if rd != 0:
                self.gprs[rd] = old_csr
            if do_write:
                self.write_csr(csr_addr, old_csr | rs1_val)
            return 0 if self.pc != saved_pc else 4

        elif fn3 == 0b011:  # CSRRC
            do_write = rs1 != 0
            validate_csr(self, csr_addr, is_write=do_write)
            old_csr = self.read_csr(csr_addr)
            rs1_val = self.gprs[rs1]  # 先读 rs1 (可能在 rd==rs1 时被覆盖)
            if rd != 0:
                self.gprs[rd] = old_csr
            if do_write:
                self.write_csr(csr_addr, old_csr & ~rs1_val)
            return 0 if self.pc != saved_pc else 4

        elif fn3 == 0b101:  # CSRRWI
            validate_csr(self, csr_addr, is_write=True)
            old = self.read_csr(csr_addr)
            if rd != 0:
                self.gprs[rd] = old
            self.write_csr(csr_addr, uimm)
            return 0 if self.pc != saved_pc else 4

        elif fn3 == 0b110:  # CSRRSI
            do_write = uimm != 0
            validate_csr(self, csr_addr, is_write=do_write)
            old = self.read_csr(csr_addr)
            if rd != 0:
                self.gprs[rd] = old
            if do_write:
                self.write_csr(csr_addr, old | uimm)
            return 0 if self.pc != saved_pc else 4

        elif fn3 == 0b111:  # CSRRCI
            do_write = uimm != 0
            validate_csr(self, csr_addr, is_write=do_write)
            old = self.read_csr(csr_addr)
            if rd != 0:
                self.gprs[rd] = old
            if do_write:
                self.write_csr(csr_addr, old & ~uimm)
            return 0 if self.pc != saved_pc else 4

        raise ValueError(f"invalid funct3={fn3:#x} for sys")


    # ----------------------------------------------------------
    #  FENCE (opcode = Opc.fence)
    # ----------------------------------------------------------
    def handle_fence(self, instr: int) -> int:
        """Execute a FENCE / FENCE.I instruction.
        In a single-hart, in-order emulator these are mostly no-ops."""
        fn3 = self._f.func3
        if fn3 == 0b000:  # FENCE
            pass  # no-op: sequential execution is already ordered
        elif fn3 == 0b001:  # FENCE.I
            pass  # TODO: flush instruction cache / pipeline
        else:
            raise ValueError(f"invalid funct3={fn3:#x} for fence")
        return 4

    # ----------------------------------------------------------
    #  JAL / LUI / AUIPC (从 exec_instr 内联代码提取)
    # ----------------------------------------------------------

    def _handle_jal(self, instr: int) -> int:
        """JAL: rd = pc+4; pc += imm.  Returns 0 (pc always modified)."""
        f = self._f
        rd = f.rd
        imm = f.imm_j
        if rd != 0:
            self.gprs[rd] = mask64((self.pc + 4))
        self.pc = mask64((self.pc + imm))
        return 0

    def _handle_lui(self, instr: int) -> int:
        """LUI: rd = imm20 << 12.  Returns 4 (pc advances normally)."""
        f = self._f
        imm20 = sext32(f.imm20_raw << 12)
        rd = f.rd
        if rd != 0:
            self.gprs[rd] = mask64(imm20)
        return 4

    def _handle_auipc(self, instr: int) -> int:
        """AUIPC: rd = pc + (imm20 << 12).  Returns 4 (pc advances normally)."""
        f = self._f
        imm20 = sext32(f.imm20_raw << 12)
        rd = f.rd
        if rd != 0:
            self.gprs[rd] = mask64((self.pc + imm20))
        return 4

    @staticmethod
    def _creg(n: int) -> int:
        """3-bit 压缩寄存器号 -> 完整寄存器号 (x8–x15)."""
        return (n & 0x7) + 8

    # -- C0: Quadrant 0 (低 2 位 = 00) --
    # C.ADDI4SPN / C.LW / C.LD / C.SW / C.SD (FLD/FSD 暂未实现)

    def _handle_compressed_c0(
        self,
        instr: int,
    ) -> int:
        cf = self._cf
        funct3 = cf.funct3
        rd = cf.rd  # C0: creg-mapped by Rust

        if funct3 == 0b000:
            # C.ADDI4SPN — nzuimm pre-decoded by Rust
            nzuimm = cf.imm
            if nzuimm == 0:
                deliver_trap(self, TrapType.IllInstr, tval=instr, is_interrupt=False)
                return 0
            self.gprs[rd] = mask64((self.gprs[2] + nzuimm))
            return 2

        rs1 = cf.rs1p  # C0: creg-mapped rs1'

        # C.FLD (RV64DC): fpr[rd'] = mem[rs1' + uimm]
        if funct3 == 0b001:
            if not self._fp_enabled():
                raise ValueError("C.FLD: FP disabled (mstatus.FS=Off)")
            uimm = cf.imm
            addr = mask64((self.gprs[rs1] + uimm))
            saved_pc = self.pc
            mem = mem_read(self, addr, 8)
            if self.pc != saved_pc:
                return 0
            val = sum(mem[i] << (8 * i) for i in range(8))
            self._fpr_bits[rd] = val
            self._fp_mark_dirty()
            return 2

        # C.LW / C.SW: uimm pre-decoded by Rust
        if funct3 in (0b010, 0b110):
            uimm = cf.imm
            addr = mask64((self.gprs[rs1] + uimm))
            if funct3 == 0b010:  # C.LW
                saved_pc = self.pc
                mem = mem_read(self, addr, 4)
                if self.pc != saved_pc:
                    return 0
                val = mem[0] | (mem[1] << 8) | (mem[2] << 16) | (mem[3] << 24)
                self.gprs[rd] = sext32(val)
            else:  # C.SW
                rs2 = cf.rdp
                v = mask32(self.gprs[rs2])
                data = bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF])
                saved_pc = self.pc
                mem_write(self, addr, data)
                if self.pc != saved_pc:
                    return 0

        # C.LD / C.SD / C.FSD (RV64C): uimm pre-decoded by Rust (8-byte aligned)
        elif funct3 in (0b011, 0b111, 0b101):
            uimm = cf.imm
            addr = mask64((self.gprs[rs1] + uimm))
            if funct3 == 0b011:  # C.LD
                saved_pc = self.pc
                mem = mem_read(self, addr, 8)
                if self.pc != saved_pc:
                    return 0
                val = sum(mem[i] << (8 * i) for i in range(8))
                self.gprs[rd] = val
            elif funct3 == 0b101:  # C.FSD
                if not self._fp_enabled():
                    raise ValueError("C.FSD: FP disabled (mstatus.FS=Off)")
                v = self._fpr_bits[rd]
                data = bytes([(v >> (8 * i)) & 0xFF for i in range(8)])
                saved_pc = self.pc
                mem_write(self, addr, data)
                if self.pc != saved_pc:
                    return 0
                self._fp_mark_dirty()
            else:  # C.SD
                rs2 = cf.rdp
                v = self.gprs[rs2]
                data = bytes([(v >> (8 * i)) & 0xFF for i in range(8)])
                saved_pc = self.pc
                mem_write(self, addr, data)
                if self.pc != saved_pc:
                    return 0

        else:
            raise NotImplementedError(f"C0 funct3={funct3:#05b} (reserved)")
        return 2

    # -- C1: Quadrant 1 (低 2 位 = 01) --
    # C.ADDI / C.ADDIW / C.LI / C.LUI / C.ADDI16SP /
    # C.SRLI/C.SRAI/C.ANDI / C.SUB/C.XOR/C.OR/C.AND /
    # C.J / C.BEQZ / C.BNEZ

    def _handle_compressed_c1(
        self,
        instr: int,
    ) -> int:
        cf = self._cf
        funct3 = cf.funct3
        rd_raw = cf.rd  # C1: rd/rs1 在 bits [11:7], 5-bit 非 creg 映射

        # C.ADDI / C.ADDIW / C.LI — imm 已由 Rust 符号扩展 (rd≠0)
        if funct3 in (0b000, 0b001, 0b010):
            imm = cf.imm
            if funct3 == 0b000:
                self.gprs[rd_raw] = mask64((self.gprs[rd_raw] + imm))
            elif funct3 == 0b001:
                r = mask32((self.gprs[rd_raw] + imm))
                self.gprs[rd_raw] = sext32(r)
            else:
                self.gprs[rd_raw] = mask64(imm)
            return 2

        # C.LUI (rd≠{0,2}) / C.ADDI16SP (rd=2) — imm 已由 Rust 符号扩展
        if funct3 == 0b011:
            nz = cf.imm
            if rd_raw == 2:
                # C.ADDI16SP: nz 已 10-bit 符号扩展 (低 4 bit 恒零)
                if nz == 0:
                    raise ValueError("C.ADDI16SP: nzuimm must be non-zero")
                self.gprs[2] = mask64((self.gprs[2] + nz))
            else:
                # C.LUI: nz 已 6-bit 符号扩展 -> 左移 12
                if nz == 0:
                    raise ValueError("C.LUI: nzuimm must be non-zero")
                self.gprs[rd_raw] = mask64((nz << 12))
            return 2

        # C1 ALU ops
        if funct3 == 0b100:
            return self._handle_compressed_c1_alu(instr)

        # C.J — offset 已由 Rust 符号扩展
        if funct3 == 0b101:
            self.pc = mask64((self.pc + cf.imm))
            return 0

        # C.BEQZ / C.BNEZ — rs1 = cf.rs1p (creg-mapped from bits[9:7])
        if funct3 in (0b110, 0b111):
            rs1 = cf.rs1p
            offset = cf.imm  # 已由 Rust 符号扩展
            taken = self.gprs[rs1] == 0
            if funct3 == 0b111:
                taken = not taken
            if taken:
                self.pc = mask64((self.pc + offset))
                return 0
            return 2

        raise NotImplementedError(f"C1 funct3={funct3:#05b}")

    def _handle_compressed_c1_alu(
        self,
        instr: int,
    ) -> int:
        """C1 ALU (funct3=100).  All fields pre-decoded by Rust into ``self._cf``.

        sf=00 -> C.SRLI  (shamt = {bit12, bits[6:2]}, 1-63 for RV64C)
        sf=01 -> C.SRAI  (shamt = {bit12, bits[6:2]}, 1-63 for RV64C)
        sf=10 -> C.ANDI (imm[5]=bit12, imm[4:0]=bits[6:2])
        sf=11 bit12=0 -> C.SUB/C.XOR/C.OR/C.AND (bits[6:5]: 00=SUB,01=XOR,10=OR,11=AND)
        sf=11 bit12=1 -> C.SUBW/C.ADDW (bits[6:5]: 00=SUBW, 01=ADDW)
        """
        cf = self._cf
        sf = cf.sf
        rd_rs1 = cf.rs1p  # creg-mapped from bits[9:7]
        v1 = self.gprs[rd_rs1]

        if sf == 0b00:
            # C.SRLI: shamt = {bit12, bits[6:2]} (1-63 for RV64C)
            shamt = (cf.bit12 << 5) | (cf.rs2 & 0x1F)
            v1 = mask64((_uint64(v1).value >> shamt))
            self.gprs[rd_rs1] = v1
        elif sf == 0b01:
            # C.SRAI: shamt = {bit12, bits[6:2]} (1-63 for RV64C)
            shamt = (cf.bit12 << 5) | (cf.rs2 & 0x1F)
            v1 = mask64(_sint64(_sint64(v1).value >> shamt).value)
            self.gprs[rd_rs1] = v1
        elif sf == 0b10:
            # C.ANDI — imm[5:0] = {bit12, bits[6:2]}, sign-extended from 6 bits
            imm = sext((cf.bit12 << 5) | (cf.rs2 & 0x1F), 6)
            self.gprs[rd_rs1] = mask64((v1 & imm))
        elif sf == 0b11:
            rs2 = cf.rdp  # creg-mapped from bits[4:2]
            v2 = self.gprs[rs2]
            bit_6_5 = cf.bit65
            if cf.bit12 == 0:
                # C.SUB / C.XOR / C.OR / C.AND
                if bit_6_5 == 0b00:
                    r = mask64((v1 - v2))
                elif bit_6_5 == 0b01:
                    r = v1 ^ v2
                elif bit_6_5 == 0b10:
                    r = v1 | v2
                else:
                    r = v1 & v2
                self.gprs[rd_rs1] = r
            # C.SUBW / C.ADDW (RV64C only)
            elif bit_6_5 == 0b00:
                r = mask32((v1 - v2))
                self.gprs[rd_rs1] = sext32(r)
            elif bit_6_5 == 0b01:
                r = mask32((v1 + v2))
                self.gprs[rd_rs1] = sext32(r)
            else:
                raise NotImplementedError(
                    f"C.SUBW/C.ADDW reserved bit[6:5]={bit_6_5:#03b}"
                )
        else:
            raise NotImplementedError(f"C1 ALU sub_fn={sf:#03b}")
        return 2

    # -- C2: Quadrant 2 (低 2 位 = 10) --
    # C.SLLI / C.LWSP / C.LDSP / C.JR/C.MV/C.EBREAK/C.JALR / C.SWSP / C.SDSP

    def _handle_compressed_c2(
        self,
        instr: int,
    ) -> int:
        cf = self._cf
        funct3 = cf.funct3
        rd_rs1 = cf.rd  # bits[11:7] — rs1(C.JR/C.JALR) 或 rd(C.MV/C.ADD)
        rs2 = cf.rs2   # bits[6:2] — rs2(C.MV/C.ADD) 或 0(C.JR/C.JALR)

        if funct3 == 0b000:
            # C.SLLI (RV64): shamt = cf.imm (Rust pre-decoded)
            self.gprs[rd_rs1] = mask64((self.gprs[rd_rs1] << cf.imm))
            return 2

        # C.FLDSP: fpr[rd] = mem[sp + uimm] (RV64DC)
        if funct3 == 0b001:
            if not self._fp_enabled():
                raise ValueError("C.FLDSP: FP disabled (mstatus.FS=Off)")
            uimm = cf.imm
            addr = mask64((self.gprs[2] + uimm))
            saved_pc = self.pc
            mem = mem_read(self, addr, 8)
            if self.pc != saved_pc:
                return 0
            val = sum(mem[i] << (8 * i) for i in range(8))
            self._fpr_bits[rd_rs1] = val
            self._fp_mark_dirty()
            return 2

        # C.LWSP: uimm = cf.imm (4-byte aligned)
        if funct3 == 0b010:
            uimm = cf.imm
            addr = mask64((self.gprs[2] + uimm))
            saved_pc = self.pc
            mem = mem_read(self, addr, 4)
            if self.pc != saved_pc:
                return 0
            val = mem[0] | (mem[1] << 8) | (mem[2] << 16) | (mem[3] << 24)
            self.gprs[rd_rs1] = sext32(val)
            return 2

        # C.LDSP: uimm = cf.imm (8-byte aligned)
        if funct3 == 0b011:
            uimm = cf.imm
            addr = mask64((self.gprs[2] + uimm))
            saved_pc = self.pc
            mem = mem_read(self, addr, 8)
            if self.pc != saved_pc:
                return 0
            val = sum(mem[i] << (8 * i) for i in range(8))
            self.gprs[rd_rs1] = val
            return 2

        if funct3 == 0b100:
            # C.JR (bit12=0, rs2==0, rd_rs1≠0) / C.JALR (bit12=1, rs2==0, rd_rs1≠0)
            # C.MV (rs2≠0, rd_rs1≠0) / C.EBREAK (rd_rs1==0, rs2==0)
            is_jalr = cf.bit12
            if rd_rs1 == 0 and rs2 == 0:
                # C.EBREAK
                trap_ebreak(self)
                return 0
            elif rs2 == 0:
                # C.JR 或 C.JALR
                if is_jalr:
                    target = self.gprs[rd_rs1]  # 先读跳转目标 (rd_rs1 可能 == 1)
                    self.gprs[1] = mask64((self.pc + 2))  # ra
                else:
                    target = self.gprs[rd_rs1]
                self.pc = mask64(target & ~1)
                return 0
            # else:
            if is_jalr:
                # C.ADD (bit12=1, rs2≠0): rd += rs2
                result = self.gprs[rd_rs1] + self.gprs[rs2]
                self.gprs[rd_rs1] = mask64(result)
            else:
                # C.MV (bit12=0, rs2≠0): rd = rs2
                self.gprs[rd_rs1] = self.gprs[rs2]
            return 2

        # C.FSDSP: mem[sp + uimm] = fpr[rs2] (RV64DC)
        if funct3 == 0b101:
            if not self._fp_enabled():
                raise ValueError("C.FSDSP: FP disabled (mstatus.FS=Off)")
            uimm = cf.imm2
            addr = mask64((self.gprs[2] + uimm))
            v = self._fpr_bits[rs2]
            data = bytes([(v >> (8 * i)) & 0xFF for i in range(8)])
            saved_pc = self.pc
            mem_write(self, addr, data)
            if self.pc != saved_pc:
                return 0
            self._fp_mark_dirty()
            return 2

        # C.SWSP: uimm = cf.imm2 (4-byte aligned)
        if funct3 == 0b110:
            uimm = cf.imm2
            addr = mask64((self.gprs[2] + uimm))
            v = mask32(self.gprs[rs2])
            data = bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF])
            saved_pc = self.pc
            mem_write(self, addr, data)
            if self.pc != saved_pc:
                return 0
            return 2

        # C.SDSP: uimm = cf.imm2 (8-byte aligned)
        if funct3 == 0b111:
            uimm = cf.imm2
            addr = mask64((self.gprs[2] + uimm))
            v = self.gprs[rs2]
            data = bytes([(v >> (8 * i)) & 0xFF for i in range(8)])
            saved_pc = self.pc
            mem_write(self, addr, data)
            if self.pc != saved_pc:
                return 0
            return 2

        raise NotImplementedError(f"C2 funct3={funct3:#05b} (reserved)")

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
        # 预解码全部字段 — 避免 C0/C1/C2 处理器中重复的位提取
        self._cf = _native_decode_compressed(instr)
        op = self._cf.quadrant
        if op == 0:
            return self._handle_compressed_c0(instr)
        elif op == 1:
            return self._handle_compressed_c1(instr)
        elif op == 2:
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
        # 预解码全部字段 — 一次 FFI 调用替代逐个 parse_* Python 调用
        f = _native_decode_fields(instr)
        self._f = f

        # 16-bit 压缩指令 — 非法编码触发 IllInstr 陷态
        if f.is_compressed:
            try:
                return self.handle_compressed(instr & 0xFFFF)
            except MemoryAccessFault:
                return 0
            except (ValueError, NotImplementedError, CsrAccessError):
                deliver_trap(self, TrapType.IllInstr, tval=instr, is_interrupt=False)
                return 0

        # 32-bit 标准指令 — 所有未识别的编码一律触发非法指令陷态
        opcode = f.opcode

        method_name = self._DISPATCH.get(opcode)
        if method_name is None:
            deliver_trap(self, TrapType.IllInstr, tval=instr, is_interrupt=False)
            return 0

        try:
            handler = getattr(self, method_name)
            return handler(instr)
        except MemoryAccessFault:
            return 0
        except (ValueError, NotImplementedError, CsrAccessError):
            # 操作码合法但编码字段无效 (如非法 funct3/funct12/nzuimm=0 等)
            deliver_trap(self, TrapType.IllInstr, tval=instr, is_interrupt=False)
            return 0


"""
TODO: 也该考虑一下
执行状态暂存的事情了
"""
