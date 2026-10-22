#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
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
    decode_compressed,
    decode_fields,
    exec_alu_op,
    exec_op32,
    exec_op_imm,
    exec_op_imm32,
    fp_exec_fma,
    fp_exec_op,
    native_available,
)
from pyremu.core.hart import HartGroup, HartWithRegs
from pyremu.core.mem_check_aux import (
    mem_read,
    mem_write,
    MemoryAccessError,
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
from pyremu.utils.disassem import (
    AMOF5_MAP,
    AmoFunct5,
    AMOW_MAP,
    AmoWidth,
    BrFn3,
    BrFn3_MAP,
    LdFn3,
    LdFn3_MAP,
    StFn3,
    StFn3_MAP,
)
from pyremu.utils.mask import (
    mask16,
    MASK32,
    mask32,
    MASK64,
    mask64,
    rol32,
    rol64,
    ror32,
    ror64,
    sext,
    sext8,
    sext16,
    sext32,
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


# ============================================================
#  Zbb (Basic bit-manipulation) 辅助函数
# ============================================================


def _zbb_imm(funct3: int, v: int, shamt: int) -> int:
    """Zbb OP-IMM 分发 (funct7=0x30, funct3=001 或 101).

    funct3=001 时, shamt 编码子指令 (imm[4:0]):
      0=clz, 1=ctz, 2=cpop, 4=sext.b, 5=sext.h
    funct3=101 时, shamt 为 rori 的旋转量.
    """
    if funct3 == 0b001:
        if shamt == 0:  # clz
            return 64 - mask64(v).bit_length() if v else 64
        if shamt == 1:  # ctz
            return (v & -v).bit_length() - 1 if v else 64
        if shamt == 2:  # cpop
            return bin(mask64(v)).count("1")
        if shamt == 4:  # sext.b
            byte = v & 0xFF
            return byte | (0xFFFF_FFFF_FFFF_FF00 if byte & 0x80 else 0)
        if shamt == 5:  # sext.h
            hw = v & 0xFFFF
            return hw | (0xFFFF_FFFF_FFFF_0000 if hw & 0x8000 else 0)
    elif funct3 == 0b101:
        # rori: rotate right immediate (64-bit)
        return ror64(v, shamt)
    raise ValueError(f"invalid Zbb funct3={funct3:#b} shamt={shamt}")


def _zbb_imm32(funct3: int, v: int, shamt: int) -> int:
    """Zbb OP-IMM-32 分发 (funct7=0x30).

    funct3=001: shamt 编码子指令 — 0=clzw, 1=ctzw, 2=cpopw
    funct3=101: roriw (32-bit rotate right immediate, 结果 sign-extend)
    """
    v32 = mask32(v)
    if funct3 == 0b001:
        if shamt == 0:  # clzw
            r = 32 - v32.bit_length() if v32 else 32
            return r
        if shamt == 1:  # ctzw
            r = (v32 & -v32).bit_length() - 1 if v32 else 32
            return r
        if shamt == 2:  # cpopw
            return bin(v32).count("1")
    elif funct3 == 0b101:
        # roriw: rotate right 32-bit by immediate
        return sext32(ror32(v32, shamt))
    raise ValueError(f"invalid Zbb imm32 funct3={funct3:#b} shamt={shamt}")


def _zbb_orc_b(v: int) -> int:
    """orc.b: 对每个字节, 若任一 bit 为 1 则该字节全 1, 否则全 0."""
    result = 0
    for i in range(8):
        byte = (v >> (i * 8)) & 0xFF
        if byte:
            result |= 0xFF << (i * 8)
    return result


def _zbb_rev8(v: int) -> int:
    """rev8.d (RV64): 字节序反转."""
    result = 0
    for i in range(8):
        result |= ((mask64(v) >> (i * 8)) & 0xFF) << ((7 - i) * 8)
    return result


# Zbb R-type 子指令表: 键为 (funct7, funct3)
_ZBB_ALU_OPS = {
    (0x20, 0b111): lambda a, b: mask64(a & ~b),  # andn
    (0x20, 0b110): lambda a, b: mask64(a | ~b),  # orn
    (0x20, 0b100): lambda a, b: mask64(~(a ^ b)),  # xnor
    (0x05, 0b100): lambda a, b: mask64(min(_sint64(a).value, _sint64(b).value)),  # min
    (0x05, 0b110): lambda a, b: mask64(max(_sint64(a).value, _sint64(b).value)),  # max
    (0x05, 0b101): lambda a, b: min(mask64(a), mask64(b)),  # minu
    (0x05, 0b111): lambda a, b: max(mask64(a), mask64(b)),  # maxu
    (0x30, 0b001): lambda a, b: rol64(a, _uint64(b).value),  # rol
    (0x30, 0b101): lambda a, b: ror64(a, _uint64(b).value),  # ror
}


def _zbb_alu(
    funct3: int,
    funct7: int,
    v1: int,
    v2: int,
) -> int:
    """Zbb R-type 分发 (funct7 ∈ {0x05, 0x20, 0x30})."""
    op = _ZBB_ALU_OPS.get((funct7, funct3))
    if op is None:
        raise ValueError(f"invalid Zbb funct3={funct3:#b} funct7={funct7:#x}")
    return op(v1, v2)


def _zbb_alu32(
    funct3: int,
    funct7: int,
    v1: int,
    v2: int,
) -> int:
    """Zbb OP-32 R-type 分发 (funct7=0x30): rolw, rorw.

    结果 sign-extend 到 64-bit.
    """
    s = _uint64(v2).value
    if funct3 == 0b001:  # rolw
        return sext32(rol32(v1, s))
    if funct3 == 0b101:  # rorw
        return sext32(ror32(v1, s))
    raise ValueError(f"invalid Zbb op32 funct3={funct3:#b} funct7={funct7:#x}")


# ============================================================
#  R 型 ALU 的纯 Python 计算, 按 funct3 分组
# ============================================================


def _alu_py_000(funct7: int, v1: int, v2: int) -> int:
    """funct3=000: add / mul / sub."""
    if funct7 == 0:
        return mask64(v1 + v2)
    if funct7 == 1:
        return mask64(v1 * v2)
    if funct7 == 0x20:
        return mask64(v1 - v2)
    raise ValueError(f"invalid funct7={funct7:#x} for funct3=000")


def _alu_py_001(funct7: int, v1: int, v2: int) -> int:
    """funct3=001: sll / mulh / rol."""
    if funct7 == 0:
        return mask64(v1 << (v2 & 0x3F))
    if funct7 == 1:
        s1 = _sint64(v1).value
        s2 = _sint64(v2).value
        return mask64(_sint64((s1 * s2) >> 64).value)
    if funct7 == 0x30:
        return _zbb_alu(0b001, funct7, v1, v2)
    raise ValueError(f"invalid funct7={funct7:#x} for funct3=001")


def _alu_py_010(funct7: int, v1: int, v2: int) -> int:
    """funct3=010: slt / mulhsu."""
    if funct7 == 0:
        return 1 if _sint64(v1).value < _sint64(v2).value else 0
    if funct7 == 1:
        s1 = _sint64(v1).value
        u2 = _uint64(v2).value
        return mask64(_sint64((s1 * u2) >> 64).value)
    raise ValueError(f"invalid funct7={funct7:#x} for funct3=010")


def _alu_py_011(funct7: int, v1: int, v2: int) -> int:
    """funct3=011: sltu / mulhu."""
    if funct7 == 0:
        return 1 if _uint64(v1).value < _uint64(v2).value else 0
    if funct7 == 1:
        u1 = _uint64(v1).value
        u2 = _uint64(v2).value
        return mask64((u1 * u2) >> 64)
    raise ValueError(f"invalid funct7={funct7:#x} for funct3=011")


def _alu_py_100(funct7: int, v1: int, v2: int) -> int:
    """funct3=100: xor / div / xnor / min."""
    if funct7 == 0:
        return v1 ^ v2
    if funct7 == 1:
        return mask64(_trunc_div(_sint64(v1).value, _sint64(v2).value))
    return _zbb_alu(0b100, funct7, v1, v2)


def _alu_py_101(funct7: int, v1: int, v2: int) -> int:
    """funct3=101: srl / sra / divu / ror."""
    if funct7 == 0:
        return mask64(_uint64(v1).value >> (v2 & 0x3F))
    if funct7 == 1:
        return mask64(_trunc_div(_uint64(v1).value, _uint64(v2).value))
    if funct7 == 0x20:
        return mask64(_sint64(_sint64(v1).value >> (v2 & 0x3F)).value)
    if funct7 == 0x30:
        return _zbb_alu(0b101, funct7, v1, v2)
    raise ValueError(f"invalid funct7={funct7:#x} for funct3=101")


def _alu_py_110(funct7: int, v1: int, v2: int) -> int:
    """funct3=110: or / rem / orn / max."""
    if funct7 == 0:
        return v1 | v2
    if funct7 == 1:
        return mask64(_trunc_rem(_sint64(v1).value, _sint64(v2).value))
    return _zbb_alu(0b110, funct7, v1, v2)


def _alu_py_111(funct7: int, v1: int, v2: int) -> int:
    """funct3=111: and / remu / andn / maxu."""
    if funct7 == 0:
        return v1 & v2
    if funct7 == 1:
        return mask64(_trunc_rem(_uint64(v1).value, _uint64(v2).value))
    return _zbb_alu(0b111, funct7, v1, v2)


_ALU_PY_OPS = {
    0b000: _alu_py_000,
    0b001: _alu_py_001,
    0b010: _alu_py_010,
    0b011: _alu_py_011,
    0b100: _alu_py_100,
    0b101: _alu_py_101,
    0b110: _alu_py_110,
    0b111: _alu_py_111,
}


def _exec_alu_via_native(fields, v1: int, v2: int) -> int:
    """经加速执行引擎计算 R 型 ALU 指令, Zbb 编码交由 _zbb_alu 计算."""
    r = exec_alu_op(fields.func3, fields.func7, v1, v2)
    if not r.trap:
        return r.value
    # funct7 属于 Zbb 扩展的 {0x05, 0x20, 0x30}: 加速执行引擎的 alu 辅助函数
    # 对这些编码返回无效编码, 故在此交由 _zbb_alu 计算
    if fields.func7 in (0x05, 0x20, 0x30):
        return _zbb_alu(fields.func3, fields.func7, v1, v2)
    raise ValueError(f"invalid funct3={fields.func3:#b} funct7={fields.func7:#x}")


# ============================================================
#  I 型立即数 ALU 的纯 Python 计算, 按 funct3 分组
#
#  各函数的 funct6 为 funct7 右移一位的结果; shamt 为立即数的低 6 位.
# ============================================================


def _op_imm_py_000(funct6: int, v1: int, imm: int, shamt: int) -> int:
    """funct3=000: addi."""
    return mask64(v1 + imm)


def _op_imm_py_001(funct6: int, v1: int, imm: int, shamt: int) -> int:
    """funct3=001: slli 与 Zbb 的 clz/ctz/cpop/sext.b/sext.h."""
    if funct6 == 0:
        return mask64(v1 << shamt)
    if funct6 == 0x18:
        # Zbb: 立即数的低 5 位选择子指令
        return _zbb_imm(0b001, v1, shamt)
    raise ValueError(f"invalid funct6={funct6:#x} for SLLI")


def _op_imm_py_010(funct6: int, v1: int, imm: int, shamt: int) -> int:
    """funct3=010: slti."""
    return 1 if _sint64(v1).value < imm else 0


def _op_imm_py_011(funct6: int, v1: int, imm: int, shamt: int) -> int:
    """funct3=011: sltiu."""
    return 1 if _uint64(v1).value < _uint64(imm).value else 0


def _op_imm_py_100(funct6: int, v1: int, imm: int, shamt: int) -> int:
    """funct3=100: xori."""
    return v1 ^ imm


def _op_imm_py_101(funct6: int, v1: int, imm: int, shamt: int) -> int:
    """funct3=101: srli/srai 与 Zbb 的 rori/orc.b/rev8."""
    if funct6 == 0:
        return mask64(_uint64(v1).value >> shamt)
    if funct6 == 0x10:
        return mask64(_sint64(_sint64(v1).value >> shamt).value)
    if funct6 == 0x18:
        return _zbb_imm(0b101, v1, shamt)
    if funct6 == 0x0A:
        return _zbb_orc_b(v1)
    if funct6 == 0x1A:
        return _zbb_rev8(v1)
    raise ValueError(f"invalid funct6={funct6:#x} for SRLI/SRAI")


def _op_imm_py_110(funct6: int, v1: int, imm: int, shamt: int) -> int:
    """funct3=110: ori."""
    return v1 | imm


def _op_imm_py_111(funct6: int, v1: int, imm: int, shamt: int) -> int:
    """funct3=111: andi."""
    return v1 & imm


_OP_IMM_PY_OPS = {
    0b000: _op_imm_py_000,
    0b001: _op_imm_py_001,
    0b010: _op_imm_py_010,
    0b011: _op_imm_py_011,
    0b100: _op_imm_py_100,
    0b101: _op_imm_py_101,
    0b110: _op_imm_py_110,
    0b111: _op_imm_py_111,
}


def _exec_op_imm_via_native(fields, v1: int) -> int:
    """经加速执行引擎计算 I 型立即数 ALU 指令, Zbb 编码交由 _zbb_imm 计算."""
    r = exec_op_imm(fields.func3, fields.func7, v1, fields.imm12_se)
    if not r.trap:
        return r.value
    # funct7 属于 Zbb 扩展的 0x30: 加速执行引擎的 alu 辅助函数对该编码返回
    # 无效编码, 故在此交由 _zbb_imm 计算
    if fields.func7 != 0x30:
        raise ValueError(f"invalid funct3={fields.func3:#b} funct7={fields.func7:#x}")
    return _zbb_imm(fields.func3, v1, fields.imm12_se & 0x3F)


# ============================================================
#  RV64 32 位字运算的纯 Python 计算, 按 funct3 分组
#
#  各函数收到的 v1 与 v2 已截断为 32 位.
# ============================================================


def _op32_py_000(funct7: int, v1: int, v2: int) -> int:
    """funct3=000: addw / mulw / subw."""
    if funct7 == 0:
        return sext32(mask32(v1 + v2))
    if funct7 == 1:
        return sext32(mask32(v1 * v2))
    if funct7 == 0x20:
        return sext32(mask32(v1 - v2))
    raise ValueError(f"invalid funct7={funct7:#x} for op32 funct3=000")


def _op32_py_001(funct7: int, v1: int, v2: int) -> int:
    """funct3=001: sllw / mulw 的高 32 位 / rolw."""
    if funct7 == 0:
        return sext32(mask32(v1 << (v2 & 0x1F)))
    if funct7 == 1:
        s1 = _sint64(sext32(v1)).value
        s2 = _sint64(sext32(v2)).value
        return mask64(_sint64((s1 * s2) >> 32).value)
    if funct7 == 0x30:
        return _zbb_alu32(0b001, funct7, v1, v2)
    raise ValueError(f"invalid funct7={funct7:#x} for op32 funct3=001")


def _op32_py_010(funct7: int, v1: int, v2: int) -> int:
    """funct3=010: sltw / mulhsuw."""
    if funct7 == 0:
        return 1 if _sint64(sext32(v1)).value < _sint64(sext32(v2)).value else 0
    if funct7 == 1:
        s1 = _sint64(sext32(v1)).value
        u2 = _uint64(v2).value
        return mask64(_sint64((s1 * u2) >> 32).value)
    raise ValueError(f"invalid funct7={funct7:#x} for op32 funct3=010")


def _op32_py_011(funct7: int, v1: int, v2: int) -> int:
    """funct3=011: sltuw / mulhuw."""
    if funct7 == 0:
        return 1 if v1 < v2 else 0
    if funct7 == 1:
        return mask64((v1 * v2) >> 32)
    raise ValueError(f"invalid funct7={funct7:#x} for op32 funct3=011")


def _op32_py_100(funct7: int, v1: int, v2: int) -> int:
    """funct3=100: xorw / divw."""
    if funct7 == 0:
        return sext32(mask32(v1 ^ v2))
    if funct7 == 1:
        return sext32(mask32(_trunc_div(_sint64(sext32(v1)).value, _sint64(sext32(v2)).value)))
    raise ValueError(f"invalid funct7={funct7:#x} for op32 funct3=100")


def _op32_py_101(funct7: int, v1: int, v2: int) -> int:
    """funct3=101: srlw / sraw / divuw / rorw."""
    if funct7 == 0:
        return sext32(mask32(v1 >> (v2 & 0x1F)))
    if funct7 == 1:
        return sext32(mask32(_trunc_div(v1, v2)))
    if funct7 == 0x20:
        return sext32(mask32(_sint64(sext32(v1) >> (v2 & 0x1F)).value))
    if funct7 == 0x30:
        return _zbb_alu32(0b101, funct7, v1, v2)
    raise ValueError(f"invalid funct7={funct7:#x} for op32 funct3=101")


def _op32_py_110(funct7: int, v1: int, v2: int) -> int:
    """funct3=110: orw / remw."""
    if funct7 == 0:
        return sext32(mask32(v1 | v2))
    if funct7 == 1:
        return sext32(mask32(_trunc_rem(_sint64(sext32(v1)).value, _sint64(sext32(v2)).value)))
    raise ValueError(f"invalid funct7={funct7:#x} for op32 funct3=110")


def _op32_py_111(funct7: int, v1: int, v2: int) -> int:
    """funct3=111: andw / remuw."""
    if funct7 == 0:
        return sext32(mask32(v1 & v2))
    if funct7 == 1:
        return sext32(mask32(_trunc_rem(v1, v2)))
    raise ValueError(f"invalid funct7={funct7:#x} for op32 funct3=111")


_OP32_PY_OPS = {
    0b000: _op32_py_000,
    0b001: _op32_py_001,
    0b010: _op32_py_010,
    0b011: _op32_py_011,
    0b100: _op32_py_100,
    0b101: _op32_py_101,
    0b110: _op32_py_110,
    0b111: _op32_py_111,
}


def _exec_op32_via_native(fields, v1: int, v2: int) -> int:
    """经加速执行引擎计算 RV64 32 位字运算指令, Zbb 编码交由 _zbb_alu32 计算."""
    r = exec_op32(fields.func3, fields.func7, v1, v2)
    if not r.trap:
        return r.value
    # funct7 属于 Zbb 扩展的 0x30 且 funct3 为 001 或 101: 加速执行引擎的
    # alu 辅助函数对这些编码返回无效编码, 故在此交由 _zbb_alu32 计算
    if fields.func7 == 0x30 and fields.func3 in (0b001, 0b101):
        return _zbb_alu32(fields.func3, fields.func7, v1, v2)
    raise ValueError(f"invalid funct3={fields.func3:#b} funct7={fields.func7:#x}")


# ============================================================
#  I 型 32 位字立即数运算的纯 Python 计算, 按 funct3 分组
# ============================================================


def _op_imm32_py_000(funct7: int, v1: int, imm: int, shamt: int) -> int:
    """funct3=000: addiw."""
    return sext32(mask32(v1 + imm))


def _op_imm32_py_001(funct7: int, v1: int, imm: int, shamt: int) -> int:
    """funct3=001: slliw 与 Zbb 的 clzw/ctzw/cpopw."""
    if funct7 == 0:
        return sext32(mask32(mask32(v1) << shamt))
    if funct7 == 0x30:
        return _zbb_imm32(0b001, v1, shamt)
    raise ValueError(f"invalid funct7={funct7:#x} for SLLIW")


def _op_imm32_py_101(funct7: int, v1: int, imm: int, shamt: int) -> int:
    """funct3=101: srliw/sraiw 与 Zbb 的 roriw."""
    if funct7 == 0:
        return sext32(mask32(mask32(v1) >> shamt))
    if funct7 == 0x20:
        return sext32(mask32(_sint64(sext32(mask32(v1)) >> shamt).value))
    if funct7 == 0x30:
        return _zbb_imm32(0b101, v1, shamt)
    raise ValueError(f"invalid funct7={funct7:#x} for SRLIW/SRAIW")


_OP_IMM32_PY_OPS = {
    0b000: _op_imm32_py_000,
    0b001: _op_imm32_py_001,
    0b101: _op_imm32_py_101,
}


def _op_imm32_py(func3: int, funct7: int, v1: int, imm: int, shamt: int) -> int:
    """按 funct3 分发 I 型 32 位字立即数运算."""
    op = _OP_IMM32_PY_OPS.get(func3)
    if op is None:
        raise ValueError(f"invalid funct3={func3:#x} for op_imm32")
    return op(funct7, v1, imm, shamt)


def _exec_op_imm32_via_native(fields, v1: int) -> int:
    """经加速执行引擎计算 I 型 32 位字立即数运算, Zbb 编码交由 _zbb_imm32 计算."""
    r = exec_op_imm32(fields.func3, fields.func7, v1, fields.imm12_se)
    if not r.trap:
        return r.value
    # funct7 属于 Zbb 扩展的 0x30: 加速执行引擎的 alu 辅助函数对该编码返回
    # 无效编码, 故在此交由 _zbb_imm32 计算
    if fields.func7 != 0x30:
        raise ValueError(f"invalid funct3={fields.func3:#b} funct7={fields.func7:#x}")
    return _zbb_imm32(fields.func3, v1, fields.imm12_se & 0x1F)


_LD_READ_SIZE = {
    LdFn3.lb: 1,
    LdFn3.lbu: 1,
    LdFn3.lh: 2,
    LdFn3.lhu: 2,
    LdFn3.lw: 4,
    LdFn3.lwu: 4,
    LdFn3.ld: 8,
}

_LD_SIGN_EXTEND = {
    LdFn3.lb: sext8,
    LdFn3.lh: sext16,
    LdFn3.lw: sext32,
}


def _load_value(kind: LdFn3, mem: bytes) -> int:
    """按加载指令的宽度解码内存字节, 有符号形式做符号扩展."""
    raw = int.from_bytes(mem, "little")
    extend = _LD_SIGN_EXTEND.get(kind)
    return raw if extend is None else extend(raw)


def _amo_width(width: AmoWidth) -> tuple[int, int]:
    """返回该 AMO 宽度的字节数与位掩码."""
    return (8, MASK64) if width == AmoWidth.D else (4, MASK32)


def _amo_signed(val: int, width: AmoWidth) -> int:
    """按 AMO 宽度做符号扩展后取有符号值."""
    return _sint64(sext(val, 64 if width == AmoWidth.D else 32)).value


def _amo_result(op: AmoFunct5, mem_val: int, op_val: int, width: AmoWidth, mask: int) -> int:
    """按 AMO 子操作计算写回内存的新值."""
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
        result = min(op_val, mem_val, key=lambda v: _amo_signed(v, width))
    elif op == AmoFunct5.MAX:
        result = max(op_val, mem_val, key=lambda v: _amo_signed(v, width))
    elif op == AmoFunct5.MINU:
        result = op_val if op_val < mem_val else mem_val
    elif op == AmoFunct5.MAXU:
        result = op_val if op_val > mem_val else mem_val
    else:
        raise ValueError(f"unhandled AMO op: {op}")
    return result & mask


def _flush_tlbs(origin, vpn: int | None, all_harts: HartGroup | None) -> None:
    """SFENCE.VMA 辅助函数: 把刷新广播到全部 hart.

    origin 为发起 SFENCE.VMA 的 hart, 本身也参与刷新; vpn 为 None 时刷新全部条目,
    否则为 VA>>12 的页号; all_harts 为全部 hart 的弱引用视图, 由调用方传入以避免
    每 hart 持有重复引用. 本次刷新另计入 TLB 刷新计数, 使加速执行引擎内既有的
    条目作废.
    """
    origin.note_tlb_flush()
    if vpn is None:
        origin.itlb.flush_all()
        origin.dtlb.flush_all()
        for h in all_harts or []:
            if h is origin:
                continue
            h.itlb.flush_all()
            h.dtlb.flush_all()
        return
    origin.itlb.flush(vpn)
    origin.dtlb.flush(vpn)
    for h in all_harts or []:
        if h is origin:
            continue
        h.itlb.flush(vpn)
        h.dtlb.flush(vpn)


# ---------------------------------------------------------------
#  系统指令辅助
# ---------------------------------------------------------------


def _sys_ecall(hart: HartWithRegs, instr: int) -> None:
    """ECALL."""
    trap_ecall(hart)


def _sys_ebreak(hart: HartWithRegs, instr: int) -> None:
    """EBREAK."""
    trap_ebreak(hart)


def _sys_mret(hart: HartWithRegs, instr: int) -> None:
    """MRET."""
    trap_mret(hart)


def _sys_sret(hart: HartWithRegs, instr: int) -> None:
    """SRET."""
    trap_sret(hart)


def _sys_sfence_vma(hart: HartWithRegs, instr: int) -> None:
    """SFENCE.VMA: rs1=x0 时刷新全部 TLB, 否则仅刷新该虚地址对应的条目.

    刷新同步到全部 hart. RISC-V 只要求刷新本 hart, 扩大范围不影响正确性,
    且省去核间中断.
    """
    rs1 = hart._f.rs1
    vpn = None if rs1 == 0 else hart.gprs[rs1] >> 12
    _flush_tlbs(hart, vpn, hart._all_harts)


def _sys_mfence_did(hart: HartWithRegs, instr: int) -> None:
    """MFENCE.DID: 按内存域刷新全部 hart 的 TLB 与共享 L2 缓存."""
    mdid_val = hart.mdid_val
    for h in hart._all_harts or [hart]:
        h.itlb.flush_by_mdid(mdid_val)
        h.dtlb.flush_by_mdid(mdid_val)
    # 加速执行引擎按页号与内存域查询条目, 无按域刷新的入口, 故整表作废.
    hart.note_tlb_flush()
    # 刷新共享 L2 缓存中匹配的条目
    if hart._bus is not None:
        l2 = getattr(hart._bus, "_l2", None)
        if l2 is not None:
            l2.flush_by_mdid(mdid_val)


_SYS_PRIV_OPS = {
    0x000: _sys_ecall,
    0x001: _sys_ebreak,
    0x302: _sys_mret,
    0x102: _sys_sret,
    0x105: handle_wfi,
    0x120: _sys_sfence_vma,
    0x5A0: _sys_mfence_did,
}

# CSR 指令: funct3 -> (新值算法, 是否按源操作数是否为零决定写入)
_CSR_OPS = {
    0b001: (lambda old, src: src, False),  # CSRRW
    0b010: (lambda old, src: old | src, True),  # CSRRS
    0b011: (lambda old, src: old & ~src, True),  # CSRRC
    0b101: (lambda old, src: src, False),  # CSRRWI
    0b110: (lambda old, src: old | src, True),  # CSRRSI
    0b111: (lambda old, src: old & ~src, True),  # CSRRCI
}

# ---------------------------------------------------------------
#  压缩指令辅助
# ---------------------------------------------------------------


def _c0_addi4spn(hart: HartWithRegs, cf, instr: int) -> int:
    """C.ADDI4SPN: rd' = sp + nzuimm, nzuimm 为 0 属保留编码."""
    if cf.imm == 0:
        deliver_trap(hart, TrapType.IllInstr, tval=instr, is_interrupt=False)
        return 0
    hart.gprs[cf.rd] = mask64(hart.gprs[2] + cf.imm)
    return 2


def _c0_mem_addr(hart: HartWithRegs, cf) -> int:
    """C0 访存地址: rs1' + uimm."""
    return mask64(hart.gprs[cf.rs1p] + cf.imm)


def _c0_fld(hart: HartWithRegs, cf) -> int:
    """C.FLD: fpr[rd'] = mem[rs1' + uimm]."""
    if not hart._fp_enabled():
        raise ValueError("C.FLD: FP disabled (mstatus.FS=Off)")
    addr = _c0_mem_addr(hart, cf)
    saved_pc = hart.pc
    mem = mem_read(hart, addr, 8)
    if hart.pc != saved_pc:
        return 0
    hart._fpr_bits[cf.rd] = int.from_bytes(mem, "little")
    hart._fp_mark_dirty()
    return 2


def _c0_lw(hart: HartWithRegs, cf) -> int:
    """C.LW: rd' = sext32(mem[rs1' + uimm])."""
    addr = _c0_mem_addr(hart, cf)
    saved_pc = hart.pc
    mem = mem_read(hart, addr, 4)
    if hart.pc != saved_pc:
        return 0
    hart.gprs[cf.rd] = sext32(int.from_bytes(mem, "little"))
    return 2


def _c0_ld(hart: HartWithRegs, cf) -> int:
    """C.LD: rd' = mem[rs1' + uimm]."""
    addr = _c0_mem_addr(hart, cf)
    saved_pc = hart.pc
    mem = mem_read(hart, addr, 8)
    if hart.pc != saved_pc:
        return 0
    hart.gprs[cf.rd] = int.from_bytes(mem, "little")
    return 2


def _c0_sw(hart: HartWithRegs, cf) -> int:
    """C.SW: mem[rs1' + uimm] = rs2'[31:0]."""
    addr = _c0_mem_addr(hart, cf)
    data = mask32(hart.gprs[cf.rdp]).to_bytes(4, "little")
    saved_pc = hart.pc
    mem_write(hart, addr, data)
    return 0 if hart.pc != saved_pc else 2


def _c0_sd(hart: HartWithRegs, cf) -> int:
    """C.SD: mem[rs1' + uimm] = rs2'."""
    addr = _c0_mem_addr(hart, cf)
    data = hart.gprs[cf.rdp].to_bytes(8, "little")
    saved_pc = hart.pc
    mem_write(hart, addr, data)
    return 0 if hart.pc != saved_pc else 2


def _c0_fsd(hart: HartWithRegs, cf) -> int:
    """C.FSD: mem[rs1' + uimm] = fpr[rd']."""
    if not hart._fp_enabled():
        raise ValueError("C.FSD: FP disabled (mstatus.FS=Off)")
    addr = _c0_mem_addr(hart, cf)
    data = hart._fpr_bits[cf.rd].to_bytes(8, "little")
    saved_pc = hart.pc
    mem_write(hart, addr, data)
    if hart.pc != saved_pc:
        return 0
    hart._fp_mark_dirty()
    return 2


_C0_OPS = {
    0b001: _c0_fld,
    0b010: _c0_lw,
    0b011: _c0_ld,
    0b101: _c0_fsd,
    0b110: _c0_sw,
    0b111: _c0_sd,
}


def _c1_shamt(cf) -> int:
    """C.SRLI/C.SRAI/C.ANDI 的 6 位立即数, 由 bit12 与 bits[6:2] 组成."""
    return (cf.bit12 << 5) | (cf.rs2 & 0x1F)


def _c1_addi(hart: HartWithRegs, instr: int) -> int:
    """C.ADDI: rd += imm."""
    cf = hart._cf
    hart.gprs[cf.rd] = mask64(hart.gprs[cf.rd] + cf.imm)
    return 2


def _c1_addiw(hart: HartWithRegs, instr: int) -> int:
    """C.ADDIW: rd = sext32(rd + imm)."""
    cf = hart._cf
    hart.gprs[cf.rd] = sext32(mask32(hart.gprs[cf.rd] + cf.imm))
    return 2


def _c1_li(hart: HartWithRegs, instr: int) -> int:
    """C.LI: rd = imm."""
    cf = hart._cf
    hart.gprs[cf.rd] = mask64(cf.imm)
    return 2


def _c1_lui_or_addi16sp(hart: HartWithRegs, instr: int) -> int:
    """C.LUI (rd≠2) 或 C.ADDI16SP (rd=2), nzuimm 为 0 属保留编码."""
    cf = hart._cf
    if cf.rd == 2:
        if cf.imm == 0:
            raise ValueError("C.ADDI16SP: nzuimm must be non-zero")
        hart.gprs[2] = mask64(hart.gprs[2] + cf.imm)
    else:
        if cf.imm == 0:
            raise ValueError("C.LUI: nzuimm must be non-zero")
        hart.gprs[cf.rd] = mask64(cf.imm << 12)
    return 2


def _c1_j(hart: HartWithRegs, instr: int) -> int:
    """C.J: pc += offset."""
    hart.pc = mask64(hart.pc + hart._cf.imm)
    return 0


def _c1_branch(hart: HartWithRegs, is_eqz: bool) -> int:
    """C.BEQZ 与 C.BNEZ 的公共部分, is_eqz 为真时按 rs1' == 0 取跳转条件."""
    cf = hart._cf
    if (hart.gprs[cf.rs1p] == 0) == is_eqz:
        hart.pc = mask64(hart.pc + cf.imm)
        return 0
    return 2


def _c1_beqz(hart: HartWithRegs, instr: int) -> int:
    """C.BEQZ: rs1' == 0 时跳转."""
    return _c1_branch(hart, True)


def _c1_bnez(hart: HartWithRegs, instr: int) -> int:
    """C.BNEZ: rs1' != 0 时跳转."""
    return _c1_branch(hart, False)


_C1_ALU_REGS = {
    0b00: lambda a, b: mask64(a - b),  # C.SUB
    0b01: lambda a, b: a ^ b,  # C.XOR
    0b10: lambda a, b: a | b,  # C.OR
    0b11: lambda a, b: a & b,  # C.AND
}

_C1_OPS = {
    0b000: _c1_addi,
    0b001: _c1_addiw,
    0b010: _c1_li,
    0b011: _c1_lui_or_addi16sp,
    0b100: lambda hart, instr: hart._handle_compressed_c1_alu(),
    0b101: _c1_j,
    0b110: _c1_beqz,
    0b111: _c1_bnez,
}


def _sp_addr(hart: HartWithRegs, uimm: int) -> int:
    """栈指针相对的访存地址."""
    return mask64(hart.gprs[2] + uimm)


def _c2_slli(hart: HartWithRegs, instr: int) -> int:
    """C.SLLI: rd <<= shamt."""
    cf = hart._cf
    hart.gprs[cf.rd] = mask64(hart.gprs[cf.rd] << cf.imm)
    return 2


def _c2_fldsp(hart: HartWithRegs, instr: int) -> int:
    """C.FLDSP: fpr[rd] = mem[sp + uimm]."""
    if not hart._fp_enabled():
        raise ValueError("C.FLDSP: FP disabled (mstatus.FS=Off)")
    cf = hart._cf
    addr = _sp_addr(hart, cf.imm)
    saved_pc = hart.pc
    mem = mem_read(hart, addr, 8)
    if hart.pc != saved_pc:
        return 0
    hart._fpr_bits[cf.rd] = int.from_bytes(mem, "little")
    hart._fp_mark_dirty()
    return 2


def _c2_lwsp(hart: HartWithRegs, instr: int) -> int:
    """C.LWSP: rd = sext32(mem[sp + uimm])."""
    cf = hart._cf
    if cf.rd == 0:
        raise ValueError("C.LWSP rd=0 (reserved)")
    addr = _sp_addr(hart, cf.imm)
    saved_pc = hart.pc
    mem = mem_read(hart, addr, 4)
    if hart.pc != saved_pc:
        return 0
    hart.gprs[cf.rd] = sext32(int.from_bytes(mem, "little"))
    return 2


def _c2_ldsp(hart: HartWithRegs, instr: int) -> int:
    """C.LDSP: rd = mem[sp + uimm]."""
    cf = hart._cf
    if cf.rd == 0:
        raise ValueError("C.LDSP rd=0 (reserved)")
    addr = _sp_addr(hart, cf.imm)
    saved_pc = hart.pc
    mem = mem_read(hart, addr, 8)
    if hart.pc != saved_pc:
        return 0
    hart.gprs[cf.rd] = int.from_bytes(mem, "little")
    return 2


def _c2_jump_or_mv(hart: HartWithRegs, instr: int) -> int:
    """funct3=100: C.JR / C.JALR / C.MV / C.ADD / C.EBREAK, 由 bit12 与 rs2 区分."""
    cf = hart._cf
    rd_rs1 = cf.rd  # C.JR/C.JALR 的 rs1, C.MV/C.ADD 的 rd
    rs2 = cf.rs2
    if rd_rs1 == 0 and rs2 == 0:
        trap_ebreak(hart)
        return 0
    if rs2 == 0:
        # 先读跳转目标, rd_rs1 可能为 x1 并在随后被写入
        target = hart.gprs[rd_rs1]
        if cf.bit12:
            hart.gprs[1] = mask64(hart.pc + 2)
        hart.pc = mask64(target & ~1)
        return 0
    if cf.bit12:
        hart.gprs[rd_rs1] = mask64(hart.gprs[rd_rs1] + hart.gprs[rs2])
    else:
        hart.gprs[rd_rs1] = hart.gprs[rs2]
    return 2


def _c2_fsdsp(hart: HartWithRegs, instr: int) -> int:
    """C.FSDSP: mem[sp + uimm] = fpr[rs2]."""
    if not hart._fp_enabled():
        raise ValueError("C.FSDSP: FP disabled (mstatus.FS=Off)")
    cf = hart._cf
    addr = _sp_addr(hart, cf.imm2)
    data = hart._fpr_bits[cf.rs2].to_bytes(8, "little")
    saved_pc = hart.pc
    mem_write(hart, addr, data)
    if hart.pc != saved_pc:
        return 0
    hart._fp_mark_dirty()
    return 2


def _c2_swsp(hart: HartWithRegs, instr: int) -> int:
    """C.SWSP: mem[sp + uimm] = rs2[31:0]."""
    cf = hart._cf
    addr = _sp_addr(hart, cf.imm2)
    data = mask32(hart.gprs[cf.rs2]).to_bytes(4, "little")
    saved_pc = hart.pc
    mem_write(hart, addr, data)
    return 0 if hart.pc != saved_pc else 2


def _c2_sdsp(hart: HartWithRegs, instr: int) -> int:
    """C.SDSP: mem[sp + uimm] = rs2."""
    cf = hart._cf
    addr = _sp_addr(hart, cf.imm2)
    data = hart.gprs[cf.rs2].to_bytes(8, "little")
    saved_pc = hart.pc
    mem_write(hart, addr, data)
    return 0 if hart.pc != saved_pc else 2


_C2_OPS = {
    0b000: _c2_slli,
    0b001: _c2_fldsp,
    0b010: _c2_lwsp,
    0b011: _c2_ldsp,
    0b100: _c2_jump_or_mv,
    0b101: _c2_fsdsp,
    0b110: _c2_swsp,
    0b111: _c2_sdsp,
}


class Hart(HartWithRegs):
    # 32-bit 指令分发表: opcode 映射到处理方法名, 索引取值为整数 0 到 127, 分发复杂度为 O(1)
    _DISPATCH: dict[int, str] = {
        0b01100_11: "handle_alu",  # Opc.op
        0b00100_11: "handle_op_imm",  # Opc.op_imm
        0b00110_11: "handle_op_imm32",  # Opc.op_imm32
        0b01110_11: "handle_op32",  # Opc.op32
        0b00000_11: "handle_ld",  # Opc.ld
        0b01000_11: "handle_st",  # Opc.st
        0b11000_11: "handle_br",  # Opc.br
        0b11001_11: "handle_jalr",  # Opc.jalr
        0b11011_11: "_handle_jal",  # Opc.jal
        0b01101_11: "_handle_lui",  # Opc.lui
        0b00101_11: "_handle_auipc",  # Opc.auipc
        0b11100_11: "handle_sys",  # Opc.sys
        0b00011_11: "handle_fence",  # Opc.fence
        0b01011_11: "handle_amo",  # Opc.amo
        0b00001_11: "handle_fp_load",  # Opc.ld_fp (FLW/FLD)
        0b01001_11: "handle_fp_store",  # Opc.stfp (FSW/FSD)
        0b10100_11: "handle_fp_op",  # Opc.fp_op (OP-FP)
        0b10000_11: "handle_fp_fma",  # Opc.fmadd
        0b10001_11: "handle_fp_fma",  # Opc.fmsub
        0b10010_11: "handle_fp_fma",  # Opc.fnmsub
        0b10011_11: "handle_fp_fma",  # Opc.fnmadd
    }

    # ----------------------------------------------------------
    #  R-type ALU (opcode = Opc.op)
    # ----------------------------------------------------------
    def handle_alu(self, instr: int) -> int:
        """执行 R 型 ALU 指令."""
        f = self._f
        v1, v2 = self.gprs[f.rs1], self.gprs[f.rs2]
        if native_available():
            result = _exec_alu_via_native(f, v1, v2)
        else:
            result = _ALU_PY_OPS[f.func3](f.func7, v1, v2)
        if f.rd != 0:
            self.gprs[f.rd] = result
        return 4

    # ----------------------------------------------------------
    #  I-type ALU (opcode = Opc.op_imm)
    # ----------------------------------------------------------
    def handle_op_imm(self, instr: int) -> int:
        """执行 I 型立即数 ALU 指令."""
        f = self._f
        v1 = self.gprs[f.rs1]
        if native_available():
            result = _exec_op_imm_via_native(f, v1)
        else:
            op = _OP_IMM_PY_OPS[f.func3]
            result = op(f.func7 >> 1, v1, f.imm12_se, f.imm12_se & 0x3F)
        if f.rd != 0:
            self.gprs[f.rd] = result
        return 4

    # ----------------------------------------------------------
    #  RV64 32-bit word operations (opcode = Opc.op32)
    # ----------------------------------------------------------
    def handle_op32(self, instr: int):
        """执行 RV64 的 32 位字运算指令, 如 ADDW、SUBW、SLLW 等."""
        f = self._f
        v1, v2 = self.gprs[f.rs1], self.gprs[f.rs2]
        if native_available():
            result = _exec_op32_via_native(f, v1, v2)
        else:
            result = _OP32_PY_OPS[f.func3](f.func7, mask32(v1), mask32(v2))
        if f.rd != 0:
            self.gprs[f.rd] = result
        return 4

    # ----------------------------------------------------------
    #  I-type immediate word operations (opcode = Opc.op_imm32)
    # ----------------------------------------------------------
    def handle_op_imm32(self, instr: int):
        """执行 I 型 32 位字立即数 ALU 指令, 如 ADDIW、SLLIW 等."""
        f = self._f
        v1 = self.gprs[f.rs1]
        if native_available():
            result = _exec_op_imm32_via_native(f, v1)
        else:
            result = _op_imm32_py(f.func3, f.func7, v1, f.imm12_se, f.imm12_se & 0x1F)
        if f.rd != 0:
            self.gprs[f.rd] = result
        return 4

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
        f = BrFn3_MAP.get(fn3)
        if f is None:
            return 4

        if f == BrFn3.beq:
            taken = v1 == v2
        elif f == BrFn3.bne:
            taken = v1 != v2
        elif f == BrFn3.blt:
            taken = _sint64(v1).value < _sint64(v2).value
        elif f == BrFn3.bge:
            taken = _sint64(v1).value >= _sint64(v2).value
        elif f == BrFn3.bltu:
            taken = _uint64(v1).value < _uint64(v2).value
        elif f == BrFn3.bgeu:
            taken = _uint64(v1).value >= _uint64(v2).value

        if taken:
            self.pc = mask64((self.pc + offset))
            return 0
        return 4

    def handle_ld(self, instr: int):
        """执行加载指令, 按 funct3 决定读取宽度与符号扩展方式."""
        fields = self._f
        kind = LdFn3_MAP.get(fields.func3)
        if kind is None:
            raise ValueError(f"invalid funct3={fields.func3:#x} for load")
        addr = mask64((self.gprs[fields.rs1] + fields.imm12_se))
        saved_pc = self.pc
        # 按实际操作宽度读取, 避免对 lb/lbu/lh/lhu/lw/lwu 产生虚假对齐故障
        mem = mem_read(self, addr, _LD_READ_SIZE[kind])
        # 访存故障已经投递并把 PC 改写为陷态入口时, 不覆盖 rd, 使陷态处理
        # 函数看到故障前的寄存器状态
        if self.pc != saved_pc:
            return 0
        if fields.rd != 0:
            self.gprs[fields.rd] = _load_value(kind, mem)
        return 4

    def handle_st(self, instr: int):
        """Execute a store instruction."""
        f = self._f
        fn3 = f.func3
        rs1 = f.rs1
        rs2 = f.rs2
        offset = f.imm_s

        addr = mask64((self.gprs[rs1] + offset))
        val = self.gprs[rs2]

        f = StFn3_MAP.get(fn3)
        if f is None:
            raise ValueError(f"invalid funct3={fn3:#x} for store")

        if f == StFn3.sb:
            data = bytes([val & 0xFF])
        elif f == StFn3.sh:
            data = bytes([val & 0xFF, (val >> 8) & 0xFF])
        elif f == StFn3.sw:
            data = bytes(
                [val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF]
            )
        elif f == StFn3.sd:
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
    #  F/D 浮点指令
    #
    #  计算委托给加速执行引擎的 softfloat, 入口为 fp_exec_op 与 fp_exec_fma,
    #  单精度结果的高 32 位由该引擎填为全 1 以设置为 NaN; Python 侧仅负责寄存器与内存的路由,
    #  并在 FLW 载入 32 位浮点数时做同样的填充.
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
        """OP-FP — 算术、转换、比较、符号、分类、移动指令, 计算委托给加速执行引擎."""
        if not self._fp_enabled():
            raise ValueError("FP disabled (mstatus.FS=Off)")
        f = self._f
        op5 = f.func7 >> 2
        # int->float (0x1A) 与 FMV.*.X (0x1E) 的 rs1 源为 GPR。
        rs1_bits = self.gprs[f.rs1] if op5 in (0x1A, 0x1E) else self._fpr_bits[f.rs1]
        rs2_bits = self._fpr_bits[f.rs2]
        out = fp_exec_op(f.func7, f.func3, f.rs2, (rs1_bits, rs2_bits), self._fp_frm())
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
        """FMADD/FMSUB/FNMSUB/FNMADD — 合并乘加指令, 计算委托给加速执行引擎."""
        if not self._fp_enabled():
            raise ValueError("FP disabled (mstatus.FS=Off)")
        f = self._f
        out = fp_exec_fma(
            f.opcode,
            f.func3,
            f.fmt,
            (self._fpr_bits[f.rs1], self._fpr_bits[f.rs2], self._fpr_bits[f.rs3]),
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
        op = AMOF5_MAP.get(funct5_val)
        width = AMOW_MAP.get(f.func3)
        if op is None or width is None:
            raise ValueError(
                f"invalid AMO encoding: funct5={funct5_val:#07b}, funct3={f.func3:#05b}"
            )
        addr = mask64(self.gprs[f.rs1])
        if op == AmoFunct5.LR:
            return self._amo_lr(f, addr, width)
        if op == AmoFunct5.SC:
            return self._amo_sc(f, addr, width)
        return self._amo_rmw(f, addr, width, op)

    def _amo_lr(self, f, addr: int, width: AmoWidth) -> int:
        """Load-Reserved: 读取内存并设置预留."""
        byte_len, mask = _amo_width(width)
        saved_pc = self.pc
        data_bytes = mem_read(self, addr, byte_len)
        if self.pc != saved_pc:
            return 0
        val = int.from_bytes(data_bytes, "little") & mask
        if f.rd != 0:
            # LR.D 不需要符号扩展; LR.W 由 32 位符号扩展到 64 位
            self.gprs[f.rd] = val if width == AmoWidth.D else sext32(val)
        self.set_reservation(addr, val)
        return 4

    def _amo_sc(self, f, addr: int, width: AmoWidth) -> int:
        """Store-Conditional: 仅预留有效时写入, 成功时 rd 写 0, 失败时写 1."""
        byte_len, mask = _amo_width(width)
        if self.reservation_valid and self.reservation_addr == addr:
            data = (self.gprs[f.rs2] & mask).to_bytes(byte_len, "little")
            saved_pc = self.pc
            mem_write(self, addr, data)
            if self.pc != saved_pc:
                return 0
            if f.rd != 0:
                self.gprs[f.rd] = 0
        elif f.rd != 0:
            self.gprs[f.rd] = 1
        self.clear_reservation()
        return 4

    def _amo_rmw(self, f, addr: int, width: AmoWidth, op: AmoFunct5) -> int:
        """读-改-写: 把运算结果写回内存, 旧值写入 rd."""
        byte_len, mask = _amo_width(width)
        saved_pc = self.pc
        data_bytes = mem_read(self, addr, byte_len)
        if self.pc != saved_pc:
            return 0
        mem_val = int.from_bytes(data_bytes, "little") & mask
        op_val = self.gprs[f.rs2] & mask
        result = _amo_result(op, mem_val, op_val, width, mask)
        saved_pc = self.pc
        mem_write(self, addr, result.to_bytes(byte_len, "little"))
        if self.pc != saved_pc:
            return 0
        if f.rd != 0:
            self.gprs[f.rd] = sext(mem_val, 64) if width == AmoWidth.D else sext32(mem_val)
        return 4

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

    def handle_sys(self, instr: int) -> int:
        """执行系统指令 (特权指令或 CSR 访问).

        返回 0 表示 pc 已被修改 (陷态或返回), 返回 4 表示 pc 正常推进.
        """
        saved_pc = self.pc
        f = self._f
        if f.func3 == 0:
            self._sys_privileged(f, instr)
        else:
            spec = _CSR_OPS.get(f.func3)
            if spec is None:
                raise ValueError(f"invalid funct3={f.func3:#x} for sys")
            self._sys_csr(f, spec)
        return 0 if self.pc != saved_pc else 4

    def _sys_privileged(self, f, instr: int) -> None:
        """执行 funct3=0 的特权指令, 由 funct12 决定具体操作."""
        funct12 = f.func12
        op = _SYS_PRIV_OPS.get(funct12)
        if op is None and 0x121 <= funct12 <= 0x13F:
            # SFENCE.VMA: funct7=0b0001001, 其 rs2 编码在 funct12 的低 5 位
            op = _sys_sfence_vma
        if op is None:
            raise ValueError(f"unknown privileged funct12={funct12:#05x}")
        op(self, instr)

    def _sys_csr(self, f, spec: tuple) -> None:
        """执行 CSR 指令; spec 给出新值算法与写入条件."""
        calc, is_conditional = spec
        csr_addr = f.func12
        is_imm = (f.func3 & 0b100) != 0
        src = f.rs1 if is_imm else self.gprs[f.rs1]
        is_write = (src != 0) if is_conditional else True
        validate_csr(self, csr_addr, is_write=is_write)
        old = self.read_csr(csr_addr)
        if f.rd != 0:
            self.gprs[f.rd] = old
        if is_write:
            self.write_csr(csr_addr, calc(old, src))

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
        """把 3-bit 压缩寄存器号转换为完整寄存器号, 结果范围为 x8 到 x15."""
        return (n & 0x7) + 8

    # -- C0: Quadrant 0 (低 2 位 = 00) --
    # C.ADDI4SPN / C.LW / C.LD / C.SW / C.SD (FLD/FSD 暂未实现)

    def _handle_compressed_c0(self, instr: int) -> int:
        """C0 象限分发: C.ADDI4SPN 之外按 funct3 查表."""
        cf = self._cf
        if cf.funct3 == 0b000:
            return _c0_addi4spn(self, cf, instr)
        handler = _C0_OPS.get(cf.funct3)
        if handler is None:
            raise NotImplementedError(f"C0 funct3={cf.funct3:#05b} (reserved)")
        return handler(self, cf)

    # -- C1: Quadrant 1 (低 2 位 = 01) --
    # C.ADDI / C.ADDIW / C.LI / C.LUI / C.ADDI16SP /
    # C.SRLI/C.SRAI/C.ANDI / C.SUB/C.XOR/C.OR/C.AND /
    # C.J / C.BEQZ / C.BNEZ

    def _handle_compressed_c1(self, instr: int) -> int:
        """C1 象限分发."""
        cf = self._cf
        handler = _C1_OPS.get(cf.funct3)
        if handler is None:
            raise NotImplementedError(f"C1 funct3={cf.funct3:#05b}")
        return handler(self, instr)

    def _handle_compressed_c1_alu(self) -> int:
        """C1 ALU 运算 (funct3=100), 由 sf 选择移位、与运算或寄存器运算.

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
            self.gprs[rd_rs1] = mask64(_uint64(v1).value >> _c1_shamt(cf))
        elif sf == 0b01:
            self.gprs[rd_rs1] = mask64(_sint64(v1).value >> _c1_shamt(cf))
        elif sf == 0b10:
            self.gprs[rd_rs1] = mask64(v1 & sext(_c1_shamt(cf), 6))
        elif sf == 0b11:
            self._c1_alu_register(cf, v1)
        else:
            raise NotImplementedError(f"C1 ALU sub_fn={sf:#03b}")
        return 2

    def _c1_alu_register(self, cf, v1: int) -> None:
        """sf=11: bit12=0 为 64 位运算, bit12=1 为 32 位运算."""
        rd_rs1 = cf.rs1p
        v2 = self.gprs[cf.rdp]
        k = cf.bit65
        if cf.bit12 == 0:
            self.gprs[rd_rs1] = _C1_ALU_REGS[k](v1, v2)
        elif k == 0b00:
            self.gprs[rd_rs1] = sext32(mask32(v1 - v2))
        elif k == 0b01:
            self.gprs[rd_rs1] = sext32(mask32(v1 + v2))
        else:
            raise NotImplementedError(f"C.SUBW/C.ADDW reserved bit[6:5]={k:#03b}")

    # -- C2: Quadrant 2 (低 2 位 = 10) --
    # C.SLLI / C.LWSP / C.LDSP / C.JR/C.MV/C.EBREAK/C.JALR / C.SWSP / C.SDSP

    def _handle_compressed_c2(self, instr: int) -> int:
        """C2 象限分发."""
        cf = self._cf
        handler = _C2_OPS.get(cf.funct3)
        if handler is None:
            raise NotImplementedError(f"C2 funct3={cf.funct3:#05b} (reserved)")
        return handler(self, instr)

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
        self._cf = decode_compressed(instr)
        if self._cf.quadrant == 0:
            return self._handle_compressed_c0(instr)
        elif self._cf.quadrant == 1:
            return self._handle_compressed_c1(instr)
        elif self._cf.quadrant == 2:
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
        f = decode_fields(instr)
        self._f = f

        # 16-bit 压缩指令 — 非法编码触发 IllInstr 陷态
        if f.is_compressed:
            try:
                return self.handle_compressed(mask16(instr))
            except MemoryAccessError:
                pass
            except ValueError, NotImplementedError, CsrAccessError:
                deliver_trap(self, TrapType.IllInstr, tval=instr, is_interrupt=False)
            return 0

        # 32-bit 标准指令 — 所有未识别的编码一律触发非法指令陷态
        opcode = f.opcode

        method_name = self._DISPATCH.get(opcode)
        if method_name is None:
            deliver_trap(self, TrapType.IllInstr, tval=instr, is_interrupt=False)
            return 0

        try:
            return getattr(self, method_name)(instr)
        except MemoryAccessError:
            pass
        except ValueError, NotImplementedError, CsrAccessError:
            # 操作码合法但编码字段无效 (如非法 funct3/funct12/nzuimm=0 等)
            deliver_trap(self, TrapType.IllInstr, tval=instr, is_interrupt=False)
        return 0


"""
TODO: 也该考虑一下
执行状态暂存的事情了
"""
