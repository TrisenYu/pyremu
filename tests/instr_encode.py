#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""RV64 32-bit 指令编码辅助, 供各测试模块共用.

R-type 的两个字段组以打包形式传入: ``funct`` 由 funct7 与 funct3 打包, ``src_regs``
由 rs1 与 rs2 打包. 其余格式按字段逐个传入, 参数次序为 opcode 在前, 目的寄存器其次.
"""

from __future__ import annotations


def pack_funct(funct7: int, funct3: int) -> int:
    """把 funct7 与 funct3 打包为 ``funct7 << 3 | funct3``."""
    return ((funct7 & 0x7F) << 3) | (funct3 & 0x7)


def pack_src_regs(rs1: int, rs2: int) -> int:
    """把 rs1 与 rs2 打包为 ``rs1 << 5 | rs2``."""
    return ((rs1 & 0x1F) << 5) | (rs2 & 0x1F)


def r_type(opcode: int, rd: int, funct: int, src_regs: int) -> int:
    """R-type: funct 为 funct7 左移 3 位再接 funct3, src_regs 为 rs1 左移 5 位再接 rs2."""
    funct7 = funct >> 3
    funct3 = funct & 0x7
    rs1 = src_regs >> 5
    rs2 = src_regs & 0x1F
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def i_type(opcode: int, rd: int, funct3: int, rs1: int, imm12: int) -> int:
    """I-type: 立即数写入 bits[31:20]."""
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def s_type(opcode: int, funct3: int, rs1: int, rs2: int, imm12: int) -> int:
    """S-type: 立即数按 imm[11:5] 与 imm[4:0] 两段写入."""
    return (
        (((imm12 >> 5) & 0x7F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((imm12 & 0x1F) << 7)
        | opcode
    )


def b_type(opcode: int, funct3: int, rs1: int, rs2: int, offset: int) -> int:
    """B-type: 立即数按 imm[12|10:5] 与 imm[4:1|11] 两段写入."""
    return (
        (((offset >> 12) & 0x1) << 31)
        | (((offset >> 5) & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (((offset >> 1) & 0xF) << 8)
        | (((offset >> 11) & 0x1) << 7)
        | opcode
    )


def u_type(opcode: int, rd: int, imm20: int) -> int:
    """U-type: 立即数写入 bits[31:12]."""
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode


def j_type(opcode: int, rd: int, offset: int) -> int:
    """J-type: 立即数按 imm[20|10:1|11|19:12] 写入."""
    return (
        (((offset >> 20) & 0x1) << 31)
        | (((offset >> 1) & 0x3FF) << 21)
        | (((offset >> 11) & 0x1) << 20)
        | (((offset >> 12) & 0xFF) << 12)
        | (rd << 7)
        | opcode
    )
