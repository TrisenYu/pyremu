#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
RISC-V 机器码片段 (shellcode) — 预定义的短指令序列.

每个片段都是一段位置无关的 32-bit 指令字列表, 可被注入到 RAM 中执行,
用于在模拟器内为托管程序准备运行环境 (开辟栈帧、设置中断向量、切换特权级等).

约定:
- 片段为纯 RV64 I 基础指令, 不依赖 C 扩展或浮点
- 片段末尾通常以 ret (jalr zero, ra, 0) 或 fall-through 结束
- 需要 64-bit 立即数的片段使用 PC 相对寻址 + 内联数据池
"""

from dataclasses import dataclass


@dataclass
class AsmSnippet:
    """一段位置无关的 RISC-V 机器码 (类 shellcode)."""

    words: list[int]  # 32-bit 指令字
    desc: str = ""  # 片段用途描述


# ============================================================
#  指令编码辅助
# ============================================================


def _r_type(
    funct7: int,
    rs2: int,
    rs1: int,
    funct3: int,
    rd: int,
    opcode: int,
) -> int:
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _i_type(
    imm12: int,
    rs1: int,
    funct3: int,
    rd: int,
    opcode: int,
) -> int:
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _u_type(
    imm20: int,
    rd: int,
    opcode: int,
) -> int:
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode


def _j_type(
    offset: int,
    rd: int,
) -> int:
    """J-type (JAL): offset 为有符号字节偏移 (必须 2 字节对齐)."""
    imm = offset >> 1
    return (
        ((imm >> 19) & 1) << 31
        | ((imm >> 0) & 0x3FF) << 21
        | ((imm >> 10) & 1) << 20
        | ((imm >> 11) & 0xFF) << 12
        | (rd << 7)
        | 0b1101111
    )


# ============================================================
#  常量
# ============================================================

# opcodes
_OP = 0b0110011
_OP_IMM = 0b0010011
_LD = 0b0000011
_JALR = 0b1100111
_LUI = 0b0110111
_AUIPC = 0b0010111
_SYS = 0b1110011

# funct3
_F3_ADD = 0b000
_F3_LD = 0b011
_F3_JALR = 0b000
_F3_ADDI = 0b000

# funct7
_F7_ADD = 0x00

# CSR addresses
CSR_MTVEC = 0x305
CSR_MSTATUS = 0x300
CSR_MEPC = 0x341

# mstatus bits
MSTATUS_MPP_MASK = 0b11 << 11
MSTATUS_MPP_U = 0b00 << 11  # User mode
MSTATUS_MPP_S = 0b01 << 11  # Supervisor mode
MSTATUS_MPP_M = 0b11 << 11  # Machine mode

# GPR aliases
_ZERO = 0
_RA = 1
_SP = 2
_GP = 3
_T0 = 5
_T1 = 6
_T2 = 7


# ============================================================
#  工具片段: 加载 64-bit 立即数
# ============================================================


def imm64(rd: int, value: int) -> AsmSnippet:
    r"""生成将 64-bit 立即数加载到 rd 的片段.

    布局 (共 24 字节):
        auipc rd, 0          # rd ← PC
        ld    rd, 16(rd)     # rd ← mem[PC + 16]  (8 字节对齐的内联数据)
        jal   zero, 16       # 跳过数据池
        nop                  # 填充 (维持 8 字节对齐)
        .dword value         # 内联数据 (64-bit)

    Args:
        rd: 目标寄存器 (0–31).
        value: 要加载的 64-bit 值.
    """
    return AsmSnippet(
        words=[
            _u_type(0, rd, _AUIPC),  # auipc rd, 0
            _i_type(16, rd, _F3_LD, rd, _LD),  # ld rd, 16(rd)
            _j_type(16, _ZERO),  # jal zero, 16
            _i_type(0, _ZERO, _F3_ADDI, _ZERO, _OP_IMM),  # nop (addi x0, x0, 0)
            value & 0xFFFFFFFF,  # 数据低 32 位
            (value >> 32) & 0xFFFFFFFF,  # 数据高 32 位
        ],
        desc=f"load x{rd} ← 0x{value:016x}",
    )


# ============================================================
#  栈帧片段
# ============================================================


def set_sp(sp_value: int) -> AsmSnippet:
    """设置栈指针 sp (x2) 的值."""
    return imm64(_SP, sp_value)


def set_gp(gp_value: int) -> AsmSnippet:
    """设置全局指针 gp (x3) 的值."""
    return imm64(_GP, gp_value)


# ============================================================
#  CSR 操作片段
# ============================================================


def csr_write(csr_addr: int, rs: int) -> AsmSnippet:
    """将 rs 寄存器的值写入 csr_addr 指定的 CSR.

    使用 csrrw 指令: csrrw x0, csr, rs  (交换 CSR 与 rs, 丢弃旧值到 x0).

    Args:
        csr_addr: CSR 地址 (12-bit).
        rs: 源寄存器 (0–31).
    """
    return AsmSnippet(
        words=[
            ((csr_addr & 0xFFF) << 20) | (rs << 15) | (0b001 << 12) | (_ZERO << 7) | _SYS
        ],
        desc=f"csrrw x0, 0x{csr_addr:03x}, x{rs}",
    )


def set_mtvec(vector_base: int, mode: int = 0) -> list[AsmSnippet]:
    """设置 mtvec CSR 为 vector_base | mode.

    Args:
        vector_base: trap 向量基址 (必须 4 字节对齐).
        mode: 0 = 直接模式 (所有 trap → base), 1 = 向量模式 (base + 4×cause).

    Returns:
        两段片段: [load mtvec 值到 t0, 写入 mtvec CSR].
    """
    mtvec_val = (vector_base & ~0x3) | (mode & 0x3)
    return [
        imm64(_T0, mtvec_val),
        csr_write(CSR_MTVEC, _T0),
    ]


# ============================================================
#  特权级切换片段
# ============================================================


def switch_to_umode(entry_addr: int, sp_addr: int = 0) -> list[AsmSnippet]:
    """切换到 U 模式并跳转到 entry_addr.

    通过 mstatus.MPP 设置返回模式为 U, 然后用 mret 跳转.
    同时设置 mepc = entry_addr.

    Args:
        entry_addr: U 模式入口地址.
        sp_addr: U 模式栈指针 (0 = 不设置).
    """
    snippets: list[AsmSnippet] = []

    # 设置 mepc = entry_addr
    snippets.append(imm64(_T0, entry_addr))
    snippets.append(csr_write(CSR_MEPC, _T0))

    # 设置 mstatus.MPP = U (清除 MPP 位)
    # mstatus = (mstatus & ~MPP_MASK) | MPP_U
    # 使用 csrrc + csrrs 序列:
    #   1. csrrc x0, mstatus, t0   (t0 = MPP_MASK, 清零 MPP 位)
    #   2. csrrs x0, mstatus, t1   (t1 = 0 = MPP_U, 设置为 U)
    snippets.append(imm64(_T0, MSTATUS_MPP_MASK))
    snippets.append(AsmSnippet(
        words=[
            ((CSR_MSTATUS & 0xFFF) << 20) | (_T0 << 15) | (0b011 << 12) | (_ZERO << 7) | _SYS
        ],
        desc="csrrc x0, mstatus, t0  # 清除 MPP",
    ))
    snippets.append(AsmSnippet(
        words=[
            ((CSR_MSTATUS & 0xFFF) << 20) | (_ZERO << 15) | (0b010 << 12) | (_ZERO << 7) | _SYS
        ],
        desc="csrrs x0, mstatus, x0  # MPP = 0 (U)",
    ))

    # 设置 sp (如果需要)
    if sp_addr != 0:
        snippets.append(set_sp(sp_addr))

    # mret → 切换到 U 模式, pc = mepc
    snippets.append(AsmSnippet(
        words=[0x30200073],  # mret
        desc="mret",
    ))

    return snippets


# ============================================================
#  复合注入脚本
# ============================================================


def hosted_bootstrap(
    entry_addr: int,
    sp_value: int,
    gp_value: int | None = None,
    tvec_addr: int = 0,
) -> list[AsmSnippet]:
    """为托管程序 (非裸金属 ELF) 生成完整的环境注入脚本.

    执行顺序:
        1. 设置 mtvec (若 tvec_addr != 0)
        2. 设置 gp (若 gp_value != None)
        3. 设置 sp
        4. 通过 jalr 跳转到 entry_addr (保持在当前特权级)

    Args:
        entry_addr: 程序入口地址 (如 main).
        sp_value: 初始栈指针.
        gp_value: 初始全局指针 (None = 不设置).
        tvec_addr: mtvec 基址 (0 = 保持默认).

    Returns:
        AsmSnippet 列表, 依次执行.
    """
    snippets: list[AsmSnippet] = []

    if tvec_addr != 0:
        snippets.extend(set_mtvec(tvec_addr))

    if gp_value is not None:
        snippets.append(set_gp(gp_value))

    snippets.append(set_sp(sp_value))

    # 跳转到入口: jalr zero, t0, 0  → t0 = entry_addr
    snippets.append(imm64(_T0, entry_addr))
    snippets.append(AsmSnippet(
        words=[_i_type(0, _T0, _F3_JALR, _ZERO, _JALR)],
        desc=f"jalr zero, 0(t0)  # jump to 0x{entry_addr:x}",
    ))

    return snippets
