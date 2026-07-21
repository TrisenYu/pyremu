#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ctypes struct definitions mirroring Rust ``#[repr(C)]`` layouts.

Separated from ``__init__.py`` to keep the loader thin.  Domain modules
(``pyremu.core.decoder``, ``pyremu.memory.mmu``, etc.) import these types
from ``pyremu._native`` just like the FFI wrapper functions.
"""

import ctypes

# ============================================================
#  Instruction decode
# ============================================================


class DecodedFields(ctypes.Structure):
    """All standard RISC-V instruction fields extracted in a single pass.

    Field order mirrors ``decode::DecodedFields`` in Rust (64-byte aligned).
    """

    _fields_ = [
        ("imm12_se", ctypes.c_uint64),
        ("imm_s", ctypes.c_uint64),
        ("imm_b", ctypes.c_uint64),
        ("imm_j", ctypes.c_uint64),
        ("imm20_raw", ctypes.c_uint64),
        ("func12", ctypes.c_uint16),
        ("opcode", ctypes.c_uint8),
        ("rd", ctypes.c_uint8),
        ("func3", ctypes.c_uint8),
        ("rs1", ctypes.c_uint8),
        ("rs2", ctypes.c_uint8),
        ("func7", ctypes.c_uint8),
        ("is_compressed", ctypes.c_uint8),
        ("rs3", ctypes.c_uint8),
        ("fmt", ctypes.c_uint8),
    ]


class CompressedFields(ctypes.Structure):
    """All 16-bit compressed instruction fields, decoded in one pass.

    Field order mirrors ``decode::CompressedFields`` in Rust.
    """

    _fields_ = [
        ("imm", ctypes.c_uint64),
        ("imm2", ctypes.c_uint64),
        ("quadrant", ctypes.c_uint8),
        ("funct3", ctypes.c_uint8),
        ("rd", ctypes.c_uint8),
        ("rs1", ctypes.c_uint8),
        ("rs2", ctypes.c_uint8),
        ("rdp", ctypes.c_uint8),
        ("rs1p", ctypes.c_uint8),
        ("sf", ctypes.c_uint8),
        ("bit12", ctypes.c_uint8),
        ("bit65", ctypes.c_uint8),
    ]


# ============================================================
#  ALU
# ============================================================


class Alu64(ctypes.Structure):
    """ALU result with trap flag (0=ok, 1=invalid encoding -> IllInstr)."""

    _fields_ = [
        ("value", ctypes.c_uint64),
        ("trap", ctypes.c_uint8),
    ]


class FpOut(ctypes.Structure):
    """浮点运算结果 — mirrors Rust ``fpu::FpOut``.

    value: 结果 bits (FPR NaN-boxed / GPR 整数/布尔/分类掩码).
    fflags: 累积异常标志 (RISC-V fflags 位序).
    to_gpr: 1 = 写 rd 的 GPR, 0 = 写 rd 的 FPR.
    trap: 1 = 非法编码 -> IllInstr.
    """

    _fields_ = [
        ("value", ctypes.c_uint64),
        ("fflags", ctypes.c_uint8),
        ("to_gpr", ctypes.c_uint8),
        ("trap", ctypes.c_uint8),
    ]


# ============================================================
#  Sv39 MMU
# ============================================================


class Sv39Vpn(ctypes.Structure):
    """Decomposed Sv39 virtual address (VPN[2:0] + page offset)."""

    _fields_ = [
        ("vpn2", ctypes.c_uint64),
        ("vpn1", ctypes.c_uint64),
        ("vpn0", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
    ]


class PteFields(ctypes.Structure):
    """Parsed Sv39 page-table entry fields."""

    _fields_ = [
        ("ppn", ctypes.c_uint64),
        ("perm", ctypes.c_uint8),
        ("v", ctypes.c_uint8),
        ("is_leaf", ctypes.c_uint8),
        ("is_ptr", ctypes.c_uint8),
    ]
