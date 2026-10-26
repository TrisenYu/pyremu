#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""trap 用例共用的 PTE 常量、hart 构造与中断控制器替身."""

from pyremu.core.decoder import Hart
from pyremu.core.hart import RiscvMode

# Sv39 叶 PTE 的权限位
PTE_V = 1 << 0
PTE_R = 1 << 1
PTE_W = 1 << 2
PTE_X = 1 << 3
PTE_U = 1 << 4

MTVEC = 0x8001_0000
STVEC = 0x8002_0000
# semihosting 服务对非法参数返回 -1, 以 64 位无符号数写出
SEMIHOSTING_ERR = 0xFFFF_FFFF_FFFF_FFFF


def make_pte(perm: int = 0, ppn: int = 0) -> int:
    """构造 Sv39 PTE 原始值: perm 为权限位, ppn 为物理页号.

    PPN 按 RISC-V 标准连续编码写入: PPN[9:0] 到 bits[19:10],
    PPN[18:10] 到 bits[28:20], PPN[43:19] 到 bits[53:29].
    """
    val = perm
    val |= (ppn & 0x3FF) << 10
    val |= ((ppn >> 10) & 0x1FF) << 20
    val |= ((ppn >> 19) & 0x1FFFFFFF) << 29
    return val


class StubImsic:
    """只记录 IPI 清除调用的 IMSIC 替身."""

    def __init__(self, pending: int = 0) -> None:
        self.pending = pending
        self.cleared: list[tuple[int, int]] = []

    def clear_ipi_on_trap(self, hart_id: int, exc_code: int) -> None:
        self.cleared.append((hart_id, exc_code))

    def get_pending_mip(self, hart_id: int) -> int:
        return self.pending


class StubPlic:
    """只给出待处理中断位图的 PLIC 替身."""

    def __init__(self, pending: int) -> None:
        self.pending = pending

    def get_pending_mip(self, hart_id: int) -> int:
        return self.pending


def make_hart(mode: RiscvMode = RiscvMode.M) -> Hart:
    """构造只带 mtvec / stvec 的 hart, 供陷态投递用例使用."""
    h = Hart(id=0)
    h.csrs["mtvec"].val = MTVEC
    h.csrs["stvec"].val = STVEC
    h.mode = mode
    h.pc = 0x8000_4000
    return h
