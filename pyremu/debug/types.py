#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""调试器数据类型 — 快照、内存变更、栈帧、断点."""

from dataclasses import dataclass


@dataclass
class HartSnapshot:
    """单条指令执行前的 hart 状态."""

    pc: int
    gpr_vals: list[int]
    csr_vals: dict[str, int]
    mode: int
    reservation_valid: bool
    reservation_addr: int


@dataclass
class MemoryChange:
    """一次物理内存写入的记录 (addr, old_data)."""

    addr: int
    old: bytes


class MemWriteTracker:
    """内存写入追踪器 — 代理 hart 的 _mem_write_phy, 记录旧值供指令回滚.

    在每条指令执行前实例化, 注入 hart 的写后端; 指令执行后取出 changes
    列表供 rollback 使用, 同时恢复原始写后端.
    """

    def __init__(self, bus, orig_write):
        self._bus = bus
        self._orig_write = orig_write
        self.changes: list[MemoryChange] = []

    def __call__(self, addr: int, data: bytes) -> None:
        self._record_old(addr, len(data))
        self._orig_write(addr, data)

    def _record_old(self, addr: int, size: int) -> None:
        """Best-effort: 在覆盖前读取旧值. 读不到就跳过记录, 不阻断写入."""
        old = self._bus.try_read(addr, size)
        if old is not None:
            self.changes.append(MemoryChange(addr=addr, old=old))


@dataclass
class StackFrame:
    """栈回溯中的单帧."""

    idx: int
    fp: int
    sp: int
    ra: int
    pc: int  # call site = ra - 4 (若 ra ≥ 4)
    note: str = ""  # 非空表示特殊帧 (如 "U->S trap", "S->M ecall")
    mode: str = ""  # "M" / "S" / "U" — 用于颜色渲染


@dataclass
class Breakpoint:
    """断点 — 地址 / 指令类型 / opcode, 可选条件."""

    kind: str  # "addr" | "instr" | "opcode" | "cond"
    value: int  # address, funct12, opcode, or 0 for cond
    desc: str  # human-readable
    cond_type: str = ""  # "" | "reg" | "csr"
    cond_reg: str = ""  # 寄存器名
    cond_op: str = "=="  # 比较运算符
    cond_val: int = 0  # 期望值
