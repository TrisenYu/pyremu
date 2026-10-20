#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""调试器无状态工具 — 格式化、校验、常量.

所有函数均为模块级, 不依赖 Debugger 实例.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from pyremu.core import trap_def
from pyremu.core.hart import RiscvMode
from pyremu.core.mem_check_aux import translate_addr
from pyremu.utils.disassem import Opc
from pyremu.utils.mask import mask64

if TYPE_CHECKING:
    from pyremu.core.hart import HartWithRegs
    from pyremu.memory.bus import Bus

# ============================================================
#  节流常量
# ============================================================

_YIELD_EVERY = 500
_YIELD_INTERVAL = 0.005  # select() 超时秒数

# ============================================================
#  格式化
# ============================================================

MAX_INSTR_COUNT = 100_000  # 指令条数上限, 超界视为不可达

_SIZE_UNITS: tuple[tuple[str, int], ...] = (
    ("EB", 1 << 60),
    ("PB", 1 << 50),
    ("TB", 1 << 40),
    ("GB", 1 << 30),
    ("MB", 1 << 20),
    ("KB", 1 << 10),
)


def hex_addr(v: int, styled: bool = True) -> str:
    """格式化为 16 位 hex: ``0x0000000080000000``.

    *styled* 为 True 时, 虚地址以蓝色高亮 (Rich tag).
    """
    raw = f"0x{mask64(v):016x}"
    if styled:
        return f"[bright_blue]{raw}[/]"
    return raw


def fmt_size(n: int) -> str:
    """human-readable 尺寸, 超出最大单位时落到最远单位."""
    for name, threshold in _SIZE_UNITS:
        if n >= threshold:
            return f"{n / threshold:.1f} {name}"
    return f"{n} B"


def fmt_instr_count(n: int) -> str:
    if n < 1000:
        return str(n)
    nn = float(n / 1000.0)
    if nn < 1000:
        return f"{nn:.3f}K"
    nn /= 1000
    if nn < 1000:
        return f"{nn:.3f}M"
    nn /= 1000
    if nn < 1000:
        return f"{nn:.3f}B"
    nn /= 1000
    return f"{nn:.3f}T"


def trap_cause_name(mcause_val: int) -> str:
    """将 mcause 寄存器值翻译为可读的 trap 类型名称."""
    return trap_def.trap_cause_name(mcause_val)


# ============================================================
#  虚拟地址读取
# ============================================================


def read_va_bytes(
    bus: Bus,
    hart: HartWithRegs,
    va: int,
    size: int,
    *,
    is_execute: bool = False,
) -> bytes | None:
    """按指定的访问类型从虚拟地址读取内存, 跨越页边界时逐页翻译.

    取指语义只判定页面的可执行位; 数据读语义还要求 R=1, 只在 mstatus.MXR
    置位时才由 X=1 顶替. 两者都不施加 PMP 与 PMA 判定.

    Returns:
        读到的字节; 翻译失败或读取失败时返回 None.
    """
    if size <= 0 or size > 4096:
        return None
    if hart.mmu_mode == 0 or hart.mode == RiscvMode.M:
        return bus.try_read(va, size)
    result = bytearray()
    remain = size
    cur = va
    while remain > 0:
        page_off = cur & 0xFFF
        chunk = min(remain, 0x1000 - page_off)
        ok, pa = translate_addr(
            hart, cur, is_execute=is_execute, is_hart_access=False
        )
        if not ok:
            return None
        data = bus.try_read(pa, chunk)
        if data is None:
            return None
        result.extend(data)
        remain -= chunk
        cur += chunk
    return bytes(result)


def read_code_bytes(
    bus: Bus,
    hart: HartWithRegs,
    va: int,
    size: int,
) -> bytes | None:
    """按取指语义读取指令字节, 取指语义被拒时退回数据读语义.

    取指只判定页面的可执行位, 数据读语义还要求 R=1. 只可执行的页面须用取指
    语义读出; 取指语义被拒时退回数据读语义, 使 S 模式仍能读出 U 模式页面上
    的指令字节.

    Returns:
        读到的字节; 两种语义都读不到时返回 None.
    """
    data = read_va_bytes(bus, hart, va, size, is_execute=True)
    if data is not None:
        return data
    return read_va_bytes(bus, hart, va, size, is_execute=False)


# ============================================================
#  校验
# ============================================================


def check_rv64_addr(v: int) -> bool:
    """校验 *v* 是否在 RV64 地址范围 [0, 2^64) 内."""
    return 0 <= v < (1 << 64)


# ============================================================
#  中断/IP 位描述
# ============================================================


def ip_bits() -> list[tuple[str, int, str]]:
    """mip/mie/sip/sie 位域描述."""
    return [
        ("USIP", 0, "U 模式软件中断挂起"),
        ("SSIP", 1, "S 模式软件中断挂起"),
        ("MSIP", 3, "M 模式软件中断挂起"),
        ("UTIP", 4, "U 模式定时器中断挂起"),
        ("STIP", 5, "S 模式定时器中断挂起"),
        ("MTIP", 7, "M 模式定时器中断挂起"),
        ("UEIP", 8, "U 模式外部中断挂起"),
        ("SEIP", 9, "S 模式外部中断挂起"),
        ("MEIP", 11, "M 模式外部中断挂起"),
    ]


# ============================================================
#  异常/中断名称表
# ============================================================

EXC_NAMES: dict[int, str] = {
    0: "InstrAddrMisaligned",
    1: "InstrAccessFault",
    2: "IllInstr",
    3: "Breakpoint",
    4: "LdAddrMisaligned",
    5: "LdAccessFault",
    6: "StAddrMisaligned",
    7: "StAccessFault",
    8: "Ecall from U",
    9: "Ecall from S",
    11: "Ecall from M",
    12: "InstrPageFault",
    13: "LdPageFault",
    15: "StPageFault",
}

IRQ_NAMES: dict[int, str] = {
    1: "SSIP (S 模式软件中断)",
    3: "MSIP (M 模式软件中断)",
    5: "STIP (S 模式定时器中断)",
    7: "MTIP (M 模式定时器中断)",
    9: "SEIP (S 模式外部中断)",
    11: "MEIP (M 模式外部中断)",
}

# ============================================================
#  断点常量
# ============================================================

INSTR_BP_NAMES: dict[str, int] = {
    "ecall": 0x000,
    "ebreak": 0x001,
    "mret": 0x302,
    "sret": 0x102,
    "wfi": 0x105,
}

KNOWN_OPCODES: frozenset[int] = frozenset(o.value for o in Opc)

RAM_MUL: dict[str, int] = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}

# ============================================================
#  历史文件修剪
# ============================================================


def trim_history(max_entries: int = 10000) -> None:
    """若历史文件超过 *max_entries* 行, 仅保留最近一半."""
    hist_path = Path.home() / ".pyremu_history"
    if not hist_path.is_file():
        return
    try:
        lines = hist_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) <= max_entries:
            return
        keep = max_entries // 2
        hist_path.write_text("\n".join(lines[-keep:]) + "\n", encoding="utf-8")
    except OSError:
        pass  # 文件被锁或权限不足时静默跳过


# ============================================================
#  补全排序
# ============================================================


def group_order(key: str) -> tuple[int, str]:
    """字典序排序: 字母先 (a-z), 再数字, 再下划线, 再点, 最后其他."""
    c = key.lower()
    if "a" <= c <= "z":
        return (0, c)
    if "0" <= c <= "9":
        return (1, c)
    if c == "_":
        return (2, c)
    if c == ".":
        return (3, c)
    return (4, c)
