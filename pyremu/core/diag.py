#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中央诊断模块 — sret/mret ->U 模式寄存器状态跟踪.

与 ``pyremu/_native/cpu/src/diag.rs`` 保持一致的设计:
所有诊断逻辑集中在本模块, 调用方仅需一行条件导入 + 一行调用,
且由环境变量控制开关, 无需重编译 (Python 侧) 或 feature gate (Rust 侧).

Usage:
    from pyremu.core.diag import log_sret_to_u, log_mret_to_u, TRACE_SRET_TO_U

    if TRACE_SRET_TO_U and hart.mode == RiscvMode.U:
        log_sret_to_u(hart)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pyremu.configs_aux import cfg_bool, cfg_str

if TYPE_CHECKING:
    from pyremu.core.hart import HartWithRegs

# ============================================================
#  HartDiag — per-hart diagnostic counters (mirrors Rust HartDiag)
# ============================================================

class HartDiag:
    """Per-hart diagnostic counters populated by Rust batch engine.

    Mirrors the Rust ``HartDiag`` struct field-for-field; names use
    the Python convention (drop ``clint_`` / ``_snapshot`` suffix).
    Initialised to zero; updated by ``unmarshal_hart`` after each batch.
    """

    __slots__ = (
        "msip_set", "msip_clr", "mtc_wr",
        "msip_wr0", "msip_wr1", "wr1_remote", "wr1_self",
        "cd_start",
        "wfi_wake_msip", "wfi_wake_mtip", "wfi_wake_other",
        "trap_msip_total", "trap_msip_delegated",
        "msip_masked", "msip_no_trap",
        "nt_mip", "nt_mie", "msie_cleared_at", "wfi_no_trap",
        "nt_mode", "nt_clint_raw", "nt_pending",
        "msip_last_seen",
    )

    def __init__(self) -> None:
        self.msip_set: int = 0
        self.msip_clr: int = 0
        self.mtc_wr: int = 0
        self.msip_wr0: int = 0
        self.msip_wr1: int = 0
        self.wr1_remote: int = 0
        self.wr1_self: int = 0
        self.cd_start: int = 0
        self.wfi_wake_msip: int = 0
        self.wfi_wake_mtip: int = 0
        self.wfi_wake_other: int = 0
        self.trap_msip_total: int = 0
        self.trap_msip_delegated: int = 0
        self.msip_masked: int = 0
        self.msip_no_trap: int = 0
        self.nt_mip: int = 0
        self.nt_mie: int = 0
        self.msie_cleared_at: int = 0
        self.wfi_no_trap: int = 0
        self.nt_mode: int = 0
        self.nt_clint_raw: int = 0
        self.nt_pending: int = 0
        self.msip_last_seen: int = 0

    def load_ctypes(self, d: object) -> None:
        """Populate from a Rust ``HartDiagC`` ctypes struct (in-place)."""
        self.msip_set = d.clint_msip_set               # type: ignore[attr-defined]
        self.msip_clr = d.clint_msip_clr               # type: ignore[attr-defined]
        self.mtc_wr = d.clint_mtc_wr                   # type: ignore[attr-defined]
        self.msip_wr0 = d.clint_msip_wr0               # type: ignore[attr-defined]
        self.msip_wr1 = d.clint_msip_wr1               # type: ignore[attr-defined]
        self.wr1_remote = d.clint_wr1_remote           # type: ignore[attr-defined]
        self.wr1_self = d.clint_wr1_self               # type: ignore[attr-defined]
        self.cd_start = d.cooldown_start               # type: ignore[attr-defined]
        self.wfi_wake_msip = d.wfi_wake_msip           # type: ignore[attr-defined]
        self.wfi_wake_mtip = d.wfi_wake_mtip           # type: ignore[attr-defined]
        self.wfi_wake_other = d.wfi_wake_other         # type: ignore[attr-defined]
        self.trap_msip_total = d.trap_msip_total       # type: ignore[attr-defined]
        self.trap_msip_delegated = d.trap_msip_delegated  # type: ignore[attr-defined]
        self.msip_masked = d.msip_masked_by_msie       # type: ignore[attr-defined]
        self.msip_no_trap = d.msip_pending_no_trap     # type: ignore[attr-defined]
        self.nt_mip = d.nt_mip_snapshot                # type: ignore[attr-defined]
        self.nt_mie = d.nt_mie_snapshot                # type: ignore[attr-defined]
        self.msie_cleared_at = d.msie_cleared_at_pc    # type: ignore[attr-defined]
        self.wfi_no_trap = d.wfi_wake_no_msip_trap     # type: ignore[attr-defined]
        self.nt_mode = d.nt_mode                       # type: ignore[attr-defined]
        self.nt_clint_raw = d.nt_clint_raw             # type: ignore[attr-defined]
        self.nt_pending = d.nt_pending                 # type: ignore[attr-defined]
        self.msip_last_seen = d.msip_last_seen         # type: ignore[attr-defined]


TRACE_SRET_TO_U = cfg_bool("PYREMU_TRACE_SRET")
_DIAG_FILE = cfg_str("PYREMU_DIAG_LOG")
_SRET_DIAG_FILE = os.environ.get("PYREMU_SRET_LOG", _DIAG_FILE)

# 排查开关: 设置 PYREMU_NO_L2=1 后 Bus.read/write 对 RAM 地址完全绕过 L2 缓存.
# Rust batch 直接写 bytearray, L2 写命中合并旧数据后 flush 回写会污染 bytearray.
NO_L2 = os.environ.get("PYREMU_NO_L2") == "1"

# 诊断开关: PYREMU_DIAG_LOG 已设置时启用 PAGE_POISON (0xFE) 写追踪.
DIAG_POISON = bool(_DIAG_FILE)

# ld-linux 基址 + 大小 (固定值, 来自内核 auxv AT_BASE)
_LD_BASE: int = 0x3FF7FDC000
_LD_END: int = _LD_BASE + 0x20000

_TRAP_CAUSE_NAME: dict[int, str] = {
    12: "InstrPageFault",
    13: "LdPageFault",
    15: "StPageFault",
    8: "EcallFromUmode",
}


def log_sret_to_u(hart: HartWithRegs) -> None:
    """记录 SRET->U 时的关键寄存器状态."""
    _log_transition(hart, "sret")


def log_mret_to_u(hart: HartWithRegs) -> None:
    """记录 MRET->U 时的关键寄存器状态."""
    _log_transition(hart, "mret")


def log_ld_trap(hart: HartWithRegs, code: int, tval: int) -> None:
    """记录 ld-linux.so 范围内的 trap 投递.

    与 Rust ``diag::ld_linux_trap`` 一致, 但为纯 Python 路径服务
    (不依赖 libdecode.so 中的 diagnostic feature).
    """
    pc = hart.pc
    if not (_LD_BASE <= pc < _LD_END):
        return
    g = hart.gprs
    offset = pc - _LD_BASE
    exc_code = code & 0x7FFF_FFFF_FFFF_FFFF
    is_irq = (code >> 63) != 0
    cause_kind = "IRQ" if is_irq else "EXC"
    cause_name = _TRAP_CAUSE_NAME.get(exc_code, f"#{exc_code}")
    line = (
        f"[ld-trap py] pc={pc:#018x} (+{offset:#x}) "
        f"sepc={hart.sepc_val:#018x} "
        f"cause={cause_kind} {exc_code}({cause_name}) "
        f"tval={tval:#018x} "
        f"a0={g[10]:#018x} a1={g[11]:#018x} a3={g[13]:#018x} a5={g[15]:#018x} "
        f"sp={g[2]:#018x} ra={g[1]:#018x}\n"
    )
    try:
        with open(_DIAG_FILE, "a") as f:
            f.write(line)
    except OSError:
        pass


def _log_transition(hart: HartWithRegs, kind: str) -> None:
    """通用陷阱返回->U 模式日志."""
    try:
        offset = hart.pc - _LD_BASE
    except Exception:
        offset = 0
    g = hart.gprs
    line = (
        f"[{kind}->U] pc={hart.pc:#018x} (+{offset:#x}) "
        f"a0={g[10]:#018x} a1={g[11]:#018x} a2={g[12]:#018x} "
        f"a3={g[13]:#018x} a4={g[14]:#018x} a5={g[15]:#018x} "
        f"s0={g[8]:#018x} s1={g[9]:#018x} "
        f"sp={g[2]:#018x} ra={g[1]:#018x} "
        f"sepc={hart.sepc_val:#018x} mode_from={hart.mode.name}\n"
    )
    try:
        with open(_SRET_DIAG_FILE, "a") as f:
            f.write(line)
    except OSError:
        pass
