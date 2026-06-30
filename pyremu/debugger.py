#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""
交互式 RISC-V 调试器 (rvdb).

提供指令级单步、状态检查、指令回滚功能。

信号处理:
- 模拟器运行时/REPL中: Ctrl+C 一次暂停

回滚机制:
- 每条指令执行前保存 hart 状态快照 (PC / GPRs / CSRs / mode / reservation)
- 指令执行期间追踪所有物理内存写入 (addr → old_data)
- 回滚时恢复 hart 状态并逆序撤销内存变更
"""

import argparse
import re
import signal
import struct
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import cast
from pathlib import Path

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markup import escape as _rich_escape
from rich.panel import Panel
from rich.table import Table

from pyremu.core.decoder import (
    Hart,
    Opc,
    decode_c_sdsp,
    decode_sd_sp,
    parse_compressed,
    parse_func12,
    parse_func3,
    parse_opcode,
    parse_rd,
)
from pyremu.core.hart import RiscvMode
from pyremu.core.mem_check_aux import inject_memory_backend, translate_addr
from pyremu.core.registers import (
    csr_addr_from_name,
    gpr_alias,
    gpr_idx_from_name,
    gpr_name,
    register_csr,
    register_fpr,
    register_gpr,
)
from pyremu.core.trap import TrapType, trap_cause_name
from pyremu.core.trap_handler import check_pending_interrupts, deliver_trap
from pyremu.emulator import Emulator
from pyremu.env_inject import Preloader
from pyremu.memory.l2cache import L2Cache
from pyremu.memory.mmu import (
    pte_flags_str,
    satp_root_ppn,
    sv39_canonical_va,
    sv39_decompose_va,
    sv39_walk,
)
from pyremu.memory.pmp import (
    PMP_A_MASK,
    PMP_A_NA4,
    PMP_A_NAPOT,
    PMP_A_OFF,
    PMP_A_TOR,
    PMP_L,
    PMP_R,
    PMP_W,
    PMP_X,
    decode_napot,
)
from pyremu.memory.tlb import decode_tlb_perm
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils.disassem import disasm
from pyremu.utils.parse_bin import FirmwareImage, FirmwareSegment, parse_firmware
from pyremu.utils.str_aux import fmt_hexdump
from pyremu.utils.wrapper import seize_val_err

# ============================================================
#  状态快照 (用于后续回滚已执行指令的特性)
# ============================================================
# 节流: 每 N 条指令后短睡, 降低 CPU 占用
_YIELD_EVERY = 5000
_YIELD_SLEEP = 0.002

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
    note: str = ""  # 非空表示特殊帧 (如 "U→S trap", "S→M ecall")
    mode: str = ""  # "M" / "S" / "U" — 用于颜色渲染


# ============================================================
#  断点
# ============================================================


@dataclass
class _Breakpoint:
    """断点 — 地址 / 指令类型 / opcode, 可选条件."""

    kind: str  # "addr" | "instr" | "opcode" | "cond"
    value: int  # address, funct12, opcode, or 0 for cond
    desc: str  # human-readable
    cond_type: str = ""  # "" | "reg" | "csr"
    cond_reg: str = ""  # 寄存器名
    cond_op: str = "=="  # 比较运算符
    cond_val: int = 0  # 期望值


# 可命名的指令断点 — 与 handle_sys / _PRIV_MNEMONIC 一致
_INSTR_BP_NAMES: dict[str, int] = {
    "ecall": 0x000,
    "ebreak": 0x001,
    "mret": 0x302,
    "sret": 0x102,
    "wfi": 0x105,
}

# 已知 RV64 标准 opcode (Opc 枚举)
# 注意: opcode 断点仅匹配 32-bit 标准指令; 压缩指令需用压缩象限
# breakpoint (0x00/0x01/0x02), 但因 4-byte 取指窗口内 instr & 0x7F
# 对压缩指令不能可靠得出象限值, 故暂不支持 opcode bp on compressed.
_KNOWN_OPCODES: frozenset[int] = frozenset(o.value for o in Opc)

# human-readable 尺寸单位 (1024 进制), 从大到小排列供 _fmt_size 遍历
_SIZE_UNITS: tuple[tuple[str, int], ...] = (
    # ("YB", 1<<80),
    # ("ZB", 1<<70),
    ("EB", 1 << 60),
    ("PB", 1 << 50),
    ("TB", 1 << 40),
    ("GB", 1 << 30),
    ("MB", 1 << 20),
    ("KB", 1 << 10),
)

MAX_INSTR_COUNT = 100_000  # 指令条数上限, 超界视为不可达
# ============================================================
#  交互式调试器
# ============================================================

# 字典序排序: 字母先 (a-z), 再数字, 再下划线, 再点, 最后其他
def group_order(key: str) -> tuple[int, str]:
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


class Debugger:
    """RISC-V 交互式调试器.

    封装 Emulator, 提供单步执行、状态检查、指令回滚功能。
    默认操作单 hart (hart_id=0), 可通过 `hart <id>` 切换。
    """

    def __init__(
        self,
        emulator: Emulator,
        hart_id: int = 0,
        image: FirmwareImage | None = None,
    ) -> None:
        self._emu: Emulator = emulator
        self._hart_id: int = hart_id
        self._image: FirmwareImage | None = image
        self._load_offset: int = 0  # PIE 搬迁偏移 (供符号→地址转换)
        self._preload_path: str | None = None  # 预加载 shellcode 路径 (供 restart)
        self._running: bool = False

        self._sigint_count: int = 0
        self._paused: bool = False
        self._terminated: bool = False

        self._snapshot: HartSnapshot | None = None
        self._mem_changes: list[MemoryChange] = []
        self._instr_count: int = 0
        self._fdt_addr: int | None = None  # 设备树加载地址 (供 banner 显示)

        # Rich console — 所有交互输出经此通道 (彩色, 高亮)
        self._console: Console = Console(highlight=True)

        # 陷态上下文去重: 同一 mcause 值只展示一次，避免 handler 执行期间重复刷屏
        self._trap_displayed_mcause: int | None = None

        # 上一条命令 (按回车重复执行)
        self._last_command: str | None = None

        # disasm 重复时的自动推进地址 (指向上次输出末尾)
        self._disasm_next_addr: int | None = None
        # vdisasm Enter 重复时推进到此地址
        self._vdisasm_next_addr: int | None = None

        # disasm 步数持久化: 参考 PC 及累积步数, 供 Enter 重复时继续编号
        self._disasm_ref_pc: int | None = None
        self._disasm_base_step: int = 0
        self._disasm_past_terminator: bool = False

        # 断点
        self._breakpoints: list[_Breakpoint] = []
        self._bp_mode: str = "sync"  # "sync"=全停, "async"=仅当前 hart
        self._hart_paused: set[int] = set()  # async 模式下被暂停的 hart
        self._bp_hit_this_run: set[tuple[str, int]] = set()
        self._prev_instr_csr_addr: int = -1  # 上一条指令的 CSR 地址 (供 CSR 条件后检)

        # 写监控 (watch): 追踪对特定内存地址的写入
        self._watch_ranges: list[tuple[int, int]] = []  # [(start, end), ...]
        self._watch_hit: tuple[int, int, int, bytes] | None = None  # (pc, addr, size, data)
        self._watch_installed: bool = False

        # 最近的栈回溯帧列表 (供 frame N 选择)
        self._stack_frames: list[StackFrame] = []
        self._current_frame_idx: int = 0

        # prompt_toolkit REPL — 方向键历史, Tab 补全, 持久化历史文件
        self._history: FileHistory = FileHistory(str(Path.home() / ".pyremu_history"))
        self._trim_history(max_entries=10000)
        self._completer: WordCompleter = self._build_completer()
        self._session: PromptSession[str] = PromptSession(
            history=self._history,
            completer=self._completer,
            style=Style.from_dict(
                {
                    "prompt": "#00aa00 bold",
                    "": "#cccccc",
                }
            ),
        )

    # ==========================================================
    #  Tab 补全 — 名称全部来自 core/registers 工厂函数
    # ==========================================================

    def _build_completer(self) -> WordCompleter:
        words: list[str] = [
            # 执行控制
            "s",
            "step",
            "c",
            "continue",
            "r",
            "run",
            "undo",
            "rollback",
            "restart",
            "b",
            "bp",
            # 寄存器 / CSR 操作
            "regs",
            "gpr",
            "reg",
            "set",
            "w",
            "csr",
            "csrw",
            # 状态 / 内存
            "pc",
            "mode",
            "mstatus",
            "tlb",
            "tlbflush",
            "cache",
            "satp",
            "pt",
            "mem",
            "vmem",
            "status",
            "info",
            "symbols",
            "sym",
            "disasm",
            "vdisasm",
            "stack",
            "show",
            "bt",
            "frame",
            "f",
            # 配置 & 帮助
            "hart",
            "h",
            "help",
            "?",
            "q",
            "quit",
            "exit",
        ]
        # GPR 名称 (x0–x31 + ABI alias)
        for r in register_gpr():
            words.append(r.name)
            if r.alias:
                words.append(r.alias)
        # FPR 名称 (f0–f31 + ABI alias)
        for r in register_fpr():
            words.append(r.name)
            if r.alias:
                words.append(r.alias)
        # CSR 名称
        words.extend(register_csr().keys())
        # "csr list" 的 list 子命令
        words.append("list")
        return WordCompleter(words, ignore_case=True, sentence=True)

    # ==========================================================
    #  历史文件裁剪 — 防止 ~/.pyremu_history 无限增长
    # ==========================================================

    @staticmethod
    def _trim_history(max_entries: int = 10000) -> None:
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

    # ==========================================================
    #  hart 访问
    # ==========================================================

    @property
    def hart(self):
        return self._emu.harts[self._hart_id]

    # ==========================================================
    #  信号处理
    # ==========================================================

    def _sigint_repl(self, _signum: int, _frame) -> None:
        raise KeyboardInterrupt

    def _sigint_run(self, _signum: int, _frame) -> None:
        self._sigint_count += 1
        payload = "\n[yellow]暂停请求 — 当前指令完成后回到 REPL[/]"
        if self._sigint_count != 1:
            payload = "\n[red bold]强制终止模拟循环[/]"
            self._terminated = True
        self._console.print(payload)
        self._paused = True

    def _enter_repl_mode(self) -> None:
        signal.signal(signal.SIGINT, self._sigint_repl)

    def _enter_run_mode(self) -> None:
        signal.signal(signal.SIGINT, self._sigint_run)

    # ==========================================================
    #  断点检查
    # ==========================================================

    def _instr_may_affect_cond(self, instr: int, bp: "_Breakpoint") -> bool:
        """快速预过滤: 指令是否可能改变条件断点关注的寄存器/CSR.

        返回 False 时可安全跳过条件评估, 大幅降低 cond 断点的性能开销:
        - CSR 条件: opcode=0x73 且 CSR 地址匹配 (当前指令写 CSR),
          或上一条指令写了目标 CSR (当前指令可见写入后的新值)
        - GPR 条件: 仅 rd 字段匹配目标寄存器且 rd≠x0 时返回 True
        """
        if bp.cond_type == "csr":
            target = csr_addr_from_name(bp.cond_reg)
            if target is None:
                return True  # 未知 CSR, 保守评估
            # 当前指令写目标 CSR
            if parse_opcode(instr) == 0x73 and parse_func12(instr) == target:
                return True
            # 上一条指令写了目标 CSR (当前指令可见写入后的新值)
            if self._prev_instr_csr_addr == target:
                return True
            return False
        elif bp.cond_type == "reg":
            rd = parse_rd(instr)
            if rd == 0:  # x0 从不改变
                return False
            target = gpr_idx_from_name(bp.cond_reg)
            return target is not None and rd == target
        return True  # 未知类型, 保守评估

    def _eval_bp_condition(self, hart, bp: "_Breakpoint") -> bool:
        """评估断点的附加条件; 无条件时返回 True."""
        if not bp.cond_type:
            return True
        if bp.cond_type == "reg":
            actual = hart.read_gpr_by_name(bp.cond_reg)
        elif bp.cond_type == "csr":
            actual = hart.read_csr_by_name(bp.cond_reg)
        else:
            return True
        c = bp.cond_val
        if bp.cond_op == "==":
            return actual == c
        elif bp.cond_op == "!=":
            return actual != c
        elif bp.cond_op == "<":
            return actual < c
        elif bp.cond_op == ">":
            return actual > c
        return actual == c

    def _report_bp_hit(self, bp: "_Breakpoint", hart, pc: int) -> None:
        """打印断点命中信息并暂停."""
        asm_info = self._fetch_and_disasm(pc)
        asm_line = f"\n    {asm_info[1]}" if asm_info else ""
        cond_hint = ""
        if bp.cond_type:
            actual_val = (
                self._read_gpr_by_name(bp.cond_reg)
                if bp.cond_type == "reg"
                else self._read_csr_by_name(bp.cond_reg)
            )
            cond_hint = f" [{bp.cond_reg}=0x{actual_val:x}]"
        self._console.print(
            f"  [bold yellow]● 断点命中[/]  {bp.desc}{cond_hint}  "
            f"@ [cyan]Hart {hart.id}[/]  {self._hex(pc)}{asm_line}"
        )
        if self._bp_mode == "async":
            self._hart_paused.add(hart.id)
        else:
            self._paused = True

    def _check_breakpoints(self, hart, pc: int, instr: int) -> bool:
        """若当前指令/地址命中任何断点, 暂停并返回 True.

        在指令执行前调用, 匹配地址/指令类型/opcode 断点.
        命中时打印断点信息并设置 ``_paused=True``.
        """
        if not self._breakpoints:
            return False

        for bp in self._breakpoints:
            hit = False
            if bp.kind == "cond":
                # 快速预过滤: 仅涉及目标寄存器/CSR 的指令才评估条件
                hit = self._instr_may_affect_cond(instr, bp)
            elif bp.kind == "addr":
                hit = pc == bp.value
            elif bp.kind == "instr":
                hit = (
                    parse_opcode(instr) == 0x73
                    and parse_func3(instr) == 0
                    and parse_func12(instr) == bp.value
                )
            elif bp.kind == "opcode":
                # opcode 断点仅适用 32-bit 标准指令: 压缩指令的
                # instr & 0x7F 不能可靠得出象限值, 跳过
                if parse_compressed(instr):
                    continue
                hit = parse_opcode(instr) == bp.value

            # 条件评估
            if not hit or not self._eval_bp_condition(hart, bp):
                continue

            # 本次 continue 内同一 PC 的地址断点只停一次 (避免 c 反复卡在同一处)
            bp_key = (bp.kind, pc) if bp.kind == "addr" else (bp.kind, id(bp))
            if bp_key in self._bp_hit_this_run:
                continue

            self._bp_hit_this_run.add(bp_key)
            self._report_bp_hit(bp, hart, pc)
            return True

        # 记录当前指令的 CSR 地址, 供下一条指令的条件断点后检
        self._prev_instr_csr_addr = -1
        if parse_opcode(instr) == 0x73:
            self._prev_instr_csr_addr = parse_func12(instr)
        return False

    def _check_multi_hart_bp(self) -> None:
        """多 hart 执行后检查断点.

        在 ``emu.step()`` 之后调用, 遍历所有 hart:
        - 地址断点: 检查 PC
        - 条件断点: 评估条件 (CSR/GPR 值已是执行后的最新状态)
        - 指令/opcode 断点: 回读当前 PC 处指令 (已执行, 尽力而为)
        """
        for h in self._emu.harts:
            for bp in self._breakpoints:
                hit = False
                if bp.kind == "addr":
                    hit = h.pc == bp.value
                elif bp.kind == "cond":
                    # 多 hart 模式下没有单条 instr 参数, 预过滤器放行
                    hit = True
                elif bp.kind in ("instr", "opcode"):
                    try:
                        instr_bytes = self._emu.bus.read(h.pc, 4)
                        instr = int.from_bytes(instr_bytes, "little", signed=False)
                    except Exception:
                        continue
                    if bp.kind == "instr":
                        hit = (
                            parse_opcode(instr) == 0x73
                            and parse_func3(instr) == 0
                            and parse_func12(instr) == bp.value
                        )
                    elif not parse_compressed(instr):
                        hit = parse_opcode(instr) == bp.value
                if not hit or not self._eval_bp_condition(h, bp):
                    continue
                bp_key = (bp.kind, h.pc) if bp.kind == "addr" else (bp.kind, bp.value)
                if bp_key in self._bp_hit_this_run:
                    continue
                self._bp_hit_this_run.add(bp_key)
                self._report_bp_hit(bp, h, h.pc)
                return

    # ==========================================================
    #  快照 & 回滚
    # ==========================================================

    def _save_snapshot(self) -> HartSnapshot:
        h = self.hart
        return HartSnapshot(
            pc=h.pc,
            gpr_vals=list(h.gprs),
            csr_vals={name: csr.val for name, csr in h.csrs.items()},
            mode=h.mode.value,
            reservation_valid=h.reservation_valid,
            reservation_addr=h.reservation_addr,
        )

    def _restore_snapshot(self, snap: HartSnapshot) -> None:
        h = self.hart
        h.pc = snap.pc
        for i, val in enumerate(snap.gpr_vals):
            h.gprs[i] = val
        for name, val in snap.csr_vals.items():
            if name not in h.csrs:
                continue
            h.csrs[name].val = val
        h.mode = RiscvMode(snap.mode)
        if snap.reservation_valid:
            h.set_reservation(snap.reservation_addr)
        else:
            h.clear_reservation()

    def step_one(self) -> None:
        """执行一条指令并保存快照 (供后续回滚)."""
        h = self.hart

        if h._halted:
            return  # 已输出过转储信息, 静默跳过

        # WFI 等待状态: 仅检查中断唤醒, 不取指/执行
        if h._waiting:
            check_pending_interrupts(h)
            return

        snap = self._save_snapshot()

        orig_write = h._mem_write_phy
        assert orig_write is not None, "memory backend not attached"

        tracker = MemWriteTracker(self._emu.bus, orig_write)
        h._mem_write_phy = tracker

        try:
            pc_before = h.pc
            raw = self._try_read_va(h.pc, 4)
            if raw is None:
                deliver_trap(h, TrapType.InstrPageFault, tval=h.pc, is_interrupt=False)
                return
            instr = int.from_bytes(raw, "little", signed=False)

            # 断点检查 — 命中时停止 (同一 continue 内已命中过的地址会被跳过)
            if self._check_breakpoints(h, pc_before, instr):
                return

            try:
                advance = h.exec_instr(instr)
            except NotImplementedError:
                deliver_trap(h, TrapType.IllInstr, tval=instr, is_interrupt=False)
                advance = 0
            # 仅在 PC 未被 exec_instr 内部修改时才自动推进
            if advance != 0 and h.pc == pc_before:
                h.pc = (h.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF
                h._consecutive_traps = 0
            # 连续 trap 超过阈值 → 不可恢复
            if h._consecutive_traps >= self._emu._TRAP_LOOP_THRESHOLD:
                dump = self._emu._dump_hart_state(h)
                self._console.print(
                    Panel(dump, title="[red]Hart State Dump[/]", border_style="red")
                )
                h._halted = True
                self._err(f"Hart {h.id} 进入不可恢复陷态, 已暂停")

            """
            单条指令不可被中途打断。

            CPU 只会在当前指令彻底执行完成、准备取下一条指令时，检查中断请求。

            若当前是访存指令必须等待访存阶段完成（数据读回寄存器 / 数据写入内存），
            指令才算结束，从而才会响应中断。
            """
            check_pending_interrupts(h)
            self._instr_count += 1
        finally:
            h._mem_write_phy = orig_write

        self._mem_changes = tracker.changes
        self._snapshot = snap

        # 写监控检查 — 命中时暂停并报告
        if not self._watch_ranges:
            return
        for change in tracker.changes:
            a, sz = change.addr, len(change.old)
            for ws, we in self._watch_ranges:
                if not (max(a, ws) < min(a + sz, we)):
                    continue
                self._watch_hit = (h.pc, a, sz, change.old)
                self._paused = True
                self._console.print(
                    f"  [red] (@) 写监控命中[/] pc={self._hex(h.pc)}"
                    f"  写入 0x{self._hex(a)} ({sz}B)"
                    f"  旧值={change.old[:16].hex()}"
                )
                return

    def rollback(self) -> None:
        """回滚最近一条指令的影响."""
        if self._snapshot is None:
            self._warn("没有可回滚的指令")
            return

        for change in reversed(self._mem_changes):
            self._emu.bus.try_write(change.addr, change.old)

        self._restore_snapshot(self._snapshot)
        self._snapshot = None
        self._mem_changes = []
        self._instr_count -= 1
        self._console.print("[dim]已撤销上一条指令[/]")

    # ==========================================================
    #  断点命令
    # ==========================================================

    def _try_set_symbol_bp(self, name: str) -> bool:
        """尝试按符号名设置地址断点.  成功返回 True, 未匹配返回 False."""
        if self._image is None or not self._image.symbols:
            return False
        sym_addr = self._image.symbols.get(name)
        if sym_addr is None:
            return False
        addr = sym_addr + self._load_offset
        if not self._check_rv64_addr(addr):
            return True  # 地址非法, 但已反馈错误, 视为已处理
        bp = _Breakpoint(kind="addr", value=addr, desc=f"{name} ({self._hex(addr)})")
        self._breakpoints.append(bp)
        asm_info = self._fetch_and_disasm(addr)
        asm_str = f"  {asm_info[1]}" if asm_info else ""
        self._console.print(
            f"  [green]断点 {len(self._breakpoints)}[/]  "
            f"[yellow]{name}[/] {self._hex(addr)}"
            f"{asm_str}"
        )
        return True

    def cmd_bp_set(self, rest: str) -> None:
        """设置断点: bp <addr> | bp <symbol> | bp <type>.

        支持条件: bp <addr> if reg <name> <op> <val>
                  bp <addr> if csr <name> <op> <val>
        """
        # 解析条件从句
        cond_type, cond_reg, cond_op, cond_val = "", "", "==", 0
        parts = rest.split(None, 2)
        if len(parts) >= 3 and parts[1] == "if":
            rest = parts[0]
            cond_parts = parts[2].split(None, 3)
            if len(cond_parts) >= 3 and cond_parts[0] in ("reg", "csr"):
                cond_type = cond_parts[0]
                cond_reg = cond_parts[1]
                cond_op = cond_parts[2] if len(cond_parts) >= 3 else "=="
                cond_val = int(cond_parts[3], 0) if len(cond_parts) >= 4 else 0

        # 1) 符号名
        if self._try_set_symbol_bp(rest):
            # 补充条件到最后一个断点
            if not cond_type:
                return
            self._breakpoints[-1].cond_type = cond_type
            self._breakpoints[-1].cond_reg = cond_reg
            self._breakpoints[-1].cond_op = cond_op
            self._breakpoints[-1].cond_val = cond_val
            self._breakpoints[-1].desc += f" if {cond_type} {cond_reg}{cond_op}" + \
                f"0x{cond_val:x}"
            self._console.print(
                f"    条件: {cond_type} {cond_reg} {cond_op} 0x{cond_val:x}"
            )
            return

        # 2) 地址 (pc / 寄存器名 / 十六进制)
        addr = self._resolve_addr(rest)
        if addr is not None:
            if not self._check_rv64_addr(addr):
                return
            desc = self._hex(addr)
            label = "断点"
            if cond_type:
                desc += f" if {cond_type} {cond_reg}{cond_op}0x{cond_val:x}"
                label = "条件断点"
            bp = _Breakpoint(
                kind="addr",
                value=addr, desc=desc,
                cond_type=cond_type,
                cond_reg=cond_reg,
                cond_op=cond_op,
                cond_val=cond_val,
            )
            self._breakpoints.append(bp)
            asm_info = self._fetch_and_disasm(addr)
            asm_str = f"  {asm_info[1]}" if asm_info else ""
            cond_str = (
                f" [bold yellow]if[/] {cond_type} {cond_reg} {cond_op} [yellow]0x{cond_val:x}[/]"
                if cond_type
                else ""
            )
            self._console.print(
                f"  [green]{label} {len(self._breakpoints)}[/]{cond_str}  "
                f"[yellow]{self._hex(addr)}[/]"
                f"{asm_str}"
            )
            return

        # 3) 命名指令
        name = rest.lower()
        if name in _INSTR_BP_NAMES:
            bp = _Breakpoint(
                kind="instr",
                value=_INSTR_BP_NAMES[name],
                desc=name,
            )
            self._breakpoints.append(bp)
            self._console.print(
                f"  [green]断点 {len(self._breakpoints)}[/]  [yellow]{name}[/]"
            )
            return

        self._err(f"无法识别的断点参数: {rest}")

    @seize_val_err("opcode 需为十六进制整数 (支持 0x 前缀)")
    def cmd_bp_opcode(self, arg: str) -> None:
        """设置 opcode 断点: bp opcode <hex>."""
        raw = int(arg, 0)
        val = raw & 0x7F
        if raw < 0 or raw > 0x7F:
            self._err(f"opcode 0x{raw:x} 超出 7-bit 范围 [0, 127]")
            return
        if val not in _KNOWN_OPCODES:
            self._err(f"opcode 0x{val:02x} 不是已知的 RV64 opcode, 拒绝设置断点")
            return
        bp = _Breakpoint(
            kind="opcode", value=val,
            desc=f"opcode 0x{val:02x}",
        )
        self._breakpoints.append(bp)
        self._console.print(
            f"  [green]断点 {len(self._breakpoints)}[/]  [yellow]opcode 0x{val:02x}[/]"
        )

    def cmd_bp_list(self) -> None:
        """列出所有断点."""
        if not self._breakpoints:
            self._console.print("  [dim](无断点)[/]")
            return
        tbl = Table(title="断点列表", border_style="blue")
        tbl.add_column("#", style="cyan", justify="right")
        tbl.add_column("类型", style="yellow")
        tbl.add_column("值", style="green")
        for i, bp in enumerate(self._breakpoints, 1):
            tbl.add_row(str(i), bp.kind, bp.desc)
        self._console.print(tbl)

    @seize_val_err("断点编号需为整数")
    def cmd_bp_delete(self, idx_str: str) -> None:
        """删除断点: bp delete <n> (1-based)."""
        idx = int(idx_str, 0) - 1
        if idx < 0 or idx >= len(self._breakpoints):
            self._err(f"断点编号超出范围: {idx_str} (当前 {len(self._breakpoints)} 个)")
            return
        removed = self._breakpoints.pop(idx)
        self._console.print(f"  [dim]已删除断点 [yellow]{removed.desc}[/][/]")

    def cmd_bp_clear(self) -> None:
        """清除全部断点."""
        count = len(self._breakpoints)
        self._breakpoints.clear()
        self._console.print(f"  [dim]已清除 {count} 个断点[/]")

    def cmd_bp_mode(self, arg: str | None = None) -> None:
        """查看或设置断点模式: bp mode [sync|async]."""
        if arg is None:
            self._console.print(f"  断点模式: [yellow]{self._bp_mode}[/]")
            return
        mode = arg.lower()
        if mode not in ("sync", "async"):
            self._err(f"无效模式: {arg} (应为 sync 或 async)")
            return
        self._bp_mode = mode
        self._hart_paused.clear()
        self._console.print(f"  断点模式 → [yellow]{mode}[/]")

    def _dispatch_bp(self, rest: list[str]) -> None:
        """bp 命令子分发: 将无歧义的子命令名路由到对应 handler."""
        if not rest:
            self.cmd_bp_list()
            return
        sub = rest[0].lower()
        if sub == "mode":
            self.cmd_bp_mode(rest[1] if len(rest) > 1 else None)
        elif sub == "delete":
            if len(rest) > 1:
                self.cmd_bp_delete(rest[1])
                return
            self._warn("用法: bp delete <编号>")
        elif sub == "clear":
            self.cmd_bp_clear()
        elif sub == "if":
            # 纯条件断点: bp if reg/csr <name> <op> <val>
            if not (len(rest) >= 4 and rest[1] in ("reg", "csr")):
                self._warn("用法: bp if reg/csr <name> <op> <val>")
                return
            ct, cr, co = rest[1], rest[2], rest[3] if len(rest) > 3 else "=="
            cv = int(rest[4], 0) if len(rest) > 4 else 0
            desc = f"if {ct} {cr}{co}0x{cv:x}"
            bp = _Breakpoint(
                kind="cond", value=0,
                desc=desc, cond_type=ct, cond_reg=cr,
                cond_op=co, cond_val=cv,
            )
            self._breakpoints.append(bp)
            self._console.print(
                f"  [green]条件断点 {len(self._breakpoints)}[/]  "
                f"[bold yellow]if[/] {ct} {cr} {co} [yellow]0x{cv:x}[/]"
            )
        elif sub == "opcode":
            if len(rest) <= 1:
                self._warn("用法: bp opcode <hex>")
                return
            self.cmd_bp_opcode(rest[1])
        else:
            self.cmd_bp_set(" ".join(rest))

    def cmd_restart(self) -> None:
        """重新运行当前加载的程序.

        重置所有 hart 状态、CLINT、TLB, 重新加载固件到 RAM,
        清除调试器快照与指令计数.
        """
        emu = self._emu
        cfg = emu._cfg

        # -- 重置 CLINT --
        emu.clint._mtime = 0
        for i in range(cfg.num_harts):
            emu.clint._mtimecmp[i] = 0
            emu.clint._msip[i] = 0

        # -- 重建 harts (保证完全干净的初始状态) --
        emu.harts = []
        for i in range(cfg.num_harts):
            h = Hart(id=i, pmp_entries=cfg.pmp_entries)
            h.pc = cfg.prog_cnt
            inject_memory_backend(h, emu.bus.read, emu.bus.write)
            h.bus = emu.bus
            h.interrupt_ctrl = emu.clint
            emu.harts.append(h)

        # 互引用: mfence.did 广播 + 断点多 hart 检查需要
        for h in emu.harts:
            h.all_harts = emu.harts

        # -- 重新加载固件 (含 PIE 搬迁偏移) --
        if self._image is not None:
            emu.load_firmware(self._image, load_offset=self._load_offset)

        # -- 重新注入 DTB --
        if self._fdt_addr is not None:
            emu.load_dtb(self._fdt_addr)

        # -- 重新注入 preload --
        if self._preload_path is not None:

            preload_entry = Preloader(emu).inject_file(self._preload_path)
            # misa U-bit + fdt_get_address DTB 副本
            for h in emu.harts:
                h.csrs["misa"].val = (2 << 62) | (1 << 18) | (1 << 20)  # MXL=RV64 + S + U
                h.csrs["mscratch"].val = cfg.ram_base + cfg.ram_size - 0x100000
            if self._fdt_addr is not None:
                emu.load_dtb_blob(cfg.ram_base + 0x2200000, emu.build_dtb())
            for h in emu.harts:
                h.pc = preload_entry
        self._snapshot = None
        self._mem_changes = []
        self._instr_count = 0
        self._trap_displayed_mcause = None
        self._disasm_next_addr = None
        self._disasm_ref_pc = None
        self._disasm_base_step = 0
        self._disasm_past_terminator = False
        self._stack_frames = []
        self._current_frame_idx = 0
        self._last_command = None
        self._paused = False
        self._hart_paused.clear()
        self._bp_hit_this_run.clear()
        self._prev_instr_csr_addr = -1

        self._console.print(
            f"[dim]已重启 — {cfg.num_harts} hart(s), "
            f"PC = [yellow]{emu.harts[0].pc:#018x}[/][/]"
        )

    # ==========================================================
    #  运行循环
    # ==========================================================

    _DEFAULT_TIMEOUT = 3600.0  # 默认超时 1 小时

    def _run_loop(
        self,
        cycles: int | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        """运行循环: 每个周期所有 hart 各执行一条指令 (round-robin).

        多 hart 时通过 emu.step() 驱动全部 hart;
        单 hart 时保持 step_one() 以支持快照/回滚.

        Args:
            cycles: 最大周期数, None 表示无限.
            timeout: 墙钟超时秒数, 默认 3600 (1 小时). 设为 0 禁用.
        """
        self._running = True
        self._paused = False
        self._terminated = False
        self._sigint_count = 0
        self._enter_run_mode()

        if timeout is None:
            timeout = self._DEFAULT_TIMEOUT
        deadline = time.monotonic() + timeout if timeout > 0 else None

        multi = self._emu.num_harts > 1


        _instr_start = self._instr_count

        while not self._terminated and not self._paused:
            if cycles is not None and (self._instr_count - _instr_start) >= cycles:
                break
            if all(h._halted for h in self._emu.harts):
                self._warn("所有 Hart 已暂停")
                break
            if self.hart._halted:
                self._show_trap_context(self.hart)
                break
            # async 模式: 当前 hart 被断点暂停时回到 REPL
            if self._bp_mode == "async" and self._hart_id in self._hart_paused:
                self._console.print(f"  [dim]Hart {self._hart_id} 断点暂停, 回到 REPL[/]")
                break
            # 超时检查: 墙钟时间超过限制时终止并转储状态
            if deadline is not None and time.monotonic() >= deadline:
                self._warn(f"运行超时 ({timeout:.0f}s), 强制停止")
                self._show_trap_context(self.hart)
                break

            try:
                if multi:
                    self._emu.step()
                    self._check_multi_hart_bp()
                else:
                    self.step_one()
            except Exception:
                logger.opt(exception=True).error("step 执行异常")
                self._err("指令执行异常, 运行中止")
                self._terminated = True
                break

            self._instr_count += 1

            # 周期性地让出 CPU 时间片, 避免 100% 占用
            if self._instr_count % _YIELD_EVERY == 0:
                time.sleep(_YIELD_SLEEP)

        if self._terminated:
            self._console.print("[dim]模拟循环已终止[/]")

        self._running = False
        self._enter_repl_mode()
        if self._emu.uart is not None:
            self._emu.uart.flush_all()

    # ==========================================================
    #  格式化辅助
    # ==========================================================

    @staticmethod
    def _hex(v: int) -> str:
        return f"0x{(v & 0xFFFF_FFFF_FFFF_FFFF):016x}"

    @staticmethod
    def _fmt_size(n: int) -> str:
        """human-readable 尺寸, 超出最大单位时落到最远单位."""
        for name, threshold in _SIZE_UNITS:  # PB → TB → ... → KB (从大到小)
            if n >= threshold:
                return f"{n / threshold:.1f} {name}"
        return f"{n} B"

    @staticmethod
    def _fmt_instr_count(n: int) -> str:
        """将指令数格式化为 human-readable: B(十亿) / M(百万) / K(千).

        精度等比例保持: K 保留 3 位小数 (0.001K=1条),
        M 保留 6 位小数 (0.000001M=1条), B 保留 6 位小数.
        """
        if n < 1000:
            return str(n)
        if n < 1_000_000:
            return f"{n / 1000:.3f}K"
        if n < 1_000_000_000:
            return f"{n / 1_000_000:.6f}M"
        return f"{n / 1_000_000_000:.7f}B"

    @staticmethod
    def _trap_cause_name(mcause_val: int) -> str:
        """将 mcause 寄存器值翻译为可读的 trap 类型名称 (委托 trap.py)."""
        return trap_cause_name(mcause_val)

    def _show_trap_context(self, h) -> None:
        """hart 进入不可恢复陷态时的上下文摘要."""
        cause = h.mcause_val
        name = trap_cause_name(cause)
        is_int = (cause >> 63) & 1
        tag = "中断" if is_int else "异常"
        self._warn(f"Hart {h.id} 进入不可恢复陷态, 已暂停")
        self._console.print(
            f"  [red bold]{tag}[/] {name}  "
            f"mcause=0x{cause:x}  mepc={self._hex(h.mepc_val)}  mtval={self._hex(h.mtval_val)}"
        )

    def _check_rv64_addr(self, v: int, what: str = "地址") -> bool:
        """校验 *v* 是否在 RV64 物理地址范围 [0, 2^64) 内.

        返回 True 表示合法; False 表示非法 (已输出错误).
        """
        if v < 0 or v >= (1 << 64):
            self._err(f"{what} {v} 超出 RV64 范围 [0, 2^64)")
            return False
        return True

    def _warn(self, msg: str) -> None:
        self._console.print(f"[yellow]警告:[/] {msg}")

    def _err(self, msg: str) -> None:
        self._console.print(f"[red bold]错误:[/] {msg}")

    # ==========================================================
    #  状态显示命令
    # ==========================================================

    def cmd_step(self, count: int = 1) -> None:
        """单步或多步执行.

        step       — 执行 1 条指令, 显示 PC 及反汇编
        step <n>   — 执行 n 条指令, 仅显示最终状态

        若途中 hart 进入不可恢复陷态, 自动停机并转储状态.
        """
        if count <= 0:
            self._err("步数须 > 0")
            return
        self._paused = False
        self._hart_paused.clear()
        for _ in range(count):
            self.step_one()
            if not self.hart._halted:
                continue
            self._show_trap_context(self.hart)
            return
        self.cmd_pc()

    def cmd_continue(self) -> None:
        self._console.print("[dim]继续执行 (Ctrl+C 暂停)...[/]")
        self._run_loop()

    def cmd_run(self, n: int = 1) -> None:
        self._console.print(f"[dim]执行 {n} 条指令...[/]")
        self._run_loop(cycles=n)

    def cmd_watch(self, addr_str: str = "", size_str: str = "8") -> None:
        """设置写监控: watch <addr> [size] — 关闭: watch off.

        当 CPU 写入被监控地址范围时自动暂停并报告写入者 PC.
        """
        if not addr_str or addr_str in ("off", "none", "clear"):
            self._watch_ranges.clear()
            self._watch_hit = None
            self._console.print("[dim]写监控已关闭[/]")
            return
        addr = self._resolve_addr(addr_str)
        if addr is None:
            self._err(f"无法解析地址: {addr_str}")
            return
        size = int(size_str, 0)
        self._watch_ranges.append((addr, addr + size))
        self._console.print(
            f"  [green]● 写监控[/] 0x{self._hex(addr)}–0x{self._hex(addr + size)}"
        )

    def cmd_regs(self) -> None:
        h = self.hart
        tbl = Table(title=f"Hart {self._hart_id}  GPRs", border_style="blue")
        tbl.add_column("Reg", style="cyan", no_wrap=True)
        tbl.add_column("Value", style="green")
        tbl.add_column("Reg", style="cyan", no_wrap=True)
        tbl.add_column("Value", style="green")
        for i in range(16):
            lo_name, lo_alias = gpr_name(i), gpr_alias(i)
            hi_name, hi_alias = gpr_name(i + 16), gpr_alias(i + 16)
            lo_val, hi_val = h.gprs[i], h.gprs[i + 16]
            tbl.add_row(
                f"{lo_name} ({lo_alias})", self._hex(lo_val),
                f"{hi_name} ({hi_alias})", self._hex(hi_val),
            )
        self._console.print(tbl)

    def cmd_reg(self, raw_args: str) -> None:
        """读取 GPR: reg <name>  或  reg <n1>, <n2>, ... (逗号分隔多寄存器)."""
        h = self.hart
        wanted = [n.strip() for n in raw_args.split(",") if n.strip()]
        if not wanted:
            return

        # 验证
        idxs = []
        for n in wanted:
            idx = self._find_gpr(n)
            if idx is None:
                self._err(f"未知寄存器: {n}")
                return
            idxs.append(idx)
        # 对齐: 名称宽度 + 别名宽度
        max_w = max(len(gpr_name(idx)) for idx in idxs)
        max_a = max(len(gpr_alias(idx)) for idx in idxs)
        lines = [
            f"  [cyan]{gpr_name(idx):<{max_w}}[/] "
            f"([dim]{gpr_alias(idx):<{max_a}}[/]) = "
            f"[green]{self._hex(h.gprs[idx])}[/]" for idx in idxs
        ]
        self._console.print("\n".join(lines))

    @seize_val_err("无效值")
    def cmd_set(self, name: str, value: str) -> None:
        """写入 GPR: set <name> <value>."""
        h = self.hart
        idx = self._find_gpr(name)
        if idx is None:
            self._err(f"未知寄存器: {name}")
            return
        v = int(value, 0) & 0xFFFF_FFFF_FFFF_FFFF
        old = h.gprs[idx]
        h.gprs[idx] = v
        self._console.print(
            f"{gpr_name(idx)} ([cyan]{gpr_alias(idx)}[/]): "
            f"[yellow]{self._hex(old)}[/] → [green]{self._hex(v)}[/]"
        )

    def _resolve_addr(self, arg: str) -> int | None:
        """将字符串解析为地址: pc / 寄存器名 / CSR 名 / 数值.

        Returns:
            解析出的 64-bit 地址, 或 None (无法解析 / 寄存器不存在).
        """
        if arg.lower() == "pc":
            return self.hart.pc
        # 尝试 GPR 名
        idx = self._find_gpr(arg)
        if idx is not None:
            return self.hart.gprs[idx] & 0xFFFF_FFFF_FFFF_FFFF
        # 尝试 CSR 名 (mepc, sepc, stval, mtval 等)
        v = self._read_csr_by_name(arg)
        if v != 0 or csr_addr_from_name(arg) is not None:
            return v
        # 尝试数值
        try:
            v = int(arg, 0)
        except ValueError:
            return None
        if v < 0 or v >= (1 << 64):
            return None
        return v


    def _try_read_va(self, va: int, size: int) -> bytes | None:
        """从虚拟地址读取内存 (自动 VA→PA 翻译).

        当 satp 启用 MMU 时, 逐页翻译 VA 后读取物理 RAM;
        Bare 模式下直接作为物理地址读取.
        """
        if size <= 0 or size > 4096:
            return None
        hart = self.hart
        # Bare 模式或 M 模式 — 直接物理读取
        # (RISC-V 规范: M 模式始终使用 Bare 翻译, 无视 satp.MODE)
        if hart.mmu_mode == 0 or hart.mode == RiscvMode.M:
            return self._emu.bus.try_read(va, size)
        # MMU 使能 — 逐页翻译
        result = bytearray()
        remain = size
        cur = va
        while remain > 0:
            page_off = cur & 0xFFF
            chunk = min(remain, 0x1000 - page_off)
            ok, pa = translate_addr(hart, cur)
            if not ok:
                return None
            data = self._emu.bus.try_read(pa, chunk)
            if data is None:
                return None
            result.extend(data)
            cur += chunk
            remain -= chunk
        return bytes(result)

    def _try_read_va_forced(self, va: int, size: int) -> tuple[int, bytes] | None:
        """强制经 MMU 翻译读取虚拟地址, 无视当前特权级.

        用于 vdisasm: 即使 hart 在 M 模式, 也使用 satp 页表将 VA 翻译为 PA.
        返回 ``(physical_addr, data)`` 或 None (翻译失败/读失败).
        """
        if size <= 0 or size > 4096:
            return None
        hart = self.hart
        if hart.mmu_mode == 0 or hart._mem_read_phy is None:
            return None

        result = bytearray()
        cur = va
        remain = size
        first_pa: int | None = None
        root_ppn = satp_root_ppn(hart.satp_val)

        while remain > 0:
            page_off = cur & 0xFFF
            chunk = min(remain, 0x1000 - page_off)
            ok, pa, _level, _flags = sv39_walk(root_ppn, cur, hart._mem_read_phy)
            if not ok:
                return None
            if first_pa is None:
                first_pa = pa
            data = self._emu.bus.try_read(pa, chunk)
            if data is None:
                return None
            result.extend(data)
            cur += chunk
            remain -= chunk
        assert first_pa is not None  # while 至少执行一次, first_pa 已赋值
        return first_pa, bytes(result)

    @seize_val_err("无效的gpr索引值")
    def _find_gpr(self, name: str) -> int | None:
        """按 xN 或 ABI 名查找 GPR 索引 (0-31)."""
        if name.startswith("x"):
            idx = int(name[1:])
            if 0 <= idx <= 31:
                return idx
        # 按 ABI 名查找
        for i in range(32):
            if gpr_alias(i).lower() == name.lower():
                return i
        return None

    def cmd_csr(self, raw_args: str) -> None:
        """读取 CSR.

        csr <name>                  — 单个
        csr <name1>, <name2>, ...   — 多个 (逗号分隔), 对齐显示
        csr list                    — 列出所有可用 CSR 名称
        """
        h = self.hart

        # "list" 子命令
        if raw_args == "list":
            names = sorted(h.csrs.keys())
            self._console.print(f"可用 CSR: [dim]{', '.join(names)}[/]")
            return

        # 按逗号拆分, 过滤空白
        wanted = [n.strip() for n in raw_args.split(",") if n.strip()]
        if not wanted:
            return

        # 验证
        for n in wanted:
            if n in h.csrs:
                continue
            self._err(f"未知 CSR: {n} (用 'csr list' 查看可用列表)")
            return

        # 对齐宽度 = 最长名称
        max_w = max(len(n) for n in wanted)

        lines: list[str] = []
        for n in wanted:
            csr = h.csrs[n]
            line = f"  [cyan]{n:<{max_w}}[/] = [green]{self._hex(csr.val)}[/] (dec: {csr.val})"
            # mstatus / mstatush 附加当前特权级
            if n == "mstatus":
                line += f"  [dim]模式 [bold]{h.mode.name}[/][/]"
            elif n == "mstatush":
                line += f"  [dim]模式 [bold]{h.mode.name}[/][/]"
            # mcause / scause 解码陷态原因 (0 = 无异常)
            elif n in ("mcause", "scause"):
                payload = "  [dim](无)[/]"
                if csr.val != 0:
                    payload = f"  [dim]{self._trap_cause_name(csr.val)}[/]"
                line += payload
            lines.append(line)
        self._console.print("\n".join(lines))

    @seize_val_err("无效值")
    def cmd_csrw(self, name: str, value: str) -> None:
        """写入 CSR: csrw <name> <value>."""
        h = self.hart
        if name not in h.csrs:
            self._err(f"未知 CSR: {name} (用 'csr list' 查看可用列表)")
            return
        v = int(value, 0) & 0xFFFF_FFFF_FFFF_FFFF
        csr = h.csrs[name]
        old, csr.val = csr.val, v
        self._console.print(
            f"[cyan]{name}[/]: [yellow]{self._hex(old)}[/] → [green]{self._hex(v)}[/]"
        )

    def _fetch_and_disasm(self, pc: int) -> tuple[str, str] | None:
        """读取 PC 处指令字并反汇编, 返回 (raw_hex, asm); IO 失败返回 None.
        当 satp 启用 MMU 时, 经 VA→PA 翻译读取; Bare 模式直接物理读取."""
        raw = self._try_read_va(pc, 4)
        if raw is None:
            return None
        instr = int.from_bytes(raw, "little", signed=False)
        asm = disasm(instr, pc)
        # 压缩指令显示 4 位十六进制, 32-bit 显示 8 位
        # instr 从 4 字节读取, 压缩指令需掩码到低 16 位以免混入后续指令字节
        raw_hex = f"{instr:08x}"
        if (instr & 0x3) != 0x3:
            raw_hex = f"{instr & 0xFFFF:04x}"
        return raw_hex, asm

    def _warn_pc_if_suspect(self, h, v: int) -> None:
        """根据 MMU 模式检查 PC 并给出潜在无效地址的软警告.

        Bare 模式: v 是物理地址, 检查是否在有效 RAM/设备范围.
        Sv39 模式: v 是虚拟地址, 检查 bits[63:39] 是否等于 bit 38.
        """
        if h.mmu_mode == 0 and h._bus is not None and not h._bus.is_valid_addr(v):  # Bare — 物理地址
            self._warn(
                f"Bare 模式下 PC 0x{v:016x} 不在有效物理地址范围 "
                f"[0x{h._bus.ram_base:x}, 0x{h._bus._ram_end:x}) 或已注册设备区域"
            )
        elif h.mmu_mode == 8:  # Sv39 — 虚拟地址
            if sv39_canonical_va(v) is not None:
                return
            self._warn(
                f"Sv39 模式下 VA 0x{v:016x} bits[63:39] 不等于 bit 38 — "
                f"此地址在页表遍历时将触发缺页异常"
            )

    @seize_val_err("无效地址")
    def cmd_pc(self, addr: str | None = None) -> None:
        """读/设 PC: pc (读取并反汇编), pc <addr> (设置).

        若当前 mcause 非零 (存在待处理的 trap), 一并显示
        mepc / mcause / mtval 等信息.
        """
        h = self.hart
        if addr is not None:
            v = self._resolve_addr(addr)
            if v is None:
                self._err(f"无法解析地址: {addr}")
                return
            if not self._check_rv64_addr(v, "PC"):
                return

            # 软警告: 根据当前 MMU 模式提示可能的无效地址
            self._warn_pc_if_suspect(h, v)

            old, h.pc = h.pc, v
            self._snapshot = None
            self._mem_changes = []
            self._console.print(
                f"PC: [yellow]{self._hex(old)}[/] → [green]{self._hex(h.pc)}[/]"
            )
        pc = h.pc
        result = self._fetch_and_disasm(pc)
        raw_hex, asm = "(无法读取)", "(无法解码)"
        if result is not None:
            raw_hex, asm = result
        self._console.print(
            f"PC  = [bold yellow]{self._hex(pc)}[/]  "
            f"[dim]模式 [bold cyan]{h.mode.name}[/][/]"
            + ("  [dim]WFI等待[/]" if h._waiting else "")
            + f"\nRaw = [bright_black]{raw_hex}[/]"
            + f"\n[bold green]  {asm}[/]"
        )

        # 仅在 mcause 刚发生变化时展示陷态上下文 (避免 handler 执行期间重复刷屏)
        mcause = h.mcause_val
        if mcause == 0:
            self._trap_displayed_mcause = None
            return
        if mcause == self._trap_displayed_mcause:
            return
        self._trap_displayed_mcause = mcause

        is_intr = (mcause >> 63) & 1
        _mstatus = h.mstatus_val
        _mcause_name = self._trap_cause_name(mcause)
        _scause = h.csrs["scause"].val
        _sstatus = h.csrs["sstatus"].val
        _hex = self._hex

        # 按活跃模式设定列样式: 活跃侧正常, 非活跃侧 dim
        if h.mode in set((RiscvMode.M, RiscvMode.D)):  # M / D 模式 — M 侧亮, S 侧 dim
            m_style, s_style = "bold cyan", "dim"
            stale_note = ""
        elif h.mode.value == 1:  # S 模式 — S 侧亮, M 侧 dim
            m_style, s_style = "dim", "bold cyan"
            # 若当前 S 模式没有活跃 trap (scause==0), 则 M 侧值来自旧 trap
            stale_note = (
                "" if _scause != 0
                else " [dim](M-mode 陈旧)[/]"
            )
        else:  # U 模式
            m_style, s_style = "cyan", "cyan"
            stale_note = " [dim](陈旧)[/]"

        tbl = Table(
            title=(
                f"Hart {self._hart_id}  Trap 上下文"
                f" ({'中断' if is_intr else '异常'}){stale_note}"
            ),
            border_style="red",
            show_header=True,
        )
        tbl.add_column("M-mode", style=m_style, justify="left")
        tbl.add_column("S-mode", style=s_style, justify="left")

        tbl.add_row(
            f"mcause  = {_hex(mcause)}",
            f"scause  = {_hex(_scause)}",
        )
        tbl.add_row(
            f"mepc    = {_hex(h.mepc_val)}",
            f"sepc    = {_hex(h.csrs['sepc'].val)}",
        )
        tbl.add_row(
            f"mtval   = {_hex(h.mtval_val)}",
            f"stval   = {_hex(h.csrs['stval'].val)}",
        )
        tbl.add_row(
            f"mstatus = {_hex(_mstatus)}",
            f"sstatus = {_hex(_sstatus)}",
        )
        self._console.print(tbl)

        # 注解行: cause 名称 + 关键 status 位
        m_anno = (
            f"[cyan]{_mcause_name}[/]  "
            f"MIE={(_mstatus >> 3) & 1} MPP={(_mstatus >> 11) & 3}"
        )
        s_anno = ""
        if _scause != 0:
            _scause_name = self._trap_cause_name(_scause)
            s_anno = (
                f"[cyan]{_scause_name}[/]  "
                f"SIE={(_sstatus >> 1) & 1} SPP={(_sstatus >> 8) & 1}"
            )
        elif _sstatus != 0:
            s_anno = (
                f"SIE={(_sstatus >> 1) & 1} "
                f"SPP={(_sstatus >> 8) & 1}"
            )
        self._console.print(f"  {m_anno}    {s_anno}")

    def cmd_mode(self) -> None:
        h = self.hart
        self._console.print(f"Mode = [bold cyan]{h.mode.name}[/] ({h.mode.value})")

    def cmd_mstatus(self) -> None:
        h = self.hart
        v = h.mstatus_val
        fields = [
            ("MIE", (v >> 3) & 1),
            ("MPIE", (v >> 7) & 1),
            ("MPP", (v >> 11) & 0b11),
            ("SIE", (v >> 1) & 1),
            ("SPIE", (v >> 5) & 1),
            ("SPP", (v >> 8) & 1),
            ("MPRV", (v >> 17) & 1),
            ("SUM", (v >> 18) & 1),
            ("MXR", (v >> 19) & 1),
            ("TVM", (v >> 20) & 1),
            ("TW", (v >> 21) & 1),
            ("TSR", (v >> 22) & 1),
            ("FS", (v >> 13) & 0b11),
            ("SD", (v >> 63) & 1),
        ]
        tbl = Table(title="mstatus 内部字段情况", border_style="magenta")
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Value", style="green")
        tbl.add_row("Hex", self._hex(v))
        tbl.add_section()
        for fname, fval in fields:
            tbl.add_row(fname, str(fval))
        self._console.print(tbl)

    def cmd_mcause(self) -> None:
        """MCAUSE CSR: 中断位 + 异常码分解."""
        self._show_cause("mcause", self.hart.mcause_val)

    def cmd_scause(self) -> None:
        """SCAUSE CSR: 中断位 + 异常码分解."""
        self._show_cause("scause", self.hart.scause_val)

    def _show_cause(self, name: str, v: int) -> None:
        is_irq = (v >> 63) & 1
        code = v & 0x7FFF_FFFF_FFFF_FFFF
        cause_name = trap_cause_name(v)

        tbl = Table(title=f"{name} 内部字段", border_style="magenta")
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Value", style="green")
        tbl.add_column("说明", style="dim")
        tbl.add_row("Hex", self._hex(v), "")
        tbl.add_section()
        tbl.add_row("Interrupt", str(is_irq), "中断" if is_irq else "异常")
        tbl.add_row("Code", str(code), cause_name)
        self._console.print(tbl)

    def cmd_mtvec(self) -> None:
        """MTVEC CSR: BASE + MODE 分解."""
        self._show_tvec("mtvec", self.hart.mtvec_val)

    def cmd_stvec(self) -> None:
        """STVEC CSR: BASE + MODE 分解."""
        self._show_tvec("stvec", self.hart.stvec_val)

    def _show_tvec(self, name: str, v: int) -> None:
        mode = v & 0b11
        base = v & ~0b11
        mode_names = {0: "Direct (直接)", 1: "Vectored (向量)"}

        tbl = Table(title=f"{name} 内部字段", border_style="magenta")
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Value", style="green")
        tbl.add_column("说明", style="dim")
        tbl.add_row("Hex", self._hex(v), "")
        tbl.add_section()
        tbl.add_row("BASE", self._hex(base), f"陷态向量基址 (0x{base:016X})")
        tbl.add_row("MODE", str(mode), mode_names.get(mode, f"保留({mode})"))
        self._console.print(tbl)

    def cmd_mip(self) -> None:
        """MIP CSR: 中断挂起位分解."""
        self._show_ip("mip", self.hart.csrs["mip"].val)

    def cmd_mie(self) -> None:
        """MIE CSR: 中断使能位分解."""
        self._show_ip("mie", self.hart.csrs["mie"].val)

    def cmd_sip(self) -> None:
        """SIP CSR: S 模式中断挂起位分解."""
        self._show_ip("sip", self.hart.csrs["sip"].val)

    def cmd_sie(self) -> None:
        """SIE CSR: S 模式中断使能位分解."""
        self._show_ip("sie", self.hart.csrs["sie"].val)

    @staticmethod
    def _ip_bits() -> list[tuple[str, int, str]]:
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

    def _show_ip(self, name: str, v: int) -> None:
        tbl = Table(title=f"{name} 内部字段", border_style="magenta")
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Bit", style="yellow", justify="center")
        tbl.add_column("Value", style="green", justify="center")
        tbl.add_column("说明", style="dim")
        tbl.add_row("Hex", "", self._hex(v), "")
        tbl.add_section()
        for bit_name, bit, desc in self._ip_bits():
            val = (v >> bit) & 1
            tbl.add_row(bit_name, str(bit), str(val), desc)
        self._console.print(tbl)

    def cmd_medeleg(self) -> None:
        """MEDELEG CSR: 异常委派位分解."""
        self._show_deleg("medeleg", self.hart.csrs["medeleg"].val, self._EXC_NAMES)

    def cmd_mideleg(self) -> None:
        """MIDELEG CSR: 中断委派位分解."""
        self._show_deleg("mideleg", self.hart.csrs["mideleg"].val, self._IRQ_NAMES)

    def _show_deleg(self, name: str, v: int, names: dict[int, str]) -> None:
        tbl = Table(
            title=f"{name} 内部字段 (置位 = 委派到 S 模式)", border_style="magenta"
        )
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Bit", style="yellow", justify="center")
        tbl.add_column("Val", style="green", justify="center")
        tbl.add_column("说明", style="dim")
        tbl.add_row("Hex", "", self._hex(v), "")
        tbl.add_section()
        for bit in sorted(names.keys()):
            val = (v >> bit) & 1
            tbl.add_row(f"bit{bit:02d}", str(bit), str(val), names[bit])
        self._console.print(tbl)

    # 异常/中断名称表 (供 medeleg/mideleg 使用)
    _EXC_NAMES: dict[int, str] = {
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
    _IRQ_NAMES: dict[int, str] = {
        1: "SSIP (S 模式软件中断)",
        3: "MSIP (M 模式软件中断)",
        5: "STIP (S 模式定时器中断)",
        7: "MTIP (M 模式定时器中断)",
        9: "SEIP (S 模式外部中断)",
        11: "MEIP (M 模式外部中断)",
    }

    # ----------------------------------------------------------
    #  TLB helpers
    # ----------------------------------------------------------

    @staticmethod
    def _tlb_page_size(level: int) -> str:
        """TLB level → 页大小标签."""
        return {0: "4K", 1: "2M", 2: "1G"}.get(level, f"Lv{level}")

    def _show_tlb(self, name: str, tlb) -> None:
        """显示单个 TLB 的全部有效条目及命中率."""
        total = tlb._hits + tlb._misses
        rate = tlb._hits / total if total > 0 else 0.0

        title = (
            f"[bold]{name}[/]: {len(tlb)}/{tlb.size} entries, "
            f"hits={tlb._hits}, misses={tlb._misses}, "
            f"rate=[{'green' if rate > 0.9 else 'yellow'}]{rate:.2%}[/]"
        )

        if len(tlb) == 0:
            hint = ""
            if total == 0 and self.hart.mmu_mode == 0:
                hint = "\n[dim]# MMU 处于 Bare 模式 — TLB 不会被填充[/]"
            self._console.print(f"{title}\n  (empty){hint}")
            return

        tbl = Table(title=title, border_style="blue")
        tbl.add_column("VPN", style="cyan")
        tbl.add_column("PPN", style="green")
        tbl.add_column("Perm", style="yellow")
        tbl.add_column("Size")
        for e in tlb.entries:
            if not e.valid:
                continue
            tbl.add_row(
                f"0x{e.tag:09x}",
                f"0x{e.ppn:09x}",
                decode_tlb_perm(e.perm),
                self._tlb_page_size(e.level),
            )
        self._console.print(tbl)

    def _tlb_search(self, vpn: int) -> None:
        """在 ITLB / DTLB 中查找指定 VPN 并输出结果."""
        h = self.hart
        self._console.print(f"TLB 查找 [bold]VPN=0x{vpn:09x}[/]:")
        for name, tlb in [("ITLB", h.itlb), ("DTLB", h.dtlb)]:
            found = False
            for e in tlb.entries:
                if not e.valid or e.tag != vpn:
                    continue
                self._console.print(
                    f"  [[cyan]{name}[/]] VPN=0x{e.tag:09x} → PPN=0x{e.ppn:09x}  "
                    f"perm={decode_tlb_perm(e.perm)}  "
                    f"size={self._tlb_page_size(e.level)}"
                )
                found = True
            if not found:
                self._console.print(f"  [[cyan]{name}[/]] [dim]未命中[/]")

    @seize_val_err("无效 VPN")
    def cmd_tlb(self, arg: str | None = None) -> None:
        """显示或查找 TLB 条目.

        tlb         — 显示全部 ITLB / DTLB
        tlb <vpn>   — 查找指定 VPN
        """
        if arg is not None:
            vpn = int(arg, 0)
            if not self._check_rv64_addr(vpn, "VPN"):
                return
            self._tlb_search(vpn)
            return
        h = self.hart
        self._show_tlb("ITLB", h.itlb)
        self._show_tlb("DTLB", h.dtlb)

    @seize_val_err("无效 VPN")
    def cmd_tlbflush(self, vpn_str: str | None = None) -> None:
        """刷新 TLB: tlbflush [vpn] (无参数 = 全部)."""
        h = self.hart
        if vpn_str is None:
            h.itlb.flush_all()
            h.dtlb.flush_all()
            self._console.print("[dim]已刷新全部 ITLB + DTLB[/]")
            return
        vpn = int(vpn_str, 0)
        if not self._check_rv64_addr(vpn, "VPN"):
            return
        h.itlb.flush(vpn)
        h.dtlb.flush(vpn)
        self._console.print(f"[dim]已刷新 ITLB + DTLB 中 VPN=0x{vpn:09x}[/]")

    @staticmethod
    def _hexdump_bytes(data: bytes, indent: str = "") -> str:
        """字节数据 → hexdump 多行字符串."""
        rows = []
        for off in range(0, len(data), 16):
            chunk = data[off : off + 16]
            hex_s = " ".join(f"{b:02x}" for b in chunk)
            asc_s = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            rows.append(f"{indent}{off:04x}  {hex_s:<48s}  |{asc_s}|")
        return "\n".join(rows)

    def _fmt_cache_line(self, e, set_idx: int, way_idx: int, full: bool) -> str:
        """格式化单条 L2 缓存行: 元数据 + 可选完整 hexdump."""
        meta = (
            f"  s={set_idx:3d} w={way_idx}  tag=0x{e.tag:09x}  "
            f"st={e.mesi.value:<1}  dirty={e.dirty!s:<5}  "
            f"last={e.last_access}"
        )
        if full:
            return meta + "\n" + self._hexdump_bytes(bytes(e.data), indent="    ")
        preview = bytes(e.data[:16]).hex(" ")
        return meta + f"  data[:16]={preview}"

    _DEFAULT_CACHE_LINES = 64

    @seize_val_err("set/way 索引需为整数")
    def cmd_cache(self, arg: str | None = None) -> None:
        """显示 L2 缓存状态.

        cache               — 概览 + 前 64 条 valid 行 (预览)
        cache <set>         — 指定 set 全部 valid 行 + 完整 hexdump
        cache <set> <way>   — 指定 set/way, 含完整 64 B hexdump
        cache <start>-<end> — set 范围, 预览模式
        """
        l2 = self._emu.bus.l2
        if not isinstance(l2, L2Cache):
            self._console.print("[dim]L2 缓存未启用[/]")
            return

        entries, ways, num_sets = l2.entries, l2.ways, l2.num_sets

        # ---- 解析参数 ----
        target_set, target_way = None, None
        range_start, range_end = None, None  # 闭区间

        if arg is not None:
            parts = arg.split()
            first = parts[0]
            # 范围: <start>-<end>
            if "-" in first and not first.startswith("-"):
                st, _, ed = first.partition("-")
                if not (0 <= int(st, 0) <= int(ed, 0) < num_sets):
                    self._err(f"set 范围需在 [0, {num_sets - 1}] 内")
                    return
            else:
                target_set = int(first, 0)
                if not (0 <= target_set < num_sets):
                    self._err(f"set 索引超出范围 [0, {num_sets - 1}]")
                    return
                if len(parts) > 1:
                    target_way = int(parts[1], 0)
                    if not (0 <= target_way < ways):
                        self._err(f"way 索引超出范围 [0, {ways - 1}]")
                        return

        # ---- 统计 ----
        mesi_counts = Counter()
        valid_count, dirty_count = 0, 0
        for e in entries:
            if not e.valid:
                continue
            valid_count += 1
            mesi_counts[e.mesi.name] += 1
            dirty_count += 1 if e.dirty else 0

        header = (
            f"[bold]L2 Cache[/]: {valid_count}/{len(entries)} valid, "
            f"{dirty_count} dirty, "
            f"{ways}-way * {num_sets} sets, "
            f"line={l2.line_size} B, "
            f"hit_rate={l2.hit_rate:.3f}\n"
            "MESI: " + " ".join(
                f"{s}={mesi_counts.get(s, 0)}"
                for s in ("MODIFIED", "EXCLUSIVE", "SHARED", "INVALID")
            )
        )

        # ---- 确定要扫描的 set 范围 ----
        set_range = range(num_sets)
        full_dump = False  # 默认模式: 紧凑预览
        limit = self._DEFAULT_CACHE_LINES
        if range_start is not None and range_end is not None:
            set_range = range(range_start, range_end + 1)
            full_dump = False  # 范围模式: 紧凑预览
            limit = None  # 不限条目数
        elif target_set is not None:
            set_range = [target_set]
            full_dump = True  # 单 set 模式: 完整 hexdump
            limit = None


        # ---- 收集行 ----
        lines: list[str] = []
        for set_idx in set_range:
            for way_idx in range(ways):
                if target_way is not None and way_idx != target_way:
                    continue
                e = entries[set_idx * ways + way_idx]
                if not e.valid:
                    continue
                lines.append(self._fmt_cache_line(e, set_idx, way_idx, full_dump))
                if limit is not None and len(lines) >= limit:
                    break
            if limit is not None and len(lines) >= limit:
                break

        # 截断提示: 仅在设置了上限且确实有更多条目时显示
        if limit is not None and len(lines) >= limit and valid_count > len(lines):
            lines.append(
                f"[dim]... 还有 {valid_count - len(lines)} 条 valid 行, "
                f"用 'cache <start>-<end>' 查看范围[/]"
            )

        if lines:
            self._console.print(header + "\n" + "\n".join(lines))
        else:
            self._console.print(header + "\n[dim](无 valid 行)[/]")

    # ----------------------------------------------------------
    #  PMP 保护范围
    # ----------------------------------------------------------

    @staticmethod
    def _decode_pmp_cfg_byte(cfg_val: int, entry_idx: int) -> int:
        """从 RV64 pmpcfgN 值中提取第 *entry_idx* 个条目的 8-bit 配置."""
        shift = (entry_idx & 0x7) * 8
        return (cfg_val >> shift) & 0xFF

    def cmd_pmp(self) -> None:
        """显示所有 PMP 条目的保护地址范围与权限 (含原始寄存器值)."""
        h = self.hart
        num = h._pmp.num_entries if h._pmp is not None else 0

        if num == 0:
            self._console.print("[dim]PMP 未配置 (num_entries=0)[/]")
            return

        mode_names = {
            PMP_A_OFF: "OFF",
            PMP_A_TOR: "TOR",
            PMP_A_NA4: "NA4",
            PMP_A_NAPOT: "NAPOT",
        }

        tbl = Table(
            title=f"Hart {self._hart_id}  PMP 条目 ({num} total)",
            border_style="blue",
        )
        tbl.add_column("#", style="cyan", justify="right")
        tbl.add_column("L", style="magenta", width=2)
        tbl.add_column("R", style="green", width=2)
        tbl.add_column("W", style="yellow", width=2)
        tbl.add_column("X", style="red", width=2)
        tbl.add_column("Mode", style="cyan", width=6)
        tbl.add_column("Base", style="green")
        tbl.add_column("End", style="yellow")
        tbl.add_column("Size")
        tbl.add_column("pmpaddr", style="dim yellow")
        tbl.add_column("cfg", style="dim cyan", width=5)

        active_count = 0

        for i in range(num):
            # RV64: pmpcfg0 存条目 0-7, pmpcfg2 存 8-15, ...
            cfg_reg_idx = (i // 8) * 2
            reg_name = f"pmpcfg{cfg_reg_idx}"
            cfg_val = h.csrs[reg_name].val if reg_name in h.csrs else 0
            cfg = self._decode_pmp_cfg_byte(cfg_val, i)

            a_mode = cfg & PMP_A_MASK
            addr_name = f"pmpaddr{i}"
            addr_field = (
                h.csrs[addr_name].val if addr_name in h.csrs else 0
            ) & 0xFFFF_FFFF_FFFF_FFFF

            # 解码地址范围
            base, size = 0, 0
            if a_mode == PMP_A_TOR:
                prev = (
                    h.csrs[f"pmpaddr{i - 1}"].val if i > 0 and f"pmpaddr{i - 1}" in h.csrs else 0
                ) & 0xFFFF_FFFF_FFFF_FFFF
                lo = 0 if i == 0 else (prev << 2)
                hi = addr_field << 2
                base, size = lo, (hi - lo) & 0xFFFF_FFFF_FFFF_FFFF
            elif a_mode == PMP_A_NA4:
                base = addr_field << 2
                size = 4
            elif a_mode == PMP_A_NAPOT:
                base, size = decode_napot(addr_field)

            locked = "l" if cfg & PMP_L else "-"
            r = "r" if cfg & PMP_R else "-"
            w = "w" if cfg & PMP_W else "-"
            x = "x" if cfg & PMP_X else "-"
            mode = mode_names.get(a_mode, f"?{a_mode >> 3}?")
            addr_s = f"0x{addr_field:016x}"
            cfg_s = f"0x{cfg:02x}"

            base_s, end_s, size_s = "-", "-", "-"
            if a_mode != PMP_A_OFF:
                active_count += 1
                base_s = f"0x{base:016x}"
                end_s = "0x0000000000000000"
                size_s = "-"
                if size > 0:
                    end_s = f"0x{(base + size) & 0xFFFF_FFFF_FFFF_FFFF:016x}"
                    size_s = self._fmt_size(size)

            tbl.add_row(
                str(i), locked, r, w, x, mode, base_s, end_s, size_s, addr_s, cfg_s,
            )

        self._console.print(tbl)
        if active_count == 0:
            self._console.print("  [dim]所有条目均未激活 (OFF)[/]")

    # ----------------------------------------------------------
    #  satp / MMU
    # ----------------------------------------------------------

    def cmd_satp(self) -> None:
        """显示当前 satp 的 MODE / ASID / PPN 解码."""
        h = self.hart
        mode_names = {0: "Bare", 8: "Sv39", 9: "Sv48", 10: "Sv57"}
        v = h.satp_val
        mode = (v >> 60) & 0xF
        asid = (v >> 44) & 0xFFFF
        ppn = v & ((1 << 44) - 1)
        mn = mode_names.get(mode, f"未知({mode})")
        out = [
            f"satp = [bold]{self._hex(v)}[/]",
            f"  MODE  = [cyan]{mode}[/] ([green]{mn}[/])",
            f"  ASID  = 0x{asid:04x} ({asid})",
            f"  PPN   = 0x{ppn:011x}",
        ]
        if mode != 0:
            out.append(f"  根页表 PA = [yellow]0x{(ppn << 12):016x}[/]")
        self._console.print("\n".join(out))

    @seize_val_err("无效 VA")
    def cmd_pt(self, arg: str | None = None) -> None:
        """页表遍历: 显示指定 VA 经 Sv39 三级页表逐级翻译的完整路径.

        用法: pt [va]
          va 接受 0x...、十进制、或 sepc/mepc 寄存器名. 默认取当前 sepc.
          输出每级 PTE 的 PPN 与权限标志.
        """
        h = self.hart
        if h.mmu_mode == 0:
            self._warn("satp.MODE = Bare, 页表遍历不可用")
            return

        # 解析 VA — 默认 sepc
        if arg is None or arg == "":
            if h.sepc_val != 0:
                arg = f"0x{h.sepc_val:x}"
            elif h.mepc_val != 0:
                arg = f"0x{h.mepc_val:x}"
            else:
                self._warn("没有可用的默认 VA, 请显式指定")
                return
        va_raw = self._resolve_addr(arg)
        if va_raw is None or va_raw < 0 or va_raw > 0xFFFF_FFFF_FFFF_FFFF:
            self._warn(f"无效 VA: {arg}")
            return
        va: int = va_raw

        # satp 解码
        v = h.satp_val
        root_ppn = v & ((1 << 44) - 1)
        root_pa = root_ppn << 12
        mode_s = (v >> 60) & 0xF
        mode_names = {8: "Sv39"}
        mn = mode_names.get(mode_s, f"MODE={mode_s}")

        # VPN 分解
        vpn2, vpn1, vpn0, page_off = sv39_decompose_va(va)

        out: list[str] = [
            f"[bold]VA[/] [cyan]0x{va:016x}[/]",
            f"  VPN[2]={vpn2:#05x}  VPN[1]={vpn1:#05x}  VPN[0]={vpn0:#05x}  "
            f"offset={page_off:#05x}",
            f"  satp = {mn}, root PPN=0x{root_ppn:09x} "
            f"→ PA=0x{root_pa:016x}",
            "",
        ]

        # ---- L1 (根表) ----
        l1_raw = self._emu.bus.try_read(root_pa, 4096)
        if l1_raw is None:
            out.append("[red]根页表不可读[/]")
            self._console.print("\n".join(out))
            return
        l1_pte = int.from_bytes(
            l1_raw[vpn2 * 8:vpn2 * 8 + 8], "little", signed=False
        )
        self._pt_append_level(out, 1, vpn2, l1_pte)
        if not (l1_pte & 1):                      # V=0 → 终止
            self._console.print("\n".join(out))
            return
        if l1_pte & 0xE:                           # 叶子 (1 GiB 超级页)
            l1_pa = ((l1_pte >> 10) & 0xF_FFFF_FFFF) << 12
            final_pa = l1_pa | (va & 0x3FFF_FFFF)
            out.append(f"  最终 PA = [bold yellow]0x{final_pa:016x}[/]")
            self._console.print("\n".join(out))
            return

        # ---- L2 ----
        l2_pa = ((l1_pte >> 10) & 0xF_FFFF_FFFF) << 12
        l2_raw = self._emu.bus.try_read(l2_pa, 4096)
        if l2_raw is None:
            out.append(f"[red]L2 页表 (PA=0x{l2_pa:016x}) 不可读[/]")
            self._console.print("\n".join(out))
            return
        l2_pte = int.from_bytes(
            l2_raw[vpn1 * 8:vpn1 * 8 + 8], "little", signed=False
        )
        self._pt_append_level(out, 2, vpn1, l2_pte)
        if not (l2_pte & 1):
            self._console.print("\n".join(out))
            return
        if l2_pte & 0xE:                           # 叶子 (2 MiB 超级页)
            l2_pa_leaf = ((l2_pte >> 10) & 0xF_FFFF_FFFF) << 12
            final_pa = l2_pa_leaf | (va & 0x1F_FFFF)
            out.append(f"  最终 PA = [bold yellow]0x{final_pa:016x}[/]")
            self._console.print("\n".join(out))
            return

        # ---- L3 ----
        l3_pa = ((l2_pte >> 10) & 0xF_FFFF_FFFF) << 12
        l3_raw = self._emu.bus.try_read(l3_pa, 4096)
        if l3_raw is None:
            out.append(f"[red]L3 页表 (PA=0x{l3_pa:016x}) 不可读[/]")
            self._console.print("\n".join(out))
            return
        l3_pte = int.from_bytes(
            l3_raw[vpn0 * 8:vpn0 * 8 + 8], "little", signed=False
        )
        self._pt_append_level(out, 3, vpn0, l3_pte)
        if not (l3_pte & 1):
            self._console.print("\n".join(out))
            return

        final_pa = (((l3_pte >> 10) & 0xF_FFFF_FFFF) << 12) | page_off
        out.append(f"  最终 PA = [bold yellow]0x{final_pa:016x}[/]")
        self._console.print("\n".join(out))

    def _pt_append_level(
        self, out: list[str], level: int, idx: int, pte: int,
    ) -> None:
        """向 *out* 列表追加一级页表遍历的格式化输出行."""
        flags = pte_flags_str(pte, is_leaf=bool(pte & 0xE))
        ppn = (pte >> 10) & 0xF_FFFF_FFFF
        pa = ppn << 12
        status = (
            "[red]无效[/]"
            if not (pte & 1)
            else flags
        )
        out.append(
            f"  L{level}[{idx:#05x}] = 0x{pte:016x}"
            f"  PPN=0x{ppn:09x}"
            f"  → {status}"
        )
        if pte & 1 and not (pte & 0xE) and level < 3:
            out.append(f"        └─ 下一级页表 PA = 0x{pa:016x}")

    # -- 反汇编着色 ----------------------------------------------------------
    # 寄存器 ABI 名称，用于在 operands 中识别并跳过（不标品红）
    _GPR_ALIASES = {
        "zero", "ra", "sp", "gp", "tp",
        *(f"t{i}" for i in range(7)),    # t0–t6
        *(f"s{i}" for i in range(12)),   # s0–s11
        *(f"a{i}" for i in range(8)),    # a0–a7
    }

    # 分支/跳转 → 绿色
    _BRANCH_JUMP_MNEMONICS = frozenset({
        "beq", "bne", "blt", "bge", "bltu", "bgeu", "jal", "jalr",
        "c.j", "c.jal", "c.jr", "c.jalr", "c.beqz", "c.bnez",
    })
    # FENCE / AMO → 黄色
    _FENCE_AMO_MNEMONICS = frozenset({
        "fence", "fence.i", "sfence.vma",
        "lr.w", "lr.d", "sc.w", "sc.d",
        "amoswap.w", "amoswap.d", "amoadd.w", "amoadd.d",
        "amoxor.w", "amoxor.d", "amoand.w", "amoand.d",
        "amoor.w", "amoor.d", "amomin.w", "amomin.d",
        "amomax.w", "amomax.d", "amominu.w", "amominu.d",
        "amomaxu.w", "amomaxu.d",
    })

    @staticmethod
    def _colorize_asm(asm_text: str) -> str:
        """为反汇编文本加 Rich 颜色标记.

        助记符 (前 7 字符): 分支/跳转→绿, fence/AMO→黄, 其余→白
        operands: 立即数 (十进制/十六进制) → 品红, 寄存器名→保持原色
        """

        # 未知指令 sentinel, 不做着色处理 (避免 Rich 角括号误解释 + 硬拆分)
        if asm_text.startswith("<unknown"):
            return _rich_escape(asm_text)

        mnemonic = asm_text[:7].strip()
        operands = asm_text[7:]

        # 助记符着色
        head = asm_text[:7]
        if mnemonic in Debugger._BRANCH_JUMP_MNEMONICS:
            head = f"[green]{head}[/]"
        elif mnemonic in Debugger._FENCE_AMO_MNEMONICS:
            head = f"[yellow]{head}[/]"

        if not operands:
            return head
        # operands 中的立即数 → 品红.
        # 寄存器保护: x0–x31, f0–f31, ABI 别名.
        # 先用占位符替换寄存器名, 对剩余 token 着色数字, 再还原.
        reg_pattern = re.compile(
            r'\b(?:[xf]\d{1,2}|'
            + '|'.join(re.escape(a) for a in Debugger._GPR_ALIASES)
            + r')\b'
        )
        protected: dict[str, str] = {}
        counter = 0

        def _protect(m: re.Match) -> str:
            nonlocal counter
            key = f"\x00REG{counter}\x00"
            counter += 1
            protected[key] = m.group(0)
            return key

        padded = reg_pattern.sub(_protect, operands)

        # 对受保护后剩余的文本着色数字
        # 匹配: 可选负号, 十进制数 或 0x 十六进制数; 前后需为边界字符
        num_pattern = re.compile(r'(?<=[ ,()])(-?(?:0x[0-9a-fA-F]+|\d+))(?=[ ,()])')
        # 首尾补空格保证边界匹配
        padded = ' ' + padded + ' '
        padded = num_pattern.sub(r'[magenta]\1[/]', padded)
        padded = padded[1:-1]

        # 还原寄存器名
        for key, name in protected.items():
            padded = padded.replace(key, name)

        return head + padded

    @staticmethod
    def _ctrl_flow_kind(instr: int) -> str:
        """返回指令的控制流类型: 'term' (终止), 'branch' (条件分支), 'normal'.

        仅当 *instr* 不是压缩指令时才准确; 压缩指令需先解码.
        """
        opcode = parse_opcode(instr)
        # JAL / JALR — 无条件跳转 (含 call / j / ret / jr)
        if opcode in (0b1101111, 0b1100111):
            return "term"
        # 条件分支
        if opcode == 0b1100011:
            return "branch"
        # SYSTEM — 仅 ecall / ebreak / mret / sret 为终止
        if opcode != 0b1110011:
            return "normal"
        if parse_func3(instr) != 0:
            return "normal"
        if parse_func12(instr) in (0x000, 0x001, 0x302, 0x102):
            return "term"
        return "normal"

    @staticmethod
    def _ctrl_flow_kind_compressed(instr16: int) -> str:
        """压缩指令 (16-bit) 的控制流类型."""
        quad = instr16 & 0x3
        funct3 = (instr16 >> 13) & 0x7
        if quad == 0b01:  # C1 象限
            if funct3 in (0b001, 0b101):  # C.JAL, C.J
                return "term"
            if funct3 in (0b110, 0b111):  # C.BEQZ, C.BNEZ
                return "branch"
        elif quad == 0b10:  # C2 象限
            if funct3 == 0b100:  # C.JR / C.JALR / C.EBREAK
                return "term"
        return "normal"

    @seize_val_err("addr 和 count 需为整数 (支持 0x 前缀)")
    def cmd_disasm(self, addr_str: str, inst_count_str: str = "16") -> None:
        """反汇编指定内存区域.

        disasm <addr> [count]  — 从 addr 开始反汇编 count 条指令 (默认 16).

        自动识别 16-bit 压缩指令和 32-bit 标准指令边界,
        按地址递增顺序逐条输出。若 PC 落入范围内, 以 ``pc ->`` 标记
        当前指令, 后续行以 ``+N`` 表示相对于 PC 的指令步数。
        遇无条件跳转/ret/mret/sret/ecall 等控制流终止指令时
        插入分隔线, 其后指令不再累加步数。
        Enter 重复时自动推进地址并延续步数编号。
        """
        addr = self._resolve_addr(addr_str)
        # 地址解析失败时尝试符号名 (PIE 偏移后)
        if addr is None and self._image and self._image.symbols:
            sym_addr = self._image.symbols.get(addr_str)
            if sym_addr is not None:
                addr = sym_addr + self._load_offset
        inst_count = int(inst_count_str, 0)

        if addr is None:
            self._err(f"无法解析地址: {addr_str}")
            return

        if inst_count <= 0 or inst_count > 512:
            self._err("指令数需在 1-512 之间")
            return

        if not self._check_rv64_addr(addr):
            return

        # 非 2-字节对齐: RISC-V 指令至少 16-bit 对齐, 奇数地址解码毫无意义
        if addr & 1:
            orig = addr
            addr &= ~1
            self._warn(f"addr 至少应为 2-字节对齐, 已从 0x{orig:016x} 对齐到 0x{addr:016x}")

        # 读取足够覆盖 inst_count 条指令的字节 (每条最多 4 字节)
        raw = self._try_read_va(addr, inst_count * 4)
        if raw is None:
            self._err("无法读取指定地址")
            return

        # ---- 判断是否为重复执行 (Enter 自动推进) ----
        is_continue = self._disasm_ref_pc is not None and addr == self._disasm_next_addr

        # ---- 第一遍: 收集指令元组 (addr, raw_hex, asm, ctrl_kind) ----
        instrs: list[tuple[int, str, str, str]] = []
        offset = 0
        max_offset = len(raw)

        while offset < max_offset and len(instrs) < inst_count:
            pc_addr = addr + offset
            remaining = max_offset - offset

            chunk = raw[offset : offset + min(4, remaining)]
            instr = int.from_bytes(chunk.ljust(4, b"\x00"), "little", signed=False)

            is_compressed = parse_compressed(instr)
            inst_size = 2 if is_compressed else 4

            if remaining < inst_size:
                # 尝试从内存多读几个字节以补全指令, 避免在窗口边界截断
                extra = self._emu.bus.try_read(addr + offset, inst_size)
                if extra is None:
                    hex_s = " ".join(f"{b:02x}" for b in raw[offset:])
                    instrs.append((pc_addr, hex_s, "[dim](截断)[/]", "normal"))
                    break
                raw = raw[:offset] + extra + raw[offset + len(extra) :]
                remaining = inst_size

            asm = disasm(instr, pc_addr)
            raw_hex = f"{instr & 0xFFFF:04x}" if is_compressed else f"{instr:08x}"
            ctrl = (
                self._ctrl_flow_kind_compressed(instr & 0xFFFF)
                if is_compressed
                else self._ctrl_flow_kind(instr)
            )
            instrs.append((pc_addr, raw_hex, asm, ctrl))
            offset += inst_size

        self._disasm_next_addr = addr + offset

        if not instrs:
            self._console.print("[dim](空)[/]")
            return

        # ---- 确定参照 PC 并计算 base_step ----
        if not is_continue:
            self._disasm_ref_pc = self.hart.pc
            self._disasm_past_terminator = False

            # 找到参照 PC 在本块内的位置
            ref_idx: int | None = None
            for i, (pc_addr, _, _, _) in enumerate(instrs):
                if pc_addr == self._disasm_ref_pc:
                    ref_idx = i
                    break

            if ref_idx is not None:
                self._disasm_base_step = -ref_idx
            elif addr > self._disasm_ref_pc:
                step_cnt = self._count_instrs_between(self._disasm_ref_pc, addr)
                self._disasm_base_step = step_cnt
                if step_cnt >= MAX_INSTR_COUNT:
                    self._disasm_past_terminator = True  # 区间过大, 禁用步数
            else:
                step_cnt = self._count_instrs_between(addr, self._disasm_ref_pc)
                self._disasm_base_step = -step_cnt
                if step_cnt >= MAX_INSTR_COUNT:
                    self._disasm_past_terminator = True

        # ---- 第二遍: 格式化, 含控制流感知 ----
        ref_pc = self._disasm_ref_pc
        base = self._disasm_base_step
        past_term = self._disasm_past_terminator
        next_base = base  # 累积可达指令步数

        lines: list[str] = []
        prefix_w = 5  # "pc ->" = 5 chars, 动态扩展

        # 符号表 (供地址→名称映射)
        syms = self._image.symbols if self._image else {}
        last_scope: str | None = None  # "<.text:func_name>"

        for i, (pc_addr, raw_hex, asm, ctrl) in enumerate(instrs):
            step = base + i
            at_ref = pc_addr == ref_pc

            # 确定前缀
            prefix = ""
            if at_ref:
                prefix = "pc ->"
            elif step > 0 and not past_term:
                prefix = f"+{step}"
                prefix_w = max(prefix_w, len(prefix))

            # 段/函数边界: 仅在变化时插入 <段:函数名> 标头行
            link_addr = pc_addr - self._load_offset
            sym_name = self._resolve_symbol(syms, link_addr)
            seg = self._find_segment(link_addr)
            seg_name = seg.name if seg and seg.name else ""
            scope = (
                f"<[dim]{seg_name}[/]:[yellow]{sym_name}[/]>"
                if (seg_name and sym_name) else ""
            )
            if scope and scope != last_scope:
                last_scope = scope
                lines.append(f"  {scope}")

            asm_colored = self._colorize_asm(asm)
            lines.append(
                f"  {prefix:<{prefix_w}}  [blue]{self._hex(pc_addr)}[/]"
                f"  [bright_black]{raw_hex:<8s}[/]  {asm_colored}"
            )

            # 遇终止指令: 插入分隔, 后续不再累加步数
            # 无条件跳转 或 条件分支: 后续指令不一定执行, 截断步数
            if ctrl in ("term", "branch") and not past_term and step >= 0:
                past_term = True
                lines.append(f"  {'─' * (prefix_w + 70)}")

            # 更新累积步数 — 正负步数均推进, 使 next_base 逐步趋近 0
            # (此前仅 step>=0 时推进, 导致 ref_pc 之前的所有 Enter 重复
            #  都卡在同一负基数上, 抵达 ref_pc 后也无法显示 +N.)
            if not past_term:
                next_base = step + 1

        self._disasm_base_step = next_base
        self._disasm_past_terminator = past_term

        self._console.print(
            f"[bold]反汇编[/] [yellow]{self._hex(addr)}[/]"
            f"  {inst_count} 条指令\n" + "\n".join(lines)
        )

    def cmd_vdisasm(self, addr_str: str, inst_count_str: str = "16") -> None:
        """虚拟地址反汇编 — 强制经 MMU (satp) 翻译 VA→PA 后读取指令.

        vdisasm <vaddr> [count]  — 从 vaddr 开始反汇编 count 条指令 (默认 16).

        与 disasm 的区别:
        - vdisasm 始终使用 satp 页表翻译, 即使当前 hart 在 M 模式.
        - 显示 VA→PA 映射: 每条指令标 {vaddr:paddr}页内偏移.
        - 仅在 satp 使能 (mmu_mode != Bare) 时有效.

        用法和步数推进逻辑与 disasm 一致.
        """
        va = self._resolve_addr(addr_str)
        if va is None and self._image and self._image.symbols:
            sym_addr = self._image.symbols.get(addr_str)
            if sym_addr is not None:
                va = sym_addr + self._load_offset
        inst_count = int(inst_count_str, 0)

        if va is None:
            self._err(f"无法解析地址: {addr_str}")
            return
        if inst_count <= 0 or inst_count > 512:
            self._err("指令数需在 1-512 之间")
            return
        if not self._check_rv64_addr(va):
            return
        if self.hart.mmu_mode == 0:
            self._err("satp 未使能 (Bare 模式), vdisasm 无可用翻译; 请用 disasm")
            return

        # 非 2-字节对齐: RISC-V 指令至少 16-bit 对齐
        if va & 1:
            orig = va
            va &= ~1
            self._warn(f"addr 至少应为 2-字节对齐, 已从 0x{orig:016x} 对齐到 0x{va:016x}")

        # 先检查单页翻译是否可行, 给出明确诊断
        if self.hart._mem_read_phy is None:
            self._err("内存后端未挂载")
            return
        root_ppn = self.hart.satp_val & ((1 << 44) - 1)
        ok, _pa, _, _ = sv39_walk(
            root_ppn, va,
            self.hart._mem_read_phy
        )
        if not ok:
            self._err(
                f"VA [yellow]{self._hex(va)}[/] 页表翻译失败 (缺页);"
                f" satp root PPN=0x{root_ppn:x}"
            )
            return

        # 强制 VA→PA 翻译, 读取指令字节
        result = self._try_read_va_forced(va, inst_count * 4)
        if result is None:
            self._err(f"VA [yellow]{self._hex(va)}[/] 翻译后物理读取失败")
            return
        first_pa, raw = result

        # ---- 第一遍: 收集指令元组 (va_addr, raw_hex, asm, ctrl_kind, pa_addr) ----
        instrs: list[tuple[int, str, str, str, int]] = []
        offset = 0
        max_offset = len(raw)

        while offset < max_offset and len(instrs) < inst_count:
            pc_va = va + offset
            pc_pa = first_pa + offset  # 同页内 VA/PA 偏移一致; 跨页已由 _try_read_va_forced 处理
            remaining = max_offset - offset

            chunk = raw[offset : offset + min(4, remaining)]
            instr = int.from_bytes(chunk.ljust(4, b"\x00"), "little", signed=False)

            is_compressed = parse_compressed(instr)
            inst_size = 2 if is_compressed else 4

            if remaining < inst_size:
                # 窗口边界截断 — 从实际物理地址补读
                extra = self._emu.bus.try_read(pc_pa, inst_size)
                if extra is not None:
                    raw = raw[:offset] + extra + raw[offset + len(extra) :]
                    remaining = inst_size
                else:
                    hex_s = " ".join(f"{b:02x}" for b in raw[offset:])
                    instrs.append((pc_va, hex_s, "[dim](截断)[/]", "normal", pc_pa))
                    break

            asm = disasm(instr, pc_va)
            raw_hex = f"{instr & 0xFFFF:04x}" if is_compressed else f"{instr:08x}"
            ctrl = (
                self._ctrl_flow_kind_compressed(instr & 0xFFFF)
                if is_compressed else self._ctrl_flow_kind(instr)
            )
            instrs.append((pc_va, raw_hex, asm, ctrl, pc_pa))
            offset += inst_size

        self._vdisasm_next_addr = va + offset

        if not instrs:
            self._console.print("[dim](空)[/]")
            return

        # ---- 第二遍: 格式化 ----
        ref_pc = self.hart.pc
        syms = self._image.symbols if self._image else {}
        last_scope: str | None = None

        # 计算列宽: VPN 最长 9 位 (Sv39), 动态适配
        vpn_w, ppn_w = 0, 0
        for pc_va, _, _, _, pc_pa in instrs:
            vpn_w = max(vpn_w, len(f"{pc_va >> 12:x}"))
            ppn_w = max(ppn_w, len(f"{pc_pa >> 12:x}"))

        lines: list[str] = [
            f"  {{vpn{' ' * (vpn_w - 3)}:ppn{' ' * (ppn_w - 3)}}}  off  "
            f"raw{' ' * (8 - 3)}  asm"
        ]

        for pc_va, raw_hex, asm, ctrl, pc_pa in instrs:
            # 段/函数边界标头
            link_addr = pc_va - self._load_offset
            sym_name = self._resolve_symbol(syms, link_addr)
            seg = self._find_segment(link_addr)
            seg_name = seg.name if seg and seg.name else ""
            scope = (
                f"<[dim]{seg_name}[/]:[yellow]{sym_name}[/]>"
                if (seg_name and sym_name) else ""
            )
            if scope and scope != last_scope:
                last_scope = scope
                lines.append(f"  {scope}")

            # VPN:PPN + 页内偏移
            vpn = pc_va >> 12
            ppn = pc_pa >> 12
            off = pc_va & 0xFFF
            addr_pair = f"{{{vpn:0{vpn_w}x}:{ppn:0{ppn_w}x}}}"
            off_str = f"{off:03x}"

            # PC 标记
            at_ref = pc_va == ref_pc
            prefix = "pc ->" if at_ref else ""

            asm_colored = self._colorize_asm(asm)
            lines.append(
                f"  {prefix:<5}  {addr_pair} {off_str}"
                f"  [bright_black]{raw_hex:<8s}[/]  {asm_colored}"
            )

            if ctrl in ("term", "branch"):
                lines.append(f"  {'─' * 80}")

        self._console.print(
            f"[bold]vdisasm[/] VA [yellow]{self._hex(va)}[/]"
            f" → PA [yellow]{self._hex(first_pa)}[/]"
            f"  {inst_count} 条指令\n" + "\n".join(lines)
        )

    def _count_instrs_between(self, start: int, end: int) -> int:
        """计算从 *start* (含) 到 *end* (不含) 之间的指令条数.

        *start* 必须在 RAM 区域内 (指令仅存在于 RAM);
        若 *start* 不在 RAM 中或区间遍历超过 MAX_INSTR_COUNT 条,
        返回该上限值, 调用方应将其视为不可达区间.
        """
        # 非 RAM 地址 (MMIO / 空洞) 不含指令, 直接返回不可达
        bus = self._emu.bus
        if not bus.is_ram_addr(start):
            return MAX_INSTR_COUNT

        count = 0
        addr = start
        while addr < end:
            if count >= MAX_INSTR_COUNT:
                return count
            raw = bus.try_read(addr, 4)
            if raw is None:
                break
            instr = int.from_bytes(raw, "little", signed=False)
            addr += 2 if parse_compressed(instr) else 4
            count += 1
        return count

    @seize_val_err("addr 和 size 需为整数 (支持 0x 前缀)")
    def cmd_mem(self, addr_str: str, size_str: str = "64") -> None:
        addr = self._resolve_addr(addr_str)
        if addr is None:
            self._err(f"无法解析地址: {addr_str}")
            return
        size = int(size_str, 0)
        if size <= 0 or size > 4096:
            self._err("size 需在 1–4096 之间")
            return
        if not self._check_rv64_addr(addr):
            return
        data = self._try_read_va(addr, size)
        if data is None:
            self._err("无法读取指定地址")
            return

        self._console.print(self._emu._fmt_hexdump(addr, data))

    def _show_va_mapping_header(self, va: int, size: int) -> None:
        """显示 VA→PA 翻译及各页的 PTE 权限位.

        对 [va, va+size) 所跨越的每个 4 KiB 页, 查找 TLB 获取
        物理页号与权限 (r/w/x/u), 输出一行摘要.
        """
        h = self.hart
        page_cnt = ((va & 0xFFF) + size + 0xFFF) >> 12
        lines: list[str] = []

        for pi in range(page_cnt):
            page_va = (va & ~0xFFF) + (pi << 12)
            vpn = page_va >> 12

            # 优先查 DTLB (已缓存), 否则走 Sv39 页表遍历
            hit, ppn, perm = h.dtlb.lookup(vpn)
            if not hit and h._mem_read_phy is None:
                lines.append(
                    f"  [red]VA 0x{page_va:016x}  内存后端未挂载[/]"
                )
                continue
            elif not hit and h._mem_read_phy is not None:
                root_ppn = h.satp_val & ((1 << 44) - 1)
                ok, ppn, perm, _ = sv39_walk(
                    root_ppn, page_va, h._mem_read_phy
                )
                if ok:
                    continue
                lines.append(
                    f"  [red]VA 0x{page_va:016x}  缺页 (无法翻译)[/]"
                )

            pa = ppn << 12
            r = "r" if perm & 1 else "-"
            w = "w" if perm & 2 else "-"
            x = "x" if perm & 4 else "-"
            u = "u" if perm & 8 else "s"

            lines.append(
                f"  VA 0x{page_va:016x} → PA 0x{pa:016x}  [{r}{w}{x}{u}]"
            )

        self._console.print("\n".join(lines))

    def cmd_vmem(self, addr_str: str, size_str: str = "64") -> None:
        """虚拟地址内存查看 — 强制经 MMU (satp) 翻译 VA→PA 后读取.

        vmem <vaddr> [size]  — 显示 vaddr 处 size 字节的 hexdump (默认 64).

        与 mem 的区别: vmem 始终使用 satp 页表翻译, 即使 hart 在 M 模式.
        仅在 satp 使能 (mmu_mode != Bare) 时有效.
        """
        va = self._resolve_addr(addr_str)
        if va is None:
            self._err(f"无法解析地址: {addr_str}")
            return
        size = int(size_str, 0)
        if size <= 0 or size > 4096:
            self._err("size 需在 1–4096 之间")
            return
        if not self._check_rv64_addr(va):
            return
        if self.hart.mmu_mode == 0:
            self._err("satp 未使能 (Bare 模式), vmem 无可用翻译; 请用 mem")
            return

        # 先检查单页翻译是否可行
        if self.hart._mem_read_phy is None:
            self._err("内存后端未挂载")
            return
        root_ppn = self.hart.satp_val & ((1 << 44) - 1)
        ok, _, _, _ = sv39_walk(root_ppn, va, self.hart._mem_read_phy)
        if not ok:
            self._err(
                f"VA [yellow]{self._hex(va)}[/] 页表翻译失败 (缺页);"
                f" satp root PPN=0x{root_ppn:x}"
            )
            return

        # 强制 VA→PA 翻译
        result = self._try_read_va_forced(va, size)
        if result is None:
            self._err(f"VA [yellow]{self._hex(va)}[/] 翻译后物理读取失败")
            return
        first_pa, data = result

        # VA→PA 映射摘要
        self._show_va_mapping_header(va, size)
        # hexdump 以 VA 标注, 附加首 PA 信息
        self._console.print(
            f"[dim]VA {self._hex(va)} → PA {self._hex(first_pa)}[/]"
        )
        self._console.print(self._emu._fmt_hexdump(va, data))

    def cmd_status(self, hart_id_str: str | None = None) -> None:
        """显示 hart 状态. 无参数时显示全部 hart 概览, 带参数时显示指定 hart 详情."""
        if hart_id_str is None:
            # 无参数: 全部 hart 概览
            self._show_hart_overview()
            return
        try:
            hid = int(hart_id_str, 0)
        except ValueError:
            self._err(f"无效 hart ID: {hart_id_str}")
            return
        if hid < 0 or hid >= self._emu.num_harts:
            self._err(f"hart ID 超出范围: 0–{self._emu.num_harts - 1}")
            return
        self._show_hart_detail(hid)
        return

    def _show_hart_overview(self) -> None:
        """终端对齐的全部 hart 概览表."""
        harts = self._emu.harts
        total = len(harts)
        active_hart = self._hart_id

        tbl = Table(
            title=f"Harts ({total} total)  [dim]● = 当前[/]",
            border_style="blue",
        )
        tbl.add_column("", style="cyan", width=1)
        tbl.add_column("#", style="cyan", justify="right")
        tbl.add_column("PC", style="green")
        tbl.add_column("Mode", style="yellow", width=5)
        tbl.add_column("State", width=8)
        tbl.add_column("Instr", justify="right")

        for h in harts:
            mark = "*" if h.id == active_hart else " "
            if h._halted:
                state = "[red]halted[/]"
            elif h._waiting:
                state = "[dim]waiting[/]"
            else:
                state = "[green]running[/]"
            instr_fmt = "0"
            if h.id == active_hart:
                instr_fmt = self._fmt_instr_count(self._instr_count)
            tbl.add_row(
                mark,
                str(h.id),
                self._hex(h.pc),
                h.mode.name,
                state,
                instr_fmt,
            )
        self._console.print(tbl)

    def _show_hart_detail(self, hart_id: int) -> None:
        h = self._emu.harts[hart_id]
        halted_note = " [red]已暂停 — 不可恢复陷态[/]" if h._halted else ""
        instr_fmt = self._fmt_instr_count(self._instr_count) if hart_id == self._hart_id else "0"
        tbl = Table(
            title=(f"Hart {hart_id}  指令计数: {instr_fmt}{halted_note}"),
            border_style="blue",
        )
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Value", style="green")
        tbl.add_row("PC", self._hex(h.pc))
        tbl.add_row("Mode", h.mode.name)
        tbl.add_row("mstatus", self._hex(h.mstatus_val))
        tbl.add_row("mie / mpie / mpp", f"{h.mie}, {h.mpie}, {h.mpp.name}")
        tbl.add_row("mepc", self._hex(h.mepc_val))
        tbl.add_row("mcause", self._hex(h.mcause_val))
        tbl.add_row("mtval", self._hex(h.mtval_val))
        tbl.add_row("mtvec", self._hex(h.mtvec_val))
        tbl.add_row("satp", self._hex(h.satp_val))
        tbl.add_row(
            "连续 trap",
            f"{h._consecutive_traps}/{self._emu._TRAP_LOOP_THRESHOLD} (阈值)",
        )
        tbl.add_row(
            "reservation",
            f"valid={h.reservation_valid}, addr=0x{h.reservation_addr:x}",
        )
        self._console.print(tbl)

    def _read_gpr_by_name(self, name: str) -> int:
        """按名称读取 GPR (例: x12, t0, a0, sp)."""
        idx = gpr_idx_from_name(name)
        return 0 if idx is None else self.hart.read_gpr(idx)

    def _read_csr_by_name(self, name: str) -> int:
        """按名称读取 CSR (例: mtvec, mstatus, mepc)."""
        addr = csr_addr_from_name(name)
        return 0 if addr is None else self.hart.read_csr(addr)

    @staticmethod
    def _resolve_symbol(symbols: dict[str, int], addr: int) -> str | None:
        """在符号表中查找包含 *addr* 的函数符号.

        精确匹配优先; 否则找地址 ≤ addr 且差值最小的符号,
        视为当前执行点所在的函数 (相差 > 64 KiB 视为不在任何函数内).
        """
        # 精确匹配
        for name, a in symbols.items():
            if a == addr:
                return name
        # 最近的前驱符号 (跳过 $x / $d 等映射符号)
        best_name, best_dist = None, 0xFFFF_FFFF_FFFF_FFFF
        for name, a in symbols.items():
            if name.startswith("$"):
                continue
            if a <= addr and (addr - a) < best_dist:
                best_dist, best_name = addr - a, name
        return best_name if best_dist <= 0x10000 else None

    def _find_segment(self, addr: int) -> "FirmwareSegment | None":
        """返回包含 *addr* 的固件段.

        精确匹配优先; 若地址落在段间空隙, 返回距离最近的前驱段
        (仅当距离 ≤ 段自身 memsz 时, 防止距离过远仍错误关联).
        """
        if self._image is None:
            return None
        for seg in self._image.segments:
            if seg.vaddr <= addr < seg.vaddr + seg.memsz:
                return seg
        # 地址落在所有段外 — 找最近的前驱段 (1 MiB 内)
        best_seg, best_dist = None, 0xFFFF_FFFF_FFFF_FFFF
        for seg in self._image.segments:
            end = seg.vaddr + seg.memsz
            if end <= addr and (addr - end) < best_dist:
                best_dist, best_seg = addr - end, seg
        if best_seg is not None and best_dist <= 0x100000:
            return best_seg
        return None

    # ----------------------------------------------------------
    #  栈帧回溯辅助
    # ----------------------------------------------------------

    def _try_read_frame_link(self, fp: int) -> tuple[int, int] | None:
        """读取 fp 指向的栈帧链接 (saved_ra, saved_fp); 越界则返回 None.

        RISC-V 标准栈帧布局: fp-8 存返回地址, fp-16 存调用者的 fp.
        当 MMU 使能时 fp 为虚拟地址, 自动翻译后读取物理内存.
        """
        ra_raw = self._try_read_va(fp - 8, 8)
        if ra_raw is None:
            return None
        ra = int.from_bytes(ra_raw, "little", signed=False)
        fp_raw = self._try_read_va(fp - 16, 8)
        if fp_raw is None:
            return None
        fp = int.from_bytes(fp_raw, "little", signed=False)
        return ra, fp

    # 特权级 → Rich 颜色
    _MODE_COLORS = {"M": "red", "S": "cyan", "U": "green", "H": "yellow", "D": "magenta"}

    def _parse_trap_save_offsets(self, tvec: int, max_instrs: int = 20) -> tuple[int, int] | None:
        """解析 trap 入口代码, 提取 RA/FP 相对 trap SP 的保存偏移.

        从 *tvec* (mtvec/stvec) 开始反汇编, 找第一条保存 RA (x1)
        和第一条保存 FP (x8/s0) 到栈的指令。若找不到则返回 None。
        指令解码委托给 :func:`pyremu.core.decoder.decode_c_sdsp` /
        :func:`pyremu.core.decoder.decode_sd_sp`.
        """
        raw = self._emu.bus.try_read(tvec, max_instrs * 4)
        if raw is None:
            return None
        ra_off: int | None = None
        fp_off: int | None = None
        pos = 0
        while pos + 2 <= len(raw) and (ra_off is None or fp_off is None):
            half = int.from_bytes(raw[pos:pos + 2], "little", signed=False)
            if (half & 0x3) != 3:  # 16-bit compressed
                decoded = decode_c_sdsp(half)
                if decoded is not None:
                    rs2, uimm = decoded
                    if rs2 == 1 and ra_off is None:
                        ra_off = uimm
                    elif rs2 == 8 and fp_off is None:
                        fp_off = uimm
                pos += 2
            else:  # 32-bit
                instr = int.from_bytes(raw[pos:pos + 4], "little", signed=False)
                decoded = decode_sd_sp(instr)
                if decoded is not None:
                    rs2, imm = decoded
                    if rs2 == 1 and ra_off is None:
                        ra_off = imm
                    elif rs2 == 8 and fp_off is None:
                        fp_off = imm
                pos += 4

        if ra_off is None:
            return None
        return ra_off, (fp_off if fp_off is not None else 0)

    def _resolve_boundary_frame(
        self,
        trapped_pc: int,
        prev_name: str,
    ) -> tuple[str, str]:
        """解析跨特权级边界帧的 privilege mode.

        当 mepc 被嵌套 trap 覆写后, trapped_pc 可能指向 M-mode 代码
        而 mpp/spp 仍指向原始特权级. 按 trapped_pc 实际所在段修正模式标签.
        note 留空 — 模式标签已隐含特权级转换.
        """
        if prev_name != "M" and self._image:
            seg = self._find_segment(trapped_pc - self._load_offset)
            if seg is not None and seg.name and ".text" in seg.name:
                return ("M", "")
        return (prev_name, "")

    def _is_valid_mmode_code(self, pc: int) -> bool:
        """检查 *pc* 是否落在已知的 M 模式代码段内.
        el
        用于检测嵌套 trap: 当 mepc 指向 M 模式代码而 mpp 指示低特权级时,
        说明 mepc 已被内层 trap 覆写.
        """
        if not self._image:
            return False
        seg = self._find_segment(pc - self._load_offset)
        return seg is not None and seg.name is not None and ".text" in seg.name

    def _add_prev_mode_frame(
        self,
        frames: list[StackFrame],
        trapped_pc: int,
        prev_mode: RiscvMode,
        tvec: int,
        visited: set[int],
    ) -> None:
        """为指定 trap 上下文添加边界帧, 并尝试继续回溯低特权级 FP 链.

        与 _walk_prev_mode_frames 的主体逻辑相同, 但参数显式传入,
        供嵌套 trap 复用 (此时 tvec 可能指向不同特权级的 trap 入口).
        """
        if trapped_pc == 0 or tvec == 0:
            return
        prev_name = prev_mode.name
        boundary_mode, boundary_note = self._resolve_boundary_frame(
            trapped_pc, prev_name
        )
        offsets = self._parse_trap_save_offsets(tvec)
        if offsets is None:
            frames.append(StackFrame(
                idx=len(frames), fp=0, sp=0, ra=0,
                pc=trapped_pc,
                note=(
                    "trap 入口寄存器保存布局无法解析 (仅 PC)"
                    if not boundary_note else boundary_note
                ),
                mode=boundary_mode,
            ))
            return
        ra_off, fp_off = offsets

        # 在栈上扫描 trapped_pc 定位保存上下文
        last_fp = frames[-1].fp if frames else 0
        last_sp = frames[-1].sp if frames else 0
        search_base = min(last_fp if last_fp else last_sp,
                          last_sp if last_sp else last_fp) & ~0x7
        if search_base == 0:
            # 没有已知栈指针 → 无法定位 trap 帧
            frames.append(StackFrame(
                idx=len(frames), fp=0, sp=0, ra=0,
                pc=trapped_pc,
                note="仅 trap PC (缺少栈帧参考点)",
                mode=boundary_mode,
            ))
            return
        raw = self._emu.bus.try_read(search_base, 4096)
        if raw is None:
            return
        mepc_bytes = struct.pack("<Q", trapped_pc)
        trap_sp: int | None = None
        pos = len(raw) - 8
        while pos >= 0:
            if raw[pos:pos + 8] != mepc_bytes:
                pos -= 8
                continue
            candidate_base = search_base + pos
            for mepc_guess in (256, 0, 128, 64, 192):
                trial_sp = candidate_base - mepc_guess
                ra_addr = trial_sp + ra_off
                ra_raw = self._emu.bus.try_read(ra_addr, 8)
                if ra_raw is None:
                    continue
                ra_val = int.from_bytes(ra_raw, "little", signed=False)
                if not self._emu.bus.is_valid_addr(ra_val):
                    continue
                zero_raw = self._emu.bus.try_read(trial_sp, 8)
                if zero_raw is None:
                    continue
                zero_val = int.from_bytes(zero_raw, "little", signed=False)
                if zero_val != 0:
                    continue
                trap_sp = trial_sp
                break
            if trap_sp is not None:
                break
            pos -= 8

        # 读取被保存的 RA, FP, SP
        saved_ra: int | None = None
        saved_fp: int | None = None
        saved_sp: int | None = None
        if trap_sp is not None:
            saved_ra = self._emu.bus.read_u64(trap_sp + ra_off)
            if fp_off:
                saved_fp = self._emu.bus.read_u64(trap_sp + fp_off)
            saved_sp = self._emu.bus.read_u64(trap_sp + ra_off + 8)

        call_site = trapped_pc - 4 if trapped_pc >= 4 else 0
        note = boundary_note
        if saved_fp is None and saved_sp is None and saved_ra is None:
            note = "仅 trap PC (栈扫描无匹配)"
        frames.append(StackFrame(
            idx=len(frames),
            fp=saved_fp or 0,
            sp=saved_sp or 0,
            ra=saved_ra or 0,
            pc=call_site,
            note=note,
            mode=boundary_mode,
        ))

        # 继续走低特权级 FP 链
        if not (saved_fp and saved_fp != 0 and saved_fp not in visited):
            return
        visited.add(saved_fp)
        current_fp = saved_fp
        pm_str = prev_name
        for _ in range(64):
            if current_fp == 0:
                break
            link = self._try_read_frame_link(current_fp)
            if link is None:
                break
            next_ra, next_fp = link
            if next_fp == 0:
                break
            if next_fp <= current_fp or (next_fp & 0x7) or next_fp in visited:
                break
            if next_ra == 0:
                break
            visited.add(next_fp)
            call_site = next_ra - 4 if next_ra >= 4 else 0
            frames.append(StackFrame(
                idx=len(frames), fp=next_fp, sp=current_fp,
                ra=next_ra, pc=call_site, mode=pm_str,
            ))
            current_fp = next_fp

    def _walk_frame_chain(self) -> list[StackFrame]:
        """沿 FP 链遍历调用栈, 返回 StackFrame 列表.

        从当前 hart 的 s0/fp 出发, 按标准 RISC-V 栈帧布局
        (fp-8 存 RA, fp-16 存 saved FP) 向上回溯.
        FP 链走完后解析 trap 入口代码获取寄存器保存偏移,
        在栈上定位被中断上下文并跨特权级继续回溯.
        每帧标记所属特权级, 供颜色渲染.
        """
        h = self.hart
        current_fp: int = h.gprs[8]  # s0/fp
        visited: set[int] = {current_fp}
        cur_mode = h.mode.name  # "M" / "S" / "U" / ...
        # 帧 #0: 当前执行点
        frames: list[StackFrame] = [StackFrame(
            idx=0, fp=current_fp, sp=h.gprs[2], ra=h.gprs[1],
            pc=h.pc, mode=cur_mode,
        )]

        while True:
            if current_fp == 0:
                break
            link = self._try_read_frame_link(current_fp)
            if link is None:
                break
            saved_ra, saved_fp = link
            # RISC-V ABI: fp=0 无条件标记栈帧链尾
            if saved_fp == 0:
                break
            # 防御: fp 必须 (a) 随栈增长递增 (b) 8 字节对齐 (c) 不回环
            if saved_fp <= current_fp or (saved_fp & 0x7) or saved_fp in visited:
                break
            if saved_ra == 0:
                break
            visited.add(saved_fp)
            call_site = saved_ra - 4 if saved_ra >= 4 else 0
            frames.append(StackFrame(
                idx=len(frames), fp=saved_fp, sp=current_fp,
                ra=saved_ra, pc=call_site, mode=cur_mode,
            ))
            current_fp = saved_fp

        # -- 跨特权级回溯 --
        self._walk_prev_mode_frames(frames, cur_mode, visited)
        return frames

    def _walk_prev_mode_frames(
        self,
        frames: list[StackFrame],
        cur_mode: str,
        visited: set[int],
    ) -> None:
        """尝试解析 trap 入口, 定位被中断上下文并继续回溯低特权级 FP 链."""
        h = self.hart
        mode_val = h.mode.value

        # 获取 tvec 和 trapped PC.
        # 嵌套 trap 场景: mepc 被新 trap 覆写为 M-mode 地址, 但 mpp 仍指向
        # 原始特权级. 此时需额外记下 sepc / SPP 作为被中断的低特权级上下文.
        if h.mode in set((RiscvMode.M, RiscvMode.D)):
            tvec = h.mtvec_val
            trapped_pc = h.mepc_val
            prev_mode = h.mpp
            s_trapped_pc = h.sepc_val    # 可能保存着 S 模式被中断的地址
            s_prev_mode = h.spp
        elif h.mode == RiscvMode.S:
            tvec = h.stvec_val
            trapped_pc = h.sepc_val
            prev_mode = h.spp
            s_trapped_pc = 0
            s_prev_mode = RiscvMode.U
        else: # 暂时不考虑虚拟化
            return

        if trapped_pc == 0 or prev_mode.value >= mode_val or tvec == 0:
            return

        prev_name = prev_mode.name

        # 嵌套 trap 检测: mepc 指向 M 模式代码, 但 mpp 指示低特权级.
        # mepc 由当前 (内层) trap 覆写; sepc 保存着原始 S 模式故障地址.
        _nested = (
            h.mode == RiscvMode.M or h.mode == RiscvMode.D
            and s_trapped_pc != 0
            and prev_mode.value < RiscvMode.M.value
            and self._is_valid_mmode_code(trapped_pc)
        )

        # 1) 解析 trap 入口 → 获取 RA/FP 保存偏移
        # 当前实现仅识别 ``sd x1/x8, IMM(sp)`` 模式, 对以下场景会降级到仅显示 pc:
        #  - 用非 sp 寄存器做存数基址 (如 OpenSBI csrrw tp, mscratch 切换到异常栈)
        #  - 压缩指令 C.SDSP 保存 (只处理了 32-bit SD)
        #  - sp 经大量代数运算后存入非常规位置 (极端情况需 SMT 求解, 不追踪)
        # 降级时边界帧只含 trapped PC, 缺少 sp/fp/ra, 低特权级 FP 链不可见.
        boundary_mode, boundary_note = self._resolve_boundary_frame(
            trapped_pc, prev_name
        )

        offsets = self._parse_trap_save_offsets(tvec)
        if offsets is None:
            # 无法解析 trap 入口的寄存器保存布局 — 标记 trapped PC 后,
            # 若为嵌套 trap 则继续尝试解析 sepc 指向的低特权级帧.
            payload = boundary_note
            if not boundary_note and _nested:
                payload = "M-mode 嵌套 trap (mepc), 寄存器保存布局无法解析"
            elif not boundary_note and not _nested:
                payload = "trap 入口寄存器保存布局无法解析 (仅 PC)"
            frames.append(StackFrame(
                idx=len(frames), fp=0, sp=0, ra=0,
                pc=trapped_pc,
                note=(payload),
                mode=boundary_mode,
            ))
            if _nested:
                # 内层 M-mode trap 帧
                # 继续尝试用 sepc 构造原始 S-mode 边界帧
                self._add_prev_mode_frame(
                    frames, s_trapped_pc, s_prev_mode, h.stvec_val, visited
                )
            return
        # 2) 调用通用添加逻辑 (栈扫描 + FP 链回溯)
        self._add_prev_mode_frame(
            frames, trapped_pc, prev_mode, tvec, visited
        )

    @seize_val_err("无效帧号")
    def cmd_frame(self, arg: str | None = None) -> None:
        """栈帧回溯 — #01 起始编号, 只显示当前帧栈内存.

        stack / bt   — 显示全部帧 + 当前帧的栈内存
        frame <N>    — 切换到第 N 帧并刷新回溯 (N 为 0-indexed)
        """
        self._stack_frames = self._walk_frame_chain()
        syms = self._image.symbols if self._image else {}

        # -- 帧选择: 切换当前帧, 然后 fall through 显示回溯 --
        if arg is not None:
            n = int(arg, 0)  # ValueError caught by decorator
            if not (0 <= n < len(self._stack_frames)):
                self._err(f"帧号 {n} 超出范围 [0, {len(self._stack_frames) - 1}]")
                return
            self._current_frame_idx = n

        # -- 回溯列表 (3 列表格: #号 | 段:函数名 | 寄存器) --
        tbl = Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("frame", style="bold")
        tbl.add_column("where")
        tbl.add_column("regs", style="dim")
        for f in self._stack_frames:
            tag = f"#{f.idx + 1:02d}"
            color = self._MODE_COLORS.get(f.mode, "")
            mode_tag = f"[{color}]{f.mode}[/]" if color else ""
            if f.note:
                # 跨特权级边界帧: 显示 trapped PC + 边界标记
                fn = self._resolve_symbol(syms, f.pc - self._load_offset) or ""
                seg = self._find_segment(f.pc - self._load_offset)
                seg_name = seg.name if seg and seg.name else ""
                where = ""
                if seg_name and fn:
                    where = f"[dim]{seg_name}[/]:[{color}]{fn}[/]"
                elif fn:
                    where = f"[{color}]{fn}[/]"
                elif seg_name:
                    where = f"[dim]{seg_name}[/]"
                regs = (
                    f"pc={self._hex(f.pc)}  "
                    f"[bold magenta]{f.note}[/]"
                )
                tbl.add_row(f"{tag} {mode_tag}", where, regs)
                continue
            fn = self._resolve_symbol(syms, f.pc - self._load_offset) or ""
            seg = self._find_segment(f.pc - self._load_offset)
            seg_name = seg.name if seg and seg.name else ""
            where = ""
            if seg_name and fn:
                where = f"[dim]{seg_name}[/]:[{color}]{fn}[/]"
            elif fn:
                where = f"[{color}]{fn}[/]"
            elif seg_name:
                where = f"[dim]{seg_name}[/]"
            regs = (
                f"pc={self._hex(f.pc)}  sp={self._hex(f.sp)}  "
                f"fp={self._hex(f.fp)}  ra={self._hex(f.ra)}"
            )
            tbl.add_row(f"{tag} {mode_tag}", where, regs)


        self._console.print(
            f"[bold]执行栈帧[/] (深度 {len(self._stack_frames)})",
            tbl,
        )

        # -- 当前帧栈内存 --
        if self._current_frame_idx >= len(self._stack_frames):
            return
        cur = self._stack_frames[self._current_frame_idx]
        stack_data = self._emu.bus.try_read(cur.sp, 64)
        payload = "\n[dim](无法读取当前帧 SP 处内存)[/]"
        if stack_data is not None:
            payload = (
                "\n"
                + ("-" * 100)
                + "\n[bold]当前栈内存[/]\n"
                + fmt_hexdump(stack_data, addr=cur.sp)
            )
        self._console.print(payload)

    def cmd_symbols(self, filter_str: str = "") -> None:
        """列出固件符号表.

        无参数时按首字符分组显示各前缀的符号数量;
        带参数时列出以 *filter_str* 为前缀的全部符号, 按地址升序.
        """
        if self._image is None:
            self._warn("无可用的符号表 (非 ELF 文件)")
            return

        syms = self._image.symbols
        if not syms:
            self._console.print("[dim](符号表为空)[/]")
            return

        if not filter_str:
            # ---- 无参数: 按首字符分组统计 ----
            groups: dict[str, int] = {}
            for name in syms:
                key = name[0] if name else "?"
                groups[key] = groups.get(key, 0) + 1

            sorted_keys = sorted(groups.keys(), key=group_order)

            tbl = Table(
                title=f"符号分组 ({len(syms)} 项, {len(sorted_keys)} 组)",
                border_style="blue",
                expand=True,
            )
            tbl.add_column("Prefix", style="cyan")
            tbl.add_column("Count", style="yellow", justify="right")
            for key in sorted_keys:
                tbl.add_row(f"{key}...", str(groups[key]))
            self._console.print(tbl)
            self._console.print(
                "[dim]输入 sym <prefix> 查看具体符号 (如 sym a, sym f, sym _)[/]"
            )
            return

        # ---- 带参数: 按前缀匹配 ----
        entries = [
            (name, addr)
            for name, addr in syms.items()
            if name.lower().startswith(filter_str.lower())
        ]
        entries.sort(key=lambda x: x[1])

        if not entries:
            self._console.print(f"[dim]无以前缀 '{filter_str}' 开头的符号[/]")
            return

        # 超过阈值时展示次级分组 (字典树), 否则列出全部符号
        _sym_subgroup_threshold = 50
        def calc_order(x: str) -> tuple[int, str]:
            if not x:
                return (5, x)
            if len(x) > len(filter_str):
                x = x[-1].lower()
            else:
                x = x.lower()
            return group_order(x)
        if len(entries) > _sym_subgroup_threshold:
            # 按「当前前缀 + 下一个字符」分组
            prefix_len = len(filter_str)
            subgroups: dict[str, int] = {}
            for name, _ in entries: # name, __addr
                # 精确匹配
                next_key = name if len(name) <= prefix_len else name[: prefix_len + 1]
                subgroups[next_key] = subgroups.get(next_key, 0) + 1

            sorted_keys = sorted(subgroups.keys(), key=calc_order)
            tbl = Table(
                title=f"符号: '{filter_str}' ({len(entries)} 项, {len(sorted_keys)} 子组)",
                border_style="blue",
                expand=True,
            )
            tbl.add_column("Prefix", style="cyan")
            tbl.add_column("Count", style="yellow", justify="right")
            for key in sorted_keys:
                # 跳过空 key (不应出现)
                if not key:
                    continue
                suffix = key[prefix_len:]
                display = f"{filter_str}[bold]{suffix}[/]..."
                tbl.add_row(display, str(subgroups[key]))
            self._console.print(tbl)
            self._console.print(
                "[dim]输入 sym <prefix> 继续深入 (如 sym "
                + sorted_keys[0]
                + ", sym "
                + (sorted_keys[1] if len(sorted_keys) > 1 else sorted_keys[0])
                + ")[/]"
            )
            return

        # ---- 条目数量适中: 按地址升序列出全部 ----
        tbl = Table(
            title=f"符号: '{filter_str}' ({len(entries)} 项)",
            border_style="blue",
            expand=True,
        )
        tbl.add_column("Address", style="yellow")
        tbl.add_column("Name", style="cyan")
        for name, addr in entries:
            tbl.add_row(f"0x{addr:016x}", name)
        self._console.print(tbl)

    def cmd_hart(self, hart_id: int) -> None:
        if not (0 <= hart_id < self._emu.num_harts):
            self._err(f"Hart ID 超出范围 [0, {self._emu.num_harts - 1}]")
            return
        self._hart_id = hart_id
        self._snapshot = None
        self._mem_changes = []
        if self._emu.uart is not None:
            self._emu.uart.flush_all()
        self._console.print(f"[dim]切换到 Hart {hart_id}[/]")

    # ==========================================================
    #  REPL 命令分发
    # ==========================================================

    # show <name> 子命令 → 方法名 查表
    _SHOW_HANDLERS: dict[str, str] = {
        "mstatus": "cmd_mstatus",
        "satp": "cmd_satp",
        "mcause": "cmd_mcause",
        "scause": "cmd_scause",
        "mtvec": "cmd_mtvec",
        "stvec": "cmd_stvec",
        "mip": "cmd_mip",
        "mie": "cmd_mie",
        "sip": "cmd_sip",
        "sie": "cmd_sie",
        "medeleg": "cmd_medeleg",
        "mideleg": "cmd_mideleg",
        "pmps": "cmd_pmp",
        "tlb": "cmd_tlb",
    }

    def _dispatch_show(self, sub: str) -> None:
        if not sub:
            avail = ", ".join(sorted(self._SHOW_HANDLERS.keys()))
            self._console.print(f"[dim]show 可用子命令: {avail}[/]")
            return
        handler_name = self._SHOW_HANDLERS.get(sub)
        if handler_name is None:
            avail = ", ".join(sorted(self._SHOW_HANDLERS.keys()))
            self._warn(f"show: 未知子命令 '{sub}'. 可用: {avail}")
            return
        getattr(self, handler_name)()

    def _dispatch(self, parts: list[str]) -> bool:
        """分发 REPL 命令; 返回 False 表示退出."""
        cmd = parts[0].lower()

        # 退出
        if cmd in ("q", "quit", "exit"):
            return False

        # 执行控制
        if cmd in ("b", "bp"):
            self._dispatch_bp(parts[1:])
            return True
        if cmd in ("s", "step"):
            count = int(parts[1]) if len(parts) > 1 else 1
            self.cmd_step(count)
            return True
        if cmd in ("c", "continue"):
            self.cmd_continue()
            return True
        if cmd == "watch":
            addr = parts[1] if len(parts) > 1 else ""
            size = parts[2] if len(parts) > 2 else "8"
            self.cmd_watch(addr, size)
            return True
        if cmd in ("r", "run"):
            n = int(parts[1]) if len(parts) > 1 else 1
            self.cmd_run(n)
            return True
        if cmd in ("undo", "rollback"):
            self.rollback()
            return True
        if cmd == "restart":
            self.cmd_restart()
            return True

        # 寄存器
        if cmd in ("regs", "gpr"):
            self.cmd_regs()
            return True
        if cmd == "reg":
            if len(parts) < 2:
                self._warn("用法: reg <name>  例: reg x10, reg a0")
            else:
                self.cmd_reg(" ".join(parts[1:]))
            return True
        if cmd in ("set", "w"):
            if len(parts) < 3:
                self._warn("用法: set <name> <value>  例: set sp 0x8000")
            else:
                self.cmd_set(parts[1], parts[2])
            return True

        # CSR
        if cmd == "csr":
            if len(parts) >= 2:
                self.cmd_csr(" ".join(parts[1:]))
                return True
            self._warn(
                "用法: csr <name>  或  csr <name1>, <name2>, ...\n"
                "  例: csr mstatus\n"
                "  例: csr mscratch, mepc, pmpcfg0, pmpaddr0\n"
                "  例: csr list"
            )
            return True
        if cmd == "csrw":
            if len(parts) < 3:
                self._warn("用法: csrw <name> <value>  例: csrw mtvec 0x80000001")
            else:
                self.cmd_csrw(parts[1], parts[2])
            return True

        # 状态
        if cmd == "pc":
            self.cmd_pc(parts[1] if len(parts) > 1 else None)
            return True
        if cmd == "mode":
            self.cmd_mode()
            return True
        if cmd == "mstatus":
            self.cmd_mstatus()
            return True
        if cmd == "show":
            self._dispatch_show(parts[1] if len(parts) > 1 else "")
            return True
        if cmd == "tlb":
            self.cmd_tlb(parts[1] if len(parts) > 1 else None)
            return True
        if cmd == "tlbflush":
            self.cmd_tlbflush(parts[1] if len(parts) > 1 else None)
            return True
        if cmd == "cache":
            self.cmd_cache(parts[1] if len(parts) > 1 else None)
            return True
        if cmd == "mem":
            if len(parts) < 2:
                self._warn("用法: mem <addr> [size]  例: mem 0x80000000 64")
            else:
                self.cmd_mem(parts[1], parts[2] if len(parts) > 2 else "64")
            return True
        if cmd == "vmem":
            if len(parts) < 2:
                self._warn("用法: vmem <vaddr> [size]  例: vmem sepc 256")
            else:
                self.cmd_vmem(parts[1], parts[2] if len(parts) > 2 else "64")
            return True
        if cmd == "disasm":
            if len(parts) < 2:
                self._warn("用法: disasm <addr> [count]  例: disasm 0x80000000 32")
            else:
                self.cmd_disasm(parts[1], parts[2] if len(parts) > 2 else "16")
            return True
        if cmd == "vdisasm":
            if len(parts) < 2:
                self._warn("用法: vdisasm <vaddr> [count]  例: vdisasm sepc 16")
            else:
                self.cmd_vdisasm(parts[1], parts[2] if len(parts) > 2 else "16")
            return True
        if cmd == "pt":
            self.cmd_pt(parts[1] if len(parts) > 1 else None)
            return True
        if cmd in ("status", "info"):
            self.cmd_status(parts[1] if len(parts) > 1 else None)
            return True
        if cmd in ("stack", "bt", "frame", "f"):
            self.cmd_frame(parts[1] if len(parts) > 1 else None)
            return True
        # 符号
        if cmd in ("symbols", "sym"):
            self.cmd_symbols(parts[1] if len(parts) > 1 else "")
            return True

        # 配置
        if cmd == "hart":
            if len(parts) < 2:
                self._warn(f"用法: hart <id>  当前: {self._hart_id}")
            else:
                self.cmd_hart(int(parts[1]))
            return True

        # 帮助
        if cmd in ("h", "help", "?"):
            self._print_help()
            return True

        self._err(f"未知命令: {cmd} (输入 'help' 查看帮助)")
        return True

    # ==========================================================
    #  REPL 主循环
    # ==========================================================

    def repl(self) -> None:
        """主交互循环.

        prompt_toolkit 提供方向键浏览历史、Tab 补全、行编辑.
        rich Console 负责所有交互输出的格式化与着色.
        """
        self._enter_repl_mode()

        # Banner
        banner = Panel.fit(
            "\n".join([
                f"Harts: [bold cyan]{self._emu.num_harts}[/]   "
                f"Active hart: [bold cyan]{self._hart_id}[/]",
                f"Prog Cnt: [bold yellow]{self._emu.prog_cnt:#018x}[/]  "
                f"RAM: [dim]{Debugger._fmt_size(self._emu.bus._ram_size)} @ 0x{self._emu.bus.ram_base:x}[/]  "
                f"FDT: [dim]{f'0x{self._fdt_addr:x}' if self._fdt_addr else '—'}[/]",
                "",
                "Ctrl+C [dim]once[/]  → pause emulation",
                "Type [bold]help/h/?[/] for commands",
            ]),
            title="RISC-V Interactive Debugger [bold green](rvdb)[/]",
            border_style="green",
        )
        self._console.print(banner)

        while True:
            try:
                raw = self._session.prompt(
                    [("class:prompt", f"\nrvdbg[{self._hart_id}] ")]
                ).strip()
            except KeyboardInterrupt:
                self._console.print()
                continue
            except EOFError:
                self._console.print("[dim]退出[/]")
                break

            # 空输入 = 重复上一条命令 (GDB 风格)
            if not raw and self._last_command is None:
                continue
            elif not raw:
                assert self._last_command is not None
                raw = self._last_command
                # disasm 重复时自动推进到上次输出末尾地址
                if raw.startswith("disasm ") and self._disasm_next_addr is not None:
                    parts = raw.split()
                    parts[1] = hex(self._disasm_next_addr)
                    raw = " ".join(parts)
                if raw.startswith("vdisasm ") and self._vdisasm_next_addr is not None:
                    parts = raw.split()
                    parts[1] = hex(self._vdisasm_next_addr)
                    raw = " ".join(parts)
                self._console.print(f"[dim](重复) {raw}[/]")

            parts = raw.split()
            try:
                if not self._dispatch(parts):
                    break
            except Exception:
                self._console.print_exception()
            else:
                # dispatch 成功后存储命令 (忽略解析异常时的空命令)
                self._last_command = raw

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._console.print("[dim]调试器已退出[/]")

    def _print_help(self) -> None:
        """显示帮助 — 使用 Rich Table 以保证中英文混排对齐."""

        def _section(title: str, rows: list[tuple[str, str]]) -> Table:
            tab = Table(title=title, border_style="green", show_header=False)
            tab.add_column("cmd", style="cyan", no_wrap=True)
            tab.add_column("desc")
            for cmd, desc in rows:
                tab.add_row(cmd, desc)
            return tab

        sections: list[Table] = [
            _section("执行控制", [
                ("s/step [n]", "单步执行 n 条指令 (默认 1)"),
                ("c/continue", "连续执行 (直到 Ctrl+C 暂停)"),
                ("r/run [n]", "执行 n 条指令 (默认 1)"),
                ("undo/rollback", "回滚上一条指令"),
                ("restart", "重置 hart/CLINT, 重新加载固件"),
                ("b/bp <addr>", "在指定地址设置断点"),
                ("b/bp ecall|ebreak|mret|sret|wfi", "在指定指令类型设置断点"),
                ("b/bp opcode <hex>", "在指定 opcode 设置断点 (如 0x73)"),
                ("b/bp", "列出所有断点"),
                ("bp delete <n>", "删除编号为 n 的断点"),
                ("bp clear", "清除全部断点"),
            ]),
            _section("读寄存器命令", [
                ("regs/gpr", "显示全部 GPR (x0–x31)"),
                ("reg <name>", "显示指定 GPR (例: reg a0, reg x10)"),
                ("csr <name>", "显示指定 CSR (例: csr mstatus)"),
                ("csr list", "列出所有可用 CSR 名称"),
                ("pc", "显示当前 PC 及反汇编"),
                ("mode", "显示当前特权级"),
                ("mstatus", "显示 mstatus 各字段分解"),
                ("tlb [vpn]", "显示 ITLB / DTLB, 或查找指定 VPN"),
                ("cache [set] [way]", "显示 L2 缓存状态及数据"),
                ("satp", "显示 satp 解码 (MODE/ASID/PPN)"),
                ("pt [va]", "Sv39 页表遍历: 逐级显示 PTE 与权限"),
                ("show-pmp", "显示全部 PMP 条目的保护范围与权限"),
            ]),
            _section("写寄存器命令", [
                ("set/w <name> <val>", "写入 GPR (例: set sp 0x8000)"),
                ("csrw <name> <val>", "写入 CSR (例: csrw mtvec 0x80000001)"),
                ("pc <addr>", "设置 PC 并显示反汇编"),
            ]),
            _section("内存/符号", [
                ("mem <addr> [size]", "hexdump 给定物理地址下的内存 (默认 64 字节)"),
                ("vmem <addr> [size]", "hexdump 给定虚拟地址下的内存 (默认 64 字节)"),
                ("disasm <addr> [count]", "对物理地址起的 count 条指令做反汇编 (默认 16)"),
                ("vdisasm <addr> [count]", "对虚拟地址起的 count 条指令做反汇编 (默认 16)"),
                ("sym/symbols [filt]", "列出符号表 (可选过滤)"),
            ]),
            _section("状态", [
                ("status/info [hart]", "全部 hart 概览, 或指定 hart 详情"),
                ("stack/bt", "栈帧情况"),
                ("frame/f <N>", "切换到第 N 帧"),
            ]),
            _section("配置", [("hart <id>", "切换活跃 hart")]),
            _section("其他", [
                ("help/h/?", "显示本帮助"),
                ("quit/q/exit", "退出调试器"),
            ]),
        ]

        for tab in sections:
            self._console.print(tab)


# ============================================================
#  CLI 入口
# ============================================================


def main(args: list[str] | None = None) -> None:
    """命令行入口: 加载固件并启动交互调试器."""
    parser = argparse.ArgumentParser(
        description="RISC-V Interactive Debugger (rvdb)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  %(prog)s firmware.bin\n"
        "  %(prog)s firmware.elf --entry-symbol main\n"
        "  %(prog)s app.elf --preload crt0.bin --entry-symbol main\n"
        "  %(prog)s firmware.bin --harts 4 --ram 256M",
    )
    parser.add_argument("firmware", help="固件文件路径 (ELF / PE / raw binary)")
    parser.add_argument(
        "--base-addr",
        type=lambda x: int(x, 0), default=0x80000000,
        help="raw binary 的加载基址 (ELF/PE 时忽略, 默认 0x80000000)",
    )
    parser.add_argument(
        "--prog-cnt",
        type=lambda x: int(x, 0), default=None,
        help="程序计数器 (Program Counter) — 每个 hart 的开始执行地址 (默认使用固件入口 + 搬迁偏移)",
    )
    parser.add_argument(
        "--ram-base",
        type=lambda x: int(x, 0), default=0x8000_0000,
        help="物理内存基址 (默认 0x80000000, 加载低地址固件时需设为 0x0)",
    )
    parser.add_argument(
        "--fdt",
        type=lambda x: int(x, 0) if x is not None else None,
        default=-1, nargs="?", const=-1,
        help="设备树加载地址 (默认自动放在 RAM 顶端 −64 KiB, 接参数则放在指定地址)",
    )
    parser.add_argument(
        "--fdt-file",
        type=str, default=None,
        metavar="PATH[:ADDR]",
        help="加载预编译 DTB 文件 (可选 :地址), 例: --fdt-file pyremu_virt.dtb:0x7f0000",
    )
    parser.add_argument(
        "--no-fdt",
        action="store_true", default=False,
        help="禁用设备树 (--fdt 和 --fdt-file 均被忽略)",
    )
    parser.add_argument("--harts", type=int, default=1, help="hart 数量 (默认 1)")
    parser.add_argument(
        "--ram",
        type=str, default="128M",
        help="RAM 大小 (支持 K/M/G 后缀, 默认 128M)",
    )
    parser.add_argument(
        "--log-level",
        type=str, default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别 (默认 INFO)",
    )
    parser.add_argument(
        "--preload",
        type=str, default=None,
        help="预加载 shellcode 文件路径 (裸 RISC-V 机器码), 在目标程序之前执行",
    )
    parser.add_argument(
        "--preload-addr",
        type=lambda x: int(x, 0), default=None,
        help="预加载 shellcode 的加载地址 (默认 = RAM 顶端 − 64 KiB)",
    )
    parser.add_argument(
        "--entry-symbol",
        type=str, default=None,
        help="目标程序入口符号名 (如 'main'), 将其地址写入 a0 供 preload 使用",
    )

    ns = parser.parse_args(args)

    # 配置 loguru: 彩色输出到 stdout
    logger.remove()
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:HH:mm:ss}</green> "
            "[<yellow>{file}:{line}</yellow>|<level>{level}</level>] "
            "<level>{message}</level>"
        ),
        colorize=True,
        level=ns.log_level,
    )

    # 解析 RAM 大小
    ram_str = ns.ram.upper()
    ram_mul = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if ram_str[-1] in ram_mul:
        ram_size = int(ram_str[:-1]) * ram_mul[ram_str[-1]]
    else:
        ram_size = int(ram_str)

    # 加载固件
    fw_path = ns.firmware
    if not Path(fw_path).exists():
        logger.error(f"文件不存在: {fw_path}")
        sys.exit(1)

    logger.info(f"正在解析固件: {fw_path}")
    image = parse_firmware(fw_path, base_addr=ns.base_addr)
    if image is None:
        logger.error(f"无法解析固件: {fw_path}")
        sys.exit(1)

    logger.info(
        f"格式={image.format}, 入口=0x{image.entry_point:x}, 段数={len(image.segments)}"
    )
    for seg in image.segments:
        logger.info(
            f"  <{seg.name}> ofs=0x{seg.vaddr:08x}  " +
            f"size={Debugger._fmt_size(len(seg.data)):<8s} mem-size={Debugger._fmt_size(seg.memsz)}"
        )

    # PIE 固件迁移: 若所有段 vaddr 均 < ram_base, 整体偏移 ram_base
    load_offset = 0
    if image.format == "elf":
        min_vaddr = min(seg.vaddr for seg in image.segments)
        if min_vaddr < ns.ram_base:
            load_offset = ns.ram_base
            logger.info(
                f"PIE 固件段偏移 +0x{load_offset:x}"
                f" (min vaddr=0x{min_vaddr:x} < ram_base=0x{ns.ram_base:x})"
            )

    # 确定 prog_cnt: 用户显式指定优先, 否则用搬迁后入口
    effective_entry = image.entry_point + load_offset
    prog_cnt = ns.prog_cnt if ns.prog_cnt is not None else effective_entry

    # 创建模拟器并加载
    plat_cfg = PlatformConfig(
        num_harts=ns.harts,
        ram_size=ram_size,
        ram_base=ns.ram_base,
        prog_cnt=prog_cnt,
        periph=PeripheralConfig(),
    )
    emu = Emulator(plat_cfg)
    emu.load_firmware(image, load_offset=load_offset)
    for h in emu.harts:
        h.pc = prog_cnt

    # --fdt: 默认生成设备树, 写入 RAM, 设 a1 供固件发现外设
    # --no-fdt 可显式禁用 (固件自带 DTB 或裸金属程序不需要设备树时使用)
    # --fdt / --fdt-file: 生成或加载设备树, 写入 RAM, 设 a1
    fdt_addr = None
    if not ns.no_fdt:
        if ns.fdt_file is not None:
            # 预编译 DTB 文件: 路径[:地址]
            parts = ns.fdt_file.rsplit(":", 1)
            file_path = parts[0]
            custom_addr = int(parts[1], 0) if len(parts) > 1 else None
            fdt_addr = (
                custom_addr if custom_addr is not None else (ns.ram_base + ram_size - 0x10000)
            )
            emu.load_dtb_file(file_path, addr=fdt_addr)
        elif ns.fdt is not None:
            fdt_addr = ns.fdt if ns.fdt != -1 else (ns.ram_base + ram_size - 0x10000)
            emu.load_dtb(fdt_addr)
    fdt_note = f", FDT=0x{fdt_addr:x}" if fdt_addr is not None else ""
    if ns.no_fdt:
        fdt_note = ", DTB=off"
    logger.info(f"固件已写入 RAM, {ns.harts} hart(s) 就绪, PC=0x{prog_cnt:x}{fdt_note}")

    # 预加载 shellcode (外部提供的裸 RISC-V 机器码)
    preload_entry = None
    if ns.preload is not None:
        preloader = Preloader(emu)
        preload_entry = preloader.inject_file(ns.preload, addr=ns.preload_addr)
        logger.info(f"预加载载荷: {ns.preload} → 0x{preload_entry:x}")
        # misa 需设 U-bit (bit20), sbi_init 冷启动依赖
        for h in emu.harts:
            h.csrs["misa"].val = (
                (2 << 62) | (1 << 18) | (1 << 20)
            )  # MXL=RV64 + U-bit; h.csrs["mscratch"].val = ns.ram_base + ram_size - 0x100000
        # fw_next_arg1 返回 ram_base + 0x2200000, fdt_get_address() 也读这里
        # 在 fdt_get_address 期望的地址放一份 DTB 副本
        if fdt_addr is not None:
            emu.load_dtb_blob(ns.ram_base + 0x2200000, emu.build_dtb())

    # 若指定了 entry-symbol, 将符号地址写入 a0 (x10) 供 preload 使用
    target_entry = None
    if ns.entry_symbol is not None:
        target_entry = image.symbols.get(ns.entry_symbol)
        if target_entry is None:
            logger.error(f"符号不存在: {ns.entry_symbol}")
            sys.exit(1)
        for h in emu.harts:
            h.write_gpr(10, target_entry)  # a0 = target address
        logger.info(f"a0 ← {ns.entry_symbol} = 0x{target_entry:x}")

    # 设置 PC: preload 优先, 否则使用固件自带入口
    if preload_entry is not None:
        for h in emu.harts:
            h.pc = preload_entry
    elif target_entry is not None:
        # 无 preload 但有 entry-symbol: 直接跳转 (裸跑)
        for h in emu.harts:
            h.pc = target_entry
        logger.warning("无 --preload, 直接跳转到入口 (未设置 sp/gp, 可能出错)")

    dbg = Debugger(emulator=emu, hart_id=0, image=image)
    dbg._fdt_addr = fdt_addr
    dbg._load_offset = load_offset
    dbg._preload_path = ns.preload
    dbg.repl()

if __name__ == "__main__":
    main()
