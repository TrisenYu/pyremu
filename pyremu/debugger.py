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
import os
import signal
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pyremu.core.decoder import Hart, Opc, parse_compressed
from pyremu.core.hart import RiscvMode
from pyremu.core.mem_check_aux import inject_memory_backend
from pyremu.core.registers import (
    csr_addr_from_name,
    gpr_idx_from_name,
    register_csr,
    register_fpr,
    register_gpr,
)
from pyremu.core.trap import TrapType, trap_cause_name
from pyremu.core.trap_handler import check_pending_interrupts, deliver_trap
from pyremu.emulator import Emulator
from pyremu.env_inject import Preloader
from pyremu.memory.l2cache import L2Cache
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils.disassem import disasm
from pyremu.utils.parse_bin import FirmwareImage, FirmwareSegment, parse_firmware
from pyremu.utils.str_aux import fmt_hexdump
from pyremu.utils.wrapper import seize_val_err

# ============================================================
#  状态快照 (用于后续回滚已执行指令的特性)
# ============================================================


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


# ============================================================
#  断点
# ============================================================


@dataclass
class _Breakpoint:
    """断点 — 地址 / 指令类型 / opcode, 可选条件."""

    kind: str  # "addr" | "instr" | "opcode" | "cond"
    value: int  # address, funct12, opcode, or 0 for cond
    desc: str  # human-readable
    cond_type: str = ""        # "" | "reg" | "csr"
    cond_reg: str = ""         # 寄存器名
    cond_op: str = "=="        # 比较运算符
    cond_val: int = 0          # 期望值


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
    ("EB", 1<<60),
    ("PB", 1<<50),
    ("TB", 1<<40),
    ("GB", 1<<30),
    ("MB", 1<<20),
    ("KB", 1<<10),
)

MAX_INSTR_COUNT = 100_000  # 指令条数上限, 超界视为不可达
# ============================================================
#  交互式调试器
# ============================================================


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

        # disasm 步数持久化: 参考 PC 及累积步数, 供 Enter 重复时继续编号
        self._disasm_ref_pc: int | None = None
        self._disasm_base_step: int = 0
        self._disasm_past_terminator: bool = False

        # 断点
        self._breakpoints: list[_Breakpoint] = []
        self._bp_mode: str = "sync"       # "sync"=全停, "async"=仅当前 hart
        self._hart_paused: set[int] = set()  # async 模式下被暂停的 hart
        self._bp_hit_this_run: set[tuple[str, int]] = set()
        self._prev_instr_csr_addr: int = -1  # 上一条指令的 CSR 地址 (供 CSR 条件后检)

        # 最近的栈回溯帧列表 (供 frame N 选择)
        self._stack_frames: list[StackFrame] = []
        self._current_frame_idx: int = 0

        # prompt_toolkit REPL — 方向键历史, Tab 补全, 持久化历史文件
        self._history: FileHistory = FileHistory(
            os.path.expanduser("~/.pyremu_history")
        )
        self._completer: WordCompleter = self._build_completer()
        self._session: PromptSession[str] = PromptSession(
            history=self._history,
            completer=self._completer,
            style=Style.from_dict({
                "prompt": "#00aa00 bold",
                "": "#cccccc",
            }),
        )

    # ==========================================================
    #  Tab 补全 — 名称全部来自 core/registers 工厂函数
    # ==========================================================

    def _build_completer(self) -> WordCompleter:
        words: list[str] = [
            # 执行控制
            "s", "step", "c", "continue", "r", "run",
            "undo", "rollback", "restart",
            "b", "bp",
            # 寄存器 / CSR 操作
            "regs", "gpr", "reg", "set", "w",
            "csr", "csrw",
            # 状态 / 内存
            "pc", "mode", "mstatus",
            "tlb", "tlbflush", "cache", "satp",
            "mem",
            "status", "info",
            "symbols", "sym",
            "disasm",
            "stack", "bt", "frame", "f",
            # 配置 & 帮助
            "hart", "h", "help", "?",
            "q", "quit", "exit",
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
    #  hart 访问
    # ==========================================================

    @property
    def hart(self):
        return self._emu.harts[self._hart_id]

    # ==========================================================
    #  信号处理
    # ==========================================================

    def _sigint_repl(self, signum: int, frame) -> None:
        raise KeyboardInterrupt

    def _sigint_run(self, signum: int, frame) -> None:
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
            if (instr & 0x7F) == 0x73 and ((instr >> 20) & 0xFFF) == target:
                return True
            # 上一条指令写了目标 CSR (当前指令可见写入后的新值)
            if self._prev_instr_csr_addr == target:
                return True
            return False
        elif bp.cond_type == "reg":
            rd = (instr >> 7) & 0x1F
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
                hit = (pc == bp.value)
            elif bp.kind == "instr":
                opcode = instr & 0x7F
                funct3 = (instr >> 12) & 0x7
                funct12 = (instr >> 20) & 0xFFF
                hit = (opcode == 0x73 and funct3 == 0 and funct12 == bp.value)
            elif bp.kind == "opcode":
                # opcode 断点仅适用 32-bit 标准指令: 压缩指令的
                # instr & 0x7F 不能可靠得出象限值, 跳过
                if parse_compressed(instr):
                    continue
                hit = ((instr & 0x7F) == bp.value)

            if not hit:
                continue

            # 条件评估
            if not self._eval_bp_condition(hart, bp):
                continue

            # 本次 continue 内同一 PC 的地址断点只停一次 (避免 c 反复卡在同一处)
            bp_key = (bp.kind, pc) if bp.kind == "addr" else (bp.kind, id(bp))
            if bp_key in self._bp_hit_this_run:
                continue

            self._bp_hit_this_run.add(bp_key)
            self._report_bp_hit(bp, hart, pc)
            return True

        # 记录当前指令的 CSR 地址, 供下一条指令的条件断点后检
        if (instr & 0x7F) == 0x73:
            self._prev_instr_csr_addr = (instr >> 20) & 0xFFF
        else:
            self._prev_instr_csr_addr = -1
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
                    hit = (h.pc == bp.value)
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
                        opcode = instr & 0x7F
                        funct3 = (instr >> 12) & 0x7
                        funct12 = (instr >> 20) & 0xFFF
                        hit = (opcode == 0x73 and funct3 == 0 and funct12 == bp.value)
                    elif not parse_compressed(instr):
                        hit = ((instr & 0x7F) == bp.value)
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
            gpr_vals=[r.val for r in h.gprs],
            csr_vals={name: csr.val for name, csr in h.csrs.items()},
            mode=h.mode.value,
            reservation_valid=h.reservation_valid,
            reservation_addr=h.reservation_addr,
        )

    def _restore_snapshot(self, snap: HartSnapshot) -> None:
        h = self.hart
        h.pc = snap.pc
        for i, val in enumerate(snap.gpr_vals):
            h.gprs[i].val = val
        for name, val in snap.csr_vals.items():
            if name in h.csrs:
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
            instr_bytes = self._emu.bus.read(h.pc, 4)
            instr = int.from_bytes(instr_bytes, "little", signed=False)

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
            if cond_type:
                self._breakpoints[-1].cond_type = cond_type
                self._breakpoints[-1].cond_reg = cond_reg
                self._breakpoints[-1].cond_op = cond_op
                self._breakpoints[-1].cond_val = cond_val
                self._breakpoints[-1].desc += f" if {cond_type} {cond_reg}{cond_op}0x{cond_val:x}"
                self._console.print(f"    条件: {cond_type} {cond_reg} {cond_op} 0x{cond_val:x}")
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
            bp = _Breakpoint(kind="addr", value=addr, desc=desc,
                             cond_type=cond_type, cond_reg=cond_reg,
                             cond_op=cond_op, cond_val=cond_val)
            self._breakpoints.append(bp)
            asm_info = self._fetch_and_disasm(addr)
            asm_str = f"  {asm_info[1]}" if asm_info else ""
            cond_str = f" [bold yellow]if[/] {cond_type} {cond_reg} {cond_op} [yellow]0x{cond_val:x}[/]" if cond_type else ""
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
                kind="instr", value=_INSTR_BP_NAMES[name], desc=name,
            )
            self._breakpoints.append(bp)
            self._console.print(
                f"  [green]断点 {len(self._breakpoints)}[/]  "
                f"[yellow]{name}[/]"
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
            kind="opcode", value=val, desc=f"opcode 0x{val:02x}",
        )
        self._breakpoints.append(bp)
        self._console.print(
            f"  [green]断点 {len(self._breakpoints)}[/]  "
            f"[yellow]opcode 0x{val:02x}[/]"
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
        self._console.print(
            f"  [dim]已删除断点 [yellow]{removed.desc}[/][/]"
        )

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
        elif sub == "list":
            self.cmd_bp_list()
        elif sub == "delete":
            if len(rest) > 1:
                self.cmd_bp_delete(rest[1])
            else:
                self._warn("用法: bp delete <编号>")
        elif sub == "clear":
            self.cmd_bp_clear()
        elif sub == "if":
            # 纯条件断点: bp if reg/csr <name> <op> <val>
            if len(rest) >= 4 and rest[1] in ("reg", "csr"):
                ct, cr, co = rest[1], rest[2], rest[3] if len(rest) > 3 else "=="
                cv = int(rest[4], 0) if len(rest) > 4 else 0
                desc = f"if {ct} {cr}{co}0x{cv:x}"
                bp = _Breakpoint(kind="cond", value=0, desc=desc,
                                 cond_type=ct, cond_reg=cr, cond_op=co, cond_val=cv)
                self._breakpoints.append(bp)
                self._console.print(
                    f"  [green]条件断点 {len(self._breakpoints)}[/]  "
                    f"[bold yellow]if[/] {ct} {cr} {co} [yellow]0x{cv:x}[/]"
                )
            else:
                self._warn("用法: bp if reg/csr <name> <op> <val>")
        elif sub == "opcode":
            if len(rest) > 1:
                self.cmd_bp_opcode(rest[1])
            else:
                self._warn("用法: bp opcode <hex>")
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
            from pyremu.env_inject import Preloader
            preload_entry = Preloader(emu).inject_file(self._preload_path)
            # BSS patch: blt → j <after_bss> (0x94, 4 字节对齐)
            j_imm = (0x98 - 0x94) >> 1
            j_blt = ((j_imm >> 19) & 1) << 31 | (j_imm & 0x3FF) << 21 | ((j_imm >> 10) & 1) << 20 | ((j_imm >> 11) & 0xFF) << 12 | 0x6f
            emu.bus.write(cfg.ram_base + 0x94, struct.pack("<I", j_blt))
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

    def _run_loop(self, cycles: int | None = None) -> None:
        """运行循环: 每个周期所有 hart 各执行一条指令 (round-robin).

        多 hart 时通过 emu.step() 驱动全部 hart;
        单 hart 时保持 step_one() 以支持快照/回滚.
        """
        self._running = True
        self._paused = False
        self._terminated = False
        self._sigint_count = 0
        self._enter_run_mode()

        multi = self._emu.num_harts > 1

        try:
            executed = 0
            while not self._terminated and not self._paused:
                if cycles is not None and executed >= cycles:
                    break
                if all(h._halted for h in self._emu.harts):
                    self._warn("所有 Hart 已暂停")
                    break
                if self.hart._halted:
                    self._show_trap_context(self.hart)
                    break
                # async 模式: 当前 hart 被断点暂停时回到 REPL
                if self._bp_mode == "async" and self._hart_id in self._hart_paused:
                    self._console.print(
                        f"  [dim]Hart {self._hart_id} 断点暂停, 回到 REPL[/]"
                    )
                    break
                if multi:
                    self._emu.step()
                    self._check_multi_hart_bp()
                else:
                    self.step_one()
                self._instr_count += 1
                executed += 1
            if self._terminated:
                self._console.print("[dim]模拟循环已终止[/]")
        finally:
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
            if self.hart._halted:
                self._show_trap_context(self.hart)
                return
        self.cmd_pc()

    def cmd_continue(self) -> None:
        self._console.print("[dim]继续执行 (Ctrl+C 暂停)...[/]")
        self._run_loop()

    def cmd_run(self, n: int = 1) -> None:
        self._console.print(f"[dim]执行 {n} 条指令...[/]")
        self._run_loop(cycles=n)

    def cmd_regs(self) -> None:
        h = self.hart
        tbl = Table(title=f"Hart {self._hart_id}  GPRs", border_style="blue")
        tbl.add_column("Reg", style="cyan", no_wrap=True)
        tbl.add_column("Value", style="green")
        tbl.add_column("Reg", style="cyan", no_wrap=True)
        tbl.add_column("Value", style="green")
        for i in range(16):
            lo = h.gprs[i]
            hi = h.gprs[i + 16]
            tbl.add_row(
                f"{lo.name} ({lo.alias})", self._hex(lo.val),
                f"{hi.name} ({hi.alias})", self._hex(hi.val),
            )
        self._console.print(tbl)

    def cmd_reg(self, raw_args: str) -> None:
        """读取 GPR: reg <name>  或  reg <n1>, <n2>, ... (逗号分隔多寄存器)."""
        wanted = [n.strip() for n in raw_args.split(",") if n.strip()]
        if not wanted:
            return

        # 验证
        regs = []
        for n in wanted:
            r = self._find_gpr(n)
            if r is None:
                self._err(f"未知寄存器: {n}")
                return
            regs.append(r)

        # 对齐: 名称宽度 + 别名宽度
        max_w = max(len(r.name) for r in regs)
        max_a = max(len(r.alias) for r in regs)

        lines = []
        for r in regs:
            lines.append(
                f"  [cyan]{r.name:<{max_w}}[/] "
                f"([dim]{r.alias:<{max_a}}[/]) = "
                f"[green]{self._hex(r.val)}[/]"
            )
        self._console.print("\n".join(lines))

    @seize_val_err("无效值")
    def cmd_set(self, name: str, value: str) -> None:
        """写入 GPR: set <name> <value>."""
        r = self._find_gpr(name)
        if r is None:
            self._err(f"未知寄存器: {name}")
            return
        v = int(value, 0) & 0xFFFF_FFFF_FFFF_FFFF
        old = r.val
        r.val = v
        self._console.print(
            f"{r.name} ([cyan]{r.alias}[/]): "
            f"[yellow]{self._hex(old)}[/] → [green]{self._hex(v)}[/]"
        )

    def _resolve_addr(self, arg: str) -> int | None:
        """将字符串解析为地址: pc / 寄存器名 / 数值.

        Returns:
            解析出的 64-bit 地址, 或 None (无法解析 / 寄存器不存在).
        """
        if arg.lower() == "pc":
            return self.hart.pc
        # 尝试寄存器名
        r = self._find_gpr(arg)
        if r is not None:
            return r.val & 0xFFFF_FFFF_FFFF_FFFF
        # 尝试数值
        try:
            v = int(arg, 0)
            if v < 0 or v >= (1 << 64):
                return None
            return v
        except ValueError:
            return None

    def _find_gpr(self, name: str):
        """按 xN 或 ABI 名查找 GPR."""
        h = self.hart
        if name.startswith("x"):
            try:
                idx = int(name[1:])
                if 0 <= idx <= 31:
                    return h.gprs[idx]
            except ValueError:
                pass
        for r in h.gprs:
            if r.alias.lower() == name.lower():
                return r
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
            line = (
                f"  [cyan]{n:<{max_w}}[/] = "
                f"[green]{self._hex(csr.val)}[/]"
                f" (dec: {csr.val})"
            )
            # mstatus / mstatush 附加当前特权级
            if n == "mstatus":
                line += f"  [dim]模式 [bold]{h.mode.name}[/][/]"
            elif n == "mstatush":
                line += f"  [dim]模式 [bold]{h.mode.name}[/][/]"
            # mcause / scause 解码陷态原因
            elif n in ("mcause", "scause"):
                cause_str = self._trap_cause_name(csr.val)
                line += f"  [dim]{cause_str}[/]"
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
        old = csr.val
        csr.val = v
        self._console.print(
            f"[cyan]{name}[/]: "
            f"[yellow]{self._hex(old)}[/] → [green]{self._hex(v)}[/]"
        )

    def _fetch_and_disasm(self, pc: int) -> tuple[str, str] | None:
        """读取 PC 处指令字并反汇编, 返回 (raw_hex, asm); IO 失败返回 None."""
        raw = self._emu.bus.try_read(pc, 4)
        if raw is None:
            return None
        instr = int.from_bytes(raw, "little", signed=False)
        asm = disasm(instr, pc)
        raw_hex = " ".join(f"{b:02x}" for b in raw[:4])
        return raw_hex, asm

    def _warn_pc_if_suspect(self, h, v: int) -> None:
        """根据 MMU 模式检查 PC 并给出潜在无效地址的软警告.

        Bare 模式: v 是物理地址, 检查是否在有效 RAM/设备范围.
        Sv39 模式: v 是虚拟地址, 检查 bits[63:39] 是否等于 bit 38.
        """
        mode = h.mmu_mode
        if mode == 0:  # Bare — 物理地址
            bus = h._bus
            if bus is not None and not bus.is_valid_addr(v):
                self._warn(
                    f"Bare 模式下 PC 0x{v:016x} 不在有效物理地址范围 "
                    f"[0x{bus.ram_base:x}, 0x{bus._ram_end:x}) 或已注册设备区域"
                )
        elif mode == 8:  # Sv39 — 虚拟地址
            sign_bit = (v >> 38) & 1
            if sign_bit:
                expected = 0xFFFFFFFFC0000000 | (v & 0x3FFFFFFFFF)
            else:
                expected = v & 0x3FFFFFFFFF
            if v != expected:
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

            old = h.pc
            h.pc = v
            self._snapshot = None
            self._mem_changes = []
            self._console.print(
                f"PC: [yellow]{self._hex(old)}[/] → [green]{self._hex(h.pc)}[/]"
            )
        pc = h.pc
        result = self._fetch_and_disasm(pc)
        if result is None:
            raw_hex, asm = "(无法读取)", "(无法解码)"
        else:
            raw_hex, asm = result
        lines = [
            f"PC  = [bold yellow]{self._hex(pc)}[/]  "
            f"[dim]模式 [bold cyan]{h.mode.name}[/][/]"
            + ("  [dim]WFI等待[/]" if h._waiting else ""),
            f"Raw = [dim]{raw_hex}[/]",
            f"[bold green]  {asm}[/]",
        ]
        # 仅在 mcause 刚发生变化时展示陷态上下文 (避免 handler 执行期间重复刷屏)
        mcause = h.mcause_val
        if mcause != 0 and mcause != self._trap_displayed_mcause:
            self._trap_displayed_mcause = mcause
            kind = "中断" if (mcause >> 63) & 1 else "异常"
            cause_str = self._trap_cause_name(mcause)
            lines.extend([
                "",
                f"[bold red]▸ Trap 上下文 ({kind}):[/]",
                f"  mcause = {self._hex(mcause)} ([cyan]{cause_str}[/])",
                f"  mepc   = {self._hex(h.mepc_val)}",
                f"  mtval  = {self._hex(h.mtval_val)}",
                f"  mstatus= {self._hex(h.mstatus_val)}"
                f"  (MIE={(h.mstatus_val >> 3) & 1}, MPP={(h.mstatus_val >> 11) & 3})",
            ])
        elif mcause == 0:
            self._trap_displayed_mcause = None
        self._console.print("\n".join(lines))

    def cmd_mode(self) -> None:
        h = self.hart
        self._console.print(f"Mode = [bold cyan]{h.mode.name}[/] ({h.mode.value})")

    def cmd_mstatus(self) -> None:
        h = self.hart
        v = h.mstatus_val
        fields = [
            ("MIE", (v >> 3) & 1), ("MPIE", (v >> 7) & 1),
            ("MPP", (v >> 11) & 0b11), ("SIE", (v >> 1) & 1),
            ("SPIE", (v >> 5) & 1), ("SPP", (v >> 8) & 1),
            ("MPRV", (v >> 17) & 1), ("SUM", (v >> 18) & 1),
            ("MXR", (v >> 19) & 1), ("TVM", (v >> 20) & 1),
            ("TW", (v >> 21) & 1), ("TSR", (v >> 22) & 1),
            ("FS", (v >> 13) & 0b11), ("SD", (v >> 63) & 1),
        ]
        tbl = Table(title=f"mstatus = {self._hex(v)}", border_style="magenta")
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Value", style="green")
        for fname, fval in fields:
            tbl.add_row(fname, str(fval))
        self._console.print(tbl)

    # ----------------------------------------------------------
    #  TLB helpers
    # ----------------------------------------------------------

    @staticmethod
    def _decode_perm(perm: int) -> str:
        """TLB 权限位 → 可读字符串: 0b1111 → 'RWXU'."""
        r = "R" if perm & 1 else "-"
        w = "W" if perm & 2 else "-"
        x = "X" if perm & 4 else "-"
        u = "U" if perm & 8 else "S"
        return f"{r}{w}{x}{u}"

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
                self._decode_perm(e.perm),
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
                    f"perm={self._decode_perm(e.perm)}  "
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

    @seize_val_err("set/way索引需要为整数")
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
                a, _, b = first.partition("-")
                range_start = int(a, 0)
                range_end = int(b, 0)
                if not (0 <= range_start <= range_end < num_sets):
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
            if e.dirty:
                dirty_count += 1

        header = (
            f"[bold]L2 Cache[/]: {valid_count}/{len(entries)} valid, "
            f"{dirty_count} dirty, "
            f"{ways}-way × {num_sets} sets, "
            f"line={l2.line_size} B, "
            f"hit_rate={l2.hit_rate:.3f}\n"
            "MESI: " + " ".join(
                f"{s}={mesi_counts.get(s, 0)}"
                for s in ("MODIFIED", "EXCLUSIVE", "SHARED", "INVALID")
            )
        )

        # ---- 确定要扫描的 set 范围 ----
        if range_start is not None and range_end is not None:
            set_range = range(range_start, range_end + 1)
            full_dump = False          # 范围模式: 紧凑预览
            limit = None               # 不限条目数
        elif target_set is not None:
            set_range = [target_set]
            full_dump = True           # 单 set 模式: 完整 hexdump
            limit = None
        else:
            set_range = range(num_sets)
            full_dump = False          # 默认模式: 紧凑预览
            limit = self._DEFAULT_CACHE_LINES

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

    @staticmethod
    def _ctrl_flow_kind(instr: int) -> str:
        """返回指令的控制流类型: 'term' (终止), 'branch' (条件分支), 'normal'.

        仅当 *instr* 不是压缩指令时才准确; 压缩指令需先解码.
        """
        opcode = instr & 0x7F
        # JAL / JALR — 无条件跳转 (含 call / j / ret / jr)
        if opcode in (0b1101111, 0b1100111):
            return "term"
        # 条件分支
        if opcode == 0b1100011:
            return "branch"
        # SYSTEM — 仅 ecall / ebreak / mret / sret 为终止
        if opcode != 0b1110011:
            return "normal"
        funct3 = (instr >> 12) & 0x7
        if funct3 != 0:
            return "normal"
        funct12 = (instr >> 20) & 0xFFF
        if funct12 in (0x000, 0x001, 0x302, 0x102):
            return "term"
        return "normal"

    @staticmethod
    def _ctrl_flow_kind_compressed(instr16: int) -> str:
        """压缩指令 (16-bit) 的控制流类型."""
        quad = instr16 & 0x3
        funct3 = (instr16 >> 13) & 0x7
        if quad == 0b01:          # C1 象限
            if funct3 in (0b001, 0b101):   # C.JAL, C.J
                return "term"
            if funct3 in (0b110, 0b111):   # C.BEQZ, C.BNEZ
                return "branch"
        elif quad == 0b10:        # C2 象限
            if funct3 == 0b100:            # C.JR / C.JALR / C.EBREAK
                return "term"
        return "normal"

    @seize_val_err("addr 和 length 需为整数 (支持 0x 前缀)")
    def cmd_disasm(self, addr_str: str, length_str: str = "64") -> None:
        """反汇编指定内存区域.

        disasm <addr> [length]  — 从 addr 开始反汇编 length 字节 (默认 64).

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
        length = int(length_str, 0)

        if addr is None:
            self._err(f"无法解析地址: {addr_str}")
            return

        if length <= 0 or length > 4096:
            self._err("length 需在 1–4096 之间")
            return

        if not self._check_rv64_addr(addr):
            return

        # 非 2-字节对齐: RISC-V 指令至少 16-bit 对齐, 奇数地址解码毫无意义
        if addr & 1:
            orig = addr
            addr &= ~1
            self._warn(
                f"addr 至少应为 2-字节对齐, "
                f"已从 0x{orig:016x} 对齐到 0x{addr:016x}"
            )

        raw = self._emu.bus.try_read(addr, length)
        if raw is None:
            self._err("无法读取指定地址")
            return

        # ---- 判断是否为重复执行 (Enter 自动推进) ----
        is_continue = (
            self._disasm_ref_pc is not None
            and addr == self._disasm_next_addr
        )

        # ---- 第一遍: 收集指令元组 (addr, raw_hex, asm, ctrl_kind) ----
        instrs: list[tuple[int, str, str, str]] = []
        offset = 0
        max_offset = len(raw)

        while offset < max_offset:
            pc_addr = addr + offset
            remaining = max_offset - offset

            chunk = raw[offset : offset + min(4, remaining)]
            instr = int.from_bytes(
                chunk.ljust(4, b"\x00"), "little", signed=False
            )

            is_compressed = parse_compressed(instr)
            inst_size = 2 if is_compressed else 4

            if remaining < inst_size:
                # 尝试从内存多读几个字节以补全指令, 避免在窗口边界截断
                extra = self._emu.bus.try_read(pc_addr + offset, inst_size)
                if extra is not None:
                    raw = raw[:offset] + extra + raw[offset + len(extra):]
                    remaining = inst_size
                else:
                    leftover = raw[offset:]
                    hex_s = " ".join(f"{b:02x}" for b in leftover)
                    instrs.append((pc_addr, hex_s, "[dim](截断)[/]", "normal"))
                    break

            asm = disasm(instr, pc_addr)
            raw_hex = " ".join(f"{b:02x}" for b in raw[offset:offset + inst_size])
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
                step_cnt = self._count_instrs_between(
                    self._disasm_ref_pc, addr
                )
                self._disasm_base_step = step_cnt
                if step_cnt >= MAX_INSTR_COUNT:
                    self._disasm_past_terminator = True  # 区间过大, 禁用步数
            else:
                step_cnt = self._count_instrs_between(
                    addr, self._disasm_ref_pc
                )
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
            at_ref = (pc_addr == ref_pc)

            # 确定前缀
            if at_ref:
                prefix = "pc ->"
            elif step > 0 and not past_term:
                prefix = f"+{step}"
                prefix_w = max(prefix_w, len(prefix))
            else:
                prefix = ""

            # 段/函数边界: 仅在变化时插入 <段:函数名> 标头行
            link_addr = pc_addr - self._load_offset
            sym_name = self._resolve_symbol(syms, link_addr)
            seg = self._find_segment(link_addr)
            seg_name = seg.name if seg and seg.name else ""
            scope = f"<[dim]{seg_name}[/]:[yellow]{sym_name}[/]>" if (seg_name and sym_name) else ""
            if scope and scope != last_scope:
                last_scope = scope
                lines.append(f"  {scope}")

            lines.append(
                f"  {prefix:<{prefix_w}}  "
                f"{self._hex(pc_addr)}  {raw_hex:<12s}  {asm}"
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
            f"  +{length} bytes  ({len(instrs)} 条指令)\n"
            + "\n".join(lines)
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
        dump = self._emu.mem_hexdump(addr, size)
        self._console.print(dump)

    def cmd_status(self, hart_id_str: str | None = None) -> None:
        """显示 hart 状态. 无参数时显示全部 hart 概览, 带参数时显示指定 hart 详情."""
        if hart_id_str is not None:
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

        # 无参数: 全部 hart 概览
        self._show_hart_overview()

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
            mark = "●" if h.id == active_hart else " "
            if h._halted:
                state = "[red]halted[/]"
            elif h._waiting:
                state = "[dim]waiting[/]"
            else:
                state = "[green]running[/]"
            instr_h = self._instr_count // 100 if h.id == active_hart else 0
            instr_rem = self._instr_count % 100 if h.id == active_hart else 0
            tbl.add_row(
                mark, str(h.id), self._hex(h.pc), h.mode.name, state,
                f"{instr_h}.{instr_rem:02d}h",
            )
        self._console.print(tbl)

    def _show_hart_detail(self, hart_id: int) -> None:
        h = self._emu.harts[hart_id]
        halted_note = " [red]已暂停 — 不可恢复陷态[/]" if h._halted else ""
        instr_h = self._instr_count // 100 if hart_id == self._hart_id else 0
        instr_rem = self._instr_count % 100 if hart_id == self._hart_id else 0
        tbl = Table(
            title=(
                f"Hart {hart_id}  指令计数: {instr_h}.{instr_rem:02d}h"
                f"{halted_note}"
            ),
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
        if idx is None:
            return 0
        return self.hart.read_gpr(idx)

    def _read_csr_by_name(self, name: str) -> int:
        """按名称读取 CSR (例: mtvec, mstatus, mepc)."""
        addr = csr_addr_from_name(name)
        if addr is None:
            return 0
        return self.hart.read_csr(addr)

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
        # 最近的前驱符号
        best_name, best_dist = None, 0xFFFF_FFFF_FFFF_FFFF
        for name, a in symbols.items():
            if a <= addr and (addr - a) < best_dist:
                best_dist, best_name = addr - a, name
        return best_name if best_dist <= 0x10000 else None

    def _find_segment(self, addr: int) -> "FirmwareSegment | None":
        """返回包含 *addr* 的固件段, 若不在任何段内则返回 None."""
        if self._image is None:
            return None
        for seg in self._image.segments:
            if seg.vaddr <= addr < seg.vaddr + seg.memsz:
                return seg
        return None

    # ----------------------------------------------------------
    #  栈帧回溯辅助
    # ----------------------------------------------------------

    def _try_read_frame_link(self, fp: int) -> tuple[int, int] | None:
        """读取 fp 指向的栈帧链接 (saved_ra, saved_fp); 越界则返回 None.

        RISC-V 标准栈帧布局: fp-8 存返回地址, fp-16 存调用者的 fp.
        """
        ra_raw = self._emu.bus.try_read(fp - 8, 8)
        if ra_raw is None:
            return None
        ra = int.from_bytes(ra_raw, "little", signed=False)
        fp_raw = self._emu.bus.try_read(fp - 16, 8)
        if fp_raw is None:
            return None
        fp = int.from_bytes(fp_raw, "little", signed=False)
        return ra, fp

    def _walk_frame_chain(self) -> list[StackFrame]:
        """沿 FP 链遍历调用栈, 返回 StackFrame 列表.

        从当前 hart 的 s0/fp 出发, 按标准 RISC-V 栈帧布局
        (fp-8 存 RA, fp-16 存 saved FP) 向上回溯.
        遇非法指针、读内存失败或 RA=0 时截断.
        """
        h = self.hart
        frames: list[StackFrame] = []
        current_fp: int = h.gprs[8].val  # s0/fp
        visited: set[int] = {current_fp}
        max_frames = 32

        # 帧 #0: 当前执行点 (PC 取自 hart, SP/FP/RA 取自寄存器)
        frames.append(StackFrame(
            idx=0,
            fp=current_fp,
            sp=h.gprs[2].val,
            ra=h.gprs[1].val,
            pc=h.pc,
        ))

        for _ in range(max_frames):
            if current_fp == 0:
                break

            link = self._try_read_frame_link(current_fp)
            if link is None:
                break  # 内存读取越界 → 截断回溯
            saved_ra, saved_fp = link

            # 合法性检查: FP 必须递增、对齐、无环
            if saved_fp != 0:
                if saved_fp <= current_fp or (saved_fp & 0x7) or saved_fp in visited:
                    break
                visited.add(saved_fp)

            # RA 为 0 表示最外层 (无调用者)
            if saved_ra == 0:
                break

            call_site = saved_ra - 4 if saved_ra >= 4 else 0
            frames.append(StackFrame(
                idx=len(frames),
                fp=saved_fp,
                sp=current_fp,  # 上一帧的 FP ≈ 本帧的 SP
                ra=saved_ra,
                pc=call_site,
            ))
            current_fp = saved_fp

        return frames

    @seize_val_err("无效帧号")
    def cmd_frame(self, arg: str | None = None) -> None:
        """栈帧回溯 — #01 起始编号, 只显示当前帧栈内存.

        stack / bt   — 显示全部帧 + 当前帧的栈内存
        frame <N>    — 切换到第 N 帧并刷新回溯 (N 为 0-indexed)
        """
        self._stack_frames = self._walk_frame_chain()
        syms = self._image.symbols if self._image else {}

        def _fn_name(addr: int) -> str:
            """返回地址对应的函数名 (纯文本, 无 markup)."""
            return self._resolve_symbol(syms, addr - self._load_offset) or ""

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
            fn = _fn_name(f.pc)
            seg = self._find_segment(f.pc - self._load_offset)
            seg_name = seg.name if seg and seg.name else ""
            where = ""
            if seg_name and fn:
                where = f"[dim]{seg_name}[/]:[yellow]{fn}[/]"
            elif fn:
                where = f"[yellow]{fn}[/]"
            elif seg_name:
                where = f"[dim]{seg_name}[/]:"
            regs = (
                f"pc={self._hex(f.pc)}  sp={self._hex(f.sp)}  "
                f"fp={self._hex(f.fp)}  ra={self._hex(f.ra)}"
            )
            tbl.add_row(tag, where, regs)

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
            payload = "\n" + ("-" * 55) + "\n[bold]当前栈内存[/]\n" + \
                fmt_hexdump(stack_data, addr=cur.sp)
        self._console.print(payload)

    def cmd_symbols(self, filter_str: str = "") -> None:
        """列出固件符号表, 支持可选的名称过滤."""
        if self._image is None:
            self._warn("无可用的符号表 (非 ELF 文件)")
            return

        syms = self._image.symbols
        if not syms:
            self._console.print("[dim](符号表为空)[/]")
            return

        entries = [
            (name, addr)
            for name, addr in syms.items()
            if filter_str.lower() in name.lower()
        ]
        entries.sort(key=lambda x: x[1])

        if not entries:
            self._console.print(f"[dim]无匹配符号: '{filter_str}'[/]")
            return

        tbl = Table(
            title=f"符号表 ({len(entries)}/{len(syms)} 项)",
            border_style="blue",
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
            if len(parts) < 2:
                self._warn(
                    "用法: csr <name>  或  csr <name1>, <name2>, ...\n"
                    "  例: csr mstatus\n"
                    "  例: csr mscratch, mepc, pmpcfg0, pmpaddr0\n"
                    "  例: csr list"
                )
            else:
                self.cmd_csr(" ".join(parts[1:]))
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
        if cmd == "tlb":
            self.cmd_tlb(parts[1] if len(parts) > 1 else None)
            return True
        if cmd == "tlbflush":
            self.cmd_tlbflush(parts[1] if len(parts) > 1 else None)
            return True
        if cmd == "cache":
            self.cmd_cache(parts[1] if len(parts) > 1 else None)
            return True
        if cmd == "satp":
            self.cmd_satp()
            return True
        if cmd == "mem":
            if len(parts) < 2:
                self._warn("用法: mem <addr> [size]  例: mem 0x80000000 64")
            else:
                self.cmd_mem(parts[1], parts[2] if len(parts) > 2 else "64")
            return True
        if cmd == "disasm":
            if len(parts) < 2:
                self._warn("用法: disasm <addr> [length]  例: disasm 0x80000000 64")
            else:
                self.cmd_disasm(parts[1], parts[2] if len(parts) > 2 else "64")
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
                "Ctrl+C [dim]twice[/] → terminate emulation",
                "Type [bold]help[/] for commands",
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
                ("b/bp|bp list", "列出所有断点"),
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
            ]),
            _section("写寄存器命令", [
                ("set/w <name> <val>", "写入 GPR (例: set sp 0x8000)"),
                ("csrw <name> <val>", "写入 CSR (例: csrw mtvec 0x80000001)"),
                ("pc <addr>", "设置 PC 并显示反汇编"),
            ]),
            _section("内存/符号", [
                ("mem <addr> [size]", "hexdump 内存 (默认 64 字节)"),
                ("disasm <addr> [len]", "反汇编内存区域 (默认 64 字节)"),
                ("sym/symbols [filt]", "列出符号表 (可选过滤)"),
            ]),
            _section("状态", [
                ("status/info [hart]", "全部 hart 概览, 或指定 hart 详情"),
                ("stack/bt", "栈帧情况"),
                ("frame/f <N>", "切换到第 N 帧")
            ]),
            _section("配置", [
                ("hart <id>", "切换活跃 hart"),
            ]),
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
        "--base-addr", type=lambda x: int(x, 0), default=0x80000000,
        help="raw binary 的加载基址 (ELF/PE 时忽略, 默认 0x80000000)",
    )
    parser.add_argument(
        "--prog-cnt", type=lambda x: int(x, 0), default=None,
        help="程序计数器 (Program Counter) — 每个 hart 的开始执行地址 (默认使用固件入口 + 搬迁偏移)",
    )
    parser.add_argument(
        "--ram-base", type=lambda x: int(x, 0), default=0x8000_0000,
        help="物理内存基址 (默认 0x80000000, 加载低地址固件时需设为 0x0)",
    )
    parser.add_argument(
        "--fdt", type=lambda x: int(x, 0) if x is not None else None,
        default=-1, nargs="?", const=-1,
        help="设备树加载地址 (默认自动放在 RAM 顶端 −64 KiB, 接参数则放在指定地址)",
    )
    parser.add_argument(
        "--fdt-file", type=str, default=None, metavar="PATH[:ADDR]",
        help="加载预编译 DTB 文件 (可选 :地址), 例: --fdt-file pyremu_virt.dtb:0x7f0000",
    )
    parser.add_argument(
        "--no-fdt", action="store_true", default=False,
        help="禁用设备树 (--fdt 和 --fdt-file 均被忽略)",
    )
    parser.add_argument("--harts", type=int, default=1, help="hart 数量 (默认 1)")
    parser.add_argument(
        "--ram", type=str, default="128M",
        help="RAM 大小 (支持 K/M/G 后缀, 默认 128M)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别 (默认 INFO)",
    )
    parser.add_argument(
        "--preload", type=str, default=None,
        help="预加载 shellcode 文件路径 (裸 RISC-V 机器码), 在目标程序之前执行",
    )
    parser.add_argument(
        "--preload-addr", type=lambda x: int(x, 0), default=None,
        help="预加载 shellcode 的加载地址 (默认 = RAM 顶端 − 64 KiB)",
    )
    parser.add_argument(
        "--entry-symbol", type=str, default=None,
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
        f"格式={image.format}, 入口=0x{image.entry_point:x}, "
        f"段数={len(image.segments)}"
    )
    for seg in image.segments:
        logger.info(
            f"  [{seg.name}] vaddr=0x{seg.vaddr:016x}  "
            f"size={Debugger._fmt_size(len(seg.data)):<8s}  memsz={Debugger._fmt_size(seg.memsz)}"
        )

    # PIE 固件搬迁: 若所有段 vaddr 均 < ram_base, 整体偏移 ram_base
    load_offset = 0
    if image.format == "elf":
        min_vaddr = min(seg.vaddr for seg in image.segments)
        if min_vaddr < ns.ram_base:
            load_offset = ns.ram_base
            logger.info(
                f"PIE 固件搬迁: 段偏移 +0x{load_offset:x}"
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
            fdt_addr = custom_addr if custom_addr is not None else (
                ns.ram_base + ram_size - 0x10000
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
        logger.info(
            f"已注入预加载 shellcode: {ns.preload} → 0x{preload_entry:x}"
        )
        # preload 场景: 缩 OpenSBI BSS 零填充循环为一轮 (固件 BSS 已由 load_firmware 零填)
        j_imm = (0x98 - 0x94) >> 1
        j_blt = ((j_imm >> 19) & 1) << 31 | (j_imm & 0x3FF) << 21 | ((j_imm >> 10) & 1) << 20 | ((j_imm >> 11) & 0xFF) << 12 | 0x6f
        emu.bus.write(ns.ram_base + 0x94, struct.pack("<I", j_blt))
        # misa 需设 U-bit (bit20), sbi_init 冷启动依赖
        for h in emu.harts:
            h.csrs["misa"].val = (2 << 62) | (1 << 18) | (1 << 20)  # MXL=RV64 + U-bit; h.csrs["mscratch"].val = ns.ram_base + ram_size - 0x100000
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
