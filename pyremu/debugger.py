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

from pyremu.core.decoder import parse_compressed
from pyremu.core.hart import RiscvMode
from pyremu.core.registers import register_csr, register_fpr, register_gpr
from pyremu.utils.disassem import disasm
from pyremu.core.trap import TrapType, trap_cause_name
from pyremu.emulator import Emulator
from pyremu.env_inject import Preloader
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils.parse_bin import FirmwareImage, parse_firmware

# ============================================================
#  状态快照 (用于指令回滚)
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
        self._emu = emulator
        self._hart_id = hart_id
        self._image = image
        self._running = False

        self._sigint_count = 0
        self._paused = False
        self._terminated = False

        self._snapshot: HartSnapshot | None = None
        self._mem_changes: list[MemoryChange] = []
        self._instr_count = 0

        # Rich console — 所有交互输出经此通道 (彩色, 高亮)
        self._console = Console(highlight=True)

        # 陷态上下文去重: 同一 mcause 值只展示一次，避免 handler 执行期间重复刷屏
        self._trap_displayed_mcause: int | None = None

        # 上一条命令 (按回车重复执行)
        self._last_command: str | None = None

        # prompt_toolkit REPL — 方向键历史, Tab 补全, 持久化历史文件
        self._history = FileHistory(
            os.path.expanduser("~/.pyremu_history")
        )
        self._completer = self._build_completer()
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
            "undo", "rollback",
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
            "stack", "bt",
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
        if self._sigint_count == 1:
            self._console.print(
                "\n[yellow]暂停请求 — 当前指令完成后回到 REPL; 再次 Ctrl+C 强制终止[/]"
            )
        else:
            self._console.print("\n[red bold]强制终止模拟循环[/]")
            self._terminated = True
        self._paused = True

    def _enter_repl_mode(self) -> None:
        signal.signal(signal.SIGINT, self._sigint_repl)

    def _enter_run_mode(self) -> None:
        signal.signal(signal.SIGINT, self._sigint_run)

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

        snap = self._save_snapshot()
        self._mem_changes = []

        orig_write = h._mem_write_phy

        def tracking_write(addr: int, data: bytes) -> None:
            try:
                old = self._emu.bus.read(addr, len(data))
                self._mem_changes.append(MemoryChange(addr=addr, old=old))
            except Exception:
                pass
            orig_write(addr, data)

        h._mem_write_phy = tracking_write

        try:
            pc_before = h.pc
            instr_bytes = self._emu.bus.read(h.pc, 4)
            instr = int.from_bytes(instr_bytes, "little", signed=False)
            try:
                advance = h.exec_instr(instr)
            except NotImplementedError:
                h._take_trap(TrapType.IllInstr, tval=instr, is_interrupt=False)
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
            h.check_pending_interrupts()
            self._instr_count += 1
        finally:
            h._mem_write_phy = orig_write

        self._snapshot = snap

    def rollback(self) -> None:
        """回滚最近一条指令的影响."""
        if self._snapshot is None:
            self._warn("没有可回滚的指令")
            return

        for change in reversed(self._mem_changes):
            try:
                self._emu.bus.write(change.addr, change.old)
            except Exception:
                pass

        self._restore_snapshot(self._snapshot)
        self._snapshot = None
        self._mem_changes = []
        self._instr_count -= 1
        self._console.print("[dim]已撤销上一条指令[/]")

    # ==========================================================
    #  运行循环
    # ==========================================================

    def _run_loop(self, steps: int | None = None) -> None:
        self._running = True
        self._paused = False
        self._terminated = False
        self._sigint_count = 0
        self._enter_run_mode()

        try:
            executed = 0
            while not self._terminated and not self._paused:
                if steps is not None and executed >= steps:
                    break
                if self.hart._halted:
                    self._warn("Hart 进入不可恢复陷态, 已暂停运行")
                    break
                self.step_one()
                executed += 1
            if self._terminated:
                self._console.print("[dim]模拟循环已终止[/]")
        finally:
            self._running = False
            self._enter_repl_mode()

    # ==========================================================
    #  格式化辅助
    # ==========================================================

    @staticmethod
    def _hex(v: int) -> str:
        return f"0x{v:016x}"

    @staticmethod
    def _trap_cause_name(mcause_val: int) -> str:
        """将 mcause 寄存器值翻译为可读的 trap 类型名称 (委托 trap.py)."""
        return trap_cause_name(mcause_val)

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
        for _ in range(count):
            self.step_one()
            if self.hart._halted:
                return
        self.cmd_pc()

    def cmd_continue(self) -> None:
        self._console.print("[dim]继续执行 (Ctrl+C 暂停)...[/]")
        self._run_loop()

    def cmd_run(self, n: int = 1) -> None:
        self._console.print(f"[dim]执行 {n} 条指令...[/]")
        self._run_loop(steps=n)

    def cmd_undo(self) -> None:
        self.rollback()

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

    def cmd_reg(self, name: str) -> None:
        """读取 GPR: reg <name>."""
        r = self._find_gpr(name)
        if r is None:
            self._err(f"未知寄存器: {name}")
            return
        self._console.print(f"{r.name} ([cyan]{r.alias}[/]) = [green]{self._hex(r.val)}[/]")

    def cmd_set(self, name: str, value: str) -> None:
        """写入 GPR: set <name> <value>."""
        r = self._find_gpr(name)
        if r is None:
            self._err(f"未知寄存器: {name}")
            return
        try:
            v = int(value, 0) & 0xFFFF_FFFF_FFFF_FFFF
        except ValueError:
            self._err(f"无效值: {value}")
            return
        old = r.val
        r.val = v
        self._console.print(
            f"{r.name} ([cyan]{r.alias}[/]): "
            f"[yellow]{self._hex(old)}[/] → [green]{self._hex(v)}[/]"
        )

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

    def cmd_csr(self, name: str) -> None:
        """读取 CSR: csr <name>."""
        h = self.hart
        if name == "list":
            names = sorted(h.csrs.keys())
            self._console.print(f"可用 CSR: [dim]{', '.join(names)}[/]")
            return
        if name not in h.csrs:
            self._err(f"未知 CSR: {name} (用 'csr list' 查看可用列表)")
            return
        csr = h.csrs[name]
        self._console.print(
            f"[cyan]{name}[/] = [green]{self._hex(csr.val)}[/] (dec: {csr.val})"
        )

    def cmd_csrw(self, name: str, value: str) -> None:
        """写入 CSR: csrw <name> <value>."""
        h = self.hart
        if name not in h.csrs:
            self._err(f"未知 CSR: {name} (用 'csr list' 查看可用列表)")
            return
        try:
            v = int(value, 0) & 0xFFFF_FFFF_FFFF_FFFF
        except ValueError:
            self._err(f"无效值: {value}")
            return
        csr = h.csrs[name]
        old = csr.val
        csr.val = v
        self._console.print(
            f"[cyan]{name}[/]: "
            f"[yellow]{self._hex(old)}[/] → [green]{self._hex(v)}[/]"
        )

    def cmd_pc(self, addr: str | None = None) -> None:
        """读/设 PC: pc (读取并反汇编), pc <addr> (设置).

        若当前 mcause 非零 (存在待处理的 trap), 一并显示
        mepc / mcause / mtval 等信息.
        """
        h = self.hart
        if addr is not None:
            try:
                v = int(addr, 0) & 0xFFFF_FFFF_FFFF_FFFF
            except ValueError:
                self._err(f"无效地址: {addr}")
                return
            old = h.pc
            h.pc = v
            self._snapshot = None
            self._mem_changes = []
            self._console.print(
                f"PC: [yellow]{self._hex(old)}[/] → [green]{self._hex(h.pc)}[/]"
            )
        pc = h.pc
        try:
            raw = self._emu.bus.read(pc, 4)
            instr = int.from_bytes(raw, "little", signed=False)
            asm = disasm(instr, pc)
            raw_hex = " ".join(f"{b:02x}" for b in raw[:4])
        except Exception:
            raw_hex = "(无法读取)"
            asm = "(无法解码)"
        lines = [
            f"PC  = [bold yellow]{self._hex(pc)}[/]",
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

    def cmd_tlb(self, arg: str | None = None) -> None:
        """显示或查找 TLB 条目.

        tlb         — 显示全部 ITLB / DTLB
        tlb <vpn>   — 查找指定 VPN
        """
        if arg is not None:
            try:
                self._tlb_search(int(arg, 0))
            except ValueError:
                self._err(f"无效 VPN: {arg}")
            return
        h = self.hart
        self._show_tlb("ITLB", h.itlb)
        self._show_tlb("DTLB", h.dtlb)

    def cmd_tlbflush(self, vpn_str: str | None = None) -> None:
        """刷新 TLB: tlbflush [vpn] (无参数 = 全部)."""
        h = self.hart
        if vpn_str is None:
            h.itlb.flush_all()
            h.dtlb.flush_all()
            self._console.print("[dim]已刷新全部 ITLB + DTLB[/]")
            return
        try:
            vpn = int(vpn_str, 0)
        except ValueError:
            self._err(f"无效 VPN: {vpn_str}")
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

    def cmd_cache(self, arg: str | None = None) -> None:
        """显示 L2 缓存状态.

        cache             — 概览 (元数据 + 前 16 B 数据预览)
        cache <set>       — 指定 set, 含完整 64 B hexdump
        cache <set> <way> — 指定 set/way, 含完整 64 B hexdump
        """
        l2 = self._emu.bus.l2
        if l2 is None:
            self._console.print("[dim]L2 缓存未启用[/]")
            return

        entries, ways, num_sets = l2.entries, l2.ways, l2.num_sets

        # 解析 set[/way]
        target_set, target_way = None, None
        if arg is not None:
            parts = arg.split()
            try:
                target_set = int(parts[0], 0)
            except ValueError:
                self._err(f"无效 set 索引: {parts[0]}")
                return
            if not (0 <= target_set < num_sets):
                self._err(f"set 索引超出范围 [0, {num_sets - 1}]")
                return
            if len(parts) > 1:
                try:
                    target_way = int(parts[1], 0)
                except ValueError:
                    self._err(f"无效 way 索引: {parts[1]}")
                    return
                if not (0 <= target_way < ways):
                    self._err(f"way 索引超出范围 [0, {ways - 1}]")
                    return

        # 统计
        mesi_counts = Counter()
        valid_count = 0
        dirty_count = 0
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

        full_dump = target_set is not None
        set_range = [target_set] if full_dump else range(num_sets)

        lines = []
        for set_idx in set_range:
            for way_idx in range(ways):
                if target_way is not None and way_idx != target_way:
                    continue
                e = entries[set_idx * ways + way_idx]
                if not e.valid:
                    continue
                lines.append(self._fmt_cache_line(e, set_idx, way_idx, full_dump))

        if lines:
            self._console.print(header + "\n" + "\n".join(lines))
        else:
            self._console.print(header)

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

    def cmd_disasm(self, addr_str: str, length_str: str = "64") -> None:
        """反汇编指定内存区域.

        disasm <addr> [length]  — 从 addr 开始反汇编 length 字节 (默认 64).

        自动识别 16-bit 压缩指令和 32-bit 标准指令边界,
        按地址递增顺序逐条输出。
        """
        try:
            addr = int(addr_str, 0)
            length = int(length_str, 0)
        except ValueError:
            self._err("addr 和 length 需为整数 (支持 0x 前缀)")
            return

        if length <= 0 or length > 4096:
            self._err("length 需在 1–4096 之间")
            return

        try:
            raw = self._emu.bus.read(addr, length)
        except Exception:
            self._console.print_exception()
            return

        lines: list[str] = []
        offset = 0
        max_offset = len(raw)

        while offset < max_offset:
            pc = addr + offset
            remaining = max_offset - offset

            # 至少需要 2 字节才能判断是否压缩指令
            chunk = raw[offset : offset + min(4, remaining)]
            instr = int.from_bytes(
                chunk.ljust(4, b"\x00"), "little", signed=False
            )

            # 低 2 位 ≠ 3 则为压缩指令 (16-bit)
            if parse_compressed(instr):
                inst_size = 2
            else:
                inst_size = 4

            if remaining < inst_size:
                # 剩余不足一条完整指令，hexdump 残部
                leftover = raw[offset:]
                hex_s = " ".join(f"{b:02x}" for b in leftover)
                lines.append(f"  {self._hex(pc)}  {hex_s:<12s}  [dim](截断)[/]")
                break

            asm = disasm(instr, pc)
            raw_hex = " ".join(f"{b:02x}" for b in raw[offset:offset + inst_size])
            lines.append(f"  {self._hex(pc)}  {raw_hex:<12s}  {asm}")
            offset += inst_size

        if not lines:
            self._console.print("[dim](空)[/]")
            return

        self._console.print(
            f"[bold]反汇编[/] [yellow]{self._hex(addr)}[/]"
            f"  +{length} bytes  ({len(lines)} 条指令)\n"
            + "\n".join(lines)
        )

    def cmd_mem(self, addr_str: str, size_str: str = "64") -> None:
        try:
            addr = int(addr_str, 0)
            size = int(size_str)
        except ValueError:
            self._err("addr 和 size 需为整数 (支持 0x 前缀)")
            return
        try:
            dump = self._emu.mem_hexdump(addr, size)
            self._console.print(dump)
        except Exception:
            self._console.print_exception()

    def cmd_status(self) -> None:
        h = self.hart
        halted_note = " [red]已暂停 — 不可恢复陷态[/]" if h._halted else ""
        tbl = Table(
            title=f"Hart {self._hart_id}  指令计数: {self._instr_count}{halted_note}",
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

    @staticmethod
    def _resolve_symbol(symbols: dict[str, int], addr: int) -> str | None:
        """在符号表中查找 *addr* 最近的符号名 (精确匹配)."""
        for name, a in symbols.items():
            if a == addr:
                return name
        return None

    def cmd_frame(self) -> None:
        """显示当前栈帧和调用回溯 (backtrace).

        RISC-V 标准帧布局 (s0/fp 指向帧基):
            fp -  8 → 保存的 ra
            fp - 16 → 保存的 fp (上一层帧)
        回溯沿保存的 fp 链向上遍历, 直到 fp=0 或链断裂.
        """
        h = self.hart
        sp = h.gprs[2].val
        ra = h.gprs[1].val
        fp = h.gprs[8].val
        pc = h.pc

        syms = self._image.symbols if self._image else {}

        def _sym(addr: int) -> str:
            name = self._resolve_symbol(syms, addr)
            return f" [dim]<{name}>[/]" if name else ""

        lines = [
            "[bold]栈帧回溯[/]",
            "",
            f"  PC = [bold yellow]{self._hex(pc)}[/]{_sym(pc)}",
            f"  SP = [cyan]{self._hex(sp)}[/]",
            f"  FP = [cyan]{self._hex(fp)}[/]  (s0)",
            f"  RA = [green]{self._hex(ra)}[/]  (x1){_sym(ra)}",
        ]

        # 沿保存的 fp 链向上遍历
        current_fp = fp
        frame_idx = 0
        max_frames = 32
        fp_history: set[int] = {current_fp}  # 防环

        try:
            while current_fp != 0 and frame_idx < max_frames:
                # fp - 8: saved ra;  fp - 16: saved previous fp
                saved_ra_raw = self._emu.bus.read(current_fp - 8, 8)
                saved_ra = int.from_bytes(saved_ra_raw, "little", signed=False)
                saved_fp_raw = self._emu.bus.read(current_fp - 16, 8)
                saved_fp = int.from_bytes(saved_fp_raw, "little", signed=False)

                # 合法性检查: fp 应单调递增 (栈向下增长), 8 字节对齐, 无环
                if saved_fp != 0:
                    if saved_fp <= current_fp:
                        lines.append(
                            f"\n  [dim](回溯终止: saved_fp=0x{saved_fp:x} 不递增)[/]"
                        )
                        break
                    if saved_fp & 0x7:
                        lines.append(
                            f"\n  [dim](回溯终止: saved_fp=0x{saved_fp:x} 未对齐)[/]"
                        )
                        break
                    if saved_fp in fp_history:
                        lines.append(
                            f"\n  [dim](回溯终止: 检测到循环)[/]"
                        )
                        break
                    fp_history.add(saved_fp)

                frame_idx += 1
                # 返回地址实际是 call 的下一条; 显示 ra-4 更直观
                call_site = saved_ra - 4 if saved_ra >= 4 else 0
                lines.extend([
                    "",
                    f"  [bold]#{frame_idx}[/] FP = {self._hex(current_fp)}",
                    f"    saved RA = [green]{self._hex(saved_ra)}[/]{_sym(saved_ra)}",
                    f"    call site≈ [dim]{self._hex(call_site)}[/]{_sym(call_site)}",
                    f"    saved FP = [cyan]{self._hex(saved_fp)}[/]",
                ])

                if saved_fp == 0:
                    lines.append("\n  [dim](回溯终止: fp=0, 已达最外层)[/]")
                    break
                current_fp = saved_fp

        except Exception:
            lines.append("\n  [dim](回溯终止: 内存读取失败 — 可能 sp 未初始化或越界)[/]")

        # 展示栈顶附近的内存
        try:
            stack_data = self._emu.bus.read(sp, 64)
            lines.extend([
                "",
                f"  [bold]栈顶数据 (SP = {self._hex(sp)}):[/]",
                self._hexdump_bytes(stack_data, indent="    "),
            ])
        except Exception:
            lines.append(f"\n  [dim](无法读取 sp 处内存)[/]")

        self._console.print("\n".join(lines))

    def cmd_symbols(self, filter_str: str = "") -> None:
        """列出固件符号表, 支持可选的名称过滤."""
        if self._image is None:
            self._warn("无可用的符号表 (非 ELF 文件?)")
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
            self.cmd_undo()
            return True

        # 寄存器
        if cmd in ("regs", "gpr"):
            self.cmd_regs()
            return True
        if cmd == "reg":
            if len(parts) < 2:
                self._warn("用法: reg <name>  例: reg x10, reg a0")
            else:
                self.cmd_reg(parts[1])
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
                self._warn("用法: csr <name>  例: csr mstatus; csr list")
            else:
                self.cmd_csr(parts[1])
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
            self.cmd_status()
            return True
        if cmd in ("stack", "bt"):
            self.cmd_frame()
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
                f"Reset vector: [bold yellow]{self._emu.reset_vector:#018x}[/]",
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
                    [("class:prompt", f"\nrvdb[{self._hart_id}] ")]
                ).strip()
            except KeyboardInterrupt:
                self._console.print()
                continue
            except EOFError:
                self._console.print("[dim]退出[/]")
                break

            # 空输入 = 重复上一条命令 (GDB 风格)
            if not raw:
                if self._last_command is not None:
                    raw = self._last_command
                    self._console.print(f"[dim](重复) {raw}[/]")
                else:
                    continue

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
            tbl = Table(title=title, border_style="green", show_header=False)
            tbl.add_column("cmd", style="cyan", no_wrap=True)
            tbl.add_column("desc")
            for cmd, desc in rows:
                tbl.add_row(cmd, desc)
            return tbl

        sections: list[Table] = [
            _section("执行控制", [
                ("s, step [n]", "单步执行 n 条指令 (默认 1)"),
                ("c, continue", "连续执行 (直到 Ctrl+C 暂停)"),
                ("r, run [n]", "执行 n 条指令 (默认 1)"),
                ("undo, rollback", "回滚上一条指令"),
            ]),
            _section("寄存器 — 读", [
                ("regs, gpr", "显示全部 GPR (x0–x31)"),
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
            _section("寄存器 — 写", [
                ("set, w <name> <val>", "写入 GPR (例: set sp 0x8000)"),
                ("csrw <name> <val>", "写入 CSR (例: csrw mtvec 0x80000001)"),
                ("pc <addr>", "设置 PC 并显示反汇编"),
            ]),
            _section("内存 & 符号", [
                ("mem <addr> [size]", "hexdump 内存 (默认 64 字节)"),
                ("disasm <addr> [len]", "反汇编内存区域 (默认 64 字节)"),
                ("sym, symbols [filt]", "列出符号表 (可选过滤)"),
            ]),
            _section("状态", [
                ("status, info", "显示 hart 状态摘要"),
                ("stack, bt", "显示栈帧回溯 (backtrace)"),
            ]),
            _section("配置", [
                ("hart <id>", "切换活跃 hart"),
            ]),
            _section("其他", [
                ("help, h, ?", "显示本帮助"),
                ("quit, q, exit", "退出调试器"),
            ]),
        ]

        for tbl in sections:
            self._console.print(tbl)


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
        "--reset-vector", type=lambda x: int(x, 0), default=0x10000000,
        help="复位向量地址 (默认 0x10000000)",
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
            f"size={len(seg.data):6} bytes  memsz=0x{seg.memsz:x}"
        )

    # 创建模拟器并加载
    plat_cfg = PlatformConfig(
        num_harts=ns.harts,
        ram_size=ram_size,
        reset_vector=ns.reset_vector,
        periph=PeripheralConfig(),
    )
    emu = Emulator(plat_cfg)
    emu.load_firmware(image)
    logger.info(f"固件已写入 RAM, {ns.harts} hart(s) 就绪")

    # 预加载 shellcode (外部提供的裸 RISC-V 机器码)
    preload_entry = None
    if ns.preload is not None:
        preloader = Preloader(emu)
        preload_entry = preloader.inject_file(ns.preload, addr=ns.preload_addr)
        logger.info(
            f"已注入预加载 shellcode: {ns.preload} → 0x{preload_entry:x}"
        )

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
    dbg.repl()


if __name__ == "__main__":
    main()
