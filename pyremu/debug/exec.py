#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""ExecutionMixin — 指令执行、快照回滚、写监控、运行循环.

依赖 DebuggerBase + MemoryMixin._try_read_va + future BreakpointMixin.
"""

from __future__ import annotations

import time
from pathlib import Path as _Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

    from pyremu.debug.types import (
        Breakpoint,
        MemoryChange,
        StackFrame,
    )

from loguru import logger
from rich.panel import Panel

from pyremu._native import decode_fields
from pyremu.core.decoder import Hart
from pyremu.core.hart import RiscvMode
from pyremu.core.mem_check_aux import inject_memory_backend
from pyremu.core.trap_def import TrapType
from pyremu.core.trap_handler import check_pending_interrupts, deliver_trap
from pyremu.debug._attrs import SharedMixinAttrs
from pyremu.debug.types import HartSnapshot, MemWriteTracker
from pyremu.debug.utils import hex_addr
from pyremu.emulator import Emulator
from pyremu.env_inject import Preloader
from pyremu.utils.parse_bin import FirmwareImage
from pyremu.utils.tick import yield_cpu
from pyremu.utils.mask import mask64

_YIELD_EVERY = 500_000
_YIELD_INTERVAL = 0.001
_DEFAULT_TIMEOUT = 3600.0


class ExecutionMixin(SharedMixinAttrs):
    """指令执行、快照回滚、运行循环."""

    # Cross-mixin attributes provided by DebuggerBase / other mixins
    # (hart 为 @property 提供, __getattr__ 解析)
    _emu: Emulator
    _console: Console
    _image: FirmwareImage | None
    _load_offset: int
    _hart_id: int
    _instr_count: int
    _snapshot: HartSnapshot | None
    _mem_changes: list[MemoryChange]
    _watch_ranges: list[tuple[int, int]]
    _watch_hit: tuple[int, int, int, bytes] | None
    _paused: bool
    _running: bool
    _terminated: bool
    _sigint_count: int
    _bp_mode: str
    _hart_paused: set[int]
    _bp_hit_this_run: set[tuple[str, int]]
    _prev_instr_csr_addr: int
    _breakpoints: list[Breakpoint]
    _trap_displayed_mcause: int | None
    _disasm_next_addr: int | None
    _vdisasm_next_addr: int | None
    _disasm_ref_pc: int | None
    _disasm_base_step: int
    _disasm_past_terminator: bool
    _consecutive_stalls: int
    _stack_frames: list[StackFrame]
    _current_frame_idx: int
    _last_command: str | None
    _fdt_addr: int | None
    _kernel_path: str | None
    _kernel_addr: int
    _preload_path: str | None

    # ----------------------------------------------------------
    #  快照 & 回滚
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    #  单步执行
    # ----------------------------------------------------------

    def step_one(self) -> None:
        """执行一条指令并保存快照 (供后续回滚)."""
        h = self.hart

        if h._halted:
            return

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
                deliver_trap(
                    h, TrapType.InstrPageFault,
                    tval=h.pc, is_interrupt=False
                )
                return
            instr = int.from_bytes(raw, "little", signed=False)

            if self._check_breakpoints(h, pc_before, instr):
                return

            try:
                advance = h.exec_instr(instr)
            except NotImplementedError:
                deliver_trap(h, TrapType.IllInstr, tval=instr, is_interrupt=False)
                advance = 0
            if advance != 0 and h.pc == pc_before:
                h.pc = mask64((h.pc + advance))
                h._consecutive_traps = 0
            elif advance == 0 and h.pc == pc_before:
                h._consecutive_traps += 1
            if h._consecutive_traps >= self._emu._TRAP_LOOP_THRESHOLD:
                dump = self._emu._dump_hart_state(h)
                self._console.print(
                    Panel(dump, title="[red]Hart State Dump[/]", border_style="red")
                )
                h._halted = True
                self._err(f"Hart {h.id} 进入不可恢复陷态, 已暂停")

            check_pending_interrupts(h)
            self._instr_count += 1

            self._emu.sync_counters(1)
        finally:
            h._mem_write_phy = orig_write

        self._mem_changes = tracker.changes
        self._snapshot = snap

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
                    f"  [red] (@) 写监控命中[/] pc={hex_addr(h.pc)}"
                    f"  写入 {hex_addr(a)} ({sz}B)"
                    f"  旧值={change.old[:16].hex()}"
                )
                return

    # ----------------------------------------------------------
    #  回滚
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    #  运行循环
    # ----------------------------------------------------------


    def _tail_stage_for_updating_pc(self) -> int:
        if self._check_multi_hart_bp():
            return 1
        self._instr_count += 1
        if self._instr_count % _YIELD_EVERY == 0:
            yield_cpu(_YIELD_INTERVAL)
        return 0

    def _run_loop(
        self,
        cycles: int | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        """运行循环: 每个周期所有 hart 各执行一条指令 (round-robin).

        断点通过 Rust native batch 内联检查: 每次 ``emu.step()`` 前将 addr
        类型断点的 PC 传给 ``emu._bp_addrs``, Rust 逐指令比对后以
        ``EXIT_BREAKPOINT`` 退出, 在此处由 ``_check_multi_hart_bp`` 报告.

        Args:
            cycles: 最大周期数, None 表示无限.
            timeout: 墙钟超时秒数, 默认 3600 (1 小时). 设为 0 禁用.
        """
        self._running = True
        self._paused = False
        self._terminated = False
        self._bp_hit_this_run.clear()
        self._enter_run_mode()
        timeout = _DEFAULT_TIMEOUT if timeout is None else timeout
        deadline = time.monotonic() + timeout if timeout > 0 else None

        _instr_start = self._instr_count

        # Push addr breakpoints to the emulator so Rust checks them inline.
        # For kernel symbols resolved to physical addresses, also include the
        # VA counterparts so Rust inline matching works when MMU is enabled.
        _bp_addrs: list[int] = []
        for bp in self._breakpoints:
            if bp.kind != "addr":
                continue
            _bp_addrs.append(bp.value)
            va = self._pa_to_va.get(bp.value)
            if va is not None:
                _bp_addrs.append(va)
        self._emu._bp_addrs = _bp_addrs

        # Snapshot hart PCs for stall detection: CSR_EXIT loops cause
        # cnt == 0 even though harts are making progress; we distinguish
        # genuine stalls (PC unchanged) from CSR-explosion (PC advancing).
        _prev_pcs: dict[int, int] = {id(h): h.pc for h in self._emu.harts}

        while not self._terminated and not self._paused:
            if cycles is not None and (self._instr_count - _instr_start) >= cycles:
                break
            if all(h._halted for h in self._emu.harts):
                self._warn("所有 Hart 已暂停")
                break
            if self.hart._halted:
                self._show_trap_context(self.hart)
                break
            if self._bp_mode == "async" and self._hart_id in self._hart_paused:
                self._console.print(
                    f"  [dim]Hart {self._hart_id} 断点暂停, 回到 REPL[/]"
                )
                break
            if deadline is not None and time.monotonic() >= deadline:
                self._warn(f"运行超时 ({timeout:.0f}s), 强制停止")
                self._show_trap_context(self.hart)
                break

            # 每批次前将可用 stdin 字节转发到 UART RX,
            # 确保用户输入在本轮批次内被客机处理 (而非等到下轮),
            # 消除键入与回显之间的一批次延迟。
            self._feed_uart_stdin()

            try:
                cnt = self._emu.step()
            except Exception:
                logger.opt(exception=True).error("step 执行异常")
                self._err("指令执行异常, 运行中止")
                self._terminated = True
                break

            # UART TXDATA 写入已自动即时输出, 无需手动刷新
            self._flush_uart_if_present()

            if cnt != 0:
                self._consecutive_stalls = 0
                if self._tail_stage_for_updating_pc() > 0:
                    break
                continue

            # cnt == 0 can mean:
            #   (a) all active harts are WFI-idle — normal, wait for interrupts
            #   (b) CSR_EXIT loops — harts execute but CSR ops aren't counted
            #       (batch exits on unhandled CSR without incrementing count)
            #   (c) genuine stall — hart is "running" but PC frozen
            # Track PC deltas to distinguish (b) from (c).
            all_idle = all(
                h._halted or h._waiting for h in self._emu.harts
            )
            if all_idle:
                # 全部 WFI 空闲: _native_finalize 中的 WFI 轮询已负责
                # stdin 转发与 TX 刷新, 此处仅复位 stall 计数器.
                self._consecutive_stalls = 0
                if self._tail_stage_for_updating_pc() > 0:
                    break
            any_pc_changed = any(
                not (h._halted or h._waiting)
                and h.pc != _prev_pcs.get(id(h), 0)
                for h in self._emu.harts
            )
            _prev_pcs = {id(h): h.pc for h in self._emu.harts}
            if any_pc_changed:
                self._consecutive_stalls = 0
            else:
                self._consecutive_stalls += 1
                if self._consecutive_stalls >= 3:
                    self._warn("连续 3 次无指令执行 (非 WFI) — 强制暂停")
                    break
            if self._tail_stage_for_updating_pc() > 0:
                break

        if self._terminated:
            self._console.print("[dim]模拟循环已终止[/]")

        self._running = False
        self._enter_repl_mode()

    # ----------------------------------------------------------
    #  运行命令
    # ----------------------------------------------------------

    def cmd_step(self, count: int = 1) -> None:
        """单步或多步执行.

        step       — 执行 1 条指令, 显示 PC 及反汇编
        step <n>   — 执行 n 条指令, 仅显示最终状态.

        对于 count > 1 且 native batch 可用的情况, 委托给 ``emu.step()``,
        避免纯 Python 逐条执行的 FFI 开销 (单条 step_one 约慢 10-50 倍).
        """
        if count <= 0:
            self._err("步数须 > 0")
            return
        self._paused = False
        self._hart_paused.clear()

        # Native fast path: delegate multi-step to the batch engine.
        # Breakpoints are checked by Rust inline; WFI / trap are handled
        # by the batch exit path in ``_step_native``.
        if count > 1 and self._emu._native_batch:
            saved_max = self._emu._native_max_instrs
            self._emu._native_max_instrs = count
            try:
                real_cnt = self._emu.step()
            finally:
                self._emu._native_max_instrs = saved_max
            self._instr_count += real_cnt
            if not self.hart._halted:
                self.cmd_pc()
            else:
                self._show_trap_context(self.hart)
            return

        # Pure-Python path for single-step or when native is disabled
        for _ in range(count):
            self.step_one()
            if not self.hart._halted:
                continue
            self._show_trap_context(self.hart)
            return
        self.cmd_pc()

    def cmd_continue(self) -> None:
        self._console.print("[dim]继续执行 (Ctrl+C 暂停)...[/]")
        # 若任一 hart 的 PC 恰好落在地址断点上, 先步进越过该断点,
        # 否则 run_batch 的 pre-execution 检查会立即再次命中同一断点.
        # 必须遍历全部 hart (不仅是当前 hart), 因为断点可能命中的是
        # 用户当前未选中的 hart.
        bps_to_skip: list = []
        bp_values_to_skip: set[int] = set()
        for h in self._emu.harts:
            if h._halted or h._waiting:
                continue
            for bp in self._breakpoints:
                if not (bp.kind == "addr"
                        and self._bp_match_pc(h, h.pc, bp.value)):
                    continue
                if bp not in bps_to_skip:
                    bps_to_skip.append(bp)
                    bp_values_to_skip.add(bp.value)
                break
        if bps_to_skip:
            for bp in bps_to_skip:
                self._breakpoints.remove(bp)
            # 同步更新 _bp_addrs, 否则 Rust batch 仍会命中已移除的断点
            saved_bp_addrs = self._emu._bp_addrs
            self._emu._bp_addrs = [
                a for a in saved_bp_addrs if a not in bp_values_to_skip
            ]
            # 使用纯 Python 单步路径跳过断点 (而非 native batch).
            # native batch (run_parallel) 可执行最多 100k 条指令; 若断点处
            # 指令为自跳转循环 (j .), batch 会在此自旋 100k 次仍不推进 PC,
            # 恢复断点后立即再次命中 ->表现为 "c 原地踏步".
            # 纯 Python 路径每 hart 仅执行一条指令, 确保精确跳过.
            saved_native_batch = self._emu._native_batch
            self._emu._native_batch = False
            try:
                self._emu.step()
            finally:
                self._breakpoints.extend(bps_to_skip)
                self._refresh_bp_cache()
                self._emu._bp_addrs = saved_bp_addrs
                self._emu._native_batch = saved_native_batch
        self._run_loop()

    def cmd_run(self, n: int = 1) -> None:
        self._console.print(f"[dim]执行 {n} 条指令...[/]")
        self._run_loop(cycles=n)

    def cmd_restart(self) -> None:
        """重新运行当前加载的程序."""
        emu = self._emu
        cfg = emu._cfg

        emu.clint._mtime = 0
        for i in range(cfg.num_harts):
            emu.clint._mtimecmp[i] = 0
            emu.clint._msip[i] = 0

        emu.harts = []
        for i in range(cfg.num_harts):
            h = Hart(id=i, pmp_entries=cfg.pmp_entries)
            h.pc = cfg.prog_cnt
            inject_memory_backend(h, emu.bus.read, emu.bus.write)
            h.bus = emu.bus
            h.interrupt_ctrl = emu.clint
            h.plic = emu.plic
            emu.harts.append(h)

        for h in emu.harts:
            h.all_harts = emu.harts

        # 清除 native batch 侧断点地址 (旧 harts 已销毁, 地址可能变化)
        emu._bp_addrs.clear()

        # 若 emu.__init__ 后 hart 数量未变, _native_states 仍有效;
        # 标记 _native_stop_flag 重置以防上一轮 Ctrl+C 未清理.
        emu._native_stop_flag.value = 0

        # 确保终端恢复 + SIGINT 设为 REPL 处理器;
        # 前次运行若异常退出 (未执行 _enter_repl_mode) 会残留
        # cbreak 模式与 _sigint_run handler, 导致 Ctrl+C 失灵.
        self._enter_repl_mode()

        if self._image is not None:
            emu.load_firmware(self._image, load_offset=self._load_offset)

        # 重建 DTB: 用原始 bootargs + 完整设备树 (含 virtio-blk)
        # 写入 emulator 记录的 DTB 地址, 固件 ELF 可能已覆写该区域.
        if emu._dtb_addr is not None:
            emu.load_dtb(emu._dtb_addr)
        elif self._fdt_addr is not None:
            emu.load_dtb(self._fdt_addr)
        elif emu._bootargs is not None:
            _dtb_addr = emu._cfg.ram_base + emu._cfg.ram_size - 0x10000
            emu.load_dtb(_dtb_addr)

        if self._kernel_path is not None:
            kernel_data = _Path(self._kernel_path).read_bytes()
            emu.bus.write_ram_direct(self._kernel_addr, kernel_data)

        if self._preload_path is not None:
            preload_entry = Preloader(emu).inject_file(self._preload_path)
            for h in emu.harts:
                h.csrs["misa"].val = (
                    (2 << 62) | (1 << 18) | (1 << 20)
                )  # MXL=RV64 + S + U
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

    # ----------------------------------------------------------
    #  写监控
    # ----------------------------------------------------------

    def cmd_watch(self, addr_str: str = "", size_str: str = "8") -> None:
        """设置写监控: watch <addr> [size] — 关闭: watch off."""
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
            f"  [green]* 写监控[/] {hex_addr(addr)}–{hex_addr(addr + size)}"
        )

    # ----------------------------------------------------------
    #  多 hart 断点检查 (供 _run_loop 调用)
    # ----------------------------------------------------------

    def _check_single_bp(self, h, bp) -> bool | None:
        """检查单个断点是否命中, 处理去重与报告.

        Returns:
            None — 断点命中且已报告, 调用方应退出 _check_multi_hart_bp.
            True — 未命中/已去重/条件不满足, 调用方应继续下一个断点.
        """
        if bp.kind == "addr":
            hit = self._bp_match_pc(h, h.pc, bp.value)
        elif bp.kind == "cond":
            hit = True
        elif bp.kind in ("instr", "opcode"):
            instr_raw = self._emu.bus.read_u32(h.pc)
            if instr_raw is None:
                return True
            f = decode_fields(instr_raw)
            if bp.kind == "instr":
                hit = (
                    f.opcode == 0x73
                    and f.func3 == 0
                    and f.func12 == bp.value
                )
            elif not f.is_compressed:
                hit = f.opcode == bp.value
            else:
                return True
        else:
            return True
        if not hit or not self._eval_bp_condition(h, bp):
            return True
        bp_key = (
            (bp.kind, h.pc) if bp.kind == "addr" else (bp.kind, bp.value)
        )
        if bp_key in self._bp_hit_this_run:
            return True
        self._bp_hit_this_run.add(bp_key)
        self._report_bp_hit(bp, h, h.pc)
        return None

    def _check_multi_hart_bp(self) -> bool:
        """多 hart 执行后检查断点 (仅检查需要 Python 侧求值的类型).

        addr (无条件) — 由 Rust 内联检查, 此处跳过.
        instr/opcode/cond/addr_with_cond — 需 Python 侧求值.

        Returns:
            True if a breakpoint was hit and the caller should pause.
        """
        if not self._has_non_addr_bps:
            return False
        for h in self._emu.harts:
            for bp in self._breakpoints:
                if bp.kind == "addr" and not bp.cond_type:
                    continue  # 纯 addr 断点 — Rust 已逐指令比对
                if self._check_single_bp(h, bp) is None:
                    return True
        return False
