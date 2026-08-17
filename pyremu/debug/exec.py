#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""ExecutionMixin — 指令执行、快照回滚、写监控、运行循环.

依赖 DebuggerBase + MemoryMixin._try_read_va + future BreakpointMixin.
"""

from __future__ import annotations

from pathlib import Path as _Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

    from pyremu.debug.types import (
        Breakpoint,
        MemoryChange,
        StackFrame,
    )


from pyremu._native import decode_fields
from pyremu.core.decoder import Hart
from pyremu.core.hart import RiscvMode
from pyremu.core.mem_check_aux import inject_memory_backend
from pyremu.core.trap_def import TrapType
from pyremu.core.trap_handler import check_pending_interrupts, deliver_trap
from pyremu.debug._attrs import SharedMixinAttrs
from pyremu.debug.types import HartSnapshot, MemWriteTracker
from pyremu.debug.utils import hex_addr
from pyremu.emulator import Emulator, RunStopReason, TimeoutError
from pyremu.env_inject import Preloader
from pyremu.utils.mask import mask64
from pyremu.utils.parse_bin import FirmwareImage

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


    def _run_loop(
        self,
        *,
        timeout: float | None = None,
    ) -> None:
        """连续执行 — 引擎无指令配额, 直至停止事件.

        停止事件与接管:
        - Ctrl+Q 设备暂停事件 — 引擎逐指令检查、即时退出, 状态保留
          在 live 数组中; stop_flag 由本函数消费 -> _paused.
        - 固件停机序列 (semihosting SYS_EXIT) -> 调试器接管.
        - addr 断点命中 (Rust 内联比对, EXIT_BREAKPOINT) -> 报告.
        - 全部 hart halted.
        - 时钟源超时 (TimeoutError).

        Args:
            timeout: 时钟源超时秒数, 默认 3600 (1 小时). 设为 0 禁用.
        """
        self._running = True
        self._paused = False
        self._terminated = False
        self._bp_hit_this_run.clear()
        self._enter_run_mode()
        timeout = _DEFAULT_TIMEOUT if timeout is None else timeout

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

        try:
            self._emu.run(timeout=timeout, yield_every=0)
        except TimeoutError:
            self._warn(f"运行超时 ({timeout:.0f}s), 强制停止")
            self._show_trap_context(self.hart)
            self._paused = True
        finally:
            # 引擎可能在无退出事件时长时间运行; 一轮结束刷新 UART 输出.
            self._flush_uart_if_present()

        reason = self._emu._run_stop_reason
        if self._emu._native_stop_flag.value != 0:
            # Ctrl+Q 事件 — 全部 hart 已暂停, 消费标志后回到 REPL.
            self._emu._native_stop_flag.value = 0
            self._paused = True
        if reason == RunStopReason.BREAKPOINT:
            self._check_multi_hart_bp(include_addr=True)
            self._paused = True
        elif reason == RunStopReason.EBREAK:
            self._console.print(
                "[dim]固件执行停机序列 (semihosting SYS_EXIT) — 全部 hart 停止[/]"
            )
            self._paused = True
        elif all(h._halted for h in self._emu.harts):
            self._warn("所有 Hart 已暂停")
            self._paused = True

        self._running = False
        self._enter_repl_mode()

    # ----------------------------------------------------------
    #  运行命令
    # ----------------------------------------------------------

    def cmd_step(self, count: int = 1) -> None:
        """单步或多步执行.

        step       — 执行 1 条指令, 显示 PC 及反汇编
        step <n>   — 执行 n 条指令, 仅显示最终状态.

        恒为纯 Python 逐条执行 (step_one): 步进语义要求精确逐条
        + 快照/回滚, 指令单轮加速执行设计已去除.
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
        self._console.print("[dim]继续执行 (Ctrl+Q 暂停)...[/]")
        # 若任一 hart 的 PC 恰好落在地址断点上, 先步进越过该断点,
        # 否则引擎的 pre-execution 检查会立即再次命中同一断点.
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
            # 同步更新 _bp_addrs, 否则 Rust 引擎仍会命中已移除的断点
            saved_bp_addrs = self._emu._bp_addrs
            self._emu._bp_addrs = [
                a for a in saved_bp_addrs if a not in bp_values_to_skip
            ]
            # 纯 Python 单步 (emu.step 全 hart 各一条) 精确跳过断点处指令,
            # 恢复断点后继续连续执行.
            try:
                self._emu.step()
            finally:
                self._breakpoints.extend(bps_to_skip)
                self._refresh_bp_cache()
                self._emu._bp_addrs = saved_bp_addrs
        self._run_loop()

    def cmd_run(self, n: int = 1) -> None:
        self._console.print(f"[dim]执行 {n} 条指令...[/]")
        self._paused = False
        self._hart_paused.clear()
        for _ in range(n):
            self._emu.step()
            if self._check_multi_hart_bp(include_addr=True):
                return
            if self.hart._halted:
                self._show_trap_context(self.hart)
                return
        self.cmd_pc()

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

        # 清除加速执行所用动态链接库内管理的断点地址
        emu._bp_addrs.clear()

        # 若 emu.__init__ 后 hart 数量未变, _speedup_hart_states 仍有效;
        # 标记 _native_stop_flag 重置以防上一轮 Ctrl+Q 未清理.
        emu._native_stop_flag.value = 0

        # 确保终端恢复 + SIGINT 设为 REPL 处理器;
        # 前次运行若异常退出 (未执行 _enter_repl_mode) 会残留
        # cbreak 模式与 _sigint_run handler, 导致 Ctrl+Q 失灵.
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
            f"  [green]* 写监控[/] {hex_addr(addr)}-{hex_addr(addr + size)}"
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

    def _check_multi_hart_bp(self, include_addr: bool = False) -> bool:
        """多 hart 执行后检查断点.

        addr (无条件) — native 连续执行时由 Rust 内联比对, 引擎以
        EXIT_BREAKPOINT 退出后经 ``include_addr=True`` 在此报告;
        纯 Python 路径 (cmd_run / step_one) 无 Rust 比对, 恒传
        ``include_addr=True``. instr/opcode/cond/addr_with_cond —
        需 Python 侧求值.

        Returns:
            True if a breakpoint was hit and the caller should pause.
        """
        if not self._has_non_addr_bps and not include_addr:
            return False
        for h in self._emu.harts:
            for bp in self._breakpoints:
                if bp.kind == "addr" and not bp.cond_type and not include_addr:
                    continue  # 纯 addr 断点 — Rust 已逐指令比对
                if self._check_single_bp(h, bp) is None:
                    return True
        return False
