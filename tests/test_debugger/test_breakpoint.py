#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""测试 pyremu.debug.breakpoint — 断点设置/命中检查."""


from pyremu.debug.breakpoint import BreakpointMixin
from pyremu.debug.types import Breakpoint
from pyremu.emulator import Emulator

# ============================================================
#  Mini 测试类
# ============================================================


class _TestBpDbg(BreakpointMixin):
    """最小聚合类供 BreakpointMixin 测试."""

    def __init__(self, emu, hart_id=0):
        from rich.console import Console
        self._emu = emu
        self._hart_id = hart_id
        self._console = Console(highlight=False)
        self._warn = lambda msg: None
        self._err = lambda msg: None
        self._breakpoints: list[Breakpoint] = []
        self._bp_mode = "sync"
        self._hart_paused: set[int] = set()
        self._bp_hit_this_run: set[tuple] = set()
        self._prev_instr_csr_addr = -1
        self._image = None
        self._load_offset = 0
        self._sym_symbols = None
        self._paused = False
        self._sym_ranges = []

    @property
    def hart(self):
        return self._emu.harts[self._hart_id]

    # Stubs for cross-mixin dependencies
    def _fetch_and_disasm(self, pc):
        return ("00000013", "nop")

    def _resolve_addr(self, arg):
        try:
            return int(arg, 0)
        except (ValueError, TypeError):
            return None

    def _read_gpr_by_name(self, name):
        from pyremu.core.registers import gpr_idx_from_name
        idx = gpr_idx_from_name(name)
        return 0 if idx is None else self.hart.read_gpr(idx)

    def _read_csr_by_name(self, name):
        from pyremu.core.registers import csr_addr_from_name
        addr = csr_addr_from_name(name)
        return 0 if addr is None else self.hart.read_csr(addr)

    def _resolve_sym_addr(self, va):
        return None

    def _check_rv64_addr(self, v):
        return 0 <= v < (1 << 64)


def _make_bpdbg():
    emu = Emulator(prog_cnt=0x1000)
    emu.load_code(0x1000, b"\x13\x00\x00\x00")  # NOP
    return _TestBpDbg(emu)


# ============================================================
#  _bp_match_pc
# ============================================================


class TestBpMatchPc:
    """静态地址匹配方法."""

    def test_direct_match(self):
        dbg = _make_bpdbg()
        assert dbg._bp_match_pc(dbg.hart, 0x80000000, 0x80000000)

    def test_no_match(self):
        dbg = _make_bpdbg()
        assert not dbg._bp_match_pc(dbg.hart, 0x1000, 0x2000)

    def test_bare_mode_pc_is_pa(self):
        dbg = _make_bpdbg()
        dbg.hart.pc = 0x80001000
        assert dbg._bp_match_pc(dbg.hart, dbg.hart.pc, 0x80001000)


# ============================================================
#  _eval_bp_condition
# ============================================================


class TestEvalBpCondition:
    """条件评估."""

    def test_no_condition_always_true(self):
        dbg = _make_bpdbg()
        bp = Breakpoint(kind="addr", value=0x1000, desc="test")
        assert dbg._eval_bp_condition(dbg.hart, bp)

    def test_reg_eq_true(self):
        dbg = _make_bpdbg()
        dbg.hart.gprs[10] = 42  # a0
        bp = Breakpoint(
            kind="cond", value=0, desc="test",
            cond_type="reg", cond_reg="a0", cond_op="==", cond_val=42,
        )
        assert dbg._eval_bp_condition(dbg.hart, bp)

    def test_reg_eq_false(self):
        dbg = _make_bpdbg()
        dbg.hart.gprs[10] = 42
        bp = Breakpoint(
            kind="cond", value=0, desc="test",
            cond_type="reg", cond_reg="a0", cond_op="==", cond_val=99,
        )
        assert not dbg._eval_bp_condition(dbg.hart, bp)

    def test_reg_ne(self):
        dbg = _make_bpdbg()
        dbg.hart.gprs[10] = 42
        bp = Breakpoint(
            kind="cond", value=0, desc="test",
            cond_type="reg", cond_reg="a0", cond_op="!=", cond_val=99,
        )
        assert dbg._eval_bp_condition(dbg.hart, bp)

    def test_reg_lt(self):
        dbg = _make_bpdbg()
        dbg.hart.gprs[10] = 5
        bp = Breakpoint(
            kind="cond", value=0, desc="test",
            cond_type="reg", cond_reg="a0", cond_op="<", cond_val=10,
        )
        assert dbg._eval_bp_condition(dbg.hart, bp)

    def test_reg_gt(self):
        dbg = _make_bpdbg()
        dbg.hart.gprs[10] = 100
        bp = Breakpoint(
            kind="cond", value=0, desc="test",
            cond_type="reg", cond_reg="a0", cond_op=">", cond_val=10,
        )
        assert dbg._eval_bp_condition(dbg.hart, bp)

    def test_csr_eq(self):
        dbg = _make_bpdbg()
        dbg.hart.csrs["mstatus"].val = 0x1800
        bp = Breakpoint(
            kind="cond", value=0, desc="test",
            cond_type="csr", cond_reg="mstatus", cond_op="==", cond_val=0x1800,
        )
        assert dbg._eval_bp_condition(dbg.hart, bp)


# ============================================================
#  断点命中检查
# ============================================================


class TestCheckBreakpoints:
    """_check_breakpoints 核心逻辑."""

    def test_no_breakpoints_returns_false(self):
        dbg = _make_bpdbg()
        assert not dbg._check_breakpoints(dbg.hart, 0x1000, 0x00000013)

    def test_addr_bp_hit(self):
        dbg = _make_bpdbg()
        dbg._breakpoints = [Breakpoint(kind="addr", value=0x1000, desc="test")]
        assert dbg._check_breakpoints(dbg.hart, 0x1000, 0x00000013)

    def test_addr_bp_no_hit(self):
        dbg = _make_bpdbg()
        dbg._breakpoints = [Breakpoint(kind="addr", value=0x2000, desc="test")]
        assert not dbg._check_breakpoints(dbg.hart, 0x1000, 0x00000013)

    def test_opcode_bp_hit(self):
        dbg = _make_bpdbg()
        # 0x13 = OP-IMM (ADDI)
        dbg._breakpoints = [Breakpoint(kind="opcode", value=0x13, desc="op")]
        assert dbg._check_breakpoints(dbg.hart, 0x1000, 0x00000013)

    def test_opcode_bp_skips_compressed(self):
        """压缩指令应跳过 opcode 断点检查."""
        dbg = _make_bpdbg()
        dbg._breakpoints = [Breakpoint(kind="opcode", value=0x13, desc="op")]
        # 0x4501 = c.li a0, 0 (压缩指令, 低 2 位 = 01 ≠ 11)
        assert not dbg._check_breakpoints(dbg.hart, 0x1000, 0x4501)


# ============================================================
#  断点管理命令
# ============================================================


class TestBpCommands:
    """断点列表/删除/清除/模式."""

    def test_list_empty(self):
        dbg = _make_bpdbg()
        dbg.cmd_bp_list()  # 应不抛异常

    def test_list_with_items(self):
        dbg = _make_bpdbg()
        dbg._breakpoints = [
            Breakpoint(kind="addr", value=0x1000, desc="test"),
            Breakpoint(kind="instr", value=0, desc="ecall"),
        ]
        dbg.cmd_bp_list()

    def test_delete_valid(self):
        dbg = _make_bpdbg()
        dbg._breakpoints = [Breakpoint(kind="addr", value=0x1000, desc="test")]
        dbg.cmd_bp_delete("1")
        assert len(dbg._breakpoints) == 0

    def test_delete_out_of_range(self):
        dbg = _make_bpdbg()
        dbg.cmd_bp_delete("99")

    def test_clear(self):
        dbg = _make_bpdbg()
        dbg._breakpoints = [
            Breakpoint(kind="addr", value=0x1000, desc="a"),
            Breakpoint(kind="addr", value=0x2000, desc="b"),
        ]
        dbg.cmd_bp_clear()
        assert len(dbg._breakpoints) == 0

    def test_mode_get(self):
        dbg = _make_bpdbg()
        dbg.cmd_bp_mode()  # 显示当前模式

    def test_mode_set_sync(self):
        dbg = _make_bpdbg()
        dbg._bp_mode = "async"
        dbg.cmd_bp_mode("sync")
        assert dbg._bp_mode == "sync"

    def test_mode_set_async(self):
        dbg = _make_bpdbg()
        dbg.cmd_bp_mode("async")
        assert dbg._bp_mode == "async"

    def test_mode_invalid(self):
        dbg = _make_bpdbg()
        dbg.cmd_bp_mode("invalid")


class TestBpOpcode:
    """opcode 断点设置."""

    def test_valid_opcode(self):
        dbg = _make_bpdbg()
        dbg.cmd_bp_opcode("0x13")
        assert len(dbg._breakpoints) == 1
        assert dbg._breakpoints[0].kind == "opcode"
        assert dbg._breakpoints[0].value == 0x13

    def test_invalid_opcode_beyond_7bit(self):
        dbg = _make_bpdbg()
        dbg.cmd_bp_opcode("0xFF")

    def test_unknown_opcode(self):
        """未知 opcode 应报错."""
        dbg = _make_bpdbg()
        dbg.cmd_bp_opcode("0x7E")  # 不是已知 RV64 opcode


class TestDispatchBp:
    """_dispatch_bp 子命令路由."""

    def test_empty_shows_list(self):
        dbg = _make_bpdbg()
        dbg._dispatch_bp([])  # 等价于 bp list

    def test_mode_subcommand(self):
        dbg = _make_bpdbg()
        dbg._dispatch_bp(["mode"])

    def test_clear_subcommand(self):
        dbg = _make_bpdbg()
        dbg._breakpoints = [Breakpoint(kind="addr", value=0x1000, desc="test")]
        dbg._dispatch_bp(["clear"])
        assert len(dbg._breakpoints) == 0

    def test_delete_subcommand(self):
        dbg = _make_bpdbg()
        dbg._breakpoints = [Breakpoint(kind="addr", value=0x1000, desc="test")]
        dbg._dispatch_bp(["delete", "1"])
        assert len(dbg._breakpoints) == 0

    def test_if_cond_bp(self):
        dbg = _make_bpdbg()
        dbg._dispatch_bp(["if", "reg", "a0", "==", "42"])
        assert len(dbg._breakpoints) == 1
        assert dbg._breakpoints[0].kind == "cond"
        assert dbg._breakpoints[0].cond_type == "reg"
        assert dbg._breakpoints[0].cond_reg == "a0"

    def test_named_instr_bp(self):
        dbg = _make_bpdbg()
        dbg._dispatch_bp(["ecall"])
        assert len(dbg._breakpoints) == 1
        assert dbg._breakpoints[0].kind == "instr"

    def test_set_addr_bp(self):
        dbg = _make_bpdbg()
        dbg._dispatch_bp(["0x80000000"])
        assert len(dbg._breakpoints) == 1
        assert dbg._breakpoints[0].kind == "addr"
        assert dbg._breakpoints[0].value == 0x80000000

    def test_opcode_subcommand(self):
        dbg = _make_bpdbg()
        dbg._dispatch_bp(["opcode", "0x13"])
        assert len(dbg._breakpoints) == 1
        assert dbg._breakpoints[0].kind == "opcode"
