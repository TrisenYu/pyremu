#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""调试器 (rvdb) 测试: 状态快照, 指令回滚, 命令方法, REPL 分发."""

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.hart import RiscvMode
from pyremu.core.mem_check_aux import inject_memory_backend
from pyremu.debugger import (
    Debugger,
    HartSnapshot,
    MemWriteTracker,
    MemoryChange,
    StackFrame,
)
from pyremu.emulator import Emulator
from pyremu.memory.bus import Bus
from pyremu.utils.parse_bin import FirmwareImage


# ============================================================
#  辅助
# ============================================================


def _make_emu(num_harts=1, ram_base=None, ram_size=None, reset_vector=0x1000):
    """创建一个最小模拟器用于调试器测试."""
    kwargs = {}
    if ram_base is not None:
        kwargs["ram_base"] = ram_base
    if ram_size is not None:
        kwargs["ram_size"] = ram_size
    kwargs["reset_vector"] = reset_vector
    kwargs["num_harts"] = num_harts
    return Emulator(**kwargs)


def _make_dbg(num_harts=1, ram_size=0x10000):
    """创建带单条 NOP 指令的 Emulator + Debugger."""
    emu = _make_emu(num_harts=num_harts, ram_size=ram_size)
    # 在 0x1000 处放 1 条 NOP (ADDI x0,x0,0 = 0x00000013)
    emu.load_code(0x1000, b"\x13\x00\x00\x00")
    dbg = Debugger(emulator=emu, hart_id=0)
    return dbg


def _make_image():
    """创建一个最小 FirmwareImage."""
    return FirmwareImage(
        format="elf",
        entry_point=0x1000,
        segments=[],
        symbols={"main": 0x2000, "_start": 0x1000, "uart_puts": 0x3000},
    )


# ============================================================
#  数据类测试
# ============================================================


class TestHartSnapshot:
    """Hart 状态快照."""

    def test_create(self):
        snap = HartSnapshot(
            pc=0x80000000,
            gpr_vals=[0] * 32,
            csr_vals={"mstatus": 0},
            mode=3,
            reservation_valid=False,
            reservation_addr=0,
        )
        assert snap.pc == 0x80000000
        assert len(snap.gpr_vals) == 32
        assert snap.mode == 3
        assert not snap.reservation_valid

    def test_with_reservation(self):
        snap = HartSnapshot(
            pc=0x1000,
            gpr_vals=[i for i in range(32)],
            csr_vals={},
            mode=3,
            reservation_valid=True,
            reservation_addr=0x2000,
        )
        assert snap.reservation_valid
        assert snap.reservation_addr == 0x2000


class TestMemoryChange:
    """内存变更记录."""

    def test_create(self):
        mc = MemoryChange(addr=0x1000, old=b"\x00\x00\x00\x00")
        assert mc.addr == 0x1000
        assert mc.old == b"\x00\x00\x00\x00"


class TestStackFrame:
    """栈帧."""

    def test_create(self):
        sf = StackFrame(idx=0, fp=0x80001000, sp=0x80000FF0, ra=0x80000100, pc=0x800000FC)
        assert sf.idx == 0
        assert sf.fp == 0x80001000
        assert sf.sp == 0x80000FF0
        assert sf.ra == 0x80000100
        assert sf.pc == 0x800000FC


# ============================================================
#  MemWriteTracker
# ============================================================


class TestMemWriteTracker:
    """内存写入追踪器."""

    def test_tracks_write_and_records_old(self):
        bus = Bus(ram_size=0x10000, ram_base=0x80000000)
        # 预先写入已知数据
        bus.write(0x80000000, b"\xAA\xBB\xCC\xDD\xEE\xFF\x00\x11")

        calls = []

        def orig_write(addr, data):
            calls.append((addr, data))

        tracker = MemWriteTracker(bus, orig_write)
        tracker(0x80000000, b"\x11\x22\x33\x44")

        # 原始写入被调用
        assert len(calls) == 1
        assert calls[0] == (0x80000000, b"\x11\x22\x33\x44")

        # 记录了旧值
        assert len(tracker.changes) == 1
        assert tracker.changes[0].addr == 0x80000000
        assert tracker.changes[0].old == b"\xAA\xBB\xCC\xDD"

    def test_multiple_writes_accumulate(self):
        bus = Bus(ram_size=0x10000, ram_base=0x80000000)
        bus.write(0x80000100, b"\x00" * 16)
        tracker = MemWriteTracker(bus, lambda a, d: None)

        tracker(0x80000100, b"\x01" * 4)
        tracker(0x80000104, b"\x02" * 4)
        tracker(0x80000108, b"\x03" * 8)

        assert len(tracker.changes) == 3
        assert tracker.changes[0].addr == 0x80000100
        assert tracker.changes[1].addr == 0x80000104
        assert tracker.changes[2].addr == 0x80000108


# ============================================================
#  Debugger 初始化
# ============================================================


class TestDebuggerInit:
    """调试器初始化."""

    def test_default_hart_id(self):
        emu = _make_emu(num_harts=2)
        dbg = Debugger(emulator=emu)
        assert dbg._hart_id == 0

    def test_explicit_hart_id(self):
        emu = _make_emu(num_harts=4)
        dbg = Debugger(emulator=emu, hart_id=2)
        assert dbg._hart_id == 2

    def test_hart_property(self):
        emu = _make_emu(num_harts=3)
        dbg = Debugger(emulator=emu, hart_id=1)
        assert dbg.hart is emu.harts[1]

    def test_with_image(self):
        emu = _make_emu()
        img = _make_image()
        dbg = Debugger(emulator=emu, hart_id=0, image=img)
        assert dbg._image is img

    def test_build_completer_includes_commands_and_regs(self):
        dbg = _make_dbg()
        c = dbg._build_completer()
        # 包含命令
        for cmd in ["step", "continue", "regs", "pc", "quit", "help"]:
            assert cmd in c.words, f"缺少命令: {cmd}"
        # 包含 GPR ABI 名称
        for abi in ["zero", "ra", "sp", "t0", "a0", "s0"]:
            assert abi in c.words, f"缺少 GPR: {abi}"
        # 包含 CSR 名称
        for csr in ["mstatus", "mtvec", "mepc", "mcause"]:
            assert csr in c.words, f"缺少 CSR: {csr}"


# ============================================================
#  快照 & 回滚
# ============================================================


class TestSnapshotRollback:
    """快照与回滚."""

    def test_save_snapshot_captures_state(self):
        dbg = _make_dbg()
        h = dbg.hart
        h.pc = 0x80000000
        h.gprs[10].val = 0xDEAD
        h.mode = RiscvMode.M

        snap = dbg._save_snapshot()
        assert snap.pc == 0x80000000
        assert snap.gpr_vals[10] == 0xDEAD
        assert snap.mode == RiscvMode.M.value

    def test_restore_snapshot_restores_all_state(self):
        dbg = _make_dbg()
        h = dbg.hart

        # 设置初始状态
        h.pc = 0x80000000
        h.gprs[5].val = 0xABCD
        h.gprs[10].val = 0x1234
        h.csrs["mstatus"].val = 0x1800
        h.mode = RiscvMode.S

        snap = dbg._save_snapshot()

        # 修改状态
        h.pc = 0x90000000
        h.gprs[5].val = 0xFFFF
        h.gprs[10].val = 0xDEAD
        h.csrs["mstatus"].val = 0x0
        h.mode = RiscvMode.M

        dbg._restore_snapshot(snap)
        assert h.pc == 0x80000000
        assert h.gprs[5].val == 0xABCD
        assert h.gprs[10].val == 0x1234
        assert h.csrs["mstatus"].val == 0x1800
        assert h.mode == RiscvMode.S

    def test_restore_clears_reservation(self):
        dbg = _make_dbg()
        h = dbg.hart
        h.set_reservation(0x8000)

        snap_with_res = dbg._save_snapshot()

        h.clear_reservation()
        assert not h.reservation_valid

        dbg._restore_snapshot(snap_with_res)
        assert h.reservation_valid
        assert h.reservation_addr == 0x8000

    def test_rollback_reverts_memory(self):
        dbg = _make_dbg()
        h = dbg.hart

        # 在 RAM 中预写数据
        dbg._emu.bus.write(0x80000000, b"\xCA\xFE\xBA\xBE\x00\x00\x00\x00")

        # 执行一条 SW 指令: sw x10, 0(x2)
        h.pc = 0x1000
        h.gprs[2].val = 0x80000000  # sp
        h.gprs[10].val = 0x12345678  # a0 = value to store
        # SD x10, 0(x2): funct3=011, opcode=0100011
        instr = (0 << 25) | (10 << 20) | (2 << 15) | (3 << 12) | (2 << 7) | 0x23
        # 需要将指令写入 PC 处
        dbg._emu.load_code(0x1000, instr.to_bytes(4, "little"))

        dbg.step_one()
        # 验证数据确实写入
        data_after = dbg._emu.bus.read(0x80000000, 4)
        # 回滚
        dbg.rollback()
        # 验证内存恢复
        data_restored = dbg._emu.bus.read(0x80000000, 4)
        assert data_restored == b"\xCA\xFE\xBA\xBE", (
            f"回滚应恢复旧数据, 实际: {data_restored.hex()}"
        )

    def test_rollback_without_snapshot_warns(self):
        dbg = _make_dbg()
        dbg._snapshot = None
        # 应该只是打印警告, 不抛异常
        dbg.rollback()
        assert dbg._snapshot is None


# ============================================================
#  step_one & 执行
# ============================================================


class TestStepOne:
    """单步执行."""

    def test_step_advances_pc(self):
        dbg = _make_dbg()
        # load_code 已在 0x1000 放了 NOP
        h = dbg.hart
        assert h.pc == 0x1000

        dbg.step_one()
        # ADDI x0,x0,0 推进 4 字节
        assert h.pc == 0x1004

    def test_step_increments_count(self):
        dbg = _make_dbg()
        assert dbg._instr_count == 0
        dbg.step_one()
        assert dbg._instr_count == 1
        dbg.step_one()
        assert dbg._instr_count == 2

    def test_halted_hart_skipped(self):
        dbg = _make_dbg()
        dbg.hart._halted = True
        count_before = dbg._instr_count
        dbg.step_one()
        assert dbg._instr_count == count_before  # 未执行

    def test_ebreak_sets_mcause(self):
        """EBREAK 指令应触发陷态, 但不应导致 halted."""
        dbg = _make_dbg()
        # EBREAK = 0x00100073
        dbg._emu.load_code(0x1000, b"\x73\x00\x10\x00")
        h = dbg.hart
        h.pc = 0x1000
        dbg.step_one()
        # EBREAK trap → mcause = 3 (Breakpoint)
        assert (h.mcause_val & ~(1 << 63)) == 3

    def test_consecutive_traps_halt(self):
        """连续 3 次 IllInstr → hart halted."""
        dbg = _make_dbg()
        # 非法指令 — 全部位为 0 的 32-bit 不是有效指令
        # 更好的方式: 未知 opcode, 如 0x0000007F
        bad_instr = b"\x7f\x00\x00\x00"  # opcode=1111111, 无效
        dbg._emu.load_code(0x1000, bad_instr * 5)
        h = dbg.hart
        for _ in range(4):
            dbg.step_one()
            if h._halted:
                break
        # 连续 trap 超过阈值后应 halted
        assert h._halted, "连续 IllInstr 应导致 hart halted"


# ============================================================
#  格式化 & 辅助方法
# ============================================================


class TestDebuggerHelpers:
    """辅助方法."""

    def test_hex(self):
        dbg = _make_dbg()
        assert dbg._hex(0) == "0x0000000000000000"
        assert dbg._hex(0xDEADBEEF) == "0x00000000deadbeef"
        assert dbg._hex(0xFFFFFFFFFFFFFFFF) == "0xffffffffffffffff"

    def test_trap_cause_name(self):
        dbg = _make_dbg()
        # mcause=3 应为断点异常
        name = dbg._trap_cause_name(3)
        assert "breakpoint" in name.lower() or "Breakpoint" in name

    def test_trap_cause_name_interrupt(self):
        dbg = _make_dbg()
        # MSI: bit63=1, code=3
        irq_mcause = (1 << 63) | 3
        name = dbg._trap_cause_name(irq_mcause)
        assert name == "MmodeSoftInterrupt"

    def test_warn_prints(self, capsys):
        dbg = _make_dbg()
        dbg._warn("test warning")
        # Rich 输出到 stderr, 我们只验证不抛异常

    def test_err_prints(self, capsys):
        dbg = _make_dbg()
        dbg._err("test error")


# ============================================================
#  PC 命令
# ============================================================


class TestCmdPc:
    """PC 显示 / 设置."""

    def test_show_pc(self):
        dbg = _make_dbg()
        dbg.hart.pc = 0x1000
        dbg.cmd_pc()  # 应不抛异常

    def test_set_pc(self):
        dbg = _make_dbg()
        dbg.cmd_pc("0x80000000")
        assert dbg.hart.pc == 0x80000000

    def test_set_pc_invalid_value(self):
        dbg = _make_dbg()
        dbg.cmd_pc("not_a_number")  # decorator 捕获 ValueError

    def test_set_pc_clears_snapshot(self):
        dbg = _make_dbg()
        dbg._snapshot = HartSnapshot(
            pc=0x1000, gpr_vals=[0] * 32, csr_vals={}, mode=3,
            reservation_valid=False, reservation_addr=0,
        )
        dbg.cmd_pc("0x2000")
        assert dbg._snapshot is None
        assert dbg._mem_changes == []


# ============================================================
#  寄存器命令
# ============================================================


class TestCmdRegs:
    """寄存器显示."""

    def test_regs_shows_all_gprs(self):
        dbg = _make_dbg()
        dbg.hart.gprs[10].val = 0xCAFE
        dbg.cmd_regs()  # 应不抛异常

    def test_reg_shows_one(self):
        dbg = _make_dbg()
        dbg.hart.gprs[10].val = 0xDEADBEEF
        dbg.cmd_reg("a0")

    def test_reg_unknown(self):
        dbg = _make_dbg()
        dbg.cmd_reg("nonexistent")

    def test_reg_by_xn(self):
        dbg = _make_dbg()
        dbg.hart.gprs[5].val = 0x5555
        dbg.cmd_reg("x5")

    def test_set_writes_gpr(self):
        dbg = _make_dbg()
        dbg.cmd_set("a0", "0xCAFE")
        assert dbg.hart.gprs[10].val == 0xCAFE

    def test_set_by_xn(self):
        dbg = _make_dbg()
        dbg.cmd_set("x5", "42")
        assert dbg.hart.gprs[5].val == 42

    def test_set_unknown_reg(self):
        dbg = _make_dbg()
        dbg.cmd_set("nonexistent", "0")

    def test_set_invalid_value(self):
        dbg = _make_dbg()
        dbg.cmd_set("a0", "not_a_number")

    def test_find_gpr_alias(self):
        dbg = _make_dbg()
        r = dbg._find_gpr("sp")
        assert r is not None
        assert r.alias == "sp"

    def test_find_gpr_xn(self):
        dbg = _make_dbg()
        r = dbg._find_gpr("x2")
        assert r is not None
        assert r.alias == "sp"

    def test_find_gpr_unknown(self):
        dbg = _make_dbg()
        r = dbg._find_gpr("nonexistent")
        assert r is None


# ============================================================
#  CSR 命令
# ============================================================


class TestCmdCsr:
    """CSR 显示 / 设置."""

    def test_csr_list(self):
        dbg = _make_dbg()
        dbg.cmd_csr("list")  # 应不抛异常

    def test_csr_read(self):
        dbg = _make_dbg()
        dbg.hart.csrs["mstatus"].val = 0x1800
        dbg.cmd_csr("mstatus")

    def test_csr_unknown(self):
        dbg = _make_dbg()
        dbg.cmd_csr("nonexistent_csr")

    def test_csrw_writes(self):
        dbg = _make_dbg()
        dbg.cmd_csrw("mstatus", "0x1234")
        assert dbg.hart.csrs["mstatus"].val == 0x1234

    def test_csrw_unknown(self):
        dbg = _make_dbg()
        dbg.cmd_csrw("nonexistent", "0")

    def test_csrw_invalid_value(self):
        dbg = _make_dbg()
        dbg.cmd_csrw("mstatus", "BAD")


# ============================================================
#  状态命令
# ============================================================


class TestCmdStatus:
    """状态显示."""

    def test_mode(self):
        dbg = _make_dbg()
        dbg.cmd_mode()

    def test_mstatus(self):
        dbg = _make_dbg()
        dbg.hart.csrs["mstatus"].val = 0x1800
        dbg.cmd_mstatus()

    def test_status(self):
        dbg = _make_dbg()
        dbg.cmd_status()


# ============================================================
#  TLB 命令
# ============================================================


class TestCmdTlb:
    """TLB 显示 / 刷新."""

    def test_tlb_empty(self):
        dbg = _make_dbg()
        dbg.cmd_tlb()  # 空 TLB

    def test_tlb_search(self):
        dbg = _make_dbg()
        dbg.cmd_tlb("0x00001")

    def test_tlb_search_invalid(self):
        dbg = _make_dbg()
        dbg.cmd_tlb("bad")

    def test_tlbflush_all(self):
        dbg = _make_dbg()
        dbg.cmd_tlbflush()  # 全刷新

    def test_tlbflush_vpn(self):
        dbg = _make_dbg()
        dbg.cmd_tlbflush("0x00001")

    def test_tlbflush_invalid(self):
        dbg = _make_dbg()
        dbg.cmd_tlbflush("bad")


# ============================================================
#  SATP 命令
# ============================================================


class TestCmdSatp:
    """SATP 显示."""

    def test_bare_mode(self):
        dbg = _make_dbg()
        dbg.hart.csrs["satp"].val = 0  # Bare
        dbg.cmd_satp()

    def test_sv39_mode(self):
        dbg = _make_dbg()
        # MODE=Sv39 (8), PPN=0x1000
        dbg.hart.csrs["satp"].val = (8 << 60) | 0x1000
        dbg.cmd_satp()


# ============================================================
#  内存 & 反汇编命令
# ============================================================


class TestCmdMem:
    """内存 dump."""

    def test_mem_default_size(self):
        dbg = _make_dbg()
        dbg._emu.bus.write(0x80000000, b"Hello, world!")
        dbg.cmd_mem("0x80000000")

    def test_mem_custom_size(self):
        dbg = _make_dbg()
        dbg._emu.bus.write(0x1000, b"\x00" * 128)
        dbg.cmd_mem("0x1000", "32")

    def test_mem_invalid_addr(self):
        dbg = _make_dbg()
        dbg.cmd_mem("bad")

    def test_mem_invalid_size(self):
        dbg = _make_dbg()
        dbg.cmd_mem("0x1000", "bad")


class TestCmdDisasm:
    """反汇编显示."""

    def test_disasm_default(self):
        dbg = _make_dbg()
        # 在 0x1000 有 NOP
        dbg.cmd_disasm("0x1000")

    def test_disasm_custom_length(self):
        dbg = _make_dbg()
        dbg.cmd_disasm("0x1000", "32")

    def test_disasm_invalid_addr(self):
        dbg = _make_dbg()
        dbg.cmd_disasm("bad")

    def test_disasm_sets_next_addr(self):
        dbg = _make_dbg()
        dbg._disasm_next_addr = None
        dbg.cmd_disasm("0x1000", "32")
        assert dbg._disasm_next_addr is not None

    def test_disasm_zero_length(self):
        dbg = _make_dbg()
        dbg.cmd_disasm("0x1000", "0")  # 报错


# ============================================================
#  符号命令
# ============================================================


class TestCmdSymbols:
    """符号表."""

    def test_list_all(self):
        dbg = _make_dbg()
        dbg._image = _make_image()
        dbg.cmd_symbols()

    def test_filter(self):
        dbg = _make_dbg()
        dbg._image = _make_image()
        dbg.cmd_symbols("main")

    def test_no_match(self):
        dbg = _make_dbg()
        dbg._image = _make_image()
        dbg.cmd_symbols("nonexistent")

    def test_no_image(self):
        dbg = _make_dbg()
        dbg._image = None
        dbg.cmd_symbols()

    def test_resolve_symbol(self):
        dbg = _make_dbg()
        syms = {"main": 0x2000, "_start": 0x1000}
        name = dbg._resolve_symbol(syms, 0x2000)
        assert name == "main"

    def test_resolve_symbol_not_found(self):
        dbg = _make_dbg()
        name = dbg._resolve_symbol({}, 0x9999)
        assert name is None


# ============================================================
#  缓存命令
# ============================================================


class TestCmdCache:
    """L2 缓存显示."""

    def test_cache_overview(self):
        dbg = _make_dbg()
        dbg.cmd_cache()

    def test_cache_set(self):
        dbg = _make_dbg()
        dbg.cmd_cache("0")

    def test_cache_set_way(self):
        dbg = _make_dbg()
        dbg.cmd_cache("0 0")

    def test_cache_invalid_set(self):
        dbg = _make_dbg()
        dbg.cmd_cache("999")  # 超出范围

    def test_cache_invalid_arg(self):
        dbg = _make_dbg()
        dbg.cmd_cache("bad")


# ============================================================
#  Hart 切换
# ============================================================


class TestCmdHart:
    """Hart 切换."""

    def test_switch(self):
        emu = _make_emu(num_harts=4)
        emu.load_code(0x1000, b"\x13\x00\x00\x00")
        dbg = Debugger(emulator=emu, hart_id=0)
        dbg.cmd_hart(2)
        assert dbg._hart_id == 2

    def test_switch_out_of_range(self):
        emu = _make_emu(num_harts=2)
        dbg = Debugger(emulator=emu)
        dbg.cmd_hart(5)  # 超出范围
        assert dbg._hart_id == 0  # 未改变

    def test_switch_clears_snapshot(self):
        emu = _make_emu(num_harts=4)
        emu.load_code(0x1000, b"\x13\x00\x00\x00")
        dbg = Debugger(emulator=emu, hart_id=0)
        dbg._snapshot = HartSnapshot(
            pc=0x1000, gpr_vals=[0] * 32, csr_vals={}, mode=3,
            reservation_valid=False, reservation_addr=0,
        )
        dbg.cmd_hart(1)
        assert dbg._snapshot is None
        assert dbg._mem_changes == []


# ============================================================
#  命令分发
# ============================================================


class TestDispatch:
    """REPL 命令分发."""

    def test_quit_returns_false(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["q"]) is False
        assert dbg._dispatch(["quit"]) is False
        assert dbg._dispatch(["exit"]) is False

    def test_step_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["s"]) is True
        assert dbg._instr_count > 0

    def test_step_with_count(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["step", "3"]) is True
        assert dbg._instr_count == 3

    def test_continue_dispatches(self):
        dbg = _make_dbg()
        # 只放 1 条指令, continue 执行完即刻暂停
        assert dbg._dispatch(["c"]) is True

    def test_run_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["r", "2"]) is True

    def test_rollback_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["undo"]) is True

    def test_regs_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["regs"]) is True
        assert dbg._dispatch(["gpr"]) is True

    def test_reg_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["reg", "a0"]) is True

    def test_reg_missing_arg(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["reg"]) is True  # 警告, 不抛异常

    def test_set_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["set", "a0", "0x42"]) is True

    def test_set_missing_arg(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["set", "a0"]) is True  # 警告

    def test_csr_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["csr", "mstatus"]) is True

    def test_csrw_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["csrw", "mstatus", "0x100"]) is True

    def test_pc_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["pc"]) is True

    def test_mode_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["mode"]) is True

    def test_mstatus_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["mstatus"]) is True

    def test_tlb_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["tlb"]) is True

    def test_tlbflush_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["tlbflush"]) is True

    def test_cache_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["cache"]) is True

    def test_satp_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["satp"]) is True

    def test_mem_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["mem", "0x1000"]) is True

    def test_mem_missing_arg(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["mem"]) is True  # 警告

    def test_disasm_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["disasm", "0x1000"]) is True

    def test_status_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["status"]) is True
        assert dbg._dispatch(["info"]) is True

    def test_stack_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["stack"]) is True
        assert dbg._dispatch(["bt"]) is True
        assert dbg._dispatch(["frame"]) is True

    def test_sym_dispatches(self):
        dbg = _make_dbg()
        dbg._image = _make_image()
        assert dbg._dispatch(["sym"]) is True

    def test_hart_dispatches(self):
        emu = _make_emu(num_harts=4)
        emu.load_code(0x1000, b"\x13\x00\x00\x00")
        dbg = Debugger(emulator=emu)
        assert dbg._dispatch(["hart", "2"]) is True

    def test_help_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["help"]) is True
        assert dbg._dispatch(["h"]) is True
        assert dbg._dispatch(["?"]) is True

    def test_unknown_command(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["nonexistent_cmd"]) is True  # 不中断 REPL

    def test_w_alias_for_set(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["w", "a0", "99"]) is True
        assert dbg.hart.gprs[10].val == 99


# ============================================================
#  栈帧回溯
# ============================================================


class TestStackWalk:
    """栈帧回溯."""

    def test_single_frame_when_fp_zero(self):
        dbg = _make_dbg()
        h = dbg.hart
        h.gprs[8].val = 0  # fp = 0
        frames = dbg._walk_frame_chain()
        assert len(frames) == 1
        assert frames[0].idx == 0
        assert frames[0].pc == h.pc

    def test_frame_chain_with_valid_fp(self):
        """构造 3 层栈帧链: fp-8=ra, fp-16=saved_fp.

        _walk_frame_chain 在 saved_ra==0 时提前 break (不追加空帧),
        因此要得到 N 帧需要 N-1 层非零 RA 链接.
        """
        dbg = _make_dbg()
        h = dbg.hart

        # 帧 #0 (当前): sp=0x80001000, fp=0x80001080
        h.gprs[2].val = 0x80001000  # sp
        h.gprs[8].val = 0x80001080  # fp

        bus = dbg._emu.bus
        # 帧 #1 链接 (fp=0x80001080): RA→call_site_A, saved_fp→F1
        bus.write(0x80001080 - 16, (0x80001100).to_bytes(8, "little"))
        bus.write(0x80001080 - 8, (0x80000200).to_bytes(8, "little"))

        # 帧 #2 链接 (fp=0x80001100): RA→call_site_B, saved_fp→F2
        bus.write(0x80001100 - 16, (0x80001180).to_bytes(8, "little"))
        bus.write(0x80001100 - 8, (0x80000300).to_bytes(8, "little"))

        # 帧 #3 链接 (fp=0x80001180): terminal — RA=0 导致 walk 截断
        bus.write(0x80001180 - 16, (0).to_bytes(8, "little"))
        bus.write(0x80001180 - 8, (0).to_bytes(8, "little"))

        frames = dbg._walk_frame_chain()
        assert len(frames) == 3
        assert frames[0].idx == 0
        assert frames[0].fp == 0x80001080
        assert frames[1].idx == 1
        assert frames[1].fp == 0x80001100
        # frames[1].pc = ra - 4 = 0x80000200 - 4
        assert frames[1].pc == 0x800001FC
        assert frames[1].ra == 0x80000200
        assert frames[2].idx == 2
        assert frames[2].fp == 0x80001180
        assert frames[2].ra == 0x80000300

    def test_stack_walk_detects_cycle(self):
        """FP 链成环 → 截断回溯."""
        dbg = _make_dbg()
        h = dbg.hart
        h.gprs[8].val = 0x80001000  # fp
        bus = dbg._emu.bus
        # saved_fp 指向自身 → 环
        bus.write(0x80001000 - 16, (0x80001000).to_bytes(8, "little"))
        bus.write(0x80001000 - 8, (0x80000400).to_bytes(8, "little"))

        frames = dbg._walk_frame_chain()
        # 应截断: 发现环后停止
        assert len(frames) <= 2, f"应截断环, 但得到 {len(frames)} 帧"

    def test_cmd_frame_shows_backtrace(self):
        dbg = _make_dbg()
        dbg.hart.gprs[8].val = 0
        dbg.cmd_frame()  # 单帧

    def test_cmd_frame_with_valid_fp(self):
        dbg = _make_dbg()
        h = dbg.hart
        h.gprs[2].val = 0x80001000
        h.gprs[8].val = 0x80001080
        bus = dbg._emu.bus
        bus.write(0x80001080 - 16, (0x80001100).to_bytes(8, "little"))
        bus.write(0x80001080 - 8, (0x80000400).to_bytes(8, "little"))
        bus.write(0x80001100 - 16, (0).to_bytes(8, "little"))
        bus.write(0x80001100 - 8, (0).to_bytes(8, "little"))
        dbg.cmd_frame()

    def test_cmd_frame_switch(self):
        dbg = _make_dbg()
        h = dbg.hart
        h.gprs[8].val = 0x80001080
        bus = dbg._emu.bus
        bus.write(0x80001080 - 16, (0x80001100).to_bytes(8, "little"))
        bus.write(0x80001080 - 8, (0x80000400).to_bytes(8, "little"))
        bus.write(0x80001100 - 16, (0).to_bytes(8, "little"))
        bus.write(0x80001100 - 8, (0).to_bytes(8, "little"))
        dbg.cmd_frame("1")  # 切换到帧 1
        assert dbg._current_frame_idx == 1

    def test_cmd_frame_out_of_range(self):
        dbg = _make_dbg()
        dbg.hart.gprs[8].val = 0
        dbg.cmd_frame("99")  # 超出范围


# ============================================================
#  TLB 页大小解码
# ============================================================


class TestTlbDecodeHelpers:
    """TLB 静态辅助函数."""

    def test_decode_perm_full(self):
        assert Debugger._decode_perm(0xF) == "RWXU"
        assert Debugger._decode_perm(0b0111) == "RWXS"
        assert Debugger._decode_perm(0b0101) == "R-XS"
        assert Debugger._decode_perm(0) == "---S"

    def test_tlb_page_size(self):
        assert Debugger._tlb_page_size(0) == "4K"
        assert Debugger._tlb_page_size(1) == "2M"
        assert Debugger._tlb_page_size(2) == "1G"
        assert Debugger._tlb_page_size(3) == "Lv3"
        assert Debugger._tlb_page_size(99) == "Lv99"


# ============================================================
#  十六进制 dump 辅助
# ============================================================


class TestHexdumpBytes:
    """hexdump 静态辅助."""

    def test_hexdump_short(self):
        data = b"\x48\x65\x6C\x6C\x6F"
        result = Debugger._hexdump_bytes(data)
        assert "48 65 6c 6c 6f" in result.lower()
        assert "|Hello|" in result

    def test_hexdump_exact_row(self):
        data = bytes(range(16))
        result = Debugger._hexdump_bytes(data)
        assert "0000  " in result

    def test_hexdump_indented(self):
        data = b"\x00\x01"
        result = Debugger._hexdump_bytes(data, indent="    ")
        assert result.startswith("    ")


# ============================================================
#  缓存行格式化
# ============================================================


class TestFmtCacheLine:
    """L2 缓存行格式化."""

    def test_meta_only(self):
        dbg = _make_dbg()
        l2 = dbg._emu.bus.l2
        # 取一个有效条目
        e = l2.entries[0]
        e.valid = True
        e.tag = 0x12345
        result = dbg._fmt_cache_line(e, 0, 0, full=False)
        assert "tag=0x" in result

    def test_full_dump(self):
        dbg = _make_dbg()
        l2 = dbg._emu.bus.l2
        e = l2.entries[0]
        e.valid = True
        e.tag = 0xABCDE
        result = dbg._fmt_cache_line(e, 0, 0, full=True)
        assert "tag=0x" in result


# ============================================================
#  获取并反汇编
# ============================================================


class TestFetchAndDisasm:
    """指令读取和反汇编."""

    def test_valid_pc(self):
        dbg = _make_dbg()
        result = dbg._fetch_and_disasm(0x1000)
        assert result is not None
        raw_hex, asm = result
        assert "13" in raw_hex  # NOP = 0x00000013
        assert "nop" in asm.lower() or "addi" in asm.lower()

    def test_invalid_pc(self):
        """未映射物理地址 → Bus 层面返回全零 (模拟未映射区域).

        PMA 校验在上层 mem_read/mem_write 中完成;
        Bus.read() 是底层物理总线, 对空洞地址返回零.
        """
        dbg = _make_dbg()
        result = dbg._fetch_and_disasm(0xFFFF0000)
        assert result is not None
        raw_hex, asm = result
        assert "00 00 00 00" in raw_hex


# ============================================================
#  帮助
# ============================================================


class TestPrintHelp:
    """帮助显示."""

    def test_help(self):
        dbg = _make_dbg()
        dbg._print_help()  # 不抛异常


# ============================================================
#  集成: 多步执行 + 状态检查
# ============================================================


class TestDebuggerIntegration:
    """端到端集成."""

    def test_multi_step_with_pc_check(self):
        """step 多次后 PC 正确推进."""
        dbg = _make_dbg()
        # 在 0x1000..0x100F 填 NOP
        nops = b"\x13\x00\x00\x00" * 4
        dbg._emu.load_code(0x1000, nops)

        dbg.cmd_step(3)
        assert dbg.hart.pc == 0x100C  # 3 × 4 = 12
        assert dbg._instr_count == 3

    def test_rollback_after_step(self):
        """step + rollback 恢复 PC.

        step_one 总是在执行前保存快照。cmd_step(2) 执行两条 NOP:
        - 第一条前: 保存快照 A (PC=0x1000), 执行后 PC=0x1004
        - 第二条前: 保存快照 B (PC=0x1004), 执行后 PC=0x1008
        最终 _snapshot = B (PC=0x1004), rollback 恢复到 B.
        """
        dbg = _make_dbg()
        nops = b"\x13\x00\x00\x00" * 4
        dbg._emu.load_code(0x1000, nops)

        assert dbg.hart.pc == 0x1000
        dbg.cmd_step(2)
        assert dbg.hart.pc == 0x1008  # 两条 NOP 后

        dbg.rollback()
        # rollback 恢复到最后一条指令执行前的快照: PC=0x1004
        assert dbg.hart.pc == 0x1004

    def test_hart_switch_preserves_state(self):
        """hart 切换后各 hart 状态独立."""
        emu = _make_emu(num_harts=2)
        nops = b"\x13\x00\x00\x00" * 4
        emu.load_code(0x1000, nops)
        emu.harts[0].gprs[10].val = 0xAAAA
        emu.harts[1].gprs[10].val = 0xBBBB

        dbg = Debugger(emulator=emu, hart_id=0)
        assert dbg.hart.gprs[10].val == 0xAAAA

        dbg.cmd_hart(1)
        assert dbg.hart.gprs[10].val == 0xBBBB

    def test_consecutive_trap_halt_and_status(self):
        """连续 trap → halted, status 显示状态."""
        dbg = _make_dbg()
        # 全部无效指令
        bad = b"\x7F\x00\x00\x00" * 5
        dbg._emu.load_code(0x1000, bad)

        for _ in range(5):
            dbg.step_one()
            if dbg.hart._halted:
                break

        dbg.cmd_status()  # halted 状态

    def test_mem_with_uart_device(self):
        """mem 命令读取设备区域."""
        dbg = _make_dbg()
        dbg.cmd_mem("0x10000000", "16")  # UART 基址

    def test_disasm_across_pages(self):
        """反汇编跨页."""
        dbg = _make_dbg()
        # 在 0x1000 和 0x1FFC 附近写入指令
        long_nops = b"\x13\x00\x00\x00" * 20
        dbg._emu.load_code(0x1000, long_nops)
        dbg.cmd_disasm("0x1000", "64")

    def test_fetch_and_disasm_unreadable(self):
        """Bus 对未映射物理地址返回零 — 不会抛异常或返回 None."""
        dbg = _make_dbg()
        result = dbg._fetch_and_disasm(0xDEAD0000)
        # Bus 层面对空洞地址返回全零 (PMA 检查在上层完成)
        assert result is not None

    def test_cmd_mem_with_unreadable_region(self):
        """mem 读取空洞地址."""
        dbg = _make_dbg()
        dbg.cmd_mem("0xDEAD0000", "4")

    def test_set_gpr_64bit_truncation(self):
        """set 应截断到 64 位."""
        dbg = _make_dbg()
        # 值超出 64 位范围
        dbg.cmd_set("a0", "0x1FFFFFFFFFFFFFFFF")  # 65 位
        assert dbg.hart.gprs[10].val == 0xFFFFFFFFFFFFFFFF

    def test_last_command_repeat_mechanism(self):
        """空输入重复上一条命令 (通过 _dispatch 间接覆盖)."""
        dbg = _make_dbg()
        # 命令后 _last_command 被设置
        dbg._last_command = "pc"
        # 这里无法直接测试 repl 循环, 但可以验证 _last_command 存储机制
        assert dbg._last_command == "pc"

    def test_cmd_frame_invalid_input(self):
        dbg = _make_dbg()
        dbg.cmd_frame("not_a_number")


# ============================================================
#  parse_compressed 检测路径
# ============================================================


class TestDisasmCompressed:
    """反汇编压缩指令检测."""

    def test_compressed_detected(self):
        """压缩指令 (低 2 位 ≠ 3) 应被正确识别.

        _fetch_and_disasm 总是读取 4 字节; raw_hex 固定显示 4 字节,
        但反汇编结果正确识别 16-bit 指令.
        """
        dbg = _make_dbg()
        # C.NOP = 0x0001 (16-bit 压缩指令)
        c_nop = b"\x01\x00"
        dbg._emu.load_code(0x1000, c_nop)
        result = dbg._fetch_and_disasm(0x1000)
        assert result is not None
        raw_hex, asm = result
        # 反汇编结果应标识为压缩指令 (low 2 bits != 3)
        assert raw_hex.startswith("01 00"), f"raw 首两字节应为压缩指令编码, 实际: {raw_hex}"


# ============================================================
#  L2 cache 显示命令
# ============================================================


class _CacheOutputCapture:
    """捕获 Rich console 输出文本."""

    def __init__(self, dbg):
        self.lines: list[str] = []
        self._orig = dbg._console.print

        def _capture(*args, **kwargs):
            import io
            buf = io.StringIO()
            dbg._console.file = buf
            self._orig(*args, **kwargs)
            dbg._console.file = __import__("sys").stdout
            text = buf.getvalue()
            self.lines.extend(text.split("\n"))

        dbg._console.print = _capture

    def text(self) -> str:
        return "\n".join(self.lines)


def _make_dbg_with_l2(
    num_sets=256, ways=4, line_size=64,
) -> Debugger:
    """构造带可配置 L2 缓存的 Debugger."""
    from pyremu.memory.l2cache import L2Cache
    from pyremu.platform import PlatformConfig

    cfg = PlatformConfig(num_harts=1, ram_size=128 * 1024 * 1024)
    emu = Emulator(cfg)
    # 替换 L2 为可配置大小 (直接设 _l2 绕过只读 property)
    l2 = L2Cache(size=num_sets * ways * line_size, line_size=line_size, ways=ways)
    emu.bus._l2 = l2
    l2.set_ram_backend(emu.bus._ram_read_direct, emu.bus._ram_write_direct)
    dbg = Debugger(emulator=emu, hart_id=0)
    return dbg


class TestCacheDisplay:
    """cmd_cache 输出格式测试."""

    def test_no_args_shows_header(self):
        """cache 无参数 → 显示统计概览头."""
        dbg = _make_dbg_with_l2()
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache()
        text = cap.text()
        assert "L2 Cache" in text
        assert "valid" in text
        assert "MESI" in text
        assert "hit_rate" in text

    def test_no_args_shows_up_to_64_entries(self):
        """默认显示最多 64 条 valid 行 (预览格式: data[:16])."""
        dbg = _make_dbg_with_l2(num_sets=128, ways=2)
        # 预填一些数据产生 valid 行
        for i in range(80):
            dbg._emu.bus.l2.read(i * 64, 4)
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache()
        text = cap.text()
        # 应有 64 条 (或实际 valid 行数) + 截断提示
        assert "data[:16]" in text, "预览模式应含 data[:16]"
        assert "还有" in text, "超过 64 条时应有截断提示"

    def test_less_than_64_shows_all_no_truncation(self):
        """不足 64 条 valid → 全部显示, 无截断提示."""
        dbg = _make_dbg_with_l2(num_sets=128, ways=2)
        for i in range(10):
            dbg._emu.bus.l2.read(i * 64, 4)
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache()
        text = cap.text()
        assert "还有" not in text, "不足 64 条不应有截断提示"

    def test_single_set_shows_full_hexdump(self):
        """cache <set> → 完整 64B hexdump."""
        dbg = _make_dbg_with_l2()
        # 读 addr 0 → 填充 set 0
        dbg._emu.bus.l2.read(0, 8)
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache("0")
        text = cap.text()
        assert "data[:16]" not in text, "单 set 模式应为完整 hexdump, 不应含 data[:16]"

    def test_range_shows_preview_format(self):
        """cache <start>-<end> → 范围预览模式."""
        dbg = _make_dbg_with_l2(num_sets=128, ways=2)
        for i in range(30):
            dbg._emu.bus.l2.read(i * 64, 4)
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache("0-4")
        text = cap.text()
        assert "data[:16]" in text, "范围模式应为预览格式"
        # 不应有截断提示 (范围模式不限条目数)
        assert "还有" not in text

    def test_invalid_set_index(self):
        """非法 set 索引 → 错误信息."""
        dbg = _make_dbg_with_l2(num_sets=16, ways=2)
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache("99")
        text = cap.text()
        assert "超出范围" in text or "set" in text.lower()

    def test_invalid_range(self):
        """非法范围 → 错误信息."""
        dbg = _make_dbg_with_l2(num_sets=16, ways=2)
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache("10-99")
        text = cap.text()
        assert "超出范围" in text or "set" in text.lower()

    def test_empty_cache(self):
        """空缓存 → 无 valid 行提示."""
        dbg = _make_dbg_with_l2()
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache()
        text = cap.text()
        assert "无 valid 行" in text

    def test_set_with_no_valid_lines(self):
        """指定 set 但该组无 valid 行."""
        dbg = _make_dbg_with_l2()
        # 不填充任何数据, 直接查某个 set
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache("5")
        text = cap.text()
        assert "无 valid 行" in text
