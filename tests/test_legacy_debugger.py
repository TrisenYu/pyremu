#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""调试器 (rvdb) 测试: 状态快照, 指令回滚, 命令方法, REPL 分发."""
import io
import sys

import pytest

from pyremu.core.hart import RiscvMode
from pyremu.core.registers import gpr_alias
from pyremu.debugger import (
    Debugger,
    HartSnapshot,
    MAX_INSTR_COUNT,
    MemoryChange,
    MemWriteTracker,
    StackFrame,
)
from pyremu.emulator import Emulator
from pyremu.memory.bus import Bus
from pyremu.memory.l2cache import L2Cache
from pyremu.memory.mmu import PTE, satp_root_ppn, sv39_walk
from pyremu.memory.pmp import PMP_A_TOR, PMP_R, PMP_W, PMP_X
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import FirmwareImage, FirmwareSegment

# ============================================================
#  辅助
# ============================================================
L1_BASE = 0x80000000
L2_BASE = 0x80001000
L3_BASE = 0x80002000
target_va = 0x1000


def _make_emu(num_harts=1, ram_base=None, ram_size=None, prog_cnt=0x1000):
    """创建一个最小模拟器用于调试器测试."""
    kwargs = {}
    if ram_base is not None:
        kwargs["ram_base"] = ram_base
    if ram_size is not None:
        kwargs["ram_size"] = ram_size
    kwargs["prog_cnt"] = prog_cnt
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
        bus.write(0x80000000, b"\xaa\xbb\xcc\xdd\xee\xff\x00\x11")

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
        assert tracker.changes[0].old == b"\xaa\xbb\xcc\xdd"

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
        # WordCompleter.words 是 Sequence[str] | (() -> Sequence[str]) 联合类型
        words = c.words() if callable(c.words) else c.words
        # 包含命令
        for cmd in ["step", "continue", "regs", "pc", "quit", "help"]:
            assert cmd in words, f"缺少命令: {cmd}"
        # 包含 GPR ABI 名称
        for abi in ["zero", "ra", "sp", "t0", "a0", "s0"]:
            assert abi in words, f"缺少 GPR: {abi}"
        # 包含 CSR 名称
        for csr in ["mstatus", "mtvec", "mepc", "mcause"]:
            assert csr in words, f"缺少 CSR: {csr}"


# ============================================================
#  快照 & 回滚
# ============================================================


class TestSnapshotRollback:
    """快照与回滚."""

    def test_save_snapshot_captures_state(self):
        dbg = _make_dbg()
        h = dbg.hart
        h.pc = 0x80000000
        h.gprs[10] = 0xDEAD
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
        h.gprs[5] = 0xABCD
        h.gprs[10] = 0x1234
        h.csrs["mstatus"].val = 0x1800
        h.mode = RiscvMode.S

        snap = dbg._save_snapshot()

        # 修改状态
        h.pc = 0x90000000
        h.gprs[5] = 0xFFFF
        h.gprs[10] = 0xDEAD
        h.csrs["mstatus"].val = 0x0
        h.mode = RiscvMode.M

        dbg._restore_snapshot(snap)
        assert h.pc == 0x80000000
        assert h.gprs[5] == 0xABCD
        assert h.gprs[10] == 0x1234
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
        dbg._emu.bus.write(0x80000000, b"\xca\xfe\xba\xbe\x00\x00\x00\x00")

        # 执行一条 SW 指令: sw x10, 0(x2)
        h.pc = 0x1000
        h.gprs[2] = 0x80000000  # sp
        h.gprs[10] = 0x12345678  # a0 = value to store
        # SD x10, 0(x2): funct3=011, opcode=0100011
        instr = (0 << 25) | (10 << 20) | (2 << 15) | (3 << 12) | (2 << 7) | 0x23
        # 需要将指令写入 PC 处
        dbg._emu.load_code(0x1000, instr.to_bytes(4, "little"))

        dbg.step_one()
        # 验证数据确实写入
        dbg._emu.bus.read(0x80000000, 4)
        # 回滚
        dbg.rollback()
        # 验证内存恢复
        data_restored = dbg._emu.bus.read(0x80000000, 4)
        assert data_restored == b"\xca\xfe\xba\xbe", (
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
        # EBREAK trap -> mcause = 3 (Breakpoint)
        assert (h.mcause_val & ~(1 << 63)) == 3


# ============================================================
#  格式化 & 辅助方法
# ============================================================


class TestDebuggerHelpers:
    """辅助方法."""

    @pytest.mark.parametrize(
        "v, expected",
        [
            (0, "0x0000000000000000"),
            (0xDEADBEEF, "0x00000000deadbeef"),
            (0xFFFFFFFFFFFFFFFF, "0xffffffffffffffff"),
            (-1, "0xffffffffffffffff"),  # 负值被掩码为 64-bit
        ],
    )
    def test_hex(self, v, expected):
        assert Debugger._hex(v) == expected

    @pytest.mark.parametrize(
        "n, expected",
        [
            (0, "0 B"),
            (512, "512 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1048576, "1.0 MB"),
            (1073741824, "1.0 GB"),
            (1099511627776, "1.0 TB"),
            (1125899906842624, "1.0 PB"),
            # 超出最大单位: 落到 EB, 不退回 B
            (2**70 * 12, "12288.0 EB"),  # 12 ZB -> 最远单位 EB
        ],
    )
    def test_fmt_size(self, n, expected):
        assert Debugger._fmt_size(n) == expected

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
            pc=0x1000,
            gpr_vals=[0] * 32,
            csr_vals={},
            mode=3,
            reservation_valid=False,
            reservation_addr=0,
        )
        dbg.cmd_pc("0x2000")
        assert dbg._snapshot is None
        assert dbg._mem_changes == []

    def test_set_pc_negative(self):
        """pc -1 被硬拒绝."""
        dbg = _make_dbg()
        dbg.hart.pc = 0x1000
        dbg.cmd_pc("-1")
        assert dbg.hart.pc == 0x1000  # 未改变

    def test_set_pc_bare_out_of_ram(self):
        """Bare 模式下 PC 在 RAM 外时给出警告."""
        dbg = _make_dbg()
        dbg.hart.pc = 0x1000
        dbg.cmd_pc("0xFFFFFFFF00000000")  # 大地址, Bare 模式警告但不阻止

    def test_set_pc_sv39_bad_sign_ext(self):
        """Sv39 下 bits[63:39] 不等于 bit 38 时警告."""
        dbg = _make_dbg()
        dbg.hart.satp_val = (8 << 60) | 1  # Sv39 mode
        dbg.hart.pc = 0x1000
        dbg.cmd_pc("0xFFFF800000010000")  # bits[63:39] 全 1, bit 38 = 0 — 格式错误


# ============================================================
#  寄存器命令
# ============================================================


class TestCmdRegs:
    """寄存器显示."""

    def test_regs_shows_all_gprs(self):
        dbg = _make_dbg()
        dbg.hart.gprs[10] = 0xCAFE
        dbg.cmd_regs()  # 应不抛异常

    def test_reg_shows_one(self):
        dbg = _make_dbg()
        dbg.hart.gprs[10] = 0xDEADBEEF
        dbg.cmd_reg("a0")

    def test_reg_unknown(self):
        dbg = _make_dbg()
        dbg.cmd_reg("nonexistent")

    def test_reg_by_xn(self):
        dbg = _make_dbg()
        dbg.hart.gprs[5] = 0x5555
        dbg.cmd_reg("x5")

    def test_set_writes_gpr(self):
        dbg = _make_dbg()
        dbg.cmd_set("a0", "0xCAFE")
        assert dbg.hart.gprs[10] == 0xCAFE

    def test_set_by_xn(self):
        dbg = _make_dbg()
        dbg.cmd_set("x5", "42")
        assert dbg.hart.gprs[5] == 42

    def test_set_unknown_reg(self):
        dbg = _make_dbg()
        dbg.cmd_set("nonexistent", "0")

    def test_set_invalid_value(self):
        dbg = _make_dbg()
        dbg.cmd_set("a0", "not_a_number")

    def test_find_gpr_alias(self):
        dbg = _make_dbg()
        idx = dbg._find_gpr("sp")
        assert idx is not None
        assert gpr_alias(idx) == "sp"

    def test_find_gpr_xn(self):
        dbg = _make_dbg()
        idx = dbg._find_gpr("x2")
        assert idx is not None
        assert gpr_alias(idx) == "sp"

    def test_find_gpr_unknown(self):
        dbg = _make_dbg()
        idx = dbg._find_gpr("nonexistent")
        assert idx is None


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

    def test_status_multi_hart_overview(self):
        """多 hart 时 info 无参数显示全部 hart 概览."""
        dbg = _make_dbg(num_harts=4)
        dbg.cmd_status()  # 不抛异常即通过

    def test_status_detail(self):
        """info <hart_id> 显示指定 hart 详情."""
        dbg = _make_dbg(num_harts=2)
        dbg.cmd_status("1")  # 查看 hart 1

    def test_status_invalid_hart_id(self):
        """info 非法 hart ID."""
        dbg = _make_dbg()
        dbg.cmd_status("999")  # 不抛异常
        dbg.cmd_status("bad")  # 非整数


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

    def test_tlb_negative_vpn(self):
        dbg = _make_dbg()
        dbg.cmd_tlb("-1")  # 被拒绝, 不抛异常

    def test_tlbflush_negative_vpn(self):
        dbg = _make_dbg()
        dbg.cmd_tlbflush("-1")  # 被拒绝, 不抛异常


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

    def test_mem_negative_addr(self):
        dbg = _make_dbg()
        dbg.cmd_mem("-1")

    def test_mem_addr_overflow(self):
        dbg = _make_dbg()
        dbg.cmd_mem("1234567890123456789012345789")


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

    def test_disasm_negative_addr(self):
        dbg = _make_dbg()
        dbg.cmd_disasm("-1")

    def test_disasm_addr_overflow(self):
        dbg = _make_dbg()
        dbg.cmd_disasm("1234567890123456789012345789")

    def test_disasm_sets_next_addr(self):
        dbg = _make_dbg()
        dbg._disasm_next_addr = None
        dbg.cmd_disasm("0x1000", "32")
        assert dbg._disasm_next_addr is not None

    def test_disasm_zero_length(self):
        dbg = _make_dbg()
        dbg.cmd_disasm("0x1000", "0")  # 报错

    def test_count_instrs_non_ram_addr_returns_limit(self):
        """非 RAM 地址 (MMIO/空洞) 直接返回上限值, 不做遍历."""
        dbg = _make_dbg()
        # 0x10000000 是 UART MMIO, 不在 RAM 范围内
        result = dbg._count_instrs_between(0x10000000, 0x10000100)
        assert result >= MAX_INSTR_COUNT, f"非 RAM 起始地址应返回上限值, 实际 {result}"

    def test_count_instrs_iteration_limit(self):
        """超过 _MAX_INSTR_COUNT 条指令后停止遍历."""
        dbg = _make_dbg(ram_size=0x200000)  # 2 MiB RAM
        # RAM 范围内的大区间, 遍历因条目数超限而截断
        ram_base = dbg._emu.bus.ram_base
        result = dbg._count_instrs_between(ram_base, ram_base + 0x100000)
        assert result <= MAX_INSTR_COUNT, f"指令数不应超过 {MAX_INSTR_COUNT}, 实际 {result}"

    def test_disasm_far_from_ref_no_hang(self):
        """ref_pc 与 disasm 地址相距甚远时不挂死, 且禁用步数."""
        dbg = _make_dbg()
        dbg._disasm_ref_pc = 0x80000000  # kernel 入口
        dbg._disasm_past_terminator = False
        dbg.cmd_disasm("0x10000000", "32")  # UART 地址 — 远在 ref_pc 之上
        # 不应挂死 — 到达这里即通过
        assert dbg._disasm_past_terminator is True, "大跨距应触发 past_terminator 禁用 +N 步数"

    def test_count_instrs_empty_range(self):
        """start >= end 时返回 0."""
        dbg = _make_dbg()
        assert dbg._count_instrs_between(0x80001000, 0x80001000) == 0
        assert dbg._count_instrs_between(0x80002000, 0x80001000) == 0

    def test_disasm_step_accumulates_through_negative(self):
        """Enter 重复时负步数也累积推进, 抵达 ref_pc 后正确显示 +N."""
        dbg = _make_dbg()
        # 将 hart PC 设在较远处, 从 ref_pc 之前开始 disasm
        dbg.hart.pc = 0x80001000
        dbg._disasm_past_terminator = False

        # 第一段: ref_pc 之前, 写入 4 条 nop
        for off in range(0, 16, 4):
            dbg._emu.bus.write(0x80000FF0 + off, b"\x13\x00\x00\x00")
        dbg.cmd_disasm("0x80000ff0", "4")
        step_after_first = dbg._disasm_base_step
        # 4 条 nop 之后步数到达 ref_pc, next_base 应为 0
        assert step_after_first <= 0, f"ref_pc 之前 base_step 应 ≤0, 实际 {step_after_first}"

        # 模拟 Enter 重复: 推进到 ref_pc 所在块
        dbg._disasm_next_addr = 0x80001000
        dbg._emu.bus.write(0x80001000, b"\x13\x00\x00\x00" * 4)
        dbg.cmd_disasm(hex(dbg._disasm_next_addr), "4")
        step_after_second = dbg._disasm_base_step
        # Enter 重复后, ref_pc 已过, 步数应为正
        assert step_after_second > 0, (
            f"过 ref_pc 后 base_step 应为正, 实际 {step_after_second}"
        )
        # past_terminator 不应被误触发
        assert dbg._disasm_past_terminator is False, "非终止指令不应设置 past_terminator"

    def test_disasm_odd_addr_auto_aligns(self):
        """非 2-字节对齐地址自动向下对齐并警告."""
        dbg = _make_dbg()
        # 写入已知指令以便反汇编
        dbg._emu.bus.write(0x80000FF0, b"\x13\x00\x00\x00" * 4)
        # 奇数地址
        dbg.cmd_disasm("0x80000ff1", "16")
        # 不应挂死, 且 _disasm_next_addr 应从对齐后的地址计算
        # (没有异常即通过)
        assert dbg._disasm_next_addr is not None
        # 对齐后应从 0x80000FF0 开始, next_addr ≥ 0x80000FF0 + 4*nop
        assert dbg._disasm_next_addr >= 0x80000FF0 + 4, (
            f"应从对齐地址开始反汇编, next_addr=0x{dbg._disasm_next_addr:x}"
        )

    def test_disasm_aligned_addr_no_warning(self):
        """对齐地址不应触发警告."""
        dbg = _make_dbg()
        dbg._emu.bus.write(0x80001000, b"\x13\x00\x00\x00" * 4)
        dbg.cmd_disasm("0x80001000", "16")
        # 正常完成
        assert dbg._disasm_next_addr is not None

    def test_disasm_always_uses_physical_address(self):
        """disasm 始终直读物理地址 (即使 MMU 使能)."""
        dbg = _make_dbg(ram_size=0x200000)
        hart = dbg.hart
        bus = dbg._emu.bus

        # 在 PA 0x80010000 写入两条已知指令: NOP + ADDI a0,a0,1
        target_pa = 0x80010000
        code = (
            b"\x13\x00\x00\x00"  # nop
            b"\x13\x05\x15\x00"  # addi a0, a0, 1
        )
        bus.write(target_pa, code)

        # 切换到 S 模式并启用 Sv39 MMU
        hart.mode = RiscvMode.S
        hart.satp_val = (8 << 60) | 0x100  # 任意 root PPN, 不影响 PA 直读

        # disasm 物理地址 — 直接读 PA, 不翻译
        dbg.cmd_disasm(hex(target_pa), "2")

        # 验证下一条地址: 两条 4 字节指令 = 8 字节后
        assert dbg._disasm_next_addr is not None
        assert dbg._disasm_next_addr == target_pa + 8, (
            f"disasm PA: next_addr=0x{dbg._disasm_next_addr:x}, "
            f"期望 0x{target_pa + 8:x}"
        )

    def test_disasm_mmu_bare_falls_back_to_physical(self):
        """Bare 模式下 disasm 直接使用物理地址 (无翻译)."""
        dbg = _make_dbg()
        # 在 PA 0x1000 处写两条指令
        dbg._emu.bus.write(0x1000, b"\x13\x00\x00\x00\x13\x05\x15\x00")
        dbg.hart.satp_val = 0  # Bare
        dbg.cmd_disasm("0x1000", "2")
        # 两条 4 字节指令, next_addr 应在 0x1000 + 8
        assert dbg._disasm_next_addr == 0x1000 + 8

    def test_vdisasm_2mib_megapage_ppn_to_pa(self):
        """vdisasm 在 2 MiB 超级页下正确计算 VA->PA.

        验证两项修复:
        1. PPN 掩码 9-bit (∼0x1FF) 而非 10-bit (∼0x3FF) — bit 9 保留.
        2. sv39_walk 返回 PPN, 调用方必须计算 PA = (ppn << 12) | page_off.
        """
        dbg = _make_dbg(ram_size=0x200000)
        hart = dbg.hart
        bus = dbg._emu.bus

        # 2 MiB 超级页: L1 指针 -> L2 大页叶 (PPN bit 9=1, 触发 2026-07-02 掩码回归)
        target_va_2m = 0x200000  # VA 在 2 MiB 对齐边界
        vpn2 = (target_va_2m >> 30) & 0x1FF
        vpn1 = (target_va_2m >> 21) & 0x1FF  # 2 MiB 页: L2 索引为 vpn1

        def _write_pte(pa, ppn, **flags):
            pte = PTE()
            pte.ppn = ppn & 0xF_FFFF_FFFF
            for f, v in flags.items():
                setattr(pte, f, v)
            bus.write(pa, pte.to_int().to_bytes(8, "little"))

        # L1: 指针 -> L2 页表
        _write_pte(L1_BASE + vpn2 * 8, L2_BASE >> 12, v=True)
        # L2: 2 MiB 大页叶, PPN=0x80200 (bit 9=1, bits[8:0]=0 — Linux 启动场景复现)
        mega_ppn = 0x80200
        _write_pte(L2_BASE + vpn1 * 8, mega_ppn, v=True, r=True, w=True, x=True)

        # 在映射目标 PA 处写入两条已知指令
        target_pa = 0x80200000
        bus.write(target_pa, b"\x13\x00\x00\x00\x13\x05\x15\x00")  # nop + addi a0,a0,1

        hart.mode = RiscvMode.S
        hart.satp_val = (8 << 60) | (L1_BASE >> 12)
        hart.dtlb.flush_all()
        hart.itlb.flush_all()

        # vdisasm — 经 _try_read_va_forced -> sv39_walk -> PA = (ppn << 12) | off
        dbg.cmd_vdisasm(hex(target_va_2m), "2")

        # 验证 next_addr 正确 (两条 4 字节指令, 共 8 字节)
        assert dbg._vdisasm_next_addr is not None
        assert dbg._vdisasm_next_addr == target_va_2m + 8, (
            f"vdisasm 2M: next_addr=0x{dbg._vdisasm_next_addr:x}, "
            f"期望 0x{target_va_2m + 8:x}"
        )

    def test_vdisasm_ppn_not_pa(self):
        """sv39_walk 返回 PPN 而非 PA — vdisasm 必须经 << 12 | offset 换算.

        若调用方将 sv39_walk 的第二个返回值当作 PA 直接使用,
        读出的物理地址会少 12 个 bit, 读到全零或错误数据.
        """
        dbg = _make_dbg(ram_size=0x200000)
        hart = dbg.hart
        bus = dbg._emu.bus

        # 在 PA 0x80005000 写入 NOP (与映射的 PPN 差 12 bit 位移)
        target_pa = 0x80005000
        bus.write(target_pa, b"\x13\x00\x00\x00")  # nop

        target_va_4k = 0x5000
        vpn2 = (target_va_4k >> 30) & 0x1FF
        vpn1 = (target_va_4k >> 21) & 0x1FF
        vpn0 = (target_va_4k >> 12) & 0x1FF

        def _write_pte(pa, ppn, **flags):
            pte = PTE()
            pte.ppn = ppn & 0xF_FFFF_FFFF
            for f, v in flags.items():
                setattr(pte, f, v)
            bus.write(pa, pte.to_int().to_bytes(8, "little"))

        _write_pte(L1_BASE + vpn2 * 8, L2_BASE >> 12, v=True)
        _write_pte(L2_BASE + vpn1 * 8, L3_BASE >> 12, v=True)
        _write_pte(
            L3_BASE + vpn0 * 8, target_pa >> 12,
            v=True, r=True, w=True, x=True,
        )

        hart.mode = RiscvMode.S
        hart.satp_val = (8 << 60) | (L1_BASE >> 12)
        hart.dtlb.flush_all()
        hart.itlb.flush_all()

        # 直接调 sv39_walk: 返回 PPN, 不是 PA
        root_ppn = satp_root_ppn(hart.satp_val)
        assert hart._mem_read_phy is not None
        ok, ppn, _perm, _ps = sv39_walk(
            root_ppn, target_va_4k, hart._mem_read_phy
        )
        assert ok, "sv39_walk 应成功"
        # PPN != PA: 若调用方把 PPN 当 PA 用, 会读到错误地址
        page_off = target_va_4k & 0xFFF
        actual_pa = (ppn << 12) | page_off
        assert actual_pa == target_pa, (
            f"PA 计算错误: ppn=0x{ppn:x}, 正确 PA=0x{target_pa:x}, "
            f"若把 PPN 当 PA 则会得到 0x{ppn:x}"
        )
        # 关键回归: PPN 本身 != 目标物理地址 (差异超过页内偏移)
        assert ppn != target_pa, (
            f"PPN=0x{ppn:x} 不应等于目标 PA=0x{target_pa:x}; "
            f"sv39_walk 返回的是 PPN, 调用方必须 << 12 | offset"
        )


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

    def test_resolve_symbol_closest_predecessor(self):
        """最近前驱符号匹配: PC 在函数中间时返回所在函数名."""
        dbg = _make_dbg()
        syms = {"_start": 0x1000, "main": 0x2000, "init_warmboot": 0xE000}
        assert dbg._resolve_symbol(syms, 0x2004) == "main"  # 中间
        assert dbg._resolve_symbol(syms, 0x2000) == "main"  # 精确
        assert dbg._resolve_symbol(syms, 0x100C) == "_start"  # 中间
        assert dbg._resolve_symbol(syms, 0xE670) == "init_warmboot"  # 中间

    def test_resolve_symbol_far_away_returns_none(self):
        """距离 > 64 KiB 视为不在任何函数内."""
        dbg = _make_dbg()
        syms = {"main": 0x2000}
        # 0x12000 - 0x2000 = 0x10000 (64 KiB) 仍在范围内
        # 0x12001 - 0x2000 = 0x10001 > 64 KiB -> None
        assert dbg._resolve_symbol(syms, 0x12001) is None

    def test_resolve_symbol_with_pie_offset_applied(self):
        """调用方已减去 _load_offset, 传入链接时地址能正确匹配."""
        dbg = _make_dbg()
        dbg._load_offset = 0x80000000
        syms = {"fdt_next_tag": 0x21D58}
        # _fn_name 传运行时 PC, 内部减去 _load_offset
        runtime_pc = 0x80021EC4  # fdt_next_tag 内部
        name = dbg._resolve_symbol(syms, runtime_pc - dbg._load_offset)
        assert name == "fdt_next_tag"

    def test_resolve_symbol_range_exact_match(self):
        """范围匹配: start ≤ addr < end -> 返回正确的包含符号."""
        dbg = _make_dbg()
        # 模拟 vmlinux 中的场景: aio_complete_rw 和 vfs_coredump 距离 60KB+
        syms = {"aio_complete_rw": 0x1000, "vfs_coredump": 0x11000}
        ranges = [(0x1000, 0x1080, "aio_complete_rw"), (0x11000, 0x12000, "vfs_coredump")]
        # 0x1060 在 aio_complete_rw 范围内 — 旧算法会错误返回 vfs_coredump
        assert dbg._resolve_symbol(syms, 0x1060, ranges) == "aio_complete_rw"
        # 0x1000 精确起始
        assert dbg._resolve_symbol(syms, 0x1000, ranges) == "aio_complete_rw"
        # 0x107F 刚好在边界内
        assert dbg._resolve_symbol(syms, 0x107F, ranges) == "aio_complete_rw"
        # 0x1080 = end — 不在任何范围内, ranges 存在时返回 None
        # (不回退到最近前驱: 错误的函数名比无符号名更有害)
        assert dbg._resolve_symbol(syms, 0x1080, ranges) is None

    def test_resolve_symbol_range_no_match_returns_none(self):
        """ranges 存在但地址不在任何范围内 -> 返回 None (不回退猜测)."""
        dbg = _make_dbg()
        syms = {"func_a": 0x1000, "func_b": 0x1200}
        ranges = [(0x1000, 0x1080, "func_a"), (0x1200, 0x1280, "func_b")]
        # 0x1100 不在任何范围内, ranges 存在 -> 返回 None
        # (旧行为: 返回 "func_a" 作为最近前驱. 这在 kernel 中会错误将
        #  静态函数内的 PC 挂到前一个 GLOBAL 符号名下)
        assert dbg._resolve_symbol(syms, 0x1100, ranges) is None

    def test_resolve_symbol_no_ranges_still_works(self):
        """无 ranges 参数时回退到旧的前驱算法 (向后兼容)."""
        dbg = _make_dbg()
        syms = {"main": 0x2000}
        assert dbg._resolve_symbol(syms, 0x2004) == "main"  # 中间
        assert dbg._resolve_symbol(syms, 0x2000) == "main"  # 精确

    def test_resolve_symbol_local_func_correctly_resolved(self):
        """LOCAL 静态函数内的 PC 正确解析, 不挂到前驱 GLOBAL 符号上.

        真实回追溯源 (2026-07-04, Linux vmlinux bt):
          workqueue_sysfs_register (GLOBAL): 0x80045664, size=254
          process_scheduled_works  (LOCAL):  0x8004604c, size=912  ← PC=0x800461c6
          worker_thread            (LOCAL):  0x80047a94, size=638  ← PC=0x80047c38
          kthread_blkcg           (GLOBAL): 0x8004d770, size=34
          kthread                 (LOCAL):  0x8004d794, size=256  ← PC=0x8004d870

        旧行为: LOCAL 符号被 skip_local 过滤, ranges 中缺失, PC 经二分查找
        未命中后回退到最近前驱 GLOBAL, 导致 #03~#05 帧函数名全错.
        """
        dbg = _make_dbg()
        syms = {
            "workqueue_sysfs_register": 0x80045664,
            "process_scheduled_works": 0x8004604C,
            "worker_thread": 0x80047A94,
            "kthread_blkcg": 0x8004D770,
            "kthread": 0x8004D794,
        }
        ranges = [
            (0x80045664, 0x80045762, "workqueue_sysfs_register"),
            (0x8004604C, 0x800463DC, "process_scheduled_works"),
            (0x80047A94, 0x80047D12, "worker_thread"),
            (0x8004D770, 0x8004D792, "kthread_blkcg"),
            (0x8004D794, 0x8004D894, "kthread"),
        ]
        # 旧行为 (无 ranges): 前驱匹配 -> workqueue_sysfs_register (错误!)
        # 新行为 (有 ranges): 精确包含 -> process_scheduled_works
        assert dbg._resolve_symbol(syms, 0x800461C6, ranges) == "process_scheduled_works"
        assert dbg._resolve_symbol(syms, 0x80047C38, ranges) == "worker_thread"
        assert dbg._resolve_symbol(syms, 0x8004D870, ranges) == "kthread"
        # GLOBAL 函数仍正常匹配
        assert dbg._resolve_symbol(syms, 0x80045664, ranges) == "workqueue_sysfs_register"

    def test_resolve_symbol_gap_between_funcs_returns_none(self):
        """两函数间空隙地址: ranges 存在 -> 返回 None, 不猜测.

        kthread_blkcg 结束于 0x8004D792, kthread 始于 0x8004D794.
        间隙 [0x8004D792, 0x8004D794) 2 字节 — 既不属于前者也不属于后者.
        """
        dbg = _make_dbg()
        syms = {"kthread_blkcg": 0x8004D770, "kthread": 0x8004D794}
        ranges = [
            (0x8004D770, 0x8004D792, "kthread_blkcg"),
            (0x8004D794, 0x8004D894, "kthread"),
        ]
        assert dbg._resolve_symbol(syms, 0x8004D792, ranges) is None  # end 排除
        assert dbg._resolve_symbol(syms, 0x8004D793, ranges) is None  # 间隙中
        assert dbg._resolve_symbol(syms, 0x8004D794, ranges) == "kthread"  # 边界内

    def test_find_segment_with_pie_offset(self):
        """_find_segment 传入链接时地址 (已减 _load_offset) 匹配段."""
        image = FirmwareImage(
            entry_point=0x0,
            format="elf",
            segments=[FirmwareSegment(vaddr=0x0, data=b"", memsz=0x3EBB0, name=".text")],
            symbols={},
        )
        dbg = _make_dbg()
        dbg._image = image
        dbg._load_offset = 0x80000000
        # 运行时地址 0x80021ec4, 减 _load_offset 后 = 0x21ec4 在 .text 范围内
        seg = dbg._find_segment(0x21EC4)
        assert seg is not None
        assert seg.name == ".text"

    def test_find_segment_miss_far_address(self):
        """不落在任何段内返回 None (远地址)."""
        dbg = _make_dbg()
        seg = dbg._find_segment(0xFFFF_FFFF)
        assert seg is None

    def test_find_segment_gap_between_segments_returns_none(self):
        """段间空隙中的地址返回 None — 不做最近前驱猜测.

        模拟真实固件布局: .text 在 VA 0x0, .coffer_enclave_man 在 VA 0x180000.
        内核 Image 加载在 VA 0x200000 (fw_jump 模式), 无对应固件段.
        VA 0x201048 在 .coffer_enclave_man 之后 0x72e23 字节 — 距离足够近
        (在旧实现的 1 MiB 阈值内) 会触发错误猜测, 返回 .coffer_enclave_man.
        修复后应返回 None.
        """
        image = FirmwareImage(
            entry_point=0x0,
            format="elf",
            segments=[
                FirmwareSegment(vaddr=0x0, data=b"", memsz=0x3F260, name=".text"),
                FirmwareSegment(
                    vaddr=0x180000, data=b"",
                    memsz=0xE225, name=".coffer_enclave_man"
                ),
            ],
            symbols={},
        )
        dbg = _make_dbg()
        dbg._image = image
        dbg._load_offset = 0x80000000
        # 内核 Image 区域 (fw_jump): VA 0x200000 起, 无对应段
        seg = dbg._find_segment(0x201048)
        assert seg is None, (
            f"段间空隙地址不应匹配任何段, 但返回了 {seg.name if seg else None}"
        )

    def test_find_segment_gap_just_after_segment_returns_none(self):
        """紧接段末尾之后的地址返回 None.

        旧实现会在距离 ≤ 1 MiB 时返回最近前驱段,
        这对栈回溯是有害的 — 返回错误的段名比无段名更误导.
        """
        image = FirmwareImage(
            entry_point=0x0,
            format="elf",
            segments=[
                FirmwareSegment(vaddr=0x1000, data=b"", memsz=0x100, name=".text"),
                FirmwareSegment(vaddr=0x2000, data=b"", memsz=0x80, name=".rodata"),
            ],
            symbols={},
        )
        dbg = _make_dbg()
        dbg._image = image
        # .text 结束于 0x1100, .rodata 开始于 0x2000
        # 0x1100 在 1 MiB 内且紧接 .text — 旧实现会返回 .text
        seg = dbg._find_segment(0x1100)
        assert seg is None, (
            f"段间空隙 0x1100 不应匹配 .text, 但返回了 {seg.name if seg else None}"
        )
        # 0x1FFF 紧接 .rodata 之前 — 同样不应匹配
        seg = dbg._find_segment(0x1FFF)
        assert seg is None, (
            f"段间空隙 0x1FFF 不应匹配 .rodata, 但返回了 {seg.name if seg else None}"
        )


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
            pc=0x1000,
            gpr_vals=[0] * 32,
            csr_vals={},
            mode=3,
            reservation_valid=False,
            reservation_addr=0,
        )
        dbg.cmd_hart(1)
        assert dbg._snapshot is None
        assert dbg._mem_changes == []


# ============================================================
#  命令分发
# ============================================================


# ============================================================
#  断点命令
# ============================================================


class _BpOutputCapture:
    """捕获断点相关 Rich 输出."""

    def __init__(self, dbg) -> None:
        self._dbg = dbg
        self.lines: list[str] = []

    def __enter__(self):
        self._dbg._console.file = self
        return self

    def write(self, s: str, **_) -> None:
        self.lines.append(s)

    def __exit__(self, *_) -> None:
        self._dbg._console.file = sys.stderr

    def text(self) -> str:
        return "".join(self.lines)


class TestCmdBreakpoint:
    """断点设置、查询、命中."""

    # -- 地址断点 --

    def test_set_addr_bp(self):
        dbg = _make_dbg()
        dbg.cmd_bp_set("0x80000000")
        assert len(dbg._breakpoints) == 1
        bp = dbg._breakpoints[0]
        assert bp.kind == "addr"
        assert bp.value == 0x80000000

    def test_addr_bp_hit_in_step_one(self):
        dbg = _make_dbg()
        dbg.hart.pc = 0x1000  # 此处有 NOP (来自 _make_dbg)
        dbg.cmd_bp_set("0x1000")
        # 断点命中应阻止执行, 且设置 _paused
        dbg.step_one()
        assert dbg._paused
        # PC 不应推进 (断点在指令执行前命中)
        assert dbg.hart.pc == 0x1000

    def test_addr_bp_miss_in_step_one(self):
        dbg = _make_dbg()
        dbg.hart.pc = 0x1000
        dbg.cmd_bp_set("0x80000000")  # 不匹配
        dbg.step_one()
        assert not dbg._paused
        # PC 应已推进
        assert dbg.hart.pc != 0x1000

    # -- 指令断点 --

    @pytest.mark.parametrize(
        "name, funct12",
        [
            ("ecall", 0x000),
            ("ebreak", 0x001),
            ("mret", 0x302),
            ("sret", 0x102),
            ("wfi", 0x105),
        ],
    )
    def test_set_instr_bp_named(self, name, funct12):
        dbg = _make_dbg()
        dbg.cmd_bp_set(name)
        bp = dbg._breakpoints[0]
        assert bp.kind == "instr"
        assert bp.value == funct12

    def test_instr_bp_hit_on_ecall(self):
        dbg = _make_dbg()
        dbg.cmd_bp_set("ecall")
        # 在 PC 处写一条 ECALL (0x00000073)
        dbg._emu.bus.write(0x1000, b"\x73\x00\x00\x00")
        dbg.hart.pc = 0x1000
        dbg.step_one()
        assert dbg._paused, "ECALL 断点应命中"

    def test_instr_bp_miss_on_addi(self):
        """非 ECALL 指令不应命中 instr 断点."""
        dbg = _make_dbg()
        dbg.cmd_bp_set("ecall")
        # 0x1000 已有 NOP (addi x0, x0, 0 = 0x00000013)
        dbg.hart.pc = 0x1000
        dbg.step_one()
        assert not dbg._paused, "NOP 不应命中 ECALL 断点"

    # -- opcode 断点 --

    def test_set_opcode_bp(self):
        dbg = _make_dbg()
        dbg.cmd_bp_opcode("0x73")
        bp = dbg._breakpoints[0]
        assert bp.kind == "opcode"
        assert bp.value == 0x73

    def test_opcode_bp_hit(self):
        dbg = _make_dbg()
        dbg.cmd_bp_opcode("0x73")
        # 写一条 CSRRS (opcode=0x73, funct3=0b010)
        # csrrw x0, cycle, x0 = (0xC00 << 20) | (0b001 << 12) | 0x73
        instr = (0xC00 << 20) | (0b001 << 12) | 0x73
        dbg._emu.bus.write(0x1000, instr.to_bytes(4, "little"))
        dbg.hart.pc = 0x1000
        dbg.step_one()
        assert dbg._paused, "opcode 0x73 断点应命中"

    # -- 列表 / 删除 / 清除 --

    def test_bp_list_empty(self):
        dbg = _make_dbg()
        dbg.cmd_bp_list()
        # 不抛异常即通过

    def test_bp_list(self):
        dbg = _make_dbg()
        dbg.cmd_bp_set("0x80000000")
        dbg.cmd_bp_set("ecall")
        # 验证列表有内容
        assert len(dbg._breakpoints) == 2
        dbg.cmd_bp_list()  # 不抛异常

    def test_bp_delete(self):
        dbg = _make_dbg()
        dbg.cmd_bp_set("0x1000")
        dbg.cmd_bp_set("0x2000")
        assert len(dbg._breakpoints) == 2
        dbg.cmd_bp_delete("1")
        assert len(dbg._breakpoints) == 1
        assert dbg._breakpoints[0].value == 0x2000

    def test_bp_delete_out_of_range(self):
        dbg = _make_dbg()
        dbg.cmd_bp_set("0x1000")
        dbg.cmd_bp_delete("99")  # 不抛异常, 静默警告

    def test_bp_clear(self):
        dbg = _make_dbg()
        dbg.cmd_bp_set("0x1000")
        dbg.cmd_bp_set("0x2000")
        dbg.cmd_bp_clear()
        assert len(dbg._breakpoints) == 0

    # -- 无效参数 --

    @pytest.mark.parametrize(
        "arg",
        [
            "not_a_thing",  # 非数字非指令名
            "-1",  # 负地址
        ],
    )
    def test_bp_set_invalid(self, arg):
        dbg = _make_dbg()
        dbg.cmd_bp_set(arg)
        assert len(dbg._breakpoints) == 0

    @pytest.mark.parametrize(
        "arg",
        [
            "not_hex",  # @seize_val_err 捕获
            "-1",  # 负值超出 7-bit
            "0x80",  # 超出 7-bit 范围
            "0x55",  # 不是已知 RV64 opcode
        ],
    )
    def test_bp_opcode_invalid(self, arg):
        dbg = _make_dbg()
        dbg.cmd_bp_opcode(arg)
        assert len(dbg._breakpoints) == 0

    def test_bp_opcode_compressed_quadrant(self):
        """压缩象限 opcode (0x00/0x01/0x02) 无对应标准指令, 应被拒绝."""
        dbg = _make_dbg()
        dbg.cmd_bp_opcode("0x00")
        assert len(dbg._breakpoints) == 0  # 拒绝: 不在 Opc 中

    # -- restart 后断点持久化 --

    def test_bp_survives_restart(self):
        dbg = _make_dbg()
        dbg.cmd_bp_set("0x1000")
        assert len(dbg._breakpoints) == 1
        dbg.cmd_restart()
        assert len(dbg._breakpoints) == 1, "restart 后断点应保留"
        assert dbg._breakpoints[0].value == 0x1000

    def test_bp_runtime_state_cleared_on_restart(self):
        dbg = _make_dbg()
        dbg._paused = True
        dbg._hart_paused.add(0)
        dbg._bp_hit_this_run.add(("addr", 0x1000))
        dbg.cmd_restart()
        assert not dbg._paused
        assert len(dbg._hart_paused) == 0
        assert len(dbg._bp_hit_this_run) == 0

    def test_bp_c_skips_already_hit_addr(self):
        """同一 continue 内, 已命中过的地址断点不应重复停."""
        dbg = _make_dbg()
        dbg.cmd_bp_set("0x1000")
        dbg.hart.pc = 0x1000
        # 第一次命中
        assert dbg._check_breakpoints(dbg.hart, 0x1000, 0x00000013)
        assert dbg._paused
        assert ("addr", 0x1000) in dbg._bp_hit_this_run
        # 同一 continue 内再次检查 -> 应跳过
        dbg._paused = False
        hit_again = dbg._check_breakpoints(dbg.hart, 0x1000, 0x00000013)
        assert not hit_again, "同一 continue 内不应重复命中"
        assert not dbg._paused

    def test_bp_c_second_call_skips_same_addr(self):
        """第二遍 c 应跳过已命中地址, 执行指令推进 PC."""
        dbg = _make_dbg()
        dbg.cmd_bp_set("0x1000")
        dbg.hart.pc = 0x1000
        # 第一遍: 命中
        assert dbg._check_breakpoints(dbg.hart, 0x1000, 0x00000013)
        assert ("addr", 0x1000) in dbg._bp_hit_this_run
        # 模拟 c 后回到 REPL, 再 c 一次: 同一地址应被跳过
        dbg._paused = False
        hit = dbg._check_breakpoints(dbg.hart, 0x1000, 0x00000013)
        assert not hit, "第二次 c 应跳过同一地址的断点"
        # 推进 PC 后命中记录应失效 (不同地址不受影响)
        dbg.hart.pc = 0x1004
        assert ("addr", 0x1004) not in dbg._bp_hit_this_run

    # -- 条件断点 --

    def test_cond_bp_if_csr_dispatch(self):
        """bp if csr mtvec == 0x4a8 应创建 kind=cond 断点."""
        dbg = _make_dbg()
        dbg._dispatch(["bp", "if", "csr", "mtvec", "==", "0x4a8"])
        assert len(dbg._breakpoints) == 1
        bp = dbg._breakpoints[0]
        assert bp.kind == "cond"
        assert bp.cond_type == "csr"
        assert bp.cond_reg == "mtvec"
        assert bp.cond_op == "=="
        assert bp.cond_val == 0x4A8

    def test_cond_bp_if_reg_dispatch(self):
        """bp if reg x0 == 0 应创建 kind=cond 断点."""
        dbg = _make_dbg()
        dbg._dispatch(["bp", "if", "reg", "x0", "==", "0"])
        assert len(dbg._breakpoints) == 1
        bp = dbg._breakpoints[0]
        assert bp.kind == "cond"
        assert bp.cond_type == "reg"
        assert bp.cond_reg == "x0"

    def test_cond_bp_hit_on_match(self):
        """条件匹配时 cond 断点应命中 (指令写入目标 CSR)."""
        dbg = _make_dbg()
        # mtvec 初始值为 0, 条件 mtvec==0 -> CSR 写指令触发检查
        dbg._dispatch(["bp", "if", "csr", "mtvec", "==", "0"])
        dbg.hart.pc = 0x1000
        # csrrw x0, mtvec, x1 — 写 x1=0 到 mtvec (保持 0)
        csr_instr = (0x305 << 20) | (1 << 15) | (1 << 12) | (0 << 7) | 0x73
        dbg._emu.bus.write(0x1000, csr_instr.to_bytes(4, "little"))
        hit = dbg._check_breakpoints(dbg.hart, 0x1000, csr_instr)
        assert hit
        assert dbg._paused

    def test_cond_bp_miss_on_mismatch(self):
        """条件不匹配时 cond 断点不应命中."""
        dbg = _make_dbg()
        dbg._dispatch(["bp", "if", "csr", "mtvec", "==", "0x4a8"])
        dbg.hart.pc = 0x1000
        # csrrw x0, mtvec, x1 — 写 x1=0 到 mtvec, 但 0 != 0x4a8
        csr_instr = (0x305 << 20) | (1 << 15) | (1 << 12) | (0 << 7) | 0x73
        dbg._emu.bus.write(0x1000, csr_instr.to_bytes(4, "little"))
        hit = dbg._check_breakpoints(dbg.hart, 0x1000, csr_instr)
        assert not hit
        assert not dbg._paused

    def test_addr_bp_with_cond_match(self):
        """地址断点+条件: 地址和条件都匹配才命中."""
        dbg = _make_dbg()
        # bp 0x1000 if reg x0 == 0
        dbg._dispatch(["bp", "0x1000", "if", "reg", "x0", "==", "0"])
        dbg.hart.pc = 0x1000
        hit = dbg._check_breakpoints(dbg.hart, 0x1000, 0x00000013)
        assert hit

    def test_addr_bp_with_cond_mismatch(self):
        """地址匹配但条件不匹配 -> 不命中."""
        dbg = _make_dbg()
        dbg._dispatch(["bp", "0x1000", "if", "reg", "x0", "==", "1"])
        dbg.hart.pc = 0x1000
        # x0 永远是 0, 不等于 1 -> 不命中
        hit = dbg._check_breakpoints(dbg.hart, 0x1000, 0x00000013)
        assert not hit

    @pytest.mark.parametrize(
        "op,val,expect_hit",
        [
            ("==", 0, True),
            ("!=", 1, True),
            ("!=", 0, False),
            (">", -1, True),  # 0 > -1 (signed)
            ("<", 1, True),  # 0 < 1
        ],
    )
    def test_cond_bp_operators(self, op, val, expect_hit):
        """条件断点运算符: == != < > — 用 a0 (x10) 验证."""
        dbg = _make_dbg()
        # a0 初始值为 0
        dbg._dispatch(["bp", "if", "reg", "a0", op, str(val)])
        dbg.hart.pc = 0x1000
        # ADDI a0, x0, 0 — 写 a0=0, rd=a0 (x10), 预过滤器放行
        addi_instr = (0 << 20) | (0 << 15) | (0 << 12) | (10 << 7) | 0x13
        dbg._emu.bus.write(0x1000, addi_instr.to_bytes(4, "little"))
        hit = dbg._check_breakpoints(dbg.hart, 0x1000, addi_instr)
        assert hit == expect_hit

    def test_cond_bp_gpr_by_alias(self):
        """条件断点应按 GPR 别名查找 (例: t0 -> x5)."""
        dbg = _make_dbg()
        # 写 t0 (x5) = 0x42
        dbg.hart.write_gpr(5, 0x42)
        dbg._dispatch(["bp", "if", "reg", "t0", "==", "0x42"])
        dbg.hart.pc = 0x1000
        # ADDI t0, x0, 0 — 写 t0=0 (与 0x42 不匹配, 但指令 rd=t0 会触发检查)
        addi_instr = (0 << 20) | (0 << 15) | (0 << 12) | (5 << 7) | 0x13
        dbg._emu.bus.write(0x1000, addi_instr.to_bytes(4, "little"))
        hit = dbg._check_breakpoints(dbg.hart, 0x1000, addi_instr)
        assert hit

    def test_cond_bp_respects_bp_mode_async(self):
        """async 模式: 条件断点命中时设置 _hart_paused 而非 _paused."""
        dbg = _make_dbg(num_harts=2)
        dbg._bp_mode = "async"
        dbg._dispatch(["bp", "if", "csr", "mtvec", "==", "0"])
        dbg.hart.pc = 0x1000
        # csrrw x0, mtvec, x1 — 写 mtvec, CSR 指令触发条件检查
        csr_instr = (0x305 << 20) | (1 << 15) | (1 << 12) | (0 << 7) | 0x73
        dbg._emu.bus.write(0x1000, csr_instr.to_bytes(4, "little"))
        hit = dbg._check_breakpoints(dbg.hart, 0x1000, csr_instr)
        assert hit
        assert dbg.hart.id in dbg._hart_paused
        assert not dbg._paused  # async 模式不设 _paused

    def test_cond_bp_invalid_syntax(self):
        """bp if 缺参数时警告, 不崩溃."""
        dbg = _make_dbg()
        assert dbg._dispatch(["bp", "if"]) is True
        assert dbg._dispatch(["bp", "if", "csr"]) is True
        assert len(dbg._breakpoints) == 0

    # -- 分发 --

    def test_dispatch_bp_addr(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["bp", "0x80000000"]) is True
        assert len(dbg._breakpoints) == 1

    def test_dispatch_bp_list(self):
        dbg = _make_dbg()
        dbg._dispatch(["bp", "0x1000"])
        assert dbg._dispatch(["bp", "list"]) is True

    def test_dispatch_bp_clear(self):
        dbg = _make_dbg()
        dbg.cmd_bp_set("0x1000")
        assert dbg._dispatch(["bp", "clear"]) is True
        assert len(dbg._breakpoints) == 0

    # -- 内核符号断点 (vmlinux) --

    def test_kernel_sym_bp_lookup(self):
        """_try_set_symbol_bp 在固件符号中未命中时应查 vmlinux 符号."""
        dbg = _make_dbg()
        dbg._image = _make_image()
        dbg._load_offset = 0x80000000
        # 模拟加载了 Linux vmlinux 外部符号
        dbg._sym_symbols = {"handle_page_fault": 0xFFFFFFFF8001393C}
        dbg._sym_load_offset = 0x200000 - 0xFFFFFFFF80000000

        result = dbg._try_set_symbol_bp("handle_page_fault")
        assert result is True
        assert len(dbg._breakpoints) == 1
        bp = dbg._breakpoints[0]
        assert bp.kind == "addr"
        # 运行时地址应为 Bare 模式下的 PA (经 load_offset 映射)
        assert bp.value != 0, "断点地址不应为 0"
        assert "handle_page_fault" in bp.desc

    def test_kernel_sym_bp_not_found(self):
        """内核符号表中无匹配符号时返回 False (供 cmd_bp_set 继续尝试地址解析)."""
        dbg = _make_dbg()
        dbg._image = _make_image()
        dbg._sym_symbols = {"some_func": 0xFFFFFFFF80001000}
        dbg._sym_load_offset = 0

        result = dbg._try_set_symbol_bp("nonexistent")
        assert result is False
        assert len(dbg._breakpoints) == 0

    def test_firmware_sym_bp_takes_priority(self):
        """固件符号与内核符号同名时, 固件符号优先 (先查 _image.symbols)."""
        dbg = _make_dbg()
        dbg._image = _make_image()  # 含 _start @ 0x1000
        dbg._load_offset = 0
        dbg._sym_symbols = {"_start": 0xFFFFFFFF80001000}  # 同名内核符号
        dbg._sym_load_offset = 0x200000

        result = dbg._try_set_symbol_bp("_start")
        assert result is True
        assert len(dbg._breakpoints) == 1
        # 应命中固件符号 (0x1000 + 0), 而非内核符号
        bp = dbg._breakpoints[0]
        assert bp.value == 0x1000, f"应使用固件符号地址 0x1000, 实际 {bp.value:#x}"


class TestDispatch:
    """REPL 命令分发."""

    @pytest.mark.parametrize(
        "v, expected",
        [
            ("q", False),
            ("quit", False),
            ("exit", False),
            ("e", True),  # 未知命令, 返回 True 保持 REPL 运行
        ],
    )
    def test_quit_returns_false(self, v, expected: bool):
        dbg = _make_dbg()
        assert dbg._dispatch([v]) is expected

    def test_step_dispatches(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["s"]) is True
        assert dbg._instr_count > 0

    def test_step_with_count(self):
        dbg = _make_dbg()
        assert dbg._dispatch(["step", "3"]) is True
        assert dbg._instr_count == 3

    def test_continue_dispatches(self):
        # ram_base=0 使 0x1000 落在 RAM 内 (native 与纯 Python 取指路径一致);
        # 在 NOP 之后的 0x1004 设地址断点, continue 执行完 NOP 后命中断点
        # 即刻暂停 — 无连续 trap 兜底, 断点是连续执行的自然停止条件.
        emu = _make_emu(ram_base=0, ram_size=0x10000, prog_cnt=0x1000)
        emu.load_code(0x1000, b"\x13\x00\x00\x00")
        dbg = Debugger(emulator=emu, hart_id=0)
        dbg.cmd_bp_set("0x1004")
        assert dbg._dispatch(["c"]) is True

    @pytest.mark.parametrize(
        "parts",
        [
            ["r", "2"],
            ["s", "-1"],  # 负计数被拒绝, 不崩溃
            ["r", "-1"],  # 同上
            ["undo"],
            ["regs"],
            ["gpr"],
            ["reg", "a0"],
            ["reg"],  # 缺参数, 警告
            ["set", "a0", "0x42"],
            ["set", "a0"],  # 缺参数, 警告
            ["csr", "mstatus"],
            ["csrw", "mstatus", "0x100"],
            ["pc"],
            ["mode"],
            ["mstatus"],
            ["tlb"],
            ["tlbflush"],
            ["cache"],
            ["satp"],
            ["mem", "0x1000"],
            ["mem"],  # 缺参数, 警告
            ["disasm", "0x1000"],
            ["status"],
            ["info"],
            ["stack"],
            ["bt"],
            ["frame"],
        ],
    )
    def test_dispatch_returns_true(self, parts):
        dbg = _make_dbg()
        assert dbg._dispatch(parts) is True

    def test_sym_dispatches(self):
        dbg = _make_dbg()
        dbg._image = _make_image()
        assert dbg._dispatch(["sym"]) is True

    def test_sym_no_args_shows_grouped_counts(self):
        """sym (无参数) 按首字符分组显示各前缀的符号数量."""
        dbg = _make_dbg()
        dbg._image = _make_image()
        # _make_image 提供: _start, main, uart_puts
        assert dbg._dispatch(["sym"]) is True

    def test_sym_prefix_lists_matching_symbols(self):
        """sym <prefix> 列出以前缀开头的符号, 按地址升序."""
        dbg = _make_dbg()
        dbg._image = _make_image()
        assert dbg._dispatch(["sym", "m"]) is True  # 匹配 main

    def test_sym_prefix_underscore(self):
        """sym _ 匹配以下划线开头的符号."""
        dbg = _make_dbg()
        dbg._image = _make_image()
        assert dbg._dispatch(["sym", "_"]) is True  # 匹配 _start

    def test_sym_prefix_no_match(self):
        """sym <prefix> 无匹配时不报错."""
        dbg = _make_dbg()
        dbg._image = _make_image()
        assert dbg._dispatch(["sym", "zzz"]) is True  # 无匹配

    def test_sym_empty_symbols(self):
        """空符号表不应崩溃."""
        dbg = _make_dbg()
        dbg._image = FirmwareImage(
            format="elf", entry_point=0x1000, segments=[], symbols={}
        )
        assert dbg._dispatch(["sym"]) is True

    def test_sym_no_image(self):
        """无 ELF 镜像时显示警告, 不崩溃."""
        dbg = _make_dbg()
        assert dbg._image is None
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
        assert dbg.hart.gprs[10] == 99


# ============================================================
#  栈帧回溯
# ============================================================


class TestStackWalk:
    """栈帧回溯."""

    def test_single_frame_when_fp_zero(self):
        dbg = _make_dbg()
        h = dbg.hart
        h.gprs[8] = 0  # fp = 0
        frames = dbg._walk_frame_chain()
        assert len(frames) == 1
        assert frames[0].idx == 0
        assert frames[0].pc == h.pc

    def test_frame_chain_with_valid_fp(self):
        """构造 3 层栈帧链: fp-8=ra, fp-16=saved_fp.

        _walk_frame_chain 在 saved_ra==0 时提前 break (不追加空帧),
        因此要得到 N 帧需要 N-1 层非零 RA 链接.

        FP 链终止后, 若最后一帧 RA 非零, RA 推断逻辑会追加一个调用者帧.
        此处帧 #2 的 RA=0x80000300 -> 追加帧 #3 (pc=0x800002FC).
        """
        dbg = _make_dbg()
        h = dbg.hart

        # 帧 #0 (当前): sp=0x80001000, fp=0x80001080
        h.gprs[2] = 0x80001000  # sp
        h.gprs[8] = 0x80001080  # fp

        bus = dbg._emu.bus
        # 帧 #1 链接 (fp=0x80001080): RA->call_site_A, saved_fp->F1
        bus.write(0x80001080 - 16, (0x80001100).to_bytes(8, "little"))
        bus.write(0x80001080 - 8, (0x80000200).to_bytes(8, "little"))

        # 帧 #2 链接 (fp=0x80001100): RA->call_site_B, saved_fp->F2
        bus.write(0x80001100 - 16, (0x80001180).to_bytes(8, "little"))
        bus.write(0x80001100 - 8, (0x80000300).to_bytes(8, "little"))

        # 帧 #3 链接 (fp=0x80001180): terminal — RA=0 导致 walk 截断
        bus.write(0x80001180 - 16, (0).to_bytes(8, "little"))
        bus.write(0x80001180 - 8, (0).to_bytes(8, "little"))

        frames = dbg._walk_frame_chain()
        assert len(frames) == 4  # FP 链 3 帧 + RA 推断 1 帧
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
        # 帧 #3 (RA 推断): 由帧 #2 的 RA=0x80000300 推导
        assert frames[3].idx == 3
        assert frames[3].fp == 0
        assert frames[3].pc == 0x800002FC  # frames[2].ra - 4
        assert frames[3].ra == 0
        assert "FP 链终止" in frames[3].note

    def test_stack_walk_detects_cycle(self):
        """FP 链成环 -> 截断回溯."""
        dbg = _make_dbg()
        h = dbg.hart
        h.gprs[8] = 0x80001000  # fp
        bus = dbg._emu.bus
        # saved_fp 指向自身 -> 环
        bus.write(0x80001000 - 16, (0x80001000).to_bytes(8, "little"))
        bus.write(0x80001000 - 8, (0x80000400).to_bytes(8, "little"))

        frames = dbg._walk_frame_chain()
        # 应截断: 发现环后停止
        assert len(frames) <= 2, f"应截断环, 但得到 {len(frames)} 帧"

    def test_cmd_frame_shows_backtrace(self):
        dbg = _make_dbg()
        dbg.hart.gprs[8] = 0
        dbg.cmd_frame()  # 单帧

    def test_cmd_frame_with_valid_fp(self):
        dbg = _make_dbg()
        h = dbg.hart
        h.gprs[2] = 0x80001000
        h.gprs[8] = 0x80001080
        bus = dbg._emu.bus
        bus.write(0x80001080 - 16, (0x80001100).to_bytes(8, "little"))
        bus.write(0x80001080 - 8, (0x80000400).to_bytes(8, "little"))
        bus.write(0x80001100 - 16, (0).to_bytes(8, "little"))
        bus.write(0x80001100 - 8, (0).to_bytes(8, "little"))
        dbg.cmd_frame()

    def test_cmd_frame_switch(self):
        dbg = _make_dbg()
        h = dbg.hart
        h.gprs[8] = 0x80001080
        bus = dbg._emu.bus
        bus.write(0x80001080 - 16, (0x80001100).to_bytes(8, "little"))
        bus.write(0x80001080 - 8, (0x80000400).to_bytes(8, "little"))
        bus.write(0x80001100 - 16, (0).to_bytes(8, "little"))
        bus.write(0x80001100 - 8, (0).to_bytes(8, "little"))
        dbg.cmd_frame("1")  # 切换到帧 1
        assert dbg._current_frame_idx == 1

    def test_cmd_frame_out_of_range(self):
        dbg = _make_dbg()
        dbg.hart.gprs[8] = 0
        dbg.cmd_frame("99")  # 超出范围

    @pytest.mark.skip(
        reason=(
            "Bare 翻译下 SPP=U + RAM sepc 无法与合法 S→U trap 现场区分: "
            "同寄存器状态在 test_smode_to_umode_boundary / "
            "test_fallback_trap_offsets_for_s_to_u 中被断言为'应生成边界帧'。"
            "二者矛盾, 代码选择保留边界帧 (Bare 下 U-mode PA 即 RAM 地址属合法 "
            "trap); Sv39 残留 sepc 跳过由 test_stale_sepc_sv39_ram_addr_skipped 覆盖。"
        )
    )
    def test_stale_sepc_from_ms_transition_skipped(self):
        """M->S mret 残留 sepc (裸 PA) 不生成虚假边界帧 — 已跳过.

        原意图: M->S 启动后 SPP=0, sepc 仍保留 M 模式设置的内核入口 PA
        (如 0x80201048), 这不是真实 trap 现场, 不应显示为边界帧。

        但在 Bare 翻译下该状态与合法 S→U trap 现场完全同构 (SPP=U + RAM
        sepc), 无法仅凭寄存器状态区分, 故与上述 S→U 边界帧测试直接矛盾。
        """
        dbg = _make_dbg(ram_size=0x800000)
        h = dbg.hart
        # S 模式, 无有效 FP 链 (fp=0 使 FP 回溯立即终止)
        h.mode = RiscvMode.S
        h.gprs[8] = 0  # fp = 0
        h.gprs[2] = 0x80001000  # sp
        # 设置 mstatus: SPP=U (0)
        h.csrs["mstatus"].val = (h.csrs["mstatus"].val & ~(1 << 8))  # SPP=0
        # 模拟 M->S mret 后的 sepc: 内核入口 PA
        h.csrs["sepc"].val = 0x80201048
        # stvec 需指向合法代码 (用于 _parse_trap_save_offsets 反汇编)
        # 写一条 ret (jalr x0, x1, 0) 到 stvec 位置
        stvec_addr = 0x80008000
        h.csrs["stvec"].val = stvec_addr
        dbg._emu.bus.write(stvec_addr, (0x00008067).to_bytes(4, "little"))

        frames = dbg._walk_frame_chain()
        # 应只有帧 #0 (当前执行点), 不应有虚假的边界帧
        assert len(frames) == 1, (
            f"M->S 残留 sepc 不应生成边界帧, 但得到 {len(frames)} 帧"
        )
        assert frames[0].idx == 0

    def test_stale_sepc_sv39_ram_addr_skipped(self):
        """Sv39 启用后 SPP=S 时 sepc 为裸 RAM 地址 -> 残留 sepc, 不生成边界帧.

        Kernel 在 S 模式运行 (SPP=S), Sv39 启用, 但 sepc 指向 0x80201048
        (物理 RAM 地址).  这不是真实的 S->S trap 现场 — 内核代码运行在高
        VA (0xffffffc6XXXXXXXX), sepc 中的低地址是 M->S 过渡后的残留值.
        即使 SPP=S 也应跳过.
        """
        dbg = _make_dbg(ram_size=0x800000)
        h = dbg.hart
        # S 模式, Sv39 启用, 无有效 FP 链
        h.mode = RiscvMode.S
        h.gprs[8] = 0  # fp = 0
        h.gprs[2] = 0x80001000  # sp
        # 设置 mstatus: SPP=S (bit 8), MPP=S
        h.csrs["mstatus"].val = (
            h.csrs["mstatus"].val | (1 << 8)
        )  # SPP=1
        # 启用 Sv39 (satp MODE=8) — 必须经 satp_val setter 更新 _mmu_mode
        root_ppn = 0x80000
        h.satp_val = (8 << 60) | root_ppn
        # 模拟残留 sepc: 内核入口 PA (非 Sv39 规范高 VA)
        h.csrs["sepc"].val = 0x80201048
        # stvec 指向合法代码
        stvec_addr = 0x80008000
        h.csrs["stvec"].val = stvec_addr
        dbg._emu.bus.write(stvec_addr, (0x00008067).to_bytes(4, "little"))

        frames = dbg._walk_frame_chain()
        # Sv39 启用时 sepc 为裸 RAM 地址 -> 残留, 不生成边界帧
        assert len(frames) == 1, (
            f"Sv39 + RAM sepc 不应生成边界帧 (即使是 SPP=S), 但得到 {len(frames)} 帧"
        )
        assert frames[0].idx == 0

    def test_ra_corruption_live_x1_falls_back_to_saved_ra(self):
        """live x1 被函数覆盖 (如 blake2s) 时降级使用栈上保存的 RA.

        blake2s_compress_generic 的 G 宏将 x1 用作临时寄存器,
        导致 live x1 变为非合法代码地址 (0xed462571a7333171).
        但栈上 fp-8 的 saved_ra 仍然有效, 回溯应降级使用.
        """
        dbg = _make_dbg()
        h = dbg.hart
        bus = dbg._emu.bus

        # 帧 #0 (当前): 模拟 blake2s_compress_generic
        h.pc = 0x80001000
        h.gprs[1] = 0xDEADBEEF  # live x1 — 被哈希函数覆盖
        h.gprs[2] = 0x8000F000  # sp
        h.gprs[8] = 0x8000F080  # fp

        # 帧 #1 链接 (fp=0x8000F080):
        #   fp-16: saved_fp -> 0x8000F100 (caller's fp)
        #   fp-8:  saved_ra -> 0x80002000 (valid return address)
        bus.write(0x8000F080 - 16, (0x8000F100).to_bytes(8, "little"))
        bus.write(0x8000F080 - 8, (0x80002000).to_bytes(8, "little"))

        # 帧 #2 链接 (fp=0x8000F100): terminal
        bus.write(0x8000F100 - 16, (0).to_bytes(8, "little"))
        bus.write(0x8000F100 - 8, (0).to_bytes(8, "little"))

        frames = dbg._walk_frame_chain()

        assert len(frames) >= 2, f"应至少有帧 #0 和帧 #1, 但只有 {len(frames)} 帧"
        # 帧 #1 (caller): pc 应使用 saved_ra (0x80002000) 而非 live x1 (0xDEADBEEF)
        frame1 = frames[1]
        assert frame1.pc == 0x80001FFC, (
            f"帧 #1 pc 应 = saved_ra - 4 = 0x80001FFC, 但得到 0x{frame1.pc:x}"
        )
        # 帧 #1 ra 应为栈上保存的 0x80002000
        assert frame1.ra == 0x80002000, (
            f"帧 #1 ra 应为 saved_ra = 0x80002000, 但得到 0x{frame1.ra:x}"
        )
        # 帧 #0 应无 note (当前帧不检查 RA 合法性)
        assert not frames[0].note

    def test_ra_corruption_both_live_and_saved_invalid(self):
        """live x1 和 saved_ra 均为非法地址时, 帧带损坏标记.

        极端情况: 函数不仅覆盖了 x1, 栈帧 fp-8 也被覆写.
        """
        dbg = _make_dbg()
        h = dbg.hart
        bus = dbg._emu.bus

        h.pc = 0x80001000
        h.gprs[1] = 0xDEADBEEF  # live x1 — 非法
        h.gprs[2] = 0x8000F000
        h.gprs[8] = 0x8000F080

        # saved_ra 也是非法地址 (栈被哈希中间数据覆写)
        bus.write(0x8000F080 - 16, (0x8000F100).to_bytes(8, "little"))
        bus.write(0x8000F080 - 8, (0xCAFEBABE).to_bytes(8, "little"))  # 非法

        # 终端帧
        bus.write(0x8000F100 - 16, (0).to_bytes(8, "little"))
        bus.write(0x8000F100 - 8, (0).to_bytes(8, "little"))

        frames = dbg._walk_frame_chain()
        assert len(frames) >= 2
        # 帧 #1 应带有损坏标记
        frame1 = frames[1]
        assert frame1.note, "帧 #1 在 RA 完全不可恢复时应带有损坏标记"


# ============================================================
#  U-mode 栈回溯 (backtrace)
# ============================================================


class TestUmodeBacktrace:
    """U-mode 直接回溯与 S→U 边界帧恢复."""

    @staticmethod
    def _make_dbg(ram_size: int = 0x10000, ram_base: int = 0x80000000) -> Debugger:
        emu = _make_emu(ram_size=ram_size, ram_base=ram_base)
        emu.load_code(0x1000, b"\x13\x00\x00\x00")
        return Debugger(emulator=emu, hart_id=0)

    # ---------- 直接 U-mode FP 链 ----------

    def test_umode_fp_chain_walk(self):
        """hart 处于 U-mode 时, FP 链帧全部标记为 U.

        使用 Bare 翻译 (mmu_mode=0), VA=PA.
        """
        dbg = self._make_dbg(ram_size=0x20000, ram_base=0x80000000)
        h = dbg.hart
        bus = dbg._emu.bus

        h.mode = RiscvMode.U
        h._mmu_mode = 0  # Bare 翻译
        h.pc = 0x80001000
        h.gprs[1] = 0x80002000  # ra (frame #1 uses live x1)
        h.gprs[2] = 0x8000F000  # sp
        h.gprs[8] = 0x8000F080  # fp

        # 帧 #1 链接: saved_ra=0x80004000, saved_fp=0x8000F100
        bus.write(0x8000F080 - 16, (0x8000F100).to_bytes(8, "little"))
        bus.write(0x8000F080 - 8, (0x80004000).to_bytes(8, "little"))
        # 帧 #2 链接: terminal
        bus.write(0x8000F100 - 16, (0).to_bytes(8, "little"))
        bus.write(0x8000F100 - 8, (0).to_bytes(8, "little"))

        frames = dbg._walk_frame_chain()
        assert len(frames) >= 3, f"应有 ≥3 帧 (含 RA 推断), 但只有 {len(frames)}"
        for i, f in enumerate(frames):
            assert f.mode == "U", f"帧 #{i} mode={f.mode!r}, 预期 U"
        assert frames[0].pc == 0x80001000
        assert frames[1].pc == 0x80001FFC  # ra - 4
        assert frames[1].ra == 0x80004000

    def test_umode_single_frame_when_fp_zero(self):
        """U-mode 且 fp=0 时仅有当前帧."""
        dbg = self._make_dbg()
        h = dbg.hart
        h.mode = RiscvMode.U
        h._mmu_mode = 0
        h.pc = 0x80001000
        h.gprs[8] = 0  # fp = 0

        frames = dbg._walk_frame_chain()
        assert len(frames) == 1
        assert frames[0].mode == "U"

    def test_umode_ra_inference_when_fp_chain_broken(self):
        """FP 链不可达时 (fp=0), 由 live RA 推断一个调用者帧."""
        dbg = self._make_dbg()
        h = dbg.hart
        h.mode = RiscvMode.U
        h._mmu_mode = 0
        h.pc = 0x80001000
        h.gprs[1] = 0x80002000  # ra — 有效返回地址
        h.gprs[8] = 0  # fp = 0 ->FP 链不可达

        frames = dbg._walk_frame_chain()
        assert len(frames) == 2, (
            f"RA 推断应产生 1 个调用者帧 (共 2 帧), 但只有 {len(frames)}"
        )
        assert frames[0].mode == "U"
        assert frames[1].mode == "U"
        assert frames[1].pc == 0x80001FFC  # ra - 4
        assert "RA 推断" in frames[1].note
        assert frames[1].fp == 0  # 推断帧无有效 FP

    # ---------- S→U 边界帧恢复 ----------

    def test_smode_to_umode_boundary(self):
        """S-mode 下 SPP=U: 从 sepc 恢复 U-mode FP 链."""
        dbg = self._make_dbg(ram_size=0x20000, ram_base=0x80000000)
        h = dbg.hart
        bus = dbg._emu.bus

        # S-mode 当前状态
        h.mode = RiscvMode.S
        h._mmu_mode = 0
        h.pc = 0x80010000  # S-mode kernel code
        h.gprs[2] = 0x8001F000  # S-mode sp
        h.gprs[8] = 0x8001F080  # S-mode fp

        # S-mode FP 链 (内核)
        bus.write(0x8001F080 - 16, (0x8001F100).to_bytes(8, "little"))
        bus.write(0x8001F080 - 8, (0x80010200).to_bytes(8, "little"))
        bus.write(0x8001F100 - 16, (0).to_bytes(8, "little"))
        bus.write(0x8001F100 - 8, (0).to_bytes(8, "little"))

        # Trap CSRs: S-mode 捕获 U-mode 异常
        h.csrs["sepc"].val = 0x80003AE8  # U-mode PC (fault site, valid PA)
        h.csrs["scause"].val = 0xD  # LoadPageFault
        # mstatus.SPP = U
        h.csrs["mstatus"].val = h.csrs["mstatus"].val & ~(1 << 8)

        # stvec 指向含 RA/FP 保存的 trap 入口
        # addi sp,sp,-240; sd x1,0(sp); sd x8,8(sp)
        tvec_addr = 0x80018000
        h.csrs["stvec"].val = tvec_addr
        trap_entry = bytes([
            0x13, 0x01, 0x01, 0xF1,  # addi sp, sp, -240
            0x23, 0x30, 0x11, 0x00,  # sd   x1, 0(sp)
            0x23, 0x34, 0x81, 0x00,  # sd   x8, 8(sp)
        ])
        bus.write(tvec_addr, trap_entry)

        # S-mode 内核栈上的 trap 帧 — 由栈扫描找到
        trap_sp = 0x8001E000
        bus.write(trap_sp, (0x80003AE8).to_bytes(8, "little"))  # trapped PC
        bus.write(trap_sp + 0, (0).to_bytes(8, "little"))        # zero verify (sp+0)
        bus.write(trap_sp + 8, (0x80002000).to_bytes(8, "little"))   # saved_ra @ ra_off=8
        bus.write(trap_sp + 16, (0x8000F080).to_bytes(8, "little"))  # saved_fp @ fp_off=16

        # U-mode FP 链 (位于 saved_fp=0x8000F080)
        bus.write(0x8000F080 - 16, (0x8000F100).to_bytes(8, "little"))
        bus.write(0x8000F080 - 8, (0x80003000).to_bytes(8, "little"))
        bus.write(0x8000F100 - 16, (0).to_bytes(8, "little"))
        bus.write(0x8000F100 - 8, (0).to_bytes(8, "little"))

        frames = dbg._walk_frame_chain()

        umode_frames = [f for f in frames if f.mode == "U"]
        assert len(umode_frames) >= 1, (
            f"应有 ≥1 个 U-mode 帧, 实际帧列表: "
            f"{[(f.idx, f.mode, hex(f.pc)) for f in frames]}"
        )

    # ---------- Fallback trap offsets ----------

    def test_fallback_trap_offsets_for_s_to_u(self):
        """S→U 边界且 stvec 无 sd x1/sd x8 时, 回退到常见布局扫描."""
        dbg = self._make_dbg(ram_size=0x30000, ram_base=0x80000000)
        h = dbg.hart
        bus = dbg._emu.bus

        # S-mode 当前状态 (fp=0 ->FP 链立即终止)
        h.mode = RiscvMode.S
        h._mmu_mode = 0
        h.pc = 0x80010000
        h.gprs[8] = 0  # fp=0
        h.gprs[2] = 0x8001F000

        # Trap CSRs: SPP=U
        h.csrs["sepc"].val = 0x80001000  # U-mode PC
        h.csrs["mstatus"].val = h.csrs["mstatus"].val & ~(1 << 8)

        # stvec: ret 指令 — _parse_trap_save_offsets 返回 None
        tvec_addr = 0x80018000
        h.csrs["stvec"].val = tvec_addr
        bus.write(tvec_addr, (0x00008067).to_bytes(4, "little"))  # ret

        # 在 S-mode 栈上按紧凑布局 (ra_off=0, fp_off=8) 放 trap 帧
        trap_sp = 0x8001E000
        bus.write(trap_sp, (0x80001000).to_bytes(8, "little"))  # trapped PC
        bus.write(trap_sp + 0, (0).to_bytes(8, "little"))        # zero verify
        # saved_ra @ ra_off=0 — 但 zero verify 在同一个位置, 所以 ra_off=0 布局无法通过
        # 用 ra_off=8, fp_off=64 (Linux pt_regs) 布局
        # 在 trap_sp 上方找位置放 saved_ra
        bus.write(trap_sp + 8, (0x80002000).to_bytes(8, "little"))   # saved_ra @ ra_off=8
        bus.write(trap_sp + 64, (0x8000F080).to_bytes(8, "little"))  # saved_fp @ fp_off=64

        # U-mode FP 链
        bus.write(0x8000F080 - 16, (0x8000F100).to_bytes(8, "little"))
        bus.write(0x8000F080 - 8, (0x80003000).to_bytes(8, "little"))
        bus.write(0x8000F100 - 16, (0).to_bytes(8, "little"))
        bus.write(0x8000F100 - 8, (0).to_bytes(8, "little"))

        frames = dbg._walk_frame_chain()
        assert len(frames) >= 2, (
            f"fallback 应至少生成 1 个 U-mode 边界帧, 但只有 {len(frames)} 帧"
        )
        modes = [f.mode for f in frames]
        assert "U" in modes, f"fallback 后应有 U-mode 帧, 模式列表: {modes}"

    def test_fallback_all_fail_gives_pc_only_frame(self):
        """所有 fallback 布局均失败时, 回退到仅 PC 帧且不崩溃."""
        dbg = self._make_dbg(ram_size=0x20000, ram_base=0x80000000)
        h = dbg.hart
        bus = dbg._emu.bus

        h.mode = RiscvMode.S
        h._mmu_mode = 0
        h.pc = 0x80010000
        h.gprs[8] = 0  # fp=0

        h.csrs["sepc"].val = 0x80001000  # U-mode PC
        h.csrs["mstatus"].val = h.csrs["mstatus"].val & ~(1 << 8)  # SPP=U

        # stvec: ret — _parse_trap_save_offsets 返回 None
        tvec_addr = 0x80018000
        h.csrs["stvec"].val = tvec_addr
        bus.write(tvec_addr, (0x00008067).to_bytes(4, "little"))

        # 不写任何 trap 帧数据 ->所有 fallback 布局在栈扫描中找不到 trapped PC
        # _add_prev_mode_frame_fallback 捕获此情况并回退到仅 PC 帧

        frames = dbg._walk_frame_chain()
        assert len(frames) >= 2, (
            f"fallback 应回退到仅 PC 帧 (≥2 帧), 但只有 {len(frames)}: "
            f"{[(f.idx, f.mode, f.note) for f in frames]}"
        )
        last = frames[-1]
        assert "无法解析" in last.note, (
            f"fallback 失败时应带 '无法解析' 备注, 实际: {last.note!r}"
        )


# ============================================================
#  TLB 页大小解码
# ============================================================


class TestTlbDecodeHelpers:
    """TLB 静态辅助函数."""

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
        data = b"\x48\x65\x6c\x6c\x6f"
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
        l2 = dbg._emu.bus._l2
        assert l2 is not None
        # 取一个有效条目
        e = l2.entries[0]
        e.valid = True
        e.tag = 0x12345
        result = dbg._fmt_cache_line(e, 0, 0, full=False)
        assert "tag=0x" in result

    def test_full_dump(self):
        dbg = _make_dbg()
        l2 = dbg._emu.bus._l2
        assert l2 is not None
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
        """未映射物理地址 -> Bus 层面返回全零 (模拟未映射区域).

        PMA 校验在上层 mem_read/mem_write 中完成;
        Bus.read() 是底层物理总线, 对空洞地址返回零.
        """
        dbg = _make_dbg()
        result = dbg._fetch_and_disasm(0xFFFF0000)
        assert result is not None
        raw_hex, asm = result
        assert raw_hex == "0000"


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
        emu.harts[0].gprs[10] = 0xAAAA
        emu.harts[1].gprs[10] = 0xBBBB

        dbg = Debugger(emulator=emu, hart_id=0)
        assert dbg.hart.gprs[10] == 0xAAAA

        dbg.cmd_hart(1)
        assert dbg.hart.gprs[10] == 0xBBBB

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
        assert dbg.hart.gprs[10] == 0xFFFFFFFFFFFFFFFF

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

        _fetch_and_disasm 总是读取 4 字节; raw_hex 显示指令十六进制值,
        压缩指令 4 位/32-bit 8 位, 带 [dim] 灰色标记.
        """
        dbg = _make_dbg()
        # C.NOP = 0x0001 (16-bit 压缩指令)
        c_nop = b"\x01\x00"
        dbg._emu.load_code(0x1000, c_nop)
        result = dbg._fetch_and_disasm(0x1000)
        assert result is not None
        raw_hex, asm = result
        # 反汇编结果: 压缩指令显示 4 位十六进制指令值
        assert raw_hex == "0001", f"压缩指令 hex 应为 0001, 实际: {raw_hex}"

    def test_compressed_raw_hex_masked_to_16bit(self):
        """压缩指令 raw_hex 应掩码到低 16 位, 不混入后续指令字节.

        当 4 字节读取包含压缩指令 + 后续指令字节时,
        raw_hex 应只显示压缩指令的 2 字节 (4 hex digits).
        例如 c.addi16sp (0x7159) 后跟 c.sdsp (0xf486),
        4 字节为 0xf4867159, raw_hex 应为 "7159" 而非 "f4867159".
        """
        dbg = _make_dbg()
        # C.ADDI16SP sp, -112 编码 0x7159 后跟 C.SDSP x1, 104(sp) 编码 0xf486
        # 4 字节 little-endian: 59 71 86 f4 -> int = 0xf4867159
        code = b"\x59\x71\x86\xf4"
        dbg._emu.load_code(0x1000, code)
        result = dbg._fetch_and_disasm(0x1000)
        assert result is not None
        raw_hex, asm = result
        assert len(raw_hex) == 4, f"压缩指令 hex 应为 4 位, 实际 {len(raw_hex)} 位: {raw_hex}"
        assert raw_hex == "7159", f"应为 7159 (仅 C.ADDI16SP), 实际: {raw_hex}"
        assert "c.addi16sp" in asm.lower(), f"应为 C.ADDI16SP, 实际: {asm}"

    def test_32bit_raw_hex_unmasked_8_digits(self):
        """32-bit 指令 raw_hex 应显示完整 8 位十六进制."""
        dbg = _make_dbg()
        # ADDI x2, x2, -112 (32-bit) = 0xf9010113
        code = b"\x13\x01\x01\xf9"
        dbg._emu.load_code(0x1000, code)
        result = dbg._fetch_and_disasm(0x1000)
        assert result is not None
        raw_hex, _ = result
        assert len(raw_hex) == 8, \
            f"32-bit 指令 hex 应为 8 位, 实际 {len(raw_hex)} 位: {raw_hex}"
        assert raw_hex == "f9010113", f"应为 f9010113, 实际: {raw_hex}"


# ============================================================
#  L2 cache 显示命令
# ============================================================


class _CacheOutputCapture:
    """捕获 Rich console 输出文本."""

    def __init__(self, dbg):
        self.lines: list[str] = []
        self._orig = dbg._console.print

        def _capture(*args, **kwargs):

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
    num_sets=256,
    ways=4,
    line_size=64,
) -> Debugger:
    """构造带可配置 L2 缓存的 Debugger."""

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
        """cache 无参数 -> 显示统计概览头."""
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
        l2 = dbg._emu.bus._l2
        assert l2 is not None
        # 预填一些数据产生 valid 行
        for i in range(80):
            l2.read(i * 64, 4)
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache()
        text = cap.text()
        # 应有 64 条 (或实际 valid 行数) + 截断提示
        assert "data[:16]" in text, "预览模式应含 data[:16]"
        assert "还有" in text, "超过 64 条时应有截断提示"

    def test_less_than_64_shows_all_no_truncation(self):
        """不足 64 条 valid -> 全部显示, 无截断提示."""
        dbg = _make_dbg_with_l2(num_sets=128, ways=2)
        l2 = dbg._emu.bus._l2
        assert l2 is not None
        for i in range(10):
            l2.read(i * 64, 4)
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache()
        text = cap.text()
        assert "还有" not in text, "不足 64 条不应有截断提示"

    def test_single_set_shows_full_hexdump(self):
        """cache <set> -> 完整 64B hexdump."""
        dbg = _make_dbg_with_l2()
        l2 = dbg._emu.bus._l2
        assert l2 is not None
        l2.read(0, 8)
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache("0")
        text = cap.text()
        assert "data[:16]" not in text, "单 set 模式应为完整 hexdump, 不应含 data[:16]"

    def test_range_shows_preview_format(self):
        """cache <start>-<end> -> 范围预览模式."""
        dbg = _make_dbg_with_l2(num_sets=128, ways=2)
        l2 = dbg._emu.bus._l2
        assert l2 is not None
        for i in range(30):
            l2.read(i * 64, 4)
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache("0-4")
        text = cap.text()
        assert "data[:16]" in text, "范围模式应为预览格式"
        # 不应有截断提示 (范围模式不限条目数)
        assert "还有" not in text

    def test_invalid_set_index(self):
        """非法 set 索引 -> 错误信息."""
        dbg = _make_dbg_with_l2(num_sets=16, ways=2)
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache("99")
        text = cap.text()
        assert "超出范围" in text or "set" in text.lower()

    def test_invalid_range(self):
        """非法范围 -> 错误信息."""
        dbg = _make_dbg_with_l2(num_sets=16, ways=2)
        cap = _CacheOutputCapture(dbg)
        dbg.cmd_cache("10-99")
        text = cap.text()
        assert "超出范围" in text or "set" in text.lower()

    def test_empty_cache(self):
        """空缓存 -> 无 valid 行提示."""
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


# ============================================================
#  _parse_trap_save_offsets — c.sdsp / sd 等价测试
# ============================================================


class TestParseTrapSaveOffsets:
    """测试 _parse_trap_save_offsets 对 c.sdsp 指令的识别.

    使用等价压缩/未压缩指令, 验证两者产生相同的 RA/FP 偏移.
    """

    @staticmethod
    def _encode_sd(rs2: int, rs1: int, imm: int) -> int:
        imm_4_0 = imm & 0x1F
        imm_11_5 = (imm >> 5) & 0x7F
        return (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
            (0x3 << 12) | (imm_4_0 << 7) | 0x23

    @staticmethod
    def _encode_c_sdsp(rs2: int, uimm: int) -> int:
        assert uimm % 8 == 0 and uimm < 512
        uimm5_3 = (uimm >> 3) & 0x7
        uimm8_6 = (uimm >> 6) & 0x7
        return (0x7 << 13) | (uimm5_3 << 10) | (uimm8_6 << 7) | (rs2 << 2) | 0x2

    @staticmethod
    def _make_dbg_with_tvec(raw_bytes: bytes):
        emu = Emulator(num_harts=1, ram_size=0x10000)
        tvec = 0x1000
        emu.load_code(tvec, raw_bytes)
        return Debugger(emulator=emu, hart_id=0), tvec

    # ---- c.sdsp 测试 ----

    def test_c_sdsp_ra_only(self):
        """c.sdsp x1, 8(sp) -> (8, 0)"""
        code = self._encode_c_sdsp(1, 8).to_bytes(2, "little")
        dbg, tvec = self._make_dbg_with_tvec(code)
        result = dbg._parse_trap_save_offsets(tvec)
        assert result == (8, 0), f"expected (8, 0), got {result}"

    def test_c_sdsp_fp_only_returns_none(self):
        """仅保存 fp 无 ra -> None"""
        code = self._encode_c_sdsp(8, 64).to_bytes(2, "little")
        dbg, tvec = self._make_dbg_with_tvec(code)
        result = dbg._parse_trap_save_offsets(tvec)
        assert result is None, f"fp-only should return None, got {result}"

    def test_c_sdsp_both_ra_and_fp(self):
        """c.sdsp x1,8(sp) + c.sdsp x8,64(sp) -> (8, 64)"""
        code = (
            self._encode_c_sdsp(1, 8).to_bytes(2, "little")
            + self._encode_c_sdsp(8, 64).to_bytes(2, "little")
        )
        dbg, tvec = self._make_dbg_with_tvec(code)
        result = dbg._parse_trap_save_offsets(tvec)
        assert result == (8, 64), f"expected (8, 64), got {result}"

    def test_c_sdsp_fp_first_then_ra(self):
        """先保存 fp 后 ra 的顺序也应正确识别"""
        code = (
            self._encode_c_sdsp(8, 64).to_bytes(2, "little")
            + self._encode_c_sdsp(1, 8).to_bytes(2, "little")
        )
        dbg, tvec = self._make_dbg_with_tvec(code)
        result = dbg._parse_trap_save_offsets(tvec)
        assert result == (8, 64), f"expected (8, 64), got {result}"

    # ---- 32-bit sd 测试 ----

    def test_sd_ra_only(self):
        """sd x1, 8(sp) -> (8, 0)"""
        code = self._encode_sd(1, 2, 8).to_bytes(4, "little")
        dbg, tvec = self._make_dbg_with_tvec(code)
        result = dbg._parse_trap_save_offsets(tvec)
        assert result == (8, 0), f"expected (8, 0), got {result}"

    def test_sd_fp_only_returns_none(self):
        """sd x8, 64(sp) 但无 ra -> None"""
        code = self._encode_sd(8, 2, 64).to_bytes(4, "little")
        dbg, tvec = self._make_dbg_with_tvec(code)
        result = dbg._parse_trap_save_offsets(tvec)
        assert result is None, f"fp-only should return None, got {result}"

    def test_sd_both_ra_and_fp(self):
        """sd x1,8(sp) + sd x8,64(sp) -> (8, 64)"""
        code = (
            self._encode_sd(1, 2, 8).to_bytes(4, "little")
            + self._encode_sd(8, 2, 64).to_bytes(4, "little")
        )
        dbg, tvec = self._make_dbg_with_tvec(code)
        result = dbg._parse_trap_save_offsets(tvec)
        assert result == (8, 64), f"expected (8, 64), got {result}"

    # ---- 等价性: 压缩 vs 未压缩 ----

    def test_equivalent_ra_offset(self):
        """c.sdsp 和 sd 对 ra 产生相同偏移"""
        c_code = self._encode_c_sdsp(1, 8).to_bytes(2, "little")
        sd_code = self._encode_sd(1, 2, 8).to_bytes(4, "little")

        dbg_c, tv_c = self._make_dbg_with_tvec(c_code)
        dbg_sd, tv_sd = self._make_dbg_with_tvec(sd_code)

        assert dbg_c._parse_trap_save_offsets(tv_c) == dbg_sd._parse_trap_save_offsets(tv_sd)

    def test_equivalent_fp_offset(self):
        """c.sdsp 和 sd 对 fp 产生相同偏移 (需同时有 ra)"""
        c_code = (
            self._encode_c_sdsp(1, 8).to_bytes(2, "little")
            + self._encode_c_sdsp(8, 64).to_bytes(2, "little")
        )
        sd_code = (
            self._encode_sd(1, 2, 8).to_bytes(4, "little")
            + self._encode_sd(8, 2, 64).to_bytes(4, "little")
        )

        dbg_c, tv_c = self._make_dbg_with_tvec(c_code)
        dbg_sd, tv_sd = self._make_dbg_with_tvec(sd_code)

        assert dbg_c._parse_trap_save_offsets(tv_c) == dbg_sd._parse_trap_save_offsets(tv_sd)

    # ---- 边界 ----

    def test_empty_tvec_returns_none(self):
        """空 trap entry -> None"""
        dbg, tvec = self._make_dbg_with_tvec(b"\x00\x00\x00\x00")
        result = dbg._parse_trap_save_offsets(tvec)
        assert result is None, f"empty entry should return None, got {result}"

    def test_max_instrs_bound(self):
        """max_instrs 限制查找范围"""
        c_ra = self._encode_c_sdsp(1, 8).to_bytes(2, "little")
        # 填充足够多的 NOP 使 c.sdsp 位于 max_instrs 范围外
        padding = b"\x01\x00" * 10  # c.nop × 10
        code = padding + c_ra
        dbg, tvec = self._make_dbg_with_tvec(code)
        # max_instrs=5, 每条压缩指令 2 字节, 最多读 10 字节
        # padding 10×2=20 字节, ra 在第 20 字节之后 -> 找不到
        result = dbg._parse_trap_save_offsets(tvec, max_instrs=5)
        assert result is None, f"ra beyond max_instrs should return None, got {result}"


# ============================================================
#  _fmt_instr_count — 指令数 human-readable
# ============================================================


class TestFmtInstrCount:
    """_fmt_instr_count 格式化."""

    def test_small(self):
        assert Debugger._fmt_instr_count(0) == "0"
        assert Debugger._fmt_instr_count(999) == "999"

    def test_k(self):
        assert Debugger._fmt_instr_count(1000) == "1.000K"
        assert Debugger._fmt_instr_count(500000) == "500.000K"
        assert Debugger._fmt_instr_count(999999) == "999.999K"

    def test_m(self):
        assert Debugger._fmt_instr_count(1_000_000) == "1.000000M"
        assert Debugger._fmt_instr_count(500_000_000) == "500.000000M"

    def test_b(self):
        assert Debugger._fmt_instr_count(1_000_000_000) == "1.0000000B"


# ============================================================
#  _colorize_asm — 汇编语法着色
# ============================================================


class TestColorizeAsm:
    """_colorize_asm 着色."""

    def test_unknown_sentinel_escaped(self):
        """<unknown 开头的指令不做 Rich 标记, 原样返回."""
        result = Debugger._colorize_asm("<unknown>")
        # unknown sentinel 不做着色, 原样返回
        assert result == "<unknown>"
        assert "[green]" not in result
        assert "[yellow]" not in result

    def test_branch_is_green(self):
        """分支指令标记为绿色."""
        result = Debugger._colorize_asm("beq     x10,x11,0x1000")
        assert "[green]" in result

    def test_jump_is_green(self):
        """跳转指令标记为绿色."""
        result = Debugger._colorize_asm("jal     ra,0x80000000")
        assert "[green]" in result

    def test_fence_is_yellow(self):
        """fence/AMO 指令标记为黄色."""
        result = Debugger._colorize_asm("fence   iorw,iorw")
        assert "[yellow]" in result

    def test_amo_is_yellow(self):
        """AMO/LR/SC 指令标记为黄色."""
        result = Debugger._colorize_asm("lr.d    x5,(x6)")
        assert "[yellow]" in result

    def test_normal_is_plain(self):
        """普通指令不着色助记符."""
        result = Debugger._colorize_asm("addi    x5,x6,42")
        assert "[green]" not in result
        assert "[yellow]" not in result

    def test_immediate_colored_magenta(self):
        """立即数标记为品红."""
        result = Debugger._colorize_asm("addi    x5,x6,0x1000")
        assert "[magenta]0x1000[/]" in result

    def test_register_not_colored(self):
        """寄存器名不应被品红着色."""
        result = Debugger._colorize_asm("addi    x5,x6,42")
        # x5, x6 不应被着色
        assert "[magenta]x5[/]" not in result
        assert "[magenta]x6[/]" not in result

    def test_no_operands(self):
        """无操作数指令仅着色助记符."""
        result = Debugger._colorize_asm("ecall   ")
        assert "ecall" in result


# ============================================================
#  _ctrl_flow_kind / _ctrl_flow_kind_compressed
# ============================================================


class TestCtrlFlowKind:
    """控制流分类."""

    # -- 32-bit 指令 --

    def test_jal_is_term(self):
        """JAL 为无条件终止."""
        # jal zero, 16
        instr = (16 << 21) | (0 << 12) | (0 << 7) | 0b1101111
        assert Debugger._ctrl_flow_kind(instr) == "jal"

    def test_jalr_is_term(self):
        """JALR 为无条件终止."""
        instr = (0 << 20) | (1 << 15) | (0b000 << 12) | (0 << 7) | 0b1100111
        assert Debugger._ctrl_flow_kind(instr) == "jalr"

    def test_beq_is_branch(self):
        """BEQ 为条件分支."""
        # beq x0, x0, 8
        instr = (0 << 25) | (0 << 20) | (0 << 15) | (0b000 << 12) | (1 << 7) | 0b1100011
        assert Debugger._ctrl_flow_kind(instr) == "branch"

    def test_ecall_is_term(self):
        """ECALL 为终止."""
        instr = 0x00000073  # ecall
        assert Debugger._ctrl_flow_kind(instr) == "ecall"

    def test_ebreak_is_term(self):
        """EBREAK 为终止."""
        instr = 0x00100073  # ebreak
        assert Debugger._ctrl_flow_kind(instr) == "ebreak"

    def test_mret_is_term(self):
        """MRET 为终止."""
        instr = 0x30200073  # mret
        assert Debugger._ctrl_flow_kind(instr) == "mret"

    def test_sret_is_term(self):
        """SRET 为终止."""
        instr = 0x10200073  # sret
        assert Debugger._ctrl_flow_kind(instr) == "sret"

    def test_addi_is_normal(self):
        """ADDI 为普通指令."""
        instr = (0 << 20) | (0 << 15) | (0b000 << 12) | (5 << 7) | 0b0010011
        assert Debugger._ctrl_flow_kind(instr) == "normal"

    # -- 压缩指令 --

    def test_c_jal_is_term(self):
        """C.JAL 为终止."""
        # funct3=001, quad=01, rd=ra
        instr16 = (0b001 << 13) | (1 << 7) | 0b01
        assert Debugger._ctrl_flow_kind_compressed(instr16) == "jal"

    def test_c_j_is_term(self):
        """C.J 为终止."""
        instr16 = (0b101 << 13) | 0b01
        assert Debugger._ctrl_flow_kind_compressed(instr16) == "jal"

    def test_c_beqz_is_branch(self):
        """C.BEQZ 为条件分支."""
        instr16 = (0b110 << 13) | 0b01
        assert Debugger._ctrl_flow_kind_compressed(instr16) == "branch"

    def test_c_bnez_is_branch(self):
        """C.BNEZ 为条件分支."""
        instr16 = (0b111 << 13) | 0b01
        assert Debugger._ctrl_flow_kind_compressed(instr16) == "branch"

    def test_c_jr_is_term(self):
        """C.JR 为终止."""
        # funct3=100, quad=10, rs1≠0
        instr16 = (0b100 << 13) | (1 << 7) | 0b10  # c.jr ra
        assert Debugger._ctrl_flow_kind_compressed(instr16) == "jalr"

    def test_c_addi_is_normal(self):
        """C.ADDI 为普通指令."""
        instr16 = (0b000 << 13) | (1 << 7) | 0b01
        assert Debugger._ctrl_flow_kind_compressed(instr16) == "normal"

    def test_c_mv_is_normal_not_term(self):
        """C.MV (0x8512) 是 rs2≠0 的普通指令, 不应被误判为 C.JR/C.JALR 终止.

        C2 quad funct3=100 编码族内, rs2=0 才是跳转 (C.JR/C.JALR/C.EBREAK),
        rs2≠0 是 C.MV / C.ADD 普通指令.
        """
        instr16 = 0x8512  # c.mv x10, x4 — 真实触发 bug 的编码
        assert Debugger._ctrl_flow_kind_compressed(instr16) == "normal"

    def test_c_add_is_normal_not_term(self):
        """C.ADD 是 rs2≠0 的普通指令, 不应被误判为终止."""
        # funct3=100, quad=10, rs1=x10, rs2=x6 -> c.add x10, x6
        instr16 = (0b100 << 13) | (10 << 7) | (6 << 2) | 0b10  # 0x9526
        assert Debugger._ctrl_flow_kind_compressed(instr16) == "normal"


# ============================================================
#  _ip_bits — 中断位定义表
# ============================================================


class TestIpBits:
    """_ip_bits 静态方法."""

    def test_returns_correct_count(self):
        bits = Debugger._ip_bits()
        assert len(bits) == 9

    def test_all_bit_numbers_unique(self):
        bits = Debugger._ip_bits()
        bit_nums = {b[1] for b in bits}
        assert len(bit_nums) == 9

    def test_msip_at_bit3(self):
        bits = Debugger._ip_bits()
        msip = next(b for b in bits if b[0] == "MSIP")
        assert msip[1] == 3


# ============================================================
#  CSR detail commands (mcause, scause, mtvec, stvec, mip, mie, sip, sie, medeleg, mideleg)
# ============================================================


class TestCsrDetailCommands:
    """CSR 详情显示命令."""

    @pytest.fixture
    def dbg(self):
        return _make_dbg()

    def test_cmd_mcause_runs(self, dbg):
        """cmd_mcause 不抛异常."""
        dbg.hart.csrs["mcause"].val = 0x8000_0000_0000_0003  # MSI
        dbg.cmd_mcause()  # 不应抛异常

    def test_cmd_scause_runs(self, dbg):
        """cmd_scause 不抛异常."""
        dbg.hart.csrs["scause"].val = 8  # ECALL from U
        dbg.cmd_scause()

    def test_cmd_mtvec_runs(self, dbg):
        """cmd_mtvec 不抛异常."""
        dbg.hart.csrs["mtvec"].val = 0x80000000  # direct mode
        dbg.cmd_mtvec()

    def test_cmd_stvec_runs(self, dbg):
        """cmd_stvec 不抛异常."""
        dbg.hart.csrs["stvec"].val = 0x80000001  # vectored mode
        dbg.cmd_stvec()

    def test_cmd_mip_runs(self, dbg):
        """cmd_mip 不抛异常."""
        dbg.hart.csrs["mip"].val = 1 << 7  # MTIP
        dbg.cmd_mip()

    def test_cmd_mie_runs(self, dbg):
        """cmd_mie 不抛异常."""
        dbg.hart.csrs["mie"].val = 1 << 3  # MSIE
        dbg.cmd_mie()

    def test_cmd_sip_runs(self, dbg):
        """cmd_sip 不抛异常."""
        dbg.hart.csrs["sip"].val = 1 << 5  # STIP
        dbg.cmd_sip()

    def test_cmd_sie_runs(self, dbg):
        """cmd_sie 不抛异常."""
        dbg.hart.csrs["sie"].val = 1 << 5  # STIE
        dbg.cmd_sie()

    def test_cmd_medeleg_runs(self, dbg):
        """cmd_medeleg 不抛异常."""
        dbg.hart.csrs["medeleg"].val = 1 << 8  # delegate ECALL-U
        dbg.cmd_medeleg()

    def test_cmd_mideleg_runs(self, dbg):
        """cmd_mideleg 不抛异常."""
        dbg.hart.csrs["mideleg"].val = 1 << 5  # delegate STI
        dbg.cmd_mideleg()


# ============================================================
#  cmd_pmp — PMP 条目显示
# ============================================================


class TestCmdPmp:
    """cmd_pmp PMP 条目显示."""

    def test_no_pmp_entries(self):
        """PMP 未配置时显示提示."""
        dbg = _make_dbg()
        dbg.cmd_pmp()  # 不应抛异常

    def test_pmp_with_entries(self):
        """PMP 有配置条目时正常显示."""
        dbg = _make_dbg()
        h = dbg.hart
        # 配置一条 TOR entry
        pmp = h._pmp
        if pmp is not None:
            # pmpcfg0: entry 0 = TOR, R/W/X
            h.csrs["pmpcfg0"].val = PMP_A_TOR | PMP_R | PMP_W | PMP_X
            h.csrs["pmpaddr0"].val = 0x20000000 >> 2  # TOR hi bound
            dbg.cmd_pmp()  # 不应抛异常


# ============================================================
#  cmd_pt — Sv39 页表遍历显示
# ============================================================


class TestCmdPt:
    """cmd_pt 页表遍历显示."""

    def test_pt_bare_mode(self):
        """Bare 模式下 pt 显示相应信息."""
        dbg = _make_dbg()
        dbg.hart.csrs["satp"].val = 0  # Bare mode
        # 在 Bare 模式下 sv39_walk 会失败, cmd_pt 应妥善处理
        dbg.cmd_pt("0x1000")  # 不应抛异常

    def test_pt_with_table(self):
        """Sv39 模式下 pt 遍历显示."""
        dbg = _make_dbg()
        h = dbg.hart
        # 构建三级页表 (identity map 0x1000 -> 0x80001000)
        # 写入 L1 (根页表) -> L2
        l2_pte = PTE()
        l2_pte.v = True
        l2_pte.ppn = L2_BASE >> 12  # 软件 PPN
        dbg._emu.bus.write(L1_BASE, l2_pte.to_int().to_bytes(8, "little"))

        # 写入 L2 -> L3 (4 KiB page)
        l3_pte = PTE()
        l3_pte.v = True
        l3_pte.ppn = L3_BASE >> 12
        dbg._emu.bus.write(L2_BASE, l3_pte.to_int().to_bytes(8, "little"))

        # 写入 L3 -> 目标页 (R/W/X)
        leaf = PTE()
        leaf.v = True
        leaf.r = True
        leaf.w = True
        leaf.x = True
        leaf.ppn = 0x80001  # -> PA 0x80001000
        dbg._emu.bus.write(L3_BASE, leaf.to_int().to_bytes(8, "little"))

        # 设置 satp
        h.csrs["satp"].val = (8 << 60) | (L1_BASE >> 12)
        dbg.cmd_pt("0x1000")  # 不应抛异常


# ============================================================
#  _resolve_boundary_frame / privilege boundary
# ============================================================


class TestPrivilegeBoundary:
    """特权级边界帧解析."""

    def test_resolve_mmode_frame(self):
        """_resolve_boundary_frame 对 M 模式返回 M."""
        dbg = _make_dbg()
        boundary = dbg._resolve_boundary_frame(0x80000000, "M")
        assert boundary is not None
        assert boundary[0] == "M"  # (label, boundary_mode)

    def test_is_valid_mmode_code_true(self):
        """M 模式代码段判断 (需要 _image 中 .text 段名)."""
        dbg = _make_dbg()
        # 注入 image 使 _is_valid_mmode_code 能检查段名
        img = _make_image()
        img.segments = [
            FirmwareSegment(
                vaddr=0x80000000, memsz=0x10000,
                data=b"\x00" * 0x10000, name=".text",
            )
        ]
        dbg._image = img
        assert dbg._is_valid_mmode_code(0x80000000) is True

    def test_is_valid_mmode_code_false(self):
        """非 M 模式代码段判断."""
        dbg = _make_dbg()
        # 任意地址不在 RAM 内
        assert dbg._is_valid_mmode_code(0x1000) is False
