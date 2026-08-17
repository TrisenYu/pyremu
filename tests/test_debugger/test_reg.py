#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""测试 pyremu.debug.reg — GPR/CSR 读写命令."""

from pyremu.core.registers import gpr_alias
from pyremu.debug.reg import RegisterMixin
from pyremu.emulator import Emulator

# ============================================================
#  Mini 测试类: RegisterMixin + 必要属性
# ============================================================


class _TestRegDbg(RegisterMixin):
    """最小聚合类供 RegisterMixin 命令测试."""

    def __init__(self, emu, hart_id=0):
        from rich.console import Console
        self._emu = emu
        self._hart_id = hart_id
        self._console = Console(highlight=False)
        self._err = lambda msg: None  # no-op

    @property
    def hart(self):
        return self._emu.harts[self._hart_id]


def _make_tdbg(num_harts=1, ram_size=0x10000):
    emu = Emulator(num_harts=num_harts, ram_size=ram_size, prog_cnt=0x1000)
    emu.load_code(0x1000, b"\x13\x00\x00\x00")  # NOP
    return _TestRegDbg(emu)


# ============================================================
#  GPR 查找
# ============================================================


class TestFindGpr:
    """_find_gpr — 按 xN 或 ABI 名查找."""

    def test_find_by_alias(self):
        dbg = _make_tdbg()
        idx = dbg._find_gpr("sp")
        assert idx is not None
        assert gpr_alias(idx) == "sp"

    def test_find_by_xn(self):
        dbg = _make_tdbg()
        idx = dbg._find_gpr("x2")
        assert idx is not None
        assert gpr_alias(idx) == "sp"

    def test_find_by_name_case_insensitive(self):
        dbg = _make_tdbg()
        idx = dbg._find_gpr("SP")
        assert idx is not None
        assert gpr_alias(idx) == "sp"

    def test_find_unknown_returns_none(self):
        dbg = _make_tdbg()
        assert dbg._find_gpr("nonexistent") is None

    def test_find_all_abi_names(self):
        dbg = _make_tdbg()
        aliases = [
            "zero", "ra", "sp", "gp", "tp",
            "t0", "t1", "t2", "t3", "t4", "t5", "t6",
            "s0", "s1", "s2", "s3", "s4", "s5", "s6",
            "s7", "s8", "s9", "s10", "s11",
            "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
        ]
        for alias in aliases:
            idx = dbg._find_gpr(alias)
            assert idx is not None, f"找不到 ABI 名: {alias}"
            assert gpr_alias(idx) == alias

    def test_xn_out_of_range_returns_none(self):
        dbg = _make_tdbg()
        assert dbg._find_gpr("x32") is None

    def test_xn_zero(self):
        dbg = _make_tdbg()
        idx = dbg._find_gpr("x0")
        assert idx == 0


# ============================================================
#  GPR 命令
# ============================================================


class TestCmdRegs:
    """cmd_regs — 全部 GPR 显示."""

    def test_cmd_regs_does_not_raise(self):
        dbg = _make_tdbg()
        dbg.cmd_regs()  # 应不抛异常


class TestCmdReg:
    """cmd_reg — 单/多寄存器读取."""

    def test_reg_single(self):
        dbg = _make_tdbg()
        dbg.hart.gprs[10] = 0xCAFE  # a0
        dbg.cmd_reg("a0")

    def test_reg_by_xn(self):
        dbg = _make_tdbg()
        dbg.hart.gprs[5] = 0x42
        dbg.cmd_reg("x5")

    def test_reg_multi_comma_separated(self):
        dbg = _make_tdbg()
        dbg.hart.gprs[10] = 0xA
        dbg.hart.gprs[11] = 0xB
        dbg.cmd_reg("a0, a1")

    def test_reg_unknown(self):
        dbg = _make_tdbg()
        dbg.cmd_reg("nonexistent")

    def test_reg_empty_args(self):
        dbg = _make_tdbg()
        dbg.cmd_reg("")


class TestCmdSet:
    """cmd_set — GPR 写入."""

    def test_set_writes_gpr(self):
        dbg = _make_tdbg()
        dbg.cmd_set("a0", "0xCAFE")
        assert dbg.hart.gprs[10] == 0xCAFE

    def test_set_by_xn(self):
        dbg = _make_tdbg()
        dbg.cmd_set("x5", "42")
        assert dbg.hart.gprs[5] == 42

    def test_set_unknown_reg(self):
        dbg = _make_tdbg()
        dbg.cmd_set("nonexistent", "0")

    def test_set_invalid_value(self):
        dbg = _make_tdbg()
        dbg.cmd_set("a0", "not_a_number")

    def test_set_64bit_mask(self):
        """写入值应被 64-bit 掩码."""
        dbg = _make_tdbg()
        dbg.cmd_set("a0", "0xDEADBEEF_DEADBEEF_DEAD")
        assert dbg.hart.gprs[10] == 0xBEEF_DEADBEEF_DEAD


# ============================================================
#  CSR 命令
# ============================================================


class TestCmdCsr:
    """cmd_csr / cmd_csrw."""

    def test_csr_list(self):
        dbg = _make_tdbg()
        dbg.cmd_csr("list")  # 应不抛异常

    def test_csr_read(self):
        dbg = _make_tdbg()
        dbg.hart.csrs["mstatus"].val = 0x1800
        dbg.cmd_csr("mstatus")

    def test_csr_multi_comma_separated(self):
        dbg = _make_tdbg()
        dbg.hart.csrs["mstatus"].val = 0x1800
        dbg.hart.csrs["mtvec"].val = 0x80000000
        dbg.cmd_csr("mstatus, mtvec")

    def test_csr_mcause_decodes(self):
        """mcause 应解码陷态原因."""
        dbg = _make_tdbg()
        dbg.hart.csrs["mcause"].val = 11  # Ecall from M-mode
        dbg.cmd_csr("mcause")

    def test_csr_scause_decodes(self):
        dbg = _make_tdbg()
        dbg.hart.csrs["scause"].val = 8  # Ecall from U-mode
        dbg.cmd_csr("scause")

    def test_csr_unknown(self):
        dbg = _make_tdbg()
        dbg.cmd_csr("nonexistent_csr")

    def test_csr_empty_args(self):
        dbg = _make_tdbg()
        dbg.cmd_csr("")

    def test_csrw_writes(self):
        dbg = _make_tdbg()
        dbg.cmd_csrw("mstatus", "0x1234")
        assert dbg.hart.csrs["mstatus"].val == 0x1234

    def test_csrw_unknown(self):
        dbg = _make_tdbg()
        dbg.cmd_csrw("nonexistent", "0")

    def test_csrw_invalid_value(self):
        dbg = _make_tdbg()
        dbg.cmd_csrw("mstatus", "BAD")
