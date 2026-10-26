#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""ECALL 与 EBREAK: 各特权级的分派, 断点陷阱与 semihosting 服务."""

import struct

import pytest

from pyremu.core import trap_handler
from pyremu.core.decoder import Hart
from pyremu.core.hart import (
    RiscvMode,
)
from pyremu.core.mem_check_aux import (
    inject_memory_backend,
)
from pyremu.core.trap_handler import (
    trap_ebreak,
    trap_ecall,
)
from pyremu.memory.bus import Bus
from tests.test_trap.support import (
    MTVEC,
    SEMIHOSTING_ERR,
)


class TestEcallEbreak:
    """验证 ECALL 和 EBREAK 的具体行为."""

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x2000
        return h

    def test_ecall_from_m_mode(self, hart):
        hart.mode = RiscvMode.M
        trap_ecall(hart)
        assert hart.mcause_val == 11, "M 模式 ecall -> mcause=11"
        assert hart.mepc_val == 0x2000

    def test_ecall_from_s_mode(self, hart):
        hart.mode = RiscvMode.S
        trap_ecall(hart)
        assert hart.mcause_val == 9, "S 模式 ecall -> mcause=9"

    def test_ecall_from_u_mode(self, hart):
        hart.mode = RiscvMode.U
        trap_ecall(hart)
        assert hart.mcause_val == 8, "U 模式 ecall -> mcause=8"

    def test_ebreak(self, hart):
        trap_ebreak(hart)
        assert hart.mcause_val == 3, "ebreak -> mcause=3"
        assert hart.mtval_val == 0x2000, "mtval 应保存断点地址"


class TestSemihostingEbreak:
    """带 marker 的 semihosting ebreak 须与 native 引擎一致内联服务.

    修复前 python 引擎只识别 SYS_EXIT (a0=0x18), SYS_WRITE/SYS_WRITEC 的
    ebreak 被投递为普通断点 trap — 当固件在持有 console_out_lock 的 M 模式
    打印路径内执行 semihosting 写时, 嵌套 trap 使控制台锁无法释放, 造成自旋
    锁死锁. 本类锁定通用 semihosting 服务的回归底线.
    """

    _PRE = 0x01F01013  # slli x0, x0, 0x1f
    _POST = 0x40705013  # srai x0, x0, 7
    _EBREAK_PC = 0x80001000

    @pytest.fixture(autouse=True)
    def enable_semihosting(self, monkeypatch):
        """本类覆盖内联服务的处理, 显式打开开关, 与编译选项无关.

        关闭开关的用例在自身内部再把它置 0.
        """
        monkeypatch.setattr(trap_handler, "PYREMU_HAS_SEMIHOSTING", 1)

    @pytest.fixture
    def sh(self):
        bus = Bus(ram_size=0x100000, ram_base=0x80000000)
        hart = Hart(id=0)
        inject_memory_backend(hart, bus.read, bus.write)
        hart.bus = bus
        hart.csrs["mtvec"].val = 0x80010000
        return hart, bus

    @staticmethod
    def _put(bus: Bus, addr: int, data: bytes) -> None:
        off = addr - bus.ram_base
        bus._ram[off : off + len(data)] = data

    def _place_ebreak(self, bus: Bus) -> None:
        """在 hart.pc 两侧写入 semihosting marker (中段 4 字节内容不限)."""
        self._put(bus, self._EBREAK_PC - 4, struct.pack("<I", self._PRE))
        self._put(bus, self._EBREAK_PC + 4, struct.pack("<I", self._POST))

    def test_non_marker_ebreak_still_traps(self, sh):
        """无 marker 的 ebreak 维持断点 trap 语义."""
        hart, _ = sh
        hart.pc = self._EBREAK_PC
        trap_ebreak(hart)
        assert hart.mcause_val == 3, "普通 ebreak -> mcause=3"

    def test_sys_write_emits_and_advances(self, sh, capfd):
        """SYS_WRITE (op=0x5) 读到 RAM 缓冲 -> 写到 stderr, a0=0, pc 前进 8."""
        hart, bus = sh
        self._place_ebreak(bus)
        param, buf = 0x80002000, 0x80003000
        text = b"hello semihost\n"
        self._put(bus, buf, text)
        self._put(bus, param, struct.pack("<QQQ", 1, buf, len(text)))
        hart.pc = self._EBREAK_PC
        hart.gprs[10] = 0x05
        hart.gprs[11] = param
        trap_ebreak(hart)
        assert text in capfd.readouterr().err.encode()
        assert hart.gprs[10] == 0, "SYS_WRITE 成功 -> a0=0"
        assert hart.pc == self._EBREAK_PC + 8, "越过 ebreak + 退出 marker"
        assert hart.mcause_val == 0, "不得投递断点 trap"

    def test_sys_write_oob_buffer_returns_minus1(self, sh, capfd):
        """缓冲为 S-mode VA (超出 RAM) -> -1, 不输出 — 与 native 越界语义一致."""
        hart, bus = sh
        self._place_ebreak(bus)
        param = 0x80002000
        va = 0xFFFF_FFE0_0000_0000  # 飞地 DBCN 打印常传 S-mode VA
        self._put(bus, param, struct.pack("<QQQ", 1, va, 5))
        hart.pc = self._EBREAK_PC
        hart.gprs[10] = 0x05
        hart.gprs[11] = param
        trap_ebreak(hart)
        assert hart.gprs[10] == 0xFFFF_FFFF_FFFF_FFFF, "越界缓冲 -> -1"
        assert hart.pc == self._EBREAK_PC + 8
        assert capfd.readouterr().err == ""

    def test_sys_writec_emits_single_char(self, sh, capfd):
        """SYS_WRITEC (op=0x3) 参数指向 RAM 中单个字符."""
        hart, bus = sh
        self._place_ebreak(bus)
        param = 0x80002000
        self._put(bus, param, b"Z")
        hart.pc = self._EBREAK_PC
        hart.gprs[10] = 0x03
        hart.gprs[11] = param
        trap_ebreak(hart)
        assert capfd.readouterr().err == "Z"
        assert hart.gprs[10] == 0
        assert hart.pc == self._EBREAK_PC + 8

    def test_sys_exit_sets_stop_flag_not_trap(self, sh):
        """SYS_EXIT (op=0x18) 维持停机标记, 不投递断点 trap."""
        hart, bus = sh
        self._place_ebreak(bus)
        hart.pc = self._EBREAK_PC
        hart.gprs[10] = 0x18
        trap_ebreak(hart)
        assert getattr(hart, "_semihosting_sys_exit", False) is True
        assert hart.mcause_val == 0, "SYS_EXIT 不得投递断点 trap"

    def test_marker_ebreak_traps_when_semihosting_disabled(self, sh, monkeypatch):
        """PYREMU_HAS_SEMIHOSTING=0 时 marker ebreak 按断点 trap 投递.

        固件的 semihosting 探测依赖该 trap 判定宿主不支持, 故该序列不得被
        内联服务吞掉; pc 与 a0 保持原值.
        """
        hart, bus = sh
        self._place_ebreak(bus)
        hart.pc = self._EBREAK_PC
        hart.gprs[10] = 0x18
        monkeypatch.setattr(trap_handler, "PYREMU_HAS_SEMIHOSTING", 0)
        trap_ebreak(hart)
        assert hart.mcause_val == 3, "内联服务关闭 -> 断点 trap"
        assert hart.mepc_val == self._EBREAK_PC, "mepc 指向 ebreak 本身"
        assert hart.pc == 0x80010000, "跳转到 mtvec"
        assert hart.gprs[10] == 0x18, "不得改写 a0"
        assert getattr(hart, "_semihosting_sys_exit", False) is False

    def test_unknown_op_returns_minus1(self, sh, capfd):
        """native 不识别操作码 -> -1, 空写, pc 前进 8."""
        hart, bus = sh
        self._place_ebreak(bus)
        hart.pc = self._EBREAK_PC
        hart.gprs[10] = 0x63
        trap_ebreak(hart)
        assert hart.gprs[10] == 0xFFFF_FFFF_FFFF_FFFF
        assert hart.pc == self._EBREAK_PC + 8
        assert capfd.readouterr().err == ""


class TestSemihostingBranches:
    _PRE = 0x01F01013
    _POST = 0x40705013
    _EBREAK_PC = 0x8000_1000

    @pytest.fixture(autouse=True)
    def enable_semihosting(self, monkeypatch):
        monkeypatch.setattr(trap_handler, "PYREMU_HAS_SEMIHOSTING", 1)

    @pytest.fixture
    def sh(self):
        bus = Bus(ram_size=0x1_0000, ram_base=0x8000_0000)
        hart = Hart(id=0)
        inject_memory_backend(hart, bus.read, bus.write)
        hart.bus = bus
        hart.csrs["mtvec"].val = MTVEC
        hart.pc = self._EBREAK_PC
        self._put(bus, self._EBREAK_PC - 4, struct.pack("<I", self._PRE))
        self._put(bus, self._EBREAK_PC + 4, struct.pack("<I", self._POST))
        return hart, bus

    @staticmethod
    def _put(bus: Bus, addr: int, data: bytes) -> None:
        assert bus.try_write(addr, data)

    def _write_param(self, bus: Bus, values: tuple[int, ...]) -> int:
        addr = 0x8000_2000
        self._put(bus, addr, struct.pack("<" + "Q" * len(values), *values))
        return addr

    def test_missing_bus_falls_back_to_breakpoint(self, sh):
        hart, _ = sh
        hart.bus = None
        trap_ebreak(hart)
        assert hart.mcause_val == 3

    def test_short_marker_read_falls_back_to_breakpoint(self, sh):
        hart, _ = sh

        class _ShortBus:
            @staticmethod
            def try_read(addr: int, size: int) -> bytes:
                return b"\x13"

        hart.bus = _ShortBus()
        trap_ebreak(hart)
        assert hart.mcause_val == 3

    def test_writec_outside_ram_emits_nul(self, sh, capfd):
        """参数指针不在 RAM 内时读到全零, 输出一个 NUL 字节."""
        hart, _ = sh
        hart.gprs[10] = 0x03
        hart.gprs[11] = 0x9000_0000
        trap_ebreak(hart)
        assert hart.gprs[10] == 0
        assert capfd.readouterr().err == "\x00"
        assert hart.pc == self._EBREAK_PC + 8

    def test_zero_length_write_succeeds_silently(self, sh, capfd):
        hart, bus = sh
        hart.gprs[10] = 0x05
        hart.gprs[11] = self._write_param(bus, (1, 0x8000_3000, 0))
        trap_ebreak(hart)
        assert hart.gprs[10] == 0
        assert capfd.readouterr().err == ""

    def test_write_with_unexpected_fd_fails(self, sh, capfd):
        hart, bus = sh
        hart.gprs[10] = 0x05
        hart.gprs[11] = self._write_param(bus, (3, 0x8000_3000, 4))
        trap_ebreak(hart)
        assert hart.gprs[10] == SEMIHOSTING_ERR
        assert capfd.readouterr().err == ""

    def test_open_reports_valid_fd(self, sh):
        hart, _ = sh
        hart.gprs[10] = 0x01
        trap_ebreak(hart)
        assert hart.gprs[10] == 1
        assert hart.pc == self._EBREAK_PC + 8

    def test_istty_reports_standard_streams(self, sh):
        hart, bus = sh
        hart.gprs[10] = 0x09
        hart.gprs[11] = self._write_param(bus, (2,))
        trap_ebreak(hart)
        assert hart.gprs[10] == 1

        hart.pc = self._EBREAK_PC
        hart.gprs[10] = 0x09
        hart.gprs[11] = self._write_param(bus, (3,))
        trap_ebreak(hart)
        assert hart.gprs[10] == 0

    def test_stderr_failure_is_ignored(self, sh, monkeypatch):
        hart, bus = sh
        hart.gprs[10] = 0x03
        hart.gprs[11] = self._write_param(bus, (ord("Q"),))

        def _raise_oserror(fd: int, data: bytes) -> int:
            raise OSError(5, "input/output error")

        monkeypatch.setattr(trap_handler.os, "write", _raise_oserror)
        trap_ebreak(hart)
        assert hart.gprs[10] == 0
        assert hart.pc == self._EBREAK_PC + 8
