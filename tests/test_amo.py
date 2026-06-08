#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""原子指令测试: LR/SC 预留机制, AMO 读-改-写."""

import pytest

from pyremu.core.decoder import (
    AmoFunct5,
    AmoWidth,
    Hart,
)


def _make_amo_instr(op: AmoFunct5, width: AmoWidth, rd: int, rs1: int, rs2: int) -> int:
    """构造一条 AMO 指令字."""
    return (
        (op.value << 27)
        | (width.value << 12)
        | (rs2 << 20)
        | (rs1 << 15)
        | (rd << 7)
        | 0b0101111  # AMO opcode
    )


def _make_ram():
    ram = bytearray(2 * 1024 * 1024)

    def read_fn(addr, size):
        return bytes(ram[addr : addr + size])

    def write_fn(addr, data):
        for i, b in enumerate(data):
            ram[addr + i] = b

    return ram, read_fn, write_fn


class TestLRSC:
    """LR/SC 预留机制."""

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        ram, rf, wf = _make_ram()
        h.set_memory_backend(rf, wf)
        return h

    def test_lr_sets_reservation(self, hart):
        """LR 执行后应设置预留."""
        hart.gprs[10].val = 0x1000  # x10 = addr
        instr = _make_amo_instr(AmoFunct5.LR, AmoWidth.D, rd=11, rs1=10, rs2=0)
        hart.exec_instr(instr)
        assert hart.reservation_valid
        assert hart.reservation_addr == 0x1000

    def test_sc_succeeds_with_valid_reservation(self, hart):
        """预留有效时 SC 应成功, rd←0."""
        hart.gprs[10].val = 0x1000  # addr
        hart.gprs[12].val = 0xDEADBEEF  # store value
        # 先做 LR
        lr = _make_amo_instr(AmoFunct5.LR, AmoWidth.D, rd=11, rs1=10, rs2=0)
        hart.exec_instr(lr)
        # 再做 SC
        sc = _make_amo_instr(AmoFunct5.SC, AmoWidth.D, rd=13, rs1=10, rs2=12)
        hart.exec_instr(sc)
        assert hart.gprs[13].val == 0, "SC 成功应返回 0"

    def test_sc_fails_without_reservation(self, hart):
        """无预留时 SC 应失败, rd←非零."""
        hart.gprs[10].val = 0x1000
        hart.gprs[12].val = 0xDEADBEEF
        sc = _make_amo_instr(AmoFunct5.SC, AmoWidth.D, rd=13, rs1=10, rs2=12)
        hart.exec_instr(sc)
        assert hart.gprs[13].val != 0, "SC 失败应返回非零"

    def test_sc_clears_reservation(self, hart):
        """SC 执行后应清除预留 (无论成功与否)."""
        hart.gprs[10].val = 0x1000
        hart.gprs[12].val = 0x42
        lr = _make_amo_instr(AmoFunct5.LR, AmoWidth.D, rd=11, rs1=10, rs2=0)
        hart.exec_instr(lr)
        assert hart.reservation_valid
        sc = _make_amo_instr(AmoFunct5.SC, AmoWidth.D, rd=13, rs1=10, rs2=12)
        hart.exec_instr(sc)
        assert not hart.reservation_valid

    def test_lr_stores_memory_value(self, hart):
        """LR 应将内存值加载到 rd."""
        # 预先写内存
        val = 0xFEED_FACE_CAFE_BABE & 0xFFFF_FFFF_FFFF_FFFF
        hart._mem_write_phy = lambda a, d: None  # 占位
        ram, rf, wf = _make_ram()
        hart.set_memory_backend(rf, wf)
        data = val.to_bytes(8, "little")
        wf(0x2000, data)

        hart.gprs[10].val = 0x2000
        lr = _make_amo_instr(AmoFunct5.LR, AmoWidth.D, rd=11, rs1=10, rs2=0)
        hart.exec_instr(lr)
        assert hart.gprs[11].val == val & 0xFFFF_FFFF_FFFF_FFFF


class TestAMOArithmetic:
    """AMO 算术/逻辑操作."""

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        ram, rf, wf = _make_ram()
        h.set_memory_backend(rf, wf)
        return h

    def _setup_mem_and_regs(self, hart, addr: int, mem_val: int, op_val: int, is_64: bool = True):
        """准备内存和寄存器."""
        width = AmoWidth.D if is_64 else AmoWidth.W
        byte_len = 8 if is_64 else 4
        mask = 0xFFFF_FFFF_FFFF_FFFF if is_64 else 0xFFFF_FFFF
        hart.gprs[10].val = addr  # rs1 = addr
        hart.gprs[12].val = op_val & mask  # rs2 = operand
        data = mem_val.to_bytes(byte_len, "little")
        hart._mem_write_phy(addr, data)
        return width, mask

    def _do_amo(self, hart, op: AmoFunct5, addr: int, mem_val: int, op_val: int, expected_result: int,
                expected_rd: int, is_64: bool = True):
        width, mask = self._setup_mem_and_regs(hart, addr, mem_val, op_val, is_64)
        instr = _make_amo_instr(op, width, rd=15, rs1=10, rs2=12)
        hart.exec_instr(instr)
        # rd 应获得原始内存值
        actual_rd = hart.gprs[15].val & mask
        assert actual_rd == (expected_rd & mask), (
            f"{op.name}: rd expected {expected_rd:#x}, got {actual_rd:#x}"
        )
        # 内存应被更新为 result
        byte_len = 8 if is_64 else 4
        new_mem = int.from_bytes(hart._mem_read_phy(addr, byte_len), "little")
        assert new_mem == (expected_result & mask), (
            f"{op.name}: mem expected {expected_result:#x}, got {new_mem:#x}"
        )

    def test_amoswap_d(self, hart):
        self._do_amo(hart, AmoFunct5.SWAP, 0x1000, mem_val=0xAAAA, op_val=0xBBBB,
                     expected_result=0xBBBB, expected_rd=0xAAAA)

    def test_amoadd_d(self, hart):
        self._do_amo(hart, AmoFunct5.ADD, 0x1000, mem_val=10, op_val=3,
                     expected_result=13, expected_rd=10)

    def test_amoxor_d(self, hart):
        self._do_amo(hart, AmoFunct5.XOR, 0x1000, mem_val=0xFF00, op_val=0x0FF0,
                     expected_result=0xF0F0, expected_rd=0xFF00)

    def test_amoand_d(self, hart):
        self._do_amo(hart, AmoFunct5.AND, 0x1000, mem_val=0xFF0F, op_val=0xF0FF,
                     expected_result=0xF00F, expected_rd=0xFF0F)

    def test_amoor_d(self, hart):
        self._do_amo(hart, AmoFunct5.OR, 0x1000, mem_val=0xFF00, op_val=0x00FF,
                     expected_result=0xFFFF, expected_rd=0xFF00)

    def test_amomin_d(self, hart):
        """AMOMIN.D: mem ← min(mem, op_val) 有符号比较."""
        # mem=-5, op=10 → min = -5
        self._do_amo(hart, AmoFunct5.MIN, 0x1000,
                     mem_val=(-5) & 0xFFFF_FFFF_FFFF_FFFF, op_val=10,
                     expected_result=(-5) & 0xFFFF_FFFF_FFFF_FFFF,
                     expected_rd=(-5) & 0xFFFF_FFFF_FFFF_FFFF)

    def test_amomax_d(self, hart):
        """AMOMAX.D: mem ← max(mem, op_val) 有符号比较."""
        # mem=-5, op=10 → max = 10
        self._do_amo(hart, AmoFunct5.MAX, 0x1000,
                     mem_val=(-5) & 0xFFFF_FFFF_FFFF_FFFF, op_val=10,
                     expected_result=10,
                     expected_rd=(-5) & 0xFFFF_FFFF_FFFF_FFFF)

    def test_amominu_d(self, hart):
        self._do_amo(hart, AmoFunct5.MINU, 0x1000, mem_val=5, op_val=10,
                     expected_result=5, expected_rd=5)

    def test_amomaxu_d(self, hart):
        self._do_amo(hart, AmoFunct5.MAXU, 0x1000, mem_val=5, op_val=10,
                     expected_result=10, expected_rd=5)

    def test_amoswap_w(self, hart):
        self._do_amo(hart, AmoFunct5.SWAP, 0x2000, mem_val=0xAAAA, op_val=0xBBBB,
                     expected_result=0xBBBB, expected_rd=0xAAAA, is_64=False)

    def test_amoadd_w(self, hart):
        self._do_amo(hart, AmoFunct5.ADD, 0x2000, mem_val=7, op_val=8,
                     expected_result=15, expected_rd=7, is_64=False)


class TestReservationInvalidation:
    """预留失效场景."""

    def test_trap_clears_reservation(self):
        """trap 发生时应清除预留."""
        h = Hart(id=0)
        h.set_reservation(0x4000)
        assert h.reservation_valid
        h._take_trap(
            __import__("pyremu.core.trap", fromlist=["TrapType"]).TrapType.IllInstr,
            is_interrupt=False,
        )
        assert not h.reservation_valid
