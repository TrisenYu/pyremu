#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""Trap 处理测试: cause code 映射、ECALL/EBREAK/MRET/SRET 流程."""

import pytest

from pyremu.core.decoder import Hart, Opc
from pyremu.core.hart import (
    MSTATUS_MIE,
    MSTATUS_SIE,
    MSTATUS_SPIE,
    MSTATUS_SPP,
    RiscvMode,
)
from pyremu.core.trap import TrapType, trap_cause_code, trap_is_interrupt

# ============================================================
#  trap_cause_code / trap_is_interrupt
# ============================================================


class TestTrapCauseCode:
    """验证每个 TrapType 对应的 mcause 编码是否正确."""

    def test_exception_codes(self):
        """异常: bit 63 = 0, 编码 = 规范定义的编号."""
        cases = [
            (TrapType.InstrAddrMisaligned, 0),
            (TrapType.InstrAccessFault, 1),
            (TrapType.IllInstr, 2),
            (TrapType.Breakpoint, 3),
            (TrapType.LdAddrMisaligned, 4),
            (TrapType.LdAccessFault, 5),
            (TrapType.StAddrMisaligned, 6),
            (TrapType.StAccessFault, 7),
            (TrapType.EcallFromUmode, 8),
            (TrapType.EcallFromSmode, 9),
            (TrapType.EcallFromMmode, 11),
            (TrapType.InstrPageFault, 12),
            (TrapType.LdPageFault, 13),
            (TrapType.StPageFault, 15),
        ]
        for trap, expected in cases:
            code = trap_cause_code(trap)
            assert code == expected, (
                f"{trap.name}: expected mcause={expected}, got {code}"
            )
            assert not trap_is_interrupt(trap), (
                f"{trap.name} should NOT be an interrupt"
            )

    def test_interrupt_codes(self):
        """中断: bit 63 = 1, 低 63 位 = 中断编号."""
        cases = [
            (TrapType.UmodeSoftInterrupt, 0),
            (TrapType.SmodeSoftInterrupt, 1),
            (TrapType.MmodeSoftInterrupt, 3),
            (TrapType.UmodeTimerInterrupt, 4),
            (TrapType.SmodeTimerInterrupt, 5),
            (TrapType.MmodeTimerInterrupt, 7),
            (TrapType.UmodeExternInterrupt, 8),
            (TrapType.SmodeExternInterrupt, 9),
            (TrapType.MmodeExternInterrupt, 11),
        ]
        for trap, exc_code in cases:
            code = trap_cause_code(trap)
            expected = exc_code | (1 << 63)
            assert code == expected, (
                f"{trap.name}: expected mcause=0x{expected:016x}, got 0x{code:016x}"
            )
            assert trap_is_interrupt(trap), (
                f"{trap.name} SHOULD be an interrupt"
            )

    def test_all_trap_types_have_code(self):
        """确保所有 TrapType 成员都有对应的编码."""
        for trap in TrapType:
            code = trap_cause_code(trap)
            assert code != 0 or trap in (
                TrapType.InstrAddrMisaligned,
                TrapType.UmodeSoftInterrupt,
            ), f"{trap.name} returned unexpected zero code"


# ============================================================
#  Hart._take_trap 基础流程
# ============================================================


class TestTakeTrap:
    """验证 _take_trap 的寄存器保存/恢复流程."""

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        # 预设 mstatus 值: MIE=1, mode=M
        h.mstatus_val = MSTATUS_MIE
        return h

    def test_take_trap_saves_context(self, hart):
        """异常触发后 mepc/mcause/mtval/mstatus 应正确保存."""
        hart._take_trap(TrapType.IllInstr, tval=0xDEAD, is_interrupt=False)

        assert hart.mepc_val == 0x1000, "mepc 应保存触发异常的 PC"
        assert hart.mcause_val == 2, "mcause 应为 IllInstr 编码 2"
        assert hart.mtval_val == 0xDEAD, "mtval 应保存附加信息"
        # MIE=1 应被保存到 MPIE, MIE 清零
        assert hart.mpie is True, "进入 trap 前 MIE=1 应保存为 MPIE=1"
        assert hart.mie is False, "进入 trap 后 MIE 应为 0"
        assert hart.mpp == RiscvMode.M, "MPP 应保存先前的 M 模式"

    def test_take_trap_jumps_to_mtvec_direct(self, hart):
        """直接模式: PC 应跳转到 mtvec.BASE."""
        hart.csrs["mtvec"].val = 0xC0000000  # MODE=0 (direct)
        hart._take_trap(TrapType.Breakpoint, is_interrupt=False)
        assert hart.pc == 0xC0000000, "直接模式应跳转到 mtvec BASE"

    def test_take_trap_jumps_to_mtvec_vectored(self, hart):
        """向量模式 + 中断: PC = BASE + 4 * code."""
        hart.csrs["mtvec"].val = 0xC0000001  # MODE=1 (vectored)
        hart._take_trap(TrapType.SmodeTimerInterrupt, is_interrupt=True)
        # SmodeTimerInterrupt code = 5
        expected_pc = 0xC0000000 + 4 * 5
        assert hart.pc == expected_pc, (
            f"向量模式应跳转到 BASE+4*5 = 0x{expected_pc:x}"
        )

    def test_take_trap_vectored_ignored_for_exception(self, hart):
        """向量模式对异常不生效, 仍使用直接模式."""
        hart.csrs["mtvec"].val = 0xC0000001
        hart._take_trap(TrapType.IllInstr, is_interrupt=False)
        assert hart.pc == 0xC0000000, "异常在向量模式下仍应使用直接模式"

    def test_take_trap_switches_to_m_mode(self, hart):
        """trap 后应切换到 M 模式."""
        hart.mode = RiscvMode.U
        hart._take_trap(TrapType.EcallFromUmode, is_interrupt=False)
        assert hart.mode == RiscvMode.M, "trap 后应进入 M 模式"


# ============================================================
#  ECALL / EBREAK
# ============================================================


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
        hart._trap_ecall()
        assert hart.mcause_val == 11, "M 模式 ecall → mcause=11"
        assert hart.mepc_val == 0x2000

    def test_ecall_from_s_mode(self, hart):
        hart.mode = RiscvMode.S
        hart._trap_ecall()
        assert hart.mcause_val == 9, "S 模式 ecall → mcause=9"

    def test_ecall_from_u_mode(self, hart):
        hart.mode = RiscvMode.U
        hart._trap_ecall()
        assert hart.mcause_val == 8, "U 模式 ecall → mcause=8"

    def test_ebreak(self, hart):
        hart._trap_ebreak()
        assert hart.mcause_val == 3, "ebreak → mcause=3"
        assert hart.mtval_val == 0x2000, "mtval 应保存断点地址"


# ============================================================
#  MRET / SRET
# ============================================================


class TestMretSret:
    """验证从 trap 返回 (MRET/SRET) 的上下文恢复."""

    @pytest.fixture
    def hart_after_m_trap(self) -> Hart:
        """模拟: 从 M 模式进入 trap 后的 hart 状态."""
        h = Hart(id=0)
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        h.mode = RiscvMode.M
        h.mstatus_val = MSTATUS_MIE  # 初始 MIE=1
        h._take_trap(TrapType.IllInstr, is_interrupt=False)
        # 现在 hart 在 trap handler 中, MPP=M, MPIE=1, MIE=0
        return h

    def test_mret_restores_pc_and_mode(self, hart_after_m_trap):
        h = hart_after_m_trap
        assert h.mepc_val == 0x1000
        h._trap_mret()
        assert h.pc == 0x1000, "MRET 应恢复到 mepc"
        assert h.mode == RiscvMode.M, "MPP=M 应恢复到 M 模式"

    def test_mret_restores_mie_from_mpie(self, hart_after_m_trap):
        h = hart_after_m_trap
        h._trap_mret()
        assert h.mie is True, "MPIE=1 → MIE 应恢复为 1"

    def test_mret_clears_mpp_to_u(self, hart_after_m_trap):
        h = hart_after_m_trap
        h._trap_mret()
        assert h.mpp == RiscvMode.U, "MRET 后 MPP 应重置为 U"

    def test_mret_preserves_mpie(self, hart_after_m_trap):
        h = hart_after_m_trap
        h._trap_mret()
        assert h.mpie is True, "MRET 后 MPIE 应为 1"

    @pytest.fixture
    def hart_after_s_trap(self) -> Hart:
        """模拟: 从 S 模式进入 S 级 trap 后的 hart 状态."""
        h = Hart(id=0)
        h.csrs["stvec"].val = 0x80004000
        h.pc = 0x3000
        h.mode = RiscvMode.S
        # 设置 S 模式 trap 需要手动模拟 (因为当前 _take_trap 只支持 M 级别)
        h.mstatus_val = MSTATUS_SIE
        h.sepc_val = 0x3000
        h.scause_val = 9  # EcallFromSmode
        # 模拟 _take_trap 对 S 级的操作: SPIE←SIE, SIE←0, SPP←S
        mstatus = h.mstatus_val
        if mstatus & MSTATUS_SIE:
            mstatus |= MSTATUS_SPIE
        else:
            mstatus &= ~MSTATUS_SPIE
        mstatus &= ~MSTATUS_SIE
        mstatus |= MSTATUS_SPP  # SPP←1 (S mode)
        h.mstatus_val = mstatus
        h.mode = RiscvMode.M  # trap 进入 M 模式 (简化)
        return h

    def test_sret_restores_pc_and_mode(self, hart_after_s_trap):
        h = hart_after_s_trap
        h._trap_sret()
        assert h.pc == 0x3000, "SRET 应恢复到 sepc"
        assert h.mode == RiscvMode.S, "SPP=S → 应恢复到 S 模式"

    def test_sret_restores_sie_from_spie(self, hart_after_s_trap):
        h = hart_after_s_trap
        h._trap_sret()
        assert h.sie is True, "SPIE=1 → SIE 应恢复为 1"

    def test_sret_clears_spp_to_u(self, hart_after_s_trap):
        h = hart_after_s_trap
        h._trap_sret()
        assert h.spp == RiscvMode.U, "SRET 后 SPP 应重置为 U"


# ============================================================
#  SFENCE.VMA (TLB flush via handle_sys)
# ============================================================


class TestSfenceVma:
    """验证 SFENCE.VMA 指令的 TLB 刷新."""

    def test_sfence_vma_flushes_tlbs(self):
        h = Hart(id=0)
        h.itlb.insert(vpn=0x100, ppn=0x200, perm=0xF)
        h.dtlb.insert(vpn=0x300, ppn=0x400, perm=0xF)
        assert len(h.itlb) == 1
        assert len(h.dtlb) == 1

        # 通过 exec_instr 执行 SFENCE.VMA
        # SFENCE.VMA: imm[31:20]=0x104, func3=000, opcode=Op.sys
        instr_val = (0x104 << 20) | Opc.sys.value
        h.exec_instr(instr_val)

        assert len(h.itlb) == 0, "SFENCE.VMA 应刷新 itlb"
        assert len(h.dtlb) == 0, "SFENCE.VMA 应刷新 dtlb"
