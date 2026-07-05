#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""Trap 处理测试: cause code 映射、ECALL/EBREAK/MRET/SRET 流程."""

import pytest

from pyremu.core.decoder import Hart, Opc
from pyremu.core.hart import (
    MSTATUS_MIE,
    MSTATUS_MPIE,
    MSTATUS_MPP,
    MSTATUS_SIE,
    MSTATUS_SPP,
    MSTATUS_TW,
    RiscvMode,
)
from pyremu.core.mem_check_aux import inject_memory_backend, mem_read, mem_write
from pyremu.core.trap import TrapType, trap_cause_code, trap_is_interrupt
from pyremu.core.trap_handler import (
    check_pending_interrupts,
    deliver_trap,
    trap_ebreak,
    trap_ecall,
    trap_mret,
    trap_sret,
)
from pyremu.emulator import Emulator
from pyremu.interrupt.clint import CLINT
from pyremu.memory.bus import Bus
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils.parse_bin import parse_firmware

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
            assert code == expected, f"{trap.name}: expected mcause={expected}, got {code}"
            assert not trap_is_interrupt(trap), f"{trap.name} should NOT be an interrupt"

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
            assert trap_is_interrupt(trap), f"{trap.name} SHOULD be an interrupt"

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
        deliver_trap(hart, TrapType.IllInstr, tval=0xDEAD, is_interrupt=False)

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
        deliver_trap(hart, TrapType.Breakpoint, is_interrupt=False)
        assert hart.pc == 0xC0000000, "直接模式应跳转到 mtvec BASE"

    def test_take_trap_jumps_to_mtvec_vectored(self, hart):
        """向量模式 + 中断: PC = BASE + 4 * code."""
        hart.csrs["mtvec"].val = 0xC0000001  # MODE=1 (vectored)
        deliver_trap(hart, TrapType.SmodeTimerInterrupt, is_interrupt=True)
        # SmodeTimerInterrupt code = 5
        expected_pc = 0xC0000000 + 4 * 5
        assert hart.pc == expected_pc, f"向量模式应跳转到 BASE+4*5 = 0x{expected_pc:x}"

    def test_take_trap_vectored_ignored_for_exception(self, hart):
        """向量模式对异常不生效, 仍使用直接模式."""
        hart.csrs["mtvec"].val = 0xC0000001
        deliver_trap(hart, TrapType.IllInstr, is_interrupt=False)
        assert hart.pc == 0xC0000000, "异常在向量模式下仍应使用直接模式"

    def test_take_trap_switches_to_m_mode(self, hart):
        """trap 后应切换到 M 模式."""
        hart.mode = RiscvMode.U
        deliver_trap(hart, TrapType.EcallFromUmode, is_interrupt=False)
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
        deliver_trap(h, TrapType.IllInstr, is_interrupt=False)
        # 现在 hart 在 trap handler 中, MPP=M, MPIE=1, MIE=0
        return h

    def test_mret_restores_pc_and_mode(self, hart_after_m_trap):
        h = hart_after_m_trap
        assert h.mepc_val == 0x1000
        trap_mret(h)
        assert h.pc == 0x1000, "MRET 应恢复到 mepc"
        assert h.mode == RiscvMode.M, "MPP=M 应恢复到 M 模式"

    def test_mret_restores_mie_from_mpie(self, hart_after_m_trap):
        h = hart_after_m_trap
        trap_mret(h)
        assert h.mie is True, "MPIE=1 -> MIE 应恢复为 1"

    def test_mret_clears_mpp_to_u(self, hart_after_m_trap):
        h = hart_after_m_trap
        trap_mret(h)
        assert h.mpp == RiscvMode.U, "MRET 后 MPP 应重置为 U"

    def test_mret_preserves_mpie(self, hart_after_m_trap):
        h = hart_after_m_trap
        trap_mret(h)
        assert h.mpie is True, "MRET 后 MPIE 应为 1"

    @pytest.fixture
    def hart_after_s_trap(self) -> Hart:
        """通过 medeleg 委派获得 S 模式 trap 后的状态."""
        h = Hart(id=0)
        h.csrs["stvec"].val = 0x80004000
        h.pc = 0x3000
        h.mode = RiscvMode.S
        h.mstatus_val = MSTATUS_SIE  # SIE=1 before trap
        # 委派 S 模式 ecall: medeleg[9]=1 -> trap 留在 S 模式
        h.csrs["medeleg"].val = 1 << 9
        deliver_trap(h, TrapType.EcallFromSmode, tval=0, is_interrupt=False)
        return h

    def test_sret_restores_pc_and_mode(self, hart_after_s_trap):
        h = hart_after_s_trap
        trap_sret(h)
        assert h.pc == 0x3000, "SRET 应恢复到 sepc"
        assert h.mode == RiscvMode.S, "SPP=S -> 应恢复到 S 模式"

    def test_sret_restores_sie_from_spie(self, hart_after_s_trap):
        h = hart_after_s_trap
        trap_sret(h)
        assert h.sie is True, "SPIE=1 -> SIE 应恢复为 1"

    def test_sret_clears_spp_to_u(self, hart_after_s_trap):
        h = hart_after_s_trap
        trap_sret(h)
        assert h.spp == RiscvMode.U, "SRET 后 SPP 应重置为 U"


# ============================================================
#  _take_trap 方法解析 (父类 stub vs 子类实现)
# ============================================================


class TestTakeTrapResolution:
    """验证 Hart (继承 TrapHandler mixin) 的 _take_trap 实现."""

    def test_trap_handler_mixin_provides_implementation(self):
        """Hart 通过 TrapHandler mixin 获得 _take_trap 实现,
        而非 HartWithRegs 的 stub (已移除)."""
        h = Hart(id=0)
        h.csrs["mtvec"].val = 0x80000000
        h.mstatus_val = 0
        # _take_trap 由 TrapHandler mixin 提供, 不应抛 NotImplementedError
        try:
            deliver_trap(h, TrapType.IllInstr, tval=0x42, is_interrupt=False)
        except NotImplementedError:
            pytest.fail("Hart._take_trap 应由 TrapHandler mixin 提供实现")
        assert h.mcause_val == 2
        assert h.mtval_val == 0x42

    def test_check_pending_uses_overridden_method(self):
        """check_pending_interrupts 定义在 Hart, 正确调用覆写后的 _take_trap."""
        h = Hart(id=0)
        clint = CLINT(num_harts=1)
        h.interrupt_ctrl = clint
        h.mie = True
        h.csrs["mie"].val = 1 << 3  # MSIP enable
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x2000

        clint.send_ipi(0)
        interrupted = check_pending_interrupts(h)
        assert interrupted
        # MSI: bit63=1, code=3
        assert h.mcause_val == (1 << 63) | 3


# ============================================================
#  Trap 委派 (medeleg / mideleg)
# ============================================================


class TestTrapDelegation:
    """验证 medeleg/mideleg 将异常/中断委派到 S 模式的流程.

    RISC-V Privileged Spec §3.1.9:
    - M 模式下发生的 trap 永不委派
    - 委派时跳转 stvec、写入 sepc/scause/stval、更新 SPP/SPIE/SIE
    """

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        h.csrs["mtvec"].val = 0x80000000
        h.csrs["stvec"].val = 0x80004000
        h.pc = 0x1000
        h.mstatus_val = MSTATUS_MIE
        h.mode = RiscvMode.M
        return h

    # ---- Exception delegation ----

    def test_medeleg_ecall_u_to_s(self, hart):
        """medeleg[8]=1: U 模式 ecall 委派到 S 模式."""
        hart.csrs["medeleg"].val = 1 << 8
        hart.mode = RiscvMode.U
        deliver_trap(hart, TrapType.EcallFromUmode, tval=0, is_interrupt=False)

        assert hart.pc == 0x80004000, "委派后应跳转到 stvec"
        assert hart.mode == RiscvMode.S, "应进入 S 模式"
        assert hart.sepc_val == 0x1000, "sepc 应保存 PC"
        assert hart.scause_val == 8, "scause 应为 8"
        assert hart.mcause_val == 0, "不应修改 mcause"

    def test_medeleg_not_from_m_mode(self, hart):
        """即使 medeleg 已设置, M 模式下永不委派."""
        hart.csrs["medeleg"].val = 1 << 8
        hart.mode = RiscvMode.M
        deliver_trap(hart, TrapType.EcallFromUmode, tval=0, is_interrupt=False)

        assert hart.pc == 0x80000000, "应走 mtvec"
        assert hart.mode == RiscvMode.M
        assert hart.mepc_val == 0x1000
        assert hart.mcause_val == 8

    def test_medeleg_ill_instr_to_s(self, hart):
        """medeleg[2]=1: 非法指令异常委派到 S 模式."""
        hart.csrs["medeleg"].val = 1 << 2
        hart.mode = RiscvMode.U
        deliver_trap(hart, TrapType.IllInstr, tval=0xBAD, is_interrupt=False)

        assert hart.pc == 0x80004000
        assert hart.mode == RiscvMode.S
        assert hart.scause_val == 2
        assert hart.stval_val == 0xBAD

    # ---- Interrupt delegation ----

    def test_mideleg_timer_to_s(self, hart):
        """mideleg[5]=1: 定时器中断委派到 S 模式."""
        hart.csrs["mideleg"].val = 1 << 5
        hart.mode = RiscvMode.U
        deliver_trap(hart, TrapType.SmodeTimerInterrupt, tval=0, is_interrupt=True)

        assert hart.pc == 0x80004000
        assert hart.mode == RiscvMode.S
        assert hart.sepc_val == 0x1000
        expected_scause = (1 << 63) | 5
        assert hart.scause_val == expected_scause, (
            f"scause 应为: bit63=1 code=5, 实际 {hart.scause_val:#x}"
        )

    def test_mideleg_not_set_goes_to_m(self, hart):
        """mideleg=0: 中断照旧进入 M 模式."""
        hart.mode = RiscvMode.U
        deliver_trap(hart, TrapType.SmodeTimerInterrupt, tval=0, is_interrupt=True)

        assert hart.pc == 0x80000000
        assert hart.mode == RiscvMode.M
        assert hart.mepc_val == 0x1000

    # ---- mstatus S-level fields ----

    def test_delegated_trap_saves_spp(self, hart):
        """委派 trap 应将当前模式编码为 SPP."""
        hart.csrs["medeleg"].val = 1 << 8
        hart.mode = RiscvMode.U
        deliver_trap(hart, TrapType.EcallFromUmode, is_interrupt=False)
        assert hart.spp == RiscvMode.U

    def test_delegated_trap_clears_sie_sets_spie(self, hart):
        """委派 trap: SIE -> SPIE, SIE ← 0."""
        hart.csrs["medeleg"].val = 1 << 8
        hart.mode = RiscvMode.U
        hart.mstatus_val = MSTATUS_SIE
        deliver_trap(hart, TrapType.EcallFromUmode, is_interrupt=False)

        assert hart.sie is False, "进入 S 模式 trap 后 SIE 应为 0"
        assert hart.spie is True, "SPIE 应保存先前的 SIE=1"

    def test_delegated_trap_spie_zero_when_sie_zero(self, hart):
        """委派 trap 前 SIE=0: SPIE 亦记录为 0."""
        hart.csrs["medeleg"].val = 1 << 8
        hart.mode = RiscvMode.U
        hart.mstatus_val = 0  # SIE=0
        deliver_trap(hart, TrapType.EcallFromUmode, is_interrupt=False)

        assert hart.spie is False

    # ---- SRET after delegated trap ----

    def test_sret_after_delegated_trap(self, hart):
        """委派 trap -> SRET 应恢复到原模式."""
        hart.csrs["medeleg"].val = 1 << 8
        hart.mode = RiscvMode.U
        hart.pc = 0x4000
        hart.mstatus_val = MSTATUS_SIE  # SIE=1
        deliver_trap(hart, TrapType.EcallFromUmode, is_interrupt=False)

        trap_sret(hart)
        assert hart.pc == 0x4000
        assert hart.mode == RiscvMode.U
        assert hart.sie is True, "SRET 应恢复 SIE"

    # ---- Vectored stvec ----

    def test_delegated_vector_stvec(self, hart):
        """委派中断 + stvec MODE=1: PC = BASE + 4*code."""
        hart.csrs["stvec"].val = 0xC0001001
        hart.csrs["mideleg"].val = 1 << 7  # delegate MTI (code=7)
        hart.mode = RiscvMode.U
        deliver_trap(hart, TrapType.MmodeTimerInterrupt, is_interrupt=True)

        expected = 0xC0001000 + 4 * 7
        assert hart.pc == expected, f"应为 0x{expected:x}, 实际 0x{hart.pc:x}"

    def test_delegated_vector_ignored_for_exception(self, hart):
        """委派异常在 stvec 向量模式下仍使用 BASE."""
        hart.csrs["stvec"].val = 0xC0001001
        hart.csrs["medeleg"].val = 1 << 8
        hart.mode = RiscvMode.U
        deliver_trap(hart, TrapType.EcallFromUmode, is_interrupt=False)

        assert hart.pc == 0xC0001000, "异常不使用向量偏移"

    # ---- check_pending_interrupts + CLINT + delegation ----

    def test_check_pending_msip_delegated_to_s(self, hart):
        """CLINT MSIP + mideleg[3]=1 -> 中断进入 S 模式."""
        clint = CLINT(num_harts=1)
        hart.interrupt_ctrl = clint
        hart.mode = RiscvMode.U
        hart.mie = True
        hart.csrs["mie"].val = 1 << 3  # MSIE
        hart.csrs["mideleg"].val = 1 << 3  # delegate MSI
        hart.csrs["stvec"].val = 0x80004000
        hart.pc = 0x2000

        clint.send_ipi(0)
        interrupted = check_pending_interrupts(hart)
        assert interrupted
        assert hart.mode == RiscvMode.S, "委派后应进入 S 模式"
        assert hart.pc == 0x80004000  # stvec
        assert hart.sepc_val == 0x2000

    def test_check_pending_not_delegated_stays_m(self, hart):
        """mideleg=0 -> MSIP 照旧进入 M 模式."""
        clint = CLINT(num_harts=1)
        hart.interrupt_ctrl = clint
        hart.mode = RiscvMode.U
        hart.mie = True
        hart.csrs["mie"].val = 1 << 3
        # mideleg = 0
        hart.pc = 0x2000

        clint.send_ipi(0)
        interrupted = check_pending_interrupts(hart)
        assert interrupted
        assert hart.mode == RiscvMode.M, "非委派应进入 M 模式"
        assert hart.mepc_val == 0x2000

    def test_s_mode_sie_zero_blocks_delegated_interrupt(self, hart):
        """S 模式 + SIE=0: 已委派中断被阻塞; 非委派中断仍可抢占."""
        clint = CLINT(num_harts=1)
        hart.interrupt_ctrl = clint
        hart.mode = RiscvMode.S
        hart.sie = False  # S 级全局关中断
        hart.mie = True  # M 级全局开中断 (允许非委派抢占)
        hart.csrs["mie"].val = 1 << 3  # MSIE
        hart.csrs["mideleg"].val = 1 << 3  # delegate MSI -> S

        clint.send_ipi(0)
        interrupted = check_pending_interrupts(hart)
        # 已委派 + SIE=0 -> 不应触发
        assert not interrupted, "SIE=0 应阻塞已委派中断"

    def test_s_mode_sie_zero_allows_non_delegated(self, hart):
        """S 模式 + SIE=0: 非委派 M 级中断仍可抢占."""
        clint = CLINT(num_harts=1)
        hart.interrupt_ctrl = clint
        hart.mode = RiscvMode.S
        hart.sie = False
        hart.mie = True
        hart.csrs["mie"].val = 1 << 3  # MSIE
        # mideleg = 0 -> MSI stays M-level, preempts S
        hart.csrs["mtvec"].val = 0x80000000

        clint.send_ipi(0)
        interrupted = check_pending_interrupts(hart)
        assert interrupted, "非委派 M 级中断应能抢占 S 模式"
        assert hart.mode == RiscvMode.M


class TestUmodePrivilegedInstructionTraps:
    """U 模式执行特权指令/CSR -> 陷态, medeleg 委派到 S 模式."""

    @pytest.fixture
    def h(self) -> Hart:
        bus = Bus(ram_size=0x10000, ram_base=0x80000000)
        hart = Hart(id=0)
        inject_memory_backend(hart, bus.read, bus.write)
        hart.bus = bus
        hart.pc = 0x80000000
        hart.csrs["mtvec"].val = 0x80001000
        hart.csrs["stvec"].val = 0x80002000
        hart.mstatus_val = MSTATUS_MIE
        return hart

    # -- helpers --

    @staticmethod
    def _ecall() -> int:
        return 0x73  # ecall

    @staticmethod
    def _csrrw(rd: int, csr: int, rs1: int) -> int:
        return (csr << 20) | (rs1 << 15) | (0b001 << 12) | (rd << 7) | 0b1110011

    @staticmethod
    def _wfi() -> int:
        return 0x10500073

    # -- U -> S delegation of ECALL --

    def test_u_ecall_delegated_to_s(self, h):
        """U 模式 ecall -> medeleg 委派 -> S 模式 trap."""
        h.csrs["medeleg"].val = 1 << 8
        h.mode = RiscvMode.U
        h.exec_instr(self._ecall())
        assert h.mode == RiscvMode.S
        assert h.pc == 0x80002000, f"应跳转 stvec, 实际 PC={h.pc:#x}"
        assert h.scause_val == 8, f"scause 应为 8, 实际 {h.scause_val:#x}"

    def test_u_ecall_no_delegation_stays_m(self, h):
        """未委派: U 模式 ecall -> M 模式 trap."""
        h.mode = RiscvMode.U
        h.exec_instr(self._ecall())
        assert h.mode == RiscvMode.M
        assert h.pc == 0x80001000

    # -- U -> S delegation of CSR access --

    def test_u_csr_read_mstatus_traps(self, h):
        """U 模式读 mstatus (M-mode CSR) -> IllInstr -> 委派到 S."""
        h.csrs["medeleg"].val = 1 << 2  # 委派 IllInstr
        h.mode = RiscvMode.U
        instr = self._csrrw(5, 0x300, 0)  # csrrw t0, mstatus, x0
        h.exec_instr(instr)
        assert h.scause_val == 2, f"IllInstr=2, 实际 scause={h.scause_val:#x}"
        assert h.mode == RiscvMode.S

    def test_u_csr_write_satp_traps(self, h):
        """U 模式写 satp (S-mode CSR) -> 也应陷态 (U 模式不能直接写 S CSR)."""
        h.csrs["medeleg"].val = 1 << 2
        h.mode = RiscvMode.U
        instr = self._csrrw(0, 0x180, 5)  # csrrw x0, satp, t0
        h.exec_instr(instr)
        assert h.scause_val == 2
        assert h.mode == RiscvMode.S

    # -- U -> S delegation of WFI (mstatus.TW) --

    def test_u_wfi_with_tw_traps(self, h):
        """mstatus.TW=1 + U 模式 WFI -> IllInstr -> 委派到 S."""
        h.csrs["medeleg"].val = 1 << 2
        h.mode = RiscvMode.U
        h.mstatus_val = MSTATUS_TW  # TW=1
        h.exec_instr(self._wfi())
        assert h.scause_val == 2
        assert h.mode == RiscvMode.S

    # -- U -> S delegation of MRET --

    def test_u_mret_traps(self, h):
        """U 模式 mret -> IllInstr (U 模式不可执行 mret)."""
        h.csrs["medeleg"].val = 1 << 2
        h.mode = RiscvMode.U
        h.exec_instr(0x30200073)  # mret
        assert h.scause_val == 2, f"应为 IllInstr, scause={h.scause_val:#x}"
        assert h.mode == RiscvMode.S

    # -- 未委派时进入 M 模式 --

    def test_u_ill_instr_no_delegation_stays_m(self, h):
        """medeleg=0: U 模式 mret -> M 模式 trap."""
        h.mode = RiscvMode.U
        h.exec_instr(0x30200073)  # mret
        assert h.mode == RiscvMode.M
        assert h.pc == 0x80001000, f"应跳转 mtvec, PC={h.pc:#x}"


# ============================================================
#  SFENCE.VMA (TLB flush via handle_sys)
# ============================================================


class TestSfenceVma:
    """验证 SFENCE.VMA 指令的 TLB 刷新."""

    def test_sfence_vma_flushes_tlbs(self):
        h = Hart(id=0)
        h.itlb.insert(vpn=0x100, ppn=0x200, perm=0xF)
        h.dtlb.insert(vpn=0x300, ppn=0x400, perm=0xF)
        assert len(h.itlb) == 1 and len(h.dtlb) == 1

        # SFENCE.VMA 正确编码:
        #   funct7=0b0001001 (bits 31:25), rs2=0, rs1=0
        #   -> imm[31:20] = (0b0001001 << 5) | 0 = 0x120
        instr_val = (0x120 << 20) | Opc.sys.value
        h.exec_instr(instr_val)

        assert len(h.itlb) == 0, "SFENCE.VMA 应刷新 itlb"
        assert len(h.dtlb) == 0, "SFENCE.VMA 应刷新 dtlb"

    def test_sfence_vma_rejects_wrong_funct12(self):
        """错误的 funct12 编码 (如旧的 0x104) 应触发 IllInstr (mcause=2)."""

        h = Hart(id=0)
        # 0x104 是旧代码中错误的 funct12 — 不是合法的特权指令编码
        instr_val = (0x104 << 20) | Opc.sys.value
        h.exec_instr(instr_val)
        assert h.mcause_val == 2, (
            f"0x104 不是合法的 SFENCE.VMA 编码, 应触发 IllInstr (mcause=2),"
            f" 实际 mc={h.mcause_val}"
        )


# ============================================================
#  CSRRW rd==rs1 读写竞争 (regression)
# ============================================================


class TestCsrrwRdRs1:
    """csrrw rd, csr, rs1 在 rd==rs1 时必须先读 rs1 再写 rd."""

    def test_csrrw_rd_eq_rs1_preserves_new_value(self):
        """csrrw sp, sscratch, sp 应正确交换 sp 和 sscratch."""

        h = Hart(id=0)
        # 用 mscratch (M-mode 可访问) 代替 sscratch 测试
        old_sp = 0x80101000
        old_scratch = 0x8000A000
        h.gprs[2] = old_sp
        h.write_csr(0x340, old_scratch)  # mscratch = old_scratch

        # 执行 csrrw sp, mscratch, sp
        instr_val = (0x340 << 20) | (2 << 15) | (1 << 12) | (2 << 7) | Opc.sys.value
        h.exec_instr(instr_val)

        assert h.gprs[2] == old_scratch, (
            f"csrrw 后 sp 应为旧 mscratch 值 0x{old_scratch:08x}, 实际 0x{h.gprs[2]:08x}"
        )
        assert h.csrs["mscratch"].val == old_sp, (
            f"csrrw 后 mscratch 应为旧 sp 值 0x{old_sp:08x}, "
            f"实际 0x{h.csrs['mscratch'].val:08x}"
        )

    def test_csrrw_rd_neq_rs1_still_works(self):
        """csrrw t0, mscratch, sp 在 rd!=rs1 时仍应正常工作."""

        h = Hart(id=0)
        old_sp = 0x80101000
        old_scratch = 0x8000A000
        h.gprs[2] = old_sp
        h.write_csr(0x340, old_scratch)

        # csrrw t0, mscratch, sp  (rd=5, rs1=2)
        instr_val = (0x340 << 20) | (2 << 15) | (1 << 12) | (5 << 7) | Opc.sys.value
        h.exec_instr(instr_val)

        assert h.gprs[5] == old_scratch, "rd(t0) 应为旧 CSR 值"
        assert h.csrs["mscratch"].val == old_sp, "CSR 应写入 rs1(sp) 的原始值"

    def test_csrrs_rd_eq_rs1_preserves_rs1(self):
        """csrrs t0, mscratch, t0: rd==rs1 时 SET 位应使用原始 rs1 值."""

        h = Hart(id=0)
        old_csr = 0x00000000
        rs1_orig = 0x0000000F  # t0 = 0xF (要 SET 的位)
        h.gprs[5] = rs1_orig  # t0
        h.write_csr(0x340, old_csr)

        # csrrs t0, mscratch, t0  (rd=5, rs1=5)
        instr_val = (0x340 << 20) | (5 << 15) | (2 << 12) | (5 << 7) | Opc.sys.value
        h.exec_instr(instr_val)

        assert h.gprs[5] == old_csr, f"rd 应为旧 CSR 值 0, 实际 0x{h.gprs[5]:x}"
        assert h.csrs["mscratch"].val == rs1_orig, (
            f"CSR 应为 old|rs1=0x{rs1_orig:x}, 实际 0x{h.csrs['mscratch'].val:x}"
        )

    def test_csrrc_rd_eq_rs1_preserves_rs1(self):
        """csrrc t0, mscratch, t0: rd==rs1 时 CLEAR 位应使用原始 rs1 值."""

        h = Hart(id=0)
        old_csr = 0x000000FF
        rs1_orig = 0x0000000F  # t0 = 0xF (要 CLEAR 的位)
        h.gprs[5] = rs1_orig  # t0
        h.write_csr(0x340, old_csr)

        # csrrc t0, mscratch, t0  (rd=5, rs1=5)
        instr_val = (0x340 << 20) | (5 << 15) | (3 << 12) | (5 << 7) | Opc.sys.value
        h.exec_instr(instr_val)

        expected_csr = old_csr & ~rs1_orig  # 0xFF & ~0x0F = 0xF0
        assert h.gprs[5] == old_csr, f"rd 应为旧 CSR 值 0xFF, 实际 0x{h.gprs[5]:x}"
        assert h.csrs["mscratch"].val == expected_csr, (
            f"CSR 应为 0x{expected_csr:x}, 实际 0x{h.csrs['mscratch'].val:x}"
        )


# ============================================================
#  CSR 特权级访问控制
# ============================================================


class TestCsrPrivilege:
    """低特权级访问高特权 CSR 应触发 IllInstr 陷态."""

    SYS = Opc.sys.value  # 0x73

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        return h

    # ---- 辅助: 构建 CSR 指令编码 ----

    @staticmethod
    def _csrrw(rd: int, rs1: int, csr: int) -> int:
        """CSRRW rd, csr, rs1."""
        return (csr << 20) | (rs1 << 15) | (1 << 12) | (rd << 7) | Opc.sys.value

    @staticmethod
    def _csrrwi(rd: int, uimm: int, csr: int) -> int:
        """CSRRWI rd, csr, uimm."""
        return (csr << 20) | (uimm << 15) | (5 << 12) | (rd << 7) | Opc.sys.value

    @staticmethod
    def _csrrs(rd: int, rs1: int, csr: int) -> int:
        """CSRRS rd, csr, rs1."""
        return (csr << 20) | (rs1 << 15) | (2 << 12) | (rd << 7) | Opc.sys.value

    @staticmethod
    def _csrrc(rd: int, rs1: int, csr: int) -> int:
        """CSRRC rd, csr, rs1."""
        return (csr << 20) | (rs1 << 15) | (3 << 12) | (rd << 7) | Opc.sys.value

    # ---- U 模式访问高特权 CSR ----

    def test_umode_read_mstatus_traps(self, hart):
        """U 模式 CSRRW 读取 mstatus -> IllInstr."""
        hart.mode = RiscvMode.U
        # CSRRW x5, mstatus, x0  (只读)
        instr = self._csrrw(rd=5, rs1=0, csr=0x300)
        # exec_instr 不抛异常, 通过 _take_trap 注入
        hart.exec_instr(instr)
        assert hart.mcause_val == 2, "应触发 IllInstr"
        assert hart.mepc_val == 0x1000

    def test_umode_write_mstatus_traps(self, hart):
        """U 模式 CSRRW 写入 mstatus -> IllInstr."""
        hart.mode = RiscvMode.U
        # CSRRW x0, mstatus, x10
        instr = self._csrrw(rd=0, rs1=10, csr=0x300)
        hart.gprs[10] = 0xDEAD
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_umode_csrrwi_mepc_traps(self, hart):
        """U 模式 CSRRWI 写 mepc -> IllInstr."""
        hart.mode = RiscvMode.U
        instr = self._csrrwi(rd=0, uimm=7, csr=0x341)  # mepc
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_umode_csrrs_mie_traps(self, hart):
        """U 模式 CSRRS mie -> IllInstr."""
        hart.mode = RiscvMode.U
        instr = self._csrrs(rd=5, rs1=0, csr=0x304)  # mie
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    # ---- S 模式访问 M 模式 CSR ----

    def test_smode_read_mstatus_traps(self, hart):
        """S 模式访问 M-only CSR -> IllInstr."""
        hart.mode = RiscvMode.S
        instr = self._csrrw(rd=5, rs1=0, csr=0x300)  # mstatus
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_smode_write_mtvec_traps(self, hart):
        """S 模式写入 mtvec (M-only) -> IllInstr."""
        hart.mode = RiscvMode.S
        instr = self._csrrw(rd=0, rs1=10, csr=0x305)  # mtvec
        hart.gprs[10] = 0x8888
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_smode_can_access_stvec(self, hart):
        """S 模式可以访问 S-mode CSR stvec."""
        hart.mode = RiscvMode.S
        hart.csrs["stvec"].val = 0xABCD0000
        # CSRRW x5, stvec, x0 (只读)
        instr = self._csrrw(rd=5, rs1=0, csr=0x105)  # stvec
        advance = hart.exec_instr(instr)
        assert advance == 4, "合法 CSR 应正常推进 PC"
        assert hart.gprs[5] == 0xABCD0000

    # ---- 写入只读 CSR ----

    def test_umode_write_readonly_csr_traps(self, hart):
        """U 模式写入只读 CSR (cycle) -> IllInstr."""
        hart.mode = RiscvMode.U
        # CSRRW x0, cycle, x10
        instr = self._csrrw(rd=0, rs1=10, csr=0xC00)  # cycle (u_ro)
        hart.gprs[10] = 42
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_mmode_write_readonly_csr_traps(self, hart):
        """M 模式写入只读 CSR (mvendorid) -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=0, rs1=10, csr=0xF11)  # mvendorid (m_ro)
        hart.gprs[10] = 99
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    # ---- CSRRW rs1=x0 写只读 CSR (CSRRW 始终为写操作, 即使 rs1=x0) ----

    def test_csrrw_rs1_zero_to_readonly_traps(self, hart):
        """CSRRW x0, cycle, x0: rs1=x0 仍写入 CSR -> IllInstr.

        RISC-V spec: CSRRW 始终是写操作, 即使 rs1=x0 也会把 0 写入 CSR.
        写入只读 CSR 必然触发非法指令陷态. 此处复现用户调试器中的
        mtval=0xc0001073 IllInstr 场景."""
        hart.mode = RiscvMode.M
        # csrrw x0, cycle, x0 — funct3=001(CSRRW), rd=0, rs1=0, csr=0xC00
        instr = self._csrrw(rd=0, rs1=0, csr=0xC00)  # cycle (u_ro)
        hart.exec_instr(instr)
        assert hart.mcause_val == 2, "CSRRW rs1=x0 仍为写操作, 应触发 IllInstr"

    def test_csrrw_rs1_zero_to_mvendorid_traps(self, hart):
        """CSRRW x0, mvendorid, x0: M 模式 CSRRS 正确, 但 CSRRW 仍非法."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=0, rs1=0, csr=0xF11)  # mvendorid (m_ro)
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    # ---- CSRRWI 写只读 CSR ----

    def test_csrrwi_to_readonly_cycle_traps(self, hart):
        """CSRRWI x0, cycle, 7: 立即数形式写入只读 CSR -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrwi(rd=0, uimm=7, csr=0xC00)  # cycle (u_ro)
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_csrrwi_uimm_zero_to_readonly_traps(self, hart):
        """CSRRWI x0, cycle, 0: uimm=0 仍为写操作 -> IllInstr.

        CSRRWI 与 CSRRW 一样始终为写操作, uimm=0 时写 0 仍然非法."""
        hart.mode = RiscvMode.M
        instr = self._csrrwi(rd=0, uimm=0, csr=0xC00)  # cycle (u_ro)
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    # ---- CSRRS/CSRRC rs1≠0 写只读 CSR ----

    def test_csrrs_write_to_readonly_traps(self, hart):
        """CSRRS x0, cycle, x10: rs1≠0 时 SET 位是写操作 -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrs(rd=0, rs1=10, csr=0xC00)  # cycle (u_ro)
        hart.gprs[10] = 0x1
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_csrrc_write_to_readonly_traps(self, hart):
        """CSRRC x0, cycle, x10: rs1≠0 时 CLEAR 位是写操作 -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrc(rd=0, rs1=10, csr=0xC00)  # cycle (u_ro)
        hart.gprs[10] = 0x1
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    # ---- CSRRSI/CSRRCI uimm≠0 写只读 CSR ----

    def test_csrrsi_to_readonly_traps(self, hart):
        """CSRRSI x0, cycle, 3: uimm≠0 的 SET 为写操作 -> IllInstr."""
        hart.mode = RiscvMode.M
        # CSRRSI funct3=110
        instr = (0xC00 << 20) | (3 << 15) | (6 << 12) | (0 << 7) | Opc.sys.value
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_csrrci_to_readonly_traps(self, hart):
        """CSRRCI x0, cycle, 3: uimm≠0 的 CLEAR 为写操作 -> IllInstr."""
        hart.mode = RiscvMode.M
        # CSRRCI funct3=111
        instr = (0xC00 << 20) | (3 << 15) | (7 << 12) | (0 << 7) | Opc.sys.value
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    # ---- 合法访问不受影响 ----

    def test_umode_can_read_umode_csr(self, hart):
        """U 模式可以读取 U-mode CSR (cycle)."""
        hart.mode = RiscvMode.U
        # CSRRS x5, cycle, x0: rs1=0 表示只读 (不写 CSR)
        instr = self._csrrs(rd=5, rs1=0, csr=0xC00)  # cycle
        advance = hart.exec_instr(instr)
        assert advance == 4, "合法 CSR 读应正常推进 PC"

    def test_mmode_can_read_write_mstatus(self, hart):
        """M 模式可以读写 mstatus."""
        hart.mode = RiscvMode.M
        hart.mstatus_val = 0xA0000000
        # CSRRW x0, mstatus, x10
        instr = self._csrrw(rd=0, rs1=10, csr=0x300)
        hart.gprs[10] = 0xB0000000
        advance = hart.exec_instr(instr)
        assert advance == 4
        assert hart.mstatus_val == 0xB0000000

    # ---- 机器信息只读寄存器 (mvendorid/marchid/mimpid/mhartid) ----

    def test_write_mvendorid_traps(self, hart):
        """写入只读 mvendorid -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=0, rs1=10, csr=0xF11)
        hart.gprs[10] = 99
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_write_marchid_traps(self, hart):
        """写入只读 marchid -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=0, rs1=10, csr=0xF12)
        hart.gprs[10] = 99
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_write_mimpid_traps(self, hart):
        """写入只读 mimpid -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=0, rs1=10, csr=0xF13)
        hart.gprs[10] = 99
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_write_mhartid_traps(self, hart):
        """写入只读 mhartid -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=0, rs1=10, csr=0xF14)
        hart.gprs[10] = 99
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_read_mvendorid_succeeds(self, hart):
        """读取 mvendorid 返回平台预设值."""
        hart.mode = RiscvMode.M
        instr = self._csrrs(rd=5, rs1=0, csr=0xF11)
        advance = hart.exec_instr(instr)
        assert advance == 4
        assert hart.gprs[5] == hart.csrs["mvendorid"].val

    # ---- 非法 CSR 地址 ----

    def test_unknown_csr_traps(self, hart):
        """访问不存在的 CSR 地址 -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=5, rs1=0, csr=0xFFF)  # 未定义
        hart.exec_instr(instr)
        assert hart.mcause_val == 2


# ============================================================
#  访存异常: 地址未对齐 / PMA 访问违例
# ============================================================


class TestMemoryAccessFaults:
    """验证 _mem_read / _mem_write 的对齐检查和 PMA 检查."""

    RAM_BASE = 0x8000_0000

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        bus = Bus(ram_size=1024 * 1024, ram_base=0x8000_0000)
        inject_memory_backend(h, bus.read, bus.write)
        h.bus = bus
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        return h

    # ---- 对齐检查 ----

    def test_lw_misaligned_traps(self, hart):
        """LW 从奇地址读取 -> LdAddrMisaligned."""
        mem_read(hart, 0x80001001, 4)
        assert hart.mcause_val == 4, f"应为 LdAddrMisaligned(4), 实际={hart.mcause_val}"

    def test_lh_misaligned_traps(self, hart):
        """LH 从奇地址读取 -> LdAddrMisaligned."""
        mem_read(hart, 0x80001001, 2)
        assert hart.mcause_val == 4

    def test_ld_misaligned_traps(self, hart):
        """LD 从非 8 字节对齐地址读取 -> LdAddrMisaligned."""
        mem_read(hart, 0x80001004, 8)
        assert hart.mcause_val == 4

    def test_sw_misaligned_traps(self, hart):
        """SW 到奇地址 -> StAddrMisaligned."""
        mem_write(hart, 0x80001001, b"\x00\x01\x02\x03")
        assert hart.mcause_val == 6, f"应为 StAddrMisaligned(6), 实际={hart.mcause_val}"

    def test_sd_misaligned_traps(self, hart):
        """SD 到非 8 字节对齐地址 -> StAddrMisaligned."""
        mem_write(hart, 0x80001004, b"\x00" * 8)
        assert hart.mcause_val == 6

    def test_lw_aligned_succeeds(self, hart):
        """LW 从对齐地址正常读取."""
        # 先写入数据确保可读
        hart._mem_write_phy(0x80001000, b"\x01\x02\x03\x04")
        data = mem_read(hart, 0x80001000, 4)
        assert data == b"\x01\x02\x03\x04"
        assert hart.mcause_val == 0

    def test_byte_access_always_aligned(self, hart):
        """1 字节 load/store 总是对齐."""
        hart._mem_write_phy(0x80001001, b"\x42")
        data = mem_read(hart, 0x80001001, 1)
        assert data == b"\x42"
        assert hart.mcause_val == 0

    # ---- PMA 检查 ----

    def test_read_empty_hole_traps(self, hart):
        """读取空洞地址 (非 RAM、非设备) -> LdAccessFault."""
        mem_read(hart, 0x0000_0000, 4)
        assert hart.mcause_val == 5, f"应为 LdAccessFault(5), 实际={hart.mcause_val}"

    def test_write_empty_hole_traps(self, hart):
        """写入空洞地址 -> StAccessFault."""
        mem_write(hart, 0x4000_0000, b"\xff")
        assert hart.mcause_val == 7, f"应为 StAccessFault(7), 实际={hart.mcause_val}"

    def test_read_ram_ok_pma(self, hart):
        """RAM 范围内读不触发 PMA 错误."""
        hart._mem_write_phy(0x80000000, b"\xaa\xbb")
        data = mem_read(hart, 0x80000000, 2)
        assert data == b"\xaa\xbb"
        assert hart.mcause_val == 0

    # ---- 对齐 + PMA 组合 ----

    def test_alignment_checked_before_translation(self, hart):
        """未对齐检测先于 MMU 翻译 (即使 PA 有效也不放过)."""
        mem_read(hart, 0x80000001, 4)  # misaligned but in RAM range
        assert hart.mcause_val == 4, "应先触发对齐错误而非页错误或访问错误"


# ============================================================
#  缺页异常 (Page Fault) — Sv39 页表遍历失败
# ============================================================


class TestPageFault:
    """验证 Sv39 模式下页表遍历失败时触发 LdPageFault / StPageFault.

    使用 bytearray 模拟物理内存 (绕过 Bus PMA 检查),
    在低物理地址 (0x1000–0x3FFF) 构建页表结构.
    """

    PAGE_SHIFT = 12
    SATP_MODE_SV39 = 8

    # 页表物理基址
    L1_BASE = 0x1000  # 根页表 (PPN=1)
    L2_BASE = 0x2000  # 二级页表 (PPN=2)
    L3_BASE = 0x3000  # 三级页表 (PPN=3)
    DATA_PA = 0x100000  # 数据页物理地址 (PPN=0x100)

    @pytest.fixture
    def ram_ctx(self):
        """2 MiB 物理内存, 地址 [0, 2 MiB)."""
        ram = bytearray(2 * 1024 * 1024)

        def read_fn(addr: int, size: int) -> bytes:
            return bytes(ram[addr : addr + size])

        def write_fn(addr: int, data: bytes):
            for i, b in enumerate(data):
                ram[addr + i] = b

        return ram, read_fn, write_fn

    @pytest.fixture
    def hart(self, ram_ctx):
        """hart 注入 bytearray 内存后端, 初始为 Bare 模式."""
        _ram, read_fn, write_fn = ram_ctx
        h = Hart(id=0)
        inject_memory_backend(h, read_fn, write_fn)
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        return h

    # -- helper --

    @staticmethod
    def _write_pte(ram: bytearray, table_base: int, index: int, pte_val: int):
        """在 *table_base* 页表的第 *index* 项写入 8 字节 PTE."""
        addr = table_base + index * 8
        data = pte_val.to_bytes(8, "little")
        ram[addr : addr + 8] = data

    @staticmethod
    def _make_pte(
        v: bool = False,
        r: bool = False,
        w: bool = False,
        x: bool = False,
        u: bool = False,
        ppn: int = 0,
    ) -> int:
        """构造 PTE 原始值."""
        val = 0
        if v:
            val |= 1 << 0
        if r:
            val |= 1 << 1
        if w:
            val |= 1 << 2
        if x:
            val |= 1 << 3
        if u:
            val |= 1 << 4
        # PPN 拆分 (RISC-V 标准连续编码): PPN[9:0]->bits[19:10],
        # PPN[18:10]->bits[28:20], PPN[43:19]->bits[53:29].
        val |= (ppn & 0x3FF) << 10
        val |= ((ppn >> 10) & 0x1FF) << 20
        val |= ((ppn >> 19) & 0x1FFFFFFF) << 29
        return val

    def _setup_valid_4k(
        self,
        ram: bytearray,
        va: int = 0,
        target_pa: int | None = None,
    ):
        """构建完整 Sv39 三级 4 KiB 映射 va -> target_pa.

        使用 vpn2/vpn1/vpn0 作为各级页表的索引, 适用于任意 VA.
        """
        if target_pa is None:
            target_pa = self.DATA_PA
        # 内联 _sv39_vpn, 避免导入私有符号
        vpn0 = (va >> 12) & 0x1FF
        vpn1 = (va >> 21) & 0x1FF
        vpn2 = (va >> 30) & 0x1FF
        target_ppn = target_pa >> self.PAGE_SHIFT

        # L1 (根) -> L2
        self._write_pte(
            ram,
            self.L1_BASE,
            vpn2,
            self._make_pte(v=True, ppn=self.L2_BASE >> self.PAGE_SHIFT),
        )
        # L2 -> L3
        self._write_pte(
            ram,
            self.L2_BASE,
            vpn1,
            self._make_pte(v=True, ppn=self.L3_BASE >> self.PAGE_SHIFT),
        )
        # L3 叶
        self._write_pte(
            ram,
            self.L3_BASE,
            vpn0,
            self._make_pte(v=True, r=True, w=True, x=True, ppn=target_ppn),
        )

    def _enable_sv39(self, hart: "Hart", root_ppn: int = 1):
        """将 hart 切换到 S 模式并启用 Sv39 (MMU 翻译仅在 S/U 模式生效).

        同时配置 PMP NAPOT 开放全部地址空间, 否则 S 模式下无 PMP 条目时
        PMP 默认拒绝所有访问.
        """
        hart.mode = RiscvMode.S
        # PMP: 1 条 NAPOT 规则覆盖全部地址空间 R+W+X
        hart.csrs["pmpcfg0"].val = 0x1F  # NAPOT, R+W+X
        hart.csrs["pmpaddr0"].val = 0x003F_FFFF_FFFF_FFFF
        hart.satp_val = (self.SATP_MODE_SV39 << 60) | root_ppn

    # ---- LdPageFault ----

    def test_ld_page_fault_root_pte_invalid(self, hart, ram_ctx):
        """根页表条目 V=0 -> LdPageFault (13)."""
        ram, _read_fn, _write_fn = ram_ctx
        # 所有 L1 条目均为默认的 0 (V=0)
        self._enable_sv39(hart)
        mem_read(hart, 0x0, 4)
        assert hart.mcause_val == 13, f"应为 LdPageFault(13), 实际={hart.mcause_val}"

    def test_ld_page_fault_l2_pte_invalid(self, hart, ram_ctx):
        """L2 条目 V=0 -> LdPageFault (13)."""
        ram, _read_fn, _write_fn = ram_ctx
        # L1 有效, 指向 L2; 但 L2 全为 0
        self._write_pte(
            ram,
            self.L1_BASE,
            0,
            self._make_pte(v=True, ppn=self.L2_BASE >> self.PAGE_SHIFT),
        )
        self._enable_sv39(hart)
        mem_read(hart, 0x0, 4)
        assert hart.mcause_val == 13, f"应为 LdPageFault(13), 实际={hart.mcause_val}"

    def test_ld_page_fault_l3_pte_invalid(self, hart, ram_ctx):
        """L3 叶条目 V=0 -> LdPageFault (13)."""
        ram, _read_fn, _write_fn = ram_ctx
        # L1 -> L2
        self._write_pte(
            ram,
            self.L1_BASE,
            0,
            self._make_pte(v=True, ppn=self.L2_BASE >> self.PAGE_SHIFT),
        )
        # L2 -> L3
        self._write_pte(
            ram,
            self.L2_BASE,
            0,
            self._make_pte(v=True, ppn=self.L3_BASE >> self.PAGE_SHIFT),
        )
        # L3 全为 0 (V=0)
        self._enable_sv39(hart)
        mem_read(hart, 0x0, 4)
        assert hart.mcause_val == 13, f"应为 LdPageFault(13), 实际={hart.mcause_val}"

    def test_ld_page_fault_unsupported_mode(self, hart, ram_ctx):
        """未实现的 satp 模式 (如 Sv48=9) -> LdPageFault."""
        _ram, _read_fn, _write_fn = ram_ctx
        hart.mode = RiscvMode.S  # MMU 翻译仅在 S/U 模式生效
        hart.satp_val = (9 << 60) | 1  # Sv48, 未实现
        mem_read(hart, 0x0, 4)
        assert hart.mcause_val == 13, f"应为 LdPageFault(13), 实际={hart.mcause_val}"

    # ---- StPageFault ----

    def test_st_page_fault_invalid_pte(self, hart, ram_ctx):
        """Sv39 下 store 遇到无效 PTE -> StPageFault (15)."""
        _ram, _read_fn, _write_fn = ram_ctx
        # 无任何页表, L1 全为 V=0
        self._enable_sv39(hart)
        mem_write(hart, 0x1000, b"\x01\x02\x03\x04")
        assert hart.mcause_val == 15, f"应为 StPageFault(15), 实际={hart.mcause_val}"

    # ---- 有效映射 — 不触发 PageFault ----

    def test_ld_succeeds_valid_mapping(self, hart, ram_ctx):
        """完整 Sv39 映射下 load 应成功."""
        ram, _read_fn, _write_fn = ram_ctx
        self._setup_valid_4k(ram)
        self._enable_sv39(hart)
        # 在数据页写入预期值
        val = b"\xde\xad\xbe\xef"
        ram[self.DATA_PA : self.DATA_PA + 4] = val
        data = mem_read(hart, 0x0, 4)
        assert data == val
        assert hart.mcause_val == 0, f"不应 trap, mcause={hart.mcause_val}"

    def test_st_succeeds_valid_mapping(self, hart, ram_ctx):
        """完整 Sv39 映射下 store 应成功."""
        ram, _read_fn, _write_fn = ram_ctx
        self._setup_valid_4k(ram)
        self._enable_sv39(hart)
        val = b"\xca\xfe\xba\xbe"
        mem_write(hart, 0x0, val)
        assert hart.mcause_val == 0, f"不应 trap, mcause={hart.mcause_val}"
        # 从物理内存验证写入
        assert bytes(ram[self.DATA_PA : self.DATA_PA + 4]) == val

    def test_ld_st_nonzero_va(self, hart, ram_ctx):
        """非零 VA 的映射 (验证 VPN 索引正确)."""

        ram, _read_fn, _write_fn = ram_ctx
        va = 0x7FFFFFE000  # 一个非零 VA (Sv39 最高合法地址附近)
        self._setup_valid_4k(ram, va=va, target_pa=self.DATA_PA)
        self._enable_sv39(hart)
        val = b"\x11\x22\x33\x44"
        ram[self.DATA_PA : self.DATA_PA + 4] = val
        data = mem_read(hart, va, 4)
        assert data == val
        assert hart.mcause_val == 0

    # ---- Bare 模式 — 无 PageFault ----

    def test_bare_mode_no_page_fault(self, hart, ram_ctx):
        """Bare 模式: VA = PA, 不做翻译, 不触发缺页异常."""
        ram, _read_fn, _write_fn = ram_ctx
        # 不设置 satp -> 默认 Bare (mode=0)
        # VA 0x5000 直接当 PA 用
        val = b"\x42\x42\x42\x42"
        ram[0x5000:0x5004] = val
        data = mem_read(hart, 0x5000, 4)
        assert data == val
        assert hart.mcause_val == 0

    def test_bare_mode_st_no_page_fault(self, hart, ram_ctx):
        """Bare 模式 store 也不应触发缺页异常."""
        ram, _read_fn, _write_fn = ram_ctx
        val = b"\xff\xee\xdd\xcc"
        mem_write(hart, 0x6000, val)
        assert hart.mcause_val == 0
        assert bytes(ram[0x6000:0x6004]) == val

    # ---- 页错误优先于 PMA ----

    def test_page_fault_before_pma(self, hart, ram_ctx):
        """Sv39 翻译失败直接报 PageFault, 不会走到 PMA 检查."""
        _ram, _read_fn, _write_fn = ram_ctx
        self._enable_sv39(hart)
        # VA 通过 Sv39 翻译, 但 L1 条目无效
        # 即使 PA 可能落在有效 RAM 范围也不应到达 PMA
        mem_read(hart, 0x80000000, 4)
        assert hart.mcause_val == 13, (
            f"应为 LdPageFault(13), 翻译失败不应转为 LdAccessFault, 实际={hart.mcause_val}"
        )

    # ---- TLB 缓存后仍正确 ----

    def test_tlb_caches_and_page_fault_on_miss(self, hart, ram_ctx):
        """TLB miss -> 页表遍历, 无效 PTE -> PageFault; 重复也不走运."""
        ram, _read_fn, _write_fn = ram_ctx
        self._setup_valid_4k(ram, va=0x0)
        self._enable_sv39(hart)
        # 第一次: TLB miss, 页表遍历成功
        ram[self.DATA_PA : self.DATA_PA + 4] = b"\x01\x02\x03\x04"
        assert mem_read(hart, 0x0, 4) == b"\x01\x02\x03\x04"
        # 第二次: TLB hit
        assert mem_read(hart, 0x0, 4) == b"\x01\x02\x03\x04"

        # 不同 VA -> TLB miss -> 无效 -> PageFault
        hart._consecutive_traps = 0
        mem_read(hart, 0x1000, 4)
        assert hart.mcause_val == 13, (
            f"TLB miss 后无效 PTE 应触发 LdPageFault, 实际={hart.mcause_val}"
        )


# ============================================================
#  定时器中断 (MTI) — CLINT mtime/mtimecmp 触发
# ============================================================


class TestTimerInterrupt:
    """验证 CLINT 定时器中断 (MTI) 通过 check_pending_interrupts 正确投递.

    CLINT.check_interrupt() 在 mtime >= mtimecmp (且 mtimecmp > 0) 时
    设置 mip 的 MTIP 位 (bit 7), 之后由 check_pending_interrupts(Hart)
    按优先级 (MSI > MTI > ...) 和 mideleg 委派投递.

    覆盖场景:
    - M-mode 直接接收 MTI
    - MIE=0 阻塞
    - mtimecmp=0 禁用
    - mideleg 委派到 S-mode
    - S-mode 全局中断使能阻塞
    """

    MTECMP_OFFSET = 0x4000  # CLINT mtimecmp 区域偏移

    @pytest.fixture
    def clint(self) -> CLINT:
        """单 hart CLINT."""
        return CLINT(num_harts=1)

    @pytest.fixture
    def hart(self, clint) -> Hart:
        """hart 连接到 CLINT, M 模式, MIE 使能, MTIE 使能."""
        h = Hart(id=0)
        h.csrs["mtvec"].val = 0x80000000
        h.csrs["stvec"].val = 0x80004000
        h.pc = 0x1000
        h.mode = RiscvMode.M
        h.mstatus_val = MSTATUS_MIE
        # 使能 M 模式定时器中断 (MTIE = bit 7)
        h.csrs["mie"].val = 1 << 7
        h.interrupt_ctrl = clint
        return h

    @staticmethod
    def _set_mtimecmp(clint: CLINT, hart_id: int, val: int):
        """通过内存映射接口写入 hart 的 mtimecmp."""
        clint.write(TestTimerInterrupt.MTECMP_OFFSET + hart_id * 8, val.to_bytes(8, "little"))

    @staticmethod
    def _cause_is_timer(mcause: int) -> bool:
        """mcause 表示 MTI (code=7, bit63=1 -> 0x8000000000000007)."""
        return mcause == 0x8000_0000_0000_0007

    # ---- M-mode 定时器中断 ----

    def test_mti_taken_mmode(self, hart, clint):
        """mtime >= mtimecmp -> MTI 投递到 M 模式."""
        self._set_mtimecmp(clint, 0, 50)
        clint.tick(100)  # mtime = 100 >= 50

        taken = check_pending_interrupts(hart)
        assert taken, "应识别待处理定时器中断"
        assert hart.pc == 0x80000000, "应跳转到 mtvec"
        assert hart.mode == RiscvMode.M
        assert self._cause_is_timer(hart.mcause_val), (
            f"mcause 应为 MTI (0x8000000000000007), 实际 {hart.mcause_val:#018x}"
        )
        # MIE 应在进入 trap 后清零
        assert not hart.mie, "进入 trap 后 MIE 应清零"

    def test_mti_not_taken_before_threshold(self, hart, clint):
        """mtime < mtimecmp -> 无中断."""
        self._set_mtimecmp(clint, 0, 100)
        clint.tick(50)  # mtime = 50 < 100

        taken = check_pending_interrupts(hart)
        assert not taken
        assert hart.pc == 0x1000, "PC 不应变化"
        assert hart.mode == RiscvMode.M

    def test_mti_not_taken_mtimecmp_zero(self, hart, clint):
        """mtimecmp == 0 -> 禁用定时器, 不触发中断."""
        self._set_mtimecmp(clint, 0, 0)
        clint.tick(100)  # mtime = 100 >= 0, 但 mtimecmp=0 表示禁用

        taken = check_pending_interrupts(hart)
        assert not taken

    def test_mti_not_taken_mie_off(self, hart, clint):
        """MIE=0 -> 全局关中断, MTI 被阻塞."""
        self._set_mtimecmp(clint, 0, 50)
        clint.tick(100)
        hart.mie = False  # 关 M 模式全局中断

        taken = check_pending_interrupts(hart)
        assert not taken
        assert hart.pc == 0x1000

    def test_mti_not_taken_mtie_off(self, hart, clint):
        """mie[7] (MTIE) = 0 -> 即使 mip[7] 挂起也不响应."""
        self._set_mtimecmp(clint, 0, 50)
        clint.tick(100)
        hart.csrs["mie"].val = 0  # 清除全部中断使能

        taken = check_pending_interrupts(hart)
        assert not taken

    # ---- 委派到 S-mode ----

    def test_mti_delegated_to_smode(self, hart, clint):
        """mideleg[7]=1, U 模式: MTI 委派到 S 模式."""
        self._set_mtimecmp(clint, 0, 50)
        clint.tick(100)
        hart.csrs["mideleg"].val = 1 << 7  # 委派 MTI
        hart.mode = RiscvMode.U
        hart.mstatus_val = MSTATUS_MIE | MSTATUS_SIE  # SIE 也需要使能

        taken = check_pending_interrupts(hart)
        assert taken
        assert hart.pc == 0x80004000, "委派后应跳转到 stvec"
        assert hart.mode == RiscvMode.S, "应进入 S 模式"
        # scause = bit63=1, code=7 (原 cause 不变)
        expected_scause = (1 << 63) | 7
        assert hart.scause_val == expected_scause, (
            f"scause 应为 0x{expected_scause:x}, 实际 0x{hart.scause_val:x}"
        )
        assert hart.sepc_val == 0x1000
        assert not hart.sie, "进入 trap 后 SIE 应清零"

    def test_mti_delegated_but_sie_off_blocked(self, hart, clint):
        """已委派的 MTI 在 S 模式且 SIE=0 时被阻塞."""
        self._set_mtimecmp(clint, 0, 50)
        clint.tick(100)
        hart.csrs["mideleg"].val = 1 << 7
        hart.mode = RiscvMode.S
        hart.sie = False  # S-mode 全局关中断

        taken = check_pending_interrupts(hart)
        assert not taken

    # ---- 优先级 ----

    def test_msi_higher_priority_than_mti(self, hart, clint):
        """MSI 优先级高于 MTI: 同时挂起时先响应 MSI."""
        self._set_mtimecmp(clint, 0, 50)
        clint.tick(100)
        clint.send_ipi(0)  # 同时设置 MSI
        # 需同时使能 MSIE (bit 3) 和 MTIE (bit 7)
        hart.csrs["mie"].val = (1 << 3) | (1 << 7)

        taken = check_pending_interrupts(hart)
        assert taken
        # MSI = cause code 3 | (1<<63)
        assert hart.mcause_val == 0x8000_0000_0000_0003, (
            f"MSI 优先于 MTI, 应为 MSI (0x8000000000000003), 实际 {hart.mcause_val:#018x}"
        )

    # ---- 连续触发 ----

    def test_mti_fires_after_clearing_mip(self, hart, clint):
        """清除 mip 后再次 tick -> MTI 再次触发."""
        self._set_mtimecmp(clint, 0, 50)
        clint.tick(100)

        # 第一次中断
        taken = check_pending_interrupts(hart)
        assert taken
        assert self._cause_is_timer(hart.mcause_val)

        # 模拟 MRET 后恢复, 设置新的 mtimecmp
        hart.mie = True
        hart.mode = RiscvMode.M
        hart.pc = 0x2000
        hart._consecutive_traps = 0
        self._set_mtimecmp(clint, 0, 200)
        clint.tick(150)  # mtime = 250 >= 200

        # 第二次中断
        taken = check_pending_interrupts(hart)
        assert taken
        assert self._cause_is_timer(hart.mcause_val)


# ============================================================
#  WFI (Wait For Interrupt) — 低功耗等待与唤醒
# ============================================================


class TestWfi:
    """验证 WFI 指令: NOP (中断已挂起)、等待、中断唤醒、TW 陷态.

    RISC-V Privileged Spec §3.3.5:
    - WFI 是 hint, 实现可将其视为 NOP
    - 若中断已挂起且使能 (mip & mie ≠ 0), WFI 立即返回
    - 否则 hart 可进入等待状态, 由挂起且使能的中断唤醒
    - 唤醒后 PC 指向 WFI 下一条指令, 中断走正常处理流程
    - mstatus.TW=1 且非 M 模式执行 WFI -> IllInstr
    """

    # WFI instruction encoding: funct12=0x105, funct3=0, opcode=0x73 (sys)
    WFI_INSTR = 0x10500073

    MTECMP_OFFSET = 0x4000

    @staticmethod
    def _set_mtimecmp(clint: CLINT, hart_id: int, val: int):
        clint.write(
            TestWfi.MTECMP_OFFSET + hart_id * 8,
            val.to_bytes(8, "little"),
        )

    # ============================================================
    #  NOP — 中断已挂起时立即返回
    # ============================================================

    def test_wfi_nop_when_interrupt_pending(self):
        """mip & mie ≠ 0 时 WFI 视作 NOP, PC 正常 +4, 不进入等待."""
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.M
        h.csrs["mtvec"].val = 0x80000000
        # 模拟 mip[3] (MSIP) 和 mie[3] (MSIE) 都已置位
        h.csrs["mip"].val = 1 << 3
        h.csrs["mie"].val = 1 << 3

        advance = h.exec_instr(self.WFI_INSTR)
        assert advance == 4, "WFI 在中断已挂起时应返回 PC+4"
        # exec_instr 只返回 advance, PC 由调用方 (Emulator.step) 推进
        h.pc += advance
        assert not h._waiting, "不应进入等待状态"
        assert h.pc == 0x1004, "PC 应前进到 WFI 的下一条指令"

    def test_wfi_nop_when_timer_interrupt_pending(self):
        """MTI 挂起且 MTIE 使能 -> WFI 立即返回."""
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.M
        h.csrs["mtvec"].val = 0x80000000
        # MTIP (bit 7) + MTIE (bit 7)
        h.csrs["mip"].val = 1 << 7
        h.csrs["mie"].val = 1 << 7

        advance = h.exec_instr(self.WFI_INSTR)
        assert advance == 4
        assert not h._waiting

    # ============================================================
    #  等待状态
    # ============================================================

    def test_wfi_enters_wait_when_no_interrupt(self):
        """无挂起且使能的中断时 WFI 进入等待状态."""
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.M
        h.csrs["mtvec"].val = 0x80000000
        # mip 和 mie 都为 0

        advance = h.exec_instr(self.WFI_INSTR)
        assert advance == 4, "PC 应越过 WFI 再进入等待"
        # exec_instr 只返回 advance, PC 由调用方 (Emulator.step) 推进
        h.pc += advance
        assert h._waiting, "应进入等待状态"
        assert h.pc == 0x1004, "PC 指向 WFI 下一条指令"

    def test_wfi_not_enter_wait_when_masked(self):
        """mip 有硬件中断但 mie 未使能 -> 仍进入等待 (中断不可被响应)."""
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.M
        h.csrs["mtvec"].val = 0x80000000
        h.csrs["mip"].val = 1 << 7  # MTIP 挂起
        h.csrs["mie"].val = 0  # 但 MTIE 未使能

        advance = h.exec_instr(self.WFI_INSTR)
        assert advance == 4
        assert h._waiting, "中断未使能 -> 仍应进入等待"

    # ============================================================
    #  中断唤醒 (通过 CLINT)
    # ============================================================

    def test_wfi_wake_by_timer(self):
        """WFI 等待后 CLINT 定时器中断唤醒 hart."""
        clint = CLINT(num_harts=1)
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.M
        h.mstatus_val = MSTATUS_MIE
        h.csrs["mtvec"].val = 0x80000000
        h.csrs["mie"].val = 1 << 7  # MTIE
        h.interrupt_ctrl = clint

        # 设置总线 (满足 PMA 检查)
        bus = Bus(ram_size=1024 * 1024, ram_base=0x8000_0000)
        inject_memory_backend(h, bus.read, bus.write)
        h.bus = bus

        # 设置 mtimecmp 为未来值, mtime 还未到达
        self._set_mtimecmp(clint, 0, 100)
        clint.tick(10)  # mtime = 10

        # 执行 WFI -> 应进入等待
        advance = h.exec_instr(self.WFI_INSTR)
        assert advance == 4
        assert h._waiting, "应进入等待状态"

        # 推进时钟使 mtime >= mtimecmp
        clint.tick(200)  # mtime = 210 >= 100

        # 模拟 emulator 循环: 对等待中的 hart 调用 check_pending_interrupts
        taken = check_pending_interrupts(h)
        assert taken, "应识别中断并唤醒"
        assert not h._waiting, "应退出等待状态"
        assert h.pc == 0x80000000, "应跳转到 mtvec"
        cause = h.mcause_val
        assert cause == 0x8000_0000_0000_0007, f"应为 MTI, 实际 mcause={cause:#018x}"

    def test_wfi_wake_by_ipi(self):
        """WFI 等待后 IPI 唤醒 hart."""
        clint = CLINT(num_harts=1)
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.M
        h.mstatus_val = MSTATUS_MIE
        h.csrs["mtvec"].val = 0x80000000
        h.csrs["mie"].val = 1 << 3  # MSIE
        h.interrupt_ctrl = clint

        bus = Bus(ram_size=1024 * 1024, ram_base=0x8000_0000)
        inject_memory_backend(h, bus.read, bus.write)
        h.bus = bus

        # 执行 WFI -> 等待
        h.exec_instr(self.WFI_INSTR)
        assert h._waiting

        # 发送 IPI
        clint.send_ipi(0)

        # 检查中断 -> 应唤醒
        taken = check_pending_interrupts(h)
        assert taken
        assert not h._waiting
        assert h.pc == 0x80000000, "应跳转到 mtvec"
        cause = h.mcause_val
        assert cause == 0x8000_0000_0000_0003, f"应为 MSI, 实际 mcause={cause:#018x}"

    def test_wfi_stays_waiting_no_interrupt(self):
        """无中断时 check_pending_interrupts 不唤醒 hart."""
        clint = CLINT(num_harts=1)
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.M
        h.mstatus_val = MSTATUS_MIE
        h.csrs["mtvec"].val = 0x80000000
        h.csrs["mie"].val = 1 << 7
        h.interrupt_ctrl = clint

        # 执行 WFI -> 等待
        h.exec_instr(self.WFI_INSTR)
        assert h._waiting

        # 无任何中断 -> check 返回 False, 继续等待
        taken = check_pending_interrupts(h)
        assert not taken
        assert h._waiting, "无中断应继续等待"

    # ============================================================
    #  TW (Timeout Wait) — 非 M 模式下 mstatus.TW=1 -> IllInstr
    # ============================================================

    def test_wfi_tw_trap_smode(self):
        """S 模式下 TW=1 -> WFI 触发 IllInstr."""
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.S
        h.csrs["mtvec"].val = 0x80000000
        h.mstatus_val = MSTATUS_TW  # M-mode 设置 TW

        advance = h.exec_instr(self.WFI_INSTR)
        assert advance == 0, "trap 时不应推进 PC"
        assert h.mcause_val == 2, f"应为 IllInstr(2), 实际={h.mcause_val}"
        assert not h._waiting, "trap 后不应处于等待"

    def test_wfi_tw_trap_umode(self):
        """U 模式下 TW=1 -> WFI 触发 IllInstr."""
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.U
        h.csrs["mtvec"].val = 0x80000000
        h.mstatus_val = MSTATUS_TW

        advance = h.exec_instr(self.WFI_INSTR)
        assert advance == 0
        assert h.mcause_val == 2
        assert not h._waiting

    def test_wfi_tw_ok_mmode(self):
        """M 模式下即使 TW=1 也不 trap (TW 仅针对低特权级)."""
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.M
        h.csrs["mtvec"].val = 0x80000000
        h.mstatus_val = MSTATUS_TW
        # M-mode 下 WFI 应合法 (进入等待, 因为无中断)
        advance = h.exec_instr(self.WFI_INSTR)
        assert advance == 4, "M-mode WFI 应正常执行"
        assert h._waiting

    def test_wfi_smode_no_tw_ok(self):
        """S 模式下 TW=0 -> WFI 正常执行, 可进入等待."""
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.S
        h.csrs["mtvec"].val = 0x80000000
        h.mstatus_val = 0  # TW=0

        advance = h.exec_instr(self.WFI_INSTR)
        assert advance == 4, "S-mode WFI (TW=0) 应正常"
        assert h._waiting

    # ============================================================
    #  其他 trap 也唤醒 WFI
    # ============================================================

    def test_trap_wakes_wfi(self):
        """任何 trap (如 ECALL) 也应唤醒 WFI 等待中的 hart."""
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.M
        h.mstatus_val = MSTATUS_MIE
        h.csrs["mtvec"].val = 0x80000000
        h.csrs["mie"].val = 1 << 7

        bus = Bus(ram_size=1024 * 1024, ram_base=0x8000_0000)
        inject_memory_backend(h, bus.read, bus.write)
        h.bus = bus

        # 执行 WFI -> 等待
        h.exec_instr(self.WFI_INSTR)
        assert h._waiting

        # 注入 ECALL trap
        deliver_trap(h, TrapType.EcallFromMmode, tval=0, is_interrupt=False)
        assert not h._waiting, "trap 应清除等待状态"
        assert h.mcause_val == 11, f"应为 EcallFromMmode(11), 实际={h.mcause_val}"

    # ============================================================
    #  WFI 在 emulator 中的行为
    # ============================================================

    def test_emulator_step_skips_waiting_hart(self):
        """Emulator.step 对等待中的 hart 不取指/执行, 但检查中断."""
        clint = CLINT(num_harts=1)
        emu = Emulator(
            PlatformConfig(
                num_harts=1,
                ram_size=1024 * 1024,
                periph=PeripheralConfig(clint_base=0x0200_0000),
            ),
        )
        # 替换 CLINT 为我们的实例
        emu.clint = clint
        emu.harts[0].interrupt_ctrl = clint
        hart = emu.harts[0]
        hart.pc = 0x80000000
        hart.mode = RiscvMode.M
        hart.mstatus_val = MSTATUS_MIE
        hart.csrs["mtvec"].val = 0x80000100
        hart.csrs["mie"].val = 1 << 7

        # 将 WFI 指令写入 RAM
        emu.bus.write(0x80000000, self.WFI_INSTR.to_bytes(4, "little"))

        # 设置定时器: mtimecmp = 50, 当前 mtime = 0
        self._set_mtimecmp(clint, 0, 50)

        # step 1: 执行 WFI -> 进入等待
        emu.step()
        assert hart._waiting, "step 1: 应进入 WFI 等待"
        assert hart.pc == 0x80000004, "PC 应指向 WFI 后一条指令"

        # step 2: 仍等待, mtime 从 0->1, 不到 50
        emu.step()
        assert hart._waiting, "step 2: 仍应等待"
        assert hart.pc == 0x80000004, "PC 不应变化"

        # 推进 mtime 越过 mtimecmp
        clint.tick(100)  # mtime 变成 ~102, 大于 50

        # step 3: 中断唤醒
        emu.step()
        assert not hart._waiting, "step 3: 应被定时器中断唤醒"
        assert hart.pc == 0x80000100, "应跳转到 mtvec"
        cause = hart.mcause_val
        assert cause == 0x8000_0000_0000_0007, f"应为 MTI, 实际 mcause={cause:#018x}"

    # ============================================================
    #  多 hart WFI 交叉唤醒
    # ============================================================

    def _make_emu(self, num_harts: int) -> Emulator:
        """创建多 hart Emulator, 全部 hart 开启 MSIE + mtvec."""
        clint = CLINT(num_harts=num_harts)
        emu = Emulator(
            PlatformConfig(
                num_harts=num_harts,
                ram_size=1024 * 1024,
                periph=PeripheralConfig(clint_base=0x0200_0000),
            ),
        )
        emu.clint = clint
        for h in emu.harts:
            h.interrupt_ctrl = clint
            h.mode = RiscvMode.M
            h.mstatus_val = MSTATUS_MIE
            h.csrs["mtvec"].val = 0x80000100
            h.csrs["mie"].val = 1 << 3  # MSIE
        return emu

    @staticmethod
    def _wfi_targets(num_harts: int, max_waiters: int = 3):
        """返回应执行 WFI 的 hart 列表 (跳过 hart 0 的唤醒者)."""
        return list(range(1, min(num_harts, max_waiters + 1)))

    # fmt: off
    @pytest.mark.parametrize("num_harts", [2, 8, 9, 10, 12, 16])
    # fmt: on
    def test_wfi_wake_by_cross_hart_ipi(self, num_harts: int):
        """Hart 0 执行 WFI 等待, Hart 1 通过 CLINT 发 IPI 唤醒 Hart 0."""
        emu = self._make_emu(num_harts)
        hart0 = emu.harts[0]

        # hart 0 放 WFI; 其余 hart 放 NOP
        emu.bus.write(0x80000000, self.WFI_INSTR.to_bytes(4, "little"))
        for tid in range(1, num_harts):
            emu.bus.write(0x80000100 + tid * 0x100, (0x00000013).to_bytes(4, "little"))
            emu.harts[tid].pc = 0x80000100 + tid * 0x100
        hart0.pc = 0x80000000

        # step 1: hart 0 执行 WFI -> 进入等待; 其余 hart 执行 NOP
        emu.step()
        assert hart0._waiting, f"hart 0 应进入 WFI 等待 (num_harts={num_harts})"
        assert hart0.pc == 0x80000004, "hart 0 PC 应越过 WFI"

        # 发送 IPI 到 hart 0
        emu.clint.send_ipi(0)

        # step 2: hart 0 被 IPI 唤醒
        emu.step()
        assert not hart0._waiting, f"hart 0 应被 IPI 唤醒 (num_harts={num_harts})"
        assert hart0.pc == 0x80000100, (
            f"hart 0 应跳转到 mtvec, PC={hart0.pc:#x} (num_harts={num_harts})"
        )
        cause = hart0.mcause_val
        assert cause == 0x8000_0000_0000_0003, (
            f"应为 MSI, 实际={cause:#018x} (num_harts={num_harts})"
        )

    # fmt: off
    @pytest.mark.parametrize("num_harts", [2, 8, 9, 10, 12, 16])
    # fmt: on
    def test_wfi_only_target_wakes(self, num_harts: int):
        """多 hart 同时 WFI; 仅目标 hart 收到 IPI 后唤醒, 其余继续等待."""
        emu = self._make_emu(num_harts)
        waiters = self._wfi_targets(num_harts)  # harts 1, 2, ... (最多 3 个)

        # 所有 waiter + hart 0 都写入 WFI
        for tid in [0] + waiters:
            emu.bus.write(0x80000000 + tid * 0x100, self.WFI_INSTR.to_bytes(4, "little"))
            emu.harts[tid].pc = 0x80000000 + tid * 0x100

        # step 1: 所有 hart 进入 WFI 等待
        emu.step()
        for tid in [0] + waiters:
            assert emu.harts[tid]._waiting, (
                f"hart {tid} 应进入等待 (num_harts={num_harts})"
            )

        # 仅向 hart 0 发送 IPI
        emu.clint.send_ipi(0)

        # step 2: hart 0 唤醒, 其余继续等待
        emu.step()
        assert not emu.harts[0]._waiting, f"hart 0 应被 IPI 唤醒 (num_harts={num_harts})"
        assert emu.harts[0].pc == 0x80000100, (
            f"hart 0 应跳转到 mtvec, PC={emu.harts[0].pc:#x}"
        )
        for tid in waiters:
            assert emu.harts[tid]._waiting, (
                f"hart {tid} 未收到 IPI, 应继续等待 (num_harts={num_harts})"
            )

    # fmt: off
    @pytest.mark.parametrize("num_harts", [2, 8, 9, 10, 12, 16])
    # fmt: on
    def test_wfi_wake_by_ipi_then_repeat(self, num_harts: int):
        """Hart 0 被 IPI 唤醒, 清除后再次 WFI 可被再次唤醒."""
        emu = self._make_emu(num_harts)
        hart0 = emu.harts[0]

        emu.bus.write(0x80000000, self.WFI_INSTR.to_bytes(4, "little"))
        hart0.pc = 0x80000000

        # 第一轮: WFI -> IPI -> 唤醒
        emu.step()
        assert hart0._waiting, f"第一轮: 应进入等待 (num_harts={num_harts})"
        emu.clint.send_ipi(0)
        emu.step()
        assert not hart0._waiting, f"第一轮: 应被唤醒 (num_harts={num_harts})"

        # 清除 IPI 并将 PC 复位到 WFI
        emu.clint.clear_ipi(0)
        hart0.csrs["mip"].val &= ~(1 << 3)
        hart0._waiting = False
        hart0._consecutive_traps = 0
        hart0.mode = RiscvMode.M
        hart0.mstatus_val |= MSTATUS_MIE
        hart0.pc = 0x80000000

        # 第二轮: 再次 WFI -> IPI -> 唤醒
        emu.step()
        assert hart0._waiting, f"第二轮: 应再次进入等待 (num_harts={num_harts})"
        emu.clint.send_ipi(0)
        emu.step()
        assert not hart0._waiting, f"第二轮: 应再次被唤醒 (num_harts={num_harts})"


# ============================================================
#  栈溢出防护 -- 覆盖 M / S / U 三种特权级
# ============================================================

# 测试共用常量
_SO_RAM_BASE = 0x8000_0000
_SO_RAM_SIZE = 64 * 1024  # 64 KiB

# S-mode Sv39 guard page 物理地址 (使用低地址以适配 2 MiB bytearray)
_SO_SMODE_STACK_VA = 0x2000  # VPN all zero, only vpn0 differs
_SO_SMODE_GUARD_VA = 0x1000  # guard page, unmapped
_SO_SMODE_STACK_PA = 0x5000  # stack physical page


class TestStackOverflowMmode:
    """M-mode stack overflow: PMA physical memory boundary detection."""

    RAM_BASE = _SO_RAM_BASE
    RAM_SIZE = _SO_RAM_SIZE
    RAM_TOP = _SO_RAM_BASE + _SO_RAM_SIZE

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        bus = Bus(ram_size=self.RAM_SIZE, ram_base=self.RAM_BASE)
        inject_memory_backend(h, bus.read, bus.write)
        h.bus = bus
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x80000000
        return h

    # fmt: off
    @pytest.mark.parametrize("addr,size,is_write,expected_cause", [
        # stack overflow below RAM base
        (RAM_BASE - 8,  8, True,  7),   # store below base -> StAccessFault
        (RAM_BASE - 4,  4, False, 5),   # load  below base -> LdAccessFault
        # stack overflow above RAM top
        (RAM_TOP,       4, True,  7),   # store above top  -> StAccessFault
        (RAM_TOP,       4, False, 5),   # load  above top  -> LdAccessFault
        # valid stack range
        (RAM_BASE + 0x1000, 4, False, 0),  # valid load
        (RAM_BASE + 0x1000, 4, True,  0),  # valid store
        # boundary access -- exactly within valid range
        (RAM_BASE,            4, False, 0),  # bottom edge load
        (RAM_TOP - 4,         4, False, 0),  # top edge load
        # 8-byte aligned store at top edge (all bytes in RAM, natural alignment OK)
        (RAM_TOP - 8,         8, True,  0),  # 8-byte at top edge
    ])
    # fmt: on
    def test_stack_boundary_access(self, hart, addr, size, is_write, expected_cause):
        """Stack boundary access: covers overflow direction, op, and edge positions."""
        if not is_write and expected_cause == 0:
            hart._mem_write_phy(addr, b"\x42" * size)

        if is_write:
            mem_write(hart, addr, b"\x00" * size)
        else:
            result = mem_read(hart, addr, size)
            if expected_cause == 0:
                assert result is not None

        assert hart.mcause_val == expected_cause, (
            f"addr=0x{addr:x} size={size} {'write' if is_write else 'read'}: "
            f"expected cause={expected_cause}, actual={hart.mcause_val}"
        )

    def test_consecutive_stack_overflows_count(self, hart):
        """Repeated stack overflows increment consecutive_traps counter."""
        for i in range(5):
            mem_write(hart, self.RAM_BASE - 8, b"\x00" * 4)
            assert hart._consecutive_traps == i + 1, (
                f"trap #{i+1}: consecutive_traps should be {i+1}"
            )


class TestStackOverflowSmode:
    """S-mode stack overflow: Sv39 guard page -> PageFault -> M-mode.

    guard page (0x1000) just below S stack page (0x2000), unmapped.
    Page tables and data all within 2 MiB bytearray.
    PMP NAPOT full address space injected so S-mode accesses pass PMP.
    """

    PAGE_SHIFT = 12
    SATP_MODE_SV39 = 8

    # page table physical layout (low bytearray addresses)
    L1_BASE = 0x10000
    L2_BASE = 0x11000
    L3_BASE = 0x12000

    # VA / PA constants
    STACK_VA = _SO_SMODE_STACK_VA  # 0x2000
    GUARD_VA = _SO_SMODE_GUARD_VA  # 0x1000
    STACK_PA = _SO_SMODE_STACK_PA  # 0x5000

    @pytest.fixture
    def ram_ctx(self):
        """2 MiB bytearray as physical memory."""
        ram = bytearray(2 * 1024 * 1024)

        def read_fn(addr: int, size: int) -> bytes:
            return bytes(ram[addr : addr + size])

        def write_fn(addr: int, data: bytes):
            for i, b in enumerate(data):
                ram[addr + i] = b

        return ram, read_fn, write_fn

    @pytest.fixture
    def hart(self, ram_ctx):
        _ram, read_fn, write_fn = ram_ctx
        h = Hart(id=0)
        inject_memory_backend(h, read_fn, write_fn)
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        return h

    # -- helpers --

    @staticmethod
    def _write_pte(ram: bytearray, table_base: int, index: int, pte_val: int):
        addr = table_base + index * 8
        ram[addr : addr + 8] = pte_val.to_bytes(8, "little")

    @staticmethod
    def _make_pte(*, v=False, r=False, w=False, x=False, u=False, ppn=0) -> int:
        val = 0
        if v:
            val |= 1 << 0
        if r:
            val |= 1 << 1
        if w:
            val |= 1 << 2
        if x:
            val |= 1 << 3
        if u:
            val |= 1 << 4
        val |= (ppn & 0x3FF) << 10
        val |= ((ppn >> 10) & 0x1FF) << 20
        val |= ((ppn >> 19) & 0x1FFFFFFF) << 29
        return val

    def _setup_sv39_stack_only(self, ram: bytearray):
        """Map only S-mode stack page (R+W), guard page stays V=0.

        VA=0x2000: vpn2=0, vpn1=0, vpn0=2
        VA=0x1000: vpn2=0, vpn1=0, vpn0=1
        """
        # L1[0] -> L2
        self._write_pte(
            ram,
            self.L1_BASE,
            0,
            self._make_pte(v=True, ppn=self.L2_BASE >> self.PAGE_SHIFT),
        )
        # L2[0] -> L3
        self._write_pte(
            ram,
            self.L2_BASE,
            0,
            self._make_pte(v=True, ppn=self.L3_BASE >> self.PAGE_SHIFT),
        )
        # L3[2] -> stack physical page (R+W)
        self._write_pte(
            ram,
            self.L3_BASE,
            2,
            self._make_pte(v=True, r=True, w=True, ppn=self.STACK_PA >> self.PAGE_SHIFT),
        )
        # L3[1] stays V=0 -> guard page

    def _enable_sv39(self, hart: Hart):
        hart.satp_val = (self.SATP_MODE_SV39 << 60) | (self.L1_BASE >> self.PAGE_SHIFT)

    def _prep_smode(self, hart: Hart, ram: bytearray):
        """Full S-mode + Sv39 + PMP NAPOT full address space setup."""
        self._setup_sv39_stack_only(ram)
        self._enable_sv39(hart)
        # PMP NAPOT full address space R+W+X, otherwise S-mode accesses are denied
        hart.csrs["pmpcfg0"].val = 0x1F  # NAPOT, R+W+X
        hart.csrs["pmpaddr0"].val = 0x003F_FFFF_FFFF_FFFF  # full address space
        hart.mode = RiscvMode.S

    # fmt: off
    @pytest.mark.parametrize("va,size,is_write,expected_cause", [
        # guard page (unmapped) -> PageFault
        (_SO_SMODE_GUARD_VA, 8, True,  15),  # store guard   -> StPageFault
        (_SO_SMODE_GUARD_VA, 4, False, 13),  # load  guard   -> LdPageFault
        # valid stack page -> success
        (_SO_SMODE_STACK_VA + 0xE0, 4, True,   0),  # store valid
        (_SO_SMODE_STACK_VA,        4, False,  0),  # load  valid
        # cross-page: starts in guard page, extends into stack page
        (_SO_SMODE_GUARD_VA + 0xFF8, 8, True,  15),  # cross-page store
        (_SO_SMODE_GUARD_VA + 0xFF8, 8, False, 13),  # cross-page load
        # last byte of stack page (single-byte, no cross)
        (_SO_SMODE_STACK_VA + 0xFFF, 1, True,   0),  # byte at page end
    ])
    # fmt: on
    def test_guard_page_access(self, hart, ram_ctx, va, size, is_write, expected_cause):
        """Sv39 guard page: parametrized over access direction/boundary/cross-page."""
        ram, _read_fn, _write_fn = ram_ctx
        self._prep_smode(hart, ram)

        # pre-fill physical data for read tests
        if not is_write:
            pa_offset = va - self.STACK_VA
            if 0 <= pa_offset < 0x1000:
                fill_sz = min(size, 0x1000 - pa_offset)
                ram[self.STACK_PA + pa_offset : self.STACK_PA + pa_offset + fill_sz] = (
                    b"\x99" * fill_sz
                )

        if is_write:
            mem_write(hart, va, b"\x00" * size)
        else:
            mem_read(hart, va, size)

        assert hart.mcause_val == expected_cause, (
            f"va=0x{va:x} size={size} {'write' if is_write else 'read'}: "
            f"expected cause={expected_cause}, actual={hart.mcause_val}"
        )
        # PageFault should enter M-mode (medeleg not set for page faults)
        if expected_cause in (13, 15):
            assert hart.mode == RiscvMode.M, (
                f"PageFault should enter M-mode, actual={hart.mode.name}"
            )

    def test_mmode_bare_no_page_fault(self, hart, ram_ctx):
        """M-mode Bare: guard VA treated as PA, no PageFault."""
        ram, _read_fn, _write_fn = ram_ctx
        self._setup_sv39_stack_only(ram)
        # no Sv39 -> Bare; M-mode MPRV=0 bypasses PMP
        hart.mode = RiscvMode.M

        ram[self.GUARD_VA : self.GUARD_VA + 4] = b"\xAB\xCD\xEF\x01"
        data = mem_read(hart, self.GUARD_VA, 4)
        assert data == b"\xAB\xCD\xEF\x01"
        assert hart.mcause_val == 0

    def test_consecutive_guard_page_traps_count(self, hart, ram_ctx):
        """Repeated guard page hits increment consecutive_traps counter."""
        ram, _read_fn, _write_fn = ram_ctx
        self._prep_smode(hart, ram)

        for i in range(5):
            # 每次写前 restore S-mode: deliver_trap 会切到 M 模式,
            # M 模式使用 Bare 翻译 (无视 satp), 后续 guard page 写不再 page fault.
            hart.mode = RiscvMode.S
            hart.pc = 0x1000  # 虚设 S-mode PC, 避免空指针检查问题
            mem_write(hart, self.GUARD_VA, b"\x00" * 4)
            assert hart._consecutive_traps == i + 1, (
                f"guard page trap #{i+1}: consecutive_traps should be {i+1}"
            )


class TestStackOverflowUmode:
    """U-mode stack overflow: Sv39 guard page -> StPageFault delegated to S-mode.

    Uses pre-built firmware u_mode_run_fib.elf:
      M: PMP / medeleg(ECALL+PageFault->S) / MRET->S
      S: Sv39 page tables (U stack 1 page + guard) / stvec / sscratch
      U: well-behaved (bounded input fib) or pathological (stack_bomb)
    """

    ELF_PATH = "tests/bins/elf/u_mode_run_fib.elf"

    @pytest.fixture
    def emu_and_fw(self):
        fw = parse_firmware(self.ELF_PATH)
        if fw is None:
            pytest.skip(f"{self.ELF_PATH} not found or failed to parse")
        emu = Emulator(PlatformConfig.qemu_virt())
        emu.load_firmware(fw)
        assert emu.uart is not None
        return emu, fw

    @staticmethod
    def _boot_to_umode(emu: Emulator):
        """Run firmware until S->U mode transition occurs."""
        h = emu.harts[0]
        for _ in range(3000):
            prev = h.mode
            emu.step()
            if prev == RiscvMode.S and h.mode == RiscvMode.U:
                return

    def _run_from(self, emu: Emulator, fw, entry_sym: str, uart_input: bytes, max_steps=3000):
        """Boot to U-mode, redirect to entry, inject input, run until termination."""
        hart = emu.harts[0]
        uart = emu.uart
        assert uart is not None
        entry = fw.symbols.get(entry_sym)
        assert entry is not None, f"symbol {entry_sym} not found"

        self._boot_to_umode(emu)
        hart.pc = entry
        uart.preload(uart_input)

        for _ in range(max_steps):
            emu.step()
            if b"Process terminated" in uart.tx_data():
                break
        return uart.tx_data().decode("latin-1", errors="replace")

    # fmt: off
    @pytest.mark.parametrize("entry_sym,uart_input,expect_fault,expect_output", [
        # pathological: input 0 -> stack_bomb -> StPageFault -> S terminates
        ("u_mode_bad", b"0\n",  True,  "Fault caught"),
        # well-behaved: input 5 -> fib(5)=5, no fault
        ("u_mode_main", b"5\n", False, "fib(5)=5"),
    ])
    # fmt: on
    def test_stack_guard(self, emu_and_fw, entry_sym, uart_input, expect_fault, expect_output):
        """U-mode guard page: overflow caught by S-mode; normal input passes."""
        emu, fw = emu_and_fw
        out = self._run_from(emu, fw, entry_sym, uart_input)

        if expect_fault:
            assert "Fault caught" in out, (
                f"PageFault should have been caught by S-mode: {out!r}"
            )
        else:
            assert "Fault caught" not in out, f"Should not fault: {out!r}"
            assert "scause" not in out, f"No scause expected: {out!r}"

        assert expect_output in out, f"Expected output {expect_output!r}: {out!r}"

    def test_sscratch_preserved_after_fault(self, emu_and_fw):
        """After stack_bomb fault, S-mode stays alive and sscratch is valid."""
        emu, fw = emu_and_fw
        hart = emu.harts[0]
        uart = emu.uart
        assert uart is not None

        self._boot_to_umode(emu)
        sscratch_before = hart.csrs["sscratch"].val
        assert sscratch_before != 0, "sscratch should be initialized before U-mode"

        hart.pc = fw.symbols["u_mode_bad"]
        uart.preload(b"0\n")

        for _ in range(3000):
            emu.step()
            if b"Process terminated" in uart.tx_data():
                break

        # After fault handling, hart should be in S-mode idle, alive
        assert not hart._halted, "S-mode should remain alive after fault"
        assert hart.mode == RiscvMode.S, (
            f"Should return to S-mode idle, actual={hart.mode.name}"
        )
        # sscratch should be a valid non-zero stack pointer
        assert hart.csrs["sscratch"].val != 0, "sscratch should be non-zero"


# ============================================================
#  stimecmp CSR — Sstc 扩展 (S-mode 直接定时器)
# ============================================================


class TestStimecmpSTI:
    """验证 stimecmp CSR (Sstc) 触发 STI 中断.

    RISC-V Sstc 扩展允许 S 模式通过直接写 stimecmp CSR (0x14D)
    设置定时器, 无需 SBI ecall 往返 M 模式.
    硬件: mtime >= stimecmp > 0 时 STIP 置位 (mip bit 5),
    STIE 使能且 S 模式全局中断使能时投递 STI.
    """

    @pytest.fixture
    def clint(self) -> CLINT:
        return CLINT(num_harts=1)

    @pytest.fixture
    def hart(self, clint) -> Hart:
        h = Hart(id=0)
        h.csrs["mtvec"].val = 0x80000000
        h.csrs["stvec"].val = 0x80004000
        h.pc = 0x1000
        h.mode = RiscvMode.S
        # S 模式全局中断使能 + STIE (bit 5) + mideleg[5]=1 (委派 STI 到 S)
        h.mstatus_val = MSTATUS_SIE
        h.csrs["mie"].val = 1 << 5  # STIE
        h.csrs["mideleg"].val = 1 << 5  # 委派 STI
        h.interrupt_ctrl = clint
        return h

    @staticmethod
    def _cause_is_sti(mcause: int) -> bool:
        """mcause bit63=1, code=5 (STI)."""
        return mcause == 0x8000_0000_0000_0005

    # ---- S-mode 直接触发 ----

    def test_sti_taken_smode(self, hart, clint):
        """mtime >= stimecmp > 0 -> STI 投递到 S 模式."""
        hart.csrs["stimecmp"].val = 50
        clint.tick(100)  # mtime = 100 >= 50

        taken = check_pending_interrupts(hart)
        assert taken, "STI 应被触发"
        # STI 是非委派的 S 模式中断 (S 模式本身就处理它)
        # 在此配置下 (S-mode, 无 mideleg 位), 走 deliver_trap -> S 模式投递
        assert hart.pc == 0x80004000, (
            f"应跳转到 stvec=0x80004000, 实际 pc={hart.pc:#x}"
        )
        assert self._cause_is_sti(hart.scause_val), (
            f"scause 应为 STI (0x8000000000000005), 实际 {hart.scause_val:#018x}"
        )
        assert not hart.sie, "进入 trap 后 SIE 应清零"

    def test_sti_not_taken_before_threshold(self, hart, clint):
        """mtime < stimecmp -> 无中断."""
        hart.csrs["stimecmp"].val = 100
        clint.tick(50)  # mtime = 50 < 100

        taken = check_pending_interrupts(hart)
        assert not taken
        assert hart.pc == 0x1000

    def test_sti_not_taken_stimecmp_zero(self, hart, clint):
        """stimecmp == 0 -> 禁用定时器, 不触发中断."""
        hart.csrs["stimecmp"].val = 0
        clint.tick(100)

        taken = check_pending_interrupts(hart)
        assert not taken

    def test_sti_not_taken_sie_off(self, hart, clint):
        """SIE=0 -> S 模式全局关中断, STI 被阻塞."""
        hart.csrs["stimecmp"].val = 50
        clint.tick(100)
        hart.mstatus_val = 0  # SIE=0

        taken = check_pending_interrupts(hart)
        assert not taken
        assert hart.pc == 0x1000

    def test_sti_not_taken_stie_off(self, hart, clint):
        """mie[5] (STIE) = 0 -> 即使 STIP 挂起也不响应."""
        hart.csrs["stimecmp"].val = 50
        clint.tick(100)
        hart.csrs["mie"].val = 0  # 清除全部中断使能

        taken = check_pending_interrupts(hart)
        assert not taken

    # ---- 委派到 S-mode (来自 M/U 模式) ----

    def test_sti_delegated_from_umode(self, hart, clint):
        """U 模式: mideleg[5]=1 -> STI 委派到 S 模式."""
        hart.csrs["stimecmp"].val = 50
        clint.tick(100)
        hart.mode = RiscvMode.U
        hart.csrs["mideleg"].val = 1 << 5  # 委派 STI
        # U 模式进入时需要有 SIE 使能
        hart.mstatus_val = MSTATUS_MIE | MSTATUS_SIE

        taken = check_pending_interrupts(hart)
        assert taken
        assert hart.pc == 0x80004000, "委派后应跳转到 stvec"
        assert hart.mode == RiscvMode.S
        assert self._cause_is_sti(hart.scause_val), (
            f"scause 应为 STI, 实际 {hart.scause_val:#018x}"
        )
        assert hart.sepc_val == 0x1000

    def test_sti_mip_stip_set(self, hart, clint):
        """验证 mip.STIP 位在条件满足时被置位."""
        hart.csrs["stimecmp"].val = 30
        clint.tick(50)

        taken = check_pending_interrupts(hart)
        assert taken
        # mip bit 5 (STIP) 应被设置
        assert hart.mip_val & (1 << 5), "mip[5] (STIP) 应被置位"

    def test_sti_mip_stip_cleared_after_threshold_update(self, hart, clint):
        """更新 stimecmp 到未来值后 STIP 应清除."""
        hart.csrs["stimecmp"].val = 30
        clint.tick(50)  # mtime >= 30 -> STIP set

        taken = check_pending_interrupts(hart)
        assert taken
        assert hart.mip_val & (1 << 5)

        # 在 trap handler 中: 写 stimecmp 到未来值
        hart.csrs["stimecmp"].val = 1000
        # 清除 mip.STIP (模拟 trap handler 中的 csrc sip, STIP)
        hart.mip_val &= ~(1 << 5)
        # 恢复 SIE (模拟 sret 后)
        hart.sie = True
        hart.pc = 0x1000

        # 下一次检查: stimecmp=1000 > mtime=50 -> 不应再触发
        taken2 = check_pending_interrupts(hart)
        assert not taken2, "stimecmp 已更新到未来值, STI 不应再次触发"


class TestSstatusMstatusLinkage:
    """验证 sstatus (0x100) 是 mstatus (0x300) 的受限视图.

    RISC-V 规范: sstatus 读写必须通过 mstatus 对应位, 两者不可独立.
    """

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.S
        return h

    # ---- SPP 位联动 ----

    def test_sstatus_read_spp_reflects_mstatus(self, hart):
        """读 sstatus.SPP 应返回 mstatus.SPP 的值."""
        # mstatus 初始 SPP=0
        assert (hart.read_csr(0x100) & MSTATUS_SPP) == 0
        # 写 mstatus.SPP=1
        hart.mstatus_val |= MSTATUS_SPP
        assert (hart.read_csr(0x100) & MSTATUS_SPP) != 0, (
            "sstatus.SPP 应反映 mstatus.SPP 的值"
        )

    def test_sstatus_write_spp_updates_mstatus(self, hart):
        """写 sstatus.SPP 应更新 mstatus.SPP."""
        hart.mstatus_val &= ~MSTATUS_SPP  # 清零
        # 通过 write_csr 写 sstatus = SPP
        hart.write_csr(0x100, MSTATUS_SPP)
        assert hart.mstatus_val & MSTATUS_SPP, (
            "写 sstatus.SPP=1 应更新 mstatus.SPP"
        )

    def test_sstatus_read_sie_reflects_mstatus(self, hart):
        """读 sstatus.SIE 应返回 mstatus.SIE 的值."""
        hart.mstatus_val |= MSTATUS_SIE
        assert (hart.read_csr(0x100) & MSTATUS_SIE) != 0, (
            "sstatus.SIE 应反映 mstatus.SIE 的值"
        )
        hart.mstatus_val &= ~MSTATUS_SIE
        assert (hart.read_csr(0x100) & MSTATUS_SIE) == 0

    def test_sstatus_write_sie_updates_mstatus(self, hart):
        """写 sstatus.SIE 应更新 mstatus.SIE."""
        hart.mstatus_val &= ~MSTATUS_SIE
        hart.write_csr(0x100, MSTATUS_SIE)
        assert hart.mstatus_val & MSTATUS_SIE, (
            "写 sstatus.SIE=1 应更新 mstatus.SIE"
        )

    def test_sstatus_write_does_not_leak_to_higher_bits(self, hart):
        """sstatus 写只影响受限视图内的位, 不影响 mstatus 高位 (如 MPP)."""
        original_mstatus = hart.mstatus_val
        # 写 sstatus 为全 1
        hart.write_csr(0x100, 0xFFFF_FFFF_FFFF_FFFF)
        # MPP (bits 11-12) 是 mstatus 独有的, 不应被 sstatus 写改变
        assert (hart.mstatus_val & MSTATUS_MPP) == (original_mstatus & MSTATUS_MPP), (
            "sstatus 写不应影响 mstatus.MPP 等高位字段"
        )

    def test_sstatus_read_excludes_mstatus_only_bits(self, hart):
        """读 sstatus 不应返回 MPP/MPIE 等 mstatus 专有位."""
        hart.mstatus_val = MSTATUS_MPP | MSTATUS_MPIE | MSTATUS_SPP | MSTATUS_SIE
        sstatus_val = hart.read_csr(0x100)
        assert (sstatus_val & MSTATUS_MPP) == 0, "sstatus 不应暴露 mstatus.MPP"
        assert (sstatus_val & MSTATUS_MPIE) == 0, "sstatus 不应暴露 mstatus.MPIE"
        assert (sstatus_val & MSTATUS_SPP) != 0, "sstatus 应包含 SPP"
        assert (sstatus_val & MSTATUS_SIE) != 0, "sstatus 应包含 SIE"

    # ---- 与 trap handler 集成 ----

    def test_spp_set_after_smode_trap(self, hart):
        """S 模式触发 ebreak -> sstatus.SPP 应 = 1 (来自 S 模式)."""
        from pyremu.core.trap_handler import deliver_trap

        hart.mode = RiscvMode.S
        hart.csrs["medeleg"].val = 1 << 3  # 委派 breakpoint
        hart.csrs["stvec"].val = 0x80004000
        hart.pc = 0x2000
        hart.mstatus_val = 0  # 清零 mstatus

        deliver_trap(hart, TrapType.Breakpoint, hart.pc, is_interrupt=False)

        # 陷阱后 mstatus.SPP 应为 1 (来自 S 模式)
        assert hart.mstatus_val & MSTATUS_SPP, (
            "S 模式 ebreak 陷阱后 mstatus.SPP 必须为 1"
        )
        # sstatus 读应反映 SPP=1
        assert hart.read_csr(0x100) & MSTATUS_SPP, (
            "sstatus.SPP 应反映 mstatus.SPP=1"
        )


class TestStimecmpClintSync:
    """验证 stimecmp/stimecmph CSR 写入同步到 CLINT mtimecmp."""

    @pytest.fixture
    def clint(self) -> CLINT:
        return CLINT(num_harts=1)

    @pytest.fixture
    def hart(self, clint) -> Hart:
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.S
        h.interrupt_ctrl = clint
        return h

    def test_stimecmp_write_syncs_to_mtimecmp(self, hart, clint):
        """写 stimecmp CSR -> CLINT mtimecmp 同步更新."""
        hart.write_csr(0x14D, 0x12345_6789_ABCD)
        assert clint._mtimecmp[0] == 0x12345_6789_ABCD, (
            "stimecmp 写入后 CLINT mtimecmp 应同步"
        )

    def test_stimecmp_zero_does_not_trigger_timer(self, hart, clint):
        """stimecmp=0 时 mtimecmp=0, 不触发定时器中断 (条件 mtimecmp>0)."""
        hart.write_csr(0x14D, 0)
        assert clint._mtimecmp[0] == 0
        has, mip, src = clint.check_interrupt(0)
        assert not (mip & (1 << 7)), "stimecmp=0 不应触发 MTIP"

    def test_stimecmp_future_value_triggers_after_tick(self, hart, clint):
        """stimecmp=100, tick(200) -> MTIP 置位."""
        hart.write_csr(0x14D, 100)
        clint.tick(200)
        has, mip, src = clint.check_interrupt(0)
        assert mip & (1 << 7), "mtime(200) >= stimecmp(100) 应触发 MTIP"

    def test_stimecmph_write_merges_to_stimecmp(self, hart):
        """stimecmph (0x15D) 写入高 32 位合并到 stimecmp (RV64)."""
        # 先写低 32 位
        hart.write_csr(0x14D, 0xDEAD_BEEF)
        # 再写高 32 位
        hart.write_csr(0x15D, 0x1234_5678)
        expected = 0x1234_5678_DEAD_BEEF
        assert hart.csrs["stimecmp"].val == expected, (
            f"stimecmph 写入后 stimecmp 应为 {expected:#018x}"
        )
