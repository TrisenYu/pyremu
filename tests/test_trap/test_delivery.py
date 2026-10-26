#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""陷态投递: M/S 模式入口、委派、嵌套投递与 IPI 清除."""

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.hart import (
    MSTATUS_MIE,
    MSTATUS_SIE,
    RiscvMode,
)
from pyremu.core.mem_check_aux import (
    inject_memory_backend,
)
from pyremu.core.trap_def import TrapType
from pyremu.core.trap_handler import (
    check_pending_interrupts,
    deliver_trap,
    deliver_trap_nested_enabled,
    trap_ecall,
    trap_sret,
)
from pyremu.interrupt.clint import CLINT
from pyremu.memory.bus import Bus
from tests.test_trap.support import (
    make_hart,
    MTVEC,
    StubImsic,
    STVEC,
)


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


class TestNestedDelivery:
    def test_exception_in_mmode_uses_mtvec(self):
        h = make_hart()
        deliver_trap_nested_enabled(h, TrapType.IllInstr, tval=0xDEAD)
        assert h.mcause_val == 2
        assert h.mtval_val == 0xDEAD
        assert h.mepc_val == 0x8000_4000
        assert h.pc == MTVEC
        assert h.mode is RiscvMode.M

    def test_exception_delegated_by_medeleg(self):
        h = make_hart(RiscvMode.S)
        h.csrs["medeleg"].val = 1 << 12
        deliver_trap_nested_enabled(h, TrapType.InstrPageFault, tval=0x1000)
        assert h.mode is RiscvMode.S
        assert h.pc == STVEC
        assert h.scause_val == 12
        assert h.stval_val == 0x1000

    def test_interrupt_delegated_by_mideleg(self):
        h = make_hart(RiscvMode.U)
        h.csrs["mideleg"].val = 1 << 7
        deliver_trap_nested_enabled(h, TrapType.MmodeTimerInterrupt, is_interrupt=True)
        assert h.mode is RiscvMode.S
        assert h.scause_val == (1 << 63) | 7
        assert h.sepc_val == 0x8000_4000

    def test_reservation_is_cleared(self):
        h = make_hart()
        h.set_reservation(0x8000_2000, 8)
        deliver_trap_nested_enabled(h, TrapType.Breakpoint)
        assert not h.reservation_valid

    def test_wfi_waiter_is_woken(self):
        h = make_hart()
        h._waiting = True
        deliver_trap_nested_enabled(h, TrapType.Breakpoint)
        assert not h._waiting
        assert h._wfi_woken


class TestDeliveryBranches:
    def test_ssip_trap_clears_imsic_eip(self):
        h = make_hart(RiscvMode.U)
        imsic = StubImsic()
        h.imsic = imsic
        h.csrs["mideleg"].val = 1 << 1
        deliver_trap(h, TrapType.SmodeSoftInterrupt, is_interrupt=True)
        assert imsic.cleared == [(0, 1)]
        assert not (h.mip_val & (1 << 1))

    def test_msip_trap_clears_imsic_eip(self):
        h = make_hart()
        imsic = StubImsic()
        h.imsic = imsic
        deliver_trap(h, TrapType.MmodeSoftInterrupt, is_interrupt=True)
        assert imsic.cleared == [(0, 3)]

    def test_hypervisor_ecall_traps_ill_instr(self):
        h = make_hart(RiscvMode.H)
        trap_ecall(h)
        assert h.mcause_val == 2
        assert h.mtval_val == 0x8000_4000


class TestMsipClearingOnTrapDelivery:
    """验证 MSIP 中断投递时 CLINT 硬件源 (_msip) 被正确清零.

    RISC-V 规范: mip.MSIP 是只读位, 反映 CLINT MSIP 寄存器.
    若 M-mode (或委派到 S-mode) 响应 MSIP 后不清零硬件源,
    mret/sret 后 mip.MSIP 持续为 1 -> 重复投递 -> 死循环 (MSIP 风暴).

    回归: 多核 TLB shootdown 中 Hart 0 在 tlb_sync 自旋等待,
    Hart 1 的 MSIP handler 若未清零 CLINT -> Hart 0 永不休眠 -> 饿死.
    """

    WFI_INSTR = 0x10500073
    NOP_INSTR = 0x00000013

    @staticmethod
    def _make_hart_with_clint(
        hart_id: int = 0,
        num_harts: int = 2,
        mode=RiscvMode.M,
    ):
        """创建带 Bus + CLINT 的 Hart."""
        clint = CLINT(num_harts=num_harts)
        clint.base_addr = 0x0200_0000
        bus = Bus(ram_size=1024 * 1024, ram_base=0x8000_0000)
        bus.add_device(clint.base_addr, clint)

        h = Hart(id=hart_id)
        h.pc = 0x80000000
        h.mode = mode
        # M-mode: MIE.  S-mode: SIE (required for delegated interrupt delivery).
        if mode == RiscvMode.S:
            h.mstatus_val = MSTATUS_MIE | MSTATUS_SIE
        else:
            h.mstatus_val = MSTATUS_MIE
        h.csrs["mtvec"].val = 0x80000100
        h.csrs["stvec"].val = 0x80000400
        h.csrs["mie"].val = 1 << 3  # MSIE
        h.interrupt_ctrl = clint
        inject_memory_backend(h, bus.read, bus.write)
        h.bus = bus

        return h, clint, bus

    def test_deliver_msip_to_mmode_clears_clint_msip(self):
        """MSIP 投递到 M 模式后 CLINT _msip 必须为 0."""
        h, clint, _ = self._make_hart_with_clint(hart_id=0, mode=RiscvMode.S)

        # S 模式运行 (模拟 Linux kernel), MSIP 已置位
        clint.send_ipi(0)
        assert clint._msip[0] == 1, "前置: _msip 应为 1"

        # 触发中断检查 -> deliver_trap(MmodeSoftInterrupt) -> _trap_deliver_mmode
        taken = check_pending_interrupts(h)
        assert taken, "MSIP 应被投递"
        assert h.mode == RiscvMode.M, "应切换到 M 模式"
        assert h.mcause_val == 0x8000_0000_0000_0003, "mcause 应为 MSI, 实际={h.mcause_val:#x}"

        # 关键断言: CLINT _msip 已被清零
        assert clint._msip[0] == 0, (
            f"投递 MSIP 到 M 模式后 _msip 必须清零, 实际={clint._msip[0]}"
        )
        # mip CSR 也应已清零 MSIP 位
        assert not (h.mip_val & (1 << 3)), f"投递后 mip.MSIP 必须为 0, 实际 mip={h.mip_val:#x}"

    def test_deliver_msip_delegated_to_smode_clears_clint_msip(self):
        """MSIP 委派到 S 模式 (mideleg[3]=1) 后 CLINT _msip 必须为 0.

        回归: 若 MSIP 被 mideleg 委派到 S 模式, _trap_deliver_smode 必须
        清理 CLINT 硬件源, 与 _trap_deliver_mmode 对称.
        """
        h, clint, _ = self._make_hart_with_clint(hart_id=0, mode=RiscvMode.S)

        # 委派 MSIP 到 S 模式
        h.csrs["mideleg"].val |= 1 << 3  # mideleg[3] = 1
        assert h.csrs["mideleg"].val & (1 << 3), "前置: mideleg[3] 必须置位"

        # S 模式运行, MSIP 已置位
        clint.send_ipi(0)
        assert clint._msip[0] == 1, "前置: _msip 应为 1"

        # 触发中断检查 -> deliver_trap -> 委派到 _trap_deliver_smode
        taken = check_pending_interrupts(h)
        assert taken, "MSIP 应被投递"
        assert h.mode == RiscvMode.S, "委派中断应保持 S 模式"
        assert h.scause_val == 0x8000_0000_0000_0003, (
            f"scause 应为 MSI, 实际={h.scause_val:#x}"
        )

        # 关键断言: CLINT _msip 已被清零
        assert clint._msip[0] == 0, (
            f"投递 MSIP 到 S 模式后 _msip 必须清零, 实际={clint._msip[0]}"
        )
        assert not (h.mip_val & (1 << 3)), f"投递后 mip.MSIP 必须为 0, 实际 mip={h.mip_val:#x}"

    def test_repeated_msip_delivery_does_not_storm(self):
        """连续投递 MSIP 30 次 -> 每次 CLINT _msip 被清零, 不形成风暴.

        模拟 bug_rec.txt 中观察到的场景: MSIP 连续投递 30 次且 mepc
        始终不变.  若每次投递后 _msip 被清零但被立即重新断言 (如另一个
        hart 持续发 IPI), 则 mepc 不变是正常的 — 中断每次都在同一
        指令处发生.
        """
        h, clint, _ = self._make_hart_with_clint(hart_id=0, mode=RiscvMode.S)

        mepc_values: list[int] = []
        msip_after_clear: list[int] = []

        for _ in range(30):
            # 重新断言 MSIP (模拟另一个 hart 发 IPI)
            clint.send_ipi(0)
            assert clint._msip[0] == 1

            saved_pc = h.pc
            taken = check_pending_interrupts(h)

            if taken:
                # 记录 mepc (中断发生时 PC)
                mepc_values.append(h.mepc_val)

            # 验证: 投递后 CLINT _msip 必须为 0
            msip_after_clear.append(clint._msip[0])
            assert clint._msip[0] == 0, (
                f"第 {_} 次投递后 _msip 必须清零, 实际={clint._msip[0]}"
            )
            assert not (h.mip_val & (1 << 3)), f"第 {_} 次投递后 mip.MSIP 必须为 0"

            # 模拟 MRET 回到中断上下文 (相同的 PC)
            h.mode = RiscvMode.S
            h.pc = saved_pc
            # 恢复 mstatus (MIE=1, MPIE=1)
            h.mstatus_val = MSTATUS_MIE | (1 << 7)

        # 所有 30 次投递后 _msip 均为 0
        assert all(v == 0 for v in msip_after_clear), (
            f"所有 30 次投递后 _msip 必须为 0, 实际={msip_after_clear}"
        )
        # mepc 每次相同 (因为每次重新断言 MSIP 时 PC 不变)
        assert len(mepc_values) == 30

    def test_deliver_msip_then_mret_no_residual_msip(self):
        """MSIP 投递 -> MRET -> 下一指令边界检查: 无残留 MSIP.

        完整的 trap -> return 循环: MSIP 投递到 M 模式, handler 执行后
        mret 返回 S 模式, check_pending_interrupts 不应再检测到 MSIP.
        """
        h, clint, bus = self._make_hart_with_clint(hart_id=0, mode=RiscvMode.S)

        # 在 mepc 处放一条 NOP (将从中断返回处执行)
        ret_addr = 0x80001000
        bus.write(ret_addr, self.NOP_INSTR.to_bytes(4, "little"))

        # 1. 断言 MSIP
        clint.send_ipi(0)
        h.pc = ret_addr  # 模拟在 ret_addr 处被中断

        # 2. 投递 MSIP -> M 模式
        taken = check_pending_interrupts(h)
        assert taken
        assert h.mode == RiscvMode.M
        assert h.mepc_val == ret_addr, f"mepc 应为 ret_addr, 实际={h.mepc_val:#x}"

        # 3. 验证 MSIP 已清零
        assert clint._msip[0] == 0
        assert not (h.mip_val & (1 << 3))

        # 4. 执行 MRET -> 返回 S 模式, PC = mepc
        h.exec_instr(0x30200073)  # MRET
        assert h.mode == RiscvMode.S, f"MRET 后应为 S 模式, 实际={h.mode}"
        assert h.pc == ret_addr, f"MRET 后 PC 应为 ret_addr, 实际={h.pc:#x}"

        # 5. 执行 NOP, 然后检查中断 — 不应有 MSIP
        h.exec_instr(self.NOP_INSTR)
        taken2 = check_pending_interrupts(h)
        assert not taken2, (
            f"MRET 后不应再有 MSIP 中断, mip={h.mip_val:#x} _msip={clint._msip[0]}"
        )
        assert clint._msip[0] == 0, "MRET 后 _msip 仍应为 0"
        assert not (h.mip_val & (1 << 3)), "MRET 后 mip.MSIP 仍应为 0"
