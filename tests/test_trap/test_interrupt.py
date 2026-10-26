#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""中断: 定时器、Sstc, SIP/MIP 只读掩码与中断检查顺序."""

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.hart import (
    MSTATUS_MIE,
    MSTATUS_SIE,
    RiscvMode,
)
from pyremu.core.trap_handler import (
    check_pending_interrupts,
    check_pending_interrupts_nested_enabled,
)
from pyremu.interrupt.clint import CLINT
from tests.test_trap.support import (
    make_hart,
    MTVEC,
    StubImsic,
    StubPlic,
)


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
        self._set_mtimecmp(clint, 0, 200)
        clint.tick(150)  # mtime = 250 >= 200

        # 第二次中断
        taken = check_pending_interrupts(hart)
        assert taken
        assert self._cause_is_timer(hart.mcause_val)


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
        assert hart.pc == 0x80004000, f"应跳转到 stvec=0x80004000, 实际 pc={hart.pc:#x}"
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


class TestStimecmpClintSync:
    """验证 stimecmp 与 CLINT mtimecmp 互相独立 (匹配 QEMU / 真实硬件).

    QEMU 验证: time_helper.c 中 riscv_timer_write_timecmp() 仅管理 STIP;
    riscv_aclint.c 中 riscv_aclint_mtimer_write_timecmp() 仅管理 MTIP.
    两者是完全独立的比较器。"""

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

    def test_stimecmp_independent_of_mtimecmp(self, hart, clint):
        """写 stimecmp CSR 不影响 CLINT mtimecmp (独立比较器)."""
        # Set mtimecmp directly (simulating OpenSBI)
        clint.set_mtimecmp(0, 0xFFFF_FFFF_FFFF_FFFF)
        # Write stimecmp — must NOT overwrite mtimecmp
        hart.write_csr(0x14D, 0x12345_6789_ABCD)
        assert clint._mtimecmp[0] == 0xFFFF_FFFF_FFFF_FFFF, (
            "stimecmp 写入后 CLINT mtimecmp 应保持不变 (独立比较器)"
        )
        assert hart.csrs["stimecmp"].val == 0x12345_6789_ABCD, "stimecmp CSR 应正确存储写入值"

    def test_mtimecmp_independent_of_stimecmp(self, hart, clint):
        """写 CLINT mtimecmp (MMIO) 不影响 stimecmp CSR."""
        hart.write_csr(0x14D, 0xAAAA_BBBB_CCCC_DDDD)
        clint.set_mtimecmp(0, 0xDEAD_BEEF)
        assert hart.csrs["stimecmp"].val == 0xAAAA_BBBB_CCCC_DDDD, (
            "CLINT mtimecmp 写入后 stimecmp 应保持不变 (独立比较器)"
        )

    def test_stimecmp_zero_does_not_affect_mtimecmp(self, hart, clint):
        """stimecmp=0 不影响已设置的 mtimecmp."""
        clint.set_mtimecmp(0, 0x100)
        hart.write_csr(0x14D, 0)
        assert clint._mtimecmp[0] == 0x100, "stimecmp=0 写入后 mtimecmp 应保持不变"

    def test_clint_check_interrupt_uses_mtimecmp_not_stimecmp(self, hart, clint):
        """CLINT.check_interrupt 的 MTIP 仅取决于 mtimecmp, 不取决于 stimecmp."""
        # Set stimecmp far in the "past" — MTIP should NOT fire
        hart.write_csr(0x14D, 100)
        clint.set_mtimecmp(0, 0xFFFFFFFF_FFFFFFFF)  # far future
        clint.tick(200)
        has, mip, _src = clint.check_interrupt(0)
        assert not (mip & (1 << 7)), "MTIP 仅取决于 CLINT mtimecmp; stimecmp 到期不应触发 MTIP"

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


class TestSipMipReadonlyMask:
    """验证 sip (0x144) / mip (0x344) 的硬件只读位不得被软件覆写.

    与 sie 不同, sip 并非对所有 mideleg 委派位都可写: 唯一可写位是 SSIP (bit 1),
    STIP (bit 5) 由 SSTC 定时器、SEIP (bit 9) 由 PLIC 硬件驱动. 飞地上下文切换
    (csr_write(CSR_SIP, saved)) 写回陈旧 sip, 若把 SEIP/STIP 一并覆写会丢失挂起的
    外设/定时器中断 (飞地回归测试后输入冻结)."""

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.S
        return h

    def test_csrw_sip_preserves_stip_and_seip(self, hart):
        """`csrw sip, 0` 只清除 SSIP, 保留硬件驱动的 STIP/SEIP."""
        hart.csrs["mideleg"].val = (1 << 5) | (1 << 9) | (1 << 1)  # STIP/SEIP/SSIP 委派
        hart.csrs["mip"].val = (1 << 5) | (1 << 9) | (1 << 1)
        hart.write_csr(0x144, 0)  # csrw sip, 0
        mip = hart.mip_val
        assert mip & (1 << 5), "STIP 由 SSTC 硬件驱动, 不得被 csrw sip 覆写"
        assert mip & (1 << 9), "SEIP 由 PLIC 硬件驱动, 不得被 csrw sip 覆写"
        assert not (mip & (1 << 1)), "SSIP 是 sip 唯一可写位, 应被清除"

    def test_csrw_mip_preserves_stip_and_seip(self, hart):
        """`csrw mip, 0` 只清除软件可写位 (SSIP/USIP), 保留硬件只读位."""
        hart.csrs["mip"].val = (1 << 5) | (1 << 9) | (1 << 1)
        hart.write_csr(0x344, 0)  # csrw mip, 0
        mip = hart.mip_val
        assert mip & (1 << 5), "STIP 只读, 不得被 csrw mip 覆写"
        assert mip & (1 << 9), "SEIP 只读, 不得被 csrw mip 覆写"
        assert not (mip & (1 << 1)), "SSIP 软件可写, 应被清除"


class TestNestedInterruptCheck:
    @pytest.fixture
    def clint(self) -> CLINT:
        return CLINT(num_harts=1)

    @pytest.fixture
    def hart(self, clint) -> Hart:
        h = make_hart()
        h.mstatus_val = MSTATUS_MIE
        h.interrupt_ctrl = clint
        return h

    def test_missing_controller_reports_none(self):
        h = make_hart()
        assert not check_pending_interrupts_nested_enabled(h)

    def test_idle_timer_is_cached_then_delivered(self, hart, clint):
        """首次全量检查无中断, 之后命中缓存, 到期后投递."""
        clint.set_mtimecmp(0, 100)
        hart.csrs["mie"].val = 1 << 7
        clint.tick(50)
        assert not check_pending_interrupts_nested_enabled(hart)
        assert not check_pending_interrupts_nested_enabled(hart), "第二次应命中缓存"

        clint.tick(50)
        assert check_pending_interrupts_nested_enabled(hart)
        assert hart.mcause_val == (1 << 63) | 7
        assert hart.pc == MTVEC

    def test_mmode_global_disable_blocks(self, hart, clint):
        clint.send_ipi(0)
        hart.csrs["mie"].val = 1 << 3
        hart.mstatus_val = 0
        assert not check_pending_interrupts_nested_enabled(hart)

    def test_source_disable_blocks(self, hart, clint):
        clint.set_mtimecmp(0, 50)
        clint.tick(100)
        hart.csrs["mie"].val = 0
        assert not check_pending_interrupts_nested_enabled(hart)

    def test_delegated_interrupt_needs_smode_global_enable(self, hart, clint):
        clint.set_mtimecmp(0, 50)
        clint.tick(100)
        hart.csrs["mie"].val = 1 << 7
        hart.csrs["mideleg"].val = 1 << 7
        hart.mode = RiscvMode.S
        hart.mstatus_val = MSTATUS_MIE
        assert not check_pending_interrupts_nested_enabled(hart)

    def test_stimecmp_sets_stip(self, hart, clint):
        """STIP 由 stimecmp 与 mtime 的比较结果给出."""
        clint.tick(100)
        hart.csrs["stimecmp"].val = 50
        hart.csrs["mie"].val = 1 << 5
        assert check_pending_interrupts_nested_enabled(hart)
        assert hart.mcause_val == (1 << 63) | 5

    def test_plic_pending_bit_is_delivered(self, hart):
        hart.plic = StubPlic(1 << 11)
        hart.csrs["mie"].val = 1 << 11
        assert check_pending_interrupts_nested_enabled(hart)
        assert hart.mcause_val == (1 << 63) | 11

    def test_imsic_pending_bit_is_delivered(self, hart):
        hart.imsic = StubImsic(1 << 9)
        hart.csrs["mie"].val = 1 << 9
        assert check_pending_interrupts_nested_enabled(hart)
        assert hart.mcause_val == (1 << 63) | 9

    def test_wfi_waiter_is_woken_by_nested_delivery(self, hart, clint):
        clint.send_ipi(0)
        hart.csrs["mie"].val = 1 << 3
        hart._waiting = True
        assert check_pending_interrupts_nested_enabled(hart)
        assert not hart._waiting
        assert hart._wfi_woken
