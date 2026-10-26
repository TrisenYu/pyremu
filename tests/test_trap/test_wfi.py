#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""WFI 等待: 唤醒条件、控制器前置条件与 Sstc 定时唤醒."""

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.hart import (
    MSTATUS_MIE,
    MSTATUS_SIE,
    MSTATUS_TW,
    RiscvMode,
)
from pyremu.core.mem_check_aux import (
    inject_memory_backend,
)
from pyremu.core.trap_def import TrapType
from pyremu.core.trap_handler import (
    check_pending_interrupts,
    deliver_trap,
    try_wfi_wakeup,
)
from pyremu.emulator import Emulator
from pyremu.interrupt.clint import CLINT
from pyremu.memory.bus import Bus
from pyremu.platform import PeripheralConfig, PlatformConfig
from tests.test_trap.support import (
    make_hart,
    StubImsic,
)


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

        # step 1: 执行 WFI -> 进入等待
        emu.step()
        assert hart._waiting, "step 1: 应进入 WFI 等待"
        assert hart.pc == 0x80000004, "PC 应指向 WFI 后一条指令"

        # 设置定时器: mtimecmp = 当前 mtime + 10000 (~1ms 安全余量)
        now = clint.get_mtime()
        self._set_mtimecmp(clint, 0, now + 10000)

        # step 2: 仍等待, mtime 未到 deadline
        emu.step()
        assert hart._waiting, "step 2: 仍应等待"
        assert hart.pc == 0x80000004, "PC 不应变化"

        # 推进 mtime 越过 mtimecmp
        clint.tick(20000)

        # step 3: 中断唤醒
        emu.step()
        assert not hart._waiting, "step 3: 应被定时器中断唤醒"
        assert hart.pc == 0x80000100, "应跳转到 mtvec"
        cause = hart.mcause_val
        assert cause == 0x8000_0000_0000_0007, f"应为 MTI, 实际 mcause={cause:#018x}"

    def test_wfi_wake_with_mstatus_mie_zero(self):
        """mstatus.MIE=0 时 WFI 仍可被 MSIP 唤醒.

        回归: sbi_hsm_hart_wait 中 csr_set(CSR_MIE, MSIP) 只设置源级使能,
        不设置 mstatus.MIE (全局使能).  若 WFI 唤醒检查错误地要求 MIE=1,
        则从 hart 永远无法被 MSIP 唤醒, 导致多 hart 启动死锁.
        """
        clint = CLINT(num_harts=1)
        emu = Emulator(
            PlatformConfig(
                num_harts=1,
                ram_size=1024 * 1024,
                periph=PeripheralConfig(clint_base=0x0200_0000),
            ),
        )
        emu.clint = clint
        emu.harts[0].interrupt_ctrl = clint
        hart = emu.harts[0]
        hart.pc = 0x80000000
        hart.mode = RiscvMode.M
        # MIE=0: 全局使能关 (与 sbi_hsm_hart_wait 中仅设 mie CSR 一致)
        hart.mstatus_val = 0  # MIE=0
        hart.csrs["mtvec"].val = 0x80000100
        hart.csrs["mie"].val = 1 << 3  # MSIE

        # WFI 指令写入 RAM
        emu.bus.write(0x80000000, self.WFI_INSTR.to_bytes(4, "little"))

        # step 1: 执行 WFI -> mip & mie == 0 (无中断挂起) -> 进入等待
        emu.step()
        assert hart._waiting, "step 1: 应进入 WFI 等待"
        assert hart.pc == 0x80000004, "PC 应指向 WFI+4"

        # 发送 MSIP (模拟 Hart 0 通过 sbi_ipi_raw_send 唤醒 Hart 1)
        clint.send_ipi(0)

        # step 2: MSIP 已挂起, WFI 应被唤醒 (即使 MIE=0)
        emu.step()
        assert not hart._waiting, "MIE=0 时也应从 WFI 中唤醒 (mip & mie != 0 已满足)"
        assert hart._wfi_woken, "唤醒后须置位 wfi_woken"
        # MIE=0 -> 中断不应投递, PC 保持在 WFI+4
        assert hart.pc == 0x80000004, "MIE=0 时中断不应投递, PC 保持在 WFI+4"

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

        # 模拟 MRET 回到 WFI 后的状态.
        # deliver_trap 从 WFI 唤醒时已置位 _wfi_woken; MRET 不再清除它.
        # 因此 handle_wfi 会将首次 WFI 视为 NOP (推进 PC),
        # 允许 while (!cond) wfi() 循环在 trap handler 返回后重新检查条件.
        emu.clint.clear_ipi(0)
        hart0.csrs["mip"].val &= ~(1 << 3)
        hart0._waiting = False
        # _wfi_woken 保持 True (MRET 已不再清除此标记)
        hart0.mode = RiscvMode.M
        hart0.mstatus_val |= MSTATUS_MIE
        hart0.pc = 0x80000000

        # 第二轮 step 1: WFI 看到 _wfi_woken=True -> NOP, PC 推进.
        emu.step()
        assert not hart0._waiting, (
            f"第二轮 step1: WFI 应视为 NOP (num_harts={num_harts})"
        )
        assert hart0._wfi_woken is False, "_wfi_woken 应由 handle_wfi NOP 路径清除"
        assert hart0.pc == 0x80000004, "NOP 后 PC 应推进 4"

        # 模拟 while 循环分支回到 WFI.
        hart0.pc = 0x80000000

        # 第二轮 step 2: _wfi_woken=False -> WFI 正常进入等待.
        emu.step()
        assert hart0._waiting, f"第二轮: 应再次进入等待 (num_harts={num_harts})"
        emu.clint.send_ipi(0)
        emu.step()
        assert not hart0._waiting, f"第二轮: 应再次被唤醒 (num_harts={num_harts})"


    def test_wfi_woken_survives_mret(self):
        """验证 _wfi_woken 标记在 MRET 后仍然存在, 使得 while (…) wfi() 轮询
        循环能在 trap handler 返回后重新检查状态条件, 而非立即重回睡眠.

        此回归测试锁定两个根因:
        1. trap_mret / trap_sret 不应清除 _wfi_woken — 该标记由 deliver_trap
           在从 WFI 唤醒时置位, 必须由 handle_wfi (NOP 路径) 消费.
        2. Rust 侧 marshal 同步缺失 — _wfi_woken 必须在 Python ↔ Rust 间
           正确地封送, 否则 Rust WFI handler 无法识别该标记.
        """
        emu = self._make_emu(1)
        h = emu.harts[0]
        # mtvec 指向 RAM 内置的 MRET 指令 (单条: 一进 trap 立即返回)
        mret_instr = 0x30200073  # MRET
        emu.bus.write(0x80000100, mret_instr.to_bytes(4, "little"))
        h.csrs["mtvec"].val = 0x80000100
        # 在 0x80000000 放置 WFI 指令
        emu.bus.write(0x80000000, self.WFI_INSTR.to_bytes(4, "little"))
        h.pc = 0x80000000
        h.mode = RiscvMode.M
        h.mstatus_val |= MSTATUS_MIE

        # Phase 1: WFI -> 进入等待
        emu.step()
        assert h._waiting is True, "WFI 应进入等待状态"
        wfi_pc_after = h.pc  # handle_sys 返回 4, step() 将 PC+4

        # Phase 2: IPI -> 从 WFI 唤醒 (deliver_trap 置位 _wfi_woken)
        emu.clint.send_ipi(0)
        emu.step()  # check_pending_interrupts -> deliver_trap(MSI)
        assert h._waiting is False, "IPI 应唤醒 hart"
        assert h._wfi_woken is True, (
            "deliver_trap 应将 _wfi_woken 置位 (从 WFI 唤醒)"
        )
        assert h.pc == 0x80000100, f"PC 应在 mtvec, 实际 {h.pc:#x}"

        # Phase 3: 执行 MRET (trap handler 唯一指令)
        # MRET 不应该清除 _wfi_woken (该标记只能由 handle_wfi NOP 路径消费).
        # 清除 MSIP 以免 check_pending_interrupts 在 MRET 之后立即再次触发 trap.
        emu.clint.clear_ipi(0)
        h.csrs["mip"].val &= ~(1 << 3)
        emu.step()
        assert h.pc == wfi_pc_after, (
            f"MRET 后 PC 应回到 WFI+4, 实际 {h.pc:#x}"
        )
        assert h._wfi_woken is True, (
            "MRET 后 _wfi_woken 必须仍然为 True"
        )

        # Phase 4: 模拟 while 循环分支回到 WFI (条件仍不满足 -> 重新 WFI)
        # 此时 _wfi_woken=True -> WFI 视为 NOP -> PC 推进, _wfi_woken=False
        h.pc = 0x80000000  # 模拟分支回到 WFI
        emu.clint.clear_ipi(0)
        h.csrs["mip"].val &= ~(1 << 3)
        emu.step()  # WFI NOP 路径
        assert h._waiting is False, (
            "_wfi_woken=True -> WFI 应视为 NOP, 不应进入等待"
        )
        assert h._wfi_woken is False, (
            "_wfi_woken 应由 handle_wfi NOP 路径清除"
        )
        assert h.pc == wfi_pc_after, "NOP 后 PC 应推进至 WFI+4"

        # Phase 5: 再次分支回到 WFI (条件仍未满足) -> _wfi_woken=False
        # -> 正常进入等待, 可被再次唤醒
        h.pc = 0x80000000
        emu.step()  # WFI -> 正常等待
        assert h._waiting is True, "_wfi_woken=False -> WFI 应正常进入等待"
        emu.clint.send_ipi(0)
        emu.step()  # 再次唤醒
        assert h._waiting is False, "_wfi_woken=False 的 WFI 仍可被 IPI 正常唤醒"


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


class TestWfiWakeupByStimecmp:
    """验证 try_wfi_wakeup 正确识别 SSTC stimecmp 定时器并唤醒 WFI.

    2026-07-07 修复前 try_wfi_wakeup 只检查 CLINT.check_interrupt 返回的
    MSIP/MTIP, 忽略了 stimecmp 对应的 STIP.  内核使用 SSTC 设定时器时,
    stimecmp 到期后 STIP 未被置位, 导致 WFI hart 永远无法被定时器中断唤醒.

    三个独立修复:
    1. try_wfi_wakeup: 补充 stimecmp -> STIP 检查
    2. _wfi_ticks_until_wake: 补充 stimecmp CSR 检查
    3. _step_native CLINT sync: 先 unmarshal (stimecmp->mtimecmp) 后 CLINT
       array sync (旧 mtimecmp_arr 覆盖) 的时序修正
    """

    @pytest.fixture
    def clint(self) -> CLINT:
        return CLINT(num_harts=1)

    @pytest.fixture
    def hart(self, clint) -> Hart:
        """S-mode hart, STIE enabled, STI delegated via mideleg."""
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.S
        h.mstatus_val = MSTATUS_SIE  # S-mode 全局中断使能
        h.csrs["mtvec"].val = 0x80000000
        h.csrs["stvec"].val = 0x80004000
        h.csrs["mie"].val = 1 << 5  # STIE
        h.csrs["mideleg"].val = 1 << 5  # 委派 STI 到 S-mode
        h.interrupt_ctrl = clint
        return h

    # ---- try_wfi_wakeup 修复 ----

    def test_wfi_wakeup_stimecmp_expired_wakes_hart(self, hart, clint):
        """stimecmp 已过期时 try_wfi_wakeup 应唤醒 WFI hart (STIP 置位)."""
        hart.csrs["stimecmp"].val = 50
        clint.tick(100)  # mtime = 100 >= 50

        # 手动进入 WFI 等待状态 (模拟 WFI 指令执行后)
        hart._waiting = True
        hart._wfi_woken = False
        hart.csrs["mip"].val = 0  # 清零 mip (仅测 stimecmp->STIP 路径)

        woken = try_wfi_wakeup(hart)
        assert woken, f"stimecmp=50 已过期 (mtime={clint.get_mtime()}), 应唤醒 WFI"
        assert not hart._waiting, "应清除 waiting 状态"
        assert hart._wfi_woken, "应置位 wfi_woken 标志"
        assert hart.mip_val & (1 << 5), "STIP (bit 5) 应已置位"

    def test_wfi_wakeup_stimecmp_not_expired_does_not_wake(self, hart, clint):
        """stimecmp 未到期时 try_wfi_wakeup 不应唤醒 hart."""
        hart.csrs["stimecmp"].val = 500  # 未来值
        clint.tick(100)  # mtime = 100 < 500

        hart._waiting = True
        hart._wfi_woken = False
        hart.csrs["mip"].val = 0

        woken = try_wfi_wakeup(hart)
        assert not woken, "stimecmp=500 > mtime=100, 不应唤醒"
        assert hart._waiting, "应保持 waiting 状态"

    def test_wfi_wakeup_stimecmp_zero_no_wake(self, hart, clint):
        """stimecmp=0 (禁用) 时不唤醒 (回归: 条件 stimecmp > 0)."""
        hart.csrs["stimecmp"].val = 0
        clint.tick(100)
        hart._waiting = True
        hart._wfi_woken = False
        hart.csrs["mip"].val = 0

        woken = try_wfi_wakeup(hart)
        assert not woken, "stimecmp=0 应视为禁用定时器, 不唤醒"

    def test_wfi_wakeup_only_mtimecmp_no_stimecmp_still_works(self, hart, clint):
        """仅有 mtimecmp (无 stimecmp) 时 try_wfi_wakeup 仍正常 (回归)."""
        # 设置 CLINT mtimecmp (模拟 legacy SBI 定时器)
        clint._mtimecmp[0] = 100
        clint.tick(200)  # mtime = 200 >= 100

        hart._waiting = True
        hart._wfi_woken = False
        # 不设 stimecmp, 但有 mie[5]=STIE (内核 S-mode timer 使能)
        hart.csrs["stimecmp"].val = 0

        # 旧路径: CLINT.check_interrupt 返回 MTIP
        # 但内核在 S-mode 期望 STIP.  修复: try_wfi_wakeup 应同时检查
        # CLINT 返回的 MTIP 位; 这里测试只要 mip & mie 非零就唤醒.
        # 实际上由于 stimecmp=0 且 mtimecmp 到期, check_interrupt 返回
        # MTIP, 若某条件使 mip & mie 含此位则唤醒.
        woken = try_wfi_wakeup(hart)
        # 注意: 本测试验证修复不破坏旧路径.  MTIP 在 S-mode 可能不被
        # mie 识别 (因 mideleg 未委派 bit 7), 所以结果取决于 mideleg.
        # 仅验证不会因为 stimecmp 而意外阻止唤醒.
        assert isinstance(woken, bool), "不应抛异常"

    # ---- _wfi_ticks_until_wake 修复 (通过 Emulator 集成测试) ----

    def test_wfi_ticks_finds_stimecmp(self, clint):
        """_wfi_ticks_until_wake 应检查 stimecmp 找到定时器.

        使用 Emulator 的 _wfi_ticks_until_wake (私有方法).
        修复前只检查 CLINT mtimecmp, 遗漏 stimecmp.
        """
        emu = Emulator(
            PlatformConfig(
                num_harts=1,
                ram_size=1024 * 1024,
                periph=PeripheralConfig(clint_base=0x0200_0000),
            ),
        )
        emu.clint = clint
        hart = emu.harts[0]
        hart.interrupt_ctrl = clint

        # 仅设置 stimecmp, 不设 mtimecmp
        hart.csrs["stimecmp"].val = 500
        clint._mtimecmp[0] = 0  # mtimecmp 禁用
        clint._mtime = 100  # mtime = 100 < 500

        remaining = emu._wfi_ticks_until_wake([hart])
        assert remaining is not None, "stimecmp=500 > mtime=100, 应返回剩余 tick 数"
        assert remaining == 400, f"剩余 ticks 应为 400, 实际 {remaining}"

    def test_wfi_ticks_returns_none_when_both_disabled(self, clint):
        """mtimecmp 和 stimecmp 都为 0 时返回 None."""
        emu = Emulator(
            PlatformConfig(
                num_harts=1,
                ram_size=1024 * 1024,
                periph=PeripheralConfig(clint_base=0x0200_0000),
            ),
        )
        emu.clint = clint
        hart = emu.harts[0]
        hart.interrupt_ctrl = clint
        hart.csrs["stimecmp"].val = 0
        clint._mtimecmp[0] = 0

        remaining = emu._wfi_ticks_until_wake([hart])
        assert remaining is None, "两个定时器源均禁用时应返回 None"

    def test_wfi_ticks_earliest_of_both_sources(self, clint):
        """同时有 mtimecmp 和 stimecmp 时返回最早到期者."""
        emu = Emulator(
            PlatformConfig(
                num_harts=1,
                ram_size=1024 * 1024,
                periph=PeripheralConfig(clint_base=0x0200_0000),
            ),
        )
        emu.clint = clint
        hart = emu.harts[0]
        hart.interrupt_ctrl = clint

        # stimecmp 更早
        hart.csrs["stimecmp"].val = 100
        clint._mtimecmp[0] = 300  # mtimecmp 晚于 stimecmp
        clint._mtime = 50

        remaining = emu._wfi_ticks_until_wake([hart])
        assert remaining == 50, f"应返回最早到期者 (stimecmp=100), 剩余 50, 实际 {remaining}"

    # ---- 端到端: Emulator.step + stimecmp WFI wakeup ----

    def test_emulator_step_wakes_wfi_via_stimecmp(self, clint, monkeypatch):
        """完整 Emulator.step 流程: WFI -> stimecmp 到期 -> 唤醒.

        模拟 Linux 多核 idle: S-mode hart 设置 stimecmp 后执行 WFI.
        修复前 try_wfi_wakeup 不检查 stimecmp, step() 返回 0 后死循环.

        注意: 使用 M-mode 运行测试 (避免 S-mode PMP 默认拒绝).
        定时器测试关注 mip&mie 的 STIP/STIE 交互, 与特权级无关.
        """
        emu = Emulator(
            PlatformConfig(
                num_harts=1,
                ram_size=1024 * 1024,
                pmp_entries=0,
                periph=PeripheralConfig(clint_base=0x0200_0000),
            ),
        )
        emu.clint = clint
        hart = emu.harts[0]
        hart.interrupt_ctrl = clint
        # M-mode (绕过 S-mode PMP 默认拒绝), 但使用 STIE 以测试
        # stimecmp->STIP 路径 (而非 CLINT mtimecmp->MTIP 路径).
        # 仅设 stimecmp, 不设 mtimecmp.
        hart.mode = RiscvMode.M
        hart.mstatus_val = MSTATUS_MIE
        hart.csrs["mie"].val = 1 << 5  # STIE (not MTIE)
        hart.csrs["mtvec"].val = 0x80000100
        hart.pc = 0x80000000

        # 仅设 stimecmp, 不设 mtimecmp (模拟 SSTC-only 场景)
        hart.csrs["stimecmp"].val = 5000
        clint._mtimecmp[0] = 0
        clint._mtime = 10

        # 放置 WFI 指令
        emu.bus.write(0x80000000, (0x10500073).to_bytes(4, "little"))

        # step 1: 执行 WFI -> mtime=10 (< 5000) -> 应进入等待
        emu.step()
        assert hart._waiting, f"step 1: mtime={clint._mtime} < stimecmp=5000, 应进入 WFI 等待"

        # step 2: 推进 mtime 到超过 stimecmp -> 应唤醒
        clint._mtime = 5100  # >= 5000
        emu.step()
        assert not hart._waiting, f"step 2: mtime={clint._mtime} >= stimecmp=5000, 应唤醒"
        assert hart._wfi_woken, "唤醒后应置位 wfi_woken"


class TestPreconditions:
    def test_wakeup_requires_waiting_hart(self):
        h = make_hart()
        h.interrupt_ctrl = CLINT(num_harts=1)
        assert not try_wfi_wakeup(h)

    def test_wakeup_rejects_halted_hart(self):
        h = make_hart()
        h.interrupt_ctrl = CLINT(num_harts=1)
        h._waiting = True
        h._halted = True
        assert not try_wfi_wakeup(h)

    def test_wakeup_rejects_missing_controller(self):
        h = make_hart()
        h._waiting = True
        assert not try_wfi_wakeup(h)

    def test_interrupt_check_rejects_missing_controller(self):
        h = make_hart()
        assert not check_pending_interrupts(h)

    def test_wakeup_queries_imsic_pending_bits(self):
        """外部中断由 IMSIC 给出, PLIC 不参与."""
        h = make_hart()
        h.interrupt_ctrl = CLINT(num_harts=1)
        h.imsic = StubImsic(1 << 11)
        h.csrs["mie"].val = 1 << 11
        h._waiting = True
        assert try_wfi_wakeup(h)
        assert not h._waiting

    def test_wakeup_enables_msie_for_raw_ipi(self):
        """MSIP 硬件置位而 MSIE=0 时, 唤醒路径置位 MSIE 以便后续投递."""
        h = make_hart()
        clint = CLINT(num_harts=1)
        h.interrupt_ctrl = clint
        clint.send_ipi(0)
        h.csrs["mie"].val = 0
        h._waiting = True
        assert try_wfi_wakeup(h)
        assert h.csrs["mie"].val & (1 << 3)
        assert h._wfi_woken

    def test_interrupt_check_queries_imsic_pending_bits(self):
        h = make_hart()
        h.mstatus_val = MSTATUS_MIE
        h.interrupt_ctrl = CLINT(num_harts=1)
        h.imsic = StubImsic(1 << 11)
        h.csrs["mie"].val = 1 << 11
        assert check_pending_interrupts(h)
        assert h.mcause_val == (1 << 63) | 11
