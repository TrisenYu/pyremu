#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""Trap handler mixin: trap delivery, delegation, interrupt checking, WFI.

Provides TrapHandler — a mixin class for Hart that implements:
- _take_trap (trap delivery with medeleg/mideleg delegation)
- _trap_deliver_smode / _trap_deliver_mmode
- _trap_ecall / _trap_ebreak / _trap_mret / _trap_sret
- _handle_wfi (wait-for-interrupt with TW check)
- check_pending_interrupts (instruction-boundary interrupt polling)
"""

from pyremu.core.hart import (
    MSTATUS_MIE,
    MSTATUS_MPIE,
    MSTATUS_MPP,
    MSTATUS_SIE,
    MSTATUS_SPIE,
    MSTATUS_SPP,
    MSTATUS_TW,
    RiscvMode,
)
from pyremu.core.trap import TrapType, trap_cause_code


class TrapHandler:
    """Mixin: trap delivery, delegation, interrupt handling.

    Requires the host class to provide:
    - self.pc, self.mode, self.mstatus_val, self.mie, self.sie
    - self.csrs dict (mtvec, stvec, mepc, sepc, mcause, scause, mtval, stval,
                       medeleg, mideleg, mip, mie)
    - self.mip_val, self.mie_val
    - self.clear_reservation()
    - self._waiting (bool, set by WFI, cleared by any trap)
    - self._interrupt_ctrl (InterruptController or None)
    - self._consecutive_traps (int)
    """

    # ----------------------------------------------------------
    #  Trap 处理
    # ----------------------------------------------------------

    def _take_trap(
        self,
        cause: TrapType,
        tval: int = 0,
        is_interrupt: bool = False,
    ) -> None:
        """统一的 trap 入口 — 支持委派到 S 模式.

        根据 medeleg (异常委派) / mideleg (中断委派) 和当前特权级
        决定 trap 目标:
        - M 模式: 保存到 mepc/mcause/mtval, 跳转 mtvec
        - S 模式: 保存到 sepc/scause/stval, 跳转 stvec

        委派规则 (RISC-V Privileged Spec §3.1.9):
        - M 模式下发生的 trap 永不委派
        - 委派仅在 medeleg/mideleg 对应位为 1 时生效
        - 无需额外检查"不委派到更高特权级", S 是唯一委派目标且 < M

        Args:
            cause: trap 类型 (TrapType 枚举).
            tval: 关联的故障地址或附加信息.
            is_interrupt: True 表示中断, False 表示异常.
        """
        # trap 发生时清除 LR/SC 预留
        self.clear_reservation()

        # 任何 trap 都会唤醒 WFI 等待中的 hart
        self._waiting = False

        # 连续 trap 计数 (正常执行指令时由 Emulator.step 清零)
        self._consecutive_traps += 1

        # 获取 cause 编码 (中断标志已嵌入)
        code = trap_cause_code(cause)
        exc_code = code & 0x7FFF_FFFF_FFFF_FFFF  # 去掉 bit 63

        # ---- 委派检查 ----
        # M 模式下永不委派; S/U 模式下根据 medeleg/mideleg 判断
        delegate = False
        if self.mode != RiscvMode.M:
            if is_interrupt:
                delegate = bool(self.csrs["mideleg"].val & (1 << exc_code))
            else:
                delegate = bool(self.csrs["medeleg"].val & (1 << exc_code))

        if delegate:
            self._trap_deliver_smode(code, exc_code, tval, is_interrupt)
        else:
            self._trap_deliver_mmode(code, exc_code, tval, is_interrupt)

    # ---- S 模式 trap 投递 ----

    def _trap_deliver_smode(
        self,
        code: int,
        exc_code: int,
        tval: int,
        is_interrupt: bool,
    ) -> None:
        """将 trap 投递到 S 模式: 保存上下文到 S 模式 CSR 并跳转 stvec."""
        # 保存当前 PC 到 sepc
        self.sepc_val = self.pc

        # 设置 scause (中断标志位不变)
        self.scause_val = code

        # 设置 stval
        self.stval_val = tval

        # 更新 mstatus:
        #   SPIE ← SIE (保存进入 trap 前的中断使能)
        #   SIE  ← 0   (进入 trap 后关全局中断)
        #   SPP  ← 当前特权级
        mstatus = self.mstatus_val
        if mstatus & MSTATUS_SIE:
            mstatus |= MSTATUS_SPIE
        else:
            mstatus &= ~MSTATUS_SPIE
        mstatus &= ~MSTATUS_SIE

        # SPP ← 当前 mode (U=0, S=1)
        _spp_map = {RiscvMode.U: 0, RiscvMode.S: 1}
        spp_code = _spp_map.get(self.mode, 0)
        mstatus = (mstatus & ~MSTATUS_SPP) | (spp_code << 8)
        self.mstatus_val = mstatus

        # 切换到 S 模式
        self.mode = RiscvMode.S

        # 跳转到 stvec
        stvec = self.csrs["stvec"].val
        tvec_mode = stvec & 0x3
        tvec_base = stvec & ~0x3
        if tvec_mode == 0:
            self.pc = tvec_base
        else:
            # vectored: 所有异常跳转 BASE, 中断跳转 BASE + 4 * exc_code
            self.pc = (
                tvec_base if not is_interrupt
                else (tvec_base + 4 * exc_code)
            )

    # ---- M 模式 trap 投递 ----

    def _trap_deliver_mmode(
        self,
        code: int,
        exc_code: int,
        tval: int,
        is_interrupt: bool,
    ) -> None:
        """将 trap 投递到 M 模式: 保存上下文到 M 模式 CSR 并跳转 mtvec."""
        # 保存当前 PC
        self.mepc_val = self.pc

        # 设置 mcause
        self.mcause_val = code

        # 设置 mtval
        self.mtval_val = tval

        # 更新 mstatus:
        #   MPIE ← MIE
        #   MIE  ← 0
        #   MPP  ← 当前特权级
        mstatus = self.mstatus_val
        if mstatus & MSTATUS_MIE:
            mstatus |= MSTATUS_MPIE
        else:
            mstatus &= ~MSTATUS_MPIE
        mstatus &= ~MSTATUS_MIE

        # MPP ← 当前 mode
        _mpp_map = {RiscvMode.U: 0, RiscvMode.S: 1, RiscvMode.M: 3}
        mpp_code = _mpp_map.get(self.mode, 0)
        mstatus = (mstatus & ~MSTATUS_MPP) | (mpp_code << 11)
        self.mstatus_val = mstatus

        # 切换到 M 模式
        self.mode = RiscvMode.M

        # 跳转到 mtvec
        mtvec = self.csrs["mtvec"].val
        tvec_mode = mtvec & 0x3
        tvec_base = mtvec & ~0x3
        if tvec_mode == 0:
            self.pc = tvec_base
        else:
            self.pc = (
                tvec_base if not is_interrupt
                else (tvec_base + 4 * exc_code)
            )

    # ----------------------------------------------------------
    #  ECALL / EBREAK / MRET / SRET
    # ----------------------------------------------------------

    def _trap_ecall(
        self,
    ) -> None:
        """ECALL: 根据当前特权级触发相应的环境调用 trap."""
        _ecall_map: dict[RiscvMode, TrapType] = {
            RiscvMode.U: TrapType.EcallFromUmode,
            RiscvMode.S: TrapType.EcallFromSmode,
            RiscvMode.M: TrapType.EcallFromMmode,
        }
        cause = _ecall_map.get(self.mode)
        if cause is None:
            cause = TrapType.IllInstr
        self._take_trap(cause, tval=self.pc, is_interrupt=False)

    def _trap_ebreak(
        self,
    ) -> None:
        """EBREAK: 断点异常."""
        self._take_trap(TrapType.Breakpoint, tval=self.pc, is_interrupt=False)

    def _trap_mret(
        self,
    ) -> None:
        """MRET: 从 M 模式 trap 返回.

        恢复进入 M 模式 trap 前保存的特权级和中断使能状态.
        """
        mstatus = self.mstatus_val

        # 恢复特权级: mode ← MPP
        mpp = (mstatus & MSTATUS_MPP) >> 11
        _mpp_to_mode = {0: RiscvMode.U, 1: RiscvMode.S, 3: RiscvMode.M}
        self.mode = _mpp_to_mode.get(mpp, RiscvMode.U)

        # 恢复中断使能: MIE ← MPIE, 然后 MPIE ← 1
        if mstatus & MSTATUS_MPIE:
            mstatus |= MSTATUS_MIE
        else:
            mstatus &= ~MSTATUS_MIE
        mstatus |= MSTATUS_MPIE

        # MPP ← U (最低特权)
        mstatus &= ~MSTATUS_MPP
        self.mstatus_val = mstatus

        # PC ← mepc
        self.pc = self.mepc_val & 0xFFFF_FFFF_FFFF_FFFF

    def _trap_sret(
        self,
    ) -> None:
        """SRET: 从 S 模式 trap 返回.

        恢复进入 S 模式 trap 前保存的特权级和中断使能状态.
        """
        mstatus = self.mstatus_val

        # 恢复特权级: mode ← SPP
        spp = (mstatus & MSTATUS_SPP) >> 8
        self.mode = RiscvMode.U if spp == 0 else RiscvMode.S

        # 恢复中断使能: SIE ← SPIE, 然后 SPIE ← 1
        if mstatus & MSTATUS_SPIE:
            mstatus |= MSTATUS_SIE
        else:
            mstatus &= ~MSTATUS_SIE
        mstatus |= MSTATUS_SPIE

        # SPP ← U
        mstatus &= ~MSTATUS_SPP
        self.mstatus_val = mstatus

        # PC ← sepc
        self.pc = self.sepc_val & 0xFFFF_FFFF_FFFF_FFFF

    # ----------------------------------------------------------
    #  WFI (Wait For Interrupt) — 低功耗等待
    # ----------------------------------------------------------

    def _handle_wfi(
        self,
        instr: int,
    ) -> None:
        """WFI 指令: 若已有待处理的使能中断则立即返回 (NOP);
        否则 hart 进入等待状态, 由中断唤醒.

        RISC-V Privileged Spec §3.3.5:
        - WFI 是一个 hint; 实现可将其视为 NOP
        - 若中断已挂起且使能, WFI 不等待, 继续执行
        - 等待状态下中断挂起且使能时 hart 被唤醒
        - 唤醒后 PC 指向 WFI 下一条指令; 若中断可被响应则走正常中断处理
        - mstatus.TW=1 且非 M 模式时执行 WFI → IllInstr

        注意: 等待期间不检查 mstatus.MIE, 仅 mip & mie 非零即可唤醒.
        """
        # TW (Timeout Wait) 检查: 非 M 模式下 mstatus.TW=1 → 非法指令异常
        if self.mode != RiscvMode.M:
            if self.mstatus_val & MSTATUS_TW:
                self._take_trap(TrapType.IllInstr, tval=instr, is_interrupt=False)
                return

        # 检查是否已有待处理且使能的中断
        # 若 mip & mie 非零, 立即返回 (NOP, PC 将 +4)
        if self.mip_val & self.mie_val:
            return  # 正常返回, 调用方会将 PC+4

        # 无可处理中断 → 进入等待状态
        self._waiting = True

    # ----------------------------------------------------------
    #  中断检查 (指令边界)
    # ----------------------------------------------------------

    def check_pending_interrupts(self) -> bool:
        """在指令边界检查是否有待处理且使能的中断.

        若有, 则通过 _take_trap 注入中断并返回 True;
        否则返回 False.

        优先级: MEI > MSI > MTI > SEI > SSI > STI
        (RISC-V Privileged Spec §3.1.9)

        支持通过 mideleg 将中断委派到 S 模式处理:
        - M 模式 + MIE=0: 全局关中断, 不响应任何中断
        - S 模式 + SIE=0: 仅非委派 (M 级) 中断可抢占; 已委派中断被阻塞
        - U 模式: 全部中断全局使能

        NOTE: 此方法在 Hart 而非 HartWithRegs 中定义,
        因为它依赖子类覆写的 _take_trap 方法.
        """
        if self._interrupt_ctrl is None:
            return False

        has_pending, mip_bits, _ = self._interrupt_ctrl.check_interrupt(self.id)
        if not has_pending:
            return False

        # 更新 mip CSR: 合并硬件中断源和软件写入的 mip 位
        current_mip = self.mip_val
        self.mip_val = current_mip | mip_bits

        # M 模式 + MIE=0 → 全局关中断
        if self.mode == RiscvMode.M and not self.mie:
            return False

        mie = self.mie_val
        masked = self.mip_val & mie
        if masked == 0:
            return False

        # S 模式全局中断使能 (用于已委派的中断)
        mideleg = self.csrs["mideleg"].val
        s_mode_global = (
            self.mode.value < RiscvMode.S.value
            or (self.mode == RiscvMode.S and self.sie)
        )

        # 按优先级找最高优先级的使能中断
        # 完整优先级: MEI(11) > MSI(3) > MTI(7) > SEI(9) > SSI(1) > STI(5)
        int_priority = [
            (1 << 11, TrapType.MmodeExternInterrupt),  # MEI
            (1 << 3, TrapType.MmodeSoftInterrupt),  # MSI
            (1 << 7, TrapType.MmodeTimerInterrupt),  # MTI
            (1 << 9, TrapType.SmodeExternInterrupt),  # SEI
            (1 << 1, TrapType.SmodeSoftInterrupt),  # SSI
            (1 << 5, TrapType.SmodeTimerInterrupt),  # STI
        ]
        for mask, trap_type in int_priority:
            if not (masked & mask):
                continue
            # 已委派到 S 的中断: 需检查 S 模式全局使能
            if (mideleg & mask) and not s_mode_global:
                continue
            self._take_trap(trap_type, tval=0, is_interrupt=True)
            return True

        return False
