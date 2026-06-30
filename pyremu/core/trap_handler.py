#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""Trap 处理函数: trap delivery, delegation, interrupt checking, WFI.

原 TrapHandler mixin 已拆分为本模块中的独立函数,
所有函数将 hart 作为显式第一参数, 消除 mixin 的隐式依赖和静态分析告警.

Public functions:
    deliver_trap              — trap delivery with medeleg/mideleg delegation
    trap_ecall                — ECALL instruction trap
    trap_ebreak               — EBREAK instruction trap
    trap_mret                 — MRET return from M-mode trap
    trap_sret                 — SRET return from S-mode trap
    handle_wfi                — WFI wait-for-interrupt with TW check
    check_pending_interrupts  — instruction-boundary interrupt polling
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

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

if TYPE_CHECKING:
    from pyremu.core.hart import HartWithRegs

_TRACE_TRAPS = os.environ.get("PYREMU_TRACE_TRAPS", "") == "1"

# 中断优先级列表 (按优先级从高到低排列).
# 预计算为模块级常量, 避免 check_pending_interrupts
# 每条指令分配一次 (节省 1M list + 6M tuple/s).
_INT_PRIORITY: list[tuple[int, TrapType]] = [
    (1 << 11, TrapType.MmodeExternInterrupt),  # MEI
    (1 << 3, TrapType.MmodeSoftInterrupt),  # MSI
    (1 << 7, TrapType.MmodeTimerInterrupt),  # MTI
    (1 << 9, TrapType.SmodeExternInterrupt),  # SEI
    (1 << 1, TrapType.SmodeSoftInterrupt),  # SSI
    (1 << 5, TrapType.SmodeTimerInterrupt),  # STI
]

# RiscvMode 枚举值预计算, 避免每条指令访问 .value property
_MODE_M = RiscvMode.M.value
_MODE_S = RiscvMode.S.value
_MODE_U = RiscvMode.U.value

# ============================================================
#  Trap 处理
# ============================================================


def deliver_trap(
    hart: HartWithRegs,
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
        hart: 目标硬件线程.
        cause: trap 类型 (TrapType 枚举).
        tval: 关联的故障地址或附加信息.
        is_interrupt: True 表示中断, False 表示异常.
    """
    # trap 发生时清除 LR/SC 预留
    hart.clear_reservation()

    # 记录是否从 WFI 唤醒 (用于排除中断 handler 的指令计数)
    was_wfi = hart._waiting

    # 任何 trap 都会唤醒 WFI 等待中的 hart
    hart._waiting = False
    if was_wfi:
        hart._wfi_woken = True

    # 连续 trap 计数 (正常执行指令时由 Emulator.step 清零)
    hart._consecutive_traps += 1

    # 获取 cause 编码 (中断标志已嵌入)
    code = trap_cause_code(cause)
    exc_code = code & 0x7FFF_FFFF_FFFF_FFFF  # 去掉 bit 63

    # ---- 委派检查 ----
    # M 模式下永不委派; S/U 模式下根据 medeleg/mideleg 判断
    delegate = False
    if hart.mode != RiscvMode.M:
        if is_interrupt:
            delegate = bool(hart.csrs["mideleg"].val & (1 << exc_code))
        else:
            delegate = bool(hart.csrs["medeleg"].val & (1 << exc_code))

    if _TRACE_TRAPS:
        _target = "S" if delegate else "M"
        _ctx = (
            f"[trap:{hart.id}] {hart.mode.name}→{_target}: {cause.name} "
            f"tval={tval:#018x} pc={hart.pc:#018x} "
            f"exc_code={exc_code} mstatus={hart.mstatus_val:#018x}"
        )
        logger.debug(_ctx)

    if delegate:
        _trap_deliver_smode(hart, code, exc_code, tval, is_interrupt)
    else:
        _trap_deliver_mmode(hart, code, exc_code, tval, is_interrupt)


# ---- S 模式 trap 投递 ----


def _trap_deliver_smode(
    hart: HartWithRegs,
    code: int,
    exc_code: int,
    tval: int,
    is_interrupt: bool,
) -> None:
    """将 trap 投递到 S 模式: 保存上下文到 S 模式 CSR 并跳转 stvec."""
    # 保存当前 PC 到 sepc
    hart.sepc_val = hart.pc

    # 设置 scause (中断标志位不变)
    hart.scause_val = code

    # 设置 stval
    hart.stval_val = tval

    # 更新 mstatus:
    #   SPIE ← SIE (保存进入 trap 前的中断使能)
    #   SIE  ← 0   (进入 trap 后关全局中断)
    #   SPP  ← 当前特权级
    mstatus = hart.mstatus_val
    if mstatus & MSTATUS_SIE:
        mstatus |= MSTATUS_SPIE
    else:
        mstatus &= ~MSTATUS_SPIE
    mstatus &= ~MSTATUS_SIE

    # SPP ← 当前 mode (U=0, S=1)
    _spp_map = {RiscvMode.U: 0, RiscvMode.S: 1}
    spp_code = _spp_map.get(hart.mode, 0)
    mstatus = (mstatus & ~MSTATUS_SPP) | (spp_code << 8)
    hart.mstatus_val = mstatus

    # 切换到 S 模式
    hart.mode = RiscvMode.S

    # 跳转到 stvec
    stvec = hart.csrs["stvec"].val
    tvec_mode = stvec & 0x3
    tvec_base = stvec & ~0x3
    if tvec_mode == 0:
        hart.pc = tvec_base
    else:
        # vectored: 所有异常跳转 BASE, 中断跳转 BASE + 4 * exc_code
        hart.pc = tvec_base if not is_interrupt else (tvec_base + 4 * exc_code)


# ---- M 模式 trap 投递 ----


def _trap_deliver_mmode(
    hart: HartWithRegs,
    code: int,
    exc_code: int,
    tval: int,
    is_interrupt: bool,
) -> None:
    """将 trap 投递到 M 模式: 保存上下文到 M 模式 CSR 并跳转 mtvec."""
    # 保存当前 PC
    hart.mepc_val = hart.pc

    # 设置 mcause
    hart.mcause_val = code

    # 设置 mtval
    hart.mtval_val = tval

    # 更新 mstatus:
    #   MPIE ← MIE
    #   MIE  ← 0
    #   MPP  ← 当前特权级
    mstatus = hart.mstatus_val
    if mstatus & MSTATUS_MIE:
        mstatus |= MSTATUS_MPIE
    else:
        mstatus &= ~MSTATUS_MPIE
    mstatus &= ~MSTATUS_MIE

    # MPP ← 当前 mode
    _mpp_map = {RiscvMode.U: 0, RiscvMode.S: 1, RiscvMode.M: 3}
    mpp_code = _mpp_map.get(hart.mode, 0)
    mstatus = (mstatus & ~MSTATUS_MPP) | (mpp_code << 11)
    hart.mstatus_val = mstatus

    # 切换到 M 模式
    hart.mode = RiscvMode.M

    # 跳转到 mtvec
    mtvec = hart.csrs["mtvec"].val
    tvec_mode = mtvec & 0x3
    tvec_base = mtvec & ~0x3
    if tvec_mode == 0:
        hart.pc = tvec_base
    else:
        hart.pc = tvec_base if not is_interrupt else (tvec_base + 4 * exc_code)


# ============================================================
#  ECALL / EBREAK / MRET / SRET
# ============================================================


def trap_ecall(
    hart: HartWithRegs,
) -> None:
    """ECALL: 根据当前特权级触发相应的环境调用 trap."""
    _ecall_map: dict[RiscvMode, TrapType] = {
        RiscvMode.U: TrapType.EcallFromUmode,
        RiscvMode.S: TrapType.EcallFromSmode,
        RiscvMode.M: TrapType.EcallFromMmode,
    }
    cause = _ecall_map.get(hart.mode)
    if cause is None:
        cause = TrapType.IllInstr
    deliver_trap(hart, cause, tval=hart.pc, is_interrupt=False)


def trap_ebreak(
    hart: HartWithRegs,
) -> None:
    """EBREAK: 断点异常."""
    deliver_trap(hart, TrapType.Breakpoint, tval=hart.pc, is_interrupt=False)


def trap_mret(
    hart: HartWithRegs,
) -> None:
    """MRET: 从 M 模式 trap 返回.

    恢复进入 M 模式 trap 前保存的特权级和中断使能状态.
    仅在 M 模式下合法; 否则触发 IllInstr.
    """
    if hart.mode != RiscvMode.M:
        deliver_trap(hart, TrapType.IllInstr, tval=0x30200073, is_interrupt=False)
        return
    mstatus = hart.mstatus_val

    # 恢复特权级: mode ← MPP
    mpp = (mstatus & MSTATUS_MPP) >> 11
    _mpp_to_mode = {0: RiscvMode.U, 1: RiscvMode.S, 3: RiscvMode.M}
    hart.mode = _mpp_to_mode.get(mpp, RiscvMode.U)

    # 恢复中断使能: MIE ← MPIE, 然后 MPIE ← 1
    if mstatus & MSTATUS_MPIE:
        mstatus |= MSTATUS_MIE
    else:
        mstatus &= ~MSTATUS_MIE
    mstatus |= MSTATUS_MPIE

    # MPP ← U (最低特权)
    mstatus &= ~MSTATUS_MPP
    hart.mstatus_val = mstatus

    # PC ← mepc
    hart.pc = hart.mepc_val & 0xFFFF_FFFF_FFFF_FFFF

    # 从 trap 返回 → 清除 WFI 唤醒标记, 后续指令正常计数
    hart._wfi_woken = False


def trap_sret(
    hart: HartWithRegs,
) -> None:
    """SRET: 从 S 模式 trap 返回.

    恢复进入 S 模式 trap 前保存的特权级和中断使能状态.
    在 S/M 模式下合法; U 模式下触发 IllInstr.
    """
    if hart.mode == RiscvMode.U:
        deliver_trap(hart, TrapType.IllInstr, tval=0x10200073, is_interrupt=False)
        return
    mstatus = hart.mstatus_val

    # 恢复特权级: mode ← SPP
    spp = (mstatus & MSTATUS_SPP) >> 8
    hart.mode = RiscvMode.U if spp == 0 else RiscvMode.S

    # 恢复中断使能: SIE ← SPIE, 然后 SPIE ← 1
    if mstatus & MSTATUS_SPIE:
        mstatus |= MSTATUS_SIE
    else:
        mstatus &= ~MSTATUS_SIE
    mstatus |= MSTATUS_SPIE

    # SPP ← U
    mstatus &= ~MSTATUS_SPP
    hart.mstatus_val = mstatus

    # PC ← sepc
    hart.pc = hart.sepc_val & 0xFFFF_FFFF_FFFF_FFFF

    # 从 trap 返回 → 清除 WFI 唤醒标记, 后续指令正常计数
    hart._wfi_woken = False


# ============================================================
#  WFI (Wait For Interrupt) — 低功耗等待
# ============================================================


def handle_wfi(
    hart: HartWithRegs,
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
    if hart.mode != RiscvMode.M:
        if hart.mstatus_val & MSTATUS_TW:
            deliver_trap(hart, TrapType.IllInstr, tval=instr, is_interrupt=False)
            return

    # 检查是否已有待处理且使能的中断
    # 若 mip & mie 非零, 立即返回 (NOP, PC 将 +4)
    if hart.mip_val & hart.mie_val:
        return  # 正常返回, 调用方会将 PC+4

    # 无可处理中断 → 进入等待状态, 重置唤醒标记
    hart._waiting = True
    hart._wfi_woken = False


# ============================================================
#  中断检查 (指令边界)
# ============================================================


def check_pending_interrupts(
    hart: HartWithRegs,
) -> bool:
    """在指令边界检查是否有待处理且使能的中断.

    若有, 则通过 _take_trap 注入中断并返回 True;
    否则返回 False.

    优先级: MEI > MSI > MTI > SEI > SSI > STI
    (RISC-V Privileged Spec §3.1.9)

    支持通过 mideleg 将中断委派到 S 模式处理:
    - M 模式 + MIE=0: 全局关中断, 不响应任何中断
    - S 模式 + SIE=0: 仅非委派 (M 级) 中断可抢占; 已委派中断被阻塞
    - U 模式: 全部中断全局使能
    """
    if hart._interrupt_ctrl is None:
        return False

    has_pending, mip_bits, _ = hart._interrupt_ctrl.check_interrupt(hart.id)
    if not has_pending:
        return False

    # 更新 mip CSR: 合并硬件中断源和软件写入的 mip 位
    current_mip = hart.mip_val
    hart.mip_val = current_mip | mip_bits

    # M 模式 + MIE=0 → 全局关中断
    if hart.mode == RiscvMode.M and not hart.mie:
        return False

    mie = hart.mie_val
    masked = hart.mip_val & mie
    if masked == 0:
        return False

    # S 模式全局中断使能 (用于已委派的中断).
    # 用预计算 mode int 替代 .value property 访问.
    mode_int = hart.mode.value
    mideleg = hart.csrs["mideleg"].val
    s_mode_global = mode_int < _MODE_S or (
        hart.mode == RiscvMode.S and hart.sie
    )

    # 按优先级找最高优先级的使能中断.
    # 使用模块级预计算常量, 避免每条指令分配 list + 6 tuple.
    for mask, trap_type in _INT_PRIORITY:
        if not (masked & mask):
            continue
        # 已委派到 S 的中断: 需检查 S 模式全局使能
        if (mideleg & mask) and not s_mode_global:
            continue
        deliver_trap(hart, trap_type, tval=0, is_interrupt=True)
        return True

    return False


# ============================================================
#  Nested-interrupt-enabled 变体
# ============================================================


def deliver_trap_nested_enabled(
    hart: HartWithRegs,
    cause: TrapType,
    tval: int = 0,
    is_interrupt: bool = False,
) -> None:
    """deliver_trap 的嵌套中断变体.

    与 deliver_trap 基本相同, 但对中断 trap 不递增 _consecutive_traps,
    因为嵌套中断是预期中的正常行为, 不应触发连续 trap 保护机制.
    """
    hart.clear_reservation()
    was_wfi = hart._waiting
    hart._waiting = False
    if was_wfi:
        hart._wfi_woken = True

    # 嵌套中断不递增连续 trap 计数 (允许正常的多层嵌套)
    if not is_interrupt:
        hart._consecutive_traps += 1

    code = trap_cause_code(cause)
    exc_code = code & 0x7FFF_FFFF_FFFF_FFFF

    delegate = False
    if hart.mode != RiscvMode.M:
        if is_interrupt:
            delegate = bool(hart.csrs["mideleg"].val & (1 << exc_code))
        else:
            delegate = bool(hart.csrs["medeleg"].val & (1 << exc_code))

    if delegate:
        _trap_deliver_smode(hart, code, exc_code, tval, is_interrupt)
    else:
        _trap_deliver_mmode(hart, code, exc_code, tval, is_interrupt)


def check_pending_interrupts_nested_enabled(
    hart: HartWithRegs,
) -> bool:
    """check_pending_interrupts 的嵌套中断变体.

    使用 deliver_trap_nested_enabled 而非 deliver_trap,
    使得嵌套中断不会触发连续 trap 计数.
    由开启嵌套中断的 trap handler (如汇编中的 s_trap_handler_nested_enabled)
    通过重新使能 SIE 后交由指令边界的本函数处理.
    """
    if hart._interrupt_ctrl is None:
        return False

    has_pending, mip_bits, _ = hart._interrupt_ctrl.check_interrupt(hart.id)
    if not has_pending:
        return False

    current_mip = hart.mip_val
    hart.mip_val = current_mip | mip_bits

    if hart.mode == RiscvMode.M and not hart.mie:
        return False

    mie = hart.mie_val
    masked = hart.mip_val & mie
    if masked == 0:
        return False

    mideleg = hart.csrs["mideleg"].val
    s_mode_global = hart.mode.value < RiscvMode.S.value or (
        hart.mode == RiscvMode.S and hart.sie
    )

    int_priority = [
        (1 << 11, TrapType.MmodeExternInterrupt),
        (1 << 3, TrapType.MmodeSoftInterrupt),
        (1 << 7, TrapType.MmodeTimerInterrupt),
        (1 << 9, TrapType.SmodeExternInterrupt),
        (1 << 1, TrapType.SmodeSoftInterrupt),
        (1 << 5, TrapType.SmodeTimerInterrupt),
    ]
    for mask, trap_type in int_priority:
        if not (masked & mask):
            continue
        if (mideleg & mask) and not s_mode_global:
            continue
        deliver_trap_nested_enabled(
            hart,
            trap_type,
            tval=0,
            is_interrupt=True,
        )
        return True

    return False
