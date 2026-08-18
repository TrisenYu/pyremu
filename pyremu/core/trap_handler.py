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

from typing import TYPE_CHECKING

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
from pyremu.core.trap_def import trap_cause_code, TrapType
from pyremu.interrupt.controller import INT_SOURCE_MIP_MASK, IntSource
from pyremu.utils.mask import mask64

if TYPE_CHECKING:
    from pyremu.core.hart import HartWithRegs

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

# 硬件源管理的 mip 位掩码 (只读位, 反映外部中断信号).
# 这些位在 check_pending_interrupts 中必须用硬件当前状态**替换** (非 OR 累加),
# 否则一旦硬件源 (如 CLINT MSIP) 撤除后旧位仍残留在 mip CSR 中,
# 导致无限中断重投递 (MSIP 风暴).
_HW_MIP_MASK: int = (
    INT_SOURCE_MIP_MASK[IntSource.MSI]  # bit 3 — CLINT _msip[hart_id]
    | INT_SOURCE_MIP_MASK[IntSource.MTI]  # bit 7 — CLINT _mtimecmp
    | INT_SOURCE_MIP_MASK[IntSource.STI]  # bit 5 — Sstc stimecmp
    | INT_SOURCE_MIP_MASK[IntSource.MEI]  # bit 11 — PLIC
    | INT_SOURCE_MIP_MASK[IntSource.SEI]  # bit 9 — PLIC
)

# RiscvMode 枚举值预计算, 避免每条指令访问 .value property
_MODE_M = RiscvMode.M.value
_MODE_S = RiscvMode.S.value
_MODE_U = RiscvMode.U.value


def _update_hw_mip(hart: HartWithRegs, hw_mip_bits: int) -> None:
    """用当前硬件状态替换 mip CSR 中的硬件源位.

    与简单的 `current_mip | hw_mip_bits` 不同, 此函数**清除**已不再被
    硬件源断言的位, 防止中断源撤除后无限重投递 (如 CLINT MSIP 清零后
    mip.MSIP 仍残留导致 MmodeSoftInterrupt 风暴).
    """
    current = hart._csr_read_raw("mip")
    hart._csr_write_raw("mip", (current & ~_HW_MIP_MASK) | hw_mip_bits)



# ============================================================
#  IMSIC eip trap-entry cleanup (legacy mode)
# ============================================================


def _imsic_clear_ipi_on_trap(hart: HartWithRegs, exc_code: int) -> None:
    """Clear mip + IMSIC eip for software interrupt trap entry.

    ``mip`` bit = ``1 << exc_code`` (MSIP=3→mip[3], SSIP=1→mip[1]).
    IMSIC: MSIP→M-file IID=3, SSIP→S-file IID=1.
    In AIA mode (eidelivery==1) eip is claimed via MTOPEI/STOPEI, not here.
    """
    hart.mip_val &= ~(1 << exc_code)
    imsic = hart._imsic
    if imsic is None:
        return
    if exc_code == 3:
        file = imsic._files[hart.id][0]  # M-file
    elif exc_code == 1:
        file = imsic._files[hart.id][1]  # S-file
    else:
        return
    if file.eidelivery == 0:
        file.clear_pending(exc_code)

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

    # 获取 cause 编码 (中断标志已嵌入)
    code = trap_cause_code(cause)
    exc_code = code & 0x7FFF_FFFF_FFFF_FFFF  # 去掉 bit 63

    # ---- 委派检查 ----
    # M 模式下永不委派; S/U 模式下根据 medeleg/mideleg 判断
    delegate = False
    if hart.mode != RiscvMode.M:
        choice = hart.csrs["mideleg"].val if is_interrupt else hart.csrs["medeleg"].val
        delegate = bool(choice & (1 << exc_code))

    fn = _trap_deliver_smode if delegate else _trap_deliver_mmode
    fn(hart, code, exc_code, tval, is_interrupt)


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
    if tvec_mode == 0 or not is_interrupt:
        hart.pc = tvec_base
    else:
        # vectored: 所有异常跳转 BASE, 中断跳转 BASE + 4 * exc_code
        hart.pc = (tvec_base + 4 * exc_code)

    # MSIP/SSIP 中断投递到 S 模式: 同步清零 CLINT MSIP + IMSIC eip.
    # 在 AIA 模式下 IPI 通过 MEIP/SEIP (cause 11/9) 投递，eip 由 MTOPEI/
    # STOPEI claim 清除，trap entry 不清理。
    if not is_interrupt:
        return
    if exc_code == 3:  # MSIP (delegated to S)
        ctrl = hart._interrupt_ctrl
        if ctrl is not None:
            ctrl.clear_ipi(hart.id)
        _imsic_clear_ipi_on_trap(hart, exc_code)
    elif exc_code == 1:  # SSIP
        _imsic_clear_ipi_on_trap(hart, exc_code)


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

    # RISC-V spec: MPRV is cleared on trap entry to M-mode so the
    # handler can safely access its own stack/data without going
    # through the MMU translation of the previous privilege mode.
    mstatus &= ~(1 << 17)  # clear MPRV

    hart.mstatus_val = mstatus

    # 切换到 M 模式
    hart.mode = RiscvMode.M

    # 跳转到 mtvec
    mtvec = hart.csrs["mtvec"].val
    tvec_mode = mtvec & 0x3
    tvec_base = mtvec & ~0x3
    if tvec_mode == 0 or not is_interrupt:
        hart.pc = tvec_base
    else:
        hart.pc = (tvec_base + 4 * exc_code)

    # MSIP 中断: 同步清零 CLINT MSIP + IMSIC eip (legacy mode).
    if is_interrupt and exc_code == 3:  # MSIP (mcause code 3)
        ctrl = hart._interrupt_ctrl
        if ctrl is not None:
            ctrl.clear_ipi(hart.id)
        _imsic_clear_ipi_on_trap(hart, exc_code)


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

    # MPP设置为U模式
    mstatus &= ~MSTATUS_MPP
    hart.mstatus_val = mstatus

    hart.pc = mask64(hart.mepc_val)

    # 不在此处清除 _wfi_woken.
    # _wfi_woken 由 deliver_trap 在从 WFI 唤醒时置位, 意在让紧随其后的
    # handle_wfi 将 WFI 视为 NOP 并推进 PC, 从而允许 while (...) wfi()
    # 轮询循环在 trap handler 返回后重新检查状态条件.
    # 若在此处提前清除, handle_wfi 将看不到该标记, hart 立即重回睡眠.


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
    hart.pc = mask64(hart.sepc_val)

    # 不在此处清除 _wfi_woken (同 trap_mret 的注释说明).


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
    - mstatus.TW=1 且非 M 模式时执行 WFI -> IllInstr

    注意: 等待期间不检查 mstatus.MIE, 仅 mip & mie 非零即可唤醒.
    """
    # TW (Timeout Wait) 检查: 非 M 模式下 mstatus.TW=1 -> 非法指令异常
    if hart.mode != RiscvMode.M and hart.mstatus_val & MSTATUS_TW:
        deliver_trap(hart, TrapType.IllInstr, tval=instr, is_interrupt=False)
        return

    # 刚被中断唤醒 (MRET 回到 WFI): 视为 NOP, 推进 PC 以允许
    # while (state != READY) wfi() 轮询循环在 trap handler 返回后
    # 重新检查状态条件. 若仍不满足, 下次 WFI 会正常进入等待.
    if hart._wfi_woken:
        hart._wfi_woken = False
        return

    # 检查是否已有待处理且使能的中断
    # 若 mip & mie 非零, 立即返回 (NOP, PC 将 +4)
    if hart.mip_val & hart.mie_val:
        return  # 正常返回, 调用方会将 PC+4

    # 无可处理中断 -> 进入等待状态, 重置唤醒标记
    hart._waiting = True
    hart._wfi_woken = False


# ============================================================
#  中断检查 (指令边界)
# ============================================================


def _compute_next_timer(
    hart: HartWithRegs,
    ctrl,  # InterruptController
) -> int:
    """计算最早定时器唤醒时间 (mtime 值); 0 = 无定时器使能.

    供中断缓存快速路径: 若 mtime < wakeup, 可跳过全量中断检查.
    """
    mie = hart._csr_read_raw("mie")
    wakeup = 0
    # MTIE (bit 7): CLINT mtimecmp
    if mie & (1 << 7):
        clint_wake = ctrl.get_next_timer_wakeup(hart.id)
        if clint_wake:
            wakeup = clint_wake
    # STIE (bit 5): Sstc stimecmp
    if mie & (1 << 5):
        s_cmp = hart._csr_read_raw("stimecmp")
        if s_cmp and (wakeup == 0 or s_cmp < wakeup):
            wakeup = s_cmp
    return wakeup


def try_wfi_wakeup(
    hart: HartWithRegs,
) -> bool:
    """WFI 唤醒检查 — 仅检查 mip & mie (中断源级使能), 不检查 mstatus.MIE.

    WFI 的 NOP 条件 (RISC-V spec §3.6.1) 只要求 ''任一中断挂起且该中断源
    在 mie CSR 中使能'', 不要求全局中断使能 (mstatus.MIE). 因此 WFI 唤醒
    的检查条件也应只使用 mip & mie, 不引入 mstatus.MIE 这一额外门槛.

    特殊处理 CLINT MSIP: 真实硬件上 CLINT 将 MSIP 位断言为独立物理中断线,
    WFI 由此唤醒不依赖 mie.MSIE (只要求 CLINT MSIP 寄存器非零).
    mie.MSIE 仅控制该中断是否被 *投递*.  这与加速所用的动态链接库中定义的函数
    ``try_wfi_wakeup`` 的行为一致 — 多核 TLB shootdown 场景中若
    mie.MSIE 因固件代码路径被意外清零, 跳过该条件可避免发送核在
    ``tlb_sync`` 中永远自旋的死锁.

    若从 WFI 中被唤醒, 置位 _wfi_woken 标志以供 WFI handler 消费 (while
    循环的 WFI NOP 路径). 返回 True 表示唤醒成功.
    """
    if not hart._waiting or hart._halted:
        return False

    ctrl = hart._interrupt_ctrl
    if ctrl is None:
        return False

    # 将 CLINT 中断状态同步到 mip CSR (与 check_pending_interrupts 相同)
    has_pending, mip_bits, _ = ctrl.check_interrupt(hart.id)

    # CLINT MSIP 直接读取 — 硬件中断线, 不依赖 mie.MSIE.
    msip_raw = (mip_bits & (1 << 3)) != 0

    # SSTC stimecmp -> STIP: S 模式直接写 stimecmp CSR 设定时器,
    # 无需 SBI ecall 往返.  硬件语义: mtime >= stimecmp > 0 ⇒ STIP 置位.
    # CLINT.check_interrupt 只返回 MSIP/MTIP, 不含 STIP, 故需在此补齐.
    stimecmp_val = hart._csr_read_raw("stimecmp")
    if stimecmp_val > 0 and ctrl.get_mtime() >= stimecmp_val:
        mip_bits |= INT_SOURCE_MIP_MASK[IntSource.STI]
        has_pending = True

    # AIA IMSIC 或 PLIC 外部中断 (MEIP/SEIP).
    # Always query IMSIC when present — get_pending_mip's raw-eip
    # fallback correctly reports IPIs even when eidelivery=0.
    ext_mip = 0
    if hart._imsic is not None:
        ext_mip = hart._imsic.get_pending_mip(hart.id)
    if ext_mip == 0 and hart._plic is not None:
        ext_mip = hart._plic.get_pending_mip(hart.id)
    mip_bits = mip_bits | ext_mip

    _update_hw_mip(hart, mip_bits)

    if not has_pending and not msip_raw and ext_mip == 0:
        return False

    # WFI 唤醒: 检查 mip & mie (源级).  特殊处理 MSIP: 即使 mie.MSIE=0,
    # 只要 CLINT MSIP 硬件寄存器非零即可唤醒 (与 Rust 行为一致).
    if (hart.mip_val & hart.mie_val) == 0 and not msip_raw:
        return False

    # MSIP 硬件活跃但 mie.MSIE=0: 临时置位 MSIE, 确保后续
    # check_pending_interrupts 可投递该中断到 M 模式 trap handler.
    # (与 Rust ``exec.rs`` line 390-392 逻辑一致)
    if msip_raw and (hart.csrs["mie"].val & (1 << 3)) == 0:
        hart.csrs["mie"].val |= 1 << 3

    hart._waiting = False
    hart._wfi_woken = True
    return True


def check_pending_interrupts(hart: HartWithRegs) -> bool:
    """在指令边界检查是否有待处理且使能的中断.

    若有, 则通过 _take_trap 注入中断并返回 True;
    否则返回 False.

    优先级: MEI > MSI > MTI > SEI > SSI > STI
    (RISC-V Privileged Spec §3.1.9)

    支持通过 mideleg 将中断委派到 S 模式处理:
    - M 模式 + MIE=0: 全局关中断, 不响应任何中断
    - S 模式 + SIE=0: 仅非委派 (M 级) 中断可抢占; 已委派中断被阻塞
    - U 模式: 全部中断全局使能

    中断状态缓存: _int_state_version 跟踪所有中断相关状态变化.
    若版本号未变且 mtime 未到下一唤醒点, 直接返回 False (~50ns),
    避免每条指令遍历 CLINT+CSR+PLIC (~1272ns).
    """
    if hart._interrupt_ctrl is None:
        return False

    ctrl = hart._interrupt_ctrl

    # ---- 快速路径: 缓存命中 ----
    if hart._int_cache_version == hart._int_state_version:
        next_timer = hart._int_cache_next_timer
        if next_timer == 0 or ctrl.get_mtime() < next_timer:
            return False

    # ---- 全量中断检查 ----
    # 1. CLINT: 定时器 + 软件中断
    has_pending, mip_bits, _ = ctrl.check_interrupt(hart.id)

    # 2. STIP via stimecmp (Sstc 扩展)
    #    S 模式直接写 stimecmp CSR 设置定时器, 无需 SBI ecall 往返.
    #    硬件: mtime >= stimecmp > 0 ⇒ STIP 置位; 否则 STIP 清零.
    stimecmp_val = hart._csr_read_raw("stimecmp")
    if stimecmp_val > 0 and hart._interrupt_ctrl.get_mtime() >= stimecmp_val:
        mip_bits |= INT_SOURCE_MIP_MASK[IntSource.STI]
        has_pending = True

    # 3. 外部中断: IMSIC (AIA) 或 PLIC (legacy) — MEIP/SEIP
    #    IMSIC 通过 eidelivery 控制当前谁在驱动外部中断线:
    #    eidelivery=1 → MSI 模式, IMSIC eip/eie 驱动 MEIP/SEIP.
    #    eidelivery=0 → legacy 模式, ext_irq drain 驱动 MEIP/SEIP.
    #
    #    Always query IMSIC when present — get_pending_mip's raw-eip
    #    fallback correctly reports IPIs (IID=1,3) even when eidelivery=0.
    #    Without this, cross-hart IPIs sent before the kernel sets
    #    eidelivery=1 are invisible → SMP boot stalls until the ~1 s
    #    cpu_up timeout expires.  PLIC is still consulted as a fallback
    #    when IMSIC reports nothing.
    ext_mip = 0
    if hart._imsic is not None:
        ext_mip = hart._imsic.get_pending_mip(hart.id)
    if ext_mip == 0 and hart._plic is not None:
        ext_mip = hart._plic.get_pending_mip(hart.id)

    # 合并全部硬件中断源
    mip_bits = mip_bits | ext_mip

    # 更新 mip CSR: 用当前硬件状态替换硬件源位, 保留软件写入位.
    # 必须在 early return 之前执行 — 即使无 pending 中断, 也要清除已撤除的
    # 硬件源位 (如 MSIP 清零后 mip.MSIP 需同步为 0), 否则旧位残留
    # 导致下次 mip & mie 仍命中 -> MSIP 风暴.
    _update_hw_mip(hart, mip_bits)

    if not has_pending and ext_mip == 0:
        # 无中断挂起 -> 更新缓存供后续快速路径使用
        hart._int_cache_version = hart._int_state_version
        hart._int_cache_next_timer = _compute_next_timer(hart, ctrl)
        return False

    # M 模式 + MIE=0 -> 全局关中断
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

    与 deliver_trap 相同, 但用于嵌套中断投递场景 (中断 handler 内再次触发中断).
    """
    hart.clear_reservation()
    was_wfi = hart._waiting
    hart._waiting = False
    if was_wfi:
        hart._wfi_woken = True

    code = trap_cause_code(cause)
    exc_code = code & 0x7FFF_FFFF_FFFF_FFFF

    delegate = False
    if hart.mode != RiscvMode.M:
        choice = hart.csrs["mideleg"].val if is_interrupt else hart.csrs["medeleg"].val
        delegate = bool(choice & (1 << exc_code))

    fn = _trap_deliver_smode if delegate else _trap_deliver_mmode
    fn(hart, code, exc_code, tval, is_interrupt)


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

    ctrl = hart._interrupt_ctrl

    # ---- 快速路径: 缓存命中 ----
    if hart._int_cache_version == hart._int_state_version:
        next_timer = hart._int_cache_next_timer
        if next_timer == 0 or ctrl.get_mtime() < next_timer:
            return False

    # ---- 全量中断检查 ----
    # 1. CLINT: 定时器 + 软件中断
    has_pending, mip_bits, _ = ctrl.check_interrupt(hart.id)

    # 2. STIP via stimecmp (Sstc 扩展)
    #    S 模式直接写 stimecmp CSR 设置定时器, 无需 SBI ecall 往返.
    #    硬件: mtime >= stimecmp > 0 ⇒ STIP 置位; 否则 STIP 清零.
    stimecmp_val = hart._csr_read_raw("stimecmp")
    if stimecmp_val > 0 and hart._interrupt_ctrl.get_mtime() >= stimecmp_val:
        mip_bits |= INT_SOURCE_MIP_MASK[IntSource.STI]
        has_pending = True

    # 3. 外部中断: IMSIC (AIA) 或 PLIC (legacy) — MEIP/SEIP
    #    IMSIC 通过 eidelivery 控制当前谁在驱动外部中断线:
    #    eidelivery=1 → MSI 模式, IMSIC eip/eie 驱动 MEIP/SEIP.
    #    eidelivery=0 → legacy 模式, ext_irq drain 驱动 MEIP/SEIP.
    #
    #    Always query IMSIC when present — get_pending_mip's raw-eip
    #    fallback correctly reports IPIs (IID=1,3) even when eidelivery=0.
    #    Without this, cross-hart IPIs sent before the kernel sets
    #    eidelivery=1 are invisible → SMP boot stalls until the ~1 s
    #    cpu_up timeout expires.  PLIC is still consulted as a fallback
    #    when IMSIC reports nothing.
    ext_mip = 0
    if hart._imsic is not None:
        ext_mip = hart._imsic.get_pending_mip(hart.id)
    if ext_mip == 0 and hart._plic is not None:
        ext_mip = hart._plic.get_pending_mip(hart.id)

    # 合并全部硬件中断源
    mip_bits = mip_bits | ext_mip

    # 更新 mip CSR: 用当前硬件状态替换硬件源位, 保留软件写入位.
    _update_hw_mip(hart, mip_bits)

    if not has_pending and ext_mip == 0:
        # 无中断挂起 -> 更新缓存
        hart._int_cache_version = hart._int_state_version
        hart._int_cache_next_timer = _compute_next_timer(hart, ctrl)
        return False

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

    # 使用模块级预计算常量, 避免每条指令分配 list + 6 tuple.
    for mask, trap_type in _INT_PRIORITY:
        if not (masked & mask) or ((mideleg & mask) and not s_mode_global):
            continue
        deliver_trap_nested_enabled(
            hart,
            trap_type,
            tval=0,
            is_interrupt=True,
        )
        return True

    return False
