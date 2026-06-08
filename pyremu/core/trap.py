#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 23:19:24
# Last modified at 2026/06/08 星期一

"""
RISC-V 特权架构定义的 trap 类型及对应的 mcause/scause 编码。
"""

from enum import Enum

TrapType = Enum(
    "TrapType",
    (
        # === 异常 (Exception) ===
        "InstrAddrMisaligned",  # 0: 指令地址未对齐
        "InstrAccessFault",  # 1: 指令访问错误
        "IllInstr",  # 2: 非法指令
        "Breakpoint",  # 3: 断点 (ebreak)
        "LdAddrMisaligned",  # 4: 载入地址未对齐
        "LdAccessFault",  # 5: 载入访问错误
        "StAddrMisaligned",  # 6: 存储地址未对齐
        "StAccessFault",  # 7: 存储访问错误
        "EcallFromUmode",  # 8: U 模式 ecall
        "EcallFromSmode",  # 9: S 模式 ecall
        "EcallFromMmode",  # 11: M 模式 ecall
        "InstrPageFault",  # 12: 指令页错误
        "LdPageFault",  # 13: 载入页错误
        "StPageFault",  # 15: 存储页错误
        # === 中断 (Interrupt) — mcause bit 63 置 1 ===
        "SoftInterrupt",  # 软件中断 (通用)
        "UmodeSoftInterrupt",  # 0: U 模式软件中断
        "SmodeSoftInterrupt",  # 1: S 模式软件中断
        "MmodeSoftInterrupt",  # 3: M 模式软件中断
        "UmodeTimerInterrupt",  # 4: U 模式定时器中断
        "SmodeTimerInterrupt",  # 5: S 模式定时器中断
        "MmodeTimerInterrupt",  # 7: M 模式定时器中断
        "UmodeExternInterrupt",  # 8: U 模式外部中断
        "SmodeExternInterrupt",  # 9: S 模式外部中断
        "MmodeExternInterrupt",  # 11: M 模式外部中断
    ),
)

# ============================================================
#  TrapType → mcause/scause 编码映射
#  异常: code = 直接编号 (bit 63 = 0)
#  中断: code = 直接编号 | (1 << 63)
# ============================================================

# 每个 TrapType 对应的 mcause 异常/中断编号
_TRAP_CAUSE_CODE: dict[TrapType, int] = {
    # 异常
    TrapType.InstrAddrMisaligned: 0,
    TrapType.InstrAccessFault: 1,
    TrapType.IllInstr: 2,
    TrapType.Breakpoint: 3,
    TrapType.LdAddrMisaligned: 4,
    TrapType.LdAccessFault: 5,
    TrapType.StAddrMisaligned: 6,
    TrapType.StAccessFault: 7,
    TrapType.EcallFromUmode: 8,
    TrapType.EcallFromSmode: 9,
    TrapType.EcallFromMmode: 11,
    TrapType.InstrPageFault: 12,
    TrapType.LdPageFault: 13,
    TrapType.StPageFault: 15,
    # 中断 (编号 | 中断标志位)
    TrapType.SoftInterrupt: 1 | (1 << 63),
    TrapType.UmodeSoftInterrupt: 0 | (1 << 63),
    TrapType.SmodeSoftInterrupt: 1 | (1 << 63),
    TrapType.MmodeSoftInterrupt: 3 | (1 << 63),
    TrapType.UmodeTimerInterrupt: 4 | (1 << 63),
    TrapType.SmodeTimerInterrupt: 5 | (1 << 63),
    TrapType.MmodeTimerInterrupt: 7 | (1 << 63),
    TrapType.UmodeExternInterrupt: 8 | (1 << 63),
    TrapType.SmodeExternInterrupt: 9 | (1 << 63),
    TrapType.MmodeExternInterrupt: 11 | (1 << 63),
}


def trap_cause_code(trap: TrapType) -> int:
    """返回 TrapType 对应的 mcause/scause 编码值 (含中断标志位)."""
    return _TRAP_CAUSE_CODE.get(trap, 0)


def trap_is_interrupt(trap: TrapType) -> bool:
    """判断该 TrapType 是否为中断 (而非异常)."""
    code = _TRAP_CAUSE_CODE.get(trap, 0)
    return (code >> 63) & 1 == 1
