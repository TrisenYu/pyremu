#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""内存访问辅助函数 — VA->PA 翻译, CSR 权限校验, PMP/PMA 检查.

原 MemoryAccessor mixin 已拆分为本模块中的独立函数,
所有函数将 hart 作为显式第一参数, 消除 mixin 的隐式依赖和静态分析告警.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pyremu.core.hart import RiscvMode
from pyremu.core.registers import CsrAccessError, check_csr_access
from pyremu.core.trap import TrapType
from pyremu.core.trap_handler import deliver_trap
from pyremu.memory.mmu import PAGE_SIZE, SATP_MODE_BARE, translate_va
from pyremu.memory.pmp import PmpAccessInfo

if TYPE_CHECKING:
    from pyremu.core.hart import HartWithRegs
    from pyremu.memory.bus import Bus
    from pyremu.memory.pmp import Pmp
    from pyremu.memory.tlb import TLB

# ============================================================
#  PMP CSR 地址范围验证
# ============================================================

_PMPADDR_BASE, _PMPCFG_BASE = 0x3B0, 0x3A0


def _pmp_csr_valid(csr_addr: int, max_entries: int) -> bool:
    """验证 PMP CSR 地址是否在合法范围内 (基于 hart 的条目数).

    越界的 pmpaddr / pmpcfg 访问触发 IllInstr.
    """
    if _PMPADDR_BASE <= csr_addr <= 0x3EF:
        return csr_addr - _PMPADDR_BASE < max_entries
    if _PMPCFG_BASE <= csr_addr <= 0x3AF and (csr_addr & 1) == 0:
        group = (csr_addr - _PMPCFG_BASE) // 2
        return group * 8 < max_entries
    return True  # 非 PMP CSR


# ============================================================
#  内存后端注入
# ============================================================


def inject_memory_backend(
    hart: HartWithRegs,
    read_fn: Callable[[int, int], bytes],
    write_fn: Callable[[int, bytes], None],
) -> None:
    """注入物理内存后端到 hart.

    调用后 mem_read / mem_write 将使用注入的回调访问物理内存.
    未注入时内存访问会触发 hart 的 _take_trap (AccessFault).
    """
    hart._mem_read_phy = read_fn
    hart._mem_write_phy = write_fn


# ============================================================
#  CSR 访问校验
# ============================================================


def validate_csr(
    hart: HartWithRegs,
    csr_addr: int,
    is_write: bool = False,
) -> None:
    """验证 CSR 访问的合法性 (PMP 范围 + 特权级 + 读写权限).

    检查顺序:
    1. PMP CSR 地址范围 (越界 -> CsrAccessError -> IllInstr)
    2. 特权级权限 (低特权访问高特权 CSR -> CsrAccessError -> IllInstr)
    3. 只读检查 (写入只读 CSR -> CsrAccessError -> IllInstr)

    由 handle_sys 的各 CSR 指令 handler 在访问前调用.
    """
    # PMP CSR 范围检查
    if not _pmp_csr_valid(csr_addr, hart._pmp_entries):
        raise CsrAccessError(csr_addr, "pmp out of range")

    # 特权级检查
    check_csr_access(csr_addr, hart.mode.value, is_write)


# ============================================================
#  地址翻译 (VA -> PA, 经 TLB 缓存)
# ============================================================


def translate_addr(
    hart: HartWithRegs,
    va: int,
) -> tuple[bool, int]:
    """完整地址翻译: TLB 查找 + 页表遍历. 供 mem_read / mem_write 和调试器使用.

    MMIO 设备地址 (经 Bus.is_device_addr 判断) 不会插入 TLB,
    因为设备寄存器读写可能有副作用, 不能被缓存.

    Returns:
        (success, pa) — success=False 表示翻译失败.
    """
    mode = hart.mmu_mode
    # RISC-V 规范: M 模式始终使用 Bare 翻译, 无视 satp.MODE.
    #
    # 例外 — MPRV (mstatus bit 17): 置位时 M-mode loads/stores 按 MPP
    # 所指示的特权级进行地址翻译和 PMP 检查. 这是 sbi_unpriv 系列 API
    # (sbi_get_insn 等) 能够读取 S/U-mode 虚拟地址的基础.
    if mode == SATP_MODE_BARE:
        return True, va & 0xFFFF_FFFF_FFFF_FFFF

    if hart.mode == RiscvMode.M:
        mprv = (hart.mstatus_val >> 17) & 1
        if not mprv:
            return True, va & 0xFFFF_FFFF_FFFF_FFFF
        # MPRV=1: 取 MPP 作为有效特权级. 若 MPP=M 则仍走 Bare.
        mpp = (hart.mstatus_val >> 11) & 0x3
        if mpp == RiscvMode.M.value:
            return True, va & 0xFFFF_FFFF_FFFF_FFFF
        # MPP 为 S 或 U 模式 — 继续走 MMU 翻译 + PMP 检查

    # TLB 查找
    vpn = va >> 12
    tlb: TLB = hart.dtlb
    hit, ppn, perm = tlb.lookup(vpn)
    if hit:
        offset = va & (PAGE_SIZE - 1)
        pa = (ppn << 12 | offset) & 0xFFFF_FFFF_FFFF_FFFF
        return True, pa

    # TLB miss — 执行页表遍历
    if hart._mem_read_phy is None:
        return False, 0

    satp = hart.satp_val
    ok, pa = translate_va(va, satp, hart._mem_read_phy)
    if not ok:
        return False, 0

    # MMIO 地址不可缓存 — 跳过 TLB 插入
    # 设备寄存器读写有副作用, 缓存会导致重复读写时绕过设备
    bus: Bus | None = hart._bus
    if bus is not None and bus.is_device_addr(pa):
        return True, pa

    # 将翻译结果插入 TLB 缓存 (标记当前 hart 的 mdid, 供 mfence.did 按域刷新)
    new_vpn = va >> 12
    new_ppn = pa >> 12
    tlb.insert(new_vpn, new_ppn, perm=0xF, level=0, mdid=hart.mdid_val)

    return True, pa


# ============================================================
#  虚拟内存读写
# ============================================================


def mem_read(
    hart: HartWithRegs,
    addr: int,
    size: int,
) -> bytes:
    """从虚拟地址 *addr* 读取 *size* 字节.

    经过路径: 对齐检查 -> VA->PA (TLB/页表) -> PMP -> PMA -> 物理内存后端.

    可能触发的陷态:
    - LdAddrMisaligned: 地址未对齐
    - LdPageFault: 页表翻译失败
    - LdAccessFault: PMP 拒绝 或 PMA 违例
    """
    if hart._mem_read_phy is None:
        raise NotImplementedError(
            f"Memory read @ {addr:#018x} ({size} B): no memory backend attached"
        )

    # 对齐检查 (RISC-V Privileged Spec §3.6.1)
    if size > 1 and (addr & (size - 1)) != 0:
        deliver_trap(hart, TrapType.LdAddrMisaligned, tval=addr, is_interrupt=False)
        return b"\x00" * size

    # 地址翻译
    ok, pa = translate_addr(hart, addr)
    if not ok:
        deliver_trap(hart, TrapType.LdPageFault, tval=addr, is_interrupt=False)
        return b"\x00" * size

    # PMP 检查 — 物理内存保护 (对 M 模式且 MPRV=0 自动放行)
    pmp: Pmp = hart._pmp
    if not pmp.check(PmpAccessInfo(
        pa=pa, size=size, mode_val=hart.mode.value, mstatus_val=hart.mstatus_val,
        is_write=False, pmpsplit=hart.pmpsplit_val, mdid=hart.mdid_val,
    )):
        deliver_trap(hart, TrapType.LdAccessFault, tval=addr, is_interrupt=False)
        return b"\x00" * size

    # PMA 检查 — 物理地址必须落在有效区域 (RAM 或已注册设备)
    bus: Bus | None = hart._bus
    if bus is not None and not bus.is_valid_addr(pa):
        deliver_trap(hart, TrapType.LdAccessFault, tval=addr, is_interrupt=False)
        return b"\x00" * size

    return hart._mem_read_phy(pa, size)


def mem_write(
    hart: HartWithRegs,
    addr: int,
    data: bytes,
) -> None:
    """向虚拟地址 *addr* 写入 *data*.

    经过路径: 对齐检查 -> VA->PA (TLB/页表) -> PMP -> PMA -> 物理内存后端.

    可能触发的陷态:
    - StAddrMisaligned: 地址未对齐
    - StPageFault: 页表翻译失败
    - StAccessFault: PMP 拒绝 或 PMA 违例
    """
    if hart._mem_write_phy is None:
        raise NotImplementedError(
            f"Memory write @ {addr:#018x} ({len(data)} B): no memory backend attached"
        )

    size = len(data)

    # 对齐检查
    if size > 1 and (addr & (size - 1)) != 0:
        deliver_trap(hart, TrapType.StAddrMisaligned, tval=addr, is_interrupt=False)
        return

    # 地址翻译
    ok, pa = translate_addr(hart, addr)
    if not ok:
        deliver_trap(hart, TrapType.StPageFault, tval=addr, is_interrupt=False)
        return

    # PMP 检查
    pmp: Pmp = hart._pmp
    if not pmp.check(PmpAccessInfo(
        pa=pa, size=size, mode_val=hart.mode.value, mstatus_val=hart.mstatus_val,
        is_write=True, pmpsplit=hart.pmpsplit_val, mdid=hart.mdid_val,
    )):
        deliver_trap(hart, TrapType.StAccessFault, tval=addr, is_interrupt=False)
        return

    # PMA 检查
    bus: Bus | None = hart._bus
    if bus is not None and not bus.is_valid_addr(pa):
        deliver_trap(hart, TrapType.StAccessFault, tval=addr, is_interrupt=False)
        return

    hart._mem_write_phy(pa, data)


# ============================================================
#  取指校验 (VA -> PA via itlb -> PMP execute check)
# ============================================================


def check_instruction_fetch(
    hart: HartWithRegs,
    va: int,
) -> tuple[bool, int]:
    """校验从虚拟地址 *va* 取指的合法性, 返回 (ok, pa).

    路径: VA -> PA (itlb 或页表遍历) -> PMP (is_execute=True).

    可能触发的陷态:
    - InstrPageFault: 页表翻译失败 (非 Bare 模式)
    - InstrAccessFault: PMP 拒绝取指 (X=0 或未匹配)

    调用方在 ok=True 时使用返回的 pa 读取指令字节.
    """
    mode = hart.mmu_mode
    # RISC-V 规范: M 模式取指始终走物理地址 (MPRV 不影响取指)
    if mode == SATP_MODE_BARE or hart.mode == RiscvMode.M:
        pa = va & 0xFFFF_FFFF_FFFF_FFFF
    else:
        # itlb 查找
        vpn = va >> 12
        tlb: TLB = hart.itlb
        hit, ppn, perm = tlb.lookup(vpn)
        if hit:
            offset = va & (PAGE_SIZE - 1)
            pa = (ppn << 12 | offset) & 0xFFFF_FFFF_FFFF_FFFF
        else:
            # itlb miss -> 页表遍历
            if hart._mem_read_phy is None:
                deliver_trap(
                    hart, TrapType.InstrPageFault, tval=va, is_interrupt=False
                )
                return False, 0

            satp = hart.satp_val
            ok, pa = translate_va(va, satp, hart._mem_read_phy)
            if not ok:
                deliver_trap(
                    hart, TrapType.InstrPageFault, tval=va, is_interrupt=False
                )
                return False, 0

            # 将翻译结果插入 itlb (标记当前 mdid, 供 mfence.did 按域刷新)
            new_vpn = va >> 12
            new_ppn = pa >> 12
            # 跳过 MMIO 地址的缓存 (与 dtlb 策略一致)
            bus: Bus | None = hart._bus
            if bus is None or not bus.is_device_addr(pa):
                tlb.insert(
                    new_vpn, new_ppn, perm=0xF, level=0, mdid=hart.mdid_val
                )

    # PMP 检查 — 所有模式均需通过, is_execute=True
    pmp: Pmp = hart._pmp
    if not pmp.check(PmpAccessInfo(
        pa=pa, size=4, mode_val=hart.mode.value, mstatus_val=hart.mstatus_val,
        is_execute=True, pmpsplit=hart.pmpsplit_val, mdid=hart.mdid_val,
    )):
        deliver_trap(
            hart, TrapType.InstrAccessFault, tval=va, is_interrupt=False
        )
        return False, 0

    # PMA 检查
    bus: Bus | None = hart._bus
    if bus is not None and not bus.is_valid_addr(pa):
        deliver_trap(
            hart, TrapType.InstrAccessFault, tval=va, is_interrupt=False
        )
        return False, 0

    return True, pa
