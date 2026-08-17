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

from pyremu.core.hart import MSTATUS_MXR, MSTATUS_SUM, RiscvMode
from pyremu.core.registers import check_csr_access, CsrAccessError
from pyremu.core.trap_def import TrapType
from pyremu.core.trap_handler import deliver_trap
from pyremu.memory.mmu import PAGE_SIZE, PTE_R, PTE_U, PTE_X, SATP_MODE_BARE, translate_va
from pyremu.memory.pmp import PmpAccessInfo
from pyremu.utils.mask import mask16, mask64

if TYPE_CHECKING:
    from pyremu.core.hart import HartWithRegs
    from pyremu.memory.bus import Bus
    from pyremu.memory.pmp import Pmp
    from pyremu.memory.tlb import TLB


class MemoryAccessFault(Exception):
    """内存访问异常基类 — mem_read / mem_write 投递陷阱后的控制流哨兵.

    所有 handler 必须允许子类异常传播至 ``exec_instr``,
    后者捕获并返回 0 (PC 已重定向).
    """


class AlignmentFault(MemoryAccessFault):
    """非对齐访存 (LdAddrMisaligned / StAddrMisaligned)."""


class PageFault(MemoryAccessFault):
    """页表翻译失败 (LdPageFault / StPageFault)."""


class AccessFault(MemoryAccessFault):
    """PMP 或 PMA 拒绝访问 (LdAccessFault / StAccessFault)."""


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


def _check_pte_perm(
    effective_mode: int,
    perm: int,
    mstatus: int,
    *,
    is_write: bool = False,
    is_execute: bool = False,
) -> bool:
    """验证 PTE 权限标志是否符合当前有效特权级和 mstatus 扩展.

    在 TLB 命中或页表遍历后调用, 检查 SUM/MXR/U-bit 三个维度:

    - **SUM** (bit 18): S 模式访问 U 模式页 (PTE.U=1), 仅当 SUM=1 时允许
    - **MXR** (bit 19): 使能可执行页的可读性。MXR=1 且读请求且 PTE.X=1
      时, 即使 PTE.R=0 也允许。仅影响读 (load), 不影响写或取指。
    - **U-bit**: U 模式只能访问 U=1 的页; S/M 模式访问 U=0 的页无条件允许

    Returns:
        True 若权限通过, False 否则 (应触发 PageFault).
    """
    is_user_page = (perm & PTE_U) != 0

    if effective_mode == RiscvMode.U.value:
        # U 模式只能访问用户页 (U=1)
        if not is_user_page:
            return False
    elif effective_mode == RiscvMode.S.value:
        # S 模式访问用户页需 SUM=1 (RISC-V Privileged Spec §4.1.12)
        if is_user_page and not (mstatus & MSTATUS_SUM):
            return False

    # MXR: 可执行但不可读的页在 MXR=1 时允许 load 读取
    # (不影响 store 与取指, 仅影响数据 load — is_execute=False, is_write=False)
    if (mstatus & MSTATUS_MXR) and not is_write and not is_execute:
        if (perm & PTE_X) and not (perm & PTE_R):
            return True  # MXR 扩展: X 替代 R

    # 基础权限检查 (R/W/X)
    if is_execute and not (perm & PTE_X):
        return False
    if is_write and not (perm & (1 << 2)):  # PTE_W = bit 2
        return False
    if not is_write and not is_execute and not (perm & PTE_R):
        return False

    return True


def translate_addr(
    hart: HartWithRegs,
    va: int,
    *,
    is_write: bool = False,
    is_execute: bool = False,
) -> tuple[bool, int]:
    """完整地址翻译: TLB 查找 + 页表遍历 + SUM/MXR 权限检查.

    供 mem_read / mem_write / check_instruction_fetch 使用。

    MMIO 设备地址 (经 Bus.is_device_addr 判断) 不会插入 TLB,
    因为设备寄存器读写可能有副作用, 不能被缓存.

    Returns:
        (success, pa) — success=False 表示翻译失败 (内部已投递 trap).
    """
    mode = hart.mmu_mode
    # RISC-V 规范: M 模式始终使用 Bare 翻译, 无视 satp.MODE.
    #
    # 例外 — MPRV (mstatus bit 17): 置位时 M-mode loads/stores 按 MPP
    # 所指示的特权级进行地址翻译和 PMP 检查. 这是 sbi_unpriv 系列 API
    # (sbi_get_insn 等) 能够读取 S/U-mode 虚拟地址的基础.
    if mode == SATP_MODE_BARE:
        return True, mask64(va)

    # 计算有效特权级 — MPRV=1 时 M 模式使用 MPP 作为翻译特权级
    effective_mode = hart.mode.value
    if hart.mode == RiscvMode.M:
        mprv = (hart.mstatus_val >> 17) & 1
        mpp = (hart.mstatus_val >> 11) & 0x3
        if not mprv or mpp == RiscvMode.M.value:
            return True, mask64(va)
        # MPP 为 S 或 U 模式 — 继续走 MMU 翻译 + PMP 检查
        effective_mode = mpp

    # TLB 查找 (ASID-tagged: Bare 模式 asid=0 匹配全部)
    vpn = va >> 12
    tlb: TLB = hart.dtlb
    _asid = mask16(hart.satp_val >> 44) if hart.mmu_mode != SATP_MODE_BARE else 0
    hit, ppn, perm = tlb.lookup(vpn, asid=_asid)
    if hit:
        # _check_pte_perm 需等 TLB 存储真实 PTE 权限 (非硬编码 0xF) 后启用
        offset = va & (PAGE_SIZE - 1)
        pa = mask64((ppn << 12 | offset))
        return True, pa

    # TLB miss — 执行页表遍历
    if hart._mem_read_phy is None:
        return False, 0

    satp = hart.satp_val
    ok, pa, perm = translate_va(va, satp, hart._mem_read_phy)
    if not ok:
        return False, 0

    # _check_pte_perm 需等 TLB 存储真实 PTE 权限 (非硬编码 0xF) 后启用。
    # 当前 perm 虽由 translate_va 正确传入, 但与 TLB 命中路径 (perm=0xF)
    # 不一致 — 同一页首次访问受检而后续不受检会使 bug 更隐蔽。 待全链
    # (translate_va ->translate_addr ->tlb.insert) 统一传递真实 perm 后,
    # 两路径同步启用 _check_pte_perm。

    # MMIO 地址不可缓存 — 跳过 TLB 插入
    # 设备寄存器读写有副作用, 缓存会导致重复读写时绕过设备
    bus: Bus | None = hart._bus
    if bus is not None and bus.is_device_addr(pa):
        return True, pa

    # 将翻译结果插入 TLB 缓存 (标记当前 hart 的 mdid, 供 mfence.did 按域刷新)
    new_vpn = va >> 12
    new_ppn = pa >> 12
    tlb.insert(new_vpn, new_ppn, perm=0xF, level=0, mdid=hart.mdid_val, asid=_asid)

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
        raise AlignmentFault

    # 跨页边界检查: M/D 模式与 Bare 模式下 VA==PA, 无需拆分.
    # 仅 S/U 模式且 MMU 使能时 VA->PA 映射可能不连续.
    _mmu_active = (
        hart.mode not in (RiscvMode.M, RiscvMode.D) and hart.mmu_mode != SATP_MODE_BARE
    )
    page_end = (addr & ~0xFFF) + 0x1000
    if _mmu_active and size > 1 and addr + size > page_end:
        # 跨页: 分别翻译两页, 任一失败即抛异常
        first_size = page_end - addr
        second_size = size - first_size
        ok1, pa1 = translate_addr(hart, addr, is_write=False, is_execute=False)
        if not ok1:
            deliver_trap(hart, TrapType.LdPageFault, tval=addr, is_interrupt=False)
            raise PageFault
        ok2, pa2 = translate_addr(hart, addr + first_size, is_write=False, is_execute=False)
        if not ok2:
            deliver_trap(
                hart, TrapType.LdPageFault, tval=addr + first_size, is_interrupt=False
            )
            raise PageFault
        # 跨页 PMP: 逐页检查
        pmp: Pmp = hart._pmp
        for check_pa, check_size in ((pa1, first_size), (pa2, second_size)):
            if not pmp.check(
                PmpAccessInfo(
                    pa=check_pa,
                    size=check_size,
                    mode_val=hart.mode.value,
                    mstatus_val=hart.mstatus_val,
                    is_write=False,
                    pmpsplit=hart.pmpsplit_val,
                    mdid=hart.mdid_val,
                )
            ):
                deliver_trap(hart, TrapType.LdAccessFault, tval=addr, is_interrupt=False)
                raise AccessFault
        return hart._mem_read_phy(pa1, first_size) + hart._mem_read_phy(pa2, second_size)

    # 单页快速路径
    ok, pa = translate_addr(hart, addr, is_write=False, is_execute=False)
    if not ok:
        deliver_trap(hart, TrapType.LdPageFault, tval=addr, is_interrupt=False)
        raise PageFault

    # PMP 检查 — 物理内存保护 (对 M 模式且 MPRV=0 自动放行)
    pmp: Pmp = hart._pmp
    if not pmp.check(
        PmpAccessInfo(
            pa=pa,
            size=size,
            mode_val=hart.mode.value,
            mstatus_val=hart.mstatus_val,
            is_write=False,
            pmpsplit=hart.pmpsplit_val,
            mdid=hart.mdid_val,
        )
    ):
        deliver_trap(hart, TrapType.LdAccessFault, tval=addr, is_interrupt=False)
        raise AccessFault

    # PMA 检查 — 物理地址必须落在有效区域 (RAM 或已注册设备)
    bus: Bus | None = hart._bus
    if bus is not None and not bus.is_valid_addr(pa):
        deliver_trap(hart, TrapType.LdAccessFault, tval=addr, is_interrupt=False)
        raise AccessFault

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
        raise AlignmentFault

    # 跨页边界检查: M/D 模式与 Bare 模式下无需拆分.
    _mmu_active = (
        hart.mode not in (RiscvMode.M, RiscvMode.D) and hart.mmu_mode != SATP_MODE_BARE
    )
    page_end = (addr & ~0xFFF) + 0x1000
    if _mmu_active and size > 1 and addr + size > page_end:
        first_size = page_end - addr
        second_size = size - first_size
        ok1, pa1 = translate_addr(hart, addr, is_write=True, is_execute=False)
        if not ok1:
            deliver_trap(hart, TrapType.StPageFault, tval=addr, is_interrupt=False)
            raise PageFault
        ok2, pa2 = translate_addr(hart, addr + first_size, is_write=True, is_execute=False)
        if not ok2:
            deliver_trap(
                hart, TrapType.StPageFault, tval=addr + first_size, is_interrupt=False
            )
            raise PageFault
        pmp: Pmp = hart._pmp
        for check_pa, check_size in ((pa1, first_size), (pa2, second_size)):
            if not pmp.check(
                PmpAccessInfo(
                    pa=check_pa,
                    size=check_size,
                    mode_val=hart.mode.value,
                    mstatus_val=hart.mstatus_val,
                    is_write=True,
                    pmpsplit=hart.pmpsplit_val,
                    mdid=hart.mdid_val,
                )
            ):
                deliver_trap(hart, TrapType.StAccessFault, tval=addr, is_interrupt=False)
                raise AccessFault
        hart.clear_reservation()
        hart._mem_write_phy(pa1, data[:first_size])
        hart._mem_write_phy(pa2, data[first_size:])
        return

    # 单页快速路径
    ok, pa = translate_addr(hart, addr, is_write=True, is_execute=False)
    if not ok:
        deliver_trap(hart, TrapType.StPageFault, tval=addr, is_interrupt=False)
        raise PageFault

    # PMP 检查
    pmp: Pmp = hart._pmp
    if not pmp.check(
        PmpAccessInfo(
            pa=pa,
            size=size,
            mode_val=hart.mode.value,
            mstatus_val=hart.mstatus_val,
            is_write=True,
            pmpsplit=hart.pmpsplit_val,
            mdid=hart.mdid_val,
        )
    ):
        deliver_trap(hart, TrapType.StAccessFault, tval=addr, is_interrupt=False)
        raise AccessFault

    # PMA 检查
    bus: Bus | None = hart._bus
    if bus is not None and not bus.is_valid_addr(pa):
        deliver_trap(hart, TrapType.StAccessFault, tval=addr, is_interrupt=False)
        raise AccessFault

    # RISC-V spec §8.2: any store by any hart invalidates all LR reservations
    # on that hart.  Without this, an LR in Python mode followed by a store
    # (to any address) would leave the reservation intact, allowing a subsequent
    # SC to succeed when it should fail.
    hart.clear_reservation()
    hart._mem_write_phy(pa, data)


# ============================================================
#  取指地址翻译 (VA -> PA via itlb -> page walk)
# ============================================================


def _translate_instruction_addr(
    hart: HartWithRegs,
    va: int,
    tlb: TLB,
) -> tuple[bool, int, int]:
    """通过 itlb 或页表遍历将取指虚拟地址翻译为物理地址.

    若翻译失败, 内部投递 InstrPageFault 陷态并返回 (False, 0, 0).

    与数据侧的 ``translate_addr`` 平行: 取指用 itlb, 读写用 dtlb.

    Returns:
        (ok, pa, perm) — perm 供调用方做 SUM 权限检查.
    """
    vpn = va >> 12
    _asid = mask16(hart.satp_val >> 44) if hart.mmu_mode != SATP_MODE_BARE else 0
    hit, ppn, perm = tlb.lookup(vpn, asid=_asid)
    if hit:
        offset = va & (PAGE_SIZE - 1)
        pa = mask64((ppn << 12 | offset))
        return True, pa, perm

    # itlb miss — 执行页表遍历
    if hart._mem_read_phy is None:
        deliver_trap(hart, TrapType.InstrPageFault, tval=va, is_interrupt=False)
        return False, 0, 0

    satp = hart.satp_val
    ok, pa, perm = translate_va(va, satp, hart._mem_read_phy)
    if not ok:
        deliver_trap(hart, TrapType.InstrPageFault, tval=va, is_interrupt=False)
        return False, 0, 0

    # 将翻译结果插入 itlb (标记当前 mdid, 供 mfence.did 按域刷新)
    new_vpn = va >> 12
    new_ppn = pa >> 12
    # 跳过 MMIO 地址的缓存 (与 dtlb 策略一致)
    bus: Bus | None = hart._bus
    if bus is None or not bus.is_device_addr(pa):
        tlb.insert(new_vpn, new_ppn, perm=perm, level=0, mdid=hart.mdid_val, asid=_asid)
    return True, pa, perm


# ============================================================
#  取指校验 (VA -> PA via itlb -> PMP execute check)
# ============================================================


def check_instruction_fetch(
    hart: HartWithRegs,
    va: int,
) -> tuple[bool, int]:
    """校验从虚拟地址 *va* 取指的合法性, 返回 (ok, pa).

    路径: VA -> PA (itlb 或页表遍历) -> SUM 权限检查 -> PMP (is_execute=True).

    可能触发的陷态:
    - InstrPageFault: 页表翻译失败 或 SUM 拒绝 (S 模式取指于 U 页)
    - InstrAccessFault: PMP 拒绝取指 (X=0 或未匹配)

    调用方在 ok=True 时使用返回的 pa 读取指令字节.
    """
    mode = hart.mmu_mode
    # RISC-V 规范: M 模式取指始终走物理地址 (MPRV 不影响取指)
    if mode == SATP_MODE_BARE or hart.mode == RiscvMode.M:
        pa = mask64(va)
    else:
        ok, pa, perm = _translate_instruction_addr(hart, va, hart.itlb)
        if not ok:
            return False, 0
        # S 模式不能从 U=1 的页取指 (此乃 RISC-V 基础规则, 非 SUM 扩展).
        # 需等 TLB 存储真实 PTE 权限 (非硬编码 0xF) 后才能启用此检查,
        # 否则所有页均被误判为用户页.
        # TODO: 待 translate_va->translate_addr 完整传递 PTE perm 后启用.
        # is_user_page = (perm & PTE_U) != 0
        # if hart.mode == RiscvMode.S and is_user_page:
        #     deliver_trap(hart, TrapType.InstrPageFault, tval=va, is_interrupt=False)
        #     return False, 0

    # PMP 检查 — 所有模式均需通过, is_execute=True
    # RISC-V spec §3.1.6.3: 取指无视 MPRV, 始终用当前特权级.
    # 清除 MPRV 位确保 M 模式取指绕过 PMP (无需关心 MPP).
    pmp: Pmp = hart._pmp
    _fetch_mstatus = hart.mstatus_val & ~(1 << 17)  # MPRV=0 for instruction fetch
    if not pmp.check(
        PmpAccessInfo(
            pa=pa,
            size=4,
            mode_val=hart.mode.value,
            mstatus_val=_fetch_mstatus,
            is_execute=True,
            pmpsplit=hart.pmpsplit_val,
            mdid=hart.mdid_val,
        )
    ):
        deliver_trap(hart, TrapType.InstrAccessFault, tval=va, is_interrupt=False)
        return False, 0

    # PMA 检查
    bus: Bus | None = hart._bus
    if bus is not None and not bus.is_valid_addr(pa):
        deliver_trap(hart, TrapType.InstrAccessFault, tval=va, is_interrupt=False)
        return False, 0

    return True, pa
