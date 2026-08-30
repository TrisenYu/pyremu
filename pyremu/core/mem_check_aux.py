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
from pyremu.memory.mmu import (
    PAGE_SIZE,
    PTE_R,
    PTE_U,
    PTE_W,
    PTE_X,
    SATP_MODE_BARE,
    translate_va,
)
from pyremu.memory.pmp import PmpAccessInfo
from pyremu.utils.mask import mask16, mask64

if TYPE_CHECKING:
    from pyremu.core.hart import HartWithRegs
    from pyremu.memory.bus import Bus
    from pyremu.memory.pmp import Pmp
    from pyremu.memory.tlb import TLB


_PMPADDR_BASE, _PMPCFG_BASE = 0x3B0, 0x3A0


class MemoryAccessFault(Exception):
    """内存访问异常基类 — mem_read / mem_write 投递陷阱后的控制流哨兵.

    所有 handler 必须允许子类异常传播至 ``exec_instr``,
    后者捕获并返回 0 (PC 已重定向).
    """
    pass

class AlignmentFault(MemoryAccessFault):
    """非对齐访存 (LdAddrMisaligned / StAddrMisaligned)."""
    pass

class PageFault(MemoryAccessFault):
    """页表翻译失败 (LdPageFault / StPageFault)."""
    pass

class AccessFault(MemoryAccessFault):
    """PMP 或 PMA 拒绝访问 (LdAccessFault / StAccessFault)."""


# ============================================================
#  PMP CSR 地址范围验证
# ============================================================


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

    与 Rust 侧 ``check_pte_perm`` (translate.rs) 逐条对齐, 检查三个维度:

    - **U-bit**: U 模式只能访问 U=1 页; S 模式访问 U=1 页时, 数据访问
      需 SUM=1, 取指则永不许可 (RISC-V §4.3.2 基础规则, 与 SUM 无关).
    - **W 权限**: 写访问要求 PTE.R 且 PTE.W 同时置位 (R=0/W=1 为保留编码,
      但 Rust 侧强制 R&W, 此处保持一致).
    - **MXR** (bit 19): 读访问且 MXR=1 时, X=1 且 R=0 的页允许读取,
      仅影响数据 load, 不影响写或取指.

    Returns:
        True 若权限通过, False 否则 (应触发 PageFault).
    """
    pte_u = (perm & PTE_U) != 0
    pte_r = (perm & PTE_R) != 0
    pte_w = (perm & PTE_W) != 0
    pte_x = (perm & PTE_X) != 0

    if effective_mode == RiscvMode.U.value:
        if not pte_u:
            return False
    elif effective_mode == RiscvMode.S.value:
        # 取指从不许可从 U 页 (基础规则), 数据访问才看 SUM
        if pte_u and (is_execute or not (mstatus & MSTATUS_SUM)):
            return False

    if is_execute:
        return pte_x
    if is_write:
        return pte_r and pte_w
    # 读: MXR=1 时 X-only 页可读
    if not pte_r:
        return (mstatus & MSTATUS_MXR) != 0 and pte_x
    return True


def _effective_mode(hart: HartWithRegs) -> int | None:
    """计算地址翻译的有效特权级 (镜像 Rust ``effective_mode``).

    Returns:
        有效特权级值 (U/S), 或 None 表示绕过 MMU 走 Bare 直通:
        - D 模式恒绕过 (同 M 模式)
        - M 模式 MPRV=0, 或 MPRV=1 但 MPP=M -> 绕过
        - M 模式 MPRV=1 且 MPP=S/U -> 返回 MPP (sbi_unpriv 系列 API 依赖)
        - 其余返回当前模式
    """
    if hart.mode == RiscvMode.D:
        return None
    if hart.mode == RiscvMode.M:
        mprv = (hart.mstatus_val >> 17) & 1
        mpp = (hart.mstatus_val >> 11) & 3
        if not mprv or mpp == RiscvMode.M.value:
            return None
        return mpp
    return hart.mode.value


def translate_addr(
    hart: HartWithRegs,
    va: int,
    *,
    is_write: bool = False,
    is_execute: bool = False,
) -> tuple[bool, int]:
    """完整地址翻译: TLB 查找 + 页表遍历 + SUM/MXR/U 权限检查.

    供 mem_read / mem_write / check_instruction_fetch 使用。

    MMIO 设备地址 (经 Bus.is_device_addr 判断) 不会插入 TLB,
    因为设备寄存器读写可能有副作用, 不能被缓存.

    Returns:
        (success, pa) — success=False 表示翻译失败或权限检查失败 (调用方投递 trap).
    """
    mode = hart.mmu_mode
    if mode == SATP_MODE_BARE:
        return True, mask64(va)

    # M 模式 (无 MPRV 或 MPP=M) 与 D 模式绕过 MMU, 走 Bare 直通。
    # 例外 — MPRV (mstatus bit 17): 置位且 MPP=S/U 时, M-mode loads/stores
    # 按 MPP 指示的特权级翻译与检查 (sbi_unpriv 系列 API 依赖).
    eff_mode = _effective_mode(hart)
    if eff_mode is None:
        return True, mask64(va)

    # TLB 查找 (ASID-tagged: 非 Bare 模式按 satp.ASID 匹配)
    vpn = va >> 12
    tlb: TLB = hart.dtlb
    _asid = mask16(hart.satp_val >> 44)

    hit, ppn, perm = tlb.lookup(vpn, asid=_asid)
    if hit:
        if not _check_pte_perm(
            eff_mode, perm, hart.mstatus_val, is_write=is_write, is_execute=is_execute
        ):
            return False, 0
        offset = va & (PAGE_SIZE - 1)
        pa = mask64((ppn << 12) | offset)
        return True, pa

    # TLB miss — 执行页表遍历
    if hart._mem_read_phy is None:
        return False, 0

    ok, pa, perm = translate_va(va, hart.satp_val, hart._mem_read_phy)
    if not ok or not _check_pte_perm(
        eff_mode, perm, hart.mstatus_val, is_write=is_write, is_execute=is_execute
    ):
        return False, 0

    # MMIO 地址不可缓存 — 跳过 TLB 插入
    # 设备寄存器读写有副作用, 缓存会导致重复读写时绕过设备
    bus: Bus | None = hart._bus
    if bus is not None and bus.is_device_addr(pa):
        return True, pa

    # 将翻译结果插入 TLB 缓存 (存真实 PTE 权限, 命中路径方可复用 _check_pte_perm;
    # 标记当前 hart 的 mdid, 供 mfence.did 按域刷新)
    tlb.insert(
        va >> 12, pa >> 12,
        perm=perm, level=0,
        mdid=hart.mdid_val,
        asid=_asid
    )

    return True, pa


# ============================================================
#  虚拟内存读写
# ============================================================


def _translate_cross_page(
    hart: HartWithRegs,
    addr: int,
    size: int,
    *,
    is_write: bool,
) -> list[tuple[int, int]]:
    """跨页访问: 将 [addr, addr+size) 拆为两页并逐页翻译 + PMP + PMA 检查.

    供 mem_read / mem_write 共用 (二者仅 is_write 与陷态类型不同)。
    任一页翻译失败投递 Ld/StPageFault, PMP/PMA 拒绝投递 Ld/StAccessFault 并抛异常。

    Returns:
        [(pa1, size1), (pa2, size2)] 供调用方拼接读取或分段写入.
    """
    page_fault = TrapType.StPageFault if is_write else TrapType.LdPageFault
    access_fault = TrapType.StAccessFault if is_write else TrapType.LdAccessFault

    page_end = (addr & ~0xFFF) + 0x1000
    first_size = page_end - addr
    second_size = size - first_size

    bus: Bus | None = hart._bus
    pmp: Pmp = hart._pmp
    chunks: list[tuple[int, int]] = []
    for va, chunk_size in ((addr, first_size), (addr + first_size, second_size)):
        ok, pa = translate_addr(hart, va, is_write=is_write, is_execute=False)
        if not ok:
            deliver_trap(hart, page_fault, tval=va, is_interrupt=False)
            raise PageFault
        if not pmp.check(
            PmpAccessInfo(
                pa=pa,
                size=chunk_size,
                mode_val=hart.mode.value,
                mstatus_val=hart.mstatus_val,
                is_write=is_write,
                pmpsplit=hart.pmpsplit_val,
                mdid=hart.mdid_val,
            )
        ):
            deliver_trap(hart, access_fault, tval=addr, is_interrupt=False)
            raise AccessFault
        if bus is not None and not bus.is_valid_addr(pa):
            deliver_trap(hart, access_fault, tval=addr, is_interrupt=False)
            raise AccessFault
        chunks.append((pa, chunk_size))
    return chunks


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
        (pa1, first_size), (pa2, second_size) = _translate_cross_page(
            hart, addr, size, is_write=False
        )
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
        (pa1, first_size), (pa2, second_size) = _translate_cross_page(
            hart, addr, size, is_write=True
        )
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

    路径: VA -> PA (itlb 或页表遍历) -> PTE X/U 权限检查 -> PMP (is_execute=True).

    可能触发的陷态:
    - InstrPageFault: 页表翻译失败, 或 PTE.X=0, 或 S 模式取指于 U 页
    - InstrAccessFault: PMP 拒绝取指 (X=0 或未匹配)

    调用方在 ok=True 时使用返回的 pa 读取指令字节.
    """
    mode = hart.mmu_mode
    # RISC-V 规范: M 模式取指始终走物理地址 (MPRV 不影响取指); D 模式同 M 绕过 MMU.
    if mode == SATP_MODE_BARE or hart.mode in (RiscvMode.M, RiscvMode.D):
        pa = mask64(va)
    else:
        ok, pa, perm = _translate_instruction_addr(hart, va, hart.itlb)
        if not ok:
            return False, 0
        # PTE 权限: 取指要求 X=1; S 模式不得从 U=1 页取指
        # (RISC-V §4.3.2 基础规则, 与 SUM 无关 — 见 _check_pte_perm 的 is_execute 分支).
        if not _check_pte_perm(
            hart.mode.value, perm, hart.mstatus_val, is_write=False, is_execute=True
        ):
            deliver_trap(hart, TrapType.InstrPageFault, tval=va, is_interrupt=False)
            return False, 0

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
