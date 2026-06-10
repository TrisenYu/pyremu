#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""Memory access mixin: TLB-based VA→PA translation, CSR checking, PMP/PMA checks.

Provides MemoryAccessor — a mixin class that adds _mem_read / _mem_write /
_translate_full / _check_csr to Hart when inherited alongside HartWithRegs.
"""

from pyremu.core.registers import CsrAccessError, check_csr_access
from pyremu.core.trap import TrapType
from pyremu.memory.mmu import PAGE_SIZE, SATP_MODE_BARE, translate_va

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
#  MemoryAccessor mixin
# ============================================================


class MemoryAccessor:
    """Mixin: virtual memory read/write with TLBPMPPMA checks.

    Requires the host class to provide:
    - self.pc, self.mode, self.mmu_mode
    - self.satp_val, self.mstatus_val
    - self._pmp (Pmp instance)
    - self._bus (Bus or None)
    - self.dtlb (TLB instance)
    - self._mem_read_phy / self._mem_write_phy callbacks
    - self._take_trap(cause, tval, is_interrupt) method
    - self._pmp_entries (int)
    """

    # ----------------------------------------------------------
    #  Memory backend injection
    # ----------------------------------------------------------

    def set_memory_backend(
        self,
        read_fn,  # (addr: int, size: int) -> bytes
        write_fn,  # (addr: int, data: bytes) -> None
    ) -> None:
        """注入物理内存后端.

        调用此方法后, _mem_read 和 _mem_write 将使用注入的回调
        来读写物理内存。未注入时内存访问会抛出 NotImplementedError。
        """
        self._mem_read_phy = read_fn
        self._mem_write_phy = write_fn

    # ----------------------------------------------------------
    #  地址翻译 (VA → PA, 经 TLB 缓存)
    # ----------------------------------------------------------

    def _translate_full(
        self,
        va: int,
    ) -> tuple:
        """完整地址翻译: TLB 查找 + 页表遍历.

        供 _mem_read / _mem_write 使用的内部入口.
        Returns: (success: bool, pa: int)

        MMIO 设备地址 (经 Bus.is_device_addr 判断) 不会插入 TLB,
        因为设备寄存器读写可能有副作用, 不能被缓存.
        """
        mode = self.mmu_mode
        if mode == SATP_MODE_BARE:
            return True, va & 0xFFFF_FFFF_FFFF_FFFF

        # TLB 查找
        vpn = va >> 12
        tlb = self.dtlb
        hit, ppn, perm = tlb.lookup(vpn)
        if hit:
            offset = va & (PAGE_SIZE - 1)
            pa = (ppn << 12 | offset) & 0xFFFF_FFFF_FFFF_FFFF
            return True, pa

        # TLB miss — 执行页表遍历
        if self._mem_read_phy is None:
            return False, 0

        satp = self.satp_val
        ok, pa = translate_va(va, satp, self._mem_read_phy)
        if not ok:
            return False, 0

        # MMIO 地址不可缓存 — 跳过 TLB 插入
        # 设备寄存器读写有副作用, 缓存会导致重复读写时绕过设备
        if self._bus is not None and self._bus.is_device_addr(pa):
            return True, pa

        # 将翻译结果插入 TLB 缓存
        new_vpn = va >> 12
        new_ppn = pa >> 12
        tlb.insert(new_vpn, new_ppn, perm=0xF, level=0)

        return True, pa

    # ----------------------------------------------------------
    #  CSR 访问检查
    # ----------------------------------------------------------

    def _check_csr(
        self,
        csr_addr: int,
        is_write: bool = False,
    ) -> None:
        """验证 CSR 访问的合法性.

        检查顺序:
        1. PMP CSR 地址范围 (越界 → CsrAccessError → IllInstr)
        2. 特权级权限 (低特权访问高特权 CSR → CsrAccessError → IllInstr)
        3. 只读检查 (写入只读 CSR → CsrAccessError → IllInstr)

        由 handle_sys 的各 CSR 指令 handler 在访问前调用.
        """
        # PMP CSR 范围检查
        if not _pmp_csr_valid(csr_addr, self._pmp_entries):
            raise CsrAccessError(csr_addr, "pmp out of range")

        # 特权级检查
        check_csr_access(csr_addr, self.mode.value, is_write)

    # ----------------------------------------------------------
    #  Memory access (with TLB-based address translation)
    # ----------------------------------------------------------

    def _mem_read(
        self,
        addr: int,
        size: int,
    ) -> bytes:
        """从虚拟地址 *addr* 读取 *size* 字节.

        经过路径: 对齐检查 → VA→PA (TLB/页表) → PMA 检查 → 物理内存后端.

        可能触发的陷态:
        - LdAddrMisaligned: 地址未对齐 (size>1 且 addr 不满足对齐要求)
        - LdPageFault: 页表翻译失败
        - LdAccessFault: PMA 违例 (PA 不落在有效物理区域)
        """
        if self._mem_read_phy is None:
            raise NotImplementedError(
                f"Memory read @ {addr:#018x} ({size} B): no memory backend attached"
            )

        # 对齐检查 (RISC-V Privileged Spec §3.6.1)
        if size > 1 and (addr & (size - 1)) != 0:
            self._take_trap(
                TrapType.LdAddrMisaligned, tval=addr, is_interrupt=False
            )
            return b"\x00" * size

        # 地址翻译
        ok, pa = self._translate_full(addr)
        if not ok:
            self._take_trap(TrapType.LdPageFault, tval=addr, is_interrupt=False)
            return b"\x00" * size

        # PMP 检查 — 物理内存保护 (对 M 模式且 MPRV=0 自动放行)
        if not self._pmp.check(
            pa, size, self.mode.value, self.mstatus_val, is_write=False,
        ):
            self._take_trap(
                TrapType.LdAccessFault, tval=addr, is_interrupt=False
            )
            return b"\x00" * size

        # PMA 检查 — 物理地址必须落在有效区域 (RAM 或已注册设备)
        if self._bus is not None and not self._bus.is_valid_addr(pa):
            self._take_trap(
                TrapType.LdAccessFault, tval=addr, is_interrupt=False
            )
            return b"\x00" * size

        return self._mem_read_phy(pa, size)

    def _mem_write(
        self,
        addr: int,
        data: bytes,
    ) -> None:
        """向虚拟地址 *addr* 写入 *data*.

        经过路径: 对齐检查 → VA→PA (TLB/页表) → PMP → PMA → 物理内存后端.

        可能触发的陷态:
        - StAddrMisaligned: 地址未对齐
        - StPageFault: 页表翻译失败
        - StAccessFault: PMP 拒绝 或 PMA 违例
        """
        if self._mem_write_phy is None:
            raise NotImplementedError(
                f"Memory write @ {addr:#018x} ({len(data)} B): no memory backend attached"
            )

        size = len(data)

        # 对齐检查
        if size > 1 and (addr & (size - 1)) != 0:
            self._take_trap(
                TrapType.StAddrMisaligned, tval=addr, is_interrupt=False
            )
            return

        # 地址翻译
        ok, pa = self._translate_full(addr)
        if not ok:
            self._take_trap(TrapType.StPageFault, tval=addr, is_interrupt=False)
            return

        # PMP 检查
        if not self._pmp.check(
            pa, size, self.mode.value, self.mstatus_val, is_write=True,
        ):
            self._take_trap(
                TrapType.StAccessFault, tval=addr, is_interrupt=False
            )
            return

        # PMA 检查
        if self._bus is not None and not self._bus.is_valid_addr(pa):
            self._take_trap(
                TrapType.StAccessFault, tval=addr, is_interrupt=False
            )
            return

        self._mem_write_phy(pa, data)
