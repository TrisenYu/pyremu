#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""mem_check_aux 的分支行为.

覆盖 counteren 计数器门控、PTE 权限判定、有效特权级的计算、页表遍历被 PMP 拒绝时
投递的故障类型、TLB 命中后的权限复核、跨页访问的分段与取指翻译的失败路径.
"""

import pytest

from pyremu.core.hart import HartWithRegs, MSTATUS_MXR, MSTATUS_SUM, RiscvMode
from pyremu.core.mem_check_aux import (
    AccessFaultError,
    check_instruction_fetch,
    inject_memory_backend,
    mem_read,
    mem_write,
    PageFaultError,
    translate_addr,
    validate_csr,
)
from pyremu.core.registers import CsrAccessError
from pyremu.memory.bus import Bus
from pyremu.memory.mmu import (
    PTE_A,
    PTE_D,
    PTE_R,
    PTE_U,
    PTE_V,
    PTE_W,
    PTE_X,
    SATP_MODE_SV39,
)

RAM_BASE = 0x8000_0000
RAM_SIZE = 4 * 1024 * 1024
# 数据页所在的虚拟地址: 与页表自身占用的物理页 (RAM_BASE + 0x1000 起) 不重叠
DATA_VA = RAM_BASE + 0x10_0000
DATA_VPN0 = (DATA_VA >> 12) & 0x1FF
MTVEC = 0x8000_0100

RWX = PTE_R | PTE_W | PTE_X
# pmpcfg0: A=NAPOT (0x18) 分别与 R、R|X 组合
PMP_ALLOW_ALL = 0x1F
PMP_READ_ONLY = 0x1D
PMP_NAPOT_ALL = 0x003F_FFFF_FFFF_FFFF


def _encode_ppn(ppn: int) -> int:
    """RISC-V 连续 PPN 编码: PTE[53:10] = PPN[43:0]."""
    val = (ppn & 0x3FF) << 10
    val |= ((ppn >> 10) & 0x1FF) << 20
    val |= ((ppn >> 19) & 0x1FFF_FFFF) << 29
    return val


def _write_pte(hart: HartWithRegs, pa: int, value: int) -> None:
    assert hart._mem_write_phy is not None
    hart._mem_write_phy(pa, value.to_bytes(8, "little"))


def _install_tables(
    hart: HartWithRegs,
    leaves: dict[int, tuple[int, int]],
) -> None:
    """为 DATA_VA 所在的 2 MiB 区间安装三级页表.

    *leaves* 为 {vpn0: (物理页基址, 权限标志)}, 只写出的叶 PTE 才可翻译.
    """
    root_pa = RAM_BASE + 0x1000
    l2_pa = RAM_BASE + 0x2000
    l3_pa = RAM_BASE + 0x3000
    root_index = (DATA_VA >> 30) & 0x1FF
    l2_index = (DATA_VA >> 21) & 0x1FF
    _write_pte(hart, root_pa + root_index * 8, PTE_V | _encode_ppn(l2_pa >> 12))
    _write_pte(hart, l2_pa + l2_index * 8, PTE_V | _encode_ppn(l3_pa >> 12))
    for vpn0, (pa, flags) in leaves.items():
        value = flags | PTE_V | PTE_A | PTE_D | _encode_ppn(pa >> 12)
        _write_pte(hart, l3_pa + vpn0 * 8, value)
    hart.satp_val = (SATP_MODE_SV39 << 60) | (root_pa >> 12)


def _identity_leaves(page_count: int, flags: int = RWX) -> dict[int, tuple[int, int]]:
    """从 DATA_VA 起 identity 映射 *page_count* 个 4 KiB 页."""
    return {DATA_VPN0 + i: (DATA_VA + i * 0x1000, flags) for i in range(page_count)}


def _make_hart(bus: Bus, mode: RiscvMode = RiscvMode.S, pmpcfg: int = PMP_ALLOW_ALL):
    hart = HartWithRegs(id=0)
    inject_memory_backend(hart, bus.read, bus.write)
    hart.bus = bus
    hart.mode = mode
    hart.pc = MTVEC
    hart.csrs["mtvec"].val = MTVEC
    hart.csrs["pmpcfg0"].val = pmpcfg
    hart.csrs["pmpaddr0"].val = PMP_NAPOT_ALL
    return hart


@pytest.fixture
def bus() -> Bus:
    return Bus(ram_size=RAM_SIZE, ram_base=RAM_BASE)


@pytest.fixture
def hart(bus: Bus) -> HartWithRegs:
    return _make_hart(bus)


# ============================================================
#  counteren 计数器门控
# ============================================================


class TestCounterEnable:
    def test_high_half_counter_uses_low_half_bit(self, hart):
        """0xC80-0xC9F 的高半计数器与低半共用 counteren 的同一位."""
        hart.mode = RiscvMode.S
        hart.csrs["mcounteren"].val = 1
        validate_csr(hart, 0xC80)

    def test_mmode_counter_read_ignores_counteren(self, hart):
        """M 模式读取计数器不受 mcounteren 约束."""
        hart.mode = RiscvMode.M
        hart.csrs["mcounteren"].val = 0
        validate_csr(hart, 0xC00)

    def test_umode_requires_both_counteren_bits(self, hart):
        """U 模式要求 mcounteren 与 scounteren 的同一位同时置位."""
        hart.mode = RiscvMode.U
        hart.csrs["mcounteren"].val = 1
        with pytest.raises(CsrAccessError):
            validate_csr(hart, 0xC00)

        hart.csrs["scounteren"].val = 1
        validate_csr(hart, 0xC00)


# ============================================================
#  PTE 权限判定
# ============================================================


class TestPtePermissions:
    def test_umode_rejects_page_without_u_bit(self, hart):
        _install_tables(hart, _identity_leaves(1))
        hart.mode = RiscvMode.U
        ok, _ = translate_addr(hart, DATA_VA)
        assert not ok

    def test_smode_fetch_from_user_page_faults(self, hart):
        """S 模式从 U=1 页取指一律失败, 与 SUM 无关."""
        _install_tables(hart, _identity_leaves(1, RWX | PTE_U))
        hart.mstatus_val = MSTATUS_SUM
        ok, _ = check_instruction_fetch(hart, DATA_VA)
        assert not ok
        assert hart.mcause_val == 12
        assert hart.mtval_val == DATA_VA

    def test_smode_data_access_to_user_page_follows_sum(self, hart):
        _install_tables(hart, _identity_leaves(1, RWX | PTE_U))
        hart.mstatus_val = 0
        assert not translate_addr(hart, DATA_VA)[0]

        hart.mstatus_val = MSTATUS_SUM
        assert translate_addr(hart, DATA_VA)[0]

    def test_mxr_allows_reading_execute_only_page(self, hart):
        _install_tables(hart, _identity_leaves(1, PTE_X))
        hart.mstatus_val = 0
        assert not translate_addr(hart, DATA_VA)[0]

        hart.mstatus_val = MSTATUS_MXR
        assert translate_addr(hart, DATA_VA)[0]

    def test_write_requires_r_and_w(self, hart):
        _install_tables(hart, _identity_leaves(1, PTE_R))
        assert not translate_addr(hart, DATA_VA, is_write=True)[0]


# ============================================================
#  有效特权级 (MPRV / D 模式)
# ============================================================


class TestEffectiveMode:
    def test_debug_mode_bypasses_translation(self, hart):
        _install_tables(hart, _identity_leaves(1))
        hart.mode = RiscvMode.D
        ok, pa = translate_addr(hart, DATA_VA)
        assert ok and pa == DATA_VA

    def test_mprv_with_supervisor_mpp_translates(self, hart):
        mapped_pa = RAM_BASE + 0x20_0000
        _install_tables(hart, {DATA_VPN0: (mapped_pa, RWX)})
        hart.mode = RiscvMode.M
        hart.mstatus_val = (1 << 17) | (RiscvMode.S.value << 11)
        ok, pa = translate_addr(hart, DATA_VA)
        assert ok and pa == mapped_pa

    def test_mprv_with_machine_mpp_bypasses_translation(self, hart):
        _install_tables(hart, _identity_leaves(1))
        hart.mode = RiscvMode.M
        hart.mstatus_val = (1 << 17) | (RiscvMode.M.value << 11)
        ok, pa = translate_addr(hart, DATA_VA)
        assert ok and pa == DATA_VA


# ============================================================
#  页表遍历被 PMP 拒绝
# ============================================================


class TestPageWalkDenied:
    def test_load_reports_load_access_fault(self, bus):
        hart = _make_hart(bus, pmpcfg=0)
        _install_tables(hart, _identity_leaves(1))
        with pytest.raises(AccessFaultError):
            mem_read(hart, DATA_VA, 8)
        assert hart.mcause_val == 5
        assert hart.mtval_val == DATA_VA

    def test_store_reports_store_access_fault(self, bus):
        hart = _make_hart(bus, pmpcfg=0)
        _install_tables(hart, _identity_leaves(1))
        with pytest.raises(AccessFaultError):
            mem_write(hart, DATA_VA, bytes(8))
        assert hart.mcause_val == 7
        assert hart.mtval_val == DATA_VA

    def test_fetch_reports_instruction_access_fault(self, bus):
        hart = _make_hart(bus, pmpcfg=0)
        _install_tables(hart, _identity_leaves(1))
        ok, pa = check_instruction_fetch(hart, DATA_VA)
        assert not ok and pa == 0
        assert hart.mcause_val == 1
        assert hart.mtval_val == DATA_VA


# ============================================================
#  TLB 命中与缺少物理内存后端
# ============================================================


class TestTranslateEdges:
    def test_tlb_hit_without_write_permission_fails(self, hart):
        """TLB 命中后按本次访问重新校验权限, 只读条目拒绝写入."""
        _install_tables(hart, _identity_leaves(1))
        hart.dtlb.insert(DATA_VA >> 12, DATA_VA >> 12, perm=PTE_R)
        ok, pa = translate_addr(hart, DATA_VA, is_write=True)
        assert not ok and pa == 0

    def test_debugger_translation_without_backend_fails(self):
        """调试器路径不做 PMP 检查, 但没有物理读回调时仍无法遍历页表."""
        hart = HartWithRegs(id=0)
        hart.mode = RiscvMode.S
        hart.satp_val = (SATP_MODE_SV39 << 60) | ((RAM_BASE + 0x1000) >> 12)
        ok, pa = translate_addr(hart, DATA_VA, is_hart_access=False)
        assert not ok and pa == 0

    def test_read_without_backend_raises_not_implemented(self):
        hart = HartWithRegs(id=0)
        with pytest.raises(NotImplementedError):
            mem_read(hart, RAM_BASE, 8)

    def test_write_without_backend_raises_not_implemented(self):
        hart = HartWithRegs(id=0)
        with pytest.raises(NotImplementedError):
            mem_write(hart, RAM_BASE, bytes(8))

    def test_fetch_without_backend_reports_page_fault(self):
        hart = HartWithRegs(id=0)
        hart.mode = RiscvMode.S
        hart.csrs["mtvec"].val = MTVEC
        hart.satp_val = (SATP_MODE_SV39 << 60) | ((RAM_BASE + 0x1000) >> 12)
        ok, pa = check_instruction_fetch(hart, DATA_VA)
        assert not ok and pa == 0
        assert hart.mcause_val == 12
        assert hart.mtval_val == DATA_VA

    def test_fetch_with_invalid_pte_reports_page_fault(self, bus):
        hart = _make_hart(bus)
        hart.satp_val = (SATP_MODE_SV39 << 60) | ((RAM_BASE + 0x1000) >> 12)
        ok, pa = check_instruction_fetch(hart, DATA_VA)
        assert not ok and pa == 0
        assert hart.mcause_val == 12
        assert hart.mtval_val == DATA_VA


# ============================================================
#  跨页访问
# ============================================================


class TestCrossPage:
    def test_read_over_two_pages_concatenates_halves(self, bus, hart):
        _install_tables(hart, _identity_leaves(2))
        payload = bytes((i * 7) & 0xFF for i in range(0x2000))
        assert bus.try_write(DATA_VA, payload)
        assert mem_read(hart, DATA_VA, 0x2000) == payload

    def test_write_over_two_pages_splits_halves(self, bus, hart):
        _install_tables(hart, _identity_leaves(2))
        payload = bytes((i * 11) & 0xFF for i in range(0x2000))
        mem_write(hart, DATA_VA, payload)
        assert bus.try_read(DATA_VA, 0x2000) == payload

    def test_load_across_unmapped_page_reports_page_fault(self, bus, hart):
        _install_tables(hart, _identity_leaves(1))
        with pytest.raises(PageFaultError):
            mem_read(hart, DATA_VA, 0x2000)
        assert hart.mcause_val == 13
        assert hart.mtval_val == DATA_VA + 0x1000

    def test_store_across_unmapped_page_reports_page_fault(self, bus, hart):
        _install_tables(hart, _identity_leaves(1))
        with pytest.raises(PageFaultError):
            mem_write(hart, DATA_VA, bytes(0x2000))
        assert hart.mcause_val == 15
        assert hart.mtval_val == DATA_VA + 0x1000

    def test_store_across_read_only_pmp_reports_access_fault(self, bus):
        """存取权限按请求方向判定: 只读条目允许页表遍历, 拒绝写入."""
        hart = _make_hart(bus, pmpcfg=PMP_READ_ONLY)
        _install_tables(hart, _identity_leaves(2))
        with pytest.raises(AccessFaultError):
            mem_write(hart, DATA_VA, bytes(0x2000))
        assert hart.mcause_val == 7
        assert hart.mtval_val == DATA_VA

    def test_second_page_outside_ram_reports_access_fault(self, bus, hart):
        leaves = _identity_leaves(2)
        leaves[DATA_VPN0 + 1] = (0x9000_0000, RWX)
        _install_tables(hart, leaves)
        with pytest.raises(AccessFaultError):
            mem_read(hart, DATA_VA, 0x2000)
        assert hart.mcause_val == 5
        assert hart.mtval_val == DATA_VA
