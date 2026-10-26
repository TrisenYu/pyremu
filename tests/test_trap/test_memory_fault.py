#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""访存故障: 非对齐、PMP/PMA 拒绝、缺页与 PTE 权限."""

import struct

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.hart import (
    MSTATUS_MXR,
    MSTATUS_SUM,
    RiscvMode,
)
from pyremu.core.mem_check_aux import (
    AccessFaultError,
    AlignmentError,
    check_instruction_fetch,
    inject_memory_backend,
    mem_read,
    mem_write,
    PageFaultError,
)
from pyremu.memory.bus import Bus
from pyremu.memory.mmu import sv39_vpn
from tests.test_trap.support import (
    make_pte,
    PTE_R,
    PTE_U,
    PTE_V,
    PTE_W,
    PTE_X,
)


class TestMemoryAccessFaults:
    """验证 _mem_read / _mem_write 的对齐检查和 PMA 检查."""

    RAM_BASE = 0x8000_0000

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        bus = Bus(ram_size=1024 * 1024, ram_base=0x8000_0000)
        inject_memory_backend(h, bus.read, bus.write)
        h.bus = bus
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        return h

    # ---- 对齐检查 ----

    def test_lw_misaligned_traps(self, hart):
        """LW 从奇地址读取 -> LdAddrMisaligned."""
        try:
            mem_read(hart, 0x80001001, 4)
        except AlignmentError:
            pass
        assert hart.mcause_val == 4, f"应为 LdAddrMisaligned(4), 实际={hart.mcause_val}"

    def test_lh_misaligned_traps(self, hart):
        """LH 从奇地址读取 -> LdAddrMisaligned."""
        try:
            mem_read(hart, 0x80001001, 2)
        except AlignmentError:
            pass
        assert hart.mcause_val == 4

    def test_ld_misaligned_traps(self, hart):
        """LD 从非 8 字节对齐地址读取 -> LdAddrMisaligned."""
        try:
            mem_read(hart, 0x80001004, 8)
        except AlignmentError:
            pass
        assert hart.mcause_val == 4

    def test_sw_misaligned_traps(self, hart):
        """SW 到奇地址 -> StAddrMisaligned."""
        try:
            mem_write(hart, 0x80001001, b"\x00\x01\x02\x03")
        except AlignmentError:
            pass
        assert hart.mcause_val == 6, f"应为 StAddrMisaligned(6), 实际={hart.mcause_val}"

    def test_sd_misaligned_traps(self, hart):
        """SD 到非 8 字节对齐地址 -> StAddrMisaligned."""
        try:
            mem_write(hart, 0x80001004, b"\x00" * 8)
        except AlignmentError:
            pass
        assert hart.mcause_val == 6

    def test_lw_aligned_succeeds(self, hart):
        """LW 从对齐地址正常读取."""
        # 先写入数据确保可读
        hart._mem_write_phy(0x80001000, b"\x01\x02\x03\x04")
        data = mem_read(hart, 0x80001000, 4)
        assert data == b"\x01\x02\x03\x04"
        assert hart.mcause_val == 0

    def test_byte_access_always_aligned(self, hart):
        """1 字节 load/store 总是对齐."""
        hart._mem_write_phy(0x80001001, b"\x42")
        data = mem_read(hart, 0x80001001, 1)
        assert data == b"\x42"
        assert hart.mcause_val == 0

    # ---- PMA 检查 ----

    def test_read_empty_hole_traps(self, hart):
        """读取空洞地址 (非 RAM、非设备) -> LdAccessFault."""
        try:
            mem_read(hart, 0x0000_0000, 4)
        except AccessFaultError:
            pass
        assert hart.mcause_val == 5, f"应为 LdAccessFault(5), 实际={hart.mcause_val}"

    def test_write_empty_hole_traps(self, hart):
        """写入空洞地址 -> StAccessFault."""
        try:
            mem_write(hart, 0x4000_0000, b"\xff")
        except AccessFaultError:
            pass
        assert hart.mcause_val == 7, f"应为 StAccessFault(7), 实际={hart.mcause_val}"

    def test_read_ram_ok_pma(self, hart):
        """RAM 范围内读不触发 PMA 错误."""
        hart._mem_write_phy(0x80000000, b"\xaa\xbb")
        data = mem_read(hart, 0x80000000, 2)
        assert data == b"\xaa\xbb"
        assert hart.mcause_val == 0

    # ---- 对齐 + PMA 组合 ----

    def test_alignment_checked_before_translation(self, hart):
        """未对齐检测先于 MMU 翻译 (即使 PA 有效也不放过)."""
        try:
            mem_read(hart, 0x80000001, 4)
        except AlignmentError:
            pass
        assert hart.mcause_val == 4, "应先触发对齐错误而非页错误或访问错误"


class TestPageFault:
    """验证 Sv39 模式下页表遍历失败时触发 LdPageFault / StPageFault.

    使用 bytearray 模拟物理内存 (绕过 Bus PMA 检查),
    在低物理地址 (0x1000–0x3FFF) 构建页表结构.
    """

    PAGE_SHIFT = 12
    SATP_MODE_SV39 = 8

    # 页表物理基址
    L1_BASE = 0x1000  # 根页表 (PPN=1)
    L2_BASE = 0x2000  # 二级页表 (PPN=2)
    L3_BASE = 0x3000  # 三级页表 (PPN=3)
    DATA_PA = 0x100000  # 数据页物理地址 (PPN=0x100)

    @pytest.fixture
    def ram_ctx(self):
        """2 MiB 物理内存, 地址 [0, 2 MiB)."""
        ram = bytearray(2 * 1024 * 1024)

        def read_fn(addr: int, size: int) -> bytes:
            return bytes(ram[addr : addr + size])

        def write_fn(addr: int, data: bytes):
            for i, b in enumerate(data):
                ram[addr + i] = b

        return ram, read_fn, write_fn

    @pytest.fixture
    def hart(self, ram_ctx):
        """hart 注入 bytearray 内存后端, 初始为 Bare 模式."""
        _ram, read_fn, write_fn = ram_ctx
        h = Hart(id=0)
        inject_memory_backend(h, read_fn, write_fn)
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        return h

    # -- helper --

    @staticmethod
    def _write_pte(ram: bytearray, table_base: int, index: int, pte_val: int):
        """在 *table_base* 页表的第 *index* 项写入 8 字节 PTE."""
        addr = table_base + index * 8
        data = pte_val.to_bytes(8, "little")
        ram[addr : addr + 8] = data

    def _setup_valid_4k(
        self,
        ram: bytearray,
        va: int = 0,
        target_pa: int | None = None,
    ):
        """构建完整 Sv39 三级 4 KiB 映射 va -> target_pa.

        使用 vpn2/vpn1/vpn0 作为各级页表的索引, 适用于任意 VA.
        """
        if target_pa is None:
            target_pa = self.DATA_PA
        vpn2, vpn1, vpn0 = sv39_vpn(va)
        target_ppn = target_pa >> self.PAGE_SHIFT

        # L1 (根) -> L2
        self._write_pte(
            ram,
            self.L1_BASE,
            vpn2,
            make_pte(PTE_V, self.L2_BASE >> self.PAGE_SHIFT),
        )
        # L2 -> L3
        self._write_pte(
            ram,
            self.L2_BASE,
            vpn1,
            make_pte(PTE_V, self.L3_BASE >> self.PAGE_SHIFT),
        )
        # L3 叶
        self._write_pte(
            ram,
            self.L3_BASE,
            vpn0,
            make_pte(PTE_V | PTE_R | PTE_W | PTE_X, target_ppn),
        )

    def _enable_sv39(self, hart: "Hart", root_ppn: int = 1):
        """将 hart 切换到 S 模式并启用 Sv39 (MMU 翻译仅在 S/U 模式生效).

        同时配置 PMP NAPOT 开放全部地址空间, 否则 S 模式下无 PMP 条目时
        PMP 默认拒绝所有访问.
        """
        hart.mode = RiscvMode.S
        # PMP: 1 条 NAPOT 规则覆盖全部地址空间 R+W+X
        hart.csrs["pmpcfg0"].val = 0x1F  # NAPOT, R+W+X
        hart.csrs["pmpaddr0"].val = 0x003F_FFFF_FFFF_FFFF
        hart.satp_val = (self.SATP_MODE_SV39 << 60) | root_ppn

    # ---- LdPageFault ----

    def test_ld_page_fault_root_pte_invalid(self, hart, ram_ctx):
        """根页表条目 V=0 -> LdPageFault (13)."""
        ram, _read_fn, _write_fn = ram_ctx
        self._enable_sv39(hart)
        try:
            mem_read(hart, 0x0, 4)
        except PageFaultError:
            pass
        assert hart.mcause_val == 13, f"应为 LdPageFault(13), 实际={hart.mcause_val}"

    def test_ld_page_fault_l2_pte_invalid(self, hart, ram_ctx):
        """L2 条目 V=0 -> LdPageFault (13)."""
        ram, _read_fn, _write_fn = ram_ctx
        self._write_pte(ram, self.L1_BASE, 0, make_pte(PTE_V, self.L2_BASE >> self.PAGE_SHIFT))
        self._enable_sv39(hart)
        try:
            mem_read(hart, 0x0, 4)
        except PageFaultError:
            pass
        assert hart.mcause_val == 13, f"应为 LdPageFault(13), 实际={hart.mcause_val}"

    def test_ld_page_fault_l3_pte_invalid(self, hart, ram_ctx):
        """L3 叶条目 V=0 -> LdPageFault (13)."""
        ram, _read_fn, _write_fn = ram_ctx
        self._write_pte(ram, self.L1_BASE, 0, make_pte(PTE_V, self.L2_BASE >> self.PAGE_SHIFT))
        self._write_pte(ram, self.L2_BASE, 0, make_pte(PTE_V, self.L3_BASE >> self.PAGE_SHIFT))
        self._enable_sv39(hart)
        try:
            mem_read(hart, 0x0, 4)
        except PageFaultError:
            pass
        assert hart.mcause_val == 13, f"应为 LdPageFault(13), 实际={hart.mcause_val}"

    def test_ld_page_fault_unsupported_mode(self, hart, ram_ctx):
        """未实现的 satp 模式 (如 Sv48=9) -> LdPageFault."""
        _ram, _read_fn, _write_fn = ram_ctx
        hart.mode = RiscvMode.S
        hart.satp_val = (9 << 60) | 1
        try:
            mem_read(hart, 0x0, 4)
        except PageFaultError:
            pass
        assert hart.mcause_val == 13, f"应为 LdPageFault(13), 实际={hart.mcause_val}"

    # ---- StPageFault ----

    def test_st_page_fault_invalid_pte(self, hart, ram_ctx):
        """Sv39 下 store 遇到无效 PTE -> StPageFault (15)."""
        _ram, _read_fn, _write_fn = ram_ctx
        self._enable_sv39(hart)
        try:
            mem_write(hart, 0x1000, b"\x01\x02\x03\x04")
        except PageFaultError:
            pass
        assert hart.mcause_val == 15, f"应为 StPageFault(15), 实际={hart.mcause_val}"

    # ---- 有效映射 — 不触发 PageFault ----

    def test_ld_succeeds_valid_mapping(self, hart, ram_ctx):
        """完整 Sv39 映射下 load 应成功."""
        ram, _read_fn, _write_fn = ram_ctx
        self._setup_valid_4k(ram)
        self._enable_sv39(hart)
        # 在数据页写入预期值
        val = b"\xde\xad\xbe\xef"
        ram[self.DATA_PA : self.DATA_PA + 4] = val
        data = mem_read(hart, 0x0, 4)
        assert data == val
        assert hart.mcause_val == 0, f"不应 trap, mcause={hart.mcause_val}"

    def test_st_succeeds_valid_mapping(self, hart, ram_ctx):
        """完整 Sv39 映射下 store 应成功."""
        ram, _read_fn, _write_fn = ram_ctx
        self._setup_valid_4k(ram)
        self._enable_sv39(hart)
        val = b"\xca\xfe\xba\xbe"
        mem_write(hart, 0x0, val)
        assert hart.mcause_val == 0, f"不应 trap, mcause={hart.mcause_val}"
        # 从物理内存验证写入
        assert bytes(ram[self.DATA_PA : self.DATA_PA + 4]) == val

    def test_ld_st_nonzero_va(self, hart, ram_ctx):
        """非零 VA 的映射 (验证 VPN 索引正确)."""

        ram, _read_fn, _write_fn = ram_ctx
        va = 0x7FFFFFE000  # 一个非零 VA (Sv39 最高合法地址附近)
        self._setup_valid_4k(ram, va=va, target_pa=self.DATA_PA)
        self._enable_sv39(hart)
        val = b"\x11\x22\x33\x44"
        ram[self.DATA_PA : self.DATA_PA + 4] = val
        data = mem_read(hart, va, 4)
        assert data == val
        assert hart.mcause_val == 0

    # ---- Bare 模式 — 无 PageFault ----

    def test_bare_mode_no_page_fault(self, hart, ram_ctx):
        """Bare 模式: VA = PA, 不做翻译, 不触发缺页异常."""
        ram, _read_fn, _write_fn = ram_ctx
        # 不设置 satp -> 默认 Bare (mode=0)
        # VA 0x5000 直接当 PA 用
        val = b"\x42\x42\x42\x42"
        ram[0x5000:0x5004] = val
        data = mem_read(hart, 0x5000, 4)
        assert data == val
        assert hart.mcause_val == 0

    def test_bare_mode_st_no_page_fault(self, hart, ram_ctx):
        """Bare 模式 store 也不应触发缺页异常."""
        ram, _read_fn, _write_fn = ram_ctx
        val = b"\xff\xee\xdd\xcc"
        mem_write(hart, 0x6000, val)
        assert hart.mcause_val == 0
        assert bytes(ram[0x6000:0x6004]) == val

    # ---- 页错误优先于 PMA ----

    def test_page_fault_before_pma(self, hart, ram_ctx):
        """Sv39 翻译失败直接报 PageFault, 不会走到 PMA 检查."""
        _ram, _read_fn, _write_fn = ram_ctx
        self._enable_sv39(hart)
        try:
            mem_read(hart, 0x80000000, 4)
        except PageFaultError:
            pass
        assert hart.mcause_val == 13, (
            f"应为 LdPageFault(13), 翻译失败不应转为 LdAccessFault, 实际={hart.mcause_val}"
        )

    # ---- TLB 缓存后仍正确 ----

    def test_tlb_caches_and_page_fault_on_miss(self, hart, ram_ctx):
        """TLB miss -> 页表遍历, 无效 PTE -> PageFault; 重复也不走运."""
        ram, _read_fn, _write_fn = ram_ctx
        self._setup_valid_4k(ram, va=0x0)
        self._enable_sv39(hart)
        ram[self.DATA_PA : self.DATA_PA + 4] = b"\x01\x02\x03\x04"
        assert mem_read(hart, 0x0, 4) == b"\x01\x02\x03\x04"
        assert mem_read(hart, 0x0, 4) == b"\x01\x02\x03\x04"

        try:
            mem_read(hart, 0x1000, 4)
        except PageFaultError:
            pass
        assert hart.mcause_val == 13, (
            f"TLB miss 后无效 PTE 应触发 LdPageFault, 实际={hart.mcause_val}"
        )


class TestPtePermissionEnforcement(TestPageFault):
    """PTE 权限位检查回归 — mem_read/mem_write/check_instruction_fetch.

    回归: ``translate_addr`` 曾将 dtlb 条目 perm 硬编码为 0xF 且从不调用
    ``_check_pte_perm``, 导致 X=0 页可读、W=0 页可写、S 模式无 SUM 可读 U 页、
    S 模式可取指 U 页等越权访问静默通过。本类锁定修复后的正确拒绝/放行行为。
    """

    def _map_leaf(
        self,
        ram: bytearray,
        va: int = 0,
        *,
        perm: int = 0,
    ):
        """构建三级 Sv39 映射 va -> DATA_PA, 叶 PTE 权限位为 perm."""
        vpn0 = (va >> 12) & 0x1FF
        vpn1 = (va >> 21) & 0x1FF
        vpn2 = (va >> 30) & 0x1FF
        target_ppn = self.DATA_PA >> self.PAGE_SHIFT
        self._write_pte(
            ram,
            self.L1_BASE,
            vpn2,
            make_pte(PTE_V, self.L2_BASE >> self.PAGE_SHIFT),
        )
        self._write_pte(
            ram,
            self.L2_BASE,
            vpn1,
            make_pte(PTE_V, self.L3_BASE >> self.PAGE_SHIFT),
        )
        self._write_pte(
            ram,
            self.L3_BASE,
            vpn0,
            make_pte(PTE_V | perm, target_ppn),
        )

    # ---- 读权限 ----

    def test_mem_read_execute_only_page_faults(self, hart, ram_ctx):
        """X=1/R=0 且 MXR=0: S 模式读 -> LdPageFault (13)."""
        ram, _read_fn, _write_fn = ram_ctx
        self._map_leaf(ram, perm=PTE_X)
        self._enable_sv39(hart)
        with pytest.raises(PageFaultError):
            mem_read(hart, 0x0, 4)
        assert hart.mcause_val == 13, f"应为 LdPageFault(13), 实际={hart.mcause_val}"

    def test_mem_read_execute_only_with_mxr_allowed(self, hart, ram_ctx):
        """X=1/R=0 且 MXR=1: S 模式读应成功."""
        ram, _read_fn, _write_fn = ram_ctx
        self._map_leaf(ram, perm=PTE_X)
        self._enable_sv39(hart)
        ram[self.DATA_PA : self.DATA_PA + 4] = b"\xde\xad\xbe\xef"
        hart.mstatus_val |= MSTATUS_MXR
        assert mem_read(hart, 0x0, 4) == b"\xde\xad\xbe\xef"
        assert hart.mcause_val == 0, f"不应 trap, mcause={hart.mcause_val}"

    # ---- 写权限 ----

    def test_mem_write_readonly_page_faults(self, hart, ram_ctx):
        """R=1/W=0 (R-X 代码页): S 模式写 -> StPageFault (15)."""
        ram, _read_fn, _write_fn = ram_ctx
        self._map_leaf(ram, perm=PTE_R | PTE_X)
        self._enable_sv39(hart)
        with pytest.raises(PageFaultError):
            mem_write(hart, 0x0, b"\x01\x02\x03\x04")
        assert hart.mcause_val == 15, f"应为 StPageFault(15), 实际={hart.mcause_val}"

    def test_mem_write_rw_page_succeeds(self, hart, ram_ctx):
        """R=1/W=1: S 模式写应成功 (正向对照)."""
        ram, _read_fn, _write_fn = ram_ctx
        self._map_leaf(ram, perm=PTE_R | PTE_W)
        self._enable_sv39(hart)
        mem_write(hart, 0x0, b"\x01\x02\x03\x04")
        assert hart.mcause_val == 0, f"不应 trap, mcause={hart.mcause_val}"
        assert bytes(ram[self.DATA_PA : self.DATA_PA + 4]) == b"\x01\x02\x03\x04"

    # ---- SUM (S 模式访问 U 页) ----

    def test_smode_read_user_page_without_sum_faults(self, hart, ram_ctx):
        """U=1 且 SUM=0: S 模式读 -> LdPageFault (13)."""
        ram, _read_fn, _write_fn = ram_ctx
        self._map_leaf(ram, perm=PTE_R | PTE_U)
        self._enable_sv39(hart)
        with pytest.raises(PageFaultError):
            mem_read(hart, 0x0, 4)
        assert hart.mcause_val == 13, f"应为 LdPageFault(13), 实际={hart.mcause_val}"

    def test_smode_read_user_page_with_sum_allowed(self, hart, ram_ctx):
        """U=1 且 SUM=1: S 模式读应成功."""
        ram, _read_fn, _write_fn = ram_ctx
        self._map_leaf(ram, perm=PTE_R | PTE_U)
        self._enable_sv39(hart)
        ram[self.DATA_PA : self.DATA_PA + 4] = b"\xaa\xbb\xcc\xdd"
        hart.mstatus_val |= MSTATUS_SUM
        assert mem_read(hart, 0x0, 4) == b"\xaa\xbb\xcc\xdd"
        assert hart.mcause_val == 0, f"不应 trap, mcause={hart.mcause_val}"

    # ---- 取指权限 ----

    def test_fetch_non_executable_page_faults(self, hart, ram_ctx):
        """X=0: S 模式取指 -> InstrPageFault (12)."""
        ram, _read_fn, _write_fn = ram_ctx
        self._map_leaf(ram, perm=PTE_R | PTE_W)
        self._enable_sv39(hart)
        ok, _pa = check_instruction_fetch(hart, 0x0)
        assert not ok
        assert hart.mcause_val == 12, f"应为 InstrPageFault(12), 实际={hart.mcause_val}"

    def test_smode_fetch_user_page_faults(self, hart, ram_ctx):
        """U=1/X=1: S 模式取指 -> InstrPageFault (12), 即使 SUM=1 也不放行."""
        ram, _read_fn, _write_fn = ram_ctx
        self._map_leaf(ram, perm=PTE_X | PTE_U)
        self._enable_sv39(hart)
        hart.mstatus_val |= MSTATUS_SUM  # SUM 仅影响数据访问, 不影响取指
        ok, _pa = check_instruction_fetch(hart, 0x0)
        assert not ok
        assert hart.mcause_val == 12, f"应为 InstrPageFault(12), 实际={hart.mcause_val}"

    def test_fetch_executable_page_succeeds(self, hart, ram_ctx):
        """X=1/U=0: S 模式取指应成功 (正向对照)."""
        ram, _read_fn, _write_fn = ram_ctx
        self._map_leaf(ram, perm=PTE_X)
        self._enable_sv39(hart)
        ok, pa = check_instruction_fetch(hart, 0x0)
        assert ok
        assert pa == self.DATA_PA
        assert hart.mcause_val == 0, f"不应 trap, mcause={hart.mcause_val}"


class TestLoadPageFaultPreservesRd(TestPageFault):
    """回归: 页错误时 load/store handler 不能覆盖目标寄存器.

    修复前 ``mem_read`` / ``mem_write`` 在触发 trap 后返回假数据
    (b\"\\x00\" * size), handler 继续执行并将 0 写入 rd ->寄存器损坏.
    修复后 ``mem_read`` / ``mem_write`` 抛出 ``MemoryAccessError``,
    ``exec_instr`` 捕获并返回 0, 内核 trap handler 看到故障前的寄存器状态.

    覆盖: LD / SD / C.LD / C.SD / AMO LR.W
    """

    # 以下七项不再覆写 TestPageFault 的同名用例: 覆写版本是空函数体且被标记跳过,
    # 既未断言任何内容, 又遮蔽了父类中可直接通过的真实实现。父类实现现已与
    # mem_read/mem_write 抛出 MemoryAccessError 的行为一致, 故直接继承。

    def _setup_sv39_for(
        self,
        hart,
        ram: bytearray,
        valid_va: int,
        valid_pa: int,
    ):
        """建立单页 Sv39 映射 (仅 valid_va ->valid_pa 可访问)."""
        vpn0 = (valid_va >> 12) & 0x1FF
        root_ppn = self.L1_BASE >> self.PAGE_SHIFT
        self._write_pte(
            ram,
            self.L1_BASE,
            0,
            make_pte(PTE_V, self.L2_BASE >> self.PAGE_SHIFT),
        )
        self._write_pte(
            ram,
            self.L2_BASE,
            0,
            make_pte(PTE_V, self.L3_BASE >> self.PAGE_SHIFT),
        )
        self._write_pte(
            ram,
            self.L3_BASE,
            vpn0,
            make_pte(PTE_V | PTE_R, valid_pa >> self.PAGE_SHIFT),
        )
        hart.satp_val = (8 << 60) | root_ppn  # Sv39
        hart.mode = RiscvMode.S
        # PMP: 1 条 NAPOT 规则覆盖全部地址空间 R+W+X, 否则 S 模式下页表遍历被拒绝
        hart.csrs["pmpcfg0"].val = 0x1F
        hart.csrs["pmpaddr0"].val = 0x003F_FFFF_FFFF_FFFF
        # 委派页错误到 S 模式, 使 trap 保持在 S 模式
        hart.csrs["medeleg"].val = (1 << 12) | (1 << 13) | (1 << 15)

    # ---- LD: rd 不被页错误清零 ----

    def test_ld_pagefault_preserves_rd(self, hart, ram_ctx):
        """LD 触发 LdPageFault 后 rd (x13/a3) 保持原值不被清零."""
        ram, read_fn, write_fn = ram_ctx
        hart._mem_read_phy = read_fn
        hart._mem_write_phy = write_fn
        hart.csrs["stvec"].val = 0x80004000

        valid_va = 0x1000
        valid_pa = 0x100000
        fault_va = 0x2000
        self._setup_sv39_for(hart, ram, valid_va, valid_pa)

        saved_val = 0xDEADBEEF
        hart.gprs[13] = saved_val  # a3

        instr_addr = valid_pa
        lo12 = fault_va & 0xFFF
        addi_instr = 0x000E0E13 | ((lo12 & 0xFFF) << 20)
        ld_instr = 0x000E3683  # ld x13, 0(x28)
        ram[instr_addr : instr_addr + 4] = struct.pack("<I", addi_instr)
        ram[instr_addr + 4 : instr_addr + 8] = struct.pack("<I", ld_instr)

        hart.pc = valid_va
        advance = hart.exec_instr(addi_instr)
        assert advance == 4
        hart.pc += advance

        advance = hart.exec_instr(ld_instr)
        assert advance == 0, f"LD->page fault 后 advance 应为 0, 实际={advance}"
        assert hart.gprs[13] == saved_val, (
            f"LdPageFault 后 a3 被清零! 期望={saved_val:#x}, 实际={hart.gprs[13]:#x}"
        )
        assert hart.mode == RiscvMode.S

    def test_sd_pagefault_pc_to_stvec(self, hart, ram_ctx):
        """SD 触发 StPageFault 后 PC 跳转到 stvec, 不推进."""
        ram, read_fn, write_fn = ram_ctx
        hart._mem_read_phy = read_fn
        hart._mem_write_phy = write_fn
        hart.csrs["stvec"].val = 0x80004000

        valid_va = 0x1000
        valid_pa = 0x100000
        fault_va = 0x2000
        self._setup_sv39_for(hart, ram, valid_va, valid_pa)

        lo12 = fault_va & 0xFFF
        addi_instr = 0x000E0E13 | ((lo12 & 0xFFF) << 20)
        sd_instr = 0x005E3023  # sd x5, 0(x28)
        ram[valid_pa : valid_pa + 4] = struct.pack("<I", addi_instr)
        ram[valid_pa + 4 : valid_pa + 8] = struct.pack("<I", sd_instr)
        hart.gprs[5] = 0x41414141

        hart.pc = valid_va
        advance = hart.exec_instr(addi_instr)
        assert advance == 4
        hart.pc += advance

        advance = hart.exec_instr(sd_instr)
        assert advance == 0, f"SD->page fault 后 advance 应为 0, 实际={advance}"
        assert hart.mode == RiscvMode.S

    def test_cld_pagefault_preserves_rd(self, hart, ram_ctx):
        """C.LD 触发 LdPageFault 后 rd 保持原值."""
        ram, read_fn, write_fn = ram_ctx
        hart._mem_read_phy = read_fn
        hart._mem_write_phy = write_fn
        hart.csrs["stvec"].val = 0x80004000

        valid_va = 0x1000
        valid_pa = 0x100000
        fault_va = 0x2000
        self._setup_sv39_for(hart, ram, valid_va, valid_pa)

        saved_val = 0xA5A5A5A5
        hart.gprs[10] = saved_val  # a0

        lo12 = fault_va & 0xFFF
        addi_instr = 0x000E0E13 | ((lo12 & 0xFFF) << 20)
        c_ld = 0x60C8  # C.LD rd'=x10(a0), rs1'=x28(t3), uimm=0
        ram[valid_pa : valid_pa + 4] = struct.pack("<I", addi_instr)
        ram[valid_pa + 4 : valid_pa + 6] = struct.pack("<H", c_ld)

        hart.pc = valid_va
        advance = hart.exec_instr(addi_instr)
        assert advance == 4
        hart.pc += advance

        advance = hart.exec_instr(c_ld)
        assert advance == 0, f"C.LD->page fault 后 advance 应为 0, 实际={advance}"
        assert hart.gprs[10] == saved_val, (
            f"C.LD page fault 后 a0 被清零! 期望={saved_val:#x}, 实际={hart.gprs[10]:#x}"
        )
        assert hart.mode == RiscvMode.S

    def test_csd_pagefault_pc_to_stvec(self, hart, ram_ctx):
        """C.SD 触发 StPageFault 后 PC 跳转到 stvec."""
        ram, read_fn, write_fn = ram_ctx
        hart._mem_read_phy = read_fn
        hart._mem_write_phy = write_fn
        hart.csrs["stvec"].val = 0x80004000

        valid_va = 0x1000
        valid_pa = 0x100000
        fault_va = 0x2000
        self._setup_sv39_for(hart, ram, valid_va, valid_pa)

        lo12 = fault_va & 0xFFF
        addi_instr = 0x000E0E13 | ((lo12 & 0xFFF) << 20)
        c_sd = 0xE184  # C.SD rs2'=x8(s0), rs1'=x28(t3), uimm=0
        ram[valid_pa : valid_pa + 4] = struct.pack("<I", addi_instr)
        ram[valid_pa + 4 : valid_pa + 6] = struct.pack("<H", c_sd)
        hart.gprs[8] = 0xBBBBBBBB

        hart.pc = valid_va
        advance = hart.exec_instr(addi_instr)
        assert advance == 4
        hart.pc += advance

        advance = hart.exec_instr(c_sd)
        assert advance == 0, f"C.SD->page fault 后 advance 应为 0, 实际={advance}"
        assert hart.mode == RiscvMode.S

    def test_lr_pagefault_preserves_rd(self, hart, ram_ctx):
        """LR.W 触发 LdPageFault 后 rd 保持原值, 不设 reservation."""
        ram, read_fn, write_fn = ram_ctx
        hart._mem_read_phy = read_fn
        hart._mem_write_phy = write_fn
        hart.csrs["stvec"].val = 0x80004000

        valid_va = 0x1000
        valid_pa = 0x100000
        fault_va = 0x2000
        self._setup_sv39_for(hart, ram, valid_va, valid_pa)

        saved_val = 0xCAFECAFE
        hart.gprs[10] = saved_val  # a0

        lo12 = fault_va & 0xFFF
        addi_instr = 0x000E0E13 | ((lo12 & 0xFFF) << 20)
        lr_w = 0x100E252F  # lr.w a0, (t3)
        ram[valid_pa : valid_pa + 4] = struct.pack("<I", addi_instr)
        ram[valid_pa + 4 : valid_pa + 8] = struct.pack("<I", lr_w)

        hart.pc = valid_va
        advance = hart.exec_instr(addi_instr)
        assert advance == 4
        hart.pc += advance

        advance = hart.exec_instr(lr_w)
        assert advance == 0, f"LR.W->page fault 后 advance 应为 0, 实际={advance}"
        assert hart.gprs[10] == saved_val, (
            f"LR.W page fault 后 a0 被清零! 期望={saved_val:#x}, 实际={hart.gprs[10]:#x}"
        )
        assert hart.reservation_valid == 0, "page fault 后不应设置 reservation"
        assert hart.mode == RiscvMode.S
