#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""MMU 测试: PTE 字段操作、Sv39 页表遍历、地址翻译."""

import pytest

from pyremu.memory.mmu import (
    PAGE_SHIFT,
    PTE,
    PTE_R,
    PTE_V,
    PTE_W,
    PTE_X,
    SATP_MODE_BARE,
    SATP_MODE_SV39,
    MemAccessMode,
    _sv39_vpn,
    translate_va,
)

# ============================================================
#  辅助: 构造简单的物理内存
# ============================================================


def _make_ram(size_bytes: int = 2 * 1024 * 1024) -> tuple:
    """创建一个 bytearray 模拟物理内存, 返回 (ram, read_fn, write_fn)."""
    ram = bytearray(size_bytes)

    def read_fn(addr: int, length: int) -> bytes:
        return bytes(ram[addr : addr + length])

    def write_fn(addr: int, data: bytes):
        for i, b in enumerate(data):
            ram[addr + i] = b

    return ram, read_fn, write_fn


# ============================================================
#  PTE 字段读写
# ============================================================


class TestPTEFlags:
    """PTE 标志位的读写."""

    def test_default_zero(self):
        pte = PTE()
        assert pte.raw == 0
        assert pte.v is False
        assert pte.r is False
        assert pte.w is False
        assert pte.x is False

    def test_set_v(self):
        pte = PTE()
        pte.v = True
        assert pte.raw & PTE_V
        pte.v = False
        assert not (pte.raw & PTE_V)

    def test_set_rwx(self):
        pte = PTE()
        pte.r = True
        pte.w = True
        pte.x = True
        assert pte.r and pte.w and pte.x
        assert pte.raw & (PTE_R | PTE_W | PTE_X)

    def test_set_ugad(self):
        pte = PTE()
        pte.u = True
        pte.g = True
        pte.a = True
        pte.d = True
        assert pte.u and pte.g and pte.a and pte.d

    def test_flags_independent(self):
        """各标志位应互不干扰."""
        pte = PTE()
        pte.r = True
        assert pte.r and not pte.w and not pte.x
        pte.w = True
        assert pte.r and pte.w and not pte.x
        # 清除 r 不影响 w
        pte.r = False
        assert not pte.r and pte.w

    def test_from_int(self):
        pte = PTE.from_int(0xF)  # V=1, R=1, W=1, X=1
        assert pte.v and pte.r and pte.w and pte.x

    def test_to_int(self):
        pte = PTE()
        pte.v = True
        pte.r = True
        assert pte.to_int() == (PTE_V | PTE_R)


class TestPTEPPN:
    """PTE 物理页号 (PPN) 的拆分与组合."""

    def test_ppn_write_read(self):
        pte = PTE()
        pte.ppn = 0x12345
        assert pte.ppn == 0x12345

    def test_ppn_zero(self):
        pte = PTE()
        pte.ppn = 0
        assert pte.ppn == 0
        assert pte.ppn0 == 0
        assert pte.ppn1 == 0
        assert pte.ppn2 == 0

    def test_ppn_max(self):
        """44 位 PPN 的最大值."""
        max_ppn = (1 << 44) - 1
        pte = PTE()
        pte.ppn = max_ppn
        assert pte.ppn == max_ppn

    def test_ppn0_field(self):
        """PPN[0] 占 bits 19:10 (10 bits)."""
        pte = PTE()
        pte.raw = 0x3FF << 10  # 全部 10 位置 1
        assert pte.ppn0 == 0x3FF

    def test_ppn1_field(self):
        """PPN[1] 占 bits 28:20 (9 bits)."""
        pte = PTE()
        pte.raw = 0x1FF << 20
        assert pte.ppn1 == 0x1FF

    def test_ppn2_field(self):
        """PPN[2] 占 bits 53:29."""
        pte = PTE()
        pte.raw = 0x1FF << 29
        assert pte.ppn2 == 0x1FF

    def test_ppn_does_not_affect_flags(self):
        """写 PPN 不应影响标志位."""
        pte = PTE()
        pte.r = True
        pte.w = True
        pte.ppn = 0xABCDEF
        assert pte.r and pte.w, "标志位应保持不变"


class TestPTELeafDetection:
    """叶节点 / 指针判断."""

    def test_invalid_is_not_leaf(self):
        pte = PTE()  # V=0
        assert not pte.is_leaf()
        assert not pte.is_ptr()

    def test_pointer_is_not_leaf(self):
        """仅 V 置位, R/W/X 全 0: 非叶节点 (指针)."""
        pte = PTE()
        pte.v = True
        assert not pte.is_leaf()
        assert pte.is_ptr()

    def test_readable_is_leaf(self):
        pte = PTE()
        pte.v = True
        pte.r = True
        assert pte.is_leaf()

    def test_executable_is_leaf(self):
        pte = PTE()
        pte.v = True
        pte.x = True
        assert pte.is_leaf()

    def test_rwx_all_is_leaf(self):
        pte = PTE()
        pte.v = True
        pte.r = True
        pte.w = True
        pte.x = True
        assert pte.is_leaf()
        assert not pte.is_ptr()

    def test_w_only_is_not_leaf(self):
        """仅 W 置位 (无 R 也无 X): 按规范 W 单独不应是叶 (RISC-V 不允许 write-only)."""
        pte = PTE()
        pte.v = True
        pte.w = True
        assert not pte.is_leaf(), "仅有 W 无 R/X → 不是合法叶 PTE"


class TestPTECheckPerm:
    """权限检查."""

    @pytest.fixture
    def user_pte(self) -> PTE:
        pte = PTE()
        pte.v = True
        pte.r = True
        pte.w = True
        pte.u = True  # 用户可访问
        return pte

    @pytest.fixture
    def supervisor_pte(self) -> PTE:
        pte = PTE()
        pte.v = True
        pte.r = True
        pte.x = True
        # u=False → 仅 S/M 可访问
        return pte

    def test_user_access_allowed(self, user_pte):
        assert user_pte.check_perm(want_r=True, want_w=False, want_x=False, is_user=True)

    def test_user_access_denied_without_u(self, supervisor_pte):
        assert not supervisor_pte.check_perm(
            want_r=True, want_w=False, want_x=False, is_user=True
        )

    def test_supervisor_access_allowed_without_u(self, supervisor_pte):
        assert supervisor_pte.check_perm(
            want_r=True, want_w=False, want_x=False, is_user=False
        )

    def test_write_denied_without_w(self, user_pte):
        user_pte.w = False
        assert not user_pte.check_perm(want_r=True, want_w=True, want_x=False, is_user=False)

    def test_execute_denied_without_x(self, user_pte):
        assert not user_pte.check_perm(want_r=False, want_w=False, want_x=True, is_user=False)

    def test_invalid_pte_denies_all(self):
        pte = PTE()  # v=0
        assert not pte.check_perm(want_r=True, want_w=False, want_x=False, is_user=False)


# ============================================================
#  VPN 分解
# ============================================================


class TestVpnDecomposition:
    """虚拟地址 VPN 分解."""

    def test_sv39_zero_va(self):
        vpn2, vpn1, vpn0 = _sv39_vpn(0)
        assert vpn2 == 0 and vpn1 == 0 and vpn0 == 0

    def test_sv39_vpn0_field(self):
        """VA[20:12] → vpn0, 测试边界."""
        # 设置 VA[20:12] = 0x1FF
        va = 0x1FF << 12
        _, _, vpn0 = _sv39_vpn(va)
        assert vpn0 == 0x1FF

    def test_sv39_vpn1_field(self):
        """VA[29:21] → vpn1."""
        va = 0x1FF << 21
        _, vpn1, _ = _sv39_vpn(va)
        assert vpn1 == 0x1FF

    def test_sv39_vpn2_field(self):
        """VA[38:30] → vpn2."""
        va = 0x1FF << 30
        vpn2, _, _ = _sv39_vpn(va)
        assert vpn2 == 0x1FF

    def test_sv39_offset_not_in_vpn(self):
        """VA[11:0] 的页内偏移不应影响 VPN."""
        va = 0xFFF  # 全部 offset 位置 1
        vpn2, vpn1, vpn0 = _sv39_vpn(va)
        assert vpn2 == 0 and vpn1 == 0 and vpn0 == 0


# ============================================================
#  Sv39 页表遍历
# ============================================================


class TestSv39Walk:
    """Sv39 三级页表遍历."""

    @pytest.fixture
    def ram_ctx(self):
        """提供一个 2 MiB 的物理内存."""
        return _make_ram()

    def _write_pte(self, ram, write_fn, addr: int, pte: PTE):
        """向物理内存写入一个 PTE."""
        data = pte.to_int().to_bytes(8, "little")
        write_fn(addr, data)

    def test_simple_4kib_page(self, ram_ctx):
        """三级 4 KiB 页: 每一级都跳转到下一级页表."""
        ram, read_fn, write_fn = ram_ctx

        # 根页表在 PPN=1 (物理地址 0x1000)
        root_ppn = 1
        l1_pte = PTE()
        l1_pte.v = True
        l1_pte.ppn = 2  # → L2 表在 PPN=2
        self._write_pte(ram, write_fn, (1 << PAGE_SHIFT), l1_pte)

        # L2 表在 PPN=2 (物理地址 0x2000)
        l2_pte = PTE()
        l2_pte.v = True
        l2_pte.ppn = 3  # → L3 表在 PPN=3
        self._write_pte(ram, write_fn, (2 << PAGE_SHIFT), l2_pte)

        # L3 叶 PTE: 映射 VPN(0,0,0) → PPN=0x40
        l3_pte = PTE()
        l3_pte.v = True
        l3_pte.r = True
        l3_pte.w = True
        l3_pte.x = True
        l3_pte.ppn = 0x40
        self._write_pte(ram, write_fn, (3 << PAGE_SHIFT), l3_pte)

        satp = (SATP_MODE_SV39 << 60) | root_ppn
        va = 0x0
        ok, pa = translate_va(va, satp, read_fn)
        assert ok, "三层 4 KiB 页遍历应成功"
        assert pa == 0x40000, f"PA 应为 0x40000, 得到 0x{pa:x}"

    def test_page_offset_preserved(self, ram_ctx):
        """VA 页内偏移应正确传递到 PA (2 MiB 大页)."""
        ram, read_fn, write_fn = ram_ctx

        # 2 MiB 大页: 第 2 级命中, PTE.ppn[9:0] 由 VA.vpn[0] 提供
        root_ppn = 1
        l1_pte = PTE()
        l1_pte.v = True
        l1_pte.ppn = 2
        self._write_pte(ram, write_fn, (1 << PAGE_SHIFT), l1_pte)

        # L2: 2 MiB 大页, PPN[43:9] 部分 = 0x20 (即 PA 基址 0x2000000)
        l2_pte = PTE()
        l2_pte.v = True
        l2_pte.r = True
        l2_pte.ppn = 0x4000  # PPN[43:9] = 0x20, PPN[8:0] = 0
        self._write_pte(ram, write_fn, (2 << PAGE_SHIFT), l2_pte)

        satp = (SATP_MODE_SV39 << 60) | root_ppn
        # va: vpn0=0x1AB, offset=0xABC
        va = (0x1AB << 12) | 0xABC
        ok, pa = translate_va(va, satp, read_fn)
        assert ok
        # PA: PPN[43:9]=0x20, PPN[8:0]=vpn0=0x1AB, offset=0xABC
        expected_pa = (0x41AB << 12) | 0xABC
        assert pa == expected_pa, f"PA=0x{pa:x}, expected=0x{expected_pa:x}"

    def test_bare_mode(self, ram_ctx):
        """Bare 模式: VA 即 PA, 不做翻译."""
        _, read_fn, _ = ram_ctx
        satp = SATP_MODE_BARE << 60
        ok, pa = translate_va(0xDEADBEEF, satp, read_fn)
        assert ok
        assert pa == 0xDEADBEEF

    def test_invalid_vpn_miss(self, ram_ctx):
        """VPN 对应 PTE 无效时应失败."""
        ram, read_fn, write_fn = ram_ctx

        # 根页表存在但对应的 PTE 无效 (V=0)
        root_ppn = 1
        invalid_pte = PTE()  # v=False
        self._write_pte(ram, write_fn, (1 << PAGE_SHIFT), invalid_pte)

        satp = (SATP_MODE_SV39 << 60) | root_ppn
        ok, pa = translate_va(0x0, satp, read_fn)
        assert not ok, "无效 PTE 应导致翻译失败"

    def test_unsupported_mode_fails(self, ram_ctx):
        """未实现的模式 (如 Sv48) 应返回失败."""
        _, read_fn, _ = ram_ctx
        satp = (SATP_MODE_SV39 + 1) << 60  # 无效/未支持的模式
        ok, pa = translate_va(0x0, satp, read_fn)
        assert not ok


# ============================================================
#  MemAccessMode 枚举
# ============================================================


class TestMemAccessMode:
    def test_enum_values(self):
        modes = list(MemAccessMode)
        assert len(modes) == 4
        names = {m.name for m in modes}
        assert names == {"None", "SV32", "SV39", "SV48"}
