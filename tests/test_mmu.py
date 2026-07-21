#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""MMU 测试: PTE 字段操作、Sv39 页表遍历、地址翻译、hart 级集成."""

import pytest

from pyremu.core.hart import HartWithRegs, RiscvMode
from pyremu.core.mem_check_aux import inject_memory_backend, translate_addr
from pyremu.memory.bus import Bus
from pyremu.memory.mmu import (
    PAGE_SHIFT,
    PTE,
    PTE_A,
    PTE_D,
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
        assert not pte.v and not pte.r and \
        not pte.w and not pte.x

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
        assert not pte.is_leaf(), "仅有 W 无 R/X -> 不是合法叶 PTE"


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
        # u=False -> 仅 S/M 可访问
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
        """VA[20:12] -> vpn0, 测试边界."""
        # 设置 VA[20:12] = 0x1FF
        va = 0x1FF << 12
        _, _, vpn0 = _sv39_vpn(va)
        assert vpn0 == 0x1FF

    def test_sv39_vpn1_field(self):
        """VA[29:21] -> vpn1."""
        va = 0x1FF << 21
        _, vpn1, _ = _sv39_vpn(va)
        assert vpn1 == 0x1FF

    def test_sv39_vpn2_field(self):
        """VA[38:30] -> vpn2."""
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
        l1_pte.ppn = 2  # -> L2 表在 PPN=2
        self._write_pte(ram, write_fn, (1 << PAGE_SHIFT), l1_pte)

        # L2 表在 PPN=2 (物理地址 0x2000)
        l2_pte = PTE()
        l2_pte.v = True
        l2_pte.ppn = 3  # -> L3 表在 PPN=3
        self._write_pte(ram, write_fn, (2 << PAGE_SHIFT), l2_pte)

        # L3 叶 PTE: 映射 VPN(0,0,0) -> PPN=0x40
        l3_pte = PTE()
        l3_pte.v = True
        l3_pte.r = True
        l3_pte.w = True
        l3_pte.x = True
        l3_pte.ppn = 0x40
        self._write_pte(ram, write_fn, (3 << PAGE_SHIFT), l3_pte)

        satp = (SATP_MODE_SV39 << 60) | root_ppn
        va = 0x0
        ok, pa, _perm = translate_va(va, satp, read_fn)
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
        ok, pa, _perm = translate_va(va, satp, read_fn)
        assert ok
        # PA: PPN[43:9]=0x20, PPN[8:0]=vpn0=0x1AB, offset=0xABC
        expected_pa = (0x41AB << 12) | 0xABC
        assert pa == expected_pa, f"PA=0x{pa:x}, expected=0x{expected_pa:x}"

    def test_megapage_ppn_mask_regression(self, ram_ctx):
        """回归: Python ~0x1FF 产生负无穷精度整数, 导致 PPN 掩码异常.

        2026-06-30: ``ppn & ∼0x3FF`` -> ``ppn & 0xFFFF_FFFF_FFFF_FC00`` (显式截断 64-bit).
        2026-07-02: 掩码从 10-bit (∼0x3FF) 修正为 9-bit (∼0x1FF):
          2 MiB 大页 offset 为 VA[20:0] (21 bits), vpn[0]=VA[20:12] (9 bits)
          仅替换 PPN[8:0]; PPN[9] 为物理地址有效位, 不应清零.
        """
        ram, read_fn, write_fn = ram_ctx

        root_ppn = 1
        l1_pte = PTE()
        l1_pte.v = True
        l1_pte.ppn = 2
        self._write_pte(ram, write_fn, (1 << PAGE_SHIFT), l1_pte)

        # 2 MiB 大页: PPN=0xABCD0 (bit 9=0, 此用例确保低 9 位正确替换)
        mega_ppn = 0xABCD0
        l2_pte = PTE()
        l2_pte.v = True
        l2_pte.r = True
        l2_pte.ppn = mega_ppn
        self._write_pte(ram, write_fn, (2 << PAGE_SHIFT), l2_pte)

        satp = (SATP_MODE_SV39 << 60) | root_ppn
        # VA: vpn[0]=0xAB, offset=0xCDE
        va = (0xAB << 12) | 0xCDE
        ok, pa, _perm = translate_va(va, satp, read_fn)
        assert ok, "大页翻译应成功"
        # PPN: 高位保留, PPN[8:0] -> vpn[0], offset 不变
        expected_pa = ((mega_ppn & 0xFFFFFFFFFFFE00) | 0xAB) << 12 | 0xCDE
        assert pa == expected_pa, f"PA=0x{pa:x}, expected=0x{expected_pa:x}"

    def test_megapage_ppn_bit9_preserved(self, ram_ctx):
        """2 MiB 大页 PPN bit 9 必须保留 — 它是物理地址的有效位.

        2026-07-02 修复: 掩码从 10-bit (∼0x3FF) 改为 9-bit (∼0x1FF),
        避免清零 PPN bit 9 导致翻译到错误的物理地址.
        真实场景: Linux Sv39 映射 VA=0xffffffff80000000 -> PA=0x80200000,
        PPN=0x80200 (bit 9=1), 被旧掩码清零后 PA 落到固件区 0x80000000.
        """
        ram, read_fn, write_fn = ram_ctx

        root_ppn = 1
        l1_pte = PTE()
        l1_pte.v = True
        l1_pte.ppn = 2
        self._write_pte(ram, write_fn, (1 << PAGE_SHIFT), l1_pte)

        # 使用真实场景中的 PPN 值: bit 9=1, bits[8:0]=0
        mega_ppn = 0x80200
        l2_pte = PTE()
        l2_pte.v = True
        l2_pte.r = True
        l2_pte.ppn = mega_ppn
        self._write_pte(ram, write_fn, (2 << PAGE_SHIFT), l2_pte)

        satp = (SATP_MODE_SV39 << 60) | root_ppn
        # VA: vpn[0]=1 (匹配 Linux 启动场景), offset=0x4c
        va = (1 << 12) | 0x4C
        ok, pa, _perm = translate_va(va, satp, read_fn)
        assert ok, "大页翻译应成功"
        # 正确: PPN bit 9 保留, PPN[8:0] -> vpn[0]=0x1
        expected_pa = ((mega_ppn & 0xFFFFFFFFFFFE00) | 0x1) << 12 | 0x4C
        assert pa == expected_pa, f"PA=0x{pa:x}, expected=0x{expected_pa:x}"
        # 回归: 确保旧掩码产生的错误值不等于当前结果
        old_buggy_pa = ((mega_ppn & 0xFFFFFFFFFFFC00) | 0x1) << 12 | 0x4C
        assert pa != old_buggy_pa, (
            f"掩码仍然错用 10-bit (~0x3FF)! PA=0x{pa:x} 不应等于旧值 0x{old_buggy_pa:x}"
        )

    def test_bare_mode(self, ram_ctx):
        """Bare 模式: VA 即 PA, 不做翻译."""
        _, read_fn, _ = ram_ctx
        satp = SATP_MODE_BARE << 60
        ok, pa, _perm = translate_va(0xDEADBEEF, satp, read_fn)
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
        ok, pa, _perm = translate_va(0x0, satp, read_fn)
        assert not ok, "无效 PTE 应导致翻译失败"

    def test_unsupported_mode_fails(self, ram_ctx):
        """未实现的模式 (如 Sv48) 应返回失败."""
        _, read_fn, _ = ram_ctx
        satp = (SATP_MODE_SV39 + 1) << 60  # 无效/未支持的模式
        ok, pa, _perm = translate_va(0x0, satp, read_fn)
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


# ============================================================
#  Hart 级 MMU 集成: satp 使能后内存读写
# ============================================================


class TestHartMMUIntegration:
    """验证 hart 设置 satp 后 translate_addr 和 mem_read 的正确性.

    此套用例覆盖从 VA->PA 的完整链路:
      TLB 查找 -> sv39_walk -> PTE 读取 -> 物理内存访问.
    """

    RAM_BASE = 0x80000000
    RAM_SIZE = 4 * 1024 * 1024  # 4 MiB

    @pytest.fixture
    def hart(self) -> HartWithRegs:
        h = HartWithRegs(id=0)
        bus = Bus(ram_size=self.RAM_SIZE, ram_base=self.RAM_BASE)
        inject_memory_backend(h, bus.read, bus.write)
        h.bus = bus
        h.pc = self.RAM_BASE
        h.mode = RiscvMode.S  # S-mode: 走 MMU 翻译 (非 M 模式)
        return h

    # ---- 辅助: 构建页表 ----

    @staticmethod
    def _encode_ppn(ppn: int) -> int:
        """RISC-V 标准连续 PPN 编码: PTE[53:10] = PPN[43:0]."""
        val = 0
        val |= (ppn & 0x3FF) << 10           # PPN[9:0] -> bits[19:10]
        val |= ((ppn >> 10) & 0x1FF) << 20   # PPN[18:10] -> bits[28:20]
        val |= ((ppn >> 19) & 0x1FFFFFFF) << 29  # PPN[43:19] -> bits[53:29]
        return val

    @staticmethod
    def _make_4kib_pte(flags: int, ppn: int) -> int:
        """构造 4 KiB 叶 PTE 值."""
        return flags | PTE_V | PTE_A | PTE_D | TestHartMMUIntegration._encode_ppn(ppn)

    @staticmethod
    def _make_pointer_pte(next_ppn: int) -> int:
        """构造中间表指针 PTE (V=1, R=W=X=0)."""
        return PTE_V | TestHartMMUIntegration._encode_ppn(next_ppn)

    @staticmethod
    def _make_2mib_megapage_pte(flags: int, ppn: int) -> int:
        """构造 2 MiB 大页 PTE (R 或 X 置位 -> 叶节点)."""
        return flags | PTE_V | PTE_A | PTE_D | TestHartMMUIntegration._encode_ppn(ppn)

    def _write_pte(self, hart: HartWithRegs, pa: int, value: int) -> None:
        """向物理地址写入一个 PTE (8 字节 LE)."""
        assert hart._mem_write_phy is not None
        hart._mem_write_phy(pa, value.to_bytes(8, "little"))

    # ---- 4 KiB 普通页: patp -> L2 -> L3 三级遍历 ----

    def test_identity_4kib_read_after_satp(self, hart):
        """satp 使能后, 4 KiB identity 映射 VA 能读到预期数据."""
        root_pa = self.RAM_BASE + 0x1000
        l2_pa = self.RAM_BASE + 0x2000
        l3_pa = self.RAM_BASE + 0x3000
        root_ppn = root_pa >> 12
        l2_ppn = l2_pa >> 12
        l3_ppn = l3_pa >> 12

        # VA=0x80000000 -> vpn[2]=2, 在 root[2] 指向 L2
        self._write_pte(hart, root_pa + 2 * 8, self._make_pointer_pte(l2_ppn))
        # VPN[1]=0 -> L2[0] 指向 L3
        self._write_pte(hart, l2_pa + 0 * 8, self._make_pointer_pte(l3_ppn))
        # VPN[0]=0 -> L3[0] 叶, identity: VA=0x80000000 -> PA=0x80000000
        ram_ppn = self.RAM_BASE >> 12
        self._write_pte(
            hart, l3_pa + 0 * 8,
            self._make_4kib_pte(PTE_R | PTE_W | PTE_X, ram_ppn),
        )

        # 写入魔数到物理地址
        magic = b"\xDE\xAD\xBE\xEF\xCA\xFE\xBA\xBE"
        hart._mem_write_phy(self.RAM_BASE, magic)

        # 使能 Sv39
        hart.satp_val = (SATP_MODE_SV39 << 60) | root_ppn
        assert hart.mmu_mode == SATP_MODE_SV39

        # ---- translate_addr ----
        ok, pa = translate_addr(hart, self.RAM_BASE)
        assert ok, "identity 4 KiB 页应翻译成功"
        assert pa == self.RAM_BASE, f"PA=0x{pa:x}, 期望=0x{self.RAM_BASE:x}"

        # ---- TLB 命中: 第二次访问应走 TLB 快捷路径 ----
        ok2, pa2 = translate_addr(hart, self.RAM_BASE)
        assert ok2 and pa2 == self.RAM_BASE

        # ---- 读取数据 ----
        data = hart._mem_read_phy(pa, 8)
        assert data == magic, f"读到 0x{data.hex()}, 期望 0x{magic.hex()}"

    def test_identity_4kib_translate_only(self, hart):
        """仅测试 translate_addr, 不涉及 PMP/PMA."""
        root_pa = self.RAM_BASE
        root_ppn = root_pa >> 12

        # L3 在 root_pa + 0x1000
        l3_pa = root_pa + 0x1000
        l3_ppn = l3_pa >> 12

        # root[0] -> L3 叶 (VA=0x00000000)
        self._write_pte(hart, root_pa + 0 * 8, self._make_pointer_pte(l3_ppn))
        # L3[0] -> PA=0x80000000
        self._write_pte(
            hart, l3_pa + 0 * 8,
            self._make_4kib_pte(PTE_R | PTE_W, 0x80000),
        )

        hart.satp_val = (SATP_MODE_SV39 << 60) | root_ppn

        ok, pa = translate_addr(hart, 0x0)
        assert ok, "translate_addr 应成功"
        assert pa == 0x80000000, f"PA=0x{pa:x}"

    # ---- 2 MiB 大页: 二级命中 ----

    def test_identity_2mib_megapage_translate(self, hart):
        """2 MiB 大页 identity 映射: root[2] -> L2[0] mega page."""
        root_pa = self.RAM_BASE + 0x1000
        l2_pa = self.RAM_BASE + 0x2000
        root_ppn = root_pa >> 12
        l2_ppn = l2_pa >> 12

        # root[2] -> L2 (VA=0x80000000 -> vpn[2]=2)
        self._write_pte(hart, root_pa + 2 * 8, self._make_pointer_pte(l2_ppn))
        # L2[0] mega page: identity VA=0x80000000 -> PA=0x80000000
        # PPN 的高位部分: 0x80000000 >> 12 = 0x80000
        self._write_pte(
            hart, l2_pa + 0 * 8,
            self._make_2mib_megapage_pte(PTE_R | PTE_W | PTE_X, 0x80000),
        )

        hart.satp_val = (SATP_MODE_SV39 << 60) | root_ppn

        # VA 0x80000000 应翻译到 PA 0x80000000
        ok, pa = translate_addr(hart, 0x80000000)
        assert ok, "2 MiB mega page 应翻译成功"
        assert pa == 0x80000000, f"PA=0x{pa:x}"

    def test_2mib_megapage_offset_preserved(self, hart):
        """2 MiB 大页内偏移正确传递."""
        root_pa = self.RAM_BASE
        root_ppn = root_pa >> 12
        l2_pa = root_pa + 0x1000
        l2_ppn = l2_pa >> 12

        # root[0] -> L2 (VA 低 1 GiB)
        self._write_pte(hart, root_pa + 0 * 8, self._make_pointer_pte(l2_ppn))
        # L2 mega page 映射 VA[0, 2MiB) -> PA[0x10000, 0x30000)
        self._write_pte(
            hart, l2_pa + 0 * 8,
            self._make_2mib_megapage_pte(PTE_R | PTE_W, 0x10),
        )

        hart.satp_val = (SATP_MODE_SV39 << 60) | root_ppn

        # VA=0x1A000 -> offset 0xA000, PA 应 = 0x10000 + 0xA000 = 0x1A000
        ok, pa = translate_addr(hart, 0x1A000)
        assert ok
        assert pa == 0x1A000, f"PA=0x{pa:x} (大页内偏移应保留)"

    # ---- TLB 行为 ----

    def test_tlb_miss_then_hit(self, hart):
        """首次翻译 TLB miss -> sv39_walk -> 插入 TLB; 再次命中."""
        root_pa = self.RAM_BASE
        root_ppn = root_pa >> 12
        l2_pa = root_pa + 0x1000
        l2_ppn = l2_pa >> 12

        self._write_pte(hart, root_pa + 0 * 8, self._make_pointer_pte(l2_ppn))
        self._write_pte(
            hart, l2_pa + 0 * 8,
            self._make_2mib_megapage_pte(PTE_R | PTE_W, 0x10),
        )

        hart.satp_val = (SATP_MODE_SV39 << 60) | root_ppn

        assert len(hart.dtlb) == 0, "初始 TLB 应为空"

        # 首次: miss -> walk -> insert
        ok, _ = translate_addr(hart, 0x1000)
        assert ok
        assert len(hart.dtlb) == 1, "首次翻译后 TLB 应有 1 条"

        # 再次: hit
        ok2, _ = translate_addr(hart, 0x1000)
        assert ok2
        assert len(hart.dtlb) == 1, "TLB 命中不应增加条目"

    def test_sfence_vma_flushes_both_tlbs(self, hart):
        """SFENCE.VMA 应清空 itlb 和 dtlb."""
        root_pa = self.RAM_BASE
        root_ppn = root_pa >> 12
        l2_pa = root_pa + 0x1000
        l2_ppn = l2_pa >> 12

        self._write_pte(hart, root_pa + 0 * 8, self._make_pointer_pte(l2_ppn))
        self._write_pte(
            hart, l2_pa + 0 * 8,
            self._make_2mib_megapage_pte(PTE_R | PTE_W | PTE_X, 0x10),
        )

        hart.satp_val = (SATP_MODE_SV39 << 60) | root_ppn

        # 填充 DTLB
        translate_addr(hart, 0x1000)
        assert len(hart.dtlb) == 1

        # SFENCE.VMA 全刷新
        hart.itlb.flush_all()
        hart.dtlb.flush_all()
        assert len(hart.dtlb) == 0
        assert len(hart.itlb) == 0

    # ---- Bare 模式/M 模式直通 ----

    def test_bare_mode_passthrough(self, hart):
        """Bare 模式 (satp.mode=0): VA 即 PA, 不走页表."""
        hart.satp_val = SATP_MODE_BARE << 60
        assert hart.mmu_mode == 0

        ok, pa = translate_addr(hart, 0x80001000)
        assert ok
        assert pa == 0x80001000

    def test_mmode_bypass_mmu(self, hart):
        """M 模式始终使用 Bare 翻译 (MPRV=0 时)."""
        root_pa = self.RAM_BASE
        root_ppn = root_pa >> 12

        # 建一个页表但 M 模式不应走它
        self._write_pte(hart, root_pa + 0 * 8, self._make_pointer_pte(0))  # 无效

        hart.satp_val = (SATP_MODE_SV39 << 60) | root_ppn
        hart.mode = RiscvMode.M  # 切换到 M 模式

        ok, pa = translate_addr(hart, 0x80001000)
        assert ok
        assert pa == 0x80001000, "M 模式应绕过 MMU"

    # ---- 错误路径 ----

    def test_invalid_pte_causes_translation_failure(self, hart):
        """无效 PTE (V=0) 导致 translate_addr 失败."""
        root_pa = self.RAM_BASE
        root_ppn = root_pa >> 12

        # root[0] = 0 (V=0) -> 无效
        self._write_pte(hart, root_pa + 0 * 8, 0)

        hart.satp_val = (SATP_MODE_SV39 << 60) | root_ppn

        ok, pa = translate_addr(hart, 0x0)
        assert not ok, "无效 PTE 应导致翻译失败"

    def test_walk_past_valid_range_fails(self, hart):
        """页表遍历中途 V=0 应失败."""
        root_pa = self.RAM_BASE
        root_ppn = root_pa >> 12
        l2_pa = root_pa + 0x1000
        l2_ppn = l2_pa >> 12

        # root[0] -> L2 (有效指针)
        self._write_pte(hart, root_pa + 0 * 8, self._make_pointer_pte(l2_ppn))
        # L2[0] = 0 -> 中途断裂
        self._write_pte(hart, l2_pa + 0 * 8, 0)

        hart.satp_val = (SATP_MODE_SV39 << 60) | root_ppn

        ok, pa = translate_addr(hart, 0x0)
        assert not ok, "中间表 V=0 应导致失败"

    # ---- 非零 ASID 不影响翻译 ----

    def test_asid_ignored_in_sv39_walk(self, hart):
        """ASID 非零不影响地址翻译 (ASID 仅用于 TLB 标记匹配)."""
        root_pa = self.RAM_BASE
        root_ppn = root_pa >> 12
        l2_pa = root_pa + 0x1000
        l2_ppn = l2_pa >> 12

        self._write_pte(hart, root_pa + 0 * 8, self._make_pointer_pte(l2_ppn))
        # 2 MiB mega: PPN=0x800 -> PPN[43:9]=4, PA base=0x800000
        self._write_pte(
            hart, l2_pa + 0 * 8,
            self._make_2mib_megapage_pte(PTE_R | PTE_W, 0x800),
        )

        # ASID = 0xAB
        hart.satp_val = (SATP_MODE_SV39 << 60) | (0xAB << 44) | root_ppn
        assert hart.mmu_mode == SATP_MODE_SV39

        # VA=0x0 -> vpn[0]=0, mega PPN[9:0]=0 -> PA=0x800000
        ok, pa = translate_addr(hart, 0x0)
        assert ok and pa == 0x800000, f"PA=0x{pa:x}, ASID 不影响页表遍历"


# ============================================================
#  satp.ASID 硬连线为 0 (WARL) — 回归
# ============================================================


class TestSatpAsidHardwiredZero:
    """satp.ASID (bits[59:44]) 必须写入即被清零 (WARL 读回 0).

    回归背景: TLB (Python 与 Rust 批量引擎) 查找均不带 ASID 标签。旧行为
    原样存储 ASID → Linux 探测到 ASID 支持 → 启用 ASID 分配器 → 上下文
    切换仅改写 satp.ASID 而不执行 sfence.vma → 前一地址空间的 TLB 表项
    残留命中 → 用户进程读脏数据 SIGSEGV (实测: ls 崩于 ld.so, 现场
    satp=0x8000100000082f1b 即 ASID=1 证明分配器已激活)。
    """

    # Linux ASID 探测写法: ASID 全 1; PPN 取现场值 0x82f1b
    PROBE = (8 << 60) | (0xFFFF << 44) | 0x82F1B
    EXPECT = (8 << 60) | 0x82F1B

    def test_setter_masks_asid(self):
        """satp_val setter 写入全 1 ASID, 读回 ASID=0 且 MODE/PPN 保留."""
        h = HartWithRegs(id=0)
        h.satp_val = self.PROBE
        assert h.satp_val == self.EXPECT
        assert h.satp_val != self.PROBE, "旧行为 (ASID 原样存储) 不得重现"
        assert h.mmu_mode == 8
        assert h.satp_ppn == 0x82F1B

    def test_write_csr_masks_asid(self):
        """csrw satp 路径 (write_csr) 同样清零 ASID."""
        h = HartWithRegs(id=0)
        h.mode = RiscvMode.M
        h.write_csr(0x180, self.PROBE)
        assert h.satp_val == self.EXPECT
        assert h.mmu_mode == 8

    def test_nonzero_asid_field_from_live_session(self):
        """现场触发值 0x8000100000082f1b (ASID=1) 写入后 ASID 归零."""
        h = HartWithRegs(id=0)
        h.satp_val = 0x8000_1000_0008_2F1B
        assert h.satp_val == 0x8000_0000_0008_2F1B
