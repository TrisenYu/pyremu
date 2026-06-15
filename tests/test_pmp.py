#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""PMP (Physical Memory Protection) 测试: 地址匹配、权限控制、CSR 范围."""

import pytest

from pyremu.core.mem_check_aux import inject_memory_backend, mem_read, mem_write
from pyremu.core.decoder import Hart
from pyremu.core.hart import RiscvMode
from pyremu.memory.pmp import Pmp


# ============================================================
#  NAPOT 编解码
# ============================================================


class TestNapotDecode:
    """验证 NAPOT 格式的 pmpaddr → (base, size) 解码."""

    def test_8_byte_region(self):
        """8 字节区域 (k=0, 无 trailing 1s)."""
        from pyremu.memory.pmp import _decode_napot

        # 8 字节 @ 0x1000: size=8, k=0, 无 padding
        # pmpaddr = base >> 2 = 0x400
        base, size = _decode_napot(0x400)
        assert size == 8
        assert base == 0x1000

    def test_4k_region(self):
        """4 KiB 区域 (k=9 trailing 1s)."""
        from pyremu.memory.pmp import _decode_napot

        # 4 KiB @ 0x80000000: k=9 (2^12)
        # pmpaddr = (0x80000000 >> 2) | 0x1FF = 0x20000000 | 0x1FF
        val = (0x8000_0000 >> 2) | 0x1FF
        base, size = _decode_napot(val)
        assert size == 4096
        assert base == 0x8000_0000

    def test_64k_region(self):
        """64 KiB 区域 (k=13 trailing 1s)."""
        from pyremu.memory.pmp import _decode_napot

        val = (0x8000_0000 >> 2) | 0x1FFF
        base, size = _decode_napot(val)
        assert size == 65536
        assert base == 0x8000_0000

    def test_all_ones_entire_space(self):
        """全 1 覆盖整个地址空间."""
        from pyremu.memory.pmp import _decode_napot

        base, size = _decode_napot(0x3F_FFFF_FFFF_FFFF)
        assert base == 0
        assert size > 2**63  # 无穷大


# ============================================================
#  Pmp.check (独立检查, 不依赖 hart 上下文)
# ============================================================


class TestPmpCheck:
    """Pmp.check() 地址匹配与权限."""

    @pytest.fixture
    def pmp(self) -> Pmp:
        """返回一个空的 Pmp (0 条目)."""
        return Pmp({}, num_entries=0)

    def _set_csrs(self, pmp: Pmp, entries: list[tuple[int, int]]):
        """设置 pmpaddr 和对应的 pmpcfg.

        *entries* 为 [(cfg, addr), ...] — cfg 为 8-bit, addr 为 pmpaddr 原始值.
        """
        class FakeCSR:
            val: int
            def __init__(self, v): self.val = v

        # 构建 cfg 组 (每 8 条目一组)
        cfg_vals: dict[int, int] = {}
        for i, (cfg, _) in enumerate(entries):
            group = i // 8
            cfg_vals[group] = cfg_vals.get(group, 0) | (cfg << ((i & 7) * 8))

        pmp._num_entries = len(entries)
        # 注入伪造 CSR
        for group, val in cfg_vals.items():
            pmp._csrs[f"pmpcfg{group * 2}"] = FakeCSR(val)  # type: ignore[attr-defined]
        for i, (_, addr) in enumerate(entries):
            pmp._csrs[f"pmpaddr{i}"] = FakeCSR(addr)  # type: ignore[attr-defined]

    # ---- M-mode bypass ----

    def test_m_mode_bypass(self, pmp):
        """M 模式 + MPRV=0: PMP 不检查, 直接放行."""
        # 无 PMP 条目 — S/U 会拒绝, M 放行
        assert pmp.check(0x1000, 4, mode_val=4, mstatus_val=0, is_write=True)

    def test_m_mode_mprv_uses_mpp(self, pmp):
        """M 模式 + MPRV=1: 按 MPP 特权级检查."""
        # MPRV=1, MPP=0 (U) — 应作为 U 模式检查
        # 0 条目 → U 模式拒绝
        mstatus = (1 << 17) | (0 << 11)  # MPRV=1, MPP=U
        assert not pmp.check(0x1000, 4, mode_val=4, mstatus_val=mstatus)

    # ---- TOR ----

    def test_tor_range(self, pmp):
        """TOR: pmpaddr[0]=0x1000 → 范围 [0, 0x4000)."""
        self._set_csrs(pmp, [
            (0b1000 | 0b0001, 0x1000),  # TOR, R, entry 0: [0, 0x4000)
        ])
        # 0x1000 << 2 = 0x4000 → range [0, 0x4000)
        assert pmp.check(0x0000, 4, mode_val=0, mstatus_val=0)  # inside
        assert not pmp.check(0x4000, 1, mode_val=0, mstatus_val=0)  # at hi bound

    def test_tor_range_multi_entry(self, pmp):
        """TOR 多条目: [addr[0]<<2, addr[1]<<2)."""
        self._set_csrs(pmp, [
            (0b1000 | 0b0001, 0x1000),  # entry 0: [0, 0x4000)
            (0b1000 | 0b0001, 0x2000),  # entry 1: [0x4000, 0x8000)
        ])
        assert pmp.check(0x0000, 4, mode_val=0, mstatus_val=0)  # entry 0
        assert pmp.check(0x4000, 4, mode_val=0, mstatus_val=0)  # entry 1
        assert not pmp.check(0x8000, 4, mode_val=0, mstatus_val=0)  # out

    # ---- NA4 ----

    def test_na4_match(self, pmp):
        """NA4: 4 字节精确匹配."""
        self._set_csrs(pmp, [
            (0b1_0000 | 0b0111, 0x8000_0000 >> 2),  # NA4, RWX @ 0x8000_0000
        ])
        # addr = pmpaddr << 2 = 0x8000_0000
        assert pmp.check(0x8000_0000, 4, mode_val=0, mstatus_val=0)
        assert not pmp.check(0x8000_0004, 4, mode_val=0, mstatus_val=0)

    # ---- NAPOT ----

    def test_napot_4k_range(self, pmp):
        """NAPOT 4 KiB 区域匹配."""
        val = (0x8000_0000 >> 2) | 0x1FF  # 4KB region @ 0x8000_0000
        self._set_csrs(pmp, [
            (0b11_000 | 0b0111, val),  # NAPOT, RWX
        ])
        assert pmp.check(0x8000_0000, 4, mode_val=0, mstatus_val=0)
        assert pmp.check(0x8000_0FFC, 4, mode_val=0, mstatus_val=0)  # last 4B
        assert not pmp.check(0x8000_1000, 4, mode_val=0, mstatus_val=0)  # out

    def test_napot_cross_boundary_access(self, pmp):
        """跨边界访问被拒绝."""
        val = (0x8000_0000 >> 2) | 0x1FF  # 4KB
        self._set_csrs(pmp, [
            (0b11_000 | 0b0111, val),
        ])
        # 从 0xFFC 读 8 字节 → 超出区域
        assert not pmp.check(0x8000_0FFC, 8, mode_val=0, mstatus_val=0)

    # ---- 权限 ----

    def test_r_denied(self, pmp):
        """R=0 拒绝读 (但 W 检查不受 R 影响)."""
        val = (0x8000_0000 >> 2) | 0x1FF
        self._set_csrs(pmp, [
            (0b11_000 | 0b0010, val),  # NAPOT, W only (no R)
        ])
        assert not pmp.check(0x8000_0000, 4, mode_val=0, mstatus_val=0)
        # R=0 时 W 也拒绝 (PMP 要求 R 必须置位)
        assert not pmp.check(0x8000_0000, 4, mode_val=0, mstatus_val=0, is_write=True)

    def test_w_denied(self, pmp):
        """W=0 拒绝写."""
        val = (0x8000_0000 >> 2) | 0x1FF
        self._set_csrs(pmp, [
            (0b11_000 | 0b0101, val),  # NAPOT, R+X
        ])
        assert pmp.check(0x8000_0000, 4, mode_val=0, mstatus_val=0)  # read OK
        assert not pmp.check(0x8000_0000, 4, mode_val=0, mstatus_val=0, is_write=True)

    # ---- 无条目 ----

    def test_no_entries_s_mode_rejected(self, pmp):
        """0 条目 → S 模式拒绝."""
        assert not pmp.check(0x1000, 4, mode_val=1, mstatus_val=0)

    # ---- S-mode by default rejected with no match ----

    def test_s_mode_no_match_rejected(self, pmp):
        """S 模式无匹配 PMP 条目 → 拒绝."""
        # 每条目都 OFF, 无匹配
        pmp._num_entries = 4
        assert not pmp.check(0x1000, 4, mode_val=1, mstatus_val=0)


# ============================================================
#  PMP CSR 条目范围检查
# ============================================================


class TestPmpCsrRange:
    """验证超范围的 pmpaddr/pmpcfg 访问触发 IllInstr."""

    @pytest.fixture
    def hart(self) -> Hart:
        """8 条目的 hart."""
        return Hart(id=0, pmp_entries=8)

    def _csrrw_instr(self, csr_addr: int) -> int:
        """CSRRW x5, csr, x0 (只读)."""
        return (csr_addr << 20) | (5 << 7) | (1 << 12) | 0x73

    def test_pmpaddr_7_valid(self, hart):
        """pmpaddr7 (在 8 条目范围内) 可正常访问."""
        hart.mode = RiscvMode.M
        instr = self._csrrw_instr(0x3B7)  # pmpaddr7
        advance = hart.exec_instr(instr)
        assert advance == 4, "pmpaddr7 应可访问"

    def test_pmpaddr_8_traps(self, hart):
        """pmpaddr8 (超出 8 条目) → IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw_instr(0x3B8)  # pmpaddr8
        hart.exec_instr(instr)
        assert hart.mcause_val == 2, f"应为 IllInstr, 实际 {hart.mcause_val}"

    def test_pmpcfg2_traps(self, hart):
        """pmpcfg2 (覆盖条目 8-15, 超出 8 条目) → IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw_instr(0x3A2)  # pmpcfg2
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_pmpcfg0_valid(self, hart):
        """pmpcfg0 (覆盖条目 0-7, 在范围内) 正常访问."""
        hart.mode = RiscvMode.M
        instr = self._csrrw_instr(0x3A0)  # pmpcfg0
        advance = hart.exec_instr(instr)
        assert advance == 4

    def test_pmp_entries_zero_all_trap(self):
        """pmp_entries=0 时所有 pmpaddr 均 trap."""
        h = Hart(id=0, pmp_entries=0)
        h.mode = RiscvMode.M
        instr = self._csrrw_instr(0x3B0)  # pmpaddr0
        h.exec_instr(instr)
        assert h.mcause_val == 2

    def test_non_pmp_csr_unaffected(self, hart):
        """非 PMP CSR (mstatus) 不受 pmp_entries 影响."""
        hart.mode = RiscvMode.M
        instr = self._csrrw_instr(0x300)  # mstatus
        advance = hart.exec_instr(instr)
        assert advance == 4


# ============================================================
#  通过 hart 执行 store 触发 PMP 拒绝
# ============================================================


class TestPmpInHart:
    """在 hart 上配置 PMP 后执行访存, 验证陷态."""

    @pytest.fixture
    def hart(self) -> Hart:
        from pyremu.memory.bus import Bus

        h = Hart(id=0, pmp_entries=4)
        bus = Bus(ram_size=1024 * 1024, ram_base=0x8000_0000)
        inject_memory_backend(h, bus.read, bus.write)
        h.bus = bus
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        return h

    def _setup_napot_rw(
        self,
        hart: Hart,
        base: int,
        size_log2: int,
        r: bool = True,
        w: bool = True,
    ):
        """配置 NAPOT PMP 条目 0."""
        k = size_log2 - 3
        mask = (1 << k) - 1
        val = (base >> 2) | mask

        cfg = PMP_A_NAPOT
        if r:
            cfg |= PMP_R
        if w:
            cfg |= PMP_W

        # 写 pmpaddr0
        hart.csrs["pmpaddr0"].val = val
        # 写 pmpcfg0 (entry 0)
        hart.csrs["pmpcfg0"].val = cfg

    def test_mmode_bypass_store(self, hart):
        """M 模式 store 不被 PMP 检查."""
        self._setup_napot_rw(hart, 0x8000_1000, 12, r=True, w=True)
        # 直接通过 _mem_write 写入
        mem_write(hart, 0x8000_1000, b"\xAA\xBB\xCC\xDD")
        # 无 trap
        assert hart.mcause_val == 0

    def test_umode_store_pmp_rw_ok(self, hart):
        """U 模式, PMP 允许 RW → store 成功."""
        from pyremu.memory.pmp import PMP_A_NAPOT, PMP_R, PMP_W

        self._setup_napot_rw(hart, 0x8000_1000, 12, r=True, w=True)
        hart.mode = RiscvMode.U
        mem_write(hart, 0x8000_1000, b"\x11\x22")
        assert hart.mcause_val == 0

    def test_umode_store_pmp_w_denied(self, hart):
        """U 模式, PMP W=0 → StAccessFault."""
        from pyremu.memory.pmp import PMP_A_NAPOT, PMP_R

        self._setup_napot_rw(hart, 0x8000_1000, 12, r=True, w=False)
        hart.mode = RiscvMode.U
        mem_write(hart, 0x8000_1000, b"\x11\x22")
        assert hart.mcause_val == 7, f"应为 StAccessFault(7), 实际 {hart.mcause_val}"

    def test_umode_load_pmp_r_denied(self, hart):
        """U 模式, PMP R=0 → LdAccessFault."""
        from pyremu.memory.pmp import PMP_A_NAPOT, PMP_W

        self._setup_napot_rw(hart, 0x8000_1000, 12, r=False, w=True)
        hart.mode = RiscvMode.U
        # 先写入实际数据
        hart._mem_write_phy(0x8000_1000, b"\xDE\xAD")
        # 清 trap 计数
        hart._consecutive_traps = 0
        mem_read(hart, 0x8000_1000, 4)
        assert hart.mcause_val == 5, f"应为 LdAccessFault(5), 实际 {hart.mcause_val}"


# 复用 pmp.py 的常量
from pyremu.memory.pmp import PMP_A_NAPOT, PMP_A_TOR, PMP_R, PMP_W, PMP_X  # noqa: E402
