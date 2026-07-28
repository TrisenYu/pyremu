#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""PMP (Physical Memory Protection) 测试: 地址匹配、权限控制、CSR 范围."""

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.hart import RiscvMode
from pyremu.emulator import Emulator
from pyremu.core.mem_check_aux import (
    MemoryAccessFault,
    check_instruction_fetch,
    inject_memory_backend,
    mem_read,
    mem_write,
)
from pyremu.core.registers import _MmodeCSR
from pyremu.memory.bus import Bus
from pyremu.memory.pmp import (
    PMP_A_NAPOT,
    PMP_R,
    PMP_W,
    PMP_X,
    Pmp,
    PmpAccessInfo,
    decode_napot,
)

# ============================================================
#  NAPOT 编解码
# ============================================================


class TestNapotDecode:
    """验证 NAPOT 格式的 pmpaddr -> (base, size) 解码."""

    # es: expected_size
    # eb: expected_base
    @pytest.mark.parametrize(
        "v, es, eb",
        [
            # 8 字节 @ 0x1000: size=8, k=0, 无 padding
            # pmpaddr = base >> 2 = 0x400
            (0x400, 8, 0x1000),
            # 4 KiB @ 0x80000000: k=9 (2^12)
            # pmpaddr = (0x80000000 >> 2) | 0x1FF = 0x20000000 | 0x1FF
            ((0x8000_0000 >> 2) | 0x1FF, 4096, 0x8000_0000),
            # 64 KiB 区域 (k=13 trailing 1s)
            ((0x8000_0000 >> 2) | 0x1FFF, 65536, 0x8000_0000),
            (0x3F_FFFF_FFFF_FFFF, 2**63, 0),
        ],
    )
    def test_pmp_basic_fn(self, v: int, es: int, eb: int):
        base, size = decode_napot(v)
        assert base == eb and size >= es


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

            def __init__(self, v):
                self.val = v

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
        assert pmp.check(PmpAccessInfo(pa=0x1000, size=4, mode_val=3, mstatus_val=0, is_write=True))

    def test_m_mode_mprv_uses_mpp(self, pmp):
        """M 模式 + MPRV=1: 按 MPP 特权级检查."""
        # MPRV=1, MPP=0 (U) — 应作为 U 模式检查
        # 0 条目 -> U 模式拒绝
        mstatus = (1 << 17) | (0 << 11)  # MPRV=1, MPP=U
        assert not pmp.check(PmpAccessInfo(0x1000, 4, mode_val=3, mstatus_val=mstatus))

    # ---- TOR ----

    def test_tor_range(self, pmp):
        """TOR: pmpaddr[0]=0x1000 -> 范围 [0, 0x4000)."""
        self._set_csrs(
            pmp,
            [
                (0b1000 | 0b0001, 0x1000),  # TOR, R, entry 0: [0, 0x4000)
            ],
        )
        # 0x1000 << 2 = 0x4000 -> range [0, 0x4000)
        assert pmp.check(PmpAccessInfo(0x0000, 4, mode_val=0, mstatus_val=0))  # inside
        assert not pmp.check(PmpAccessInfo(0x4000, 1, mode_val=0, mstatus_val=0))  # at hi bound

    def test_tor_range_multi_entry(self, pmp):
        """TOR 多条目: [addr[0]<<2, addr[1]<<2)."""
        self._set_csrs(
            pmp,
            [
                (0b1000 | 0b0001, 0x1000),  # entry 0: [0, 0x4000)
                (0b1000 | 0b0001, 0x2000),  # entry 1: [0x4000, 0x8000)
            ],
        )
        assert pmp.check(PmpAccessInfo(0x0000, 4, mode_val=0, mstatus_val=0))  # entry 0
        assert pmp.check(PmpAccessInfo(0x4000, 4, mode_val=0, mstatus_val=0))  # entry 1
        assert not pmp.check(PmpAccessInfo(0x8000, 4, mode_val=0, mstatus_val=0))  # out

    # ---- NA4 ----

    def test_na4_match(self, pmp):
        """NA4: 4 字节精确匹配."""
        self._set_csrs(
            pmp,
            [
                (0b1_0000 | 0b0111, 0x8000_0000 >> 2),  # NA4, RWX @ 0x8000_0000
            ],
        )
        # addr = pmpaddr << 2 = 0x8000_0000
        assert pmp.check(PmpAccessInfo(0x8000_0000, 4, mode_val=0, mstatus_val=0))
        assert not pmp.check(PmpAccessInfo(0x8000_0004, 4, mode_val=0, mstatus_val=0))

    # ---- NAPOT ----

    def test_napot_4k_range(self, pmp):
        """NAPOT 4 KiB 区域匹配."""
        val = (0x8000_0000 >> 2) | 0x1FF  # 4KB region @ 0x8000_0000
        self._set_csrs(
            pmp,
            [
                (0b11_000 | 0b0111, val),  # NAPOT, RWX
            ],
        )
        assert pmp.check(PmpAccessInfo(0x8000_0000, 4, mode_val=0, mstatus_val=0))
        assert pmp.check(PmpAccessInfo(0x8000_0FFC, 4, mode_val=0, mstatus_val=0))  # last 4B
        assert not pmp.check(PmpAccessInfo(0x8000_1000, 4, mode_val=0, mstatus_val=0))  # out

    def test_napot_cross_boundary_access(self, pmp):
        """跨边界访问被拒绝."""
        val = (0x8000_0000 >> 2) | 0x1FF  # 4KB
        self._set_csrs(
            pmp,
            [
                (0b11_000 | 0b0111, val),
            ],
        )
        # 从 0xFFC 读 8 字节 -> 超出区域
        assert not pmp.check(PmpAccessInfo(0x8000_0FFC, 8, mode_val=0, mstatus_val=0))

    # ---- 权限 ----

    def test_r_denied(self, pmp):
        """R=0 拒绝读 (但 W 检查不受 R 影响)."""
        val = (0x8000_0000 >> 2) | 0x1FF
        self._set_csrs(
            pmp,
            [
                (0b11_000 | 0b0010, val),  # NAPOT, W only (no R)
            ],
        )
        assert not pmp.check(PmpAccessInfo(0x8000_0000, 4, mode_val=0, mstatus_val=0))
        # R=0 时 W 也拒绝 (PMP 要求 R 必须置位)
        assert not pmp.check(PmpAccessInfo(pa=0x8000_0000, size=4, mode_val=0, mstatus_val=0, is_write=True))

    def test_w_denied(self, pmp):
        """W=0 拒绝写."""
        val = (0x8000_0000 >> 2) | 0x1FF
        self._set_csrs(
            pmp,
            [
                (0b11_000 | 0b0101, val),  # NAPOT, R+X
            ],
        )
        assert pmp.check(PmpAccessInfo(0x8000_0000, 4, mode_val=0, mstatus_val=0))  # read OK
        assert not pmp.check(PmpAccessInfo(pa=0x8000_0000, size=4, mode_val=0, mstatus_val=0, is_write=True))

    # ---- 无条目 ----

    def test_no_entries_s_mode_rejected(self, pmp):
        """0 条目 -> S 模式拒绝."""
        assert not pmp.check(PmpAccessInfo(0x1000, 4, mode_val=1, mstatus_val=0))

    # ---- S-mode by default rejected with no match ----

    def test_s_mode_no_match_rejected(self, pmp):
        """S 模式无匹配 PMP 条目 -> 拒绝."""
        # 每条目都 OFF, 无匹配
        pmp._num_entries = 4
        assert not pmp.check(PmpAccessInfo(0x1000, 4, mode_val=1, mstatus_val=0))


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
        """pmpaddr8 (超出 8 条目) -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw_instr(0x3B8)  # pmpaddr8
        hart.exec_instr(instr)
        assert hart.mcause_val == 2, f"应为 IllInstr, 实际 {hart.mcause_val}"

    def test_pmpcfg2_traps(self, hart):
        """pmpcfg2 (覆盖条目 8-15, 超出 8 条目) -> IllInstr."""
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
        mem_write(hart, 0x8000_1000, b"\xaa\xbb\xcc\xdd")
        # 无 trap
        assert hart.mcause_val == 0

    def test_umode_store_pmp_rw_ok(self, hart):
        """U 模式, PMP 允许 RW -> store 成功."""
        self._setup_napot_rw(hart, 0x8000_1000, 12, r=True, w=True)
        hart.mode = RiscvMode.U
        mem_write(hart, 0x8000_1000, b"\x11\x22")
        assert hart.mcause_val == 0

    def test_umode_store_pmp_w_denied(self, hart):
        """U 模式, PMP W=0 -> StAccessFault."""
        self._setup_napot_rw(hart, 0x8000_1000, 12, r=True, w=False)
        hart.mode = RiscvMode.U
        try:
            mem_write(hart, 0x8000_1000, b"\x11\x22")
        except MemoryAccessFault:
            pass
        assert hart.mcause_val == 7, f"应为 StAccessFault(7), 实际 {hart.mcause_val}"

    def test_umode_load_pmp_r_denied(self, hart):
        """U 模式, PMP R=0 -> LdAccessFault."""
        self._setup_napot_rw(hart, 0x8000_1000, 12, r=False, w=True)
        hart.mode = RiscvMode.U
        hart._mem_write_phy(0x8000_1000, b"\xde\xad")
        hart._consecutive_traps = 0
        try:
            mem_read(hart, 0x8000_1000, 4)
        except MemoryAccessFault:
            pass
        assert hart.mcause_val == 5, f"应为 LdAccessFault(5), 实际 {hart.mcause_val}"


# ============================================================
#  check_instruction_fetch — 取指路径的 PMP 检查
# ============================================================


class TestInstrFetchPmp:
    """验证 check_instruction_fetch() 对取指路径施加 PMP execute 检查."""

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0, pmp_entries=4)
        bus = Bus(ram_size=1024 * 1024, ram_base=0x8000_0000)
        inject_memory_backend(h, bus.read, bus.write)
        h.bus = bus
        h.csrs["mtvec"].val = 0x8000_0000
        h.pc = 0x8000_1000
        return h

    def _setup_napot(self, hart: Hart, base: int, size_log2: int,
                     r: bool = True, w: bool = True, x: bool = True):
        """配置 NAPOT PMP 条目 0."""
        k = size_log2 - 3
        mask = (1 << k) - 1
        val = (base >> 2) | mask

        cfg = PMP_A_NAPOT
        if r:
            cfg |= PMP_R
        if w:
            cfg |= PMP_W
        if x:
            cfg |= PMP_X

        hart.csrs["pmpaddr0"].val = val
        hart.csrs["pmpcfg0"].val = cfg

    def test_bare_fetch_x_ok(self, hart):
        """Bare 模式, PMP X=1 -> 取指通过."""
        self._setup_napot(hart, 0x8000_1000, 12, r=True, w=False, x=True)
        hart.mode = RiscvMode.S
        ok, pa = check_instruction_fetch(hart, 0x8000_1000)
        assert ok
        assert pa == 0x8000_1000

    def test_bare_fetch_x_denied(self, hart):
        """Bare 模式, PMP X=0 -> InstrAccessFault."""
        self._setup_napot(hart, 0x8000_1000, 12, r=True, w=True, x=False)
        hart.mode = RiscvMode.S
        ok, pa = check_instruction_fetch(hart, 0x8000_1000)
        assert not ok
        assert hart.mcause_val == 1, (
            f"应为 InstrAccessFault(1), 实际 {hart.mcause_val}"
        )

    def test_bare_fetch_no_match_s_mode(self, hart):
        """S 模式无匹配 PMP 条目 -> InstrAccessFault."""
        # 所有条目 OFF -> S 模式取指被拒
        hart.mode = RiscvMode.S
        ok, pa = check_instruction_fetch(hart, 0x8000_1000)
        assert not ok
        assert hart.mcause_val == 1

    def test_mmode_bypass_fetch(self, hart):
        """M 模式取指 (MPRV=0) 不受 PMP 限制."""
        self._setup_napot(hart, 0x8000_1000, 12, r=True, w=True, x=False)
        hart.mode = RiscvMode.M
        # M 模式 MPRV=0 -> PMP 自动放行
        ok, pa = check_instruction_fetch(hart, 0x8000_1000)
        assert ok
        assert pa == 0x8000_1000

    def test_mmode_fetch_ignores_mprv(self, hart):
        """M 模式取指无视 MPRV (RISC-V spec §3.1.6.3).

        MPRV=1 + MPP=S 时, LOAD/STORE 按 S 模式检查 PMP,
        但取指始终用当前特权级 (M) 并绕过 PMP.
        回归: _do_claim 修复前 MPRV=1 导致 M 模式取指被 PMP 拒绝 → halted.
        """
        self._setup_napot(hart, 0x8000_1000, 12, r=True, w=True, x=False)
        hart.mode = RiscvMode.M
        # MPRV=1, MPP=S (bits[12:11]=01)
        hart.mstatus_val = (1 << 17) | (1 << 11)
        ok, pa = check_instruction_fetch(hart, 0x8000_1000)
        assert ok, (
            "MPRV=1 不应影响取指 — M 模式取指必须绕过 PMP"
        )
        assert pa == 0x8000_1000

    def test_fetch_pma_invalid_addr(self, hart):
        """取指地址不在有效内存范围 -> InstrAccessFault."""
        hart.mode = RiscvMode.M
        # M 模式 PMP 放行, 但 PMA 检查拒绝 (无效地址)
        ok, pa = check_instruction_fetch(hart, 0xDEAD_BEEF)
        assert not ok
        assert hart.mcause_val == 1


# ============================================================
#  Native batch PMP 取指检查 — 确保 Rust 侧与 Python 侧一致
# ============================================================


class TestNativeBatchPmpFetch:
    """验证 native batch 对 PMP 取指权限的检查与纯 Python 路径一致.

    Rust batch 通过 ``pmp_ok(fetch_pa, is_execute=true)`` 检查取指,
    与 Python ``check_instruction_fetch`` 保持语义一致.
    """

    @pytest.fixture
    def emu(self) -> Emulator:
        """单 hart 模拟器, PMP entries=64, ram_base=0x8000_0000."""
        return Emulator(
            num_harts=1,
            ram_base=0x8000_0000,
            ram_size=0x1000_0000,  # 256 MiB
            pmp_entries=64,
            prog_cnt=0x8020_0000,
        )

    def _setup_napot_entry(self, emu: Emulator, idx: int,
                           base: int, size_log2: int,
                           r: bool, w: bool, x: bool):
        """直接在 hart 上配置 NAPOT PMP 条目."""
        hart = emu.harts[0]
        k = size_log2 - 3
        mask = (1 << k) - 1
        pmpaddr_val = (base >> 2) | mask

        cfg = PMP_A_NAPOT
        if r:
            cfg |= PMP_R
        if w:
            cfg |= PMP_W
        if x:
            cfg |= PMP_X

        hart.csrs[f"pmpaddr{idx}"].val = pmpaddr_val
        # pmpcfgN: 每个存 8 条目, RV64 仅偶数 cfg (pmpcfg0, pmpcfg2, ...)
        cfg_reg = (idx // 8) * 2
        existing = hart.csrs[f"pmpcfg{cfg_reg}"].val
        shift = (idx % 8) * 8
        # 清除旧条目位
        existing &= ~(0xFF << shift)
        existing |= (cfg << shift)
        hart.csrs[f"pmpcfg{cfg_reg}"].val = existing
        # 使 PMP 缓存失效
        hart._pmp.invalidate_cache()

    def test_native_batch_smode_fetch_allowed(self, emu: Emulator):
        """S 模式取指 — PMP 覆盖该区域 (RWX) -> 正常执行, 无 trap."""
        hart = emu.harts[0]

        # 配置 PMP 条目 0: NAPOT 覆盖 [0x8000_0000, 0x8200_0000) 32 MiB RWX
        self._setup_napot_entry(emu, 0, 0x8000_0000, 25, r=True, w=True, x=True)
        # 配置 PMP 条目 1: NAPOT 覆盖 [0x0000_0000, 0x8000_0000) 2 GiB RWX
        self._setup_napot_entry(emu, 1, 0x0000_0000, 31, r=True, w=True, x=True)

        # 在 0x8020_1108 放置一条 ADDI 指令: addi x5, x0, 42
        addi_instr = (42 << 20) | (5 << 7) | 0b0010011
        emu.load_code(0x8020_1108, addi_instr.to_bytes(4, "little"))

        # 切换到 S 模式
        hart.pc = 0x8020_1108
        hart.mode = RiscvMode.S
        hart._consecutive_traps = 0
        hart.gprs[5] = 0

        emu.step()

        # ADDI 应成功执行: x5 = 42, PC += 4, 无 trap
        assert hart.gprs[5] == 42, f"x5 应为 42, 实际 {hart.gprs[5]}"
        assert hart.pc == 0x8020_110C, f"PC 应为 0x8020_110C, 实际 {hart.pc:#018x}"
        assert hart.mcause_val == 0, f"不应有 trap, mcause={hart.mcause_val}"

    def test_native_batch_smode_fetch_denied(self, emu: Emulator):
        """S 模式取指 — PMP X=0 -> InstrAccessFault."""
        hart = emu.harts[0]

        # 配置 PMP 条目 0: NAPOT 覆盖目标区域但 X=0 (仅 RW)
        self._setup_napot_entry(emu, 0, 0x8000_0000, 25, r=True, w=True, x=False)

        # 在 0x8020_1108 放置指令
        addi_instr = (42 << 20) | (5 << 7) | 0b0010011
        emu.load_code(0x8020_1108, addi_instr.to_bytes(4, "little"))

        # S 模式
        hart.pc = 0x8020_1108
        hart.mode = RiscvMode.S
        hart._consecutive_traps = 0
        hart.gprs[5] = 0

        emu.step()

        # 应触发 InstrAccessFault (mcause=1)
        assert hart.mcause_val == 1, (
            f"应为 InstrAccessFault(1), 实际 mcause={hart.mcause_val}"
        )

    def test_native_batch_smode_no_match_denied(self, emu: Emulator):
        """S 模式取指 — 所有 PMP 条目 OFF -> 拒绝."""
        hart = emu.harts[0]

        addi_instr = (42 << 20) | (5 << 7) | 0b0010011
        emu.load_code(0x8020_1108, addi_instr.to_bytes(4, "little"))

        hart.pc = 0x8020_1108
        hart.mode = RiscvMode.S
        hart._consecutive_traps = 0
        hart.gprs[5] = 0

        emu.step()

        # 无 PMP 条目匹配 -> S 模式拒绝
        assert hart.mcause_val == 1, (
            f"应为 InstrAccessFault, 实际 mcause={hart.mcause_val}"
        )

    def test_native_batch_mmode_fetch_always_ok(self, emu: Emulator):
        """M 模式 (MPRV=0) 取指 — PMP 检查始终通过."""
        hart = emu.harts[0]

        # 配置 PMP 条目 0: X=0 (不应影响 M 模式取指)
        self._setup_napot_entry(emu, 0, 0x8000_0000, 25, r=True, w=True, x=False)

        addi_instr = (42 << 20) | (5 << 7) | 0b0010011
        emu.load_code(0x8020_1108, addi_instr.to_bytes(4, "little"))

        hart.pc = 0x8020_1108
        hart.mode = RiscvMode.M
        hart._consecutive_traps = 0
        hart.gprs[5] = 0

        emu.step()

        # M 模式取指应成功
        assert hart.gprs[5] == 42, f"x5 应为 42, 实际 {hart.gprs[5]}"
        assert hart.mcause_val == 0, f"M 模式不应 trap, mcause={hart.mcause_val}"


class TestSyncFromFlat:
    """验证 _sync_from_flat 正确将 flat 数组同步回 CSR entries."""

    @staticmethod
    def _make_pmp(num_entries: int = 16) -> Pmp:
        csrs: dict[str, object] = {}
        # Create pmpcfg registers (even-numbered, RV64: 8 entries each)
        for reg_idx in range(0, (num_entries + 7) // 8 * 2, 2):
            name = f"pmpcfg{reg_idx}"
            csrs[name] = _MmodeCSR(name=name)
        # Create pmpaddr registers
        for i in range(num_entries):
            name = f"pmpaddr{i}"
            csrs[name] = _MmodeCSR(name=name)
        return Pmp(csrs, num_entries)

    def test_sync_cfg_to_entries(self):
        """_sync_from_flat 将 flat_cfg 字节写回 pmpcfg CSR."""
        pmp = self._make_pmp(16)
        # Modify flat arrays directly (as Rust would)
        pmp._flat_cfg[0] = 0x9F  # NAPOT, R=W=X=1 (locked)
        pmp._flat_cfg[3] = 0x0B  # TOR, R=W=1
        pmp._flat_cfg[8] = 0x18  # pmpcfg2, entry 8: NA4, X=1
        pmp._sync_from_flat()

        # pmpcfg0 = entries 0-7: byte 0 = 0x9F, byte 3 = 0x0B
        cfg0 = pmp._csrs["pmpcfg0"].val
        assert (cfg0 >> 0) & 0xFF == 0x9F, f"entry 0 cfg: {cfg0:016x}"
        assert (cfg0 >> 24) & 0xFF == 0x0B, f"entry 3 cfg: {cfg0:016x}"
        # pmpcfg2 = entries 8-15: byte 0 = 0x18
        cfg2 = pmp._csrs["pmpcfg2"].val
        assert (cfg2 >> 0) & 0xFF == 0x18, f"entry 8 cfg: {cfg2:016x}"

    def test_sync_addr_to_entries(self):
        """_sync_from_flat 将 flat_addr 值写回 pmpaddr CSR."""
        pmp = self._make_pmp(8)
        pmp._flat_addr[0] = 0x8000_0000
        pmp._flat_addr[5] = 0xDEAD_BEEF_CAFE
        pmp._flat_addr[7] = 0xFFFF_FFFF_FFFF_FFFF
        pmp._sync_from_flat()

        assert pmp._csrs["pmpaddr0"].val == 0x8000_0000
        assert pmp._csrs["pmpaddr5"].val == 0xDEAD_BEEF_CAFE
        assert pmp._csrs["pmpaddr7"].val == 0xFFFF_FFFF_FFFF_FFFF

    def test_sync_clears_cache_dirty(self):
        """_sync_from_flat 完成后 _cache_dirty 应为 False."""
        pmp = self._make_pmp(4)
        pmp._cache_dirty = True
        pmp._sync_from_flat()
        assert pmp._cache_dirty is False

    def test_sync_roundtrip(self):
        """_sync_from_flat -> _rebuild_cache 往返应保持数据一致."""
        pmp = self._make_pmp(12)
        # Set via flat arrays
        for i in range(12):
            pmp._flat_cfg[i] = (i * 17 + 3) & 0xFF
            pmp._flat_addr[i] = 0x8000_0000 + i * 0x1000
        pmp._sync_from_flat()

        # Rebuild flat from entries
        pmp._rebuild_cache()

        for i in range(12):
            assert pmp._flat_cfg[i] == (i * 17 + 3) & 0xFF, f"cfg[{i}] mismatch"
            assert pmp._flat_addr[i] == 0x8000_0000 + i * 0x1000, f"addr[{i}] mismatch"
