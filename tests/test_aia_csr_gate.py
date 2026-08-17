#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""AIA CSR 门控测试 — 验证 PYREMU_AIA=0/1 时 8 个 AIA CSR 的可达性.

被门控的 CSR:
  miselect (0x350), mireg (0x351), mtopei (0x35C),  mtopi (0xFB0),
  siselect (0x150), sireg (0x151), stopei (0x15C),  stopi (0xDB0).

行为:
  PYREMU_AIA=0 → 全部触发 IllInstr.
  PYREMU_AIA=1 → 全部可正常访问 (IMSIC 接管进一步的读写语义).
"""

from __future__ import annotations

import pytest

from pyremu.configs_gen import PYREMU_AIA
from pyremu.core.decoder import Hart
from pyremu.core.hart import RiscvMode
from pyremu.core.mem_check_aux import inject_memory_backend
from pyremu.core.registers import _csr_bank, check_csr_access, CsrAccessError
from pyremu.memory.bus import Bus


# ============================================================
#  常量
# ============================================================

_AIA_CSRS: tuple[int, ...] = (
    0x150,  # siselect
    0x151,  # sireg
    0x15C,  # stopei
    0x350,  # miselect
    0x351,  # mireg
    0x35C,  # mtopei
    0xDB0,  # stopi
    0xFB0,  # mtopi
)

_AIA_NAMES: dict[int, str] = {
    0x150: "siselect",
    0x151: "sireg",
    0x15C: "stopei",
    0x350: "miselect",
    0x351: "mireg",
    0x35C: "mtopei",
    0xDB0: "stopi",
    0xFB0: "mtopi",
}


# ============================================================
#  check_csr_access 层 — 直接验证 CSR 权限
# ============================================================


class TestAiaCsrGateCheckAccess:
    """验证 check_csr_access 对 AIA CSR 的门控."""

    def test_implemented_flags_uniform(self) -> None:
        """全部 8 个 AIA CSR 的 implemented 标志必须一致."""
        states = {addr: _csr_bank[addr].implemented for addr in _AIA_CSRS}
        unique = set(states.values())
        assert len(unique) == 1, f"AIA CSR implemented flags must be uniform: {states}"

    def test_mmode_access_consistent(self) -> None:
        """M 模式: implemented=False → CsrAccessError, True → 通过.
        mtopi (0xFB0) / stopi (0xDB0) 为只读 CSR, 写操作应报 readonly."""
        M = RiscvMode.M.value
        for addr in _AIA_CSRS:
            if not _csr_bank[addr].implemented:
                with pytest.raises(CsrAccessError, match="unknown"):
                    check_csr_access(addr, M, is_write=False)
                with pytest.raises(CsrAccessError, match="unknown"):
                    check_csr_access(addr, M, is_write=True)
            else:
                check_csr_access(addr, M, is_write=False)
                if (_csr_bank[addr].access.value & 1) == 0:
                    # Read-only CSR (e.g. mtopi, stopi)
                    with pytest.raises(CsrAccessError, match="readonly"):
                        check_csr_access(addr, M, is_write=True)
                else:
                    check_csr_access(addr, M, is_write=True)

    def test_privilege_boundary_smode_cannot_access_mmode_aia(self) -> None:
        """S 模式访问 M-mode AIA CSR: 权限不足 (privilege) 或 未实现 (unknown)."""
        S = RiscvMode.S.value
        m_csrs = (0x350, 0x351, 0x35C, 0xFB0)
        for addr in m_csrs:
            with pytest.raises(CsrAccessError):
                check_csr_access(addr, S, is_write=False)

    def test_non_aia_csrs_unaffected(self) -> None:
        """健全检查: 非 AIA CSR 不受门控影响."""
        M = RiscvMode.M.value
        S = RiscvMode.S.value
        check_csr_access(0x300, M, is_write=False)  # mstatus
        check_csr_access(0x300, M, is_write=True)
        check_csr_access(0x105, S, is_write=False)  # stvec
        check_csr_access(0x105, S, is_write=True)


# ============================================================
#  指令执行层 — 通过 Hart.exec_instr 验证
# ============================================================


class TestAiaCsrGateInstruction:
    """通过执行 CSR 指令验证门控 (Python 指令解码路径)."""

    @pytest.fixture
    def h(self) -> Hart:
        bus = Bus(ram_size=0x10000, ram_base=0x8000_0000)
        hart = Hart(id=0)
        inject_memory_backend(hart, bus.read, bus.write)
        hart.pc = 0x8000_0000
        hart.mode = RiscvMode.M
        return hart

    @staticmethod
    def _csrrw(rd: int, csr: int, rs1: int) -> int:
        """CSRRW rd, csr, rs1 (funct3=1)."""
        return (
            ((csr & 0xFFF) << 20) | ((rs1 & 0x1F) << 15)
            | (1 << 12) | ((rd & 0x1F) << 7) | 0x73
        )

    @staticmethod
    def _csrrs(rd: int, csr: int, rs1: int) -> int:
        """CSRRS rd, csr, rs1 (funct3=2). rs1=0 → csrr pseudo-instruction."""
        return (
            ((csr & 0xFFF) << 20) | ((rs1 & 0x1F) << 15)
            | (2 << 12) | ((rd & 0x1F) << 7) | 0x73
        )

    # ---- write tests ----

    # mtopi / stopi are read-only even when AIA is enabled.
    _RDONLY_AIA = frozenset({0xFB0, 0xDB0})

    def test_aia_csr_write(self, h: Hart) -> None:
        """CSRRW x0, AIA_CSR, a0 — 根据 PYREMU_AIA 验证 IllInstr.
        mtopi/stopi 为只读 CSR, 写操作在 AIA=1 时也应报 IllInstr."""
        for addr in _AIA_CSRS:
            h.mcause_val = 0
            h.pc = 0x8000_0000
            instr = self._csrrw(0, addr, 10)
            h.exec_instr(instr)
            if PYREMU_AIA and addr not in self._RDONLY_AIA:
                assert h.mcause_val == 0, (
                    f"{_AIA_NAMES[addr]} (0x{addr:03X}): AIA=1 must not trap"
                )
            else:
                assert h.mcause_val == 2, (
                    f"{_AIA_NAMES[addr]} (0x{addr:03X}): must trigger IllInstr"
                    f" (AIA={PYREMU_AIA}, ro={addr in self._RDONLY_AIA})"
                )

    def test_aia_csr_read(self, h: Hart) -> None:
        """CSRRS t0, AIA_CSR, x0 (csrr) — 根据 PYREMU_AIA 验证 IllInstr."""
        for addr in _AIA_CSRS:
            h.mcause_val = 0
            h.pc = 0x8000_0000
            instr = self._csrrs(5, addr, 0)
            h.exec_instr(instr)
            if PYREMU_AIA:
                assert h.mcause_val == 0, (
                    f"{_AIA_NAMES[addr]} (0x{addr:03X}): AIA=1 must not trap"
                )
            else:
                assert h.mcause_val == 2, (
                    f"{_AIA_NAMES[addr]} (0x{addr:03X}): AIA=0 must trigger IllInstr"
                )

    # ---- probe pattern (OpenSBI csr_read_allowed) ----

    def test_mtopi_probe(self, h: Hart) -> None:
        """csrrs t0, mtopi, x0 — OpenSBI 的 csr_read_allowed(CSR_MTOPI) 探测.

        AIA=0 时必须触发 IllInstr (mcause=2), 使 sbi_trap_info.cause 置位,
        从而 imsic_cold_irqchip_init 提前返回且不触碰 miselect/mireg.
        """
        h.mcause_val = 0
        h.pc = 0x8000_0000
        instr = self._csrrs(5, 0xFB0, 0)
        h.exec_instr(instr)
        if PYREMU_AIA:
            assert h.mcause_val == 0, "AIA=1: mtopi probe must not trap"
        else:
            assert h.mcause_val == 2, "AIA=0: mtopi probe must trigger IllInstr"
            assert (h.mtval_val & 0xFFFF_FFFF) == instr, (
                "mtval must carry the faulting instruction"
            )
