#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""IPI delivery chain tests: cross-hart seteipnum → SEIP → stopi → claim → clear.

Verifies the complete IMSIC IPI path at the Python level (no native batch),
then at the Emulator integration level.  Native batch is disabled by default
via conftest.py

CSR read/write: the hart's read_csr_by_name()/write_csr_by_name() methods
route AIA CSRs through the Python IMSIC backend.  CSR addresses:
  stopi  = 0xDB0  (combined S-level view, no auto-claim on read)
  stopei = 0x15C  (IMSIC S-file only, auto-claim on read)
  mtopi  = 0xFB0  (combined M-level view, no auto-claim on read)
  mtopei = 0x35C  (IMSIC M-file only, auto-claim on read)
"""

import pytest

from pyremu.core.hart import RiscvMode
from pyremu.core.trap_handler import check_pending_interrupts
from pyremu.emulator import Emulator
from pyremu.interrupt.imsic import _IID_M_IPI, _IID_S_IPI, IMSIC
from pyremu.platform import PlatformConfig

# ============================================================
#  常量
# ============================================================

_IMSIC_M_BASE = 0x2400_0000
_PAGE_STRIDE = 0x1000
_EIPNUM_SET_OFF = 0x0000

# CSR addresses (RISC-V AIA spec)
_CSR_STOPEI = 0x15C
_CSR_MTOPEI = 0x35C
_CSR_MTOPI = 0xFB0
_CSR_STOPI = 0xDB0

# ============================================================
#  IMSIC 单元级: 完整的 IPI 生命周期 (与 Rust imsic.rs 逻辑一致)
# ============================================================


class TestIpiLifecycle:
    """Single-hart IPI lifecycle: set → peek → claim → clear."""

    def test_ipi_set_peek_claim_clear(self):
        """S-file seteipnum(1) → peek sees IID=1 → claim clears eip."""
        imsic = IMSIC(num_harts=2, m_base_addr=_IMSIC_M_BASE)
        s_base = 2 * _PAGE_STRIDE  # N=2, S-files after M-files
        imsic.write(s_base + _EIPNUM_SET_OFF, (1).to_bytes(4, "little"))

        sf = imsic._file_for(0, 'S')
        assert sf is not None
        assert sf.eip[0] & (1 << _IID_S_IPI), "S-file eip[0] bit 1 should be set"

        peek = imsic.peek_topei(0, 'S')
        assert peek != 0, f"peek_topei non-zero, got {peek}"
        assert ((peek >> 16) & 0x7FF) == _IID_S_IPI

        # Claim via read_topei — clears eip
        claim = imsic.read_topei(0, 'S')
        assert claim != 0
        assert imsic.peek_topei(0, 'S') == 0, "after claim, peek should be 0"

    def test_peek_does_not_claim(self):
        """peek_topei should NOT clear eip — idempotent."""
        imsic = IMSIC(num_harts=1, m_base_addr=_IMSIC_M_BASE)
        s_base = 1 * _PAGE_STRIDE
        imsic.write(s_base + _EIPNUM_SET_OFF, (1).to_bytes(4, "little"))

        p1 = imsic.peek_topei(0, 'S')
        p2 = imsic.peek_topei(0, 'S')
        assert p1 == p2, "peek should be idempotent (no claim)"
        assert p1 != 0

    def test_eidelivery_zero_ipi_visible(self):
        """With eidelivery=0, IPI should still be visible via peek_topei."""
        imsic = IMSIC(num_harts=1, m_base_addr=_IMSIC_M_BASE)
        s_base = 1 * _PAGE_STRIDE
        imsic.write(s_base + _EIPNUM_SET_OFF, (1).to_bytes(4, "little"))
        assert imsic.peek_topei(0, 'S') != 0
        assert imsic.get_pending_mip(0) & (1 << 9), "SEIP should be set"

    def test_mfile_ipi_stays_in_mfile(self):
        """M-file seteipnum + IID=1 → stays in M-file (no cross-file routing)."""
        imsic = IMSIC(num_harts=2, m_base_addr=_IMSIC_M_BASE)
        imsic.write(0 * _PAGE_STRIDE, (1).to_bytes(4, "little"))  # M-file, IID=1

        mf = imsic._file_for(0, 'M')
        assert mf is not None
        assert mf.eip[0] & (1 << _IID_S_IPI), "M-file eip[0] bit 1 should be set"
        assert imsic.peek_topei(0, 'M') != 0, "M-file should have pending IPI"
        assert imsic.peek_topei(0, 'S') == 0, "S-file should have no pending IPI"

    def test_external_interrupt_blocked_without_eidelivery(self):
        """External IID blocked when eidelivery=0; IPI IIDs still visible."""
        imsic = IMSIC(num_harts=1, m_base_addr=_IMSIC_M_BASE)
        s_base = 1 * _PAGE_STRIDE
        # External IID=10 with eidelivery=0
        imsic.write(s_base + _EIPNUM_SET_OFF, (10).to_bytes(4, "little"))
        # Should NOT be visible — peek_topei blocks external IIDs without eidelivery
        assert imsic.peek_topei(0, 'S') == 0, \
        "external IID blocked without eidelivery"
        assert imsic.get_pending_mip(0) == 0, \
        "external IID should not set SEIP with eidelivery=0"

        # But IPI (IID=1) should still be visible via IPI fast-path
        imsic.write(s_base + _EIPNUM_SET_OFF, (1).to_bytes(4, "little"))
        assert imsic.peek_topei(0, 'S') != 0, \
        "IPI visible even with eidelivery=0"
        assert imsic.get_pending_mip(0) & (1 << 9), \
        "IPI should set SEIP via peek_topei fast-path"


# ============================================================
#  跨 hart IPI (Python IMSIC 级别)
# ============================================================


class TestCrossHartIpi:
    """Cross-hart IPI delivery via IMSIC seteipnum."""

    def test_hart0_sends_ipi_to_hart1_sfile(self):
        """Write to hart 1's S-file seteipnum → hart 1 sees IPI, hart 0 does not."""
        imsic = IMSIC(num_harts=2, m_base_addr=_IMSIC_M_BASE)
        s_base = 2 * _PAGE_STRIDE
        target = s_base + 1 * _PAGE_STRIDE + _EIPNUM_SET_OFF
        imsic.write(target, (1).to_bytes(4, "little"))

        assert imsic.peek_topei(1, 'S') != 0, "Hart 1 sees IPI"
        assert imsic.peek_topei(0, 'S') == 0, "Hart 0 unaffected"
        assert imsic.get_pending_mip(1) & (1 << 9), "Hart 1 SEIP set"

    def test_mfile_ipi_stays_in_target_mfile(self):
        """M-file seteipnum for hart 1 + IID=1 stays in hart 1's M-file."""
        imsic = IMSIC(num_harts=2, m_base_addr=_IMSIC_M_BASE)
        target = 1 * _PAGE_STRIDE + _EIPNUM_SET_OFF  # hart 1 M-file
        imsic.write(target, (1).to_bytes(4, "little"))

        mf_h1 = imsic._file_for(1, 'M')
        assert mf_h1 is not None
        assert mf_h1.eip[0] & (1 << _IID_S_IPI)
        assert imsic.get_pending_mip(1) & (1 << 11), "hart 1 MEIP should be set"

    def test_set_ip_number_all_harts_independent(self):
        """set_ip_number to hart 2 leaves harts 0,1,3 unaffected."""
        imsic = IMSIC(num_harts=4, m_base_addr=_IMSIC_M_BASE)
        imsic.set_ip_number(2, 'S', _IID_S_IPI)
        assert imsic.peek_topei(2, 'S') != 0
        for h in (0, 1, 3):
            assert imsic.peek_topei(h, 'S') == 0, f"Hart {h} unaffected"


# ============================================================
#  CSR 级别: hart.read_csr_by_name / write_csr_by_name
# ============================================================


class TestIpiCsrLevel:
    """Integration: hart CSR path for stopi / stopei / mtopi / mtopei.

    Native batch disabled by conftest.py → CSR ops stay in Python.
    """

    @staticmethod
    def _make_aia_emu(num_harts=2):
        cfg = PlatformConfig.qemu_virt_aia()
        cfg.num_harts = num_harts
        return Emulator(cfg)

    # ---- stopi (0xDB0, S-mode combined view) ----

    def test_stopi_read_after_ipi(self):
        """stopi (0xDB0) read → IID=9 (SEI); write stopi to claim → next read returns 0."""
        emu = self._make_aia_emu(num_harts=2)
        imsic = emu.imsic
        h1 = emu.harts[1]
        h1.mode = RiscvMode.S
        assert imsic
        imsic.csr_write(1, 'S', 0x70, 1)  # eidelivery=1
        imsic.set_ip_number(1, 'S', _IID_S_IPI)

        val = h1.read_csr(_CSR_STOPI)
        assert val != 0, f"stopi after IPI: expected non-zero, got {val}"
        iid = (val >> 16) & 0x7FF
        assert iid == 9, f"stopi IID: expected 9 (SEI), got {iid}"

        # Claim
        h1.write_csr(_CSR_STOPI, val)
        val2 = h1.read_csr(_CSR_STOPI)
        assert val2 == 0, f"stopi after claim: expected 0, got {val2}"

    def test_stopi_ipi_major_identity_is_sei_regression(self):
        """Regression: S-file IPI (minor 1) → stopi returns major identity 9 (SEI).

        Linux 的 riscv_intc_aia_irq() 将 stopi>>16 分发给 intc domain。
        若 stopi 对 S-file IPI 返回 1 (SSI) 而非 9 (SEI)，IMSIC 模式下
        SSI 没有 handler，IPI 丢失，发送方 hart 卡死在
        smp_call_function_many_cond 等待目标 hart 应答。
        """
        emu = self._make_aia_emu(num_harts=2)
        imsic = emu.imsic
        h1 = emu.harts[1]
        h1.mode = RiscvMode.S
        assert imsic
        imsic.csr_write(1, 'S', 0x70, 1)  # eidelivery=1
        imsic.set_ip_number(1, 'S', _IID_S_IPI)

        # stopi must report the MAJOR identity (SEI=9), not the minor (1).
        stopi = h1.read_csr(_CSR_STOPI)
        assert ((stopi >> 16) & 0x7FF) == 9, (
            f"stopi>>16 must be SEI (9) for the S-file IPI, got {(stopi >> 16) & 0x7FF}"
        )

        # stopei must still reveal the MINOR identity (1 = IPI).
        stopei = h1.read_csr(_CSR_STOPEI)
        assert ((stopei >> 16) & 0x7FF) == _IID_S_IPI, (
            f"stopei>>16 must be the minor IPI identity (1), got {(stopei >> 16) & 0x7FF}"
        )

    def test_stopi_read_idempotent_no_auto_claim(self):
        """stopi read does NOT auto-claim — same IID returned on repeated reads."""
        emu = self._make_aia_emu(num_harts=1)
        imsic = emu.imsic
        h = emu.harts[0]
        h.mode = RiscvMode.S
        assert imsic
        imsic.csr_write(0, 'S', 0x70, 1)
        imsic.set_ip_number(0, 'S', _IID_S_IPI)

        v1 = h.read_csr(_CSR_STOPI)
        v2 = h.read_csr(_CSR_STOPI)
        assert v1 == v2, "stopi read should be idempotent"
        assert v1 != 0

    def test_stopi_zero_when_no_interrupts(self):
        """stopi returns 0 when no interrupts are pending."""
        emu = self._make_aia_emu(num_harts=1)
        h = emu.harts[0]
        h.mode = RiscvMode.S
        h.mip_val = 0
        h.csrs["mie"].val = 0

        val = h.read_csr(_CSR_STOPI)
        assert val == 0, f"stopi with no interrupts should be 0, got {val}"

    # ---- stopei (0x15C, IMSIC S-file only, auto-claim on read) ----

    def test_stopei_read_auto_claims(self):
        """stopei (0x15C) read auto-claims: second read returns 0."""
        emu = self._make_aia_emu(num_harts=1)
        imsic = emu.imsic
        h = emu.harts[0]
        h.mode = RiscvMode.S
        assert imsic
        imsic.csr_write(0, 'S', 0x70, 1)  # eidelivery=1
        imsic.set_ip_number(0, 'S', _IID_S_IPI)

        v1 = h.read_csr(_CSR_STOPEI)
        assert v1 != 0, f"first stopei: expected non-zero, got {v1}"
        v2 = h.read_csr(_CSR_STOPEI)
        assert v2 == 0, f"second stopei (auto-claimed): expected 0, got {v2}"

    # ---- mtopi (0xFB0, M-mode combined view) ----

    def test_mtopi_read_after_mfile_ipi(self):
        """mtopi (0xFB0) sees IID=11 (MEI) for M-mode IPI (minor 3)."""
        emu = self._make_aia_emu(num_harts=1)
        imsic = emu.imsic
        h = emu.harts[0]
        h.mode = RiscvMode.M
        assert imsic
        imsic.csr_write(0, 'M', 0x70, 1)  # eidelivery=1
        imsic.set_ip_number(0, 'M', _IID_M_IPI)

        val = h.read_csr(_CSR_MTOPI)
        iid = (val >> 16) & 0x7FF
        assert iid == 11, f"mtopi: expected IID=11 (MEI), got {iid}"

    # ---- mtopei (0x35C, IMSIC M-file only, auto-claim on read) ----

    def test_mtopei_read_auto_claims(self):
        """mtopei (0x35C) read auto-claims IMSIC M-file."""
        emu = self._make_aia_emu(num_harts=1)
        imsic = emu.imsic
        h = emu.harts[0]
        h.mode = RiscvMode.M

        assert imsic
        imsic.csr_write(0, 'M', 0x70, 1)
        imsic.set_ip_number(0, 'M', _IID_M_IPI)
        v1 = h.read_csr(_CSR_MTOPEI)
        assert v1 != 0
        v2 = h.read_csr(_CSR_MTOPEI)
        assert v2 == 0, "mtopei auto-claim: second read should be 0"

    # ---- Integration: IPI → trap ----

    def test_ipi_raises_seip_trap(self):
        """set_ip_number → check_pending_interrupts fires for target hart."""

        emu = self._make_aia_emu(num_harts=2)
        imsic = emu.imsic
        h1 = emu.harts[1]
        h1.mode = RiscvMode.S

        # mie CSR (interrupt enable), not mstatus.MIE (global enable)
        h1.csrs["mie"].val = 1 << 9  # SEIE
        assert h1.mie_val & (1 << 9), "SEIE should be set in mie CSR"
        assert imsic
        imsic.set_ip_number(1, 'S', _IID_S_IPI)

        # check_pending_interrupts returns True if a trap was delivered
        result = check_pending_interrupts(h1)
        assert result, "check_pending_interrupts should deliver SEI trap"

        # After delivery, mcause should reflect SEI
        mcause = h1.csrs["mcause"].val
        assert (mcause & 0x7FFFFFFFFFFFFFFF) == 9, \
            f"mcause should be 9 (SEI), got {mcause}"


# ============================================================
#  CSR 状态一致性 (eidelivery / eie / eip 通过 CSR 读写)
# ============================================================


class TestCsrStateConsistency:
    """IMSIC register readback through CSR indirect access."""

    def test_eidelivery_roundtrip(self):
        """miselect=0x70 → mireg write 1 → mireg read 1."""
        imsic = IMSIC(num_harts=1, m_base_addr=_IMSIC_M_BASE)
        imsic.csr_write(0, 'M', 0x70, 1)
        assert imsic.csr_read(0, 'M', 0x70) == 1

        imsic.csr_write(0, 'S', 0x70, 1)
        assert imsic.csr_read(0, 'S', 0x70) == 1

    def test_eie_roundtrip_through_csr(self):
        """miselect=0xC0 → mireg write → mireg read."""
        imsic = IMSIC(num_harts=1, m_base_addr=_IMSIC_M_BASE)
        imsic.csr_write(0, 'M', 0xC0, 0xDEAD)
        assert imsic.csr_read(0, 'M', 0xC0) == 0xDEAD

    def test_eip_visible_through_csr_after_seteipnum(self):
        """seteipnum → eip visible via mireg read."""
        imsic = IMSIC(num_harts=1, m_base_addr=_IMSIC_M_BASE)
        imsic.csr_write(0, 'M', 0x70, 1)  # eidelivery=1
        imsic.set_ip_number(0, 'M', 3)

        val = imsic.csr_read(0, 'M', 0x80)  # eip[0]
        assert val & (1 << 3), f"eip bit 3 should be visible, got {val:#x}"

    def test_eie_controls_external_interrupt_visibility(self):
        """External interrupt needs eie: topei returns 0 when eie bit is clear."""
        imsic = IMSIC(num_harts=1, m_base_addr=_IMSIC_M_BASE)
        imsic.csr_write(0, 'M', 0x70, 1)  # eidelivery=1
        imsic.set_ip_number(0, 'M', 10)   # external IID=10

        # eie[0] bit 10 is clear → peek_topei should return 0
        assert imsic.peek_topei(0, 'M') == 0, "external IID blocked without eie"

        # Set eie for bit 10
        imsic.csr_write(0, 'M', 0xC0, 1 << 10)
        assert imsic.peek_topei(0, 'M') != 0, "external IID visible with eie set"


# ============================================================
#  DTB 布局验证: IMSIC 地址解码一致性
# ============================================================


class TestImsicAddressing:
    """IMSIC MMIO 地址解码: DTB reg 与 decode_imsic_addr 一致."""

    def test_mfile_offset_maps_to_correct_hart(self):
        """M-file offset = hart_id * 0x1000 → correct (hart_id, M-file)."""
        imsic = IMSIC(num_harts=4, m_base_addr=_IMSIC_M_BASE)
        # _resolve maps MMIO offset → (hart_id, file, file_off)
        for h in range(4):
            off = h * _PAGE_STRIDE
            resolved = imsic._resolve(off)
            assert resolved is not None
            hart_id, file, file_off = resolved
            assert hart_id == h, f"offset {off:#x} → hart {hart_id}, expected {h}"
            assert file_off == 0, f"offset {off:#x} → file_off {file_off}, expected 0"

    def test_sfile_offset_maps_to_correct_hart(self):
        """S-file offset = N*0x1000 + hart_id * 0x1000 → correct (hart_id, S-file)."""
        N = 4
        imsic = IMSIC(num_harts=N, m_base_addr=_IMSIC_M_BASE)
        for h in range(N):
            off = N * _PAGE_STRIDE + h * _PAGE_STRIDE
            resolved = imsic._resolve(off)
            assert resolved is not None
            hart_id, file, file_off = resolved
            assert hart_id == h, f"S-file offset {off:#x} → hart {hart_id}, expected {h}"
            assert file_off == 0
