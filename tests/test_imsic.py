#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""IMSIC (Incoming MSI Controller) 单元测试 — Phase 1 AIA 实现."""

import pytest

from pyremu.core.hart import RiscvMode
from pyremu.core.registers import PYREMU_AIA
from pyremu.core.trap_def import trap_cause_code, TrapType
from pyremu.core.trap_handler import check_pending_interrupts
import pyremu.configs_gen
from pyremu.emulator import Emulator
from pyremu.interrupt.imsic import IMSIC
from pyremu.platform import InterruptMode, PlatformConfig


@pytest.fixture(autouse=True)
def _disable_aia_compile_override(monkeypatch):
    """Disable PYREMU_AIA compile-time override.

    Tests use explicit configs: qemu_virt() for legacy, qemu_virt_aia() for AIA.
    """
    monkeypatch.setattr(pyremu.configs_gen, "PYREMU_AIA", False)


# ============================================================
#  辅助工厂
# ============================================================


def _make_imsic(num_harts: int = 2) -> IMSIC:
    return IMSIC(num_harts=num_harts, m_base_addr=0x2400_0000)


def _make_aia_emu(num_harts: int = 1):
    """创建 AIA 模式 Emulator."""
    cfg = PlatformConfig.qemu_virt_aia()
    cfg.num_harts = num_harts
    return Emulator(cfg)


# ============================================================
#  MMIO 基本: seteipnum / clreipnum
# ============================================================


class TestImsicMMIO:
    """IMSIC MMIO (seteipnum / clreipnum) 寄存器."""

    def test_seteipnum_sets_pending_and_meip(self):
        """seteipnum 写 -> eip 置位 -> get_pending_mip 返回 MEIP (eidelivery=1, eie 已使能)."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)  # eidelivery=1
        imsic.csr_write(0, 'M', 0xC0, 1 << 10)  # eie: enable identity 10
        imsic.write(0x000, (10).to_bytes(4, "little"))
        mip = imsic.get_pending_mip(0)
        assert mip & (1 << 11), f"MEIP should be set, got mip={mip:#x}"

    def test_seteipnum_hart_isolation(self):
        """hart 0 seteipnum 不影响 hart 1."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)
        imsic.csr_write(0, 'M', 0xC0, 1 << 10)
        imsic.csr_write(1, 'M', 0x70, 1)
        imsic.csr_write(1, 'M', 0xC0, 1 << 10)
        imsic.write(0x000, (10).to_bytes(4, "little"))  # hart 0
        assert imsic.get_pending_mip(0) & (1 << 11)
        assert imsic.get_pending_mip(1) == 0, "hart 1 should be unaffected"

    def test_clreipnum_clears_pending(self):
        """seteipnum -> clreipnum -> pending 清除."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)
        imsic.csr_write(0, 'M', 0xC0, 1 << 10)
        imsic.write(0x000, (10).to_bytes(4, "little"))
        assert imsic.get_pending_mip(0) & (1 << 11)
        imsic.write(0x008, (10).to_bytes(4, "little"))  # clreipnum
        assert imsic.get_pending_mip(0) == 0, "pending should be cleared"

    def test_seteipnum_ignores_invalid_id(self):
        """seteipnum(0) 或 seteipnum(>=2048) 被忽略."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)
        imsic.write(0x000, (0).to_bytes(4, "little"))      # identity 0 = reserved
        imsic.write(0x000, (2048).to_bytes(4, "little"))   # out of range
        assert imsic.get_pending_mip(0) == 0

    def test_sfile_seteipnum_triggers_seip(self):
        """S-mode file seteipnum -> SEIP (bit 9)."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'S', 0x70, 1)  # S-file eidelivery=1
        imsic.csr_write(0, 'S', 0xC0, 1 << 5)  # eie: enable identity 5
        # S-file hart 0: num_harts * 0x1000 + 0 * 0x1000 = 0x2000
        imsic.write(0x2000, (5).to_bytes(4, "little"))  # hart 0 S-file seteipnum
        mip = imsic.get_pending_mip(0)
        assert mip & (1 << 9), f"SEIP should be set, got mip={mip:#x}"
        assert not (mip & (1 << 11)), "MEIP should not be set"


# ============================================================
#  CSR 间接访问: miselect/mireg, siselect/sireg
# ============================================================


class TestImsicCSR:
    """AIA CSR 间接访问 (miselect→mireg)."""

    def test_eidelivery_roundtrip(self):
        """miselect=0x70 -> mireg 读写 eidelivery."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)
        assert imsic.csr_read(0, 'M', 0x70) == 1
        imsic.csr_write(0, 'M', 0x70, 0)
        assert imsic.csr_read(0, 'M', 0x70) == 0

    def test_eithreshold_roundtrip(self):
        """miselect=0x72 -> mireg 读写 eithreshold."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x72, 0xFF)
        assert imsic.csr_read(0, 'M', 0x72) == 0xFF
        # threshold clamped to 10 bits
        imsic.csr_write(0, 'M', 0x72, 0x3FF)
        assert imsic.csr_read(0, 'M', 0x72) == 0x3FF

    def test_eip_visible_via_mireg(self):
        """seteipnum 写后 eip 可通过 mireg (select 0x80+) 读取."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)
        # 注入中断号 10: word 0, bit 10
        imsic.set_ip_number(0, 'M', 10)
        # 读 eip0: select = 0x80, 应含 bit 10
        val = imsic.csr_read(0, 'M', 0x80)
        assert val & (1 << 10), f"eip[0] bit 10 should be set, got {val:#x}"

    def test_eie_visible_via_mireg(self):
        """eie 可通过 mireg (select 0xC0+) 读写."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0xC0, (1 << 15))
        val = imsic.csr_read(0, 'M', 0xC0)
        assert val & (1 << 15)

    def test_siselect_sireg(self):
        """S-mode siselect/sireg 独立于 M-mode."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)
        imsic.csr_write(0, 'S', 0x70, 0)
        assert imsic.csr_read(0, 'M', 0x70) == 1
        assert imsic.csr_read(0, 'S', 0x70) == 0

    def test_mireg_64bit_pairs_u32(self):
        """mireg 读 eip 返回 64 位 (相邻两 u32 拼装)."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)
        # 设 eip[0] bit 10 (IID 10) 和 eip[1] bit 31 (IID 63)
        imsic.set_ip_number(0, 'M', 10)   # word 0, bit 10
        imsic.set_ip_number(0, 'M', 63)   # word 1, bit 31
        val = imsic.csr_read(0, 'M', 0x80)  # select eip0 -> eip[0] | (eip[1] << 32)
        assert val & (1 << 10), "eip[0] bit 10 (IID 10) should be set"
        assert val & (1 << 63), "eip[1] bit 31 (IID 63) → mireg bit 63 should be set"

    def test_unknown_select_reads_zero(self):
        """未注册的 select 读返回 0."""
        imsic = _make_imsic()
        assert imsic.csr_read(0, 'M', 0x00) == 0
        assert imsic.csr_read(0, 'M', 0xFF) == 0


# ============================================================
#  优先级与阈值
# ============================================================


class TestImsicPriority:
    """IMSIC 优先级阈值 (eithreshold) 与多中断仲裁."""

    def test_eithreshold_filters_low_priority(self):
        """eithreshold=5: 优先级 ≤5 的中断不触发 pending."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)
        imsic.csr_write(0, 'M', 0x72, 5)  # threshold=5
        # IID=5 has priority 5, NOT > threshold 5 → filtered by topei
        imsic.csr_write(0, 'M', 0xC0, 1 << 5)  # enable identity 5
        imsic.set_ip_number(0, 'M', 5)
        topei = imsic.read_topei(0, 'M')
        assert topei == 0, f"topei should be 0 (filtered by eithreshold), got {topei:#x}"

    def test_get_pending_mip_respects_eithreshold(self):
        """get_pending_mip returns 0 when the only pending IMSIC
        interrupt is masked by eithreshold — consistent with peek_topei.

        Regression test: sync_imsic (Rust) and get_pending_mip (Python)
        previously used a raw eip&eie scan (imsic_has_any / has_pending)
        that ignored eithreshold.  compute_stopi / _read_stopi used the
        threshold-aware topei_peek.  SEIP=1 + stopi→IID=5(timer) caused
        an infinite SEI→timer→SEI loop in riscv_intc_aia_irq.
        """
        imsic = _make_imsic()
        # S-file: eidelivery=1, threshold=20
        imsic.csr_write(0, 'S', 0x70, 1)
        imsic.csr_write(0, 'S', 0x72, 20)
        # Enable & pend IID=10 (prio=10 ≤ threshold=20)
        imsic.csr_write(0, 'S', 0xC0, 1 << 10)
        imsic.set_ip_number(0, 'S', 10)

        assert imsic.get_pending_mip(0) == 0, (
            "get_pending_mip must be 0 when interrupt is masked by eithreshold"
        )

        # Lower threshold below priority → SEIP should appear
        imsic.csr_write(0, 'S', 0x72, 5)
        assert imsic.get_pending_mip(0) & (1 << 9), (
            "SEIP must be set after lowering eithreshold below priority"
        )

    def test_eidelivery_zero_suppresses_all(self):
        """eidelivery=0 -> get_pending_mip 返回 0, topei 返回 0."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0xC0, 1 << 10)
        imsic.set_ip_number(0, 'M', 10)
        assert imsic.get_pending_mip(0) == 0
        assert imsic.read_topei(0, 'M') == 0

    def test_topei_returns_highest_priority(self):
        """多个 pending: topei 返回最高 IID (即最高优先级)."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)
        # 使能 identity 5 和 100
        imsic.csr_write(0, 'M', 0xC0, (1 << 5) | (1 << 4))  # eie[0] = bits 4, 5
        # 需要使能 word 3 bit 4 for IID 100
        w100 = 100 // 32       # = 3
        b100 = 100 % 32       # = 4
        imsic.csr_write(0, 'M', 0xC0 + w100, 1 << b100)
        imsic.set_ip_number(0, 'M', 5)
        imsic.set_ip_number(0, 'M', 100)
        topei = imsic.read_topei(0, 'M')
        iid = (topei >> 16) & 0x7FF
        assert iid == 100, f"topei should return highest IID (100), got {iid}"

    def test_topei_claims_clears_pending(self):
        """topei 读后 pending 位被清除 (claim 语义)."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)
        imsic.csr_write(0, 'M', 0xC0, 1 << 10)
        imsic.set_ip_number(0, 'M', 10)
        assert imsic.read_topei(0, 'M') != 0, "first topei should return pending"
        assert imsic.read_topei(0, 'M') == 0, "second topei should return 0 (claimed)"


# ============================================================
#  topei CSR: mtopei / stopei
# ============================================================


class TestImsicTopei:
    """mtopei (0x35C) / stopei (0x15C) CSR 读取."""

    def test_mtopei_with_pending(self):
        """有 pending 时 mtopei 返回 (IID<<16)|prio."""
        emu = _make_aia_emu()
        h = emu.harts[0]
        imsic = emu.imsic
        assert imsic is not None
        # 使能 + 注入
        h.write_csr(0x350, 0xC0)     # miselect = eie0
        h.write_csr(0x351, 1 << 15)  # mireg: enable identity 15
        h.write_csr(0x350, 0x70)     # miselect = eidelivery
        h.write_csr(0x351, 1)        # mireg: eidelivery = 1
        imsic.set_ip_number(0, 'M', 15)
        val = h.read_csr(0x35C)      # mtopei
        iid = (val >> 16) & 0x7FF
        assert iid == 15, f"mtopei should report IID 15, got {iid}"

    def test_mtopei_no_pending(self):
        """无 pending 时 mtopei 返回 0."""
        emu = _make_aia_emu()
        h = emu.harts[0]
        assert h.read_csr(0x35C) == 0

    def test_stopei_with_pending(self):
        """S-mode 有 pending 时 stopei 返回 IID."""
        emu = _make_aia_emu()
        h = emu.harts[0]
        imsic = emu.imsic
        assert imsic is not None
        imsic.csr_write(0, 'S', 0x70, 1)   # S-file eidelivery
        imsic.csr_write(0, 'S', 0xC0, 1 << 7)  # enable IID 7
        imsic.set_ip_number(0, 'S', 7)
        val = h.read_csr(0x15C)             # stopei
        iid = (val >> 16) & 0x7FF
        assert iid == 7, f"stopei should report IID 7, got {iid}"


# ============================================================
# ============================================================
#  mtopi (0xFB0): Machine Top Interrupt
# ============================================================


class TestMtopi:
    """mtopi (0xFB0) AIA 模式下动态报告最高优先级 M 模式中断."""

    def test_mtopi_readable_in_aia_mode(self):
        """AIA 模式下 mtopi 可读且无中断时返回 0."""

        emu = _make_aia_emu()
        h = emu.harts[0]
        h.pc = 0x40000000

        if not PYREMU_AIA:
            # Legacy 模式: mtopi 返回 IllInstr

            instr = (0xFB0 << 20) | (5 << 7) | 0x73  # csrr t0, mtopi
            h.exec_instr(instr)
            assert h.mcause_val == trap_cause_code(TrapType.IllInstr), (
                f"mtopi should trap IllInstr in legacy mode, got mcause={h.mcause_val}"
            )
        else:
            # AIA 模式: mtopi 可读, 无中断时返回 0
            val = h.read_csr(0xFB0)
            assert val == 0, f"mtopi should return 0 when no interrupt pending, got {val:#x}"

    def test_mtopi_reports_msip_as_iid_3(self):
        """MSIP 待处理时 mtopi 返回 (IRQ_M_SOFT<<16)|1 = (3<<16)|1.

        OpenSBI 的 sbi_trap_aia_irq() 将 mtopi>>16 与 IRQ_M_SOFT(=3) 比较
        来决定是否调用 sbi_ipi_process(). 此测试锁死该行为.
        """
        if not PYREMU_AIA:
            pytest.skip("mtopi is readable only when PYREMU_AIA=1")

        emu = _make_aia_emu(num_harts=2)
        h = emu.harts[0]
        h.pc = 0x40000000
        h.mode = RiscvMode.M  # M-mode (required to read mtopi)
        # Enable MSIP in mie
        h.csrs["mie"].val |= 1 << 3  # MSIE

        # Inject MSIP: set mip.MSIP directly (and CLINT level byte)
        h.mip_val |= 1 << 3
        emu.clint._msip[0] |= 1

        assert h.mip_val & (1 << 3), "MSIP should be pending"

        # Read mtopi → should return (IRQ_M_SOFT << 16) | priority
        mtopi = h.read_csr(0xFB0)
        iid = (mtopi >> 16) & 0xFFFF
        assert iid == 3, (
            f"sbi_trap_aia_irq() expects mtopi>>16 == IRQ_M_SOFT (3), "
            f"got IID={iid} (mtopi={mtopi:#x})"
        )
        assert mtopi & 0xFFFF, f"mtopi priority should be non-zero, got {mtopi:#x}"

    def test_mtopi_reports_imsic_ipi_as_iid_11(self):
        """IMSIC M-file eip 有 IPI (minor identity 1) 时 mtopi 返回 major identity 11 (MEI).

        All IMSIC interrupts (IPI minor identity 1 and external IID>=6)
        drive MEIP/SEIP per AIA spec.  A ``seteipnum = 1`` write to the M-file
        stays in the M-file (no cross-file routing) and drives MEIP.  mtopi
        returns the MAJOR identity (11=MEI); the MINOR identity (1) is read via
        mtopei.  OpenSBI's ``sbi_trap_aia_irq`` sees mtopi>>16 == 11, then reads
        mtopei>>16 == 1 to dispatch ``sbi_ipi_process``.
        """

        if not PYREMU_AIA:
            pytest.skip("mtopi is readable only when PYREMU_AIA=1")

        emu = _make_aia_emu(num_harts=2)
        h = emu.harts[0]
        h.pc = 0x40000000
        h.mode = RiscvMode.M

        # Enable IMSIC eidelivery for M-file
        h.write_csr(0x350, 0x70)   # miselect ← eidelivery
        h.write_csr(0x351, 1)      # mireg ← 1 (enable delivery)
        h.write_csr(0x350, 0xC0)   # miselect ← eie[0]
        h.write_csr(0x351, 1 << 1) # mireg ← enable identity 1

        # Inject IMSIC interrupt: seteipnum identity 1 via M-file MMIO
        imsic = h._imsic
        assert imsic is not None, "IMSIC must be present in AIA mode"
        imsic.write(0x000, (1).to_bytes(4, 'little'))  # hart 0 M-file seteipnum

        # M-file + IID=1 stays in M-file (no cross-file routing) → MEIP, not SEIP.
        imsic_mip = imsic.get_pending_mip(h.id)
        assert (imsic_mip & (1 << 11)) != 0, (
            f"MEIP should be set when M-file eip[1] is set, got mip={imsic_mip:#x}"
        )
        assert (imsic_mip & (1 << 9)) == 0, (
            f"SEIP must NOT be set (no cross-file routing), got mip={imsic_mip:#x}"
        )

        # M-mode mtopi sees major identity 11 (MEI).
        mtopi = h.read_csr(0xFB0)
        iid = (mtopi >> 16) & 0xFFFF
        assert iid == 11, (
            f"mtopi>>16 should be IID=11 (MEI) for the M-file IPI, "
            f"got IID={iid} (mtopi={mtopi:#x})"
        )

        # M-mode mtopei sees minor identity 1 (OpenSBI IMSIC_IPI_ID).
        mtopei = h.read_csr(0x35C)
        minor = (mtopei >> 16) & 0xFFFF
        assert minor == 1, (
            f"mtopei>>16 should be minor identity 1, "
            f"got minor={minor} (mtopei={mtopei:#x})"
        )


# ============================================================
#  stopi (0xDB0): Supervisor Top Interrupt
# ============================================================


class TestStopi:
    """stopi (0xDB0) AIA 模式下动态报告最高优先级 S 模式中断."""

    def test_stopi_readable_in_aia_mode(self):
        """AIA 模式下 stopi 可读且无中断时返回 0."""

        emu = _make_aia_emu()
        h = emu.harts[0]
        h.pc = 0x40000000

        if not PYREMU_AIA:
            # Legacy 模式: stopi 返回 IllInstr
            instr = (0xDB0 << 20) | (5 << 7) | 0x73  # csrr t0, stopi
            h.exec_instr(instr)
            assert h.mcause_val == trap_cause_code(TrapType.IllInstr), (
                f"stopi should trap IllInstr in legacy mode, got mcause={h.mcause_val}"
            )
        else:
            # AIA 模式: stopi 可读, 无中断时返回 0
            val = h.read_csr(0xDB0)
            assert val == 0, f"stopi should return 0 when no interrupt pending, got {val:#x}"

    def test_stopi_reports_ssip_as_iid_1(self):
        """SSIP 待处理时 stopi 返回 (IRQ_S_SOFT<<16)|1 = (1<<16)|1.

        Linux 内核的 riscv_intc_aia_irq() 将 stopi>>16 与 IID 比较
        来决定调用哪个中断 handler. 此测试锁死该行为.
        """

        if not PYREMU_AIA:
            pytest.skip("stopi is readable only when PYREMU_AIA=1")

        emu = _make_aia_emu(num_harts=2)
        h = emu.harts[0]
        h.pc = 0x40000000
        h.mode = RiscvMode.S  # S-mode (stopi is an S-mode CSR)
        # Enable SSIP in mie
        h.csrs["mie"].val |= 1 << 1  # SSIE

        # Inject SSIP
        h.mip_val |= 1 << 1

        assert h.mip_val & (1 << 1), "SSIP should be pending"

        # Read stopi → should return (IRQ_S_SOFT << 16) | priority
        stopi = h.read_csr(0xDB0)
        iid = (stopi >> 16) & 0xFFFF
        assert iid == 1, (
            f"riscv_intc_aia_irq() expects stopi>>16 == IRQ_S_SOFT (1), "
            f"got IID={iid} (stopi={stopi:#x})"
        )
        assert stopi & 0xFFFF, f"stopi priority should be non-zero, got {stopi:#x}"

    def test_stopi_reports_imsic_ext_as_iid_9(self):
        """IMSIC S-file eip 有待处理中断时 stopi 返回 (IRQ_S_EXT << 16) | prio.

        所有外部中断在 stopi 中映射到 major identity IRQ_S_EXT (9),
        内核随后通过 STOPEI CSR 读取 minor identity (具体设备 IID).
        这是 AIA 规范 §5.3 的要求: stopi 返回 MAJOR identity, stopei 返回 MINOR.
        """

        if not PYREMU_AIA:
            pytest.skip("stopi is readable only when PYREMU_AIA=1")

        emu = _make_aia_emu(num_harts=2)
        h = emu.harts[0]
        h.pc = 0x40000000
        h.mode = RiscvMode.S

        # Enable IMSIC eidelivery + eie for IID=21 (external device interrupt)
        h.write_csr(0x150, 0x70)  # siselect ← eidelivery
        h.write_csr(0x151, 1)     # sireg ← 1 (enable delivery)
        h.write_csr(0x150, 0xC0)  # siselect ← eie[0]
        h.write_csr(0x151, 1 << 21)  # sireg ← enable identity 21

        # Inject IMSIC S-mode external interrupt via S-file seteipnum
        imsic = h._imsic
        assert imsic is not None, "IMSIC must be present in AIA mode"
        imsic.write(0x2000, (21).to_bytes(4, 'little'))  # hart 0 S-file seteipnum

        # Sync IMSIC → mip bits
        imsic_mip = imsic.get_pending_mip(h.id)
        assert imsic_mip & (1 << 9), (
            f"SEIP should be set by IMSIC S-file interrupt, got mip={imsic_mip:#x}"
        )

        # Read stopi → should return MAJOR identity IID=9 (IRQ_S_EXT).
        # Per AIA spec §5.3, stopi reports the major identity; the minor
        # identity (actual device IID=21) is read from STOPEI (0x15C).
        stopi = h.read_csr(0xDB0)
        iid = (stopi >> 16) & 0xFFFF
        assert iid == 9, (
            f"stopi>>16 should be major identity IID=9 (IRQ_S_EXT), "
            f"got IID={iid} (stopi={stopi:#x})"
        )
        assert stopi & 0xFFFF, f"stopi priority should be non-zero, got {stopi:#x}"

        # Verify the minor IID is retrievable from STOPEI
        stopei = h.read_csr(0x15C)
        minor_iid = (stopei >> 16) & 0xFFFF
        assert minor_iid == 21, (
            f"STOPEI>>16 should be minor IID 21, got {minor_iid} (stopei={stopei:#x})"
        )

    def test_stopi_reports_stip_as_iid_5(self):
        """STIP pending → stopi reports IID=5 (IRQ_S_TIMER).

        STIP is computed from LIVE mtime & stimecmp (not cached mip_val),
        matching the Rust batch engine's sync_mtip behaviour.  Using live
        values avoids the Python-side bug where mip_val.STIP could be stale
        after the kernel writes a new stimecmp, causing stopi to never
        return 0 (infinite loop in riscv_intc_aia_irq).
        """
        if not PYREMU_AIA:
            pytest.skip("stopi is readable only when PYREMU_AIA=1")

        emu = _make_aia_emu(num_harts=2)
        h = emu.harts[0]
        h.pc = 0x40000000
        h.mode = RiscvMode.S
        h.csrs["mie"].val |= 1 << 5   # STIE

        # Set stimecmp in the past → hardware STIP = 1
        h.csrs["stimecmp"].val = 100
        assert h.interrupt_ctrl, "empty aia interrupt controller"
        h.interrupt_ctrl.tick(200)

        # Read stopi → should return (IRQ_S_TIMER << 16) | priority
        stopi = h.read_csr(0xDB0)
        iid = (stopi >> 16) & 0xFFFF
        assert iid == 5, (
            f"STIP stopi>>16 should be IRQ_S_TIMER (5), "
            f"got IID={iid} (stopi={stopi:#x})"
        )
        assert stopi & 0xFFFF, f"stopi priority should be non-zero, got {stopi:#x}"

    def test_stopi_clears_after_stimecmp_written(self):
        """stopi returns 0 after stimecmp is advanced into the future.

        The kernel's timer ISR writes a new stimecmp (future value) and
        then re-reads stopi.  stopi must return 0 so the while loop in
        riscv_intc_aia_irq() exits.  Before the fix, stopi read stale
        mip_val.STIP=1 (set by check_pending_interrupts during the
        initial trap) and never returned 0 → infinite loop.
        """
        if not PYREMU_AIA:
            pytest.skip("stopi is readable only when PYREMU_AIA=1")

        emu = _make_aia_emu(num_harts=2)
        h = emu.harts[0]
        h.pc = 0x40000000
        h.mode = RiscvMode.S
        h.csrs["mie"].val |= 1 << 5   # STIE

        # Simulate stale mip_val from an earlier trap delivery
        h.mip_val |= 1 << 5  # STIP=1 (stale)

        # Kernel sets stimecmp into the future
        assert h.interrupt_ctrl, "empty aia interrupt controller"
        cur = h.interrupt_ctrl.get_mtime()
        h.csrs["stimecmp"].val = cur + 100_000

        # stopi must now return 0 because mtime < stimecmp
        stopi = h.read_csr(0xDB0)
        assert stopi == 0, (
            f"stopi must return 0 after stimecmp is advanced, got {stopi:#x}"
        )

    def test_stopi_ignores_mtip(self):
        """MTIP (bit 7) alone must NOT be reported by stopi.

        mtimecmp is frozen during native batch execution; if stopi
        checked it, a timer tick would never clear → infinite loop.
        Only STIP (bit 5, from state.stimecmp) is authoritative for
        S-mode timer interrupts.
        """
        if not PYREMU_AIA:
            pytest.skip("stopi is readable only when PYREMU_AIA=1")

        emu = _make_aia_emu(num_harts=2)
        h = emu.harts[0]
        h.pc = 0x40000000
        h.mode = RiscvMode.S
        h.csrs["mie"].val |= 1 << 7   # MTIE
        h.csrs["mideleg"].val |= 1 << 7  # delegate MTIP → S-mode

        # Set MTIP only (STIP is 0)
        h.mip_val |= 1 << 7
        h.mip_val &= ~(1 << 5)

        assert h.mip_val & (1 << 7), "MTIP should be pending"
        assert not (h.mip_val & (1 << 5)), "STIP should NOT be pending"

        # Read stopi → should return 0 (MTIP is deliberately ignored)
        stopi = h.read_csr(0xDB0)
        assert stopi == 0, (
            f"stopi must ignore MTIP (bit 7), got stopi={stopi:#x}"
        )


# ============================================================
#  CLINT + PLIC 共存
# ============================================================


class TestImsicCoexistence:
    """IMSIC 与 CLINT/PLIC 共存."""

    def test_clint_msip_works_with_imsic(self):
        """IMSIC 存在时 CLINT MSIP 仍正常工作."""
        emu = _make_aia_emu(num_harts=2)
        emu.clint.send_ipi(1)
        mip_bits = emu.clint.check_interrupt(1)[1]
        assert mip_bits & (1 << 3), "CLINT MSIP should still work"

    def test_clint_mtip_works_with_imsic(self):
        """IMSIC 存在时 CLINT MTIP 仍正常工作."""
        emu = _make_aia_emu()
        h = emu.harts[0]
        emu.clint._mtime = 1000
        emu.clint._mtimecmp[0] = 500  # mtime >= mtimecmp
        mip_bits = emu.clint.check_interrupt(0)[1]
        assert mip_bits & (1 << 7), "CLINT MTIP should still work"

        # MTIP 应通过 check_pending_interrupts 触发实际中断投递
        h.csrs["mie"].val = 1 << 7  # MTIE
        h.mip_val = 0
        h.pc = 0x40000000
        h.mode = RiscvMode.M
        h.mstatus_val |= 1 << 3  # MIE=1 (global enable)
        result = check_pending_interrupts(h)
        assert result, "MTIP with IMSIC should deliver interrupt"

    def test_plic_unaffected_by_imsic(self):
        """默认配置 (imsic_base=0): PLIC 行为不变."""
        cfg = PlatformConfig.qemu_virt()
        emu = Emulator(cfg)
        assert emu.imsic is None, "legacy config should have no IMSIC"
        assert emu.plic is not None, "legacy config should have PLIC"
        assert emu.harts[0]._imsic is None
        assert emu.harts[0]._plic is not None

    def test_imsic_mip_merged_in_check_pending(self):
        """IMSIC get_pending_mip 被 check_pending_interrupts 合并到 mip."""
        emu = _make_aia_emu()
        h = emu.harts[0]
        imsic = emu.imsic
        assert imsic is not None
        imsic.csr_write(0, 'M', 0x70, 1)   # eidelivery=1
        imsic.csr_write(0, 'M', 0xC0, 1 << 10)
        imsic.set_ip_number(0, 'M', 10)
        # 使能 MEIE
        h.csrs["mie"].val = 1 << 11  # MEIE
        h.mstatus_val |= 1 << 3  # MIE=1 (M-mode global enable)
        result = check_pending_interrupts(h)
        assert result, "IMSIC pending should trigger interrupt delivery"
        assert h.mip_val & (1 << 11), "mip.MEIP should be set"


# ============================================================
#  Hart CSR 集成
# ============================================================


class TestImsicHartIntegration:
    """Hart.write_csr / read_csr 通过 IMSIC 分发."""

    def test_miselect_mireg_roundtrip(self):
        """csrw miselect + csrr mireg 往返一致."""
        emu = _make_aia_emu()
        h = emu.harts[0]
        # miselect = eidelivery (0x70)
        h.write_csr(0x350, 0x70)
        assert h._imsic_select_m == 0x70
        # mireg: 读当前 eidelivery
        val = h.read_csr(0x351)
        assert val == 0  # default = 0
        # 写 mireg (eidelivery = 1)
        h.write_csr(0x351, 1)
        assert h.read_csr(0x351) == 1

    def test_siselect_sireg_roundtrip(self):
        """csrw siselect + csrr sireg 往返一致."""
        emu = _make_aia_emu()
        h = emu.harts[0]
        h.write_csr(0x150, 0x70)        # siselect = eidelivery
        assert h._imsic_select_s == 0x70
        h.write_csr(0x151, 1)           # sireg → S-file eidelivery=1
        assert h.read_csr(0x151) == 1

    def test_mireg_ignored_without_imsic(self):
        """无 IMSIC 时 mireg 读返回原始 CSR 值 (fallback)."""
        cfg = PlatformConfig.qemu_virt()
        emu = Emulator(cfg)
        h = emu.harts[0]
        assert h._imsic is None
        val = h.read_csr(0x351)  # mireg without IMSIC
        assert val == 0  # default CSR val


# ============================================================
#  Config 层
# ============================================================


class TestImsicConfig:
    """PlatformConfig 与 IMSIC 创建."""

    def test_qemu_virt_aia_preset(self):
        """qemu_virt_aia 预设正确配置 AIA 模式."""
        cfg = PlatformConfig.qemu_virt_aia()
        assert cfg.interrupt_mode == InterruptMode.AIA
        assert cfg.periph.imsic_m_base == 0x2400_0000
        assert cfg.periph.imsic_s_base == 0x2800_0000
        assert cfg.periph.aplic_base == 0x0C00_0000

    def test_default_config_no_imsic(self):
        """默认 qemu_virt 配置 imsic_base=0 — 不创建 IMSIC."""
        cfg = PlatformConfig.qemu_virt()
        assert cfg.periph.imsic_m_base == 0
        emu = Emulator(cfg)
        assert emu.imsic is None


# ============================================================
#  IPI 投递: M-file → MEIP, S-file → SEIP (无跨文件路由)
# ============================================================


class TestImsicIpi:
    """IMSIC IPI 通过 MEIP/SEIP 外部中断线投递 (AIA 规范).

    All IMSIC interrupts (software IPI minor identity 1 and external IID>=6)
    drive MEIP/SEIP via the file they were written to — M-file → MEIP,
    S-file → SEIP, with no cross-file routing.  M-mode handler reads MTOPEI
    for the minor identity; S-mode handler reads STOPE/STOPEI.
    """

    def test_mfile_ipi_sets_meip(self):
        """M-file seteipnum IID=3 pending → get_pending_mip 设置 MEIP.

        All IMSIC interrupts (including IPIs) drive the external interrupt
        line when eidelivery=1 per AIA spec.  A ``seteipnum`` write stays in
        the addressed file (no cross-file routing): IID=3 written to the M-file
        sets M-file eip[3] and drives MEIP.
        """
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)  # eidelivery=1
        imsic.csr_write(0, 'M', 0xC0, 1 << 3)  # eie: enable identity 3
        imsic.write(0x000, (3).to_bytes(4, 'little'))  # seteipnum 3 (M-file)
        mip = imsic.get_pending_mip(0)
        assert (mip & (1 << 11)) != 0, (
            f"MEIP should be set for M-file IID=3 (stays in M-file), got mip={mip:#x}"
        )

    def test_sfile_ipi_sets_seip(self):
        """S-file IPI IID=3 pending → get_pending_mip 设置 SEIP.

        All IMSIC interrupts drive the external interrupt line per AIA spec.
        """
        imsic = _make_imsic()
        imsic.csr_write(0, 'S', 0x70, 1)  # eidelivery=1
        imsic.csr_write(0, 'S', 0xC0, 1 << 3)  # eie: enable identity 3
        imsic.write(0x2000, (3).to_bytes(4, 'little'))  # S-file seteipnum 3
        mip = imsic.get_pending_mip(0)
        assert (mip & (1 << 9)) != 0, (
            f"SEIP should be set for ALL IMSIC interrupts (including IPI IID=3), "
            f"got mip={mip:#x}"
        )

    def test_ext_interrupt_sets_meip(self):
        """M-file 外部中断 (>=6) → get_pending_mip 返回 MEIP."""
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)
        imsic.csr_write(0, 'M', 0xC0, 1 << 10)  # eie bit 10
        imsic.write(0x000, (10).to_bytes(4, 'little'))  # seteipnum 10
        mip = imsic.get_pending_mip(0)
        assert mip & (1 << 11), f"MEIP should be set for ext int, got mip={mip:#x}"

    def test_mixed_ipi_and_ext(self):
        """IPI (IID=1) + 外部中断 (IID=10) 同时写 M-file → MEIP 置位, SEIP 不置位.

        Both a software IPI minor identity (1) and an external interrupt (10)
        written to the M-file stay in the M-file (no cross-file routing) and
        drive MEIP.  SEIP must remain clear — only an S-file write drives it.
        """
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)  # M-file eidelivery=1
        imsic.csr_write(0, 'M', 0xC0, (1 << 1) | (1 << 10))  # eie: enable IID=1,10
        imsic.write(0x000, (1).to_bytes(4, 'little'))   # M-file IID=1 → stays in M-file
        imsic.write(0x000, (10).to_bytes(4, 'little'))  # M-file IID=10 → stays in M-file
        mip = imsic.get_pending_mip(0)
        assert mip & (1 << 11), f"MEIP should be set (from IID=1,10 in M-file), got mip={mip:#x}"
        assert not (mip & (1 << 9)), (
            f"SEIP must NOT be set (no cross-file routing), got mip={mip:#x}"
        )

    def test_clreipnum_ipi_clears_pending(self):
        """clreipnum identity 1 → M-file pending 清除, MEIP cleared.

        Regression: clreipnum mirrors seteipnum's direct mapping — ``clreipnum = 1``
        written to the M-file clears M-file eip[1] (not the S-file).
        """
        imsic = _make_imsic()
        imsic.csr_write(0, 'M', 0x70, 1)  # M-file eidelivery=1
        imsic.csr_write(0, 'M', 0xC0, 1 << 1)
        imsic.write(0x000, (1).to_bytes(4, 'little'))
        # M-file IID=1 stays in M-file → MEIP, M-file topei reports pending IPI.
        assert (imsic.get_pending_mip(0) & (1 << 11)) != 0, "MEIP should be set for IPI IID=1"
        assert imsic.peek_topei(0, 'M') != 0, "M-file topei should report pending IPI"
        imsic.write(0x008, (1).to_bytes(4, 'little'))  # clreipnum 1 on M-file
        # clreipnum clears the M-file eip bit directly.
        assert imsic.get_pending_mip(0) == 0, "mip should be 0 after clear"
        assert imsic.peek_topei(0, 'M') == 0, "M-file topei should be 0 after clear"
