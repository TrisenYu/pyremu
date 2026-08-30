#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""AIA 集成测试 — emulator + IMSIC + APLIC 端到端."""

import pytest

import pyremu.configs_gen
from pyremu.core.trap_handler import check_pending_interrupts, try_wfi_wakeup
from pyremu.emulator import Emulator
from pyremu.interrupt.aplic import APLIC
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


def _make_aia_emu(num_harts: int = 1) -> Emulator:
    cfg = PlatformConfig.qemu_virt_aia()
    cfg.num_harts = num_harts
    cfg.ram_base = 0x4000_0000
    cfg.prog_cnt = 0x4000_0000
    return Emulator(cfg)


def _make_legacy_emu(num_harts: int = 1) -> Emulator:
    cfg = PlatformConfig.qemu_virt()
    cfg.num_harts = num_harts
    cfg.ram_base = 0x4000_0000
    cfg.prog_cnt = 0x4000_0000
    return Emulator(cfg)


# ============================================================
#  Emulator 创建
# ============================================================


class TestAiaEmulator:
    """Emulator 在 AIA 模式下正确创建 IMSIC+APLIC, 不创建 PLIC."""

    def test_aia_mode_creates_imsic_aplic(self):
        """AIA 模式: IMSIC 和 APLIC 存在, PLIC 为 None."""
        emu = _make_aia_emu(4)
        assert emu.imsic is not None, "IMSIC should be created"
        assert emu.aplic is not None, "APLIC should be created"
        assert emu.plic is None, "PLIC should be absent in AIA mode"

    def test_legacy_mode_creates_plic(self):
        """Legacy 模式: PLIC 存在, IMSIC 和 APLIC 为 None."""
        emu = _make_legacy_emu()
        assert emu.plic is not None, "PLIC should be created"
        assert emu.imsic is None, "IMSIC should be absent in legacy mode"
        assert emu.aplic is None, "APLIC should be absent in legacy mode"

    def test_hart_imsic_wired(self):
        """每个 hart 的 _imsic 指向共享 IMSIC 实例."""
        emu = _make_aia_emu(4)
        for h in emu.harts:
            assert h._imsic is emu.imsic

    def test_imsic_mmio_on_bus(self):
        """IMSIC MMIO 区域在总线上."""
        emu = _make_aia_emu()
        imsic_base = emu._cfg.periph.imsic_m_base
        dev = emu.bus.devices.get(imsic_base)
        assert dev is emu.imsic

    def test_aplic_mmio_on_bus(self):
        """APLIC MMIO 区域在总线上."""
        emu = _make_aia_emu()
        aplic_base = emu._cfg.periph.aplic_base
        dev = emu.bus.devices.get(aplic_base)
        assert dev is emu.aplic


# ============================================================
#  中断投递: UART → APLIC → IMSIC → check_pending_interrupts
# ============================================================


class TestAiaInterruptDelivery:
    """端到端中断投递: 外设 → APLIC → IMSIC → hart mip."""

    def test_uart_irq_raises_seip(self):
        """UART 写 → APLIC set_irq → IMSIC S-file → get_pending_mip 返回 SEIP.

        Device interrupts route to S-mode (delegate=True) so the kernel's
        IMSIC driver handles them directly without M-mode forwarding.
        """
        emu = _make_aia_emu()
        assert emu.imsic is not None and emu.aplic is not None
        # 使能 IMSIC S-file eie for IID=20 (UART via APLIC)
        emu.imsic.csr_write(0, 'S', 0x70, 1)  # eidelivery=1
        emu.imsic.csr_write(0, 'S', 0xC0, 1 << 20)  # eie: enable identity 20
        # 配置 APLIC source 10 → hart 0 S-file IID 20 (经 MMIO, 模拟内核流程)
        emu.aplic.write(0x3004 + (10 - 1) * 4, (20).to_bytes(4, "little"))  # target[10]
        emu.aplic.write(0x0004 + (10 - 1) * 4, (0x6).to_bytes(4, "little"))  # sourcecfg[10] LEVEL_HIGH
        emu.aplic.write(0x1EDC, (10).to_bytes(4, "little"))  # setienum source 10

        # 触发 UART 中断: 经 APLIC source 10 → S-file IID=20
        emu.aplic.set_irq(10, True)

        mip = emu.imsic.get_pending_mip(0)
        assert mip & (1 << 9), f"SEIP expected (S-mode routing), got mip={mip:#x}"

    def test_check_pending_interrupts_sees_imsic_meip(self):
        """check_pending_interrupts 在 IMSIC 有 pending 时检测到 MEI."""
        emu = _make_aia_emu()
        hart = emu.harts[0]
        imsic = emu.imsic
        assert imsic is not None
        imsic.csr_write(0, 'M', 0x70, 1)  # eidelivery=1
        imsic.csr_write(0, 'M', 0xC0, 1 << 10)  # eie bit 10
        imsic.set_ip_number(0, 'M', 10)  # 注入 MSI
        emu._native_sync_plic_mip()  # IMSIC path: get_pending_mip → mip
        hart.mie = True  # 全局 MIE=1
        hart.csrs["mie"].val = 1 << 11  # MEIE=1
        result = check_pending_interrupts(hart)
        assert result is not None, "should detect pending MEI"

    def test_interrupt_delivered_to_specific_hart(self):
        """中断仅投递到目标 hart (非广播)."""
        emu = _make_aia_emu(4)
        imsic = emu.imsic
        assert imsic is not None
        for i in range(4):
            imsic.csr_write(i, 'M', 0x70, 1)
            imsic.csr_write(i, 'M', 0xC0, 1 << 10)
        # 仅向 hart 2 注入
        imsic.set_ip_number(2, 'M', 10)
        emu._native_sync_plic_mip()
        assert imsic.get_pending_mip(2) & (1 << 11)
        assert imsic.get_pending_mip(0) == 0
        assert imsic.get_pending_mip(1) == 0
        assert imsic.get_pending_mip(3) == 0


# ============================================================
#  WFI 唤醒
# ============================================================


class TestAiaWfiWakeup:
    """IMSIC 外部中断唤醒 WFI."""

    def test_wfi_wakeup_from_imsic(self):
        """WFI 等待中的 hart 在 IMSIC 外部中断到达时唤醒."""
        emu = _make_aia_emu()
        hart = emu.harts[0]
        imsic = emu.imsic
        assert imsic is not None
        imsic.csr_write(0, 'M', 0x70, 1)
        imsic.csr_write(0, 'M', 0xC0, 1 << 10)
        hart.mie = True  # 全局 MIE=1
        hart.csrs["mie"].val = 1 << 11  # MEIE=1 (WFI requires source-level enable)
        # WFI: 无 pending 中断, 进入等待
        hart._waiting = True
        # IMSIC 注入中断
        imsic.set_ip_number(0, 'M', 10)
        emu._native_sync_plic_mip()
        # try_wfi_wakeup 应唤醒 hart
        result = try_wfi_wakeup(hart)
        assert result, "WFI should wake up on IMSIC external interrupt"
        assert not hart._waiting, "hart should be awoken"


# ============================================================
#  mtopei claim 语义
# ============================================================


class TestAiaTopei:
    """mtopei / stopei read-and-claim."""

    def test_mtopei_returns_iid_and_priority(self):
        """mtopei 返回 (IID << 16) | priority."""
        emu = _make_aia_emu()
        imsic = emu.imsic
        assert imsic is not None
        imsic.csr_write(0, 'M', 0x70, 1)  # eidelivery=1
        imsic.csr_write(0, 'M', 0xC0, 1 << 20)  # eie bit 20
        imsic.set_ip_number(0, 'M', 20)
        topei = imsic.read_topei(0, 'M')
        assert topei != 0, "should return non-zero topei"
        iid = (topei >> 16) & 0xFFFF
        assert iid == 20, f"expected IID=20, got {iid}"

    def test_mtopei_claim_clears_pending(self):
        """mtopei 读后 pending 位被清除."""
        emu = _make_aia_emu()
        imsic = emu.imsic
        assert imsic is not None
        imsic.csr_write(0, 'M', 0x70, 1)
        imsic.csr_write(0, 'M', 0xC0, 1 << 15)
        imsic.set_ip_number(0, 'M', 15)
        assert imsic.get_pending_mip(0) & (1 << 11), "MEIP before claim"
        # read topei → claim
        imsic.read_topei(0, 'M')
        assert imsic.get_pending_mip(0) == 0, "MEIP should be clear after claim"


# ============================================================
#  DTB 验证
# ============================================================


class TestAiaDtb:
    """AIA 模式 DTB 节点."""

    def test_dtb_contains_imsic_node(self):
        """AIA DTB 含 imsic 节点."""
        emu = _make_aia_emu(2)
        dtb = bytes(emu.build_dtb())
        assert b"imsic" in dtb, "DTB should contain imsic node"

    def test_dtb_contains_aplic_node(self):
        """AIA DTB 含 aplic 节点."""
        emu = _make_aia_emu(2)
        dtb = bytes(emu.build_dtb())
        assert b"aplic" in dtb, "DTB should contain aplic node"

    def test_dtb_no_plic_in_aia_mode(self):
        """AIA DTB 不含 plic 节点."""
        emu = _make_aia_emu(2)
        dtb = bytes(emu.build_dtb())
        assert b"riscv,plic0" not in dtb, "AIA DTB should not contain PLIC"

    def test_dtb_has_plic_in_legacy_mode(self):
        """Legacy DTB 含 plic 节点."""
        emu = _make_legacy_emu()
        dtb = bytes(emu.build_dtb())
        assert b"riscv,plic0" in dtb, "Legacy DTB should contain PLIC"


# ============================================================
#  eidelivery gate
# ============================================================


class TestEideliveryGate:
    """eidelivery=0 时 IMSIC 不参与中断."""

    def test_eidelivery_zero_no_meip(self):
        """eidelivery=0 时 get_pending_mip 返回 0."""
        emu = _make_aia_emu()
        imsic = emu.imsic
        assert imsic is not None
        imsic.csr_write(0, 'M', 0x70, 0)  # eidelivery=0
        imsic.csr_write(0, 'M', 0xC0, 1 << 10)
        imsic.set_ip_number(0, 'M', 10)
        assert imsic.get_pending_mip(0) == 0, "eidelivery=0 should block MEIP"


# ============================================================
#  配置一致性
# ============================================================


class TestInterruptModeConfig:
    """interrupt_mode 配置控制中断子系统."""

    def test_default_config_is_legacy(self):
        """默认配置为 legacy 模式."""
        cfg = PlatformConfig.qemu_virt()
        assert cfg.interrupt_mode == InterruptMode.LEGACY
        assert cfg.periph.imsic_m_base == 0  # 默认禁用

    def test_aia_config_has_imsic_base(self):
        """AIA 预设含 IMSIC/APLIC 基址."""
        cfg = PlatformConfig.qemu_virt_aia()
        assert cfg.interrupt_mode == InterruptMode.AIA
        assert cfg.periph.imsic_m_base != 0
        assert cfg.periph.aplic_base != 0
