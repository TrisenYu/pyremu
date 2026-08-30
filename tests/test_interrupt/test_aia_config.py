#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""AIA 配置层测试 — PlatformConfig 序列化 + 向后兼容."""

import json
from pathlib import Path
import tempfile

import pytest

import pyremu.configs_gen
from pyremu.emulator import Emulator
from pyremu.platform import InterruptMode, PeripheralConfig, PlatformConfig


@pytest.fixture(autouse=True)
def _disable_aia_compile_override(monkeypatch):
    """Disable PYREMU_AIA compile-time override.

    Tests use explicit configs: qemu_virt() for legacy, qemu_virt_aia() for AIA.
    """
    monkeypatch.setattr(pyremu.configs_gen, "PYREMU_AIA", False)



class TestInterruptModeEnum:
    """InterruptMode 枚举."""

    def test_legacy_aia_values(self):
        assert InterruptMode.LEGACY.value == "legacy"
        assert InterruptMode.AIA.value == "aia"

    def test_from_string(self):
        assert InterruptMode("legacy") == InterruptMode.LEGACY
        assert InterruptMode("aia") == InterruptMode.AIA

    def test_default_is_legacy(self):
        cfg = PlatformConfig()
        assert cfg.interrupt_mode == InterruptMode.LEGACY


class TestQemuVirtAiaPreset:
    """qemu_virt_aia 工厂方法."""

    def test_creates_correct_mode(self):
        cfg = PlatformConfig.qemu_virt_aia()
        assert cfg.interrupt_mode == InterruptMode.AIA
        assert cfg.periph.imsic_m_base == 0x2400_0000
        assert cfg.periph.aplic_base == 0x0C00_0000

    def test_emulator_creates_imsic(self):
        emu = Emulator(PlatformConfig.qemu_virt_aia())
        assert emu.imsic is not None
        assert emu.harts[0]._imsic is not None

    def test_legacy_qemu_virt_has_no_imsic(self):
        emu = Emulator(PlatformConfig.qemu_virt())
        assert emu.imsic is None
        assert emu.harts[0]._imsic is None


class TestConfigSerialization:
    """JSON/TOML 序列化与枚举互转."""

    def test_json_roundtrip_legacy(self):
        cfg = PlatformConfig.qemu_virt()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "num_harts": cfg.num_harts,
                "ram_size": cfg.ram_size,
                "ram_base": cfg.ram_base,
                "interrupt_mode": cfg.interrupt_mode.value,
                "periph": {
                    "imsic_m_base": cfg.periph.imsic_m_base,
                    "imsic_s_base": cfg.periph.imsic_s_base,
                    "aplic_base": cfg.periph.aplic_base,
                    "plic_base": cfg.periph.plic_base,
                },
            }, f)
            path = f.name

        loaded = PlatformConfig.from_json(path)
        assert loaded.interrupt_mode == InterruptMode.LEGACY
        assert loaded.periph.imsic_m_base == 0
        Path(path).unlink()

    def test_json_roundtrip_aia(self):
        cfg = PlatformConfig.qemu_virt_aia()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "num_harts": cfg.num_harts,
                "interrupt_mode": cfg.interrupt_mode.value,
                "periph": {
                    "imsic_m_base": cfg.periph.imsic_m_base,
                    "imsic_s_base": cfg.periph.imsic_s_base,
                    "aplic_base": cfg.periph.aplic_base,
                    "plic_base": cfg.periph.plic_base,
                },
            }, f)
            path = f.name

        loaded = PlatformConfig.from_json(path)
        assert loaded.interrupt_mode == InterruptMode.AIA
        assert loaded.periph.imsic_m_base == 0x2400_0000
        Path(path).unlink()


class TestPeripheralConfigDefault:
    """PeripheralConfig 默认值向后兼容."""

    def test_default_imsic_base_zero(self):
        p = PeripheralConfig()
        assert p.imsic_m_base == 0
        assert p.aplic_base == 0

    def test_minimal_preset_no_imsic(self):
        cfg = PlatformConfig.minimal()
        assert cfg.periph.imsic_m_base == 0
        emu = Emulator(cfg)
        assert emu.imsic is None
