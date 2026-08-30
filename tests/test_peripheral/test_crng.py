#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""模拟随机数生成器 (CRNG) 测试.

覆盖: 熵源确定性 / 寄存器读 / 写入无副作用 / Bus 集成 / DTB 节点生成.
"""

import struct

import libfdt
import pytest

from pyremu.emulator import Emulator
from pyremu.memory.bus import Bus
from pyremu.peripheral.crng import (
    CRNG,
    DEVICE_ID,
    REG_DATA,
    REG_ID,
    REG_STATUS,
    STATUS_READY,
)
from pyremu.platform import PeripheralConfig, PlatformConfig


class TestCRNGEntropy:
    """熵源行为: 确定性 (seed) 与随机 (os.urandom) 两条路径."""

    def test_read_data_returns_requested_size(self):
        """DATA 读取应精确返回 size 字节."""
        rng = CRNG(seed=42)
        for size in (1, 2, 4, 8):
            data = rng.read(REG_DATA, size)
            assert len(data) == size

    def test_same_seed_is_deterministic(self):
        """相同 seed 的两个实例产生相同字节序列 (测试复现基础)."""
        a = CRNG(seed=42)
        b = CRNG(seed=42)
        assert a.read(REG_DATA, 8) == b.read(REG_DATA, 8)
        assert a.read(REG_DATA, 8) == b.read(REG_DATA, 8)

    def test_diff_seed_differs(self):
        """不同 seed 产生不同字节 (Mersenne Twister seed 0/1 首块必然不同)."""
        assert CRNG(seed=0).read(REG_DATA, 8) != CRNG(seed=1).read(REG_DATA, 8)

    def test_sequence_advances_within_instance(self):
        """同一实例连续读取产生不同字节 (流推进)."""
        rng = CRNG(seed=7)
        assert rng.read(REG_DATA, 8) != rng.read(REG_DATA, 8)

    def test_no_seed_uses_os_urandom(self):
        """seed=None 走 os.urandom 路径 — 仅验证长度与两次读通常不同."""
        rng = CRNG()
        data = rng.read(REG_DATA, 8)
        assert len(data) == 8
        # 真随机路径: 两次 8 字节读取碰撞概率 2^-64, 断言不同是安全的.
        assert data != rng.read(REG_DATA, 8)


class TestCRNGRegisters:
    """寄存器语义."""

    @pytest.fixture
    def rng(self) -> CRNG:
        return CRNG(seed=1)

    def test_status_ready(self, rng):
        """STATUS 寄存器恒返回 ready (bit0=1)."""
        assert int.from_bytes(rng.read(REG_STATUS, 4), "little") == STATUS_READY

    def test_id_register(self, rng):
        """ID 寄存器返回 0x43524E47 ("CRNG")."""
        assert int.from_bytes(rng.read(REG_ID, 4), "little") == DEVICE_ID
        assert DEVICE_ID == 0x4352_4E47

    def test_unmapped_offset_returns_zero(self, rng):
        """未定义偏移读返回全 0."""
        assert rng.read(0x10, 4) == b"\x00" * 4
        assert rng.read(0x20, 8) == b"\x00" * 8

    def test_write_is_noop(self, rng):
        """写入无副作用 (纯被动设备) — 状态与后续读不变."""
        before_status = rng.read(REG_STATUS, 4)
        before_id = rng.read(REG_ID, 4)
        rng.write(REG_DATA, b"\xff\xff\xff\xff")
        rng.write(REG_STATUS, b"\x00\x00\x00\x00")
        rng.write(0x100, b"\x00" * 8)
        assert rng.read(REG_STATUS, 4) == before_status
        assert rng.read(REG_ID, 4) == before_id


class TestCRNGBus:
    """Bus 集成: 设备路由与 MMIO 检测."""

    def test_bus_routes_read_to_crng(self):
        """经 Bus 读取 CRNG 地址应路由到设备并返回熵."""
        bus = Bus(ram_size=1024 * 1024, ram_base=0)
        bus.add_device(0x1000_6000, CRNG(seed=3, base=0x1000_6000))
        assert bus.is_device_addr(0x1000_6000)
        data = bus.read(0x1000_6000, 8)
        assert len(data) == 8
        # 确定性 seed=3: 与直接实例化相同 seed 的结果一致.
        assert data == CRNG(seed=3).read(REG_DATA, 8)


class TestCRNGConfig:
    """平台配置与 DTB 节点生成."""

    @staticmethod
    def _cfg() -> PlatformConfig:
        return PlatformConfig(
            num_harts=1,
            ram_base=0x8000_0000,
            prog_cnt=0x8000_0000,
            periph=PeripheralConfig(),
        )

    def test_default_crng_base(self):
        """默认 PeripheralConfig 启用 CRNG (基址非零)."""
        assert PeripheralConfig().crng_base == 0x1000_6000

    def test_crng_base_does_not_collide_with_virtio_blk(self):
        """CRNG 默认基址与 virtio-blk 基址不得重叠.

        历史 bug: 两者默认同为 0x1000_5000, ``bus.add_device`` 后注册的
        CRNG 覆盖 virtio, 使 virtio 的 ACK 写被 CRNG 空 ``write()`` 吞掉,
        PLIC 电平无法拉低 -> guest "irq N: nobody cared" 停摆.
        """
        # virtio-blk 基址来自 cli.py --disk 的硬编码 _VIRTIO_BLK_BASE (0x200 字节).
        virtio_base = 0x1000_5000
        virtio_size = 0x200
        crng_base = PeripheralConfig().crng_base
        crng_size = 0x1000
        # 两段 MMIO 区间无交集 (CRNG 恒位于 virtio 之上或之下).
        assert crng_base + crng_size <= virtio_base or crng_base >= virtio_base + virtio_size

    def test_minimal_preset_disables_crng(self):
        """minimal 预设禁用 CRNG (基址 0), 保持 '仅 CLINT + UART' 语义."""
        assert PlatformConfig.minimal().periph.crng_base == 0

    def test_dtb_contains_crng_node(self):
        """启用 CRNG 时 DTB 应包含 /soc/crng@... 节点."""
        emu = Emulator(self._cfg(), bootargs="console=ttySIF0")
        blob = emu.build_dtb()
        fdt = libfdt.Fdt(blob)
        node = fdt.path_offset("/soc/crng@10006000")
        assert node >= 0
        compat = bytes(fdt.getprop(node, "compatible"))
        assert compat == b"pyremu,crng\x00"

    def test_dtb_crng_disabled_no_node(self):
        """crng_base=0 时 DTB 不应包含 crng 节点."""
        cfg = self._cfg()
        cfg.periph.crng_base = 0
        emu = Emulator(cfg, bootargs="console=ttySIF0")
        blob = emu.build_dtb()
        fdt = libfdt.Fdt(blob)
        with pytest.raises(libfdt.FdtException):
            fdt.path_offset("/soc/crng@0")

    def test_dtb_crng_reg_encoded(self):
        """crng 节点的 reg 属性编码正确的 base/size."""
        emu = Emulator(self._cfg(), bootargs="console=ttySIF0")
        blob = emu.build_dtb()
        fdt = libfdt.Fdt(blob)
        node = fdt.path_offset("/soc/crng@10006000")
        reg = bytes(fdt.getprop(node, "reg"))
        base_hi, base_lo, size_hi, size_lo = struct.unpack(">IIII", reg)
        base = (base_hi << 32) | base_lo
        size = (size_hi << 32) | size_lo
        assert base == 0x1000_6000
        assert size == 0x1000
