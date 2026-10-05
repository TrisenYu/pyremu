#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""virtio-net 设备树节点生成测试.

锁定: 设备树只给出宿主侧实例 (``virtio_net_s``) 的节点, 飞地侧实例
(``virtio_net_m``) 的节点不得出现. 树中出现的中断源会被受调试程序内核在
自己的 S context 上使能, 该中断源一置位即产生 SEIP, 而 SEIP 由宿主 S 模式支配.

注意: 必须通过 Emulator 导入链以规避 core.decoder ⟷ utils.disassem 循环导入.
"""

import struct

import libfdt
import pytest

from pyremu.emulator import Emulator
from pyremu.peripheral.virtio_net import VIRTIO_NET_M_IRQ, VIRTIO_NET_S_IRQ
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils import dtb

# 两个实例的 MMIO 基址 — 与 pyremu/debug/cli.py 的取值一致
NET_S_BASE = 0x1000_7000
NET_M_BASE = 0x1000_8000

_IRQ_TYPE_LEVEL_HIGH = 0x4


def _cfg(is_net_s: bool = True, is_net_m: bool = True) -> PlatformConfig:
    """构造只启用指定网卡实例的平台配置."""
    cfg = PlatformConfig(
        num_harts=1,
        ram_base=0x8000_0000,
        prog_cnt=0x8000_0000,
        periph=PeripheralConfig(),
    )
    if is_net_s:
        cfg.periph.virtio_net_s_base = NET_S_BASE
    if is_net_m:
        cfg.periph.virtio_net_m_base = NET_M_BASE
    return cfg


def _fdt(cfg: PlatformConfig) -> libfdt.Fdt:
    return libfdt.Fdt(Emulator(cfg, bootargs="console=ttySIF0").build_dtb())


def _find(fdt: libfdt.Fdt, path: str) -> int:
    """按路径查找节点, 不存在时返回 -1."""
    return fdt.path_offset(path, libfdt.QUIET_NOTFOUND)


def _all_interrupt_props(fdt: libfdt.Fdt, node: int = 0) -> list[bytes]:
    """递归遍历全树, 收集每个节点 ``interrupts`` 属性的原始内容."""
    out: list[bytes] = []
    child = fdt.first_subnode(node, libfdt.QUIET_NOTFOUND)
    while child >= 0:
        try:
            out.append(bytes(fdt.getprop(child, "interrupts")))
        except libfdt.FdtException:
            pass  # 该节点没有 interrupts 属性
        out.extend(_all_interrupt_props(fdt, child))
        child = fdt.next_subnode(child, libfdt.QUIET_NOTFOUND)
    return out


# ============================================================
#  节点存在性
# ============================================================


class TestNetNodePresence:
    """设备树中只有宿主侧实例的节点."""

    @pytest.fixture(autouse=True)
    def _legacy_cells(self, monkeypatch) -> None:
        """本类按 legacy 的中断单元数校验, 与编译期 AIA 开关无关."""
        monkeypatch.setattr(dtb, "PYREMU_AIA", False)

    def test_debuggee_side_node_present(self) -> None:
        """宿主侧实例的节点存在, 形制与 virtio-blk 节点一致."""
        fdt = _fdt(_cfg())
        node = _find(fdt, f"/soc/virtio@{NET_S_BASE:x}")
        assert node >= 0, "宿主侧实例应进设备树"

        assert bytes(fdt.getprop(node, "compatible")) == b"virtio,mmio\x00"
        addr_hi, addr_lo, size_hi, size_lo = struct.unpack(
            ">IIII", bytes(fdt.getprop(node, "reg"))
        )
        assert (addr_hi << 32) | addr_lo == NET_S_BASE
        assert (size_hi << 32) | size_lo == 0x200
        assert struct.unpack(">I", bytes(fdt.getprop(node, "interrupts"))) == (
            VIRTIO_NET_S_IRQ,
        )

    def test_enclave_side_node_absent(self) -> None:
        """飞地侧实例的节点不存在."""
        fdt = _fdt(_cfg())
        assert _find(fdt, f"/soc/virtio@{NET_M_BASE:x}") < 0, "飞地侧实例不得进设备树"

    def test_enclave_side_irq_absent_from_whole_tree(self) -> None:
        """飞地侧实例的中断源编号不出现在全树任何 ``interrupts`` 属性中.

        只有节点不存在还不够: 任何一处把该源号写进树里, 内核都会在自己的 S
        context 上使能它.
        """
        fdt = _fdt(_cfg())
        for raw in _all_interrupt_props(fdt):
            cells = struct.unpack(f">{len(raw) // 4}I", raw)
            assert VIRTIO_NET_M_IRQ not in cells, f"飞地侧源号出现在 interrupts={cells}"

    def test_enclave_side_only_yields_no_node(self) -> None:
        """只启用飞地侧实例时, 设备树中不存在任何 virtio 节点.

        这是上一条的加强形式: 源号本身 (3) 可能与其它属性的取值重合, 故另按节点
        路径断言. 改动前该用例失败 (M 侧实例被无条件写入节点).
        """
        fdt = _fdt(_cfg(is_net_s=False, is_net_m=True))
        assert _find(fdt, f"/soc/virtio@{NET_M_BASE:x}") < 0
        assert _find(fdt, f"/soc/virtio@{NET_S_BASE:x}") < 0

    def test_disabled_instance_yields_no_node(self) -> None:
        """两个实例都未启用时不存在 virtio 节点."""
        fdt = _fdt(_cfg(is_net_s=False, is_net_m=False))
        assert _find(fdt, f"/soc/virtio@{NET_S_BASE:x}") < 0
        assert _find(fdt, f"/soc/virtio@{NET_M_BASE:x}") < 0


# ============================================================
#  中断属性编码
# ============================================================


class TestNetNodeInterruptEncoding:
    """中断单元数随中断模式变化."""

    def test_legacy_encoding_is_one_cell(self, monkeypatch) -> None:
        """legacy 下 APLIC 不在场, interrupts 取 PLIC 的一格编码."""
        monkeypatch.setattr(dtb, "PYREMU_AIA", False)
        fdt = _fdt(_cfg())
        node = _find(fdt, f"/soc/virtio@{NET_S_BASE:x}")
        raw = bytes(fdt.getprop(node, "interrupts"))
        assert raw == struct.pack(">I", VIRTIO_NET_S_IRQ)

    def test_aia_encoding_is_two_cells(self, monkeypatch) -> None:
        """AIA 下 interrupts 取 APLIC 的 ``<中断源编号 flags>`` 两格编码."""
        monkeypatch.setattr(dtb, "PYREMU_AIA", True)
        fdt = _fdt(_cfg())
        node = _find(fdt, f"/soc/virtio@{NET_S_BASE:x}")
        raw = bytes(fdt.getprop(node, "interrupts"))
        assert raw == struct.pack(">II", VIRTIO_NET_S_IRQ, _IRQ_TYPE_LEVEL_HIGH)
