#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""initramfs (initrd) 接入测试.

锁定: DTB /chosen 中 linux,initrd-start / linux,initrd-end 的 u64 大端编码,
以及 Emulator.load_initrd 写入 RAM + 记录物理范围 + a1 (DTB) 可解析。
"""

import struct

import libfdt
import pytest

from pyremu.emulator import Emulator
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils.dtb import build_dtb, Initrd

RAM_BASE = 0x8000_0000


def _cfg() -> PlatformConfig:
    return PlatformConfig(
        num_harts=1,
        ram_size=16 * 1024 * 1024,
        ram_base=RAM_BASE,
        prog_cnt=RAM_BASE,
        periph=PeripheralConfig(),
    )


def _read_u64_prop(fdt: libfdt.Fdt, node: int, name: str) -> int:
    prop = fdt.getprop(node, name)
    return struct.unpack(">Q", bytes(prop))[0]


def test_dtb_encodes_initrd_range():
    """build_dtb 应在 /chosen 写入 initrd 起止 (u64 大端)."""
    cfg = _cfg()
    start, end = 0x9800_0000, 0x9A34_5678
    blob = build_dtb(cfg, bootargs="console=ttySIF0", initrd=Initrd(start, end))
    fdt = libfdt.Fdt(blob)
    chosen = fdt.path_offset("/chosen")
    assert _read_u64_prop(fdt, chosen, "linux,initrd-start") == start
    assert _read_u64_prop(fdt, chosen, "linux,initrd-end") == end


def test_dtb_no_initrd_when_absent():
    """未提供 initrd 时不应出现 initrd 属性."""
    cfg = _cfg()
    blob = build_dtb(cfg, bootargs="console=ttySIF0")
    fdt = libfdt.Fdt(blob)
    chosen = fdt.path_offset("/chosen")
    with pytest.raises(libfdt.FdtException):
        fdt.getprop(chosen, "linux,initrd-start")


def test_dtb_chosen_emitted_for_initrd_only():
    """即使无 bootargs, 只要有 initrd 也应生成 /chosen 节点."""
    cfg = _cfg()
    blob = build_dtb(cfg, bootargs=None, initrd=Initrd(0x9000_0000, 0x9001_0000))
    fdt = libfdt.Fdt(blob)
    chosen = fdt.path_offset("/chosen")  # 不存在会抛异常
    assert _read_u64_prop(fdt, chosen, "linux,initrd-start") == 0x9000_0000


def test_load_initrd_writes_ram_and_records_range(tmp_path):
    """load_initrd 写入 RAM, 记录范围, 并进入生成的 DTB."""
    emu = Emulator(_cfg())
    payload = b"CPIO-ROOTFS-PAYLOAD" * 16
    img = tmp_path / "initramfs.cpio"
    img.write_bytes(payload)
    addr = RAM_BASE + 0x0080_0000

    info = emu.load_initrd(str(img), addr)

    # RAM 中确实写入了 payload。
    assert emu.bus.try_read(addr, len(payload)) == payload
    assert info.start == addr
    assert info.end == addr + len(payload)

    # 生成的 DTB 包含该范围。
    blob = emu.build_dtb()
    fdt = libfdt.Fdt(blob)
    chosen = fdt.path_offset("/chosen")
    assert _read_u64_prop(fdt, chosen, "linux,initrd-start") == addr
    assert _read_u64_prop(fdt, chosen, "linux,initrd-end") == addr + len(payload)
