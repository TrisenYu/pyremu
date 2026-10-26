#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""DTB 的 /memory 与 /reserved-memory 节点.

锁定两类行为:

- reg 按 64-bit 编码为 4 个大端 u32 (addr_hi, addr_lo, size_hi, size_lo),
  地址与大小的高 32 位取自其自身取值. 覆盖 configs.mk 的 ram 默认值 6 GiB,
  该值的高 32 位非零.
- PlatformConfig.reserved_memory_ranges 非空时生成 /reserved-memory 的 no-map
  子节点, 供 Linux 内核从 enclave 池与固件映像区中排除物理内存区域.

注意: 必须通过 Emulator 导入链以规避 core.decoder ⟷ utils.disassem 循环导入.
"""

import struct

import libfdt

from pyremu.configs_gen import (
    FW_RESERVED_MEM_BASE,
    FW_RESERVED_MEM_SIZE,
    RESERVED_MEM_BASE,
    RESERVED_MEM_SIZE,
)
from pyremu.emulator import Emulator
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils import dtb

# configs.mk 的 ram 默认值. 6 GiB 的高 32 位为 1, 低 32 位为 0x8000_0000.
RAM_SIZE_6GIB = 6 * 1024 * 1024 * 1024

# 固件内嵌 .sittim 飞地运行时的物理地址: 链接地址 0x100000 + 加载偏移 0x80000000
SITTIM_PA = 0x8010_0000


def _cfg(ram_size: int | None = None) -> PlatformConfig:
    cfg = PlatformConfig(
        num_harts=1,
        ram_base=0x8000_0000,
        prog_cnt=0x8000_0000,
        periph=PeripheralConfig(),
    )
    if ram_size is not None:
        cfg.ram_size = ram_size
    return cfg


def _reg(path: str, blob: bytes) -> tuple[int, int]:
    """读取 *path* 节点 reg 属性, 返回 (地址, 大小)."""
    fdt = libfdt.Fdt(blob)
    node = fdt.path_offset(path)
    addr_hi, addr_lo, size_hi, size_lo = struct.unpack(
        ">IIII", bytes(fdt.getprop(node, "reg"))
    )
    return ((addr_hi << 32) | addr_lo, (size_hi << 32) | size_lo)


def _reserved_nodes(fdt: libfdt.Fdt) -> list[tuple[int, int, bool]]:
    """收集 /reserved-memory 全部子节点, 返回 (base, size, no-map) 列表."""
    rm = fdt.path_offset("/reserved-memory")
    out: list[tuple[int, int, bool]] = []
    node = fdt.first_subnode(rm, libfdt.QUIET_NOTFOUND)
    while node >= 0:
        reg = bytes(fdt.getprop(node, "reg"))
        base_hi, base_lo, size_hi, size_lo = struct.unpack(">IIII", reg)
        # no-map 为布尔属性, 长度 0 表示已设置
        no_map = len(bytes(fdt.getprop(node, "no-map"))) == 0
        out.append(((base_hi << 32) | base_lo, (size_hi << 32) | size_lo, no_map))
        node = fdt.next_subnode(node, libfdt.QUIET_NOTFOUND)
    return out


# ============================================================
#  /memory 与 /reserved-memory 的 reg 编码
# ============================================================


def test_memory_reg_encodes_size_above_4gib():
    """回归: ram 取 6 GiB 时 /memory 的 reg 必须把大小的高 32 位写出来.

    旧实现把 addr_hi 与 size_hi 两个单元硬写成 0, 6 GiB 超出 u32 的表示范围,
    struct.pack 抛 struct.error, 整个 DTB 无法生成, emu-linux-sh 因此中止.
    6 GiB 与 2 GiB 的高 32 位分别为 1 与 0, 两个取值一并锁定.
    """
    base, size = _reg("/memory", dtb.build_dtb(_cfg(RAM_SIZE_6GIB)))
    assert (base, size) == (0x8000_0000, RAM_SIZE_6GIB)
    assert size >> 32 == 1, "6 GiB 的高 32 位必须落在 reg 的 size_hi 单元"

    base, size = _reg("/memory", dtb.build_dtb(_cfg(2 * 1024 * 1024 * 1024)))
    assert (base, size) == (0x8000_0000, 2 * 1024 * 1024 * 1024)
    assert size >> 32 == 0, "2 GiB 的高 32 位必须为 0"


def test_reserved_memory_reg_encodes_base_above_4gib():
    """reg 的地址高于 4 GiB 时, 其高 32 位落在 addr_hi 单元."""
    ranges = [(0x1_0000_0000, 0x20_0000)]
    blob = dtb.build_dtb(_cfg(128 * 1024 * 1024), reserved_ranges=ranges)
    assert _reg("/reserved-memory/reserved-0@100000000", blob) == ranges[0]


# ============================================================
#  /reserved-memory 节点生成
# ============================================================


def test_reserved_memory_default_pool():
    """默认 reserved_memory_ranges 应为固件映像区与 enclave 池两个 no-map 节点."""
    cfg = _cfg()
    assert len(cfg.reserved_memory_ranges) == 2, "default should reserve fw + pool"

    emu = Emulator(cfg, bootargs="console=ttySIF0")
    blob = emu.build_dtb()
    fdt = libfdt.Fdt(blob)
    rm = fdt.path_offset("/reserved-memory")
    assert rm >= 0

    nodes = _reserved_nodes(fdt)
    assert [(base, size) for base, size, _ in nodes] == [
        (int(FW_RESERVED_MEM_BASE), int(FW_RESERVED_MEM_SIZE)),
        (int(RESERVED_MEM_BASE), int(RESERVED_MEM_SIZE)),
    ]
    assert all(no_map for _, _, no_map in nodes), "reserved regions must all be no-map"


def test_reserved_memory_covers_firmware_region():
    """固件映像区必须被 no-map 保留, 且覆盖内嵌 .sittim 的物理地址.

    回归: 固件映像 (ram_base 起, 含尾部内嵌的 .sittim 飞地运行时) 未保留时,
    内核分配器会把页面分到固件映像上, 设备 DMA (virtio-blk 读写) 直接覆写
    .sittim 对应的物理页面, 使 CREATE 复制进飞地池的载荷变成垃圾数据; 飞地
    入口取指即故障, 且该异常未委派到 S 模式, 表现为 M 模式静默挂起 (无任何
    后续输出). 此处锁定 "0x80100000 落在某个 no-map 保留区间内".
    """
    cfg = _cfg()
    assert int(FW_RESERVED_MEM_BASE) == cfg.ram_base, "fw 区应自 ram_base 起"

    emu = Emulator(cfg, bootargs="console=ttySIF0")
    fdt = libfdt.Fdt(emu.build_dtb())
    nodes = _reserved_nodes(fdt)

    covering = [
        (base, size)
        for base, size, no_map in nodes
        if no_map and base <= SITTIM_PA < base + size
    ]
    assert covering, f"固件内嵌 .sittim (PA {SITTIM_PA:#x}) 未被任何 no-map 区间覆盖: {nodes}"


def test_reserved_memory_multiple_ranges():
    """多个 reserved 区域应分别生成子节点."""
    cfg = _cfg()
    cfg.reserved_memory_ranges = [
        (0x8300_0000, 0x0080_0000),
        (0x9000_0000, 0x0100_0000),
    ]
    emu = Emulator(cfg, bootargs="console=ttySIF0")
    blob = emu.build_dtb()
    fdt = libfdt.Fdt(blob)

    nodes = _reserved_nodes(fdt)
    assert len(nodes) == 2, f"expected 2 children, got {len(nodes)}"
    for base, _, _ in nodes:
        assert base in (0x8300_0000, 0x9000_0000), f"unexpected base=0x{base:x}"


def test_reserved_memory_empty_no_node():
    """空 reserved_memory_ranges 不生成 /reserved-memory 节点."""
    cfg = _cfg()
    cfg.reserved_memory_ranges = []
    emu = Emulator(cfg, bootargs="console=ttySIF0")
    blob = emu.build_dtb()
    fdt = libfdt.Fdt(blob)
    try:
        fdt.path_offset("/reserved-memory")
        assert False, "/reserved-memory must be absent with empty ranges"
    except libfdt.FdtException:
        pass  # expected — node not present
