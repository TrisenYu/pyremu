"""DTB /memory 与 /reserved-memory 节点的 reg 编码测试.

锁定: reg 按 64-bit 编码为 4 个大端 u32 (addr_hi, addr_lo, size_hi, size_lo),
地址与大小的高 32 位取自其自身取值. 覆盖 configs.mk 的 ram 默认值 6 GiB,
该值的高 32 位非零.
"""

import struct

import libfdt

from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils import dtb

# configs.mk 的 ram 默认值. 6 GiB 的高 32 位为 1, 低 32 位为 0x8000_0000.
RAM_SIZE_6GIB = 6 * 1024 * 1024 * 1024


def _cfg(ram_size: int) -> PlatformConfig:
    return PlatformConfig(
        num_harts=1,
        ram_base=0x8000_0000,
        ram_size=ram_size,
        prog_cnt=0x8000_0000,
        periph=PeripheralConfig(),
    )


def _reg(path: str, blob: bytes) -> tuple[int, int]:
    """读取 *path* 节点 reg 属性, 返回 (地址, 大小)."""
    fdt = libfdt.Fdt(blob)
    node = fdt.path_offset(path)
    addr_hi, addr_lo, size_hi, size_lo = struct.unpack(
        ">IIII", bytes(fdt.getprop(node, "reg"))
    )
    return ((addr_hi << 32) | addr_lo, (size_hi << 32) | size_lo)


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
