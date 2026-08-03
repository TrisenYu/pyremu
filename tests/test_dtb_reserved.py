"""DTB reserved-memory 节点生成测试.

锁定: build_dtb 在 PlatformConfig.reserved_memory_ranges 非空时
生成 /reserved-memory no-map 子节点, 以供 Linux 内核从 enclave 池
中排除物理内存区域.

注意: 必须通过 Emulator 导入链以规避 core.decoder ⟷ utils.disassem 循环导入.
"""

import struct

import libfdt

from pyremu.configs_gen import RESERVED_MEM_BASE, RESERVED_MEM_SIZE
from pyremu.emulator import Emulator
from pyremu.platform import PeripheralConfig, PlatformConfig


def _cfg() -> PlatformConfig:
    return PlatformConfig(
        num_harts=1,
        ram_base=0x8000_0000,
        prog_cnt=0x8000_0000,
        periph=PeripheralConfig(),
    )


def test_reserved_memory_default_pool():
    """默认 reserved_memory_ranges 应在 DTB 中生成 no-map 子节点."""
    cfg = _cfg()
    # 默认 PlatformConfig 含 [(0x83000000, 0x800000)]
    assert len(cfg.reserved_memory_ranges) > 0, "default should have pool reserved"

    emu = Emulator(cfg, bootargs="console=ttySIF0")
    blob = emu.build_dtb()
    fdt = libfdt.Fdt(blob)
    rm = fdt.path_offset("/reserved-memory")
    assert rm >= 0

    node = fdt.first_subnode(rm, libfdt.QUIET_NOTFOUND)
    assert node >= 0, "expected at least one reserved child"

    # no-map
    no_map = bytes(fdt.getprop(node, "no-map"))
    assert len(no_map) == 0, "no-map must be boolean (empty) property"

    # reg: 2-cell address + 2-cell size
    reg = bytes(fdt.getprop(node, "reg"))
    base_hi, base_lo, size_hi, size_lo = struct.unpack(">IIII", reg)
    base = (base_hi << 32) | base_lo
    size = (size_hi << 32) | size_lo
    assert base == int(RESERVED_MEM_BASE), f"base=0x{base:x}"
    assert size == int(RESERVED_MEM_SIZE), f"size=0x{size:x}"

    # 仅一个子节点
    next_node = fdt.next_subnode(node, libfdt.QUIET_NOTFOUND)
    assert next_node < 0, "expected exactly one reserved child"


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
    rm = fdt.path_offset("/reserved-memory")

    node = fdt.first_subnode(rm, libfdt.QUIET_NOTFOUND)
    count = 0
    while node >= 0:
        reg = bytes(fdt.getprop(node, "reg"))
        base_hi, base_lo, _, _ = struct.unpack(">IIII", reg)
        base = (base_hi << 32) | base_lo
        assert base in (0x8300_0000, 0x9000_0000), f"unexpected base=0x{base:x}"
        count += 1
        node = fdt.next_subnode(node, libfdt.QUIET_NOTFOUND)
    assert count == 2, f"expected 2 children, got {count}"


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
