#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""btrfs 的超级块解析.

超级块位于 64 KiB 处, 给出设备大小、已用字节数、节点大小与特性位.
btrfs 的空闲空间由块组记录维护, 超级块不给出空闲数量, 因此摘要中该值为 None.
本模块只读映像, 不写入任何扇区.
"""

from dataclasses import dataclass
import io
from pathlib import Path
import struct

from pyremu.disk.base import DiskFormatError, DiskRegion, DiskSummary
from pyremu.utils.mask import is_power_of_two_in
from pyremu.utils.str_aux import bytes_to_utf8, flag_names, join_names, uuid_text

FORMAT_NAME = "btrfs"

SUPERBLOCK_OFFSET = 0x10000
SUPERBLOCK_SIZE = 4096
MAGIC = b"_BHRfS_M"

MIN_SECTOR_SIZE = 4096
MAX_SECTOR_SIZE = 65536
MAX_NODE_SIZE = 65536

CSUM_TYPES: tuple[tuple[int, str], ...] = (
    (0, "crc32c"),
    (1, "xxhash"),
    (2, "sha256"),
    (3, "blake2b"),
)

COMPAT_FEATURES: tuple[tuple[int, str], ...] = ((0x1, "mixed_backref"),)

COMPAT_RO_FEATURES: tuple[tuple[int, str], ...] = (
    (0x1, "free_space_tree"),
    (0x2, "free_space_tree_valid"),
)

INCOMPAT_FEATURES: tuple[tuple[int, str], ...] = (
    (0x1, "mixed_backref"),
    (0x2, "default_subvol"),
    (0x4, "mixed_groups"),
    (0x8, "compress_lzo"),
    (0x10, "compress_zstd"),
    (0x20, "big_metadata"),
    (0x40, "extended_iref"),
    (0x80, "raid56"),
    (0x100, "skinny_metadata"),
    (0x200, "no_holes"),
    (0x400, "metadata_uuid"),
    (0x800, "raid1c34"),
    (0x1000, "zoned"),
    (0x2000, "extent_tree_v2"),
)


@dataclass(frozen=True)
class BtrfsSuperblock:
    """超级块中的布局字段与特性位."""

    bytenr: int
    fsid: str
    label: str
    total_bytes: int
    bytes_used: int
    sectorsize: int
    nodesize: int
    stripesize: int
    num_devices: int
    csum_type: int
    compat_flags: int
    compat_ro_flags: int
    incompat_flags: int

    def summary(self) -> DiskSummary:
        regions = (DiskRegion("超级块", SUPERBLOCK_OFFSET, SUPERBLOCK_SIZE),)
        properties = (
            ("卷标", self.label or "未命名"),
            ("UUID", self.fsid),
            ("设备数", f"{self.num_devices}"),
            ("总字节数", f"{self.total_bytes:,}"),
            ("已用字节数", f"{self.bytes_used:,}"),
            ("扇区大小", f"{self.sectorsize} B"),
            ("节点大小", f"{self.nodesize} B"),
            ("条带大小", f"{self.stripesize} B"),
            ("校验和类型", _csum_name(self.csum_type)),
            ("compat 特性", join_names(flag_names(self.compat_flags, COMPAT_FEATURES))),
            (
                "ro_compat 特性",
                join_names(flag_names(self.compat_ro_flags, COMPAT_RO_FEATURES)),
            ),
            (
                "incompat 特性",
                join_names(flag_names(self.incompat_flags, INCOMPAT_FEATURES)),
            ),
        )
        return DiskSummary(
            format_name=FORMAT_NAME,
            block_unit="扇区",
            block_size=self.sectorsize,
            block_count=self.total_bytes // self.sectorsize,
            free_blocks=None,
            regions=regions,
            properties=properties,
        )


def probe(path: str | Path) -> BtrfsSuperblock | None:
    """读取 64 KiB 处的超级块并判定格式; 魔数不符时返回 None.

    Raises:
        DiskFormatError: 布局字段自相矛盾, 或映像短于声明的设备大小.
    """
    with Path(path).open("rb") as fp:
        fp.seek(SUPERBLOCK_OFFSET)
        raw = fp.read(SUPERBLOCK_SIZE)
        if len(raw) < SUPERBLOCK_SIZE or raw[0x40:0x48] != MAGIC:
            return None
        superblock = _parse_superblock(raw)
        fp.seek(0, io.SEEK_END)
        if superblock.total_bytes > fp.tell():
            raise DiskFormatError("映像短于超级块声明的总字节数")
        return superblock


def _parse_superblock(raw: bytes) -> BtrfsSuperblock:
    """解析超级块; 布局字段不合法时抛出 DiskFormatError."""
    bytenr = struct.unpack_from("<Q", raw, 0x30)[0]
    total_bytes = struct.unpack_from("<Q", raw, 0x70)[0]
    bytes_used = struct.unpack_from("<Q", raw, 0x78)[0]
    num_devices = struct.unpack_from("<Q", raw, 0x88)[0]
    sectorsize = struct.unpack_from("<I", raw, 0x90)[0]
    nodesize = struct.unpack_from("<I", raw, 0x94)[0]
    stripesize = struct.unpack_from("<I", raw, 0x9C)[0]
    compat_flags = struct.unpack_from("<Q", raw, 0xAC)[0]
    compat_ro_flags = struct.unpack_from("<Q", raw, 0xB4)[0]
    incompat_flags = struct.unpack_from("<Q", raw, 0xBC)[0]
    csum_type = struct.unpack_from("<H", raw, 0xC4)[0]

    if bytenr != SUPERBLOCK_OFFSET:
        raise DiskFormatError(f"超级块记录的自身地址 {bytenr} 不是 64 KiB")
    if total_bytes == 0:
        raise DiskFormatError("设备字节数为 0")
    if bytes_used > total_bytes:
        raise DiskFormatError("已用字节数大于设备字节数")
    if not is_power_of_two_in(sectorsize, MIN_SECTOR_SIZE, MAX_SECTOR_SIZE):
        raise DiskFormatError(f"扇区大小 {sectorsize} 不合法")
    if not is_power_of_two_in(nodesize, sectorsize, MAX_NODE_SIZE):
        raise DiskFormatError(f"节点大小 {nodesize} 不合法")
    if stripesize == 0 or stripesize & (stripesize - 1):
        raise DiskFormatError(f"条带大小 {stripesize} 不是 2 的幂")
    if num_devices == 0:
        raise DiskFormatError("设备数为 0")

    return BtrfsSuperblock(
        bytenr=bytenr,
        fsid=uuid_text(raw[0x20:0x30]),
        label=bytes_to_utf8(raw[0x12B:0x22B]),
        total_bytes=total_bytes,
        bytes_used=bytes_used,
        sectorsize=sectorsize,
        nodesize=nodesize,
        stripesize=stripesize,
        num_devices=num_devices,
        csum_type=csum_type,
        compat_flags=compat_flags,
        compat_ro_flags=compat_ro_flags,
        incompat_flags=incompat_flags,
    )


def _csum_name(csum_type: int) -> str:
    for value, name in CSUM_TYPES:
        if value == csum_type:
            return name
    return f"未知 {csum_type}"
