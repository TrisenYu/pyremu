#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""XFS 的超级块解析.

超级块位于偏移 0, 按大端序存储, 给出块大小、块总数、空闲块数与特性位.
本模块只读映像, 不写入任何扇区.
"""

from dataclasses import dataclass
import io
from pathlib import Path
import struct

from pyremu.disk.base import DiskFormatError, DiskRegion, DiskSummary
from pyremu.utils.mask import is_power_of_two_in
from pyremu.utils.str_aux import bytes_to_utf8, flag_names, join_names, uuid_text

FORMAT_NAME = "xfs"

SUPERBLOCK_OFFSET = 0
SUPERBLOCK_SIZE = 512
MAGIC = b"XFSB"

MIN_BLOCK_SIZE = 512
MAX_BLOCK_SIZE = 65536
MIN_SECTOR_SIZE = 512
MAX_SECTOR_SIZE = 32768
VALID_INODE_SIZES = (256, 512, 1024, 2048)
VALID_VERSIONS = (4, 5)

RO_COMPAT_FEATURES: tuple[tuple[int, str], ...] = (
    (0x1, "finobt"),
    (0x2, "rmapbt"),
    (0x4, "reflink"),
    (0x8, "inobtcnt"),
)

INCOMPAT_FEATURES: tuple[tuple[int, str], ...] = (
    (0x1, "ftype"),
    (0x2, "sparse_inodes"),
    (0x4, "meta_uuid"),
    (0x8, "bigtime"),
    (0x10, "needsrepair"),
    (0x20, "nrext64"),
    (0x40, "exchrange"),
    (0x80, "parent"),
)


@dataclass(frozen=True)
class XfsSuperblock:
    """超级块中的布局字段与特性位."""

    version: int
    block_size: int
    sector_size: int
    inode_size: int
    inodes_per_block: int
    block_count: int
    free_blocks: int
    realtime_blocks: int
    allocation_group_count: int
    blocks_per_group: int
    log_start_block: int
    log_block_count: int
    inode_count: int
    free_inode_count: int
    volume_label: str
    uuid: str
    ro_compat_flags: int
    incompat_flags: int

    def summary(self) -> DiskSummary:
        regions = [DiskRegion("超级块", SUPERBLOCK_OFFSET, SUPERBLOCK_SIZE)]
        if self.log_start_block > 0 and self.log_block_count > 0:
            regions.append(
                DiskRegion(
                    "日志区",
                    self.log_start_block * self.block_size,
                    self.log_block_count * self.block_size,
                )
            )
        properties = (
            ("卷标", self.volume_label or "未命名"),
            ("UUID", self.uuid),
            ("版本", f"{self.version}"),
            ("块大小", f"{self.block_size:,} B"),
            ("扇区大小", f"{self.sector_size} B"),
            ("inode 大小", f"{self.inode_size} B"),
            ("每块 inode 数", f"{self.inodes_per_block}"),
            ("分配组数", f"{self.allocation_group_count:,}"),
            ("每分配组块数", f"{self.blocks_per_group:,}"),
            ("日志起始块", f"{self.log_start_block:,}"),
            ("日志块数", f"{self.log_block_count:,}"),
            ("实时块数", f"{self.realtime_blocks:,}"),
            ("inode 数", f"{self.inode_count:,} (空闲 {self.free_inode_count:,})"),
            (
                "ro_compat 特性",
                join_names(flag_names(self.ro_compat_flags, RO_COMPAT_FEATURES)),
            ),
            (
                "incompat 特性",
                join_names(flag_names(self.incompat_flags, INCOMPAT_FEATURES)),
            ),
        )
        return DiskSummary(
            format_name=FORMAT_NAME,
            block_unit="块",
            block_size=self.block_size,
            block_count=self.block_count,
            free_blocks=self.free_blocks,
            regions=tuple(regions),
            properties=properties,
        )


def probe(path: str | Path) -> XfsSuperblock | None:
    """读取偏移 0 处的超级块并判定格式; 魔数不符时返回 None.

    Raises:
        DiskFormatError: 布局字段自相矛盾, 或映像短于声明的块总数.
    """
    with Path(path).open("rb") as fp:
        raw = fp.read(SUPERBLOCK_SIZE)
        if len(raw) < SUPERBLOCK_SIZE or raw[0:4] != MAGIC:
            return None
        superblock = _parse_superblock(raw)
        fp.seek(0, io.SEEK_END)
        if superblock.block_count * superblock.block_size > fp.tell():
            raise DiskFormatError("映像短于超级块声明的块总数")
        return superblock


def _parse_superblock(raw: bytes) -> XfsSuperblock:
    """解析超级块; 布局字段不合法时抛出 DiskFormatError."""
    block_size = struct.unpack_from(">I", raw, 0x04)[0]
    block_count = struct.unpack_from(">Q", raw, 0x08)[0]
    realtime_blocks = struct.unpack_from(">Q", raw, 0x10)[0]
    log_start_block = struct.unpack_from(">Q", raw, 0x30)[0]
    blocks_per_group = struct.unpack_from(">I", raw, 0x54)[0]
    allocation_group_count = struct.unpack_from(">I", raw, 0x58)[0]
    log_block_count = struct.unpack_from(">I", raw, 0x60)[0]
    version_number = struct.unpack_from(">H", raw, 0x64)[0]
    sector_size = struct.unpack_from(">H", raw, 0x66)[0]
    inode_size = struct.unpack_from(">H", raw, 0x68)[0]
    inodes_per_block = struct.unpack_from(">H", raw, 0x6A)[0]
    inode_count = struct.unpack_from(">Q", raw, 0x80)[0]
    free_inode_count = struct.unpack_from(">Q", raw, 0x88)[0]
    free_blocks = struct.unpack_from(">Q", raw, 0x90)[0]
    ro_compat_flags = struct.unpack_from(">I", raw, 0xD4)[0]
    incompat_flags = struct.unpack_from(">I", raw, 0xD8)[0]

    version = version_number & 0x0F
    if version not in VALID_VERSIONS:
        raise DiskFormatError(f"版本号 {version} 不合法")
    if not is_power_of_two_in(block_size, MIN_BLOCK_SIZE, MAX_BLOCK_SIZE):
        raise DiskFormatError(f"块大小 {block_size} 不合法")
    if not is_power_of_two_in(sector_size, MIN_SECTOR_SIZE, MAX_SECTOR_SIZE):
        raise DiskFormatError(f"扇区大小 {sector_size} 不合法")
    if inode_size not in VALID_INODE_SIZES:
        raise DiskFormatError(f"inode 大小 {inode_size} 不合法")
    if raw[0x78] != block_size.bit_length() - 1:
        raise DiskFormatError("块大小的指数与块大小不一致")
    if raw[0x79] != sector_size.bit_length() - 1:
        raise DiskFormatError("扇区大小的指数与扇区大小不一致")
    if inodes_per_block == 0 or inodes_per_block * inode_size != block_size:
        raise DiskFormatError("每块 inode 数与块大小不一致")
    if block_count == 0 or allocation_group_count == 0 or blocks_per_group == 0:
        raise DiskFormatError("块总数或分配组数为 0")
    if free_blocks > block_count:
        raise DiskFormatError("空闲块数大于块总数")
    if free_inode_count > inode_count:
        raise DiskFormatError("空闲 inode 数大于 inode 数")
    if log_start_block > 0 and (
        log_start_block >= block_count or log_start_block + log_block_count > block_count
    ):
        raise DiskFormatError("日志区超出块总数")

    return XfsSuperblock(
        version=version,
        block_size=block_size,
        sector_size=sector_size,
        inode_size=inode_size,
        inodes_per_block=inodes_per_block,
        block_count=block_count,
        free_blocks=free_blocks,
        realtime_blocks=realtime_blocks,
        allocation_group_count=allocation_group_count,
        blocks_per_group=blocks_per_group,
        log_start_block=log_start_block,
        log_block_count=log_block_count,
        inode_count=inode_count,
        free_inode_count=free_inode_count,
        volume_label=bytes_to_utf8(raw[0x6C:0x78]),
        uuid=uuid_text(raw[0x20:0x30]),
        ro_compat_flags=ro_compat_flags,
        incompat_flags=incompat_flags,
    )
