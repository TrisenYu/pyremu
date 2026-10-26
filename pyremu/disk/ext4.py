#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""ext2/ext3/ext4 的超级块与块组描述符解析.

超级块位于偏移 1024, 按小端序存储. 块组描述符表紧随首数据块之后, 每项 32 字节,
启用 64bit 特性时另带 32 字节的扩展字段. 本模块只读映像, 不写入任何扇区.
"""

from dataclasses import dataclass
import io
from pathlib import Path
import struct
from typing import BinaryIO

from pyremu.disk.base import DiskFormatError, DiskRegion, DiskSummary
from pyremu.utils.str_aux import bytes_to_utf8, flag_names, join_names, uuid_text

FORMAT_NAME = "ext4"

SUPERBLOCK_OFFSET = 1024
SUPERBLOCK_SIZE = 1024
MAGIC = 0xEF53
MAGIC_OFFSET = 0x38

# 超级块与块组描述符中的位域
_INCOMPAT_64BIT = 0x80
_INCOMPAT_META_BG = 0x10
_STATE_CLEAN = 0x1
_STATE_ERRORS = 0x2
_STATE_RECOVERING = 0x4

# 块大小指数的上界: 2^(10 + 6) = 64 KiB
_MAX_LOG_BLOCK_SIZE = 6

# 块组描述符的两种长度: 基础字段与 INCOMPAT_64BIT 之后的扩展字段
_DESC_SIZE_BASE = 32
_DESC_SIZE_64BIT = 64
_DESC_SIZE_MAX = 128

COMPAT_FEATURES: tuple[tuple[int, str], ...] = (
    (0x1, "dir_prealloc"),
    (0x2, "imagic_inodes"),
    (0x4, "has_journal"),
    (0x8, "ext_attr"),
    (0x10, "resize_inode"),
    (0x20, "dir_index"),
    (0x40, "lazy_bg"),
    (0x80, "exclude_inode"),
    (0x100, "exclude_bitmap"),
    (0x200, "sparse_super2"),
    (0x400, "fast_commit"),
    (0x1000, "stable_inodes"),
    (0x2000, "orphan_file"),
)

INCOMPAT_FEATURES: tuple[tuple[int, str], ...] = (
    (0x1, "compression"),
    (0x2, "filetype"),
    (0x4, "recover"),
    (0x8, "journal_dev"),
    (0x10, "meta_bg"),
    (0x40, "extents"),
    (0x80, "64bit"),
    (0x100, "mmp"),
    (0x200, "flex_bg"),
    (0x400, "ea_inode"),
    (0x1000, "dirdata"),
    (0x2000, "csum_seed"),
    (0x4000, "largedir"),
    (0x8000, "inline_data"),
    (0x10000, "encrypt"),
    (0x20000, "casefold"),
)

RO_COMPAT_FEATURES: tuple[tuple[int, str], ...] = (
    (0x1, "sparse_super"),
    (0x2, "large_file"),
    (0x4, "btree_dir"),
    (0x8, "huge_file"),
    (0x10, "gdt_csum"),
    (0x20, "dir_nlink"),
    (0x40, "extra_isize"),
    (0x100, "quota"),
    (0x200, "bigalloc"),
    (0x400, "metadata_csum"),
    (0x800, "replica"),
    (0x1000, "read-only"),
    (0x2000, "project"),
    (0x8000, "verity"),
    (0x10000, "orphan_present"),
)


@dataclass(frozen=True)
class Ext4Group:
    """一个块组的布局与占用统计."""

    index: int
    first_block: int
    block_count: int
    free_blocks: int
    inode_table_block: int
    free_inodes: int


@dataclass(frozen=True)
class Ext4Superblock:
    """超级块中的布局字段与文件系统标识."""

    block_size: int
    blocks_count: int
    free_blocks_count: int
    inodes_count: int
    free_inodes_count: int
    first_data_block: int
    blocks_per_group: int
    inodes_per_group: int
    inode_size: int
    desc_size: int
    rev_level: int
    volume_name: str
    uuid: str
    last_mounted: str
    state: int
    feature_compat: int
    feature_incompat: int
    feature_ro_compat: int

    @property
    def group_count(self) -> int:
        """块组数: 首数据块之外的块按每块组的块数向上取整."""
        span = self.blocks_count - self.first_data_block
        return max(1, (span + self.blocks_per_group - 1) // self.blocks_per_group)

    @property
    def descriptor_table_offset(self) -> int:
        """块组描述符表的字节偏移: 首个数据块之后的第一块."""
        return (self.first_data_block + 1) * self.block_size

    @property
    def has_64bit_descriptor(self) -> bool:
        """块组描述符是否带 64 位的扩展字段."""
        return (
            bool(self.feature_incompat & _INCOMPAT_64BIT)
            and self.desc_size >= _DESC_SIZE_64BIT
        )

    @property
    def state_name(self) -> str:
        names = []
        if self.state & _STATE_CLEAN:
            names.append("clean")
        if self.state & _STATE_ERRORS:
            names.append("errors")
        if self.state & _STATE_RECOVERING:
            names.append("recovering")
        return ", ".join(names) if names else f"未知 0x{self.state:x}"

    @property
    def compat_names(self) -> list[str]:
        return flag_names(self.feature_compat, COMPAT_FEATURES)

    @property
    def incompat_names(self) -> list[str]:
        return flag_names(self.feature_incompat, INCOMPAT_FEATURES)

    @property
    def ro_compat_names(self) -> list[str]:
        return flag_names(self.feature_ro_compat, RO_COMPAT_FEATURES)

    def summary(self) -> DiskSummary:
        properties = (
            ("卷标", self.volume_name or "未命名"),
            ("UUID", self.uuid),
            ("状态", self.state_name),
            ("修订号", f"{self.rev_level}"),
            ("块组数", f"{self.group_count:,}"),
            ("每块组块数", f"{self.blocks_per_group:,}"),
            ("每块组 inode 数", f"{self.inodes_per_group:,}"),
            ("inode 大小", f"{self.inode_size} B"),
            ("inode 数", f"{self.inodes_count:,} (空闲 {self.free_inodes_count:,})"),
            ("最后挂载点", self.last_mounted or "-"),
            ("compat 特性", join_names(self.compat_names)),
            ("incompat 特性", join_names(self.incompat_names)),
            ("ro_compat 特性", join_names(self.ro_compat_names)),
        )
        regions = (
            DiskRegion("超级块", SUPERBLOCK_OFFSET, SUPERBLOCK_SIZE),
            DiskRegion(
                "块组描述符表",
                self.descriptor_table_offset,
                self.desc_size * self.group_count,
            ),
        )
        return DiskSummary(
            format_name=FORMAT_NAME,
            block_unit="块",
            block_size=self.block_size,
            block_count=self.blocks_count,
            free_blocks=self.free_blocks_count,
            regions=regions,
            properties=properties,
        )


def probe(path: str | Path) -> Ext4Superblock | None:
    """读取超级块并判定格式; 映像不足一个超级块或魔数不符时返回 None.

    Raises:
        DiskFormatError: 块大小指数超出范围, 几何参数自相矛盾, 或使用 meta_bg 布局.
    """
    with Path(path).open("rb") as fp:
        raw = _read_superblock(fp)
    if raw is None or struct.unpack_from("<H", raw, MAGIC_OFFSET)[0] != MAGIC:
        return None
    return _parse_superblock(raw)


def read_groups(path: str | Path, sb: Ext4Superblock) -> list[Ext4Group]:
    """读取块组描述符表, 按块组序号升序返回.

    Raises:
        DiskFormatError: 超级块魔数不符, 或描述符表超出文件范围.
    """
    with Path(path).open("rb") as fp:
        raw = _read_superblock(fp)
        if raw is None or struct.unpack_from("<H", raw, MAGIC_OFFSET)[0] != MAGIC:
            raise DiskFormatError("超级块魔数不符")
        table_offset = sb.descriptor_table_offset
        table_size = sb.desc_size * sb.group_count
        fp.seek(0, io.SEEK_END)
        if table_offset + table_size > fp.tell():
            raise DiskFormatError("块组描述符表超出文件范围")
        fp.seek(table_offset)
        table = fp.read(table_size)
    if len(table) < table_size:
        raise DiskFormatError("块组描述符表超出文件范围")
    return [
        _parse_group_descriptor(
            table[index * sb.desc_size : (index + 1) * sb.desc_size], sb, index
        )
        for index in range(sb.group_count)
    ]


def _read_superblock(fp: BinaryIO) -> bytes | None:
    """从偏移 1024 处读取超级块; 映像不足一个超级块时返回 None."""
    fp.seek(SUPERBLOCK_OFFSET)
    raw = fp.read(SUPERBLOCK_SIZE)
    return raw if len(raw) == SUPERBLOCK_SIZE else None


def _parse_superblock(raw: bytes) -> Ext4Superblock:
    """解析超级块; 布局字段不合法时抛出 DiskFormatError."""
    log_block_size = struct.unpack_from("<I", raw, 0x18)[0]
    if log_block_size > _MAX_LOG_BLOCK_SIZE:
        raise DiskFormatError(f"块大小指数 {log_block_size} 超出范围")

    feature_incompat = struct.unpack_from("<I", raw, 0x60)[0]
    if feature_incompat & _INCOMPAT_META_BG:
        raise DiskFormatError("块组描述符采用 meta_bg 布局, 暂不支持")

    blocks_per_group = struct.unpack_from("<I", raw, 0x20)[0]
    inodes_per_group = struct.unpack_from("<I", raw, 0x28)[0]
    if blocks_per_group == 0:
        raise DiskFormatError("每块组块数为 0")
    if inodes_per_group == 0:
        raise DiskFormatError("每块组 inode 数为 0")

    is_64bit = bool(feature_incompat & _INCOMPAT_64BIT)
    desc_size = _DESC_SIZE_BASE
    if is_64bit:
        raw_desc_size = struct.unpack_from("<H", raw, 0xFE)[0]
        if _DESC_SIZE_BASE <= raw_desc_size <= _DESC_SIZE_MAX:
            desc_size = raw_desc_size
        else:
            desc_size = _DESC_SIZE_64BIT

    blocks_count = struct.unpack_from("<I", raw, 0x04)[0]
    free_blocks_count = struct.unpack_from("<I", raw, 0x0C)[0]
    if is_64bit:
        blocks_count |= struct.unpack_from("<I", raw, 0x150)[0] << 32
        free_blocks_count |= struct.unpack_from("<I", raw, 0x158)[0] << 32
    first_data_block = struct.unpack_from("<I", raw, 0x14)[0]
    if blocks_count <= first_data_block:
        raise DiskFormatError("总块数不大于首数据块")
    if free_blocks_count > blocks_count:
        raise DiskFormatError("空闲块数大于总块数")

    return Ext4Superblock(
        block_size=1024 << log_block_size,
        blocks_count=blocks_count,
        free_blocks_count=free_blocks_count,
        inodes_count=struct.unpack_from("<I", raw, 0x00)[0],
        free_inodes_count=struct.unpack_from("<I", raw, 0x10)[0],
        first_data_block=first_data_block,
        blocks_per_group=blocks_per_group,
        inodes_per_group=inodes_per_group,
        inode_size=struct.unpack_from("<H", raw, 0x58)[0] or 128,
        desc_size=desc_size,
        rev_level=struct.unpack_from("<I", raw, 0x4C)[0],
        volume_name=bytes_to_utf8(raw[0x78:0x88]),
        uuid=uuid_text(raw[0x68:0x78]),
        last_mounted=bytes_to_utf8(raw[0x88:0xC8]),
        state=struct.unpack_from("<H", raw, 0x3A)[0],
        feature_compat=struct.unpack_from("<I", raw, 0x5C)[0],
        feature_incompat=feature_incompat,
        feature_ro_compat=struct.unpack_from("<I", raw, 0x64)[0],
    )


def _parse_group_descriptor(desc: bytes, sb: Ext4Superblock, index: int) -> Ext4Group:
    """解析第 *index* 个块组描述符."""
    first_block = sb.first_data_block + index * sb.blocks_per_group
    free_blocks = struct.unpack_from("<H", desc, 0x0C)[0]
    free_inodes = struct.unpack_from("<H", desc, 0x0E)[0]
    inode_table = struct.unpack_from("<I", desc, 0x08)[0]
    if sb.has_64bit_descriptor:
        inode_table |= struct.unpack_from("<I", desc, 0x28)[0] << 32
        free_blocks |= struct.unpack_from("<H", desc, 0x2C)[0] << 16
        free_inodes |= struct.unpack_from("<H", desc, 0x2E)[0] << 16
    return Ext4Group(
        index=index,
        first_block=first_block,
        block_count=min(sb.blocks_per_group, sb.blocks_count - first_block),
        free_blocks=free_blocks,
        inode_table_block=inode_table,
        free_inodes=free_inodes,
    )
