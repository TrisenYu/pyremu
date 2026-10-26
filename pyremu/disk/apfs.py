#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""APFS 容器超级块解析.

容器超级块位于映像起点, 按小端序存储, 给出块大小、块总数、检查点区域与特性位.
卷名存放于各卷的超级块, 由 pyfsapfs 读取; 空闲块数由空间管理器维护, 本模块不读取该结构,
因此摘要中该值为 None. 本模块只读映像, 不写入任何扇区.
"""

from dataclasses import dataclass, replace
import io
from pathlib import Path
import struct

import pyfsapfs

from pyremu.disk.base import DiskFormatError, DiskRegion, DiskSummary
from pyremu.utils.mask import is_power_of_two_in
from pyremu.utils.str_aux import flag_names, join_names, uuid_text

FORMAT_NAME = "apfs"

SUPERBLOCK_OFFSET = 0
SUPERBLOCK_SIZE = 4096
MAGIC = b"NXSB"
MAGIC_OFFSET = 0x20
MIN_BLOCK_SIZE = 4096
MAX_BLOCK_SIZE = 65536
MAX_FILE_SYSTEMS = 100
FILE_SYSTEM_OID_OFFSET = 0xB8

COMPAT_FEATURES: tuple[tuple[int, str], ...] = (
    (0x1, "defrag"),
    (0x2, "lcfd"),
)

COMPAT_RO_FEATURES: tuple[tuple[int, str], ...] = ()

INCOMPAT_FEATURES: tuple[tuple[int, str], ...] = (
    (0x1, "version1"),
    (0x2, "version2"),
    (0x100, "fusion"),
)


@dataclass(frozen=True)
class ApfsContainerSuperblock:
    """容器超级块中的布局字段与特性位."""

    block_size: int
    block_count: int
    container_identifier: str
    compat_flags: int
    compat_ro_flags: int
    incompat_flags: int
    checkpoint_desc_blocks: int
    checkpoint_data_blocks: int
    checkpoint_desc_base: int
    checkpoint_data_base: int
    space_manager_oid: int
    object_map_oid: int
    reaper_oid: int
    max_file_systems: int
    file_system_oids: tuple[int, ...]
    volume_names: tuple[str, ...]

    def summary(self) -> DiskSummary:
        regions = [DiskRegion("容器超级块", SUPERBLOCK_OFFSET, self.block_size)]
        if self.checkpoint_desc_blocks:
            regions.append(
                DiskRegion(
                    "检查点描述区",
                    self.checkpoint_desc_base * self.block_size,
                    self.checkpoint_desc_blocks * self.block_size,
                )
            )
        if self.checkpoint_data_blocks:
            regions.append(
                DiskRegion(
                    "检查点数据区",
                    self.checkpoint_data_base * self.block_size,
                    self.checkpoint_data_blocks * self.block_size,
                )
            )
        names = [name for name in self.volume_names if name]
        properties = (
            ("卷名", ", ".join(names) if names else "未知"),
            ("容器标识", self.container_identifier),
            ("块大小", f"{self.block_size:,} B"),
            ("块总数", f"{self.block_count:,}"),
            ("卷数", f"{len(self.volume_names)}"),
            ("文件系统对象标识数", f"{len(self.file_system_oids)}"),
            ("最大文件系统数", f"{self.max_file_systems}"),
            ("检查点描述块数", f"{self.checkpoint_desc_blocks:,}"),
            ("检查点数据块数", f"{self.checkpoint_data_blocks:,}"),
            ("空间管理器对象标识", f"{self.space_manager_oid:,}"),
            ("对象映射对象标识", f"{self.object_map_oid:,}"),
            ("回收器对象标识", f"{self.reaper_oid:,}"),
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
            block_unit="块",
            block_size=self.block_size,
            block_count=self.block_count,
            free_blocks=None,
            regions=tuple(regions),
            properties=properties,
        )


def probe(path: str | Path) -> ApfsContainerSuperblock | None:
    """读取容器超级块并判定格式; 魔数不符时返回 None.

    Raises:
        DiskFormatError: 布局字段自相矛盾, 映像短于声明的块总数, 或卷列表无法读取.
    """
    with Path(path).open("rb") as fp:
        raw = fp.read(SUPERBLOCK_SIZE)
        if len(raw) < SUPERBLOCK_SIZE or raw[MAGIC_OFFSET : MAGIC_OFFSET + 4] != MAGIC:
            return None
        superblock = _parse_superblock(raw)
        fp.seek(0, io.SEEK_END)
        if superblock.block_count * superblock.block_size > fp.tell():
            raise DiskFormatError("映像短于容器超级块声明的块总数")
    return replace(superblock, volume_names=_read_volume_names(path))


def _parse_superblock(raw: bytes) -> ApfsContainerSuperblock:
    """解析容器超级块; 布局字段不合法时抛出 DiskFormatError."""
    block_size = struct.unpack_from("<I", raw, 0x24)[0]
    block_count = struct.unpack_from("<Q", raw, 0x28)[0]
    compat_flags = struct.unpack_from("<Q", raw, 0x30)[0]
    compat_ro_flags = struct.unpack_from("<Q", raw, 0x38)[0]
    incompat_flags = struct.unpack_from("<Q", raw, 0x40)[0]
    checkpoint_desc_blocks = struct.unpack_from("<I", raw, 0x68)[0]
    checkpoint_data_blocks = struct.unpack_from("<I", raw, 0x6C)[0]
    checkpoint_desc_base = struct.unpack_from("<Q", raw, 0x70)[0]
    checkpoint_data_base = struct.unpack_from("<Q", raw, 0x78)[0]
    space_manager_oid = struct.unpack_from("<Q", raw, 0x98)[0]
    object_map_oid = struct.unpack_from("<Q", raw, 0xA0)[0]
    reaper_oid = struct.unpack_from("<Q", raw, 0xA8)[0]
    max_file_systems = struct.unpack_from("<I", raw, 0xB4)[0]
    file_system_oids = tuple(
        oid
        for index in range(MAX_FILE_SYSTEMS)
        if (oid := struct.unpack_from("<Q", raw, FILE_SYSTEM_OID_OFFSET + index * 8)[0])
    )

    if not is_power_of_two_in(block_size, MIN_BLOCK_SIZE, MAX_BLOCK_SIZE):
        raise DiskFormatError(f"块大小 {block_size} 不合法")
    if block_count == 0:
        raise DiskFormatError("块总数为 0")
    if max_file_systems > MAX_FILE_SYSTEMS:
        raise DiskFormatError(f"最大文件系统数 {max_file_systems} 超出上限")
    if len(file_system_oids) > max_file_systems:
        raise DiskFormatError("文件系统数大于最大文件系统数")
    if not _is_within(checkpoint_desc_base, checkpoint_desc_blocks, block_count):
        raise DiskFormatError("检查点描述区超出块总数")
    if not _is_within(checkpoint_data_base, checkpoint_data_blocks, block_count):
        raise DiskFormatError("检查点数据区超出块总数")

    return ApfsContainerSuperblock(
        block_size=block_size,
        block_count=block_count,
        container_identifier=uuid_text(raw[0x48:0x58]),
        compat_flags=compat_flags,
        compat_ro_flags=compat_ro_flags,
        incompat_flags=incompat_flags,
        checkpoint_desc_blocks=checkpoint_desc_blocks,
        checkpoint_data_blocks=checkpoint_data_blocks,
        checkpoint_desc_base=checkpoint_desc_base,
        checkpoint_data_base=checkpoint_data_base,
        space_manager_oid=space_manager_oid,
        object_map_oid=object_map_oid,
        reaper_oid=reaper_oid,
        max_file_systems=max_file_systems,
        file_system_oids=file_system_oids,
        volume_names=(),
    )


def _read_volume_names(path: str | Path) -> tuple[str, ...]:
    """经 pyfsapfs 读取各卷的卷名."""
    container = pyfsapfs.container()
    try:
        container.open(str(path))
        return tuple(
            container.get_volume(index).name or ""
            for index in range(container.number_of_volumes)
        )
    except OSError as exc:
        raise DiskFormatError(f"读取卷列表失败: {exc}") from exc
    finally:
        container.close()


def _is_within(base: int, count: int, block_count: int) -> bool:
    return count == 0 or base + count <= block_count
