#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""FAT12/FAT16/FAT32 的引导扇区与 FAT 表解析.

引导扇区给出几何参数, 第一个 FAT 表中的空条目数给出空闲簇数.
本模块只读映像, 不写入任何扇区.
"""

from array import array
from dataclasses import dataclass, replace
import io
from pathlib import Path
import struct
from typing import BinaryIO

from pyremu.disk.base import DiskFormatError, DiskRegion, DiskSummary

FORMAT_NAME = "fat"

BOOT_SECTOR_SIZE = 512
EXTENDED_BOOT_SIGNATURE = b"\x55\xaa"
VALID_SECTOR_SIZES = (512, 1024, 2048, 4096)
MAX_SECTORS_PER_CLUSTER = 128
FAT12_CLUSTER_LIMIT = 4085
FAT16_CLUSTER_LIMIT = 65525
ROOT_CLUSTER_MIN = 2
MIN_MEDIA_DESCRIPTOR = 0xF8
FLOPPY_MEDIA_DESCRIPTOR = 0xF0

# 统计空条目时分块读取 FAT 表, 每块 4 MiB
FAT_SCAN_CHUNK = 4 << 20

# 按条目宽度选取 array 的元素类型, 用于逐条目统计空值
_FAT_WIDTH_TYPECODE = {2: "H", 4: "I"}


@dataclass(frozen=True)
class FatSuperblock:
    """引导扇区与 FAT 表中的布局字段."""

    fat_type: str
    bytes_per_sector: int
    sectors_per_cluster: int
    reserved_sectors: int
    fat_count: int
    root_entry_count: int
    total_sectors: int
    fat_sectors: int
    root_cluster: int
    media_descriptor: int
    volume_id: int
    volume_label: str
    cluster_count: int
    free_clusters: int

    @property
    def cluster_bytes(self) -> int:
        return self.bytes_per_sector * self.sectors_per_cluster

    @property
    def root_dir_sectors(self) -> int:
        """根目录区占用的扇区数, 仅 FAT12 与 FAT16 有该项."""
        entry_bytes = self.root_entry_count * 32
        return (entry_bytes + self.bytes_per_sector - 1) // self.bytes_per_sector

    @property
    def data_start_sector(self) -> int:
        fat_span = self.fat_count * self.fat_sectors
        return self.reserved_sectors + fat_span + self.root_dir_sectors

    def summary(self) -> DiskSummary:
        regions = [
            DiskRegion("引导扇区", 0, BOOT_SECTOR_SIZE),
            DiskRegion(
                "FAT 表",
                self.reserved_sectors * self.bytes_per_sector,
                self.fat_count * self.fat_sectors * self.bytes_per_sector,
            ),
        ]
        if self.fat_type != "FAT32":
            regions.append(
                DiskRegion(
                    "根目录区",
                    (self.reserved_sectors + self.fat_count * self.fat_sectors)
                    * self.bytes_per_sector,
                    self.root_entry_count * 32,
                )
            )
        regions.append(
            DiskRegion(
                "数据区",
                self.data_start_sector * self.bytes_per_sector,
                (self.total_sectors - self.data_start_sector) * self.bytes_per_sector,
            )
        )

        properties = (
            ("卷标", self.volume_label or "未命名"),
            ("文件系统类型", self.fat_type),
            ("卷序列号", f"0x{self.volume_id:08x}"),
            ("每扇区字节数", f"{self.bytes_per_sector}"),
            ("每簇扇区数", f"{self.sectors_per_cluster}"),
            ("总扇区数", f"{self.total_sectors:,}"),
            ("簇数", f"{self.cluster_count:,}"),
            ("空闲簇", f"{self.free_clusters:,}"),
            ("保留扇区数", f"{self.reserved_sectors}"),
            ("FAT 表数", f"{self.fat_count}"),
            ("每个 FAT 表的扇区数", f"{self.fat_sectors:,}"),
            ("介质描述符", f"0x{self.media_descriptor:02x}"),
        )
        if self.fat_type == "FAT32":
            properties += (("根目录起始簇", f"{self.root_cluster}"),)
        else:
            properties += (("根目录项数", f"{self.root_entry_count}"),)

        return DiskSummary(
            format_name=FORMAT_NAME,
            block_unit="簇",
            block_size=self.cluster_bytes,
            block_count=self.cluster_count,
            free_blocks=self.free_clusters,
            regions=tuple(regions),
            properties=properties,
        )


def probe(path: str | Path) -> FatSuperblock | None:
    """读取引导扇区与第一个 FAT 表并判定格式; 签名不符时返回 None.

    Raises:
        DiskFormatError: 映像短于引导扇区声明的总扇区数, 或 FAT 表超出映像范围.
    """
    with Path(path).open("rb") as fp:
        boot = fp.read(BOOT_SECTOR_SIZE)
        if len(boot) < BOOT_SECTOR_SIZE:
            return None
        if boot[0] not in (0xEB, 0xE9) or boot[510:512] != EXTENDED_BOOT_SIGNATURE:
            return None
        superblock = _parse_boot_sector(boot)
        if superblock is None:
            return None
        fp.seek(0, io.SEEK_END)
        file_size = fp.tell()
        if superblock.total_sectors * superblock.bytes_per_sector > file_size:
            raise DiskFormatError("映像短于引导扇区声明的总扇区数")
        free_clusters = _count_free_clusters(fp, superblock)
        return replace(superblock, free_clusters=free_clusters)


def _parse_boot_sector(boot: bytes) -> FatSuperblock | None:
    """解析引导扇区; 基本签名与几何参数不像 FAT 时返回 None."""
    (
        bytes_per_sector,
        sectors_per_cluster,
        reserved_sectors,
        fat_count,
        root_entry_count,
        total_sectors_16,
    ) = struct.unpack_from("<HBHBHH", boot, 0x0B)
    media_descriptor = boot[0x15]
    fat_sectors_16 = struct.unpack_from("<H", boot, 0x16)[0]
    total_sectors_32 = struct.unpack_from("<I", boot, 0x20)[0]
    fat_sectors_32 = struct.unpack_from("<I", boot, 0x24)[0]
    total_sectors = total_sectors_16 or total_sectors_32
    fat_sectors = fat_sectors_16 or fat_sectors_32

    is_plausible = (
        bytes_per_sector in VALID_SECTOR_SIZES
        and 0 < sectors_per_cluster <= MAX_SECTORS_PER_CLUSTER
        and (sectors_per_cluster & (sectors_per_cluster - 1)) == 0
        and (
            media_descriptor == FLOPPY_MEDIA_DESCRIPTOR
            or media_descriptor >= MIN_MEDIA_DESCRIPTOR
        )
        and reserved_sectors != 0
        and fat_count in (1, 2)
        and total_sectors != 0
        and fat_sectors != 0
    )
    if not is_plausible:
        return None

    root_dir_sectors = (root_entry_count * 32 + bytes_per_sector - 1) // bytes_per_sector
    data_sectors = (
        total_sectors - reserved_sectors - fat_count * fat_sectors - root_dir_sectors
    )
    if data_sectors <= 0:
        return None
    cluster_count = data_sectors // sectors_per_cluster
    if cluster_count < FAT12_CLUSTER_LIMIT:
        fat_type = "FAT12"
    elif cluster_count < FAT16_CLUSTER_LIMIT:
        fat_type = "FAT16"
    else:
        fat_type = "FAT32"

    if fat_type == "FAT32":
        root_cluster = struct.unpack_from("<I", boot, 0x2C)[0]
        is_valid = (
            root_entry_count == 0
            and fat_sectors_16 == 0
            and fat_sectors_32 != 0
            and ROOT_CLUSTER_MIN <= root_cluster < cluster_count + ROOT_CLUSTER_MIN
        )
        if not is_valid:
            return None
        volume_id = struct.unpack_from("<I", boot, 0x43)[0]
        label = boot[0x47:0x52]
    else:
        if root_entry_count == 0 or fat_sectors_16 == 0:
            return None
        root_cluster = 0
        volume_id = struct.unpack_from("<I", boot, 0x27)[0]
        label = boot[0x2B:0x36]

    return FatSuperblock(
        fat_type=fat_type,
        bytes_per_sector=bytes_per_sector,
        sectors_per_cluster=sectors_per_cluster,
        reserved_sectors=reserved_sectors,
        fat_count=fat_count,
        root_entry_count=root_entry_count,
        total_sectors=total_sectors,
        fat_sectors=fat_sectors,
        root_cluster=root_cluster,
        media_descriptor=media_descriptor,
        volume_id=volume_id,
        volume_label=_decode_label(label),
        cluster_count=cluster_count,
        free_clusters=0,
    )


def _count_free_clusters(fp: BinaryIO, superblock: FatSuperblock) -> int:
    """在第一个 FAT 表中统计值为 0 的条目, 即空闲簇数."""
    entry_count = superblock.cluster_count + ROOT_CLUSTER_MIN
    fp.seek(superblock.reserved_sectors * superblock.bytes_per_sector)
    if superblock.fat_type == "FAT32":
        return _count_zero_words(fp, entry_count * 4, 4)
    if superblock.fat_type == "FAT16":
        return _count_zero_words(fp, entry_count * 2, 2)

    needed = (entry_count * 3 + 1) // 2
    data = fp.read(needed)
    if len(data) < needed:
        raise DiskFormatError("FAT 表超出映像范围")
    free = 0
    for index in range(ROOT_CLUSTER_MIN, entry_count):
        offset = index * 3 // 2
        if index & 1:
            value = (data[offset] >> 4) | (data[offset + 1] << 4)
        else:
            value = data[offset] | ((data[offset + 1] & 0x0F) << 8)
        if value == 0:
            free += 1
    return free


def _count_zero_words(fp: BinaryIO, byte_count: int, width: int) -> int:
    """统计 FAT16 或 FAT32 表中字节数为 *width* 的全零条目.

    分块读取, 每块保留不足一个条目的剩余字节, 条目不会跨块漏计.
    """
    typecode = _FAT_WIDTH_TYPECODE[width]
    free = 0
    tail = b""
    remaining = byte_count
    while remaining > 0:
        chunk = fp.read(min(remaining, FAT_SCAN_CHUNK))
        if not chunk:
            raise DiskFormatError("FAT 表超出映像范围")
        data = tail + chunk
        aligned = len(data) - len(data) % width
        if aligned:
            values = array(typecode)
            values.frombytes(data[:aligned])
            free += values.count(0)
        tail = data[aligned:]
        remaining -= len(chunk)
    if tail:
        raise DiskFormatError("FAT 表长度不是条目宽度的整数倍")
    return free


def _decode_label(raw: bytes) -> str:
    return raw.rstrip(b" ").split(b"\x00", 1)[0].decode("cp437", "replace")
