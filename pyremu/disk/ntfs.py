#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""NTFS 的引导扇区解析与 $Bitmap 统计.

引导扇区给出几何参数与主文件表的位置, 主文件表与 $Bitmap 数据流的解析由 pyfsntfs 完成,
$Bitmap 中值为 1 的位给出已分配簇数. 本模块只读映像, 不写入任何扇区.
"""

from dataclasses import dataclass, replace
import io
from pathlib import Path
import struct
from typing import BinaryIO

import pyfsntfs

from pyremu.disk.base import DiskFormatError, DiskRegion, DiskSummary

FORMAT_NAME = "ntfs"

BOOT_SECTOR_SIZE = 512
OEM_NAME = b"NTFS    "
EXTENDED_BOOT_SIGNATURE = b"\x55\xaa"
VALID_SECTOR_SIZES = (512, 1024, 2048, 4096)
MAX_CLUSTER_BYTES = 2 << 20
BITMAP_PATH = "$Bitmap"

# 读取 $Bitmap 时每次至多 1 MiB
BITMAP_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class NtfsBootSector:
    """引导扇区与 $Bitmap 中的布局字段."""

    bytes_per_sector: int
    sectors_per_cluster: int
    total_sectors: int
    mft_cluster: int
    mft_mirror_cluster: int
    volume_serial: int
    cluster_count: int
    free_clusters: int
    volume_label: str
    mft_entry_size: int
    index_entry_size: int
    mft_record_count: int

    @property
    def cluster_bytes(self) -> int:
        return self.bytes_per_sector * self.sectors_per_cluster

    def summary(self) -> DiskSummary:
        regions = (
            DiskRegion("引导扇区", 0, BOOT_SECTOR_SIZE),
            DiskRegion("簇堆", 0, self.cluster_count * self.cluster_bytes),
        )
        properties = (
            ("卷标", self.volume_label or "未命名"),
            ("卷序列号", f"0x{self.volume_serial:016x}"),
            ("每扇区字节数", f"{self.bytes_per_sector}"),
            ("每簇扇区数", f"{self.sectors_per_cluster}"),
            ("总扇区数", f"{self.total_sectors:,}"),
            ("簇数", f"{self.cluster_count:,}"),
            ("空闲簇", f"{self.free_clusters:,}"),
            ("MFT 起始簇", f"{self.mft_cluster:,}"),
            ("MFT 镜像起始簇", f"{self.mft_mirror_cluster:,}"),
            ("MFT 记录大小", f"{self.mft_entry_size} B"),
            ("索引记录大小", f"{self.index_entry_size} B"),
            ("MFT 记录数", f"{self.mft_record_count:,}"),
        )
        return DiskSummary(
            format_name=FORMAT_NAME,
            block_unit="簇",
            block_size=self.cluster_bytes,
            block_count=self.cluster_count,
            free_blocks=self.free_clusters,
            regions=regions,
            properties=properties,
        )


def probe(path: str | Path) -> NtfsBootSector | None:
    """读取引导扇区与 $Bitmap 并判定格式; 签名不符时返回 None.

    Raises:
        DiskFormatError: 几何参数自相矛盾, 主文件表或 $Bitmap 无法读取.
    """
    with Path(path).open("rb") as fp:
        boot = fp.read(BOOT_SECTOR_SIZE)
        if len(boot) < BOOT_SECTOR_SIZE:
            return None
        if boot[3:11] != OEM_NAME or boot[510:512] != EXTENDED_BOOT_SIGNATURE:
            return None
        superblock = _parse_boot_sector(boot)
        fp.seek(0, io.SEEK_END)
        file_size = fp.tell()
        if superblock.total_sectors * superblock.bytes_per_sector > file_size:
            raise DiskFormatError("映像短于引导扇区声明的总扇区数")
        return _read_volume_fields(fp, superblock)


def _parse_boot_sector(boot: bytes) -> NtfsBootSector:
    """解析引导扇区; 几何参数不合法时抛出 DiskFormatError."""
    bytes_per_sector = struct.unpack_from("<H", boot, 0x0B)[0]
    sectors_per_cluster = boot[0x0D]
    total_sectors = struct.unpack_from("<Q", boot, 0x28)[0]
    mft_cluster = struct.unpack_from("<Q", boot, 0x30)[0]
    mft_mirror_cluster = struct.unpack_from("<Q", boot, 0x38)[0]
    volume_serial = struct.unpack_from("<Q", boot, 0x48)[0]

    if bytes_per_sector not in VALID_SECTOR_SIZES:
        raise DiskFormatError(f"每扇区字节数 {bytes_per_sector} 不合法")
    if sectors_per_cluster == 0 or sectors_per_cluster & (sectors_per_cluster - 1):
        raise DiskFormatError(f"每簇扇区数 {sectors_per_cluster} 不是 2 的幂")
    if bytes_per_sector * sectors_per_cluster > MAX_CLUSTER_BYTES:
        raise DiskFormatError("簇大小超出 NTFS 允许的范围")
    if total_sectors == 0:
        raise DiskFormatError("总扇区数为 0")

    cluster_count = total_sectors // sectors_per_cluster
    if cluster_count == 0:
        raise DiskFormatError("簇数为 0")
    if mft_cluster >= cluster_count or mft_mirror_cluster >= cluster_count:
        raise DiskFormatError("主文件表的起始簇超出簇堆范围")

    return NtfsBootSector(
        bytes_per_sector=bytes_per_sector,
        sectors_per_cluster=sectors_per_cluster,
        total_sectors=total_sectors,
        mft_cluster=mft_cluster,
        mft_mirror_cluster=mft_mirror_cluster,
        volume_serial=volume_serial,
        cluster_count=cluster_count,
        free_clusters=0,
        volume_label="",
        mft_entry_size=0,
        index_entry_size=0,
        mft_record_count=0,
    )


def _read_volume_fields(fp: BinaryIO, superblock: NtfsBootSector) -> NtfsBootSector:
    """经 pyfsntfs 读取卷标、主文件表参数与 $Bitmap 的已分配簇数."""
    volume = pyfsntfs.volume()
    try:
        volume.open_file_object(fp)
        entry = volume.get_file_entry_by_path(BITMAP_PATH)
        if entry is None:
            raise DiskFormatError(f"缺少 {BITMAP_PATH} 文件记录")
        if not entry.has_default_data_stream():
            raise DiskFormatError(f"{BITMAP_PATH} 没有默认数据流")
        allocated = _count_allocated_clusters(entry, superblock.cluster_count)
        return replace(
            superblock,
            volume_label=volume.name or "",
            mft_entry_size=volume.mft_entry_size,
            index_entry_size=volume.index_entry_size,
            mft_record_count=volume.number_of_file_entries,
            free_clusters=superblock.cluster_count - allocated,
        )
    except OSError as exc:
        raise DiskFormatError(f"读取主文件表失败: {exc}") from exc
    finally:
        volume.close()


def _count_allocated_clusters(entry: pyfsntfs.file_entry, cluster_count: int) -> int:
    """在 $Bitmap 的前 *cluster_count* 位中统计值为 1 的位."""
    allocated = 0
    offset = 0
    remaining = cluster_count
    while remaining > 0:
        byte_count = min((remaining + 7) // 8, BITMAP_CHUNK_BYTES)
        data = entry.read_buffer_at_offset(byte_count, offset)
        if len(data) < byte_count:
            raise DiskFormatError(f"{BITMAP_PATH} 短于簇数所需")
        bits = min(remaining, byte_count * 8)
        allocated += (int.from_bytes(data, "little") & ((1 << bits) - 1)).bit_count()
        offset += byte_count
        remaining -= bits
    return allocated
