#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""exFAT 的引导扇区、根目录与分配位图解析.

引导扇区给出几何参数, 根目录中的分配位图条目给出位图位置, 位图中的 0 位给出空闲簇数.
根目录与分配位图在 FAT 表中按簇链(cluster chain)组织, 即每个簇的 FAT 表条目给出下一个簇.
本模块只读映像, 不写入任何扇区.
"""

from dataclasses import dataclass, replace
import io
from pathlib import Path
import struct
from typing import BinaryIO

from pyremu.disk.base import DiskFormatError, DiskRegion, DiskSummary
from pyremu.utils.str_aux import flag_names, join_names

FORMAT_NAME = "exfat"

BOOT_SECTOR_SIZE = 512
OEM_NAME = b"EXFAT   "
EXTENDED_BOOT_SIGNATURE = b"\x55\xaa"

MIN_SECTOR_SHIFT = 9
MAX_SECTOR_SHIFT = 12
MAX_CLUSTER_SHIFT = 25
MIN_FAT_OFFSET_SECTORS = 24
FIRST_CLUSTER = 2

ENTRY_END = 0x00
ENTRY_ALLOCATION_BITMAP = 0x81
ENTRY_VOLUME_LABEL = 0x83
MAX_LABEL_CHARS = 11

FAT_BAD_CLUSTER = 0xFFFFFFF7
FAT_EOC_MIN = 0xFFFFFFF8

VOLUME_FLAGS: tuple[tuple[int, str], ...] = (
    (0x1, "使用第二个 FAT 表"),
    (0x2, "上次未正常卸载"),
    (0x4, "介质故障"),
    (0x8, "卷需要清零"),
)


@dataclass(frozen=True)
class ExfatSuperblock:
    """引导扇区与分配位图中的布局字段."""

    bytes_per_sector: int
    sectors_per_cluster: int
    fat_offset_sectors: int
    fat_length_sectors: int
    cluster_heap_offset_sectors: int
    cluster_count: int
    root_cluster: int
    volume_length_sectors: int
    partition_offset: int
    volume_serial: int
    revision_major: int
    revision_minor: int
    volume_flags: int
    fat_count: int
    volume_label: str
    free_clusters: int | None
    bitmap_byte_offset: int | None
    bitmap_byte_size: int

    @property
    def cluster_bytes(self) -> int:
        return self.bytes_per_sector * self.sectors_per_cluster

    def summary(self) -> DiskSummary:
        regions = [
            DiskRegion("引导扇区", 0, BOOT_SECTOR_SIZE),
            DiskRegion(
                "FAT 表",
                self.fat_offset_sectors * self.bytes_per_sector,
                self.fat_length_sectors * self.bytes_per_sector,
            ),
        ]
        if self.bitmap_byte_offset is not None:
            regions.append(
                DiskRegion("分配位图", self.bitmap_byte_offset, self.bitmap_byte_size)
            )
        regions.append(
            DiskRegion(
                "簇堆",
                self.cluster_heap_offset_sectors * self.bytes_per_sector,
                self.cluster_count * self.cluster_bytes,
            )
        )

        free_text = "未知" if self.free_clusters is None else f"{self.free_clusters:,}"
        properties = (
            ("卷标", self.volume_label or "未命名"),
            ("文件系统修订号", f"{self.revision_major}.{self.revision_minor}"),
            ("卷序列号", f"0x{self.volume_serial:08x}"),
            ("卷标志", join_names(flag_names(self.volume_flags, VOLUME_FLAGS))),
            ("每扇区字节数", f"{self.bytes_per_sector}"),
            ("每簇扇区数", f"{self.sectors_per_cluster}"),
            ("总扇区数", f"{self.volume_length_sectors:,}"),
            ("簇数", f"{self.cluster_count:,}"),
            ("空闲簇", free_text),
            ("FAT 表数", f"{self.fat_count}"),
            ("FAT 起始扇区", f"{self.fat_offset_sectors:,}"),
            ("簇堆起始扇区", f"{self.cluster_heap_offset_sectors:,}"),
            ("分区偏移", f"{self.partition_offset:,} B"),
        )
        return DiskSummary(
            format_name=FORMAT_NAME,
            block_unit="簇",
            block_size=self.cluster_bytes,
            block_count=self.cluster_count,
            free_blocks=self.free_clusters,
            regions=tuple(regions),
            properties=properties,
        )


def probe(path: str | Path) -> ExfatSuperblock | None:
    """读取引导扇区、根目录与分配位图并判定格式; 签名不符时返回 None.

    Raises:
        DiskFormatError: 几何参数自相矛盾, 或根目录与分配位图超出映像范围.
    """
    with Path(path).open("rb") as fp:
        boot = fp.read(BOOT_SECTOR_SIZE)
        if len(boot) < BOOT_SECTOR_SIZE:
            return None
        if boot[3:11] != OEM_NAME or any(boot[11:64]):
            return None
        if boot[510:512] != EXTENDED_BOOT_SIGNATURE:
            return None
        superblock = _parse_boot_sector(boot)
        fp.seek(0, io.SEEK_END)
        file_size = fp.tell()
        if superblock.volume_length_sectors * superblock.bytes_per_sector > file_size:
            raise DiskFormatError("映像短于引导扇区声明的扇区总数")

        label, bitmap = _scan_root_directory(fp, superblock)
        free_clusters = _read_free_clusters(fp, superblock, bitmap) if bitmap else None
        bitmap_offset = _cluster_offset(superblock, bitmap[0]) if bitmap else None
        return replace(
            superblock,
            volume_label=label,
            free_clusters=free_clusters,
            bitmap_byte_offset=bitmap_offset,
            bitmap_byte_size=bitmap[1] if bitmap else 0,
        )


def _parse_boot_sector(boot: bytes) -> ExfatSuperblock:
    """解析引导扇区; 几何参数不合法时抛出 DiskFormatError."""
    partition_offset = struct.unpack_from("<Q", boot, 0x40)[0]
    volume_length = struct.unpack_from("<Q", boot, 0x48)[0]
    fat_offset, fat_length, heap_offset, cluster_count = struct.unpack_from(
        "<IIII", boot, 0x50
    )
    root_cluster = struct.unpack_from("<I", boot, 0x60)[0]
    volume_serial = struct.unpack_from("<I", boot, 0x64)[0]
    revision = struct.unpack_from("<H", boot, 0x68)[0]
    volume_flags = struct.unpack_from("<H", boot, 0x6A)[0]
    sector_shift = boot[0x6C]
    cluster_shift = boot[0x6D]
    fat_count = boot[0x6E]

    if not MIN_SECTOR_SHIFT <= sector_shift <= MAX_SECTOR_SHIFT:
        raise DiskFormatError(f"每扇区字节数的指数 {sector_shift} 超出范围")
    if sector_shift + cluster_shift > MAX_CLUSTER_SHIFT:
        raise DiskFormatError("簇大小超出 exFAT 允许的范围")
    if fat_count not in (1, 2):
        raise DiskFormatError(f"FAT 表数 {fat_count} 不合法")
    if fat_offset < MIN_FAT_OFFSET_SECTORS or fat_length == 0:
        raise DiskFormatError("FAT 表的位置或长度不合法")
    if heap_offset < fat_offset + fat_length:
        raise DiskFormatError("簇堆起点位于 FAT 表之前")
    if cluster_count == 0:
        raise DiskFormatError("簇数为 0")
    if heap_offset + cluster_count * (1 << cluster_shift) > volume_length:
        raise DiskFormatError("簇堆超出引导扇区声明的扇区总数")
    if not FIRST_CLUSTER <= root_cluster < cluster_count + FIRST_CLUSTER:
        raise DiskFormatError("根目录起始簇超出簇堆范围")

    return ExfatSuperblock(
        bytes_per_sector=1 << sector_shift,
        sectors_per_cluster=1 << cluster_shift,
        fat_offset_sectors=fat_offset,
        fat_length_sectors=fat_length,
        cluster_heap_offset_sectors=heap_offset,
        cluster_count=cluster_count,
        root_cluster=root_cluster,
        volume_length_sectors=volume_length,
        partition_offset=partition_offset,
        volume_serial=volume_serial,
        revision_major=(revision >> 8) & 0xFF,
        revision_minor=revision & 0xFF,
        volume_flags=volume_flags,
        fat_count=fat_count,
        volume_label="",
        free_clusters=None,
        bitmap_byte_offset=None,
        bitmap_byte_size=0,
    )


def _scan_root_directory(
    fp: BinaryIO, superblock: ExfatSuperblock
) -> tuple[str, tuple[int, int] | None]:
    """沿簇链遍历根目录, 返回卷标与分配位图条目的起始簇和字节数."""
    label = ""
    bitmap: tuple[int, int] | None = None
    cluster = superblock.root_cluster
    visited: set[int] = set()
    while True:
        if not FIRST_CLUSTER <= cluster < superblock.cluster_count + FIRST_CLUSTER:
            raise DiskFormatError("根目录簇链超出簇堆范围")
        if cluster in visited:
            raise DiskFormatError("根目录簇链中同一簇第二次出现, 判定为环")
        visited.add(cluster)

        data = _read_cluster(fp, superblock, cluster)
        for offset in range(0, len(data), 32):
            entry_type = data[offset]
            if entry_type == ENTRY_END:
                return label, bitmap
            if entry_type == ENTRY_ALLOCATION_BITMAP and bitmap is None:
                bitmap = struct.unpack_from("<IQ", data, offset + 20)
            elif entry_type == ENTRY_VOLUME_LABEL and not label:
                length = min(data[offset + 1], MAX_LABEL_CHARS)
                raw = data[offset + 2 : offset + 2 + length * 2]
                label = raw.decode("utf-16-le", "replace")

        cluster = _next_cluster(fp, superblock, cluster)
        if cluster >= FAT_EOC_MIN:
            return label, bitmap


def _read_free_clusters(
    fp: BinaryIO, superblock: ExfatSuperblock, bitmap: tuple[int, int]
) -> int:
    """读取分配位图并统计 0 位, 即空闲簇数."""
    first_cluster, byte_size = bitmap
    if not FIRST_CLUSTER <= first_cluster < superblock.cluster_count + FIRST_CLUSTER:
        raise DiskFormatError("分配位图起始簇超出簇堆范围")
    if byte_size == 0:
        raise DiskFormatError("分配位图长度为 0")

    data = _read_cluster_chain(fp, superblock, first_cluster, byte_size)
    full_bytes = superblock.cluster_count // 8
    remainder = superblock.cluster_count % 8
    needed = full_bytes + (1 if remainder else 0)
    if len(data) < needed:
        raise DiskFormatError("分配位图短于簇数所需")

    used = sum(byte.bit_count() for byte in data[:full_bytes])
    if remainder:
        used += (data[full_bytes] & ((1 << remainder) - 1)).bit_count()
    return superblock.cluster_count - used


def _read_cluster_chain(
    fp: BinaryIO, superblock: ExfatSuperblock, first_cluster: int, byte_size: int
) -> bytes:
    """沿簇链读取至多 *byte_size* 字节."""
    chunks: list[bytes] = []
    remaining = byte_size
    cluster = first_cluster
    visited: set[int] = set()
    while remaining > 0:
        if not FIRST_CLUSTER <= cluster < superblock.cluster_count + FIRST_CLUSTER:
            raise DiskFormatError("簇链超出簇堆范围")
        if cluster in visited:
            raise DiskFormatError("簇链中同一簇第二次出现, 判定为环")
        visited.add(cluster)

        data = _read_cluster(fp, superblock, cluster)
        take = min(remaining, len(data))
        chunks.append(data[:take])
        remaining -= take
        if remaining == 0:
            break
        cluster = _next_cluster(fp, superblock, cluster)
        if cluster >= FAT_EOC_MIN:
            raise DiskFormatError("簇链短于声明的字节数")
    return b"".join(chunks)


def _read_cluster(fp: BinaryIO, superblock: ExfatSuperblock, cluster: int) -> bytes:
    fp.seek(_cluster_offset(superblock, cluster))
    data = fp.read(superblock.cluster_bytes)
    if len(data) < superblock.cluster_bytes:
        raise DiskFormatError("簇超出映像范围")
    return data


def _cluster_offset(superblock: ExfatSuperblock, cluster: int) -> int:
    sector = (
        superblock.cluster_heap_offset_sectors
        + (cluster - FIRST_CLUSTER) * superblock.sectors_per_cluster
    )
    return sector * superblock.bytes_per_sector


def _next_cluster(fp: BinaryIO, superblock: ExfatSuperblock, cluster: int) -> int:
    """读取 FAT 表中 *cluster* 的下一簇编号."""
    fp.seek(superblock.fat_offset_sectors * superblock.bytes_per_sector + cluster * 4)
    raw = fp.read(4)
    if len(raw) < 4:
        raise DiskFormatError("FAT 表超出映像范围")
    value = struct.unpack("<I", raw)[0]
    if value == FAT_BAD_CLUSTER:
        raise DiskFormatError("簇链包含坏簇")
    return value
