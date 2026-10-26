#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""HFS+ 与 HFSX 的卷头解析.

卷头位于偏移 1024, 按大端序存储, 给出分配块大小、块总数、空闲块数与卷属性.
卷标存放于目录文件, 由 pyfshfs 读取. 本模块只读映像, 不写入任何扇区.
"""

from dataclasses import dataclass, replace
import io
from pathlib import Path
import struct

import pyfshfs

from pyremu.disk.base import DiskFormatError, DiskRegion, DiskSummary
from pyremu.utils.mask import is_power_of_two_in
from pyremu.utils.str_aux import flag_names, join_names

FORMAT_NAME = "hfsplus"

VOLUME_HEADER_OFFSET = 1024
VOLUME_HEADER_SIZE = 512
VALID_SIGNATURES = {b"H+": 4, b"HX": 5}
MIN_BLOCK_SIZE = 512
MAX_BLOCK_SIZE = 65536
EXTENT_COUNT = 8

VOLUME_ATTRIBUTES: tuple[tuple[int, str], ...] = (
    (0x80, "硬件写锁"),
    (0x100, "已正常卸载"),
    (0x200, "使用备用块"),
    (0x400, "不需要缓存"),
    (0x800, "引导卷内容不一致"),
    (0x1000, "目录节点标识已复用"),
    (0x2000, "日志卷"),
    (0x8000, "软件写锁"),
)


@dataclass(frozen=True)
class HfsplusVolumeHeader:
    """卷头中的布局字段与卷属性."""

    signature: bytes
    version: int
    attributes: int
    last_mounted_version: int
    journal_info_block: int
    file_count: int
    folder_count: int
    block_size: int
    block_count: int
    free_blocks: int
    next_allocation: int
    next_catalog_id: int
    write_count: int
    allocation_total_blocks: int
    volume_identifier: str
    volume_label: str
    allocation_extents: tuple[tuple[int, int], ...]

    @property
    def is_case_sensitive(self) -> bool:
        return self.signature == b"HX"

    def summary(self) -> DiskSummary:
        regions = [DiskRegion("卷头", VOLUME_HEADER_OFFSET, VOLUME_HEADER_SIZE)]
        for start_block, block_count in self.allocation_extents:
            regions.append(
                DiskRegion(
                    "分配位图",
                    start_block * self.block_size,
                    block_count * self.block_size,
                )
            )
        properties = (
            ("卷标", self.volume_label or "未命名"),
            ("文件系统类型", "HFSX" if self.is_case_sensitive else "HFS+"),
            ("卷标识", self.volume_identifier or "未设置"),
            ("卷属性", join_names(flag_names(self.attributes, VOLUME_ATTRIBUTES))),
            ("上次挂载版本", _fourcc_text(self.last_mounted_version)),
            ("日志信息块", f"{self.journal_info_block:,}"),
            ("文件数", f"{self.file_count:,}"),
            ("文件夹数", f"{self.folder_count:,}"),
            ("块大小", f"{self.block_size:,} B"),
            ("总块数", f"{self.block_count:,}"),
            ("空闲块", f"{self.free_blocks:,}"),
            ("下一个分配块", f"{self.next_allocation:,}"),
            ("下一个目录节点标识", f"{self.next_catalog_id:,}"),
            ("写入次数", f"{self.write_count:,}"),
            ("分配位图块数", f"{self.allocation_total_blocks:,}"),
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


def probe(path: str | Path) -> HfsplusVolumeHeader | None:
    """读取卷头并判定格式; 签名不符时返回 None.

    Raises:
        DiskFormatError: 布局字段自相矛盾, 映像短于声明的总块数, 或目录文件无法读取.
    """
    with Path(path).open("rb") as fp:
        fp.seek(VOLUME_HEADER_OFFSET)
        raw = fp.read(VOLUME_HEADER_SIZE)
        if len(raw) < VOLUME_HEADER_SIZE or raw[:2] not in VALID_SIGNATURES:
            return None
        header = _parse_volume_header(raw)
        fp.seek(0, io.SEEK_END)
        if header.block_count * header.block_size > fp.tell():
            raise DiskFormatError("映像短于卷头声明的总块数")
    return replace(header, volume_label=_read_volume_label(path))


def _parse_volume_header(raw: bytes) -> HfsplusVolumeHeader:
    """解析卷头; 布局字段不合法时抛出 DiskFormatError."""
    signature = raw[:2]
    version = struct.unpack_from(">H", raw, 0x02)[0]
    if version != VALID_SIGNATURES[signature]:
        raise DiskFormatError(f"签名 {signature!r} 与版本 {version} 不一致")
    attributes = struct.unpack_from(">I", raw, 0x04)[0]
    last_mounted_version = struct.unpack_from(">I", raw, 0x08)[0]
    journal_info_block = struct.unpack_from(">I", raw, 0x0C)[0]
    file_count = struct.unpack_from(">I", raw, 0x20)[0]
    folder_count = struct.unpack_from(">I", raw, 0x24)[0]
    block_size = struct.unpack_from(">I", raw, 0x28)[0]
    block_count = struct.unpack_from(">I", raw, 0x2C)[0]
    free_blocks = struct.unpack_from(">I", raw, 0x30)[0]
    next_allocation = struct.unpack_from(">I", raw, 0x34)[0]
    next_catalog_id = struct.unpack_from(">I", raw, 0x40)[0]
    write_count = struct.unpack_from(">I", raw, 0x44)[0]
    allocation_total_blocks = struct.unpack_from(">I", raw, 0x7C)[0]

    if not is_power_of_two_in(block_size, MIN_BLOCK_SIZE, MAX_BLOCK_SIZE):
        raise DiskFormatError(f"块大小 {block_size} 不合法")
    if block_count == 0:
        raise DiskFormatError("总块数为 0")
    if free_blocks > block_count:
        raise DiskFormatError("空闲块数大于总块数")

    allocation_extents = _parse_allocation_extents(raw, block_count)
    extent_blocks = sum(count for _start, count in allocation_extents)
    if extent_blocks > allocation_total_blocks:
        raise DiskFormatError("分配位图的区段块数大于其总块数")

    volume_identifier = raw[0x68:0x70].hex()
    if volume_identifier == "0" * 16:
        volume_identifier = ""

    return HfsplusVolumeHeader(
        signature=signature,
        version=version,
        attributes=attributes,
        last_mounted_version=last_mounted_version,
        journal_info_block=journal_info_block,
        file_count=file_count,
        folder_count=folder_count,
        block_size=block_size,
        block_count=block_count,
        free_blocks=free_blocks,
        next_allocation=next_allocation,
        next_catalog_id=next_catalog_id,
        write_count=write_count,
        allocation_total_blocks=allocation_total_blocks,
        volume_identifier=volume_identifier,
        volume_label="",
        allocation_extents=allocation_extents,
    )


def _parse_allocation_extents(raw: bytes, block_count: int) -> tuple[tuple[int, int], ...]:
    """解析分配位图文件的前 8 个区段; 区段超出总块数时抛出 DiskFormatError."""
    extents: list[tuple[int, int]] = []
    for index in range(EXTENT_COUNT):
        start_block, count = struct.unpack_from(">II", raw, 0x80 + index * 8)
        if count == 0:
            continue
        if start_block + count > block_count:
            raise DiskFormatError("分配位图的区段超出总块数")
        extents.append((start_block, count))
    return tuple(extents)


def _read_volume_label(path: str | Path) -> str:
    """经 pyfshfs 读取卷标."""
    volume = pyfshfs.volume()
    try:
        volume.open(str(path))
        return volume.name or ""
    except OSError as exc:
        raise DiskFormatError(f"读取目录文件失败: {exc}") from exc
    finally:
        volume.close()


def _fourcc_text(value: int) -> str:
    """把四字符编码转成文本; 不能打印时以十六进制给出."""
    if value == 0:
        return "未挂载"
    raw = value.to_bytes(4, "big")
    if all(0x20 <= byte < 0x7F for byte in raw):
        return raw.decode("ascii")
    return f"0x{value:08x}"
