#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""磁盘映像格式解析的共用类型.

各格式模块提供 ``FORMAT_NAME``、``probe`` 与 ``summary``, 由包入口按登记顺序调用.
"""

from dataclasses import dataclass


class DiskFormatError(Exception):
    """映像不符合所声明的格式, 或布局数据超出映像范围."""


@dataclass(frozen=True)
class DiskRegion:
    """映像中的一段连续区域, 按字节偏移给出."""

    name: str
    byte_offset: int
    byte_size: int


@dataclass(frozen=True)
class DiskSummary:
    """与格式无关的布局摘要.

    *block_unit* 是该格式分配单元的名称, 例如 ext4 的块, FAT 的簇与 btrfs 的扇区.
    *free_blocks* 为该分配单元的空闲数量; 该格式的元数据不给出空闲数量时为 None.
    *properties* 是该格式自己的名值对, 按显示顺序排列.
    """

    format_name: str
    block_unit: str
    block_size: int
    block_count: int
    free_blocks: int | None
    regions: tuple[DiskRegion, ...]
    properties: tuple[tuple[str, str], ...]

    @property
    def size_bytes(self) -> int:
        return self.block_size * self.block_count

    @property
    def free_bytes(self) -> int | None:
        if self.free_blocks is None:
            return None
        return self.block_size * self.free_blocks


@dataclass(frozen=True)
class DiskLayout:
    """一次格式判别得到的布局.

    *payload* 为该格式自己的数据结构, 只有对应格式模块能解释它.
    """

    summary: DiskSummary
    payload: object
