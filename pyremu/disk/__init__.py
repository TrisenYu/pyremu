#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""磁盘映像格式的只读解析.

每种格式占一个模块, 模块提供 ``FORMAT_NAME``、``probe`` 与格式自己的读取函数.
新增格式时把模块加入 ``FORMAT_MODULES``, 即参与自动判定.
"""

from pathlib import Path

from pyremu.disk import apfs, btrfs, exfat, ext4, fat, hfsplus, ntfs, xfs
from pyremu.disk.base import (
    DiskFormatError,
    DiskLayout,
    DiskRegion,
    DiskSummary,
)

FORMAT_MODULES = (exfat, fat, ntfs, hfsplus, apfs, btrfs, xfs, ext4)


def probe_image(path: str | Path) -> DiskLayout | None:
    """按登记顺序判定映像格式, 返回首个匹配的布局; 都不匹配时返回 None.

    Raises:
        DiskFormatError: 某个格式的结构数据自相矛盾.
    """
    for module in FORMAT_MODULES:
        payload = module.probe(path)
        if payload is not None:
            return DiskLayout(summary=payload.summary(), payload=payload)
    return None


__all__ = [
    "FORMAT_MODULES",
    "DiskFormatError",
    "DiskLayout",
    "DiskRegion",
    "DiskSummary",
    "apfs",
    "btrfs",
    "exfat",
    "ext4",
    "fat",
    "hfsplus",
    "ntfs",
    "probe_image",
    "xfs",
]
