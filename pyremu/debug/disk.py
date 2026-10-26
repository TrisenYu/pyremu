#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""DiskViewOps — disk-info: 磁盘映像的格式摘要与块布局."""

from __future__ import annotations

from pathlib import Path

from rich.markup import escape
from rich.table import Table
from rich.text import Text

from pyremu.debug.attrs import SharedAttrs
from pyremu.debug.exception import UsageError
from pyremu.debug.sym.cmd_files import parse_name_arg
from pyremu.debug.utils import fmt_display_name, fmt_size
from pyremu.disk import DiskFormatError, DiskLayout, ext4, probe_image

DISK_USAGE = "用法: disk-info [<映像路径>]"
GROUP_ROWS_LIMIT = 128
GROUP_HEAD_ROWS = 96
GROUP_TAIL_ROWS = 8


class DiskViewOps(SharedAttrs):
    """解析磁盘映像的块布局; 省略路径时取 --disk 挂载的镜像."""

    def cmd_disk_info(self, rest: str) -> None:
        try:
            path_text = parse_name_arg(rest)
        except UsageError as exc:
            self._err(escape(str(exc)))
            return
        self._print_disk_layout(path_text)

    def _print_disk_layout(self, path_text: str) -> None:
        path = path_text or self._attached_disk_path()
        if not path:
            self._err(f"未挂载磁盘映像; {DISK_USAGE}")
            return
        if not Path(path).is_file():
            self._err(f"文件不存在: {fmt_display_name(path)}")
            return

        try:
            layout = probe_image(path)
        except DiskFormatError as exc:
            self._err(escape(str(exc)))
            return
        except OSError as exc:
            self._err(escape(f"无法读取 {fmt_display_name(path)}: {exc.strerror or exc}"))
            return
        if layout is None:
            self._warn(f"未识别的映像格式: {fmt_display_name(path)}")
            return

        self._print_image_summary(path, layout)
        self._print_regions(layout)
        if isinstance(layout.payload, ext4.Ext4Superblock):
            self._print_ext4_groups(path, layout.payload)

    def _attached_disk_path(self) -> str:
        """已挂载的 virtio-blk 镜像路径; 未挂载时返回空串."""
        blk = self._emu.virtio_blk
        return blk.image_path if blk is not None else ""

    def _print_image_summary(self, path: str, layout: DiskLayout) -> None:
        summary = layout.summary
        try:
            file_size = Path(path).stat().st_size
        except OSError:
            file_size = 0
        blocks = summary.block_count
        unit = summary.block_unit
        if summary.free_blocks is None:
            free_text = "未知"
        else:
            free_ratio = summary.free_blocks * 100.0 / blocks if blocks else 0.0
            free_text = f"{summary.free_blocks:,} ({free_ratio:.1f}%)"
        rows = [
            ("路径", path),
            ("格式", summary.format_name),
            ("文件大小", f"{fmt_size(file_size)} ({file_size:,} B)"),
            (f"{unit}大小", f"{summary.block_size:,} B"),
            (f"{unit}数", f"{blocks:,}"),
            (f"空闲{unit}", free_text),
        ]
        blk = self._emu.virtio_blk
        if blk is not None and blk.image_path == path:
            rows.append(("挂载方式", "只读" if blk.read_only else "读写"))
        tbl = Table(title="磁盘映像", border_style="blue", show_header=False)
        tbl.add_column("项", style="cyan", justify="right", no_wrap=True)
        tbl.add_column("值", justify="left")
        for key, value in rows:
            tbl.add_row(Text(key), Text(value))
        self._console.print(tbl)

        props = Table(title="文件系统属性", border_style="blue", show_header=False)
        props.add_column("项", style="cyan", justify="right", no_wrap=True)
        props.add_column("值", justify="left")
        for key, value in summary.properties:
            props.add_row(Text(key), Text(value))
        self._console.print(props)

    def _print_regions(self, layout: DiskLayout) -> None:
        tbl = Table(title="映像布局", border_style="blue")
        tbl.add_column("区域", style="cyan", justify="center", no_wrap=True)
        tbl.add_column("起始", justify="right", no_wrap=True)
        tbl.add_column("大小", justify="right", no_wrap=True)
        for region in layout.summary.regions:
            tbl.add_row(
                Text(region.name),
                Text(f"0x{region.byte_offset:x}"),
                Text(f"{region.byte_size:,} B", justify="right"),
            )
        self._console.print(tbl)

    def _print_ext4_groups(self, path: str, sb: ext4.Ext4Superblock) -> None:
        try:
            groups = ext4.read_groups(path, sb)
        except DiskFormatError as exc:
            self._err(escape(str(exc)))
            return
        except OSError as exc:
            self._err(escape(f"无法读取块组描述符表: {exc.strerror or exc}"))
            return

        tbl = Table(title="ext4 块组", border_style="blue")
        tbl.add_column("组", style="cyan", justify="right")
        tbl.add_column("首块", justify="right")
        tbl.add_column("块数", justify="right")
        tbl.add_column("空闲块", justify="right")
        tbl.add_column("空闲 inode", justify="right")
        tbl.add_column("inode 表块", justify="right")
        head = groups
        tail: list[ext4.Ext4Group] = []
        if len(groups) > GROUP_ROWS_LIMIT:
            head = groups[:GROUP_HEAD_ROWS]
            tail = groups[-GROUP_TAIL_ROWS:]
        for group in head:
            self._add_group_row(tbl, group)
        if tail:
            omitted = len(groups) - len(head) - len(tail)
            tbl.add_row(
                Text("…"),
                Text(f"其余 {omitted} 个块组未列出"),
                Text(""),
                Text(""),
                Text(""),
                Text(""),
            )
            for group in tail:
                self._add_group_row(tbl, group)
        self._console.print(tbl)

    @staticmethod
    def _add_group_row(tbl: Table, group: ext4.Ext4Group) -> None:
        tbl.add_row(
            Text(f"{group.index}"),
            Text(f"{group.first_block:,}"),
            Text(f"{group.block_count:,}"),
            Text(f"{group.free_blocks:,}"),
            Text(f"{group.free_inodes:,}"),
            Text(f"{group.inode_table_block:,}"),
        )
