#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""BinsInfoOps — bins-info: 命令行传入的各映像所占的物理地址范围."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text

from pyremu.debug.attrs import SharedAttrs
from pyremu.debug.utils import fmt_display_name, fmt_size

BINS_USAGE = "用法: bins-info"


class BinsInfoOps(SharedAttrs):
    """列出经命令行参数交给调试器的映像及其物理区间.

    每一项对应一个入参: 位置参数 firmware, 以及 --preload / --kernel /
    --initrd / --fdt-file 中实际给出的那些. 未给出的入参不出现在表中.
    """

    def cmd_bins_info(self, rest: str) -> None:
        if rest.strip():
            self._warn(BINS_USAGE)
            return
        self._print_ram_caption()
        images = sorted(
            self._bins_images.items(),
            key=lambda item: (item[1].start, item[1].end),
        )
        if not images:
            self._console.print(Text("没有经命令行加载的映像", style="dim"))
            return

        tbl = Table(title="命令行映像的物理地址范围", border_style="blue")
        tbl.add_column("映像", style="cyan", justify="center", no_wrap=True)
        tbl.add_column("参数", justify="center", no_wrap=True)
        tbl.add_column("起始", justify="center", no_wrap=True)
        tbl.add_column("结束", justify="center", no_wrap=True)
        tbl.add_column("大小", justify="center")
        tbl.add_column("路径", justify="center")
        for kind, image in images:
            tbl.add_row(
                Text(kind.label),
                Text(kind.option),
                Text(f"0x{image.start:016x}"),
                Text(f"0x{image.end:016x}"),
                Text(fmt_size(image.size), justify="center"),
                Text(fmt_display_name(image.path)),
            )
        self._console.print(tbl)
        self._console.print(Text("区间按 [起始, 结束) 计; 结束为不包含的上界", style="dim"))

    def _print_ram_caption(self) -> None:
        bus = self._emu.bus
        self._console.print(
            Text(
                f"RAM 0x{bus.ram_base:016x} .. "
                f"0x{bus.ram_base + bus.ram_size:016x} "
                f"({fmt_size(bus.ram_size)})",
                style="dim",
            )
        )
