#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""字符串格式化辅助 — 地址、hexdump 等可复用输出格式."""


def fmt_addr(addr: int) -> str:
    """64 位地址 -> 0x 前缀固定 16 位 hex 字符串."""
    return f"0x{addr:016x}"


def fmt_hexdump(data: bytes, addr: int = 0, *, columns: int = 16) -> str:
    """字节序列 -> 经典 hexdump 文本 (地址 | hex | ASCII).

    Args:
        data: 原始字节.
        addr: 起始地址 (用于地址列显示).
        columns: 每行字节数 (默认 16).
    """
    lines: list[str] = []
    for offset in range(0, len(data), columns):
        chunk = data[offset : offset + columns]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(
            f"  {fmt_addr(addr + offset)}  {hex_part:<{columns * 3 - 1}s}  |{ascii_part}|"
        )
    return "\n".join(lines)
