#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""字符串格式化辅助 — 地址、hexdump 等可复用输出格式."""


def join_names(names: list[str], *, empty: str = "-") -> str:
    """把名称列表以逗号分隔连接; 列表为空时返回 *empty*."""
    return ", ".join(names) if names else empty


def flag_names(value: int, table: tuple[tuple[int, str], ...]) -> list[str]:
    """按位取出 *table* 中登记的名称; 未登记的高位以十六进制附在末尾."""
    names = [name for bit, name in table if value & bit]
    known = 0
    for bit, _name in table:
        known |= bit
    unknown = value & ~known
    if unknown:
        names.append(f"unknown=0x{unknown:x}")
    return names


def bytes_to_utf8(raw: bytes) -> str:
    """按第一个 0 字节截断, 再以 UTF-8 解码, 无法解码的字节替换为 U+FFFD."""
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")


def uuid_text(raw: bytes) -> str:
    """把 16 字节标识按其自然分段写成带连字符的十六进制文本."""
    text = raw.hex()
    parts = (text[:8], text[8:12], text[12:16], text[16:20], text[20:])
    return "-".join(parts)


def fmt_addr(addr: int) -> str:
    """64 位地址转换为固定 16 位、带 0x 前缀的十六进制字符串."""
    return f"0x{addr:016x}"


def fmt_hexdump(data: bytes, addr: int = 0, *, columns: int = 16) -> str:
    """把字节序列转换为经典 hexdump 文本, 依次为地址列、十六进制列与 ASCII 列.

    Args:
        data: 原始字节.
        addr: 起始地址, 用于地址列显示.
        columns: 每行字节数, 默认为 16.
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


def dye_hexdump(data: bytes, addr: int = 0, *, columns: int = 16) -> str:
    """把字节序列转换为带 Rich 样式标记的 hexdump 文本.

    地址列用亮蓝色, ASCII 列用暗灰色.

    Args:
        data: 原始字节.
        addr: 起始地址, 用于地址列显示.
        columns: 每行字节数, 默认为 16.
    """
    lines: list[str] = []
    for offset in range(0, len(data), columns):
        chunk = data[offset : offset + columns]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        # ASCII 列写成 '[[' 后输出, 使 Rich 按字面显示该字符, 不把它解析为
        # 样式标记的开头.
        ascii_part = ascii_part.replace("[", "[[")
        lines.append(
            f"  [bright_blue]{fmt_addr(addr + offset)}[/]  "
            f"{hex_part:<{columns * 3 - 1}s}  [dim]|{ascii_part}|[/]"
        )
    return "\n".join(lines)
