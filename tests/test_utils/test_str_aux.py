#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""测试 pyremu.utils.str_aux 的地址与 hexdump 文本格式."""

from pyremu.utils.str_aux import fmt_addr, fmt_hexdump


class TestFmtAddr:
    def test_zero_fills_sixteen_digits(self):
        assert fmt_addr(0) == "0x0000000000000000"

    def test_high_bit_of_low_word(self):
        assert fmt_addr(0x8000_0000) == "0x0000000080000000"

    def test_all_sixty_four_bits_set(self):
        assert fmt_addr((1 << 64) - 1) == "0xffffffffffffffff"

    def test_lowercase_digits(self):
        assert fmt_addr(0xABCDEF) == "0x0000000000abcdef"


class TestFmtHexdump:
    def test_empty_input_yields_empty_text(self):
        assert fmt_hexdump(b"") == ""

    def test_full_line_holds_sixteen_bytes(self):
        expected = (
            "  0x0000000000001000  00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f"
            "  |................|"
        )
        assert fmt_hexdump(bytes(range(16)), 0x1000) == expected

    def test_short_line_pads_hex_column(self):
        expected = "  0x0000000000000000  00                                               |.|"
        assert fmt_hexdump(b"\x00") == expected

    def test_addr_advances_by_columns(self):
        text = fmt_hexdump(b"A" * 32, 0x2000)
        lines = text.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("  0x0000000000002000  ")
        assert lines[1].startswith("  0x0000000000002010  ")

    def test_custom_columns(self):
        expected = (
            "  0x0000000080000000  48 65 6c 6c 6f 2c 20 52  |Hello, R|\n"
            "  0x0000000080000008  49 53 43 2d 56 21        |ISC-V!|"
        )
        assert fmt_hexdump(b"Hello, RISC-V!", 0x8000_0000, columns=8) == expected

    def test_ascii_column_keeps_printable_range(self):
        """可打印区间为 0x20 至 0x7E; 0x1F 与 0x7F 各自落在区间之外."""
        text = fmt_hexdump(bytes([0x1F, 0x20, 0x7E, 0x7F, 0x80]))
        assert text.endswith("|. ~..|")
