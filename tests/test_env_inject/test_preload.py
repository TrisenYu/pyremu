#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""测试 pyremu.env_inject.preload — shellcode 注入位置与文件读取."""

import pytest

from pyremu.emulator import Emulator
from pyremu.env_inject.preload import Preloader

RAM_BASE = 0x8000_0000
RAM_SIZE = 0x0004_0000
SHELLCODE = b"\x13\x00\x00\x00\x73\x00\x10\x00"


def _make_emu(num_harts: int = 1) -> Emulator:
    """建立 RAM 足以容纳默认注入位置的 Emulator."""
    emu = Emulator(num_harts=num_harts, ram_size=RAM_SIZE, ram_base=RAM_BASE)
    emu._termio = None
    return emu


class TestInject:
    def test_default_address_sits_below_ram_top(self):
        emu = _make_emu()
        entry = Preloader(emu).inject(SHELLCODE)
        assert entry == (RAM_BASE + RAM_SIZE - Preloader.DEFAULT_ADDR_OFFSET) & ~0xF
        assert emu.bus.try_read(entry, len(SHELLCODE)) == SHELLCODE

    def test_explicit_address_is_used_verbatim(self):
        """显式地址不做对齐处理, 按调用方给定值写入."""
        emu = _make_emu()
        addr = RAM_BASE + 0x1002
        entry = Preloader(emu).inject(SHELLCODE, addr)
        assert entry == addr
        assert emu.bus.try_read(addr, len(SHELLCODE)) == SHELLCODE

    def test_every_hart_is_untouched_by_injection(self):
        """注入只写内存, 不改变 hart 的程序计数器."""
        emu = _make_emu(num_harts=2)
        for hart in emu.harts:
            hart.pc = RAM_BASE
        Preloader(emu).inject(SHELLCODE)
        assert [h.pc for h in emu.harts] == [RAM_BASE, RAM_BASE]

    def test_empty_shellcode_returns_address(self):
        emu = _make_emu()
        entry = Preloader(emu).inject(b"")
        assert entry == (RAM_BASE + RAM_SIZE - Preloader.DEFAULT_ADDR_OFFSET) & ~0xF


class TestInjectFile:
    def test_file_contents_are_injected(self, tmp_path):
        src = tmp_path / "crt0.bin"
        src.write_bytes(SHELLCODE)
        emu = _make_emu()
        entry = Preloader(emu).inject_file(str(src))
        assert entry == (RAM_BASE + RAM_SIZE - Preloader.DEFAULT_ADDR_OFFSET) & ~0xF
        assert emu.bus.try_read(entry, len(SHELLCODE)) == SHELLCODE

    def test_explicit_address_is_forwarded(self, tmp_path):
        src = tmp_path / "crt0.bin"
        src.write_bytes(SHELLCODE)
        emu = _make_emu()
        addr = RAM_BASE + 0x2000
        assert Preloader(emu).inject_file(str(src), addr) == addr

    def test_missing_file_raises(self, tmp_path):
        emu = _make_emu()
        missing = tmp_path / "none.bin"
        with pytest.raises(FileNotFoundError) as exc_info:
            Preloader(emu).inject_file(str(missing))
        assert str(missing) in str(exc_info.value)

    def test_directory_is_rejected(self, tmp_path):
        emu = _make_emu()
        with pytest.raises(FileNotFoundError):
            Preloader(emu).inject_file(str(tmp_path))
