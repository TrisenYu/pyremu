#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""测试 pyremu.debug.sym.elf_idx — ELF 头部校验、符号索引与名称查找."""

import os
import struct

import pytest

from pyremu.debug.exception import ElfFormatError
from pyremu.debug.sym import elf_idx
from tests.test_debugger.sym_fixtures import (
    build_elf,
    build_symbol_elf,
    build_symtab_from_records,
    ELF_HEADER_SIZE,
    HeaderSpec,
    LayoutSpec,
    pack_symbols,
    ProgramSpec,
    PT_LOAD,
    SECTION_HEADER_SIZE,
    SectionSpec,
    SHF_ALLOC,
    SHF_EXECINSTR,
    SHT_DYNSYM,
    SHT_NULL,
    SHT_PROGBITS,
    SHT_STRTAB,
    SHT_SYMTAB,
    SymTabSpec,
)

GLOBAL_FUNC = 0x12
LOCAL_NOTYPE = 0x00
LOCAL_FUNC = 0x02
GLOBAL_NOTYPE = 0x10

SH_OFFSET = 24
SH_SIZE = 32
SH_LINK = 40
SH_ENTSIZE = 56

SYMBOLS = [
    ("alpha", 0x1000, 4, GLOBAL_FUNC, 1),
    ("beta", 0x2000, 0, GLOBAL_FUNC, 1),
]


def _write(tmp_path, data: bytes, name: str = "probe.elf") -> str:
    target = tmp_path / name
    target.write_bytes(data)
    return str(target)


def _open_index(tmp_path, data: bytes, name: str = "probe.elf"):
    """打开 ELF 并建立符号索引."""
    path = _write(tmp_path, data, name)
    elf = elf_idx.ElfFile(path)
    found = elf.find_symbol_table()
    assert found is not None
    symtab, strtab = found
    return path, elf, elf_idx.ElfSymbolIndex(elf, symtab, strtab)


def _patch_section(
    data: bytes,
    index: int,
    field_offset: int,
    value: int,
    fmt: str = "<Q",
) -> bytes:
    """改写一个节头的某个字段."""
    buf = bytearray(data)
    shoff = struct.unpack_from("<Q", buf, 40)[0]
    struct.pack_into(fmt, buf, shoff + index * SECTION_HEADER_SIZE + field_offset, value)
    return bytes(buf)


# ============================================================
#  ELF 头部与头部表
# ============================================================


class TestElfHeaderValidation:
    def test_short_file_is_rejected(self, tmp_path):
        path = _write(tmp_path, b"\x7fELF")
        with pytest.raises(ElfFormatError, match="不足一个 ELF 头"):
            elf_idx.ElfFile(path)

    def test_bad_magic_is_rejected(self, tmp_path):
        data = build_elf(header=HeaderSpec(ident=b"XXXX\x02\x01\x01" + b"\x00" * 9))
        with pytest.raises(ElfFormatError, match="不是 ELF 文件"):
            elf_idx.ElfFile(_write(tmp_path, data))

    def test_non_64bit_class_is_rejected(self, tmp_path):
        data = build_elf(header=HeaderSpec(ident=b"\x7fELF\x01\x01\x01" + b"\x00" * 9))
        with pytest.raises(ElfFormatError, match="不是 64 位 ELF"):
            elf_idx.ElfFile(_write(tmp_path, data))

    def test_big_endian_is_rejected(self, tmp_path):
        data = build_elf(header=HeaderSpec(ident=b"\x7fELF\x02\x02\x01" + b"\x00" * 9))
        with pytest.raises(ElfFormatError, match="不是小端 ELF"):
            elf_idx.ElfFile(_write(tmp_path, data))

    def test_missing_section_table_is_rejected(self, tmp_path):
        data = build_elf(
            sections=[SectionSpec(sh_type=SHT_NULL)],
            header=HeaderSpec(shnum=0),
        )
        with pytest.raises(ElfFormatError, match="没有节头表"):
            elf_idx.ElfFile(_write(tmp_path, data))

    def test_illegal_section_header_size_is_rejected(self, tmp_path):
        data = build_elf(
            sections=[SectionSpec(sh_type=SHT_NULL)],
            header=HeaderSpec(shentsize=32),
        )
        with pytest.raises(ElfFormatError, match="节头长度非法"):
            elf_idx.ElfFile(_write(tmp_path, data))

    def test_section_table_beyond_file_is_rejected(self, tmp_path):
        data = build_elf(
            sections=[SectionSpec(sh_type=SHT_NULL)],
            header=HeaderSpec(shoff=1 << 40),
        )
        with pytest.raises(ElfFormatError, match="节头表越出文件"):
            elf_idx.ElfFile(_write(tmp_path, data))

    def test_illegal_program_header_size_is_rejected(self, tmp_path):
        data = build_elf(
            sections=[SectionSpec(sh_type=SHT_NULL)],
            header=HeaderSpec(phentsize=32),
            layout=LayoutSpec(programs=[ProgramSpec()]),
        )
        with pytest.raises(ElfFormatError, match="程序头长度非法"):
            elf_idx.ElfFile(_write(tmp_path, data))

    def test_program_table_beyond_file_is_rejected(self, tmp_path):
        data = build_elf(
            sections=[SectionSpec(sh_type=SHT_NULL)],
            header=HeaderSpec(phoff=1 << 40),
            layout=LayoutSpec(programs=[ProgramSpec()]),
        )
        with pytest.raises(ElfFormatError, match="程序头表越出文件"):
            elf_idx.ElfFile(_write(tmp_path, data))


class TestHeaderTables:
    def test_sections_and_programs_are_parsed(self, tmp_path):
        program = ProgramSpec(
            p_type=PT_LOAD,
            offset=0x100,
            vaddr=0x1000,
            filesz=0x20,
            memsz=0x40,
        )
        data = build_symbol_elf(symbols=SYMBOLS, layout=LayoutSpec(programs=[program]))
        elf = elf_idx.ElfFile(_write(tmp_path, data))
        try:
            assert elf.entry == 0x1000
            assert len(elf.sections) == 4
            code = elf.sections[1]
            assert code.sh_type == SHT_PROGBITS
            assert code.flags == SHF_ALLOC | SHF_EXECINSTR
            assert code.addr == 0x1000
            assert elf.sections[2].entsize == 24
            assert elf.sections[2].link == 3
            assert elf.sections[3].sh_type == SHT_STRTAB
            assert len(elf.programs) == 1
            assert elf.programs[0].offset == 0x100
            assert elf.programs[0].vaddr == 0x1000
            assert elf.programs[0].filesz == 0x20
            assert elf.programs[0].memsz == 0x40
        finally:
            elf.close()

    def test_pread_with_non_positive_size_returns_empty(self, tmp_path):
        elf = elf_idx.ElfFile(_write(tmp_path, build_symbol_elf()))
        try:
            assert elf.pread(0, 0) == b""
            assert elf.pread(0, -1) == b""
        finally:
            elf.close()

    def test_close_is_idempotent(self, tmp_path):
        elf = elf_idx.ElfFile(_write(tmp_path, build_symbol_elf()))
        elf.close()
        elf.close()
        assert elf._fd == -1


class TestFindSymbolTable:
    def test_absent_tables_yield_none(self, tmp_path):
        elf = elf_idx.ElfFile(_write(tmp_path, build_symbol_elf(symbols=None)))
        try:
            assert elf.find_symbol_table() is None
        finally:
            elf.close()

    def test_symtab_is_preferred(self, tmp_path):
        elf = elf_idx.ElfFile(_write(tmp_path, build_symbol_elf(symbols=SYMBOLS)))
        try:
            symtab, strtab = elf.find_symbol_table()
            assert symtab.sh_type == SHT_SYMTAB
            assert strtab.sh_type == SHT_STRTAB
        finally:
            elf.close()

    def test_dynsym_is_used_when_symtab_is_absent(self, tmp_path):
        data = build_symbol_elf(symbols=SYMBOLS, symtab=SymTabSpec(is_dynsym=True))
        elf = elf_idx.ElfFile(_write(tmp_path, data))
        try:
            symtab, _ = elf.find_symbol_table()
            assert symtab.sh_type == SHT_DYNSYM
        finally:
            elf.close()

    def test_illegal_strtab_link_is_rejected(self, tmp_path):
        data = _patch_section(
            build_symbol_elf(symbols=SYMBOLS),
            2,
            SH_LINK,
            99,
            "<I",
        )
        elf = elf_idx.ElfFile(_write(tmp_path, data))
        try:
            with pytest.raises(ElfFormatError, match="字符串表索引非法"):
                elf.find_symbol_table()
        finally:
            elf.close()

    def test_symtab_beyond_file_is_rejected(self, tmp_path):
        data = _patch_section(build_symbol_elf(symbols=SYMBOLS), 2, SH_OFFSET, 1 << 40)
        elf = elf_idx.ElfFile(_write(tmp_path, data))
        try:
            with pytest.raises(ElfFormatError, match="符号表越出文件"):
                elf.find_symbol_table()
        finally:
            elf.close()

    def test_strtab_beyond_file_is_rejected(self, tmp_path):
        data = _patch_section(build_symbol_elf(symbols=SYMBOLS), 3, SH_OFFSET, 1 << 40)
        elf = elf_idx.ElfFile(_write(tmp_path, data))
        try:
            with pytest.raises(ElfFormatError, match="字符串表越出文件"):
                elf.find_symbol_table()
        finally:
            elf.close()


# ============================================================
#  地址索引
# ============================================================


class TestAddressIndex:
    def test_records_without_section_or_value_are_dropped(self, tmp_path):
        symbols = [
            (None, 0, 0, 0, 0),
            (None, 0x3000, 4, GLOBAL_FUNC, 1),
            ("zero", 0, 4, GLOBAL_FUNC, 1),
            ("alpha", 0x1000, 4, GLOBAL_FUNC, 1),
            ("beta", 0x2000, 0, GLOBAL_FUNC, 1),
        ]
        _, elf, index = _open_index(tmp_path, build_symbol_elf(symbols=symbols))
        try:
            assert index.sym_count == 2
            assert index.elf is elf
            assert index.value_min == 0x1000
            assert index.value_max_end == 0x2001
        finally:
            index.close()

    def test_name_offset_beyond_strtab_is_dropped(self, tmp_path):
        strtab = b"\x00alpha\x00"
        data = build_symtab_from_records(
            [(0xFFFF, GLOBAL_FUNC, 0, 1, 0x1000, 4)],
            strtab,
        )
        _, _, index = _open_index(tmp_path, data)
        try:
            assert index.sym_count == 0
            assert index.value_min is None
        finally:
            index.close()

    def test_empty_symbol_table_is_rejected(self, tmp_path):
        _, elf, index = _open_index(tmp_path, build_symbol_elf(symbols=SYMBOLS))
        index.close()
        symtab, strtab = elf.find_symbol_table()
        symtab.size = 0
        with pytest.raises(ElfFormatError, match="符号表为空"):
            elf_idx.ElfSymbolIndex(elf, symtab, strtab)
        elf.close()

    def test_illegal_entry_size_is_rejected(self, tmp_path):
        _, elf, index = _open_index(tmp_path, build_symbol_elf(symbols=SYMBOLS))
        index.close()
        symtab, strtab = elf.find_symbol_table()
        symtab.entsize = 16
        with pytest.raises(ElfFormatError, match="条目长度非法"):
            elf_idx.ElfSymbolIndex(elf, symtab, strtab)
        elf.close()

    def test_symbol_table_is_scanned_in_chunks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(elf_idx, "SCAN_CHUNK_BYTES", 48)
        symbols = [(f"fn{i}", 0x1000 + 4 * i, 4, GLOBAL_FUNC, 1) for i in range(5)]
        _, _, index = _open_index(tmp_path, build_symbol_elf(symbols=symbols))
        try:
            assert index.sym_count == 5
            assert index.resolve_addr(0x1008) == "fn2"
        finally:
            index.close()

    def test_truncated_symbol_table_is_rejected(self, tmp_path):
        path, elf, _ = _open_index(tmp_path, build_symbol_elf(symbols=SYMBOLS))
        symtab, strtab = elf.find_symbol_table()
        os.truncate(path, ELF_HEADER_SIZE)
        with pytest.raises(ElfFormatError, match="符号表越出文件"):
            elf_idx.ElfSymbolIndex(elf, symtab, strtab)
        elf.close()

    def test_local_non_functions_are_skipped_for_large_tables(self, tmp_path, monkeypatch):
        monkeypatch.setattr(elf_idx, "LOCAL_SYMBOL_SKIP_THRESHOLD", 1)
        symbols = [
            ("loc_data", 0x1000, 4, LOCAL_NOTYPE, 1),
            ("loc_func", 0x2000, 4, LOCAL_FUNC, 1),
            ("glob", 0x3000, 4, GLOBAL_FUNC, 1),
            ("data", 0x4000, 4, GLOBAL_NOTYPE, 1),
        ]
        _, _, index = _open_index(tmp_path, build_symbol_elf(symbols=symbols))
        try:
            assert index.sym_count == 3
            assert index.lookup_name("loc_data") is None
            assert index.lookup_name("loc_func") == 0x2000
        finally:
            index.close()


# ============================================================
#  地址到名称
# ============================================================


class TestResolveAddr:
    @staticmethod
    def _index(tmp_path):
        symbols = [
            ("start", 0x1000, 0x10, GLOBAL_FUNC, 1),
            ("middle", 0x2000, 0, GLOBAL_FUNC, 1),
            ("$local", 0x3000, 0, GLOBAL_FUNC, 1),
        ]
        return _open_index(tmp_path, build_symbol_elf(symbols=symbols))

    @pytest.mark.parametrize(
        ("addr", "expected"),
        [
            (0x1000, "start"),
            (0x1008, "start"),
            (0x2000, "middle"),
            (0x2004, "middle"),
            (0x3000, "$local"),
            (0x3004, "middle"),
        ],
    )
    def test_symbol_is_resolved_by_range_and_predecessor(self, tmp_path, addr, expected):
        _, _, index = self._index(tmp_path)
        try:
            assert index.resolve_addr(addr) == expected
        finally:
            index.close()

    def test_address_below_all_symbols_yields_none(self, tmp_path):
        _, _, index = self._index(tmp_path)
        try:
            assert index.resolve_addr(0x0800) is None
        finally:
            index.close()

    def test_predecessor_beyond_limit_yields_none(self, tmp_path):
        _, _, index = self._index(tmp_path)
        try:
            assert index.resolve_addr(0x2000 + 0x10001) is None
        finally:
            index.close()

    def test_record_without_name_is_skipped(self, tmp_path):
        strtab = b"\x00\x00alpha\x00"
        records = [
            (2, GLOBAL_FUNC, 0, 1, 0x1000, 4),
            (1, GLOBAL_FUNC, 0, 1, 0x1004, 0),
        ]
        _, _, index = _open_index(
            tmp_path,
            build_symtab_from_records(records, strtab),
        )
        try:
            assert index.resolve_addr(0x1004) == "alpha"
        finally:
            index.close()

    def test_missing_record_continues_to_previous_symbol(self, tmp_path):
        path, elf, index = self._index(tmp_path)
        try:
            os.truncate(path, index._symtab_offset)
            assert index.resolve_addr(0x1008) is None
        finally:
            index.close()

    def test_name_cache_is_cleared_when_full(self, tmp_path, monkeypatch):
        _, _, index = self._index(tmp_path)
        try:
            monkeypatch.setattr(elf_idx, "NAME_CACHE_SIZE", 1)
            assert index.resolve_addr(0x1000) == "start"
            assert index.resolve_addr(0x2000) == "middle"
            assert index.resolve_addr(0x1000) == "start"
        finally:
            index.close()

    def test_name_at_rejects_empty_and_out_of_range_offsets(self, tmp_path):
        _, _, index = self._index(tmp_path)
        try:
            assert index._name_at(0) is None
            assert index._name_at(index._strtab_size) is None
        finally:
            index.close()


# ============================================================
#  名称到地址
# ============================================================


class TestLookupName:
    def test_index_is_built_on_first_lookup(self, tmp_path):
        _, _, index = _open_index(tmp_path, build_symbol_elf(symbols=SYMBOLS))
        try:
            assert index._name_index is None
            assert index.lookup_name("alpha") == 0x1000
            assert index._name_index is not None
        finally:
            index.close()

    def test_later_duplicate_wins(self, tmp_path):
        symbols = [
            ("dup", 0x1000, 4, GLOBAL_FUNC, 1),
            ("dup", 0x2000, 4, GLOBAL_FUNC, 1),
        ]
        _, _, index = _open_index(tmp_path, build_symbol_elf(symbols=symbols))
        try:
            assert index.lookup_name("dup") == 0x2000
        finally:
            index.close()

    def test_missing_name_yields_none(self, tmp_path):
        _, _, index = _open_index(tmp_path, build_symbol_elf(symbols=SYMBOLS))
        try:
            assert index.lookup_name("gamma") is None
        finally:
            index.close()

    def test_name_crossing_strtab_chunk_uses_pread(self, tmp_path, monkeypatch):
        monkeypatch.setattr(elf_idx, "SCAN_CHUNK_BYTES", 8)
        strtab = b"\x00alpha\x00beta\x00"
        records = [
            (1, GLOBAL_FUNC, 0, 1, 0x1000, 4),
            (7, GLOBAL_FUNC, 0, 1, 0x2000, 4),
        ]
        _, _, index = _open_index(
            tmp_path,
            build_symtab_from_records(records, strtab),
        )
        try:
            assert index.lookup_name("alpha") == 0x1000
            assert index.lookup_name("beta") == 0x2000
        finally:
            index.close()


def test_name_hash_is_stable():
    assert elf_idx._name_hash(b"abc") == elf_idx._name_hash(b"abc")
    assert elf_idx._name_hash(b"abc") != elf_idx._name_hash(b"abd")


def test_pack_symbols_reuses_identical_names():
    """同一名称只写入字符串表一次, 读取端按偏移共享."""
    symtab, strtab = pack_symbols(
        [
            ("same", 0x1000, 4, GLOBAL_FUNC, 1),
            ("same", 0x2000, 4, GLOBAL_FUNC, 1),
        ]
    )
    assert strtab.count(b"same") == 1
    assert len(symtab) == 2 * 24
