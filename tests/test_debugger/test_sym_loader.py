#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""测试 pyremu.debug.sym.loader — 部署地址解析、区间校验与内容校验."""

from __future__ import annotations

from dataclasses import dataclass
import io
import struct
from types import SimpleNamespace

from rich.console import Console

from pyremu.core.hart import RiscvMode
from pyremu.debug import Debugger
from pyremu.debug.sym import loader, model
from pyremu.emulator import Emulator
from tests.test_debugger.sym_fixtures import (
    build_symbol_elf,
    HeaderSpec,
    LayoutSpec,
    ProgramSpec,
    PT_LOAD,
    SECTION_HEADER_SIZE,
    SHF_ALLOC,
    TextSpec,
)

RAM_BASE = 0x8000_0000
RAM_SIZE = 0x0001_0000
GLOBAL_FUNC = 0x12
SYMBOLS = [
    ("alpha", 0x1000, 4, GLOBAL_FUNC, 1),
    ("beta", 0x2000, 0, GLOBAL_FUNC, 1),
]
SAMPLE_LIMIT = 256


def _make_dbg(ram_size: int = RAM_SIZE):
    emu = Emulator(num_harts=1, ram_size=ram_size, ram_base=RAM_BASE, prog_cnt=0x1000)
    emu._termio = None
    dbg = Debugger(emulator=emu, hart_id=0, image=None)
    buf = io.StringIO()
    dbg._console = Console(file=buf, highlight=False, width=200)
    return dbg, emu, buf


def _write(tmp_path, data: bytes, name: str = "probe.elf") -> str:
    target = tmp_path / name
    target.write_bytes(data)
    return str(target)


def _section_offset(data: bytes, index: int = 1) -> int:
    shoff = struct.unpack_from("<Q", data, 40)[0]
    return struct.unpack_from("<Q", data, shoff + index * SECTION_HEADER_SIZE + 24)[0]


def _patch_section_offset(data: bytes, index: int, offset: int) -> bytes:
    buf = bytearray(data)
    shoff = struct.unpack_from("<Q", buf, 40)[0]
    struct.pack_into("<Q", buf, shoff + index * SECTION_HEADER_SIZE + 24, offset)
    return bytes(buf)


def _patch_section_entsize(data: bytes, index: int, entsize: int) -> bytes:
    buf = bytearray(data)
    shoff = struct.unpack_from("<Q", buf, 40)[0]
    struct.pack_into("<Q", buf, shoff + index * SECTION_HEADER_SIZE + 56, entsize)
    return bytes(buf)


def _stage(emu, data: bytes, runtime: int, file_offset: int) -> None:
    """把文件中校验样本所在的一段写入 RAM."""
    emu.load_code(runtime, data[file_offset : file_offset + SAMPLE_LIMIT])


@dataclass
class DeployTarget:
    """符号模块的地址空间与运行时基址."""

    space: str | None = None
    base: int | None = None
    runtime: int = RAM_BASE


def _deploy(dbg, emu, path: str, data: bytes, target: DeployTarget | None = None):
    target = DeployTarget() if target is None else target
    _stage(emu, data, target.runtime, _section_offset(data))
    return loader.load_symbol_module(dbg, path, target.space, target.base)


class _StubBus:
    def __init__(self, data: bytes, is_ram: bool = True) -> None:
        self._data = data
        self._is_ram = is_ram

    def is_ram_addr(self, addr: int) -> bool:
        return self._is_ram

    def try_read(self, addr: int, size: int) -> bytes:
        return self._data[:size]


class _StubDbg:
    def __init__(self, bus: _StubBus) -> None:
        self._emu = SimpleNamespace(bus=bus)


# ============================================================
#  物理与虚拟基址
# ============================================================


class TestBaseResolution:
    def test_physical_base_is_deployed_and_verified(self, tmp_path):
        dbg, emu, _ = _make_dbg()
        data = build_symbol_elf(symbols=SYMBOLS)
        path = _write(tmp_path, data)
        mod = _deploy(dbg, emu, path, data, DeployTarget("pa", RAM_BASE))
        assert mod is not None
        assert mod.pa_base == RAM_BASE
        assert mod.va_base is None
        assert mod.cover_start == 0x1000
        assert mod.cover_end == 0x1010
        assert mod.sym_count == 2
        assert mod.is_verified
        remaining = len(data) - _section_offset(data)
        assert mod.sample_len == min(SAMPLE_LIMIT, remaining)

    def test_omitted_address_uses_link_address(self, tmp_path):
        dbg, emu, _ = _make_dbg()
        data = build_symbol_elf(
            symbols=SYMBOLS,
            header=HeaderSpec(entry=RAM_BASE),
            text=TextSpec(addr=RAM_BASE),
        )
        path = _write(tmp_path, data)
        mod = _deploy(dbg, emu, path, data)
        assert mod is not None
        assert mod.va_base == RAM_BASE
        assert mod.pa_base == RAM_BASE

    def test_virtual_base_is_translated(self, tmp_path):
        dbg, emu, _ = _make_dbg()
        data = build_symbol_elf(symbols=SYMBOLS)
        path = _write(tmp_path, data)
        mod = _deploy(dbg, emu, path, data, DeployTarget("va", RAM_BASE))
        assert mod is not None
        assert mod.va_base == RAM_BASE
        assert mod.pa_base == RAM_BASE

    def test_virtual_base_without_mapping_is_rejected(self, tmp_path):
        dbg, _, buf = _make_dbg()
        hart = dbg.hart
        hart.mode = RiscvMode.S
        hart.satp_val = (8 << 60) | (RAM_BASE >> 12)
        data = build_symbol_elf(symbols=SYMBOLS)
        path = _write(tmp_path, data)
        mod = loader.load_symbol_module(dbg, path, "va", 0x1000)
        assert mod is None
        assert "无法翻译" in buf.getvalue()

    def test_window_upper_violation_is_rejected(self, tmp_path):
        dbg, emu, buf = _make_dbg()
        data = build_symbol_elf(symbols=SYMBOLS)
        path = _write(tmp_path, data)
        mod = loader.load_symbol_module(dbg, path, "pa", RAM_BASE + RAM_SIZE)
        assert mod is None
        assert "超出 RAM 窗口" in buf.getvalue()

    def test_window_lower_violation_is_rejected(self, tmp_path):
        dbg, emu, buf = _make_dbg()
        data = build_symbol_elf(symbols=SYMBOLS)
        path = _write(tmp_path, data)
        mod = loader.load_symbol_module(dbg, path, "pa", RAM_BASE - 1)
        assert mod is None
        assert "超出 RAM 窗口" in buf.getvalue()

    def test_interval_equal_to_window_end_is_accepted(self, tmp_path):
        dbg, emu, _ = _make_dbg()
        text = b"\x13\x00\x00\x00" * 0x100
        data = build_symbol_elf(symbols=SYMBOLS, text=TextSpec(data=text))
        path = _write(tmp_path, data)
        base = RAM_BASE + RAM_SIZE - 0x400
        mod = _deploy(dbg, emu, path, data, DeployTarget("pa", base, base))
        assert mod is not None
        assert mod.pa_base == base

    def test_first_mismatching_byte_is_reported(self, tmp_path):
        dbg, emu, buf = _make_dbg()
        data = build_symbol_elf(symbols=SYMBOLS)
        path = _write(tmp_path, data)
        _stage(emu, data, RAM_BASE, _section_offset(data))
        assert emu.bus.try_write(RAM_BASE + 3, b"\xff")
        mod = loader.load_symbol_module(dbg, path, "pa", RAM_BASE)
        assert mod is None
        text = buf.getvalue()
        assert "首个不一致字节的偏移为 3" in text
        assert "内存中的字节为 0xff" in text


# ============================================================
#  加载失败路径
# ============================================================


class TestLoadFailures:
    def test_missing_file_is_reported(self, tmp_path):
        dbg, _, buf = _make_dbg()
        mod = loader.load_symbol_module(dbg, str(tmp_path / "absent.elf"), "pa", RAM_BASE)
        assert mod is None
        assert "无法读取符号文件" in buf.getvalue()

    def test_non_elf_file_is_rejected(self, tmp_path):
        dbg, _, buf = _make_dbg()
        path = _write(tmp_path, b"not an elf file at all")
        mod = loader.load_symbol_module(dbg, path, "pa", RAM_BASE)
        assert mod is None
        assert "无法加载符号文件" in buf.getvalue()

    def test_missing_symbol_table_is_rejected(self, tmp_path):
        dbg, _, buf = _make_dbg()
        path = _write(tmp_path, build_symbol_elf(symbols=None))
        mod = loader.load_symbol_module(dbg, path, "pa", RAM_BASE)
        assert mod is None
        assert "未找到 .symtab 或 .dynsym" in buf.getvalue()

    def test_illegal_symbol_entry_size_is_rejected(self, tmp_path):
        dbg, _, buf = _make_dbg()
        data = _patch_section_entsize(build_symbol_elf(symbols=SYMBOLS), 2, 16)
        path = _write(tmp_path, data)
        mod = loader.load_symbol_module(dbg, path, "pa", RAM_BASE)
        assert mod is None
        assert "条目长度非法" in buf.getvalue()

    def test_entry_limit_exceeded_is_rejected(self, tmp_path, monkeypatch):
        dbg, _, buf = _make_dbg()
        monkeypatch.setattr(loader.elf_idx, "MAX_SYMBOL_ENTRIES", 1)
        path = _write(tmp_path, build_symbol_elf(symbols=SYMBOLS))
        mod = loader.load_symbol_module(dbg, path, "pa", RAM_BASE)
        assert mod is None
        assert "超过上限" in buf.getvalue()

    def test_warn_threshold_reports_index_size(self, tmp_path, monkeypatch):
        dbg, emu, buf = _make_dbg()
        monkeypatch.setattr(loader.elf_idx, "WARN_SYMBOL_ENTRIES", 1)
        data = build_symbol_elf(symbols=SYMBOLS)
        path = _write(tmp_path, data)
        mod = _deploy(dbg, emu, path, data, DeployTarget("pa", RAM_BASE))
        assert mod is not None
        assert "地址索引约" in buf.getvalue()


# ============================================================
#  覆盖范围与校验样本定位
# ============================================================


class TestCoverageAndAnchor:
    def test_symbol_span_is_used_without_alloc_sections(self, tmp_path):
        dbg, emu, _ = _make_dbg()
        data = build_symbol_elf(symbols=SYMBOLS, text=TextSpec(flags=0))
        path = _write(tmp_path, data)
        mod = _deploy(dbg, emu, path, data, DeployTarget("pa", RAM_BASE))
        assert mod is not None
        assert (mod.cover_start, mod.cover_end) == (0x1000, 0x2001)

    def test_missing_address_range_is_rejected(self, tmp_path):
        dbg, _, buf = _make_dbg()
        symbols = [(None, 0, 0, 0, 0)]
        path = _write(tmp_path, build_symbol_elf(symbols=symbols, text=TextSpec(flags=0)))
        mod = loader.load_symbol_module(dbg, path, "pa", RAM_BASE)
        assert mod is None
        assert "没有可用的地址范围" in buf.getvalue()

    def test_anchor_falls_back_to_lowest_exec_section(self, tmp_path):
        dbg, emu, _ = _make_dbg()
        data = build_symbol_elf(symbols=SYMBOLS, header=HeaderSpec(entry=0))
        path = _write(tmp_path, data)
        mod = _deploy(dbg, emu, path, data, DeployTarget("pa", RAM_BASE))
        assert mod is not None
        assert mod.anchor_link == 0x1000

    def test_missing_exec_section_leaves_no_anchor(self, tmp_path):
        dbg, _, buf = _make_dbg()
        data = build_symbol_elf(
            symbols=SYMBOLS,
            header=HeaderSpec(entry=0),
            text=TextSpec(flags=SHF_ALLOC),
        )
        path = _write(tmp_path, data)
        mod = loader.load_symbol_module(dbg, path, "pa", RAM_BASE)
        assert mod is None
        assert "无法在文件中定位校验样本的起点" in buf.getvalue()

    def test_anchor_outside_sections_uses_program_header(self, tmp_path):
        dbg, emu, _ = _make_dbg()
        program = ProgramSpec(
            p_type=PT_LOAD,
            offset=0x40,
            vaddr=0x1000,
            filesz=0x2000,
            memsz=0x2000,
        )
        data = build_symbol_elf(
            symbols=SYMBOLS,
            header=HeaderSpec(entry=0x1800),
            text=TextSpec(flags=0),
            layout=LayoutSpec(programs=[program], extra_size=0x2000),
        )
        path = _write(tmp_path, data)
        file_offset = program.offset + (0x1800 - 0x1000)
        runtime = RAM_BASE + 0x800
        _stage(emu, data, runtime, file_offset)
        mod = loader.load_symbol_module(dbg, path, "pa", RAM_BASE)
        assert mod is not None
        assert mod.anchor_link == 0x1800

    def test_sample_shorter_than_minimum_is_rejected(self, tmp_path):
        dbg, _, buf = _make_dbg()
        raw = build_symbol_elf(symbols=SYMBOLS)
        data = _patch_section_offset(raw, 1, len(raw) - 2)
        path = _write(tmp_path, data)
        mod = loader.load_symbol_module(dbg, path, "pa", RAM_BASE)
        assert mod is None
        assert "校验样本不足 4 字节" in buf.getvalue()


# ============================================================
#  重新校验与文本
# ============================================================


class TestVerifyAndText:
    @staticmethod
    def _loaded(tmp_path):
        dbg, emu, _ = _make_dbg()
        data = build_symbol_elf(symbols=SYMBOLS)
        path = _write(tmp_path, data)
        mod = _deploy(dbg, emu, path, data, DeployTarget("pa", RAM_BASE))
        assert mod is not None
        return dbg, emu, mod

    def test_verify_succeeds_while_memory_matches(self, tmp_path):
        dbg, _, mod = self._loaded(tmp_path)
        assert loader.verify_module(dbg, mod)

    def test_verify_fails_after_memory_changes(self, tmp_path):
        dbg, emu, mod = self._loaded(tmp_path)
        assert emu.bus.try_write(RAM_BASE + 1, b"\xff")
        assert not loader.verify_module(dbg, mod)

    def test_verify_skips_module_without_index(self):
        dbg, _, _ = _make_dbg()
        mod = model.SymMod(
            file_name="x.elf",
            path="/tmp/x.elf",
            pa_base=RAM_BASE,
            va_base=None,
            cover_start=0x1000,
            cover_end=0x2000,
            sym_count=0,
            is_verified=False,
            load_order=0,
            verify_epoch=0,
            index=None,
        )
        assert loader.verify_module(dbg, mod)

    def test_verify_skips_module_without_physical_base(self, tmp_path):
        dbg, _, _ = _make_dbg()
        data = build_symbol_elf(symbols=SYMBOLS)
        path = _write(tmp_path, data)
        elf = loader._open_elf(path)
        index = loader._build_index(dbg, elf, path)
        mod = model.SymMod(
            file_name="x.elf",
            path=path,
            pa_base=None,
            va_base=0x1000,
            cover_start=0x1000,
            cover_end=0x2000,
            sym_count=0,
            is_verified=False,
            load_order=0,
            verify_epoch=0,
            index=index,
        )
        try:
            assert loader.verify_module(dbg, mod)
        finally:
            index.close()

    def test_base_text_lists_known_spaces(self):
        mod = model.SymMod(
            file_name="x.elf",
            path="/tmp/x.elf",
            pa_base=RAM_BASE,
            va_base=0x1000,
            cover_start=0x1000,
            cover_end=0x2000,
            sym_count=0,
            is_verified=True,
            load_order=0,
            verify_epoch=0,
        )
        assert loader._base_text(mod) == "PA 0x80000000, VA 0x1000"
        mod.va_base = None
        assert loader._base_text(mod) == "PA 0x80000000"

    def test_sample_outside_ram_is_rejected(self):
        dbg = _StubDbg(_StubBus(b"", is_ram=False))
        sample = loader._Sample(0x1000, 0x2000, 0x1000, 0, b"abcd")
        assert loader._compare_sample(dbg, sample) == (False, 0)

    def test_mismatch_text_reports_memory_byte(self):
        dbg = _StubDbg(_StubBus(b"XYZ"))
        sample = loader._Sample(0x1000, 0x2000, 0x1000, 0x40, b"ABCD", RAM_BASE)
        text = loader._mismatch_text(dbg, "/tmp/x.elf", sample, 0)
        assert (
            "首个不一致字节的偏移为 0" in text
            and "文件中的字节为 0x41" in text
            and "内存中的字节为 0x58" in text
        )

    def test_mismatch_text_handles_past_end_offset(self):
        dbg = _StubDbg(_StubBus(b"ABCDE"))
        sample = loader._Sample(0x1000, 0x2000, 0x1000, 0x40, b"ABCD", RAM_BASE)
        text = loader._mismatch_text(dbg, "/tmp/x.elf", sample, 4)
        assert "文件中的字节" not in text and "内存中的字节为 0x41" in text


def test_sample_length_is_capped(monkeypatch, tmp_path):
    """样本长度取剩余长度与上限中的较小者."""
    dbg, emu, _ = _make_dbg()
    data = build_symbol_elf(symbols=SYMBOLS)
    path = _write(tmp_path, data)
    monkeypatch.setattr(loader, "SAMPLE_MAX_BYTES", 8)
    mod = _deploy(dbg, emu, path, data, DeployTarget("pa", RAM_BASE))
    assert mod is not None and mod.sample_len == 8
