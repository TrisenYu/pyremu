#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""测试 pyremu.debug.sym — 符号解析、内核符号加载、段查找与代码地址校验."""

import io
from pathlib import Path

import pytest
from rich.console import Console

from pyremu import configs_gen
from pyremu.core.hart import RiscvMode
from pyremu.debug import Debugger, sym
from pyremu.debug.sym.model import SymAvailItem
from pyremu.emulator import Emulator
from pyremu.utils.parse_bin import FirmwareImage, FirmwareSegment

RAM_BASE = 0x8000_0000
RAM_SIZE = 0x0004_0000
HOLE_ADDR = 0x9000_0000
SATP_SV39 = 8 << 60
KERNEL_VA = 0xFFFF_FFC0_0000_1000
USER_VA = 0x0000_0000_8000_1000
NON_CANONICAL_VA = 0x0000_0100_0000_0000


def _make_dbg(image: FirmwareImage | None = None, num_harts: int = 1):
    """建立最小 Emulator 与 Debugger, 输出重定向到内存缓冲区."""
    emu = Emulator(num_harts=num_harts, ram_size=RAM_SIZE, ram_base=RAM_BASE, prog_cnt=0x1000)
    emu._termio = None
    emu.load_code(0x1000, b"\x13\x00\x00\x00")
    dbg = Debugger(emulator=emu, hart_id=0, image=image)
    buf = io.StringIO()
    dbg._console = Console(file=buf, highlight=False, width=200)
    return dbg, emu, buf


def _fw_image(
    entry_point: int = 0x1000,
    symbols: dict | None = None,
    symbol_ranges: list | None = None,
    fmt: str = "elf",
    segments: list | None = None,
) -> FirmwareImage:
    return FirmwareImage(
        entry_point=entry_point,
        segments=list(segments or []),
        format=fmt,
        symbols=dict(symbols or {}),
        symbol_ranges=list(symbol_ranges or []),
    )


def _many_symbols(prefix: str, count: int) -> dict:
    """构造以 *prefix* 开头的 *count* 个符号, 地址自 0x1000 起递增."""
    return {f"{prefix}{i}": 0x1000 + 4 * i for i in range(count)}


# 加载路径先核对文件是否存在, 再调用 parse_firmware; 后者由替身给出,
# 故此处只需要一个确实存在的路径.
EXISTING_PATH = sym.__file__


# ============================================================
#  按名称读取寄存器
# ============================================================


class TestReadByName:
    def test_gpr_is_read_by_number_name(self):
        dbg, emu, _ = _make_dbg()
        emu.harts[0].write_gpr(12, 0xDEAD)
        assert dbg._read_gpr_by_name("x12") == 0xDEAD

    def test_gpr_is_read_by_alias(self):
        dbg, emu, _ = _make_dbg()
        emu.harts[0].write_gpr(10, 0x1234)
        assert dbg._read_gpr_by_name("a0") == 0x1234

    def test_unknown_gpr_name_yields_zero(self):
        dbg, emu, _ = _make_dbg()
        emu.harts[0].write_gpr(0, 0xFFFF)
        assert dbg._read_gpr_by_name("x99") == 0

    def test_csr_is_read_by_name(self):
        dbg, emu, _ = _make_dbg()
        emu.harts[0].write_csr(0x305, 0x8000_1234)
        assert dbg._read_csr_by_name("mtvec") == 0x8000_1234

    def test_unknown_csr_name_yields_zero(self):
        dbg, _, _ = _make_dbg()
        assert dbg._read_csr_by_name("nosuchcsr") == 0


# ============================================================
#  _resolve_symbol
# ============================================================


class TestResolveSymbol:
    def test_range_containing_address_wins(self):
        ranges = [(0x1000, 0x1100, "probe"), (0x2000, 0x2100, "echo")]
        assert sym.SymLookupOps._resolve_symbol({}, 0x1050, ranges) == "probe"

    def test_second_range_is_selected(self):
        ranges = [(0x1000, 0x1100, "probe"), (0x2000, 0x2100, "echo")]
        assert sym.SymLookupOps._resolve_symbol({}, 0x2000, ranges) == "echo"

    def test_range_end_is_exclusive(self):
        """区间右端不包含: 命中落在 end 的地址须回退到后续判定."""
        ranges = [(0x1000, 0x1100, "probe")]
        assert sym.SymLookupOps._resolve_symbol({}, 0x1100, ranges) is None

    def test_exact_match_is_used_without_ranges(self):
        assert sym.SymLookupOps._resolve_symbol({"main": 0x8000}, 0x8000) == "main"

    def test_exact_match_runs_before_predecessor_fallback(self):
        """无区间信息时精确匹配优先于最近前驱."""
        symbols = {"_start": 0x7000, "main": 0x8000}
        assert sym.SymLookupOps._resolve_symbol(symbols, 0x8000) == "main"

    def test_gap_inside_range_span_yields_none(self):
        """地址落在区间跨度之内但不在任何区间内时放弃匹配."""
        ranges = [(0x1000, 0x1100, "probe"), (0x2000, 0x2100, "echo")]
        assert sym.SymLookupOps._resolve_symbol({"mid": 0x1500}, 0x1800, ranges) is None

    def test_address_before_range_span_falls_back_to_predecessor(self):
        ranges = [(0x1000, 0x1100, "probe")]
        assert sym.SymLookupOps._resolve_symbol({"head": 0x0F00}, 0x0F00, ranges) == "head"

    def test_address_after_range_span_falls_back_to_predecessor(self):
        ranges = [(0x1000, 0x1100, "probe")]
        assert sym.SymLookupOps._resolve_symbol({"tail": 0x3000}, 0x3000, ranges) == "tail"

    @pytest.mark.parametrize("name", ["$x", ".L0"])
    def test_assembler_local_names_are_skipped(self, name):
        symbols = {name: 0x1000}
        assert sym.SymLookupOps._resolve_symbol(symbols, 0x2000) is None

    def test_local_name_does_not_shadow_visible_predecessor(self):
        symbols = {".L0": 0x1800, "func": 0x1000}
        assert sym.SymLookupOps._resolve_symbol(symbols, 0x2000) == "func"

    def test_predecessor_beyond_limit_yields_none(self):
        """最近前驱相距超过 64 KiB 时不视为匹配."""
        assert sym.SymLookupOps._resolve_symbol({"far": 0x1000}, 0x1000 + 0x10001) is None

    def test_predecessor_at_limit_is_accepted(self):
        assert sym.SymLookupOps._resolve_symbol({"far": 0x1000}, 0x1000 + 0x10000) == "far"

    def test_empty_symbols_yield_none(self):
        assert sym.SymLookupOps._resolve_symbol({}, 0x1000) is None

    def test_successor_only_yields_none(self):
        assert sym.SymLookupOps._resolve_symbol({"later": 0x2000}, 0x1000) is None


# ============================================================
#  load_kernel_symbols
# ============================================================


class TestLoadKernelSymbols:
    def test_missing_file_reports_error(self, tmp_path):
        dbg, _, buf = _make_dbg()
        missing = tmp_path / "vmlinux"
        assert dbg.load_kernel_symbols(str(missing)) is False
        assert "符号文件不存在" in buf.getvalue()

    def test_unparsable_file_reports_error(self, tmp_path, monkeypatch):
        src = tmp_path / "vmlinux"
        src.write_bytes(b"\x00")
        monkeypatch.setattr(sym.lookup, "parse_firmware", lambda *a, **k: None)
        dbg, _, buf = _make_dbg()
        assert dbg.load_kernel_symbols(str(src)) is False
        assert "无法解析符号文件" in buf.getvalue()

    def test_non_elf_file_reports_error(self, tmp_path, monkeypatch):
        src = tmp_path / "kernel.bin"
        src.write_bytes(b"\x00")
        monkeypatch.setattr(sym.lookup, "parse_firmware", lambda *a, **k: _fw_image(fmt="raw"))
        dbg, _, buf = _make_dbg()
        assert dbg.load_kernel_symbols(str(src)) is False
        assert "需为 ELF" in buf.getvalue()

    def test_missing_symbol_table_reports_warning(self, tmp_path, monkeypatch):
        src = tmp_path / "vmlinux"
        src.write_bytes(b"\x00")
        monkeypatch.setattr(sym.lookup, "parse_firmware", lambda *a, **k: _fw_image())
        dbg, _, buf = _make_dbg()
        assert dbg.load_kernel_symbols(str(src)) is False
        assert "未找到符号表" in buf.getvalue()

    def test_default_base_address_is_reserved_mem_top(self, monkeypatch):
        image = _fw_image(
            entry_point=0x1000,
            symbols={"main": 0x1000},
            symbol_ranges=[(0x1000, 0x1100, "main")],
        )
        monkeypatch.setattr(sym.lookup, "parse_firmware", lambda *a, **k: image)
        dbg, _, _ = _make_dbg()
        assert dbg.load_kernel_symbols(EXISTING_PATH) is True
        expected_base = RAM_BASE + configs_gen.FW_RESERVED_MEM_SIZE
        assert dbg._sym_load_offset == (expected_base - 0x1000) & ((1 << 64) - 1)
        assert dbg._sym_symbols_pa["main"] == (0x1000 + dbg._sym_load_offset) & ((1 << 64) - 1)

    def test_explicit_base_address_is_used(self, monkeypatch):
        image = _fw_image(entry_point=0x2000, symbols={"main": 0x2000})
        monkeypatch.setattr(sym.lookup, "parse_firmware", lambda *a, **k: image)
        dbg, _, _ = _make_dbg()
        assert dbg.load_kernel_symbols(EXISTING_PATH, base_pa=0x9000) is True
        assert dbg._sym_load_offset == 0x7000
        assert dbg._sym_symbols_pa["main"] == 0x9000

    def test_loaded_state_is_recorded(self, monkeypatch):
        image = _fw_image(
            entry_point=0x1000,
            symbols={"main": 0x1000, "echo": 0x1200},
            symbol_ranges=[(0x1000, 0x1100, "main"), (0x1200, 0x1300, "echo")],
        )
        monkeypatch.setattr(sym.lookup, "parse_firmware", lambda *a, **k: image)
        dbg, _, buf = _make_dbg()
        assert dbg.load_kernel_symbols(EXISTING_PATH) is True
        assert dbg._sym_symbols == {"main": 0x1000, "echo": 0x1200}
        assert dbg._sym_path == EXISTING_PATH
        assert dbg._sym_symbols_pa["echo"] == 0x1200 + dbg._sym_load_offset
        assert dbg._pa_to_va[dbg._sym_symbols_pa["echo"]] == 0x1200
        assert dbg._sym_ranges_pa[0] == (
            0x1000 + dbg._sym_load_offset,
            0x1100 + dbg._sym_load_offset,
            "main",
        )
        assert "已加载" in buf.getvalue()

    def test_physical_ranges_are_sorted_by_start(self, monkeypatch):
        """范围表按物理起始地址升序排列, 供二分查找使用."""
        image = _fw_image(
            entry_point=0x1000,
            symbols={"main": 0x1000},
            symbol_ranges=[(0x2000, 0x2100, "echo"), (0x1000, 0x1100, "main")],
        )
        monkeypatch.setattr(sym.lookup, "parse_firmware", lambda *a, **k: image)
        dbg, _, _ = _make_dbg()
        dbg.load_kernel_symbols(EXISTING_PATH, base_pa=0x4000)
        starts = [r[0] for r in dbg._sym_ranges_pa]
        assert starts == sorted(starts)
        assert [r[2] for r in dbg._sym_ranges_pa] == ["main", "echo"]

    def test_reload_clears_previous_tables(self, monkeypatch):
        first = _fw_image(entry_point=0x1000, symbols={"old": 0x1000})
        second = _fw_image(entry_point=0x1000, symbols={"new": 0x1000})
        dbg, _, _ = _make_dbg()
        monkeypatch.setattr(sym.lookup, "parse_firmware", lambda *a, **k: first)
        dbg.load_kernel_symbols(EXISTING_PATH)
        monkeypatch.setattr(sym.lookup, "parse_firmware", lambda *a, **k: second)
        assert dbg.load_kernel_symbols(EXISTING_PATH) is True
        assert dbg._sym_symbols == {"new": 0x1000}
        assert "old" not in dbg._sym_symbols_pa


# ============================================================
#  地址解析
# ============================================================


class TestResolveSymAddr:
    def test_without_symbols_returns_none(self):
        dbg, _, _ = _make_dbg()
        assert dbg._resolve_sym_addr(0x1000) is None

    def test_bare_mode_adds_load_offset(self):
        dbg, emu, _ = _make_dbg()
        dbg._sym_symbols = {"main": 0x1000}
        dbg._sym_load_offset = 0x2000
        assert dbg._resolve_sym_addr(0x1000) == 0x3000

    def test_bare_mode_wraps_around_64_bits(self):
        dbg, _, _ = _make_dbg()
        dbg._sym_symbols = {"main": 0x1000}
        dbg._sym_load_offset = (1 << 64) - 0x1000
        assert dbg._resolve_sym_addr(0x2000) == 0x1000

    def test_translation_result_is_returned(self, monkeypatch):
        dbg, emu, _ = _make_dbg()
        dbg._sym_symbols = {"main": 0x1000}
        dbg._sym_load_offset = 0x2000
        emu.harts[0].satp_val = SATP_SV39
        monkeypatch.setattr(sym.lookup, "translate_addr", lambda *a, **k: (True, 0x4000))
        assert dbg._resolve_sym_addr(KERNEL_VA) == 0x4000

    def test_original_address_is_kept_when_translation_fails(self, monkeypatch):
        dbg, emu, _ = _make_dbg()
        dbg._sym_symbols = {"main": 0x1000}
        dbg._sym_load_offset = 0x2000
        emu.harts[0].satp_val = SATP_SV39
        monkeypatch.setattr(sym.lookup, "translate_addr", lambda *a, **k: (False, 0))
        assert dbg._resolve_sym_addr(KERNEL_VA) == KERNEL_VA


class TestSymLookupName:
    def test_without_symbols_returns_none(self):
        dbg, _, _ = _make_dbg()
        assert dbg._sym_lookup_name(0x1000) is None

    def test_bare_mode_with_physical_table_uses_physical(self):
        dbg, _, _ = _make_dbg()
        dbg._sym_symbols = {"va_name": 0x1000}
        dbg._sym_symbols_pa = {"pa_name": 0x2000}
        dbg._sym_ranges_pa = [(0x2000, 0x2100, "pa_name")]
        assert dbg._sym_lookup_name(0x2050) == "pa_name"

    def test_bare_mode_without_physical_table_falls_back_to_va(self):
        """未建立物理表时沿用类 VA 表."""
        dbg, _, _ = _make_dbg()
        dbg._sym_symbols = {"va_name": 0x1000}
        dbg._sym_ranges = [(0x1000, 0x1100, "va_name")]
        assert dbg._sym_lookup_name(0x1050) == "va_name"

    def test_translated_mode_uses_va_table(self):
        dbg, emu, _ = _make_dbg()
        dbg._sym_symbols = {"va_name": 0x1000}
        dbg._sym_symbols_pa = {"pa_name": 0x2000}
        emu.harts[0].satp_val = SATP_SV39
        dbg._sym_ranges = [(0x1000, 0x1100, "va_name")]
        assert dbg._sym_lookup_name(0x1050) == "va_name"

    def test_address_far_from_any_symbol_returns_none(self):
        """最近前驱相距超过 64 KiB 时不给出名称."""
        dbg, _, _ = _make_dbg()
        dbg._sym_symbols = {"va_name": 0x1000}
        dbg._sym_ranges = [(0x1000, 0x1100, "va_name")]
        assert dbg._sym_lookup_name(0x1000 + 0x10001) is None


class TestResolveAnySymbolName:
    def test_without_firmware_image_uses_kernel_symbols(self):
        dbg, _, _ = _make_dbg()
        dbg._sym_symbols = {"kern": 0x1000}
        dbg._sym_ranges = [(0x1000, 0x1100, "kern")]
        assert dbg._resolve_any_symbol_name(0x1050) == "kern"

    def test_firmware_symbol_wins(self):
        image = _fw_image(
            symbols={"fw_main": 0x1000}, symbol_ranges=[(0x1000, 0x1100, "fw_main")]
        )
        dbg, _, _ = _make_dbg(image=image)
        dbg._sym_symbols = {"kern": 0x1000}
        assert dbg._resolve_any_symbol_name(0x1050) == "fw_main"

    def test_kernel_symbols_are_consulted_after_firmware_miss(self):
        image = _fw_image(
            symbols={"fw_main": 0x1000}, symbol_ranges=[(0x1000, 0x1100, "fw_main")]
        )
        dbg, _, _ = _make_dbg(image=image)
        dbg._sym_symbols = {"kern": 0x50000}
        dbg._sym_ranges = [(0x50000, 0x50100, "kern")]
        assert dbg._resolve_any_symbol_name(0x50050) == "kern"

    def test_firmware_load_offset_is_subtracted(self):
        image = _fw_image(
            symbols={"fw_main": 0x1000}, symbol_ranges=[(0x1000, 0x1100, "fw_main")]
        )
        dbg, _, _ = _make_dbg(image=image)
        dbg._load_offset = 0x8000_0000
        assert dbg._resolve_any_symbol_name(0x8000_1050) == "fw_main"

    def test_image_without_symbols_uses_kernel_symbols(self):
        dbg, _, _ = _make_dbg(image=_fw_image())
        dbg._sym_symbols = {"kern": 0x1000}
        assert dbg._resolve_any_symbol_name(0x1000) == "kern"


# ============================================================
#  段查找
# ============================================================


class TestFindSegment:
    def test_without_image_returns_none(self):
        dbg, _, _ = _make_dbg()
        assert dbg._find_segment(0x1000) is None

    def test_containing_segment_is_returned(self):
        segs = [
            FirmwareSegment(vaddr=0x1000, data=b"\x00", memsz=0x100, name=".text"),
            FirmwareSegment(vaddr=0x2000, data=b"\x00", memsz=0x100, name=".data"),
        ]
        dbg, _, _ = _make_dbg(image=_fw_image(segments=segs))
        assert dbg._find_segment(0x2050).name == ".data"

    def test_bss_tail_is_inside_segment(self):
        """memsz 大于文件内容时, 零填充部分仍属该段."""
        segs = [FirmwareSegment(vaddr=0x1000, data=b"\x00", memsz=0x1000, name=".bss")]
        dbg, _, _ = _make_dbg(image=_fw_image(segments=segs))
        assert dbg._find_segment(0x1F00).name == ".bss"

    def test_address_at_segment_end_is_outside(self):
        segs = [FirmwareSegment(vaddr=0x1000, data=b"\x00", memsz=0x100, name=".text")]
        dbg, _, _ = _make_dbg(image=_fw_image(segments=segs))
        assert dbg._find_segment(0x1100) is None

    def test_address_in_no_segment_returns_none(self):
        segs = [FirmwareSegment(vaddr=0x1000, data=b"\x00", memsz=0x100, name=".text")]
        dbg, _, _ = _make_dbg(image=_fw_image(segments=segs))
        assert dbg._find_segment(0x9000) is None


# ============================================================
#  代码地址校验
# ============================================================


class TestIsValidCodeVa:
    def test_bare_mode_accepts_ram_address(self):
        dbg, _, _ = _make_dbg()
        assert dbg._is_valid_code_va(RAM_BASE) is True

    def test_bare_mode_rejects_hole_address(self):
        dbg, _, _ = _make_dbg()
        assert dbg._is_valid_code_va(HOLE_ADDR) is False

    def test_machine_mode_falls_back_to_physical_check(self):
        """M 模式恒用 Bare 语义, 即使 satp 已设定翻译模式."""
        dbg, emu, _ = _make_dbg()
        hart = emu.harts[0]
        hart.satp_val = SATP_SV39
        hart.mode = RiscvMode.M
        assert dbg._is_valid_code_va(RAM_BASE) is True
        assert dbg._is_valid_code_va(KERNEL_VA) is False

    @pytest.mark.parametrize("mode", [RiscvMode.S, RiscvMode.D])
    def test_supervisor_mode_accepts_kernel_va(self, mode):
        dbg, emu, _ = _make_dbg()
        hart = emu.harts[0]
        hart.satp_val = SATP_SV39
        hart.mode = mode
        assert dbg._is_valid_code_va(KERNEL_VA) is True

    @pytest.mark.parametrize("mode", [RiscvMode.S, RiscvMode.D])
    def test_supervisor_mode_rejects_user_va(self, mode):
        dbg, emu, _ = _make_dbg()
        hart = emu.harts[0]
        hart.satp_val = SATP_SV39
        hart.mode = mode
        assert dbg._is_valid_code_va(USER_VA) is False

    def test_user_mode_accepts_user_va(self):
        dbg, emu, _ = _make_dbg()
        hart = emu.harts[0]
        hart.satp_val = SATP_SV39
        hart.mode = RiscvMode.U
        assert dbg._is_valid_code_va(USER_VA) is True

    def test_user_mode_rejects_kernel_va(self):
        dbg, emu, _ = _make_dbg()
        hart = emu.harts[0]
        hart.satp_val = SATP_SV39
        hart.mode = RiscvMode.U
        assert dbg._is_valid_code_va(KERNEL_VA) is False

    def test_non_canonical_address_is_rejected(self):
        dbg, emu, _ = _make_dbg()
        hart = emu.harts[0]
        hart.satp_val = SATP_SV39
        hart.mode = RiscvMode.S
        assert dbg._is_valid_code_va(NON_CANONICAL_VA) is False


# ============================================================
#  cmd_symbols
# ============================================================


class TestCmdSymbols:
    def test_without_image_reports_warning(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_symbols()
        assert "无可用的符号表" in buf.getvalue()

    def test_empty_symbol_table_is_reported(self):
        dbg, _, buf = _make_dbg(image=_fw_image())
        dbg.cmd_symbols()
        assert "符号表为空" in buf.getvalue()

    def test_grouping_counts_symbols_by_first_character(self):
        dbg, _, buf = _make_dbg(image=_fw_image(symbols={"alpha": 1, "beta": 2, "ax": 3}))
        dbg.cmd_symbols()
        text = buf.getvalue()
        assert "符号分组 (3 项, 2 组)" in text
        assert "a..." in text
        assert "b..." in text
        assert "sym <prefix>" in text

    def test_prefix_filter_lists_matching_addresses(self):
        image = _fw_image(symbols={"alpha": 0x1000, "ax": 0x1004, "beta": 0x2000})
        dbg, _, buf = _make_dbg(image=image)
        dbg.cmd_symbols("a")
        text = buf.getvalue()
        assert "alpha" in text
        assert "ax" in text
        assert "beta" not in text

    def test_prefix_filter_is_case_insensitive(self):
        image = _fw_image(symbols={"Alpha": 0x1000})
        dbg, _, buf = _make_dbg(image=image)
        dbg.cmd_symbols("aL")
        assert "Alpha" in buf.getvalue()

    def test_unmatched_prefix_reports_nothing_found(self):
        dbg, _, buf = _make_dbg(image=_fw_image(symbols={"alpha": 0x1000}))
        dbg.cmd_symbols("zz")
        assert "无以前缀 'zz' 开头的符号" in buf.getvalue()

    def test_many_matches_are_split_into_subgroups(self):
        """超过 50 项时按 filter 之后的一位再分组."""
        symbols = {"f": 0x1000}
        symbols.update(_many_symbols("f", 60))
        dbg, _, buf = _make_dbg(image=_fw_image(symbols=symbols))
        dbg.cmd_symbols("f")
        text = buf.getvalue()
        assert "子组" in text
        assert "f0..." in text
        assert "sym f0" in text

    def test_single_subgroup_repeats_its_own_name_in_hint(self):
        """只分出一个子组时提示中两次给出同一名称."""
        symbols = _many_symbols("zzz", 60)
        dbg, _, buf = _make_dbg(image=_fw_image(symbols=symbols))
        dbg.cmd_symbols("zz")
        assert "sym zzz, sym zzz" in buf.getvalue()


# ============================================================
#  cmd_sym
# ============================================================


class TestCmdSym:
    @staticmethod
    def _with_candidate(dbg, tmp_path) -> Path:
        """在候选清单中放入一个真实存在的文件."""
        target = tmp_path / "app.elf"
        target.write_bytes(b"\x00")
        dbg._sym_mod_dir = str(tmp_path)
        dbg._sym_avail_items = [
            SymAvailItem(
                number=1,
                name="app.elf",
                path=str(target),
                size=1,
                is_readable=True,
                is_loaded=False,
            ),
        ]
        return target

    def test_load_without_path_reports_usage(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_sym("load")
        assert "用法: sym load <#n|文件名> [pa:<addr>|va:<addr>]" in buf.getvalue()

    @pytest.mark.parametrize(
        ("spec", "space", "base"),
        [
            ("", None, None),
            ("pa:0x80200000", "pa", 0x8020_0000),
            ("va:0x40000000", "va", 0x4000_0000),
        ],
    )
    def test_load_forwards_candidate_and_address(
        self, monkeypatch, tmp_path, spec, space, base
    ):
        recorded = {}

        def fake_load(dbg, path, space_arg, base_arg):
            recorded["path"] = path
            recorded["space"] = space_arg
            recorded["base"] = base_arg

        dbg, _, _ = _make_dbg()
        target = self._with_candidate(dbg, tmp_path)
        monkeypatch.setattr(sym.cmd_files.loader, "load_symbol_module", fake_load)
        dbg.cmd_sym(f"load #1 {spec}")
        assert recorded == {"path": str(target), "space": space, "base": base}

    def test_load_without_address_prefix_reports_usage(self, tmp_path):
        dbg, _, buf = _make_dbg()
        self._with_candidate(dbg, tmp_path)
        dbg.cmd_sym("load #1 0x80200000")
        assert sym.cmd_files.ADDR_USAGE in buf.getvalue()

    def test_load_rejects_filesystem_path(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_sym("load /tmp/vmlinux")
        assert "只接受候选清单" in buf.getvalue()

    def test_info_without_symbols_is_reported(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_sym("info")
        assert "未加载外部调试符号" in buf.getvalue()

    def test_info_summarises_loaded_symbols(self):
        dbg, _, buf = _make_dbg()
        dbg._sym_symbols = {"main": 0x1000, "echo": 0x2000}
        dbg._sym_path = "/tmp/vmlinux"
        dbg._sym_load_offset = 0x8000
        dbg.cmd_sym("info")
        text = buf.getvalue()
        assert "/tmp/vmlinux" in text
        assert "符号数: 2" in text
        assert "0x1000" in text
        assert "load_offset: 0x8000" in text

    def test_list_subcommand_filters_symbols(self):
        dbg, _, buf = _make_dbg(image=_fw_image(symbols={"alpha": 0x1000}))
        dbg.cmd_sym("list a")
        assert "alpha" in buf.getvalue()

    def test_empty_argument_lists_all_symbols(self):
        dbg, _, buf = _make_dbg(image=_fw_image(symbols={"alpha": 0x1000}))
        dbg.cmd_sym("")
        assert "符号分组" in buf.getvalue()

    def test_unknown_subcommand_falls_back_to_prefix_filter(self):
        """非子命令的关键字整串作为前缀过滤."""
        dbg, _, buf = _make_dbg(image=_fw_image(symbols={"alpha": 0x1000}))
        dbg.cmd_sym("alpha")
        assert "alpha" in buf.getvalue()


# ============================================================
#  目录约定
# ============================================================


def test_module_file_exists():
    """测试对象为 pyremu/debug/sym/__init__.py, 避免导入到同名替身."""
    module_path = Path(sym.__file__)
    assert module_path.name == "__init__.py"
    assert module_path.parent.name == "sym"
