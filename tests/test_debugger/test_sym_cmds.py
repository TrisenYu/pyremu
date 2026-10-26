#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""测试 sym 的子命令与模块清单管理."""

from __future__ import annotations

from dataclasses import replace
import io
import os
from pathlib import Path

import pytest
from rich.console import Console

from pyremu.debug import base, Debugger
from pyremu.debug.exception import UsageError
from pyremu.debug.sym import cmd_files, cmd_modules, registry
from pyremu.debug.sym.model import SymAvailItem, SymMod
from pyremu.debug.utils import fmt_display_name
from pyremu.emulator import Emulator

RAM_BASE = 0x8000_0000
RAM_SIZE = 0x0001_0000


class _FakeIndex:
    def __init__(self, names=None, addrs=None) -> None:
        self._names = dict(names or {})
        self._addrs = dict(addrs or {})
        self.is_closed = False

    def resolve_addr(self, link: int) -> str | None:
        return self._addrs.get(link)

    def lookup_name(self, name: str) -> int | None:
        return self._names.get(name)

    def close(self) -> None:
        self.is_closed = True


_BASE_MOD = SymMod(
    file_name="app.elf",
    path="/tmp/app.elf",
    pa_base=RAM_BASE,
    va_base=None,
    cover_start=0x1000,
    cover_end=0x2000,
    sym_count=2,
    is_verified=True,
    load_order=0,
    verify_epoch=0,
    index=None,
)


def _mod(**changes) -> SymMod:
    return replace(_BASE_MOD, **changes)


def _make_dbg():
    emu = Emulator(num_harts=1, ram_size=RAM_SIZE, ram_base=RAM_BASE, prog_cnt=0x1000)
    emu._termio = None
    dbg = Debugger(emulator=emu, hart_id=0, image=None)
    dbg._sym_mod_dir = None
    buf = io.StringIO()
    dbg._console = Console(file=buf, highlight=False, width=200)
    return dbg, buf


def _item(number: int, path, *, is_readable: bool = True) -> SymAvailItem:
    return SymAvailItem(
        number=number,
        name=path.name,
        path=str(path),
        size=path.stat().st_size if path.exists() else 0,
        is_readable=is_readable,
        is_loaded=False,
    )


def _drop_candidates(dbg, _target) -> None:
    """清空候选清单, 对应未执行 sym ls 的情形."""
    dbg._sym_avail_items = []


def _remove_candidate(_dbg, target) -> None:
    """删除候选文件, 对应 sym ls 之后文件被删除的情形."""
    target.unlink()


# ============================================================
#  sym chdir / sym ls
# ============================================================


class TestChdirAndLs:
    def test_chdir_without_argument_reports_unset_directory(self):
        dbg, buf = _make_dbg()
        dbg.cmd_sym("chdir")
        assert "未设置候选目录" in buf.getvalue()

    def test_chdir_records_directory_and_clears_candidates(self, tmp_path):
        dbg, buf = _make_dbg()
        dbg._sym_avail_items = [_item(1, tmp_path / "x.elf")]
        dbg.cmd_sym(f"chdir {tmp_path}")
        assert dbg._sym_mod_dir == str(tmp_path)
        assert dbg._sym_avail_items == []
        assert "候选目录设为" in buf.getvalue()

    @pytest.mark.parametrize("target", ["..", "../../../../../../", "~", "build/elf"])
    def test_chdir_rejects_relative_path(self, target):
        """候选目录只接受绝对路径: 相对路径既不登记, 也不清空既有候选清单."""
        dbg, buf = _make_dbg()
        dbg._sym_mod_dir = "/tmp"
        dbg._sym_avail_items = [_item(1, Path("/tmp/x.elf"))]
        dbg.cmd_sym(f"chdir {target}")
        assert "绝对路径" in buf.getvalue()
        assert dbg._sym_mod_dir == "/tmp"
        assert len(dbg._sym_avail_items) == 1

    def test_chdir_without_argument_reports_current_directory(self, tmp_path):
        dbg, buf = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path)
        dbg.cmd_sym("chdir")
        assert str(tmp_path) in buf.getvalue()

    def test_pwd_shows_candidate_directory(self, tmp_path):
        dbg, buf = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path)
        dbg.cmd_sym("pwd")
        assert str(tmp_path) in buf.getvalue()

    def test_pwd_reports_unset_directory(self):
        dbg, buf = _make_dbg()
        dbg.cmd_sym("pwd")
        assert "未设置候选目录" in buf.getvalue()

    def test_pwd_is_not_treated_as_symbol_prefix(self, monkeypatch):
        """pwd 是保留子命令, 不再作为固件符号前缀过滤."""
        dbg, _buf = _make_dbg()
        calls: list[str] = []
        monkeypatch.setattr(dbg, "cmd_symbols", calls.append)
        dbg.cmd_sym("pwd")
        assert calls == []

    def test_ls_without_directory_reports_hint(self):
        dbg, buf = _make_dbg()
        dbg.cmd_sym("ls")
        assert "先执行 sym chdir" in buf.getvalue()

    def test_ls_reports_os_error(self, tmp_path):
        dbg, buf = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path / "absent")
        dbg.cmd_sym("ls")
        assert "无法列出目录" in buf.getvalue()
        assert dbg._sym_avail_items == []

    def test_ls_lists_regular_files_in_order(self, tmp_path):
        (tmp_path / "b.elf").write_bytes(b"\x00\x01")
        (tmp_path / "a.elf").write_bytes(b"\x00")
        (tmp_path / "nested").mkdir()
        dbg, buf = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path)
        dbg.cmd_sym("ls")
        items = dbg._sym_avail_items
        assert [item.name for item in items] == ["a.elf", "b.elf"]
        assert [item.number for item in items] == [1, 2]
        assert items[1].size == 2
        assert all(item.is_readable for item in items)
        assert "候选文件" in buf.getvalue()

    def test_ls_marks_loaded_module(self, tmp_path):
        target = tmp_path / "app.elf"
        target.write_bytes(b"\x00")
        dbg, _ = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path)
        dbg._sym_mods.append(_mod(path=str(target)))
        dbg.cmd_sym("ls")
        assert dbg._sym_avail_items[0].is_loaded

    def test_ls_on_empty_directory_reports_no_files(self, tmp_path):
        dbg, buf = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path)
        dbg.cmd_sym("ls")
        assert "中没有普通文件" in buf.getvalue()

    def test_unreadable_file_is_listed_as_unreadable(self, tmp_path, monkeypatch):
        target = tmp_path / "secret.elf"
        target.write_bytes(b"\x00")
        monkeypatch.setattr(cmd_files, "is_file_readable", lambda path: False)
        dbg, _ = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path)
        dbg.cmd_sym("ls")
        assert not dbg._sym_avail_items[0].is_readable

    def test_file_size_error_is_reported_as_zero(self, tmp_path, monkeypatch):
        target = tmp_path / "app.elf"
        target.write_bytes(b"\x00")

        def fail(path):
            raise OSError("stat failed")

        monkeypatch.setattr(cmd_files, "file_size_bytes", fail)
        dbg, _ = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path)
        dbg.cmd_sym("ls")
        assert dbg._sym_avail_items[0].size == 0


# ============================================================
#  sym load
# ============================================================


class TestLoadCommand:
    @staticmethod
    def _candidate(dbg, tmp_path, name: str = "app.elf"):
        target = tmp_path / name
        target.write_bytes(b"\x00")
        dbg._sym_mod_dir = str(tmp_path)
        dbg._sym_avail_items = [_item(1, target)]
        return target

    @pytest.mark.parametrize(
        "sym_name, expected_str",
        [
            ("load", cmd_files.LOAD_USAGE),
            ("load #1", "未设置候选目录"),
            ("load #-1", "未设置候选目录"),
            ("load #12", "未设置候选目录"),
            ("load #0x123", "未设置候选目录"),
            ("load #abc", "未设置候选目录"),
            ("load ##", "未设置候选目录"),
            ("load /tmp/app.elf", "只接受候选清单"),
            ("load /etc/passwd", "只接受候选清单"),
            ("load ../../../../../root/.bashrc", "只接受候选清单"),
        ],
    )
    def test_cmd_sym(self, sym_name: str, expected_str: str):
        dbg, buf = _make_dbg()
        dbg.cmd_sym(sym_name)
        assert expected_str in buf.getvalue()

    @pytest.mark.parametrize(
        ("arguments", "expected", "prepare"),
        [
            ("#1", "候选清单为空", _drop_candidates),
            ("other.elf", "候选清单中没有文件", None),
            ("#x", "编号不是整数", None),
            ("#9", "候选清单中没有编号", None),
            ("#1", "文件不存在", _remove_candidate),
            ('#1 ""', cmd_files.ADDR_USAGE, None),
            ("#1 0x80000000", cmd_files.ADDR_USAGE, None),
            ("#1 pa:zz", "zz", None),
            ("#1 pa:-1", "-1", None),
        ],
    )
    def test_load_rejects_invalid_selection(self, tmp_path, arguments, expected, prepare):
        dbg, buf = _make_dbg()
        target = self._candidate(dbg, tmp_path)
        if prepare is not None:
            prepare(dbg, target)
        dbg.cmd_sym(f"load {arguments}")
        assert expected in buf.getvalue()
        assert dbg._sym_mods == []

    def test_load_rejects_unreadable_candidate(self, tmp_path):
        dbg, buf = _make_dbg()
        target = tmp_path / "app.elf"
        target.write_bytes(b"\x00")
        dbg._sym_mod_dir = str(tmp_path)
        dbg._sym_avail_items = [_item(1, target, is_readable=False)]
        dbg.cmd_sym("load #1")
        assert "文件不可读" in buf.getvalue()

    def test_load_rejects_file_that_became_unreadable(self, tmp_path, monkeypatch):
        dbg, buf = _make_dbg()
        self._candidate(dbg, tmp_path)
        monkeypatch.setattr(cmd_files, "is_file_readable", lambda path: False)
        dbg.cmd_sym("load #1")
        assert "文件不可读" in buf.getvalue()

    def test_load_reads_bare_digits_as_hex(self, tmp_path, monkeypatch):
        dbg, _ = _make_dbg()
        self._candidate(dbg, tmp_path)
        recorded = {}

        def fake_load(dbg_arg, path, space, base):
            recorded.update(space=space, base=base)

        monkeypatch.setattr(cmd_files.loader, "load_symbol_module", fake_load)
        dbg.cmd_sym("load #1 pa:10")
        assert recorded == {"space": "pa", "base": 0x10}

    def test_load_by_number_forwards_address_and_registers(self, tmp_path, monkeypatch):
        dbg, buf = _make_dbg()
        target = self._candidate(dbg, tmp_path)
        recorded = {}
        mod = _mod(path=str(target), pa_base=RAM_BASE, va_base=0x4000_0000)

        def fake_load(dbg_arg, path, space, base):
            recorded.update(path=path, space=space, base=base)
            return mod

        monkeypatch.setattr(cmd_files.loader, "load_symbol_module", fake_load)
        dbg.cmd_sym("load #1 va:0x40000000")
        assert recorded == {"path": str(target), "space": "va", "base": 0x4000_0000}
        assert dbg._sym_mods == [mod] and dbg._pa_to_va[RAM_BASE] == 0x4000_0000
        assert "模块序号 1" in buf.getvalue()

    def test_load_by_name_uses_link_address_by_default(self, tmp_path, monkeypatch):
        dbg, _ = _make_dbg()
        self._candidate(dbg, tmp_path)
        recorded = {}

        def fake_load(dbg_arg, path, space, base):
            recorded.update(space=space, base=base)

        monkeypatch.setattr(cmd_files.loader, "load_symbol_module", fake_load)
        dbg.cmd_sym("load app.elf")
        assert recorded == {"space": None, "base": None}
        assert dbg._sym_mods == []

    def test_load_failure_does_not_register(self, tmp_path, monkeypatch):
        dbg, _ = _make_dbg()
        self._candidate(dbg, tmp_path)
        monkeypatch.setattr(cmd_files.loader, "load_symbol_module", lambda *args: None)
        dbg.cmd_sym("load #1")
        assert dbg._sym_mods == []


class TestAddressSpec:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("", (None, None)),
            ("   ", (None, None)),
            ("PA:0x80000000", ("pa", RAM_BASE)),
            ("pa:0xffffffffffffffff", ("pa", (1 << 64) - 1)),
            ("pa:80000000", ("pa", 0x8000_0000)),
            ("pa:0b1010", ("pa", 0xB1010)),
        ],
    )
    def test_valid_spec_is_parsed(self, text, expected):
        assert cmd_files._parse_addr_spec(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            ":",
            "pa:",
            "vA:",
            "pa:0x",
            "pa:0x10000000000000000",
            "pa:1_0",
            "pa:1'0",
            "pa:0o17",
            "pa:1g",
        ],
    )
    def test_incomplete_or_invalid_spec_is_rejected(self, text):
        with pytest.raises(UsageError):
            cmd_files._parse_addr_spec(text)


# ============================================================
#  sym detail / sym rm / sym info
# ============================================================


class TestModuleCommands:
    def test_detail_without_modules_reports_empty(self):
        dbg, buf = _make_dbg()
        dbg.cmd_sym("detail")
        assert "没有已加载的符号模块" in buf.getvalue()

    def test_detail_lists_loaded_modules(self):
        dbg, buf = _make_dbg()
        dbg._sym_mods.append(_mod(index=_FakeIndex(), is_verified=False))
        dbg.cmd_sym("detail")
        text = buf.getvalue()
        assert "符号模块" in text and "未校验" in text and "/tmp/app.elf" in text

    def test_detail_shows_both_bases(self):
        dbg, buf = _make_dbg()
        dbg._sym_mods.append(_mod(va_base=0x4000_0000, is_verified=False))
        dbg.cmd_sym("detail")
        assert "pa 0x80000000 / va 0x40000000" in buf.getvalue()

    def test_detail_lists_modules_covering_address(self):
        dbg, buf = _make_dbg()
        index = _FakeIndex(addrs={0x1000: "main"})
        dbg._sym_mods.append(_mod(index=index, is_verified=False))
        dbg.cmd_sym(f"detail 0x{RAM_BASE:x}")
        text = buf.getvalue()
        assert "覆盖该地址的模块" in text and "main" in text and "是" in text

    def test_detail_reports_uncovered_address(self):
        dbg, buf = _make_dbg()
        dbg._sym_mods.append(_mod(index=_FakeIndex(), is_verified=False))
        dbg.cmd_sym("detail 0x1000")
        assert "没有符号模块覆盖该地址" in buf.getvalue()

    def test_detail_reports_unresolvable_text(self):
        dbg, buf = _make_dbg()
        dbg.cmd_sym("detail nosuchthing")
        assert "无法解析地址或符号名" in buf.getvalue()

    def test_rm_without_selector_reports_usage(self):
        dbg, buf = _make_dbg()
        dbg.cmd_sym("rm")
        assert "用法: sym rm" in buf.getvalue()

    def test_rm_without_modules_reports_hint(self):
        dbg, buf = _make_dbg()
        dbg.cmd_sym("rm 1")
        assert "没有已加载的符号模块" in buf.getvalue()

    def test_rm_rejects_number_out_of_range(self):
        dbg, buf = _make_dbg()
        dbg._sym_mods.append(_mod())
        dbg.cmd_sym("rm 2")
        assert "模块序号超出范围" in buf.getvalue()

    def test_rm_rejects_unknown_file_name(self):
        dbg, buf = _make_dbg()
        dbg._sym_mods.append(_mod())
        dbg.cmd_sym("rm other.elf")
        assert "没有文件名为 other.elf 的模块" in buf.getvalue()

    def test_rm_requires_number_when_name_is_ambiguous(self):
        dbg, buf = _make_dbg()
        dbg._sym_mods.append(_mod())
        dbg._sym_mods.append(_mod(path="/tmp/other/app.elf"))
        dbg.cmd_sym("rm app.elf")
        assert "对应多个模块" in buf.getvalue()

    def test_rm_by_number_removes_module(self):
        dbg, buf = _make_dbg()
        mod = _mod()
        dbg._sym_mods.append(mod)
        dbg.cmd_sym("rm 1")
        assert dbg._sym_mods == []
        assert "已移除" in buf.getvalue()

    def test_rm_by_unique_name_removes_module(self):
        dbg, _ = _make_dbg()
        dbg._sym_mods.append(_mod())
        dbg.cmd_sym("rm app.elf")
        assert dbg._sym_mods == []

    def test_rm_legacy_module_clears_symbol_tables(self):
        dbg, _ = _make_dbg()
        dbg._sym_mods.append(_mod(is_verified=False))
        dbg._sym_symbols = {"main": 0x1000}
        dbg._sym_path = "/tmp/vmlinux"
        dbg._sym_load_offset = 0x8000
        dbg._sym_ranges = [(0x1000, 0x2000, "main")]
        dbg.cmd_sym("rm 1")
        assert dbg._sym_symbols == {}
        assert dbg._sym_path is None
        assert dbg._sym_load_offset == 0
        assert dbg._sym_ranges == []

    def test_info_without_symbols_but_with_modules(self):
        dbg, buf = _make_dbg()
        dbg._sym_mods.append(_mod())
        dbg.cmd_sym("info")
        text = buf.getvalue()
        assert "未加载外部调试符号" in text
        assert "符号模块: 1 个" in text

    def test_info_appends_module_count(self):
        dbg, buf = _make_dbg()
        dbg._sym_symbols = {"main": 0x1000}
        dbg._sym_path = "/tmp/vmlinux"
        dbg._sym_mods.append(_mod())
        dbg.cmd_sym("info")
        text = buf.getvalue()
        assert "/tmp/vmlinux" in text and "符号模块: 1 个" in text

    def test_base_text_marks_unknown_bases(self):
        assert cmd_modules._base_text(_mod(pa_base=None, va_base=0x1000)) == "va 0x1000"
        assert cmd_modules._base_text(_mod(pa_base=None, va_base=None)) == "-"


# ============================================================
#  模块清单
# ============================================================


class TestRegistry:
    def test_load_order_increases(self):
        dbg, _ = _make_dbg()
        first = _mod()
        second = _mod()
        registry.assign_load_order(dbg, first)
        registry.assign_load_order(dbg, second)
        assert (first.load_order, second.load_order) == (0, 1)

    def test_register_replaces_same_path(self, tmp_path):
        dbg, _ = _make_dbg()
        path = str(tmp_path / "app.elf")
        old_index = _FakeIndex()
        dbg._sym_mods.append(_mod(path=path, index=old_index))
        new = _mod(path=path)
        registry.register_module(dbg, new)
        assert dbg._sym_mods == [new]
        assert old_index.is_closed

    def test_find_module_by_path_uses_absolute_paths(self, tmp_path):
        dbg, _ = _make_dbg()
        dbg._sym_mods.append(_mod(path=str(tmp_path / "app.elf")))
        found = registry.find_module_by_path(dbg, str(tmp_path / "app.elf"))
        assert found is not None
        assert registry.find_module_by_path(dbg, str(tmp_path / "other.elf")) is None

    def test_remove_legacy_modules_keeps_verified_modules(self, tmp_path):
        dbg, _ = _make_dbg()
        legacy = _mod(path="/tmp/legacy.elf", is_verified=False)
        verified = _mod(path="/tmp/verified.elf", is_verified=True)
        dbg._sym_mods.extend([legacy, verified])
        registry.remove_legacy_modules(dbg)
        assert dbg._sym_mods == [verified]

    def test_refresh_marks_invalid_and_reports_once(self, monkeypatch):
        dbg, buf = _make_dbg()
        mod = _mod(verify_epoch=0)
        dbg._sym_mods.append(mod)
        dbg._instr_count = 1
        monkeypatch.setattr(registry.loader, "verify_module", lambda *args: False)
        registry.refresh_modules(dbg)
        assert not mod.is_valid
        registry.refresh_modules(dbg)
        assert buf.getvalue().count("已标记失效") == 1

    def test_refresh_skips_unverified_and_unchanged_modules(self, monkeypatch):
        dbg, _ = _make_dbg()
        unverified = _mod(is_verified=False)
        unchanged = _mod(verify_epoch=0)
        dbg._sym_mods.extend([unverified, unchanged])
        dbg._instr_count = 0
        calls = []

        def verify(*args):
            calls.append(args)
            return True

        monkeypatch.setattr(registry.loader, "verify_module", verify)
        registry.refresh_modules(dbg)
        assert calls == []

    def test_refresh_keeps_valid_module(self, monkeypatch):
        dbg, _ = _make_dbg()
        mod = _mod(verify_epoch=0)
        dbg._sym_mods.append(mod)
        dbg._instr_count = 5
        monkeypatch.setattr(registry.loader, "verify_module", lambda *args: True)
        registry.refresh_modules(dbg)
        assert mod.is_valid
        assert mod.verify_epoch == 5

    def test_lookup_modules_skips_invalid(self, monkeypatch):
        dbg, _ = _make_dbg()
        valid = _mod()
        invalid = _mod(is_verified=True, is_valid=False)
        dbg._sym_mods.extend([valid, invalid])
        monkeypatch.setattr(registry.loader, "verify_module", lambda *args: True)
        assert registry.lookup_modules(dbg) == [valid]


def test_sym_command_dispatch_reaches_subcommands(tmp_path):
    """cmd_sym 把各子命令交给对应的实现."""
    dbg, buf = _make_dbg()
    dbg._sym_mod_dir = str(tmp_path)
    dbg.cmd_sym("chdir " + str(tmp_path))
    dbg.cmd_sym("ls")
    dbg.cmd_sym("detail")
    dbg.cmd_sym("info")
    dbg.cmd_sym("rm 1")
    text = buf.getvalue()
    assert "候选目录设为" in text
    assert "中没有普通文件" in text
    assert "没有已加载的符号模块" in text
    assert "未加载外部调试符号" in text
    assert "没有已加载的符号模块; 执行 sym load 加载" in text


# ============================================================
#  含空白与控制字符的文件名
# ============================================================


class TestQuotedArguments:
    def _named_candidate(self, tmp_path, name: str):
        target = tmp_path / name
        target.write_bytes(b"\x00")
        dbg, buf = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path)
        dbg._sym_avail_items = [_item(1, target)]
        return dbg, buf, target

    def test_quoted_name_with_space_is_selected(self, tmp_path, monkeypatch):
        dbg, _buf, target = self._named_candidate(tmp_path, "has space.elf")
        recorded = {}

        def fake_load(dbg_arg, path, space, base):
            recorded.update(path=path)

        monkeypatch.setattr(cmd_files.loader, "load_symbol_module", fake_load)
        dbg.cmd_sym('load "has space.elf" pa:0x80000000')
        assert recorded == {"path": str(target)}

    def test_unquoted_name_with_spaces_reports_quote_hint(self, tmp_path):
        dbg, buf, _target = self._named_candidate(tmp_path, "has space.elf")
        dbg.cmd_sym("load has space name.elf")
        assert cmd_files.QUOTE_HINT in buf.getvalue()
        assert dbg._sym_mods == []

    def test_two_token_name_without_prefix_reports_address_usage(self, tmp_path):
        dbg, buf, _target = self._named_candidate(tmp_path, "has space.elf")
        dbg.cmd_sym("load has space.elf")
        assert cmd_files.ADDR_USAGE in buf.getvalue()
        assert dbg._sym_mods == []

    def test_unbalanced_quotes_are_rejected(self, tmp_path):
        dbg, buf, _target = self._named_candidate(tmp_path, "has space.elf")
        dbg.cmd_sym('load "has space.elf')
        assert "引号不配对" in buf.getvalue()

    def test_chdir_accepts_quoted_directory(self, tmp_path):
        target = tmp_path / "dir with space"
        target.mkdir()
        dbg, _buf = _make_dbg()
        dbg.cmd_sym(f'chdir "{target}"')
        assert dbg._sym_mod_dir == str(target)

    def test_chdir_keeps_unquoted_directory_with_space(self, tmp_path):
        target = tmp_path / "dir with space"
        target.mkdir()
        dbg, _buf = _make_dbg()
        dbg.cmd_sym(f"chdir {target}")
        assert dbg._sym_mod_dir == str(target)

    def test_rm_accepts_quoted_name_with_space(self, tmp_path):
        target = tmp_path / "has space.elf"
        dbg, buf = _make_dbg()
        dbg._sym_mods.append(_mod(file_name=target.name, path=str(target)))
        dbg.cmd_sym('rm "has space.elf"')
        assert dbg._sym_mods == []
        assert "已移除" in buf.getvalue()


class TestCandidateDisplay:
    @pytest.mark.parametrize(
        ("name", "expected", "absent"),
        [
            ("has space.elf", '"has space.elf"', None),
            ("esc\x1b[31m.elf", "\\x1b", "\x1b"),
            ("符号.elf", "符号.elf", None),
        ],
    )
    def test_ls_renders_candidate_name(self, tmp_path, name, expected, absent):
        (tmp_path / name).write_bytes(b"\x00")
        dbg, buf = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path)
        dbg.cmd_sym("ls")
        text = buf.getvalue()
        assert expected in text
        if absent is not None:
            assert absent not in text


class TestDisplayNameFormat:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("vmlinux", "vmlinux"),
            ("has space.elf", '"has space.elf"'),
            ("a\x1bb", '"a\\x1bb"'),
            ("a\udcffb", '"a\\udcffb"'),
            ("a\"b' c", '"a\\"b\' c"'),
        ],
    )
    def test_name_is_written_for_display(self, text, expected):
        assert fmt_display_name(text) == expected


class TestSymDispatch:
    def test_dispatch_forwards_all_arguments(self, tmp_path, monkeypatch):
        target = tmp_path / "app.elf"
        target.write_bytes(b"\x00")
        dbg, _buf = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path)
        dbg._sym_avail_items = [_item(1, target)]
        recorded = {}

        def fake_load(dbg_arg, path, space, base):
            recorded.update(path=path, space=space, base=base)

        monkeypatch.setattr(cmd_files.loader, "load_symbol_module", fake_load)
        dbg._dispatch(["sym", "load", "#1", "pa:0x80000000"])
        assert recorded == {"path": str(target), "space": "pa", "base": RAM_BASE}

    def test_dispatch_uses_repl_rest_text(self, tmp_path, monkeypatch):
        target = tmp_path / "has space.elf"
        target.write_bytes(b"\x00")
        dbg, _buf = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path)
        dbg._sym_avail_items = [_item(1, target)]
        recorded = {}

        def fake_load(dbg_arg, path, space, base):
            recorded.update(path=path)

        monkeypatch.setattr(cmd_files.loader, "load_symbol_module", fake_load)
        line = 'sym load "has space.elf"'
        dbg._dispatch(line.split(), line[len("sym") :].lstrip())
        assert recorded == {"path": str(target)}

    def test_dispatch_without_rest_falls_back_to_parts(self):
        dbg, buf = _make_dbg()
        dbg._dispatch(["sym", "help"])
        assert "chdir" in buf.getvalue()
        assert "pwd" in buf.getvalue()


class TestInitialCandidateDir:
    def test_initial_value_is_current_directory(self):
        assert base.init_sym_mod_dir() == os.getcwd()

    def test_falls_back_to_home_when_cwd_is_gone(self, monkeypatch):
        def raise_oserror():
            raise OSError

        monkeypatch.setattr(base.os, "getcwd", raise_oserror)
        monkeypatch.setenv("HOME", "/tmp/home")
        assert base.init_sym_mod_dir() == "/tmp/home"

    def test_returns_none_without_cwd_and_home(self, monkeypatch):
        def raise_oserror():
            raise OSError

        monkeypatch.setattr(base.os, "getcwd", raise_oserror)
        monkeypatch.delenv("HOME", raising=False)
        assert base.init_sym_mod_dir() is None


class TestUnknownCandidateNumber:
    def test_out_of_range_number_reports_valid_range(self, tmp_path):
        (tmp_path / "a.elf").write_bytes(b"\x00")
        (tmp_path / "b.elf").write_bytes(b"\x00")
        dbg, buf = _make_dbg()
        dbg._sym_mod_dir = str(tmp_path)
        dbg.cmd_sym("ls")
        dbg.cmd_sym("load #9")
        assert "有效编号为 1 至 2" in buf.getvalue()
