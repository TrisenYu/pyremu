#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""测试 pyremu.debug.sym.model — 模块选取、地址换算与状态文本."""

from __future__ import annotations

from dataclasses import replace

from pyremu.debug.sym import model


class _FakeIndex:
    def __init__(self, names=None, addrs=None):
        self._names = dict(names or {})
        self._addrs = dict(addrs or {})

    def resolve_addr(self, link: int) -> str | None:
        return self._addrs.get(link)

    def lookup_name(self, name: str) -> int | None:
        return self._names.get(name)


_BASE_MOD = model.SymMod(
    file_name="app.elf",
    path="/tmp/app.elf",
    pa_base=0x8000_0000,
    va_base=None,
    cover_start=0x1000,
    cover_end=0x2000,
    sym_count=1,
    is_verified=True,
    load_order=0,
    verify_epoch=0,
    index=None,
    is_valid=True,
)


def _mod(**changes) -> model.SymMod:
    """以默认模块为基准, 按字段覆盖."""
    return replace(_BASE_MOD, **changes)


# ============================================================
#  基本属性
# ============================================================


def test_span_is_cover_range_length():
    assert _mod(cover_start=0x1000, cover_end=0x1800).span == 0x800


def test_status_name_distinguishes_unverified_valid_and_invalid():
    assert _mod(is_verified=False).status_name == "未校验"
    assert _mod(is_verified=True, is_valid=True).status_name == "有效"
    assert _mod(is_verified=True, is_valid=False).status_name == "失效"


def test_module_spaces_lists_known_bases():
    assert model.module_spaces(_mod(pa_base=0x1000, va_base=None)) == ("pa",)
    assert model.module_spaces(_mod(pa_base=None, va_base=0x1000)) == ("va",)
    assert model.module_spaces(_mod(pa_base=0x1000, va_base=0x2000)) == ("pa", "va")
    assert model.module_spaces(_mod(pa_base=None, va_base=None)) == ()


# ============================================================
#  链接地址换算
# ============================================================


class TestLinkAddrOf:
    def test_physical_address_is_converted(self):
        mod = _mod(pa_base=0x8000_0000)
        assert model.link_addr_of(mod, 0x8000_0100, model.SPACE_PA) == 0x1100

    def test_virtual_address_is_converted(self):
        mod = _mod(pa_base=None, va_base=0x4000_0000)
        assert model.link_addr_of(mod, 0x4000_0100, model.SPACE_VA) == 0x1100

    def test_unknown_base_yields_none(self):
        mod = _mod(pa_base=None, va_base=None)
        assert model.link_addr_of(mod, 0x1000, model.SPACE_PA) is None
        assert model.link_addr_of(mod, 0x1000, model.SPACE_VA) is None

    def test_range_bounds_are_half_open(self):
        mod = _mod(pa_base=0x8000_0000)
        assert model.link_addr_of(mod, 0x8000_0000, model.SPACE_PA) == 0x1000
        assert model.link_addr_of(mod, 0x8000_0FFF, model.SPACE_PA) == 0x1FFF
        assert model.link_addr_of(mod, 0x8000_1000, model.SPACE_PA) is None
        assert model.link_addr_of(mod, 0x7FFF_FFFF, model.SPACE_PA) is None


# ============================================================
#  按地址选取模块
# ============================================================


class TestSelectModule:
    def test_no_module_yields_none(self):
        assert model.select_module([], 0x8000_0000) is None

    def test_module_without_index_is_skipped(self):
        mod = _mod(index=None)
        assert model.select_module([mod], 0x8000_0000) is None

    def test_uncovered_address_yields_none(self):
        mod = _mod(index=_FakeIndex(addrs={0x1000: "main"}))
        assert model.select_module([mod], 0x7000_0000) is None

    def test_unresolved_symbol_is_skipped(self):
        mod = _mod(index=_FakeIndex(addrs={}))
        assert model.select_module([mod], 0x8000_0000) is None

    def test_resolved_module_is_selected(self):
        mod = _mod(index=_FakeIndex(addrs={0x1000: "main"}))
        selected = model.select_module([mod], 0x8000_0000)
        assert selected == (mod, model.SPACE_PA, "main")

    def test_virtual_space_is_selected_when_only_it_is_known(self):
        mod = _mod(
            pa_base=None,
            va_base=0x4000_0000,
            index=_FakeIndex(addrs={0x1100: "func"}),
        )
        assert model.select_module([mod], 0x4000_0100) == (mod, model.SPACE_VA, "func")

    def test_wider_range_wins(self):
        narrow = _mod(
            cover_end=0x1800,
            load_order=0,
            index=_FakeIndex(addrs={0x1100: "narrow"}),
            file_name="narrow.elf",
        )
        wide = _mod(
            cover_end=0x2000,
            load_order=1,
            index=_FakeIndex(addrs={0x1100: "wide"}),
            file_name="wide.elf",
        )
        selected = model.select_module([narrow, wide], 0x8000_0100)
        assert selected == (wide, model.SPACE_PA, "wide")

    def test_earlier_load_order_wins_for_equal_range(self):
        first = _mod(
            va_base=0x4000_0000,
            load_order=0,
            index=_FakeIndex(addrs={0x1100: "first"}),
            file_name="first.elf",
        )
        second = _mod(
            va_base=0x4000_0000,
            load_order=1,
            index=_FakeIndex(addrs={0x1100: "second"}),
            file_name="second.elf",
        )
        selected = model.select_module([second, first], 0x8000_0100)
        assert selected == (first, model.SPACE_PA, "first")

    def test_physical_space_wins_for_equal_range_and_order(self):
        physical = _mod(
            load_order=0,
            index=_FakeIndex(addrs={0x1100: "physical"}),
            file_name="physical.elf",
        )
        virtual = _mod(
            pa_base=None,
            va_base=0x8000_0000,
            load_order=0,
            index=_FakeIndex(addrs={0x1100: "virtual"}),
            file_name="virtual.elf",
        )
        selected = model.select_module([virtual, physical], 0x8000_0100)
        assert selected == (physical, model.SPACE_PA, "physical")

    def test_physical_space_is_preferred_within_one_module(self):
        mod = _mod(
            va_base=0x8000_0000,
            index=_FakeIndex(addrs={0x1100: "main"}),
        )
        assert model.select_module([mod], 0x8000_0100) == (mod, model.SPACE_PA, "main")


# ============================================================
#  按名称查找运行时地址
# ============================================================


class TestNameToRuntimeAddr:
    def test_module_without_index_is_skipped(self):
        assert model.name_to_runtime_addr([_mod(index=None)], "main") is None

    def test_missing_name_yields_none(self):
        mod = _mod(index=_FakeIndex(names={}))
        assert model.name_to_runtime_addr([mod], "main") is None

    def test_physical_base_gives_physical_address_only(self):
        mod = _mod(index=_FakeIndex(names={"main": 0x1100}))
        assert model.name_to_runtime_addr([mod], "main") == (0x8000_0100, None)

    def test_virtual_base_gives_virtual_address_only(self):
        mod = _mod(
            pa_base=None,
            va_base=0x4000_0000,
            index=_FakeIndex(names={"main": 0x1100}),
        )
        assert model.name_to_runtime_addr([mod], "main") == (None, 0x4000_0100)

    def test_both_bases_are_reported(self):
        mod = _mod(
            va_base=0x4000_0000,
            index=_FakeIndex(names={"main": 0x1100}),
        )
        assert model.name_to_runtime_addr([mod], "main") == (0x8000_0100, 0x4000_0100)

    def test_address_is_masked_to_64_bits(self):
        mod = _mod(
            pa_base=0xFFFF_FFFF_FFFF_FFFE,
            index=_FakeIndex(names={"main": 0x1004}),
        )
        pa, va = model.name_to_runtime_addr([mod], "main")
        assert pa == 2
        assert va is None

    def test_wider_range_wins(self):
        narrow = _mod(
            cover_end=0x1800,
            load_order=0,
            index=_FakeIndex(names={"main": 0x1100}),
            file_name="narrow.elf",
        )
        wide = _mod(
            cover_end=0x2000,
            load_order=1,
            index=_FakeIndex(names={"main": 0x1100}),
            file_name="wide.elf",
        )
        assert model.name_to_runtime_addr([narrow, wide], "main") == (0x8000_0100, None)
