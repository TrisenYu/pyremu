#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""测试 pyremu.debug.tlb_cache — TLB/Cache 显示."""

from pyremu.debug.tlb_cache import TlbCacheMixin
from pyremu.emulator import Emulator

# ============================================================
#  Mini 测试类
# ============================================================


class _TestTlbDbg(TlbCacheMixin):
    """最小聚合类供 TlbCacheMixin 测试."""

    def __init__(self, emu, hart_id=0):
        from rich.console import Console
        self._emu = emu
        self._hart_id = hart_id
        self._console = Console(highlight=False)
        self._warn = lambda msg: None
        self._err = lambda msg: None

    @property
    def hart(self):
        return self._emu.harts[self._hart_id]


def _make_tlbdbg():
    emu = Emulator(prog_cnt=0x1000)
    emu.load_code(0x1000, b"\x13\x00\x00\x00")  # NOP
    return _TestTlbDbg(emu)


# ============================================================
#  TLB 页大小
# ============================================================


class TestTlbPageSize:
    """_tlb_page_size 静态方法."""

    def test_level_0_is_4k(self):
        assert TlbCacheMixin._tlb_page_size(0) == "4K"

    def test_level_1_is_2m(self):
        assert TlbCacheMixin._tlb_page_size(1) == "2M"

    def test_level_2_is_1g(self):
        assert TlbCacheMixin._tlb_page_size(2) == "1G"

    def test_unknown_level(self):
        assert "Lv3" in TlbCacheMixin._tlb_page_size(3)


# ============================================================
#  hexdump
# ============================================================


class TestHexdumpBytes:
    """_hexdump_bytes 静态方法."""

    def test_empty(self):
        result = TlbCacheMixin._hexdump_bytes(b"")
        assert result == ""

    def test_short_data(self):
        data = b"\x00\x01\x02\x03"
        result = TlbCacheMixin._hexdump_bytes(data)
        assert "0000" in result
        assert "00 01 02 03" in result

    def test_sixteen_bytes_single_line(self):
        data = bytes(range(16))
        result = TlbCacheMixin._hexdump_bytes(data)
        lines = result.split("\n")
        assert len(lines) == 1

    def test_multi_line(self):
        data = bytes(range(32))
        result = TlbCacheMixin._hexdump_bytes(data)
        lines = result.split("\n")
        assert len(lines) == 2

    def test_indent(self):
        data = b"\x00\x01"
        result = TlbCacheMixin._hexdump_bytes(data, indent="  ")
        assert result.startswith("  ")


# ============================================================
#  TLB 命令
# ============================================================


class TestCmdTlb:
    """cmd_tlb — 默认/全部/范围/VPN 查找."""

    def test_default_shows_both_tlbs(self):
        dbg = _make_tlbdbg()
        dbg.cmd_tlb()  # 应不抛异常

    def test_all_flag(self):
        dbg = _make_tlbdbg()
        dbg.cmd_tlb("-a")

    def test_vpn_search(self):
        dbg = _make_tlbdbg()
        dbg.cmd_tlb("0x12345")

    def test_range_mode(self):
        dbg = _make_tlbdbg()
        dbg.cmd_tlb("10-20")

    def test_bad_range_reversed(self):
        dbg = _make_tlbdbg()
        dbg.cmd_tlb("20-10")  # end ≤ start


class TestCmdTlbflush:
    """cmd_tlbflush — 全部/单 VPN 刷新."""

    def test_flush_all(self):
        dbg = _make_tlbdbg()
        dbg.cmd_tlbflush()

    def test_flush_vpn(self):
        dbg = _make_tlbdbg()
        dbg.cmd_tlbflush("0x10000")


# ============================================================
#  Cache 命令
# ============================================================


class TestCmdCache:
    """cmd_cache — L2 缓存显示."""

    def test_no_l2_cache(self):
        """无 L2 缓存时显示提示."""
        dbg = _make_tlbdbg()
        dbg.cmd_cache()  # 应不抛异常

    def test_with_l2_cache(self):
        """有 L2 缓存时显示统计."""
        emu = Emulator(ram_size=0x100000, use_l2_cache=True, prog_cnt=0x1000)
        emu.load_code(0x1000, b"\x13\x00\x00\x00")
        dbg = _TestTlbDbg(emu)
        dbg.cmd_cache()

    def test_with_l2_cache_set_range(self):
        emu = Emulator(ram_size=0x100000, use_l2_cache=True, prog_cnt=0x1000)
        dbg = _TestTlbDbg(emu)
        dbg.cmd_cache("0-3")

    def test_with_l2_cache_single_set(self):
        emu = Emulator(ram_size=0x100000, use_l2_cache=True, prog_cnt=0x1000)
        dbg = _TestTlbDbg(emu)
        dbg.cmd_cache("0")

    def test_with_l2_cache_set_way(self):
        emu = Emulator(ram_size=0x100000, use_l2_cache=True, prog_cnt=0x1000)
        dbg = _TestTlbDbg(emu)
        dbg.cmd_cache("0 0")


# ============================================================
#  Cache 参数解析
# ============================================================


class TestParseCacheArgs:
    """_parse_cache_args 参数解析."""

    def test_none_args(self):
        dbg = _make_tlbdbg()
        err, *rest = dbg._parse_cache_args(None, 8, 4)
        assert not err
        assert rest == [None, None, None, None]

    def test_set_range(self):
        dbg = _make_tlbdbg()
        err, ts, tw, rs, re = dbg._parse_cache_args("2-5", 8, 4)
        assert not err
        assert ts is None
        assert tw is None
        assert rs == 2
        assert re == 5

    def test_single_set(self):
        dbg = _make_tlbdbg()
        err, ts, tw, rs, re = dbg._parse_cache_args("3", 8, 4)
        assert not err
        assert ts == 3
        assert tw is None

    def test_set_and_way(self):
        dbg = _make_tlbdbg()
        err, ts, tw, rs, re = dbg._parse_cache_args("3 1", 8, 4)
        assert not err
        assert ts == 3
        assert tw == 1

    def test_set_out_of_range(self):
        dbg = _make_tlbdbg()
        err, *_ = dbg._parse_cache_args("99", 8, 4)
        assert err

    def test_range_out_of_bounds(self):
        dbg = _make_tlbdbg()
        err, *_ = dbg._parse_cache_args("0-99", 8, 4)
        assert err

    def test_way_out_of_range(self):
        dbg = _make_tlbdbg()
        err, *_ = dbg._parse_cache_args("3 99", 8, 4)
        assert err
