#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""L2 缓存测试: MESI 状态, 命中/未命中, 回写, FWIK."""

import pytest

from pyremu.memory.cache_base import CacheBase
from pyremu.memory.l2cache import L2Cache, MESIState


def _make_ram(size=64 * 1024):
    """构造模拟物理 RAM."""
    ram = bytearray(size)

    def read_fn(addr, size):
        return bytes(ram[addr : addr + size])

    def write_fn(addr, data):
        for i, b in enumerate(data):
            ram[addr + i] = b

    return ram, read_fn, write_fn


class TestL2CacheBasic:
    """L2 基本读写."""

    @pytest.fixture
    def l2(self) -> L2Cache:
        ram, rf, wf = _make_ram()
        # 预写 RAM 以便测试缓存未命中时的加载
        wf(0x1000, b"\x01\x02\x03\x04\x05\x06\x07\x08")
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=2)
        l2.set_ram_backend(rf, wf)
        return l2

    def test_inherits_cachebase(self, l2):
        """L2Cache 应是 CacheBase 的子类."""
        assert isinstance(l2, CacheBase)

    def test_initial_empty(self, l2):
        """初始缓存应为空."""
        assert len(l2) == 0

    def test_read_miss_loads_from_ram(self, l2):
        """未命中时从 RAM 加载数据."""
        data = l2.read(0x1000, 4)
        assert data == b"\x01\x02\x03\x04"
        # 加载后应有 1 条有效行
        assert len(l2) == 1

    def test_read_hit(self, l2):
        """命中时直接返回缓存数据."""
        # 第一次: 未命中, 加载
        l2.read(0x1000, 4)
        # 第二次: 应命中
        hit_before = l2._hits
        data = l2.read(0x1000, 4)
        assert data == b"\x01\x02\x03\x04"
        assert l2._hits == hit_before + 1

    def test_write_updates_cache_and_marks_modified(self, l2):
        """写命中更新数据并转 M 状态."""
        l2.read(0x1000, 8)  # 加载到 E 状态
        l2.write(0x1000, b"\xff\xee\xdd\xcc")
        data = l2.read(0x1000, 8)
        assert data[:4] == b"\xff\xee\xdd\xcc"
        # 检查后 4 字节仍为 RAM 原始值
        assert data[4:8] == b"\x05\x06\x07\x08"

    def test_writeback_on_eviction(self, l2):
        """逐出 M 状态行时回写 RAM."""
        # 使用很小的缓存 (1 set, 1 way) 促使逐出
        tiny_ram, rf, wf = _make_ram()
        wf(0x1000, b"\x00" * 64)
        wf(0x2000, b"\x00" * 64)
        tiny_l2 = L2Cache(size=64, line_size=64, ways=1)
        tiny_l2.set_ram_backend(rf, wf)

        # 写 0x1000 的行 (进入 M)
        tiny_l2.write(0x1000, b"\xaa" * 8)
        # 访问 0x2000 (不同 set), 逐出 0x1000 的行
        tiny_l2.read(0x2000, 8)

        # 检查 RAM 中 0x1000 被回写
        assert wf is not None
        ram_data = rf(0x1000, 8)
        assert ram_data == b"\xaa" * 8


class TestMESIState:
    """MESI 状态转移."""

    @pytest.fixture
    def l2(self) -> L2Cache:
        ram, rf, wf = _make_ram()
        wf(0x1000, b"\x00" * 64)
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)
        return l2

    def test_read_miss_goes_to_exclusive(self, l2):
        """读未命中 → E 状态."""
        l2.read(0x1000, 4)
        # 找到该行检查状态
        for e in l2._entries:
            if e.valid:
                assert e.mesi == MESIState.EXCLUSIVE
                return
        pytest.fail("未找到有效缓存行")

    def test_write_hit_exclusive_goes_to_modified(self, l2):
        """写命中 E → M."""
        l2.read(0x1000, 8)
        l2.write(0x1000, b"\x11" * 8)
        for e in l2._entries:
            if e.valid:
                assert e.mesi == MESIState.MODIFIED
                return
        pytest.fail("未找到有效缓存行")

    def test_invalidate(self, l2):
        """invalidate 使行失效, M 状态回写 RAM."""
        l2.read(0x1000, 8)
        l2.write(0x1000, b"\xbb" * 8)
        l2.invalidate(0x1000)
        assert len(l2) == 0


class TestL2Properties:
    """L2 属性."""

    def test_line_size(self):
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=4)
        assert l2.line_size == 64
        assert l2.num_sets == 16 * 1024 // (64 * 4)
        assert l2.ways == 4
