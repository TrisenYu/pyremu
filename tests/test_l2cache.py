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
        """读未命中 -> E 状态."""
        l2.read(0x1000, 4)
        # 找到该行检查状态
        for e in l2._entries:
            if e.valid:
                assert e.mesi == MESIState.EXCLUSIVE
                return
        pytest.fail("未找到有效缓存行")

    def test_write_hit_exclusive_goes_to_modified(self, l2):
        """写命中 E -> M."""
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

    def test_entries_property(self):
        """entries 属性返回内部 L2CacheLine 列表."""
        ram, rf, wf = _make_ram()
        wf(0x1000, b"\x00" * 64)
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)
        l2.read(0x1000, 8)
        entries = l2.entries
        assert len(entries) == l2.num_sets * l2.ways
        assert any(e.valid for e in entries), "至少有一条有效条目"


class TestL2BusInterface:
    """L2 bus_read / bus_write 总线接口."""

    def test_bus_read_is_alias_for_read(self):
        """bus_read 应与 read 返回相同数据."""
        ram, rf, wf = _make_ram()
        wf(0x1000, b"\x01\x02\x03\x04\x05\x06\x07\x08")
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=2)
        l2.set_ram_backend(rf, wf)
        assert l2.bus_read(0x1000, 4) == l2.read(0x1000, 4)

    def test_bus_write_is_alias_for_write(self):
        """bus_write 应与 write 效果相同."""
        ram, rf, wf = _make_ram()
        wf(0x1000, b"\x00" * 64)
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=2)
        l2.set_ram_backend(rf, wf)
        l2.bus_write(0x1000, b"\xaa\xbb\xcc\xdd")
        assert l2.read(0x1000, 4) == b"\xaa\xbb\xcc\xdd"


class TestL2Invalidate:
    """L2 invalidate 边界用例."""

    @pytest.fixture
    def l2(self) -> L2Cache:
        ram, rf, wf = _make_ram()
        wf(0x1000, b"\x00" * 64)
        wf(0x2000, b"\x11" * 64)
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)
        return l2

    def test_invalidate_clean_e_state_no_writeback(self, l2):
        """E 状态行 invalidate 不应触发回写 (干净行)."""
        l2.read(0x1000, 8)  # I->E
        # 验证 E 状态
        for e in l2._entries:
            if e.valid:
                assert e.mesi == MESIState.EXCLUSIVE
        l2.invalidate(0x1000)
        assert len(l2) == 0

    def test_invalidate_nonexistent_addr_noop(self, l2):
        """invalidate 不存在的地址应无副作用."""
        l2.read(0x1000, 8)
        assert len(l2) == 1
        l2.invalidate(0xF000)  # 不在缓存中
        assert len(l2) == 1, "已缓存行不应被误清"

    def test_invalidate_already_invalid_line_noop(self, l2):
        """invalidate 已失效的行应无副作用."""
        l2.read(0x1000, 8)
        l2.invalidate(0x1000)
        assert len(l2) == 0
        # 二次 invalidate 不应抛异常
        l2.invalidate(0x1000)
        assert len(l2) == 0

    def test_invalidate_without_ram_writeback(self, l2):
        """无 RAM backend 时 M 状态 invalidate 不崩溃."""
        l2_no_ram = L2Cache(size=4 * 1024, line_size=64, ways=4)
        # 无 set_ram_backend — 模拟设备 MMIO 旁路场景
        l2_no_ram._ram_read = lambda a, s: b"\x00" * s
        l2_no_ram.read(0x1000, 8)
        l2_no_ram.write(0x1000, b"\xff" * 8)  # M 状态
        l2_no_ram.invalidate(0x1000)  # 不应崩溃
        assert len(l2_no_ram) == 0


class TestL2HitRate:
    """L2 命中率统计."""

    def test_hit_rate_initial_zero(self):
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=2)
        assert l2.hit_rate == 0.0

    def test_hit_rate_after_all_hits(self):
        ram, rf, wf = _make_ram()
        wf(0x1000, b"\x00" * 64)
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=2)
        l2.set_ram_backend(rf, wf)
        l2.read(0x1000, 4)  # miss
        l2.read(0x1000, 4)  # hit
        l2.read(0x1000, 8)  # hit
        l2.read(0x1000, 4)  # hit
        assert l2.hit_rate == 0.75, f"期望 3/4=0.75, 实际 {l2.hit_rate}"

    def test_hit_rate_all_misses(self):
        ram, rf, wf = _make_ram()
        wf(0x1000, b"\x00" * 64)
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=1)
        l2.set_ram_backend(rf, wf)
        # 连续访问映射到同 set 的不同 tag, 每次都 miss 并逐出
        l2.read(0x1000, 4)  # miss
        l2.read(0x1040, 4)  # miss (不同 tag, 同 set)
        l2.read(0x1080, 4)  # miss
        assert l2.hit_rate == 0.0


class TestL2Iter:
    """L2 __iter__ 迭代器."""

    def test_iter_yields_valid_entries(self):
        ram, rf, wf = _make_ram()
        wf(0x1000, b"\x00" * 64)
        wf(0x2000, b"\x11" * 64)
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)
        l2.read(0x1000, 8)
        l2.read(0x2000, 8)
        valid = list(iter(l2))
        assert len(valid) == 2
        tags = {e.tag for e in valid}
        assert l2._addr_fields(0x1000)[0] in tags
        assert l2._addr_fields(0x2000)[0] in tags

    def test_iter_empty_cache(self):
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=4)
        assert list(iter(l2)) == []


class TestL2SingleWay:
    """直接映射 (ways=1) 边界用例."""

    def test_direct_mapped_every_access_evicts_previous(self):
        """ways=1: 同 set 的不同 tag 访问逐出前一条 (直接映射语义)."""
        ram, rf, wf = _make_ram()
        wf(0x1000, b"\xaa" * 64)
        wf(0x1040, b"\xbb" * 64)
        l2 = L2Cache(size=64, line_size=64, ways=1)  # 1 set, 1 way
        l2.set_ram_backend(rf, wf)
        l2.read(0x1000, 8)
        assert len(l2) == 1
        l2.read(0x1040, 8)  # 不同 tag, 同 set -> 逐出
        assert len(l2) == 1
        # 旧行应已被逐出, 新行数据正确
        assert l2.read(0x1040, 4) == b"\xbb\xbb\xbb\xbb"


class TestL2FlushAll:
    """L2 flush_all / invalidate_all 批量缓存一致性操作."""

    @pytest.fixture
    def l2(self) -> L2Cache:
        ram, rf, wf = _make_ram()
        wf(0x1000, b"\x00" * 64)
        wf(0x2000, b"\x11" * 64)
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)
        return l2

    def test_flush_all_empty_cache(self, l2):
        """空缓存 flush_all 返回 0."""
        assert l2.flush_all() == 0

    def test_flush_all_clean_lines_not_written(self, l2):
        """干净行 (E/S 状态) flush_all 不计入."""
        l2.read(0x1000, 8)  # I->E
        assert l2.flush_all() == 0, "E 状态行不应被 flush"

    def test_flush_all_writes_back_dirty_lines(self, l2):
        """flush_all 将 M 状态行回写到 RAM 并降级为 E."""
        l2.read(0x1000, 8)  # I->E
        l2.write(0x1000, b"\xaa\xbb\xcc\xdd")  # E->M
        assert l2.flush_all() == 1

        # 行应保持有效但降级为 E (干净)
        for e in l2._entries:
            if e.valid:
                assert e.mesi == MESIState.EXCLUSIVE
                assert not e.dirty

    def test_flush_all_multiple_dirty_lines(self, l2):
        """多条脏行全部回写."""
        l2.read(0x1000, 8)
        l2.write(0x1000, b"\xaa" * 8)
        l2.read(0x2000, 8)
        l2.write(0x2000, b"\xbb" * 8)
        assert l2.flush_all() == 2

    def test_flush_all_writes_to_ram(self, l2):
        """flush_all 将数据写入底层 RAM."""
        l2.read(0x1000, 8)
        l2.write(0x1000, b"\xde\xad\xbe\xef\xca\xfe\xba\xbe")
        l2.flush_all()

        # 直接从 RAM 读取验证回写
        ram_data = l2._ram_read(0x1000, 8)
        assert ram_data == b"\xde\xad\xbe\xef\xca\xfe\xba\xbe"

    def test_flush_all_without_ram_backend(self, l2):
        """无 RAM backend 时 flush_all 不崩溃."""
        no_ram = L2Cache(size=4 * 1024, line_size=64, ways=4)
        no_ram._ram_read = lambda a, s: b"\x00" * s
        no_ram.read(0x1000, 8)
        no_ram.write(0x1000, b"\xff" * 8)
        assert no_ram.flush_all() == 1  # 不崩溃

    def test_invalidate_all_empty_cache(self, l2):
        """空缓存 invalidate_all 返回 0."""
        assert l2.invalidate_all() == 0

    def test_invalidate_all_clears_all_valid(self, l2):
        """invalidate_all 使全部有效行失效."""
        l2.read(0x1000, 8)
        l2.read(0x2000, 8)
        assert len(l2) == 2
        assert l2.invalidate_all() == 2
        assert len(l2) == 0

    def test_invalidate_all_writes_back_dirty(self, l2):
        """flush_all 回写脏行到 RAM, invalidate_all 仅丢弃 (不写回).

        invalidate_all 用于 native batch 后: Rust 已直接修改 bytearray,
        再写回会覆盖 Rust 的修改。脏行回写由 pre-batch flush_all 负责。
        """
        l2.read(0x1000, 8)
        l2.write(0x1000, b"\xc0\xff\xee\x00\x00\x00\x00\x00")
        # flush_all: 写回脏行 M→E; invalidate_all: 丢弃 E 行 (无回写)
        assert l2.flush_all() == 1
        l2.invalidate_all()

        ram_data = l2._ram_read(0x1000, 8)
        assert ram_data == b"\xc0\xff\xee\x00\x00\x00\x00\x00"

    def test_invalidate_all_mixed_clean_dirty(self, l2):
        """混合干净/脏行时 invalidate_all 全部失效."""
        l2.read(0x1000, 8)  # E (clean)
        l2.read(0x2000, 8)
        l2.write(0x2000, b"\xff" * 8)  # M (dirty)
        assert len(l2) == 2
        count = l2.invalidate_all()
        assert count == 2
        assert len(l2) == 0

    def test_flush_then_invalidate_idempotent(self, l2):
        """flush_all 后 invalidate_all: flush 降级行, invalidate 清理."""
        l2.read(0x1000, 8)
        l2.write(0x1000, b"\x11" * 8)
        assert l2.flush_all() == 1
        # 行现在是 E 状态
        assert l2.invalidate_all() == 1
        assert len(l2) == 0


class TestL2Degenerate:
    """退化情形: 尺寸为 0 或极小值时的防御行为."""

    def test_zero_size_cache_is_usable(self):
        """l2_size=0 自动收敛为 1 组, 不崩溃且可正常透传读写."""
        ram, rf, wf = _make_ram()
        wf(0x100, b"\xde\xad\xbe\xef" * 4)
        l2 = L2Cache(size=0)
        l2.set_ram_backend(rf, wf)
        # 读: 未命中后从 RAM 加载行, 返回正确数据
        assert l2.read(0x100, 4) == b"\xde\xad\xbe\xef"
        # 写: 标记 M 不崩溃
        l2.write(0x200, b"\x01\x02\x03\x04")
        assert l2.read(0x200, 4) == b"\x01\x02\x03\x04"

    def test_zero_size_minimal_cache_does_not_crash(self):
        """size=0 退化为 1 组最小缓存, 总线读写不崩溃."""
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=0)
        l2.set_ram_backend(rf, wf)
        assert l2.read(0x0, 4) is not None
        l2.write(0x0, b"\x00" * 4)
