#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""缓存抽象基类 (CacheBase) 测试: 通过 TLB 验证 CacheBase 基础功能."""

from pyremu.memory.cache_base import CacheBase, CacheLineBase, ReplacementPolicy
from pyremu.memory.tlb import TLB, TLBLine


class TestCacheBaseThroughTLB:
    """通过 TLB 测试 CacheBase 的基础框架."""

    def test_tlb_inherits_cachebase(self):
        """TLB 应是 CacheBase 的子类."""
        tlb = TLB(size=8)
        assert isinstance(tlb, CacheBase)

    def test_name_and_size(self):
        """名称和容量应正确."""
        tlb = TLB(size=16)
        assert tlb.num_entries == 16
        assert tlb.name == "TLB"

    def test_initial_empty(self):
        """新缓存应为空."""
        tlb = TLB(size=8)
        assert len(tlb) == 0

    def test_hit_rate_initial_zero(self):
        """无访问时命中率为 0."""
        tlb = TLB(size=8)
        assert tlb.hit_rate == 0.0

    def test_repr(self):
        """__repr__ 应包含名称和容量信息."""
        tlb = TLB(size=8)
        r = repr(tlb)
        assert "TLB" in r
        assert "8" in r

    def test_len_reflects_valid_entries(self):
        """__len__ 应返回有效条目数."""
        tlb = TLB(size=8)
        assert len(tlb) == 0
        tlb.insert(vpn=0x100, ppn=0x200, perm=0xF)
        assert len(tlb) == 1
        tlb.insert(vpn=0x101, ppn=0x201, perm=0xF)
        assert len(tlb) == 2
        tlb.flush(vpn=0x100)
        assert len(tlb) == 1


class TestCacheBaseReplacement:
    """替换策略测试."""

    def test_fifo_policy(self):
        """FIFO 策略: 填满后逐出最早插入的."""
        tlb = TLB(size=2, policy=ReplacementPolicy.FIFO)
        tlb.insert(vpn=0x100, ppn=0x200, perm=0xF)
        tlb.insert(vpn=0x101, ppn=0x201, perm=0xF)
        tlb.insert(vpn=0x102, ppn=0x202, perm=0xF)
        # 0x100 应被逐出, 0x101 和 0x102 保留
        hit, _, _ = tlb.lookup(0x100)
        assert not hit
        hit, _, _ = tlb.lookup(0x101)
        assert hit
        hit, _, _ = tlb.lookup(0x102)
        assert hit

    def test_lru_policy(self):
        """LRU 策略: 填满后逐出最久未用的."""
        tlb = TLB(size=2, policy=ReplacementPolicy.LRU)
        tlb.insert(vpn=0x100, ppn=0x200, perm=0xF)
        tlb.insert(vpn=0x101, ppn=0x201, perm=0xF)
        # 访问 vpn=0x100 使其成为最近使用的
        tlb.lookup(0x100)
        # 插入第三个, 应逐出 0x101 (最久未用)
        tlb.insert(vpn=0x102, ppn=0x202, perm=0xF)
        hit, _, _ = tlb.lookup(0x100)
        assert hit, "最近使用的 0x100 应保留"
        hit, _, _ = tlb.lookup(0x101)
        assert not hit, "最久未用的 0x101 应被逐出"
        hit, _, _ = tlb.lookup(0x102)
        assert hit


class TestCacheLineBase:
    """CacheLineBase 数据类基本功能."""

    def test_default_values(self):
        line = CacheLineBase()
        assert line.tag == 0
        assert line.valid is False
        assert line.dirty is False
        assert line.last_access == 0

    def test_tlb_line_extends_base(self):
        """TLBLine 应是 CacheLineBase 的子类."""
        line = TLBLine()
        assert isinstance(line, CacheLineBase)
        assert line.ppn == 0
        assert line.perm == 0
        assert line.level == 0
