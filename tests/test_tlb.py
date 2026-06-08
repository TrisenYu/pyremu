#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""TLB 测试: 查找、插入、FIFO 替换、刷新."""

import pytest

from pyremu.memory.tlb import TLB


class TestTLBBasic:
    """TLB 基本操作."""

    @pytest.fixture
    def tlb(self) -> TLB:
        return TLB(size=8)

    def test_initial_empty(self, tlb):
        """新创建的 TLB 应为空."""
        assert len(tlb) == 0
        assert tlb.size == 8

    def test_insert_and_lookup(self, tlb):
        """插入后应能命中."""
        tlb.insert(vpn=0x100, ppn=0x200, perm=0xF)
        hit, ppn, perm = tlb.lookup(0x100)
        assert hit is True
        assert ppn == 0x200 and perm == 0xF

    def test_lookup_miss(self, tlb):
        """未插入的 VPN 应未命中."""
        hit, ppn, perm = tlb.lookup(0x999)
        assert hit is False
        assert ppn == 0 and perm == 0

    def test_update_existing(self, tlb):
        """插入已存在的 VPN 应原地更新, 不占用新槽位."""
        tlb.insert(vpn=0x100, ppn=0x200, perm=0xF)
        tlb.insert(vpn=0x100, ppn=0x300, perm=0x3)
        assert len(tlb) == 1, "更新已存在条目不应增加有效条目数"
        hit, ppn, perm = tlb.lookup(0x100)
        assert ppn == 0x300 and perm == 0x3

    def test_multiple_entries(self, tlb):
        """多个不同 VPN 应能全部命中."""
        mappings = [(0x100, 0xA00), (0x101, 0xA01), (0x102, 0xA02)]
        for vpn, ppn in mappings:
            tlb.insert(vpn=vpn, ppn=ppn, perm=0xF)
        assert len(tlb) == 3
        for vpn, ppn in mappings:
            hit, result_ppn, _ = tlb.lookup(vpn)
            assert hit and result_ppn == ppn


class TestTLBFIFO:
    """FIFO 替换策略."""

    def test_fifo_eviction(self):
        """当 TLB 满时, FIFO 策略应逐出最早插入的条目."""
        tlb = TLB(size=2)

        # 填满 TLB
        tlb.insert(vpn=0x100, ppn=0x200, perm=0xF)  # slot 0 (oldest)
        tlb.insert(vpn=0x101, ppn=0x201, perm=0xF)  # slot 1
        assert len(tlb) == 2

        # 插入第三个, 应逐出 slot 0 (vpn=0x100)
        tlb.insert(vpn=0x102, ppn=0x202, perm=0xF)

        hit, _, _ = tlb.lookup(0x100)
        assert hit is False, "slot 0 (vpn=0x100) 应被逐出"

        hit, ppn, _ = tlb.lookup(0x101)
        assert hit is True and ppn == 0x201, "slot 1 应保留"

        hit, ppn, _ = tlb.lookup(0x102)
        assert hit is True and ppn == 0x202, "新条目应存在"

    def test_fifo_wraps_around(self):
        """FIFO 指针应循环."""
        tlb = TLB(size=2)
        # 插入 4 次, 验证 FIFO 指针正确回绕
        for i in range(4):
            tlb.insert(vpn=0x100 + i, ppn=0x200 + i, perm=0xF)

        # slot 0 被逐出两次 (i=0 被 i=2 替换)
        hit, _, _ = tlb.lookup(0x100)
        assert hit is False
        hit, ppn, _ = tlb.lookup(0x102)
        assert hit is True and ppn == 0x202

        # slot 1 被逐出两次 (i=1 被 i=3 替换)
        hit, _, _ = tlb.lookup(0x101)
        assert hit is False
        hit, ppn, _ = tlb.lookup(0x103)
        assert hit is True and ppn == 0x203


class TestTLBFlush:
    """TLB 刷新."""

    @pytest.fixture
    def tlb(self) -> TLB:
        tlb = TLB(size=8)
        tlb.insert(vpn=0x100, ppn=0x200, perm=0xF)
        tlb.insert(vpn=0x101, ppn=0x201, perm=0xF)
        tlb.insert(vpn=0x102, ppn=0x202, perm=0xF)
        return tlb

    def test_flush_single_vpn(self, tlb):
        """按 VPN 刷新只清除匹配的条目."""
        tlb.flush(vpn=0x101)
        assert len(tlb) == 2
        hit, _, _ = tlb.lookup(0x101)
        assert hit is False
        # 其他条目不受影响
        hit, _, _ = tlb.lookup(0x100)
        assert hit is True

    def test_flush_all_by_vpn_zero(self, tlb):
        """vpn=0 时刷新全部条目."""
        tlb.flush(vpn=0)
        assert len(tlb) == 0
        for v in (0x100, 0x101, 0x102):
            hit, _, _ = tlb.lookup(v)
            assert hit is False

    def test_flush_all_method(self, tlb):
        """flush_all() 应刷新全部条目."""
        tlb.flush_all()
        assert len(tlb) == 0

    def test_flush_nonexistent_vpn(self, tlb):
        """刷新不存在的 VPN 不应影响其他条目."""
        tlb.flush(vpn=0x999)
        assert len(tlb) == 3


class TestTLBEdgeCases:
    """边界情况."""

    def test_zero_vpn(self):
        """VPN=0 是合法的."""
        tlb = TLB(size=4)
        tlb.insert(vpn=0, ppn=0x1000, perm=0xF)
        hit, ppn, _ = tlb.lookup(0)
        assert hit and ppn == 0x1000

    def test_large_vpn(self):
        """较大的 VPN 值."""
        tlb = TLB(size=4)
        tlb.insert(vpn=0xFFFF_FFFF, ppn=0xDEAD, perm=0xF)
        hit, ppn, _ = tlb.lookup(0xFFFF_FFFF)
        assert hit and ppn == 0xDEAD

    def test_default_size(self):
        """默认构造时应使用 size=256."""
        tlb = TLB()
        assert tlb.size == 256

    def test_level_parameter(self):
        """level 参数应正确保存."""
        tlb = TLB(size=4)
        tlb.insert(vpn=0x100, ppn=0x200, perm=0xF, level=2)
        hit, ppn, perm = tlb.lookup(0x100)
        assert hit
        # level 不通过 lookup 返回, 但应不影响基本功能
