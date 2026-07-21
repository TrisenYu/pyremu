#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""RISC-V PLIC (Platform-Level Interrupt Controller) 设备测试.

测试覆盖:
- MMIO 寄存器读写 (优先级、pending、enable、阈值、claim/complete)
- 中断仲裁 (最高优先级、阈值过滤)
- set_irq / get_pending_mip 硬件集成 API
- 多 context (hart) 中断独立性
"""

from __future__ import annotations

import pytest

from pyremu.interrupt.plic import (
    PLIC,
    PLIC_CONTEXT_BASE,
    PLIC_CONTEXT_STRIDE,
    PLIC_ENABLE_BASE,
    PLIC_ENABLE_STRIDE,
    PLIC_PENDING_BASE,
    PLIC_PRIORITY_BASE,
    PLIC_PRIORITY_STRIDE,
)

# ============================================================
#  辅助: MMIO 快捷方法
# ============================================================


def _read_u32(plic: PLIC, offset: int) -> int:
    return int.from_bytes(plic.read(offset, 4), "little")


def _write_u32(plic: PLIC, offset: int, val: int) -> None:
    plic.write(offset, val.to_bytes(4, "little"))


def _priority_offset(source: int) -> int:
    return PLIC_PRIORITY_BASE + source * PLIC_PRIORITY_STRIDE


def _pending_word_offset(word: int) -> int:
    return PLIC_PENDING_BASE + word * 4


def _enable_offset(context: int, word: int) -> int:
    return PLIC_ENABLE_BASE + context * PLIC_ENABLE_STRIDE + word * 4


def _context_threshold_offset(context: int) -> int:
    return PLIC_CONTEXT_BASE + context * PLIC_CONTEXT_STRIDE


def _context_claim_offset(context: int) -> int:
    return PLIC_CONTEXT_BASE + context * PLIC_CONTEXT_STRIDE + 4


# ============================================================
#  Fixtures
# ============================================================


@pytest.fixture
def plic() -> PLIC:
    """创建 128 源、4 context 的 PLIC."""
    return PLIC(num_sources=128, num_contexts=4)


# ============================================================
#  Tests: 优先级寄存器
# ============================================================


class TestPriority:
    def test_default_zero(self, plic):
        assert _read_u32(plic, _priority_offset(1)) == 0
        assert _read_u32(plic, _priority_offset(64)) == 0

    def test_write_read_back(self, plic):
        _write_u32(plic, _priority_offset(10), 5)
        assert _read_u32(plic, _priority_offset(10)) == 5

    def test_write_clamps_to_3_bits(self, plic):
        _write_u32(plic, _priority_offset(3), 0xFF)
        assert _read_u32(plic, _priority_offset(3)) == 7  # only bits [2:0]

    def test_source_zero_reserved(self, plic):
        """source 0 保留, 写不生效."""
        _write_u32(plic, _priority_offset(0), 5)
        assert _read_u32(plic, _priority_offset(0)) == 0

    def test_out_of_range_source_returns_zero(self, plic):
        assert _read_u32(plic, _priority_offset(200)) == 0


# ============================================================
#  Tests: Pending 位 (只读)
# ============================================================


class TestPending:
    def test_all_zero_on_start(self, plic):
        assert _read_u32(plic, _pending_word_offset(0)) == 0

    def test_set_irq_sets_pending(self, plic):
        plic.set_irq(5, True)
        pending = _read_u32(plic, _pending_word_offset(0))
        assert (pending >> 5) & 1 == 1

    def test_set_irq_clear(self, plic):
        plic.set_irq(5, True)
        plic.set_irq(5, False)
        pending = _read_u32(plic, _pending_word_offset(0))
        assert (pending >> 5) & 1 == 0

    def test_pending_is_read_only(self, plic):
        """写 pending 寄存器无效 (hardware 控制)."""
        plic.set_irq(3, True)
        _write_u32(plic, _pending_word_offset(0), 0)  # 尝试写清零
        assert _read_u32(plic, _pending_word_offset(0)) & 8 == 8  # source 3 仍 pending

    def test_claim_clears_pending(self, plic):
        """Claim 会清除 pending 位."""
        plic.set_irq(7, True)
        _write_u32(plic, _priority_offset(7), 3)
        _write_u32(plic, _enable_offset(0, 0), 1 << 7)

        claim = _read_u32(plic, _context_claim_offset(0))
        assert claim == 7
        # pending 已清除
        assert _read_u32(plic, _pending_word_offset(0)) & (1 << 7) == 0


# ============================================================
#  Tests: Enable 位
# ============================================================


class TestEnable:
    def test_default_disabled(self, plic):
        assert _read_u32(plic, _enable_offset(0, 0)) == 0

    def test_write_enable_bits(self, plic):
        _write_u32(plic, _enable_offset(0, 0), (1 << 3) | (1 << 15))
        assert _read_u32(plic, _enable_offset(0, 0)) & (1 << 3)
        assert _read_u32(plic, _enable_offset(0, 0)) & (1 << 15)

    def test_different_contexts_independent(self, plic):
        _write_u32(plic, _enable_offset(0, 0), 1 << 5)
        _write_u32(plic, _enable_offset(1, 0), 1 << 8)
        assert not (_read_u32(plic, _enable_offset(0, 0)) & (1 << 8))
        assert not (_read_u32(plic, _enable_offset(1, 0)) & (1 << 5))


# ============================================================
#  Tests: 阈值
# ============================================================


class TestThreshold:
    def test_default_zero(self, plic):
        assert _read_u32(plic, _context_threshold_offset(0)) == 0

    def test_write_threshold(self, plic):
        _write_u32(plic, _context_threshold_offset(0), 4)
        assert _read_u32(plic, _context_threshold_offset(0)) == 4

    def test_threshold_blocks_lower_priority(self, plic):
        """阈值 3 -> 优先级 ≤ 3 的中断不应触发."""
        _write_u32(plic, _context_threshold_offset(0), 3)
        _write_u32(plic, _priority_offset(10), 2)  # 低于阈值
        _write_u32(plic, _enable_offset(0, 0), 1 << 10)
        plic.set_irq(10, True)
        assert _read_u32(plic, _context_claim_offset(0)) == 0  # 不触发

    def test_threshold_allows_higher_priority(self, plic):
        """阈值 3 -> 优先级 > 3 的中断正常触发."""
        _write_u32(plic, _context_threshold_offset(0), 3)
        _write_u32(plic, _priority_offset(10), 5)  # 高于阈值
        _write_u32(plic, _enable_offset(0, 0), 1 << 10)
        plic.set_irq(10, True)
        assert _read_u32(plic, _context_claim_offset(0)) == 10


# ============================================================
#  Tests: 仲裁逻辑
# ============================================================


class TestArbitration:
    def test_highest_priority_wins(self, plic):
        _write_u32(plic, _priority_offset(3), 2)
        _write_u32(plic, _priority_offset(7), 5)
        _write_u32(plic, _priority_offset(12), 3)

        _write_u32(plic, _enable_offset(0, 0), (1 << 3) | (1 << 7) | (1 << 12))
        plic.set_irq(3, True)
        plic.set_irq(7, True)
        plic.set_irq(12, True)

        assert _read_u32(plic, _context_claim_offset(0)) == 7  # 最高优先级

    def test_same_priority_lowest_id_wins(self, plic):
        _write_u32(plic, _priority_offset(5), 3)
        _write_u32(plic, _priority_offset(3), 3)
        _write_u32(plic, _priority_offset(8), 3)

        _write_u32(plic, _enable_offset(0, 0), (1 << 5) | (1 << 3) | (1 << 8))
        plic.set_irq(5, True)
        plic.set_irq(3, True)
        plic.set_irq(8, True)

        assert _read_u32(plic, _context_claim_offset(0)) == 3  # 最小 ID

    def test_not_enabled_ignored(self, plic):
        _write_u32(plic, _priority_offset(5), 7)
        plic.set_irq(5, True)
        # 不设置 enable -> 不触发
        assert _read_u32(plic, _context_claim_offset(0)) == 0

    def test_not_pending_ignored(self, plic):
        _write_u32(plic, _priority_offset(5), 7)
        _write_u32(plic, _enable_offset(0, 0), 1 << 5)
        # 不设 pending -> 不触发
        assert _read_u32(plic, _context_claim_offset(0)) == 0

    def test_zero_priority_ignored(self, plic):
        """优先级 0 表示"永不中断". """
        _write_u32(plic, _priority_offset(5), 0)
        _write_u32(plic, _enable_offset(0, 0), 1 << 5)
        plic.set_irq(5, True)
        assert _read_u32(plic, _context_claim_offset(0)) == 0


# ============================================================
#  Tests: Complete
# ============================================================


class TestComplete:
    def test_complete_allows_re_trigger(self, plic):
        _write_u32(plic, _priority_offset(5), 3)
        _write_u32(plic, _enable_offset(0, 0), 1 << 5)
        plic.set_irq(5, True)

        claim = _read_u32(plic, _context_claim_offset(0))
        assert claim == 5

        # Complete — 写回 claim 得到的值
        _write_u32(plic, _context_claim_offset(0), 5)

        # 再次触发
        plic.set_irq(5, True)
        claim2 = _read_u32(plic, _context_claim_offset(0))
        assert claim2 == 5

    def test_complete_wrong_source_ignored(self, plic):
        """写回不匹配的 source id 无效."""
        _write_u32(plic, _priority_offset(5), 3)
        _write_u32(plic, _enable_offset(0, 0), 1 << 5)
        plic.set_irq(5, True)
        _read_u32(plic, _context_claim_offset(0))  # claim source 5

        # 尝试 complete source 3 (不匹配)
        _write_u32(plic, _context_claim_offset(0), 3)

        # 再次触发 source 5 应该可行 (complete 未生效, claimed 仍为 5)
        plic.set_irq(5, True)
        claim2 = _read_u32(plic, _context_claim_offset(0))
        # claimed[0] 仍是 5, 第二次 claim 会返回谁?
        # 实际上 _claimed 还是 5, 新的 pending 不会被 claim
        # 直到 complete 原 source
        assert claim2 == 0  # 新 pending 被跳过, 因为 old claim 没完成


# ============================================================
#  Tests: Hardware Integration API
# ============================================================


class TestHardwareAPI:
    def test_get_pending_mip_no_interrupt(self, plic):
        assert plic.get_pending_mip(0) == 0

    def test_get_pending_mip_with_interrupt(self, plic):
        _write_u32(plic, _priority_offset(10), 3)
        _write_u32(plic, _enable_offset(0, 0), 1 << 10)
        plic.set_irq(10, True)
        assert plic.get_pending_mip(0) == (1 << 11)  # MEIP

    def test_get_pending_mip_threshold_blocks(self, plic):
        _write_u32(plic, _context_threshold_offset(0), 7)
        _write_u32(plic, _priority_offset(10), 3)
        _write_u32(plic, _enable_offset(0, 0), 1 << 10)
        plic.set_irq(10, True)
        assert plic.get_pending_mip(0) == 0  # 优先级低于阈值

    def test_get_pending_mip_different_contexts(self, plic):
        _write_u32(plic, _priority_offset(10), 3)
        _write_u32(plic, _enable_offset(0, 0), 1 << 10)
        plic.set_irq(10, True)
        assert plic.get_pending_mip(0) == (1 << 11)
        # context 1 未使能 source 10
        assert plic.get_pending_mip(1) == 0

    def test_get_pending_mip_out_of_range_context(self, plic):
        assert plic.get_pending_mip(999) == 0


# ============================================================
#  Tests: 标准双 context/hart (2h=M-context/MEIP, 2h+1=S-context/SEIP)
# ============================================================


class TestDualContext:
    """每 hart 两个 context: 偶=M(MEIP bit11), 奇=S(SEIP bit9)。

    fixture num_contexts=4 -> hart0: ctx0(M)/ctx1(S); hart1: ctx2(M)/ctx3(S)。
    """

    def test_s_context_returns_seip(self, plic):
        # 源使能于 hart0 的 S-context (ctx 1) -> SEIP, 而非 MEIP
        _write_u32(plic, _priority_offset(10), 3)
        _write_u32(plic, _enable_offset(1, 0), 1 << 10)
        plic.set_irq(10, True)
        assert plic.get_pending_mip(0) == (1 << 9)  # SEIP

    def test_m_context_returns_meip(self, plic):
        _write_u32(plic, _priority_offset(10), 3)
        _write_u32(plic, _enable_offset(0, 0), 1 << 10)
        plic.set_irq(10, True)
        assert plic.get_pending_mip(0) == (1 << 11)  # MEIP

    def test_both_contexts_set_both_bits(self, plic):
        _write_u32(plic, _priority_offset(10), 3)
        _write_u32(plic, _enable_offset(0, 0), 1 << 10)  # hart0 M
        _write_u32(plic, _enable_offset(1, 0), 1 << 10)  # hart0 S
        plic.set_irq(10, True)
        assert plic.get_pending_mip(0) == ((1 << 11) | (1 << 9))

    def test_hart1_s_context_isolated(self, plic):
        # hart1 的 S-context = ctx 3; hart0 未使能 -> 仅 hart1 得 SEIP
        _write_u32(plic, _priority_offset(7), 2)
        _write_u32(plic, _enable_offset(3, 0), 1 << 7)
        plic.set_irq(7, True)
        assert plic.get_pending_mip(1) == (1 << 9)  # SEIP
        assert plic.get_pending_mip(0) == 0


# ============================================================
#  Tests: 多源竞争
# ============================================================


class TestMultiSource:
    def test_claim_takes_only_highest(self, plic):
        """Claim 只取最高优先级, 其他 pending 保留."""
        for s in [3, 5, 8]:
            _write_u32(plic, _priority_offset(s), min(s, 7))
            plic.set_irq(s, True)
        _write_u32(plic, _enable_offset(0, 0), (1 << 3) | (1 << 5) | (1 << 8))

        # 最高优先级的 source 被 claim (priority 3/5/7 -> source 8 wins)
        assert _read_u32(plic, _context_claim_offset(0)) == 8
        # 检查 source 5 和 3 的 pending 仍保留
        pending0 = _read_u32(plic, _pending_word_offset(0))
        assert (pending0 >> 3) & 1 == 1
        assert (pending0 >> 5) & 1 == 1
        assert (pending0 >> 8) & 1 == 0  # claimed


# ============================================================
#  Tests: 电平中断 gateway — complete 重挂起
# ============================================================


class TestLevelRetrigger:
    """complete 时设备电平仍高 → pending 重新置位 (QEMU sifive_plic gateway 语义).

    回归背景: UART TX watermark 为电平中断; sifive 驱动 ISR 每次仅发送
    FIFO 深度 (8) 个字符, 期间 TXDATA 写由 Rust inline 处理, 不再有任何
    Python 侧设备访问调用 set_irq。旧行为 claim 清 pending 后无人重新拉线
    → complete 后中断永久丢失, 剩余 TX 数据滞留内核环形缓冲。
    """

    def test_complete_reraises_when_level_still_high(self, plic):
        plic.set_irq(7, True)
        _write_u32(plic, _priority_offset(7), 3)
        _write_u32(plic, _enable_offset(0, 0), 1 << 7)
        assert _read_u32(plic, _context_claim_offset(0)) == 7
        assert _read_u32(plic, _pending_word_offset(0)) & (1 << 7) == 0
        # 设备未拉低电平 (还有待发送数据), complete 必须重新挂起
        _write_u32(plic, _context_claim_offset(0), 7)
        assert _read_u32(plic, _pending_word_offset(0)) & (1 << 7) != 0

    def test_complete_idle_when_level_lowered(self, plic):
        plic.set_irq(7, True)
        _write_u32(plic, _priority_offset(7), 3)
        _write_u32(plic, _enable_offset(0, 0), 1 << 7)
        assert _read_u32(plic, _context_claim_offset(0)) == 7
        plic.set_irq(7, False)  # ISR 内设备已拉低 (如 IE 关闭 / RX 读空)
        _write_u32(plic, _context_claim_offset(0), 7)
        assert _read_u32(plic, _pending_word_offset(0)) & (1 << 7) == 0
