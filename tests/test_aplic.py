#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""APLIC (Advanced Platform Level Interrupt Controller) 单元测试 — AIA MSI 模式.

寄存器布局与语义对齐 RISC-V AIA 规范 §4 与 QEMU ``riscv_aplic.c`` /
Linux ``irq-riscv-aplic-msi.c``。测试通过 MMIO 配置 sourcecfg/target/setie,
再经 ``set_irq`` 模拟外设中断线, 验证 MSI 投递到 IMSIC S-file。
"""

import pytest

from pyremu.interrupt.aplic import APLIC
from pyremu.interrupt.imsic import IMSIC

# ============================================================
#  AIA 寄存器偏移 (与 aplic.py 常量一致)
# ============================================================

_DOMAINCFG = 0x0000
_SOURCECFG_BASE = 0x0004
_SETIP_BASE = 0x1C00
_SETIPNUM = 0x1CDC
_CLRIP_BASE = 0x1D00
_SETIE_BASE = 0x1E00
_SETIENUM = 0x1EDC
_CLRIE_BASE = 0x1F00
_SETIPNUM_LE = 0x2000
_TARGET_BASE = 0x3004

# sourcecfg SM 触发类型
_SM_INACTIVE = 0x0
_SM_EDGE_RISE = 0x4
_SM_LEVEL_HIGH = 0x6

# target 字段
_HART_SHIFT = 18
_EIID_MASK = 0x7FF


def _make_aplic(num_harts: int = 2, num_sources: int = 64) -> APLIC:
    imsic = IMSIC(num_harts=num_harts, m_base_addr=0x2400_0000)
    return APLIC(imsic=imsic, base_addr=0x0C00_0000, num_sources=num_sources)


def _w(dev: APLIC, offset: int, val: int) -> None:
    dev.write(offset, val.to_bytes(4, "little"))


def _r(dev: APLIC, offset: int) -> int:
    return int.from_bytes(dev.read(offset, 4), "little")


def _configure(a: APLIC, src: int, hart: int, eiid: int, *, level: bool) -> None:
    """通过 MMIO 配置一个源 — 模拟内核 aplic_msi_write_msg + set_type 流程."""
    # target[src] = (hart << 18) | eiid
    _w(a, _TARGET_BASE + (src - 1) * 4, (hart << _HART_SHIFT) | eiid)
    # sourcecfg[src] = SM 触发类型
    _w(a, _SOURCECFG_BASE + (src - 1) * 4, _SM_LEVEL_HIGH if level else _SM_EDGE_RISE)
    # 使能源 (setienum)
    _w(a, _SETIENUM, src)


# ============================================================
#  MMIO 寄存器偏移 (AIA 规范)
# ============================================================


class TestAplicMMIO:
    """APLIC MMIO 寄存器读写 — 偏移与 AIA 规范对齐."""

    def test_domaincfg_rw(self):
        """domaincfg IE 位往返."""
        a = _make_aplic()
        _w(a, _DOMAINCFG, 1 << 8)
        assert _r(a, _DOMAINCFG) == (1 << 8), "domaincfg IE bit round-trip"

    def test_sourcecfg_rw(self):
        """sourcecfg[1] 写入 SM_LEVEL_HIGH 往返."""
        a = _make_aplic()
        _w(a, _SOURCECFG_BASE, _SM_LEVEL_HIGH)
        assert a._sourcecfg[1] == _SM_LEVEL_HIGH

    def test_target_rw(self):
        """target[1] 写入 hart + eiid 往返."""
        a = _make_aplic()
        val = (3 << _HART_SHIFT) | 0x2A
        _w(a, _TARGET_BASE, val)
        assert a._target[1] == val

    def test_setip_bitmap(self):
        """setip 位图对活动源设置 pending."""
        a = _make_aplic()
        # 边沿触发源无输入电平要求, setip 位图直接置 pending
        _w(a, _SOURCECFG_BASE + 2 * 4, _SM_EDGE_RISE)  # source 3
        _w(a, _SETIP_BASE, 1 << 3)
        assert a._state[3] & 1, "source 3 pending bit should be set"


# ============================================================
#  MSI 投递路由
# ============================================================


class TestAplicRouting:
    """sourcecfg + target → set_irq → IMSIC S-file 投递."""

    def test_edge_trigger_delivers_seip(self):
        """边沿触发: 外设拉高电平 → 投递到 hart 0 S-file eiid."""
        a = _make_aplic()
        imsic = a._imsic
        imsic.csr_write(0, 'S', 0x70, 1)  # S-file eidelivery=1
        imsic.csr_write(0, 'S', 0xC0, 1 << 6)  # S-file eie bit 6
        _configure(a, 1, hart=0, eiid=6, level=False)
        a.set_irq(1, True)
        mip = imsic.get_pending_mip(0)
        assert mip & (1 << 9), f"SEIP should be set, got mip={mip:#x}"

    def test_level_trigger_keeps_pending_until_clear(self):
        """电平触发: 拉高投递, 撤除后输入电平回落."""
        a = _make_aplic()
        imsic = a._imsic
        imsic.csr_write(0, 'S', 0x70, 1)
        imsic.csr_write(0, 'S', 0xC0, 1 << 6)
        _configure(a, 1, hart=0, eiid=6, level=True)
        a.set_irq(1, True)
        assert a._state[1] & 1 << 8, "input level should be high"
        # 投递后 pending 被清除 (MSI 模式)
        assert not (a._state[1] & 1), "pending cleared after MSI delivery"
        a.set_irq(1, False)
        assert not (a._state[1] & 1 << 8), "input level should drop"

    def test_hart_routing_via_target(self):
        """target 路由: 源投递到 target 指定的 hart, 不影响其他 hart."""
        a = _make_aplic()
        imsic = a._imsic
        # hart 0 与 hart 1 均使能 eiid 7
        for h in (0, 1):
            imsic.csr_write(h, 'S', 0x70, 1)
            imsic.csr_write(h, 'S', 0xC0, 1 << 7)
        _configure(a, 1, hart=1, eiid=7, level=False)
        a.set_irq(1, True)
        assert imsic.get_pending_mip(1) & (1 << 9), "hart 1 should get SEIP"
        assert imsic.get_pending_mip(0) == 0, "hart 0 should be unaffected"

    def test_inactive_source_no_delivery(self):
        """sourcecfg=INACTIVE 时外设中断不投递."""
        a = _make_aplic()
        imsic = a._imsic
        imsic.csr_write(0, 'S', 0x70, 1)
        imsic.csr_write(0, 'S', 0xC0, 1 << 6)
        # target 已设但 sourcecfg 仍 INACTIVE
        _w(a, _TARGET_BASE, (0 << _HART_SHIFT) | 6)
        a.set_irq(1, True)
        assert imsic.get_pending_mip(0) == 0, "inactive source should not deliver"


# ============================================================
#  setipnum / EOI retrigger
# ============================================================


class TestAplicSetipnum:
    """setipnum 单源置 pending — 供 EOI retrigger 使用."""

    def test_setipnum_retriggers_delivery(self):
        """setipnum → 置 pending → 投递 MSI."""
        a = _make_aplic()
        imsic = a._imsic
        imsic.csr_write(0, 'S', 0x70, 1)
        imsic.csr_write(0, 'S', 0xC0, 1 << 15)
        _configure(a, 8, hart=0, eiid=15, level=True)
        # 拉高输入使源保持有效, EOI 后软件写 setipnum 重新检查
        a.set_irq(8, True)
        _w(a, _SETIPNUM, 8)
        mip = imsic.get_pending_mip(0)
        assert mip & (1 << 9), f"setipnum retrigger should deliver, got mip={mip:#x}"

    def test_setipnum_le_level_high_input_low_no_retrigger(self):
        """电平触发源输入已撤除时, setipnum_le 不重挂起 (AIA §4.9.2 防风暴)."""
        a = _make_aplic()
        imsic = a._imsic
        imsic.csr_write(0, 'S', 0x70, 1)
        imsic.csr_write(0, 'S', 0xC0, 1 << 15)
        _configure(a, 8, hart=0, eiid=15, level=True)
        # 拉高投递后再撤除: 输入回落, pending 已由投递清除
        a.set_irq(8, True)
        a.set_irq(8, False)
        assert not (a._state[8] & (1 << 8)), "input should be low after deassert"
        # EOI retrigger: 输入已低, 不应重新置 pending
        _w(a, _SETIPNUM_LE, 8)
        assert not (a._state[8] & 1), "pending must stay clear when input deasserted"

    def test_setipnum_le_level_high_input_high_retriggers(self):
        """电平触发源输入仍有效时, setipnum_le 重挂起并投递."""
        a = _make_aplic()
        imsic = a._imsic
        imsic.csr_write(0, 'S', 0x70, 1)
        imsic.csr_write(0, 'S', 0xC0, 1 << 15)
        _configure(a, 8, hart=0, eiid=15, level=True)
        # 拉高输入 (投递清除 pending 但输入保持有效)
        a.set_irq(8, True)
        assert not (a._state[8] & 1), "pending cleared after delivery"
        # EOI retrigger: 输入仍有效, 重新置 pending
        _w(a, _SETIPNUM_LE, 8)
        assert not (a._state[8] & 1), "pending cleared after redelivery"
        assert imsic.get_pending_mip(0) & (1 << 9), "SEIP should remain asserted"


# ============================================================
#  domaincfg IE 门控
# ============================================================


class TestAplicDomaincfg:
    """domaincfg.IE 门控 MSI 投递."""

    def test_domaincfg_ie_disabled_blocks(self):
        """IE=0 时不投递, 待 IE=1 时投递 pending 源."""
        a = _make_aplic()
        imsic = a._imsic
        imsic.csr_write(0, 'S', 0x70, 1)
        imsic.csr_write(0, 'S', 0xC0, 1 << 6)
        _configure(a, 1, hart=0, eiid=6, level=True)
        _w(a, _DOMAINCFG, 0)  # IE=0
        a.set_irq(1, True)
        assert imsic.get_pending_mip(0) == 0, "IE=0 should block delivery"
        # 恢复 IE=1 → 投递 pending 源
        _w(a, _DOMAINCFG, 1 << 8)
        assert imsic.get_pending_mip(0) & (1 << 9), "IE re-enable should deliver"
