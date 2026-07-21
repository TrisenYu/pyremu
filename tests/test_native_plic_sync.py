#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""native 引擎 PLIC -> mip 同步回归测试.

锁定修复: native 批次引擎内部只同步 CLINT (MSIP/MTIP), 不感知 PLIC。
`_step_native` 在 marshal 前必须调用 `_native_sync_plic_mip`, 把 PLIC 的
外部中断挂起 (MEIP bit11 / SEIP bit9) 合并进各 hart 的 mip —— 否则 virtio
等外设的完成中断永远到不了 hart (S 模式 Linux 收不到 SEIP)。

修复前 (无此同步): 走 native 路径时 hart.mip 的 SEIP/MEIP 恒为 0。
"""

from __future__ import annotations

import pytest

from pyremu.emulator import Emulator
from pyremu.interrupt.plic import (
    PLIC_CONTEXT_BASE,
    PLIC_CONTEXT_STRIDE,
    PLIC_ENABLE_BASE,
    PLIC_ENABLE_STRIDE,
    PLIC_PRIORITY_STRIDE,
)
from pyremu.platform import PeripheralConfig, PlatformConfig

_SRC = 1  # 中断源号 (对齐 virtio-blk)
SEIP = 1 << 9
MEIP = 1 << 11


def _make_emu(num_harts: int = 1) -> Emulator:
    cfg = PlatformConfig(
        num_harts=num_harts,
        ram_size=8 * 1024 * 1024,
        ram_base=0x8000_0000,
        prog_cnt=0x8000_0000,
        periph=PeripheralConfig(),
    )
    return Emulator(cfg)


def _plic_program(plic, context: int, source: int) -> None:
    """经 MMIO 编程 PLIC: 置源优先级并在指定 context 使能该源."""
    plic.write(source * PLIC_PRIORITY_STRIDE, (1).to_bytes(4, "little"))  # priority=1
    word = source // 32
    bit = source % 32
    enable_off = PLIC_ENABLE_BASE + context * PLIC_ENABLE_STRIDE + word * 4
    plic.write(enable_off, (1 << bit).to_bytes(4, "little"))
    # threshold 默认 0 (< priority=1), 无需设置
    _ = PLIC_CONTEXT_BASE + context * PLIC_CONTEXT_STRIDE  # (仅文档化偏移)


def test_native_sync_sets_seip_from_s_context():
    """源在 hart0 S-context(ctx1) 使能且挂起 -> 同步后 mip 含 SEIP."""
    emu = _make_emu(1)
    _plic_program(emu.plic, context=1, source=_SRC)  # ctx1 = hart0 S
    emu.plic.set_irq(_SRC, True)

    assert emu.harts[0].mip_val & SEIP == 0  # 同步前无
    emu._native_sync_plic_mip()
    assert emu.harts[0].mip_val & SEIP == SEIP  # 同步后 SEIP 置位
    assert emu.harts[0].mip_val & MEIP == 0  # 未使能 M-context


def test_native_sync_sets_meip_from_m_context():
    """源在 hart0 M-context(ctx0) 使能 -> 同步后 mip 含 MEIP."""
    emu = _make_emu(1)
    _plic_program(emu.plic, context=0, source=_SRC)  # ctx0 = hart0 M
    emu.plic.set_irq(_SRC, True)

    emu._native_sync_plic_mip()
    assert emu.harts[0].mip_val & MEIP == MEIP
    assert emu.harts[0].mip_val & SEIP == 0


def test_native_sync_clears_when_not_pending():
    """PLIC 撤除挂起后, 同步应清掉 mip 的 SEIP (不残留)."""
    emu = _make_emu(1)
    _plic_program(emu.plic, context=1, source=_SRC)
    emu.plic.set_irq(_SRC, True)
    emu._native_sync_plic_mip()
    assert emu.harts[0].mip_val & SEIP == SEIP

    emu.plic.set_irq(_SRC, False)
    emu._native_sync_plic_mip()
    assert emu.harts[0].mip_val & SEIP == 0


def test_native_sync_per_hart_isolated():
    """多 hart: 源仅在 hart1 S-context 使能 -> 只有 hart1 得 SEIP."""
    emu = _make_emu(2)
    # hart1 的 S-context = 2*1+1 = 3
    _plic_program(emu.plic, context=3, source=_SRC)
    emu.plic.set_irq(_SRC, True)

    emu._native_sync_plic_mip()
    assert emu.harts[1].mip_val & SEIP == SEIP
    assert emu.harts[0].mip_val & SEIP == 0


def test_native_sync_preserves_software_mip_bits():
    """同步只替换 MEIP/SEIP, 不动软件/CLINT 位 (如 MSIP bit3)."""
    emu = _make_emu(1)
    emu.harts[0].mip_val = 1 << 3  # 预置 MSIP
    emu._native_sync_plic_mip()  # PLIC 无挂起
    assert emu.harts[0].mip_val & (1 << 3) == (1 << 3)  # MSIP 保留
    assert emu.harts[0].mip_val & (SEIP | MEIP) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
