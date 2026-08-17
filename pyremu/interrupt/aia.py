#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""AIA (Advanced Interrupt Architecture) — IMSIC + APLIC 模块.

IMSIC (Incoming MSI Controller): 每 hart MSI 中断控制器, 处理外部 + 软件中断.
APLIC (Advanced PLIC): 有线 → MSI 桥 (Phase 2).

用法:
    from pyremu.interrupt.aia import IMSIC
"""

from pyremu.interrupt.imsic import IMSIC

__all__ = ["IMSIC"]
