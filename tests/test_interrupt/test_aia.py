#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""AIA 模块入口测试.

``pyremu.interrupt.aia`` 只把 IMSIC 转出, 供旧调用方按原路径导入. 本文件锁住该转出
关系, 使拆分实现之后旧路径不至于静默失效.
"""

from pyremu.interrupt import aia, imsic
from pyremu.interrupt.aia import IMSIC


class TestAiaExport:
    """转出的符号与实现模块一致."""

    def test_exports_imsic_class(self):
        """转出的 IMSIC 就是 imsic 模块中的同一个类."""
        assert aia.IMSIC is imsic.IMSIC

    def test_all_lists_only_imsic(self):
        """__all__ 只登记 IMSIC."""
        assert aia.__all__ == ["IMSIC"]

    def test_import_from_legacy_path(self):
        """按旧路径从 aia 导入得到同一个类."""
        assert IMSIC is imsic.IMSIC
