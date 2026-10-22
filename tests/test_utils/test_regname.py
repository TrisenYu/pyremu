#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""GPR 与 CSR 名称查找测试.

名称映射由 ``core.registers`` 在模块加载时注入, 故本文件先导入该模块使映射到位,
再核对查询结果与编号越界时的回退取值.
"""

from pyremu.utils.regname import check_csr, gpr_alias, gpr_name

GPR_COUNT = 32


class TestGprName:
    """GPR 名称查询."""

    def test_injected_name_is_returned(self):
        """编号在映射范围内时返回注入的名称."""
        assert gpr_name(0) == "x0"
        assert gpr_name(2) == "x2"
        assert gpr_name(GPR_COUNT - 1) == f"x{GPR_COUNT - 1}"

    def test_out_of_range_number_falls_back(self):
        """编号越界时按 ?x<编号> 形式回退, 负数编号同样如此."""
        assert gpr_name(GPR_COUNT) == f"?x{GPR_COUNT}"
        assert gpr_name(-1) == "?x-1"


class TestGprAlias:
    """GPR 的 ABI 别名查询."""

    def test_injected_alias_is_returned(self):
        """编号在映射范围内时返回注入的别名."""
        assert gpr_alias(0) == "zero"
        assert gpr_alias(2) == "sp"
        assert gpr_alias(10) == "a0"

    def test_out_of_range_number_yields_empty(self):
        """编号越界时回退为空串."""
        assert gpr_alias(GPR_COUNT) == ""
        assert gpr_alias(-1) == ""


class TestCheckCsr:
    """CSR 地址查找."""

    def test_known_csr_is_reported_valid(self):
        """已实现的 CSR 返回有效并给出名称."""
        assert check_csr(0x300) == (True, "mstatus")
        assert check_csr(0x100) == (True, "sstatus")

    def test_address_is_masked_to_twelve_bits(self):
        """高于 12 位的部分被截去, 与低 12 位相同的地址得到同一结果."""
        assert check_csr(0x1300) == check_csr(0x300)
        assert check_csr(0x1300) == (True, "mstatus")

    def test_unknown_csr_is_reported_invalid(self):
        """未登记或未实现的 CSR 返回无效且名称为空串."""
        assert check_csr(0xFFF) == (False, "")
        assert check_csr(0x7FF) == (False, "")
