#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""验证无 Rust .so 时纯 Python fallback 仍然正确。"""

from __future__ import annotations

import pytest

import pyremu._native as _nat
from pyremu._native import native_available
from pyremu.core.decoder import Hart
from pyremu.core.mem_check_aux import inject_memory_backend
from pyremu.utils.disassem import Opc
from tests.test_calc.test_compressed_diff import (
    _b_type,
    _c1,
    _i_type,
    _j_type,
    _make_ram,
    _u_type,
)


@pytest.fixture
def _no_native():
    """临时禁用 native 库, 测试后恢复。"""
    saved = _nat._lib
    _nat._lib = None
    try:
        yield
    finally:
        _nat._lib = saved


def test_fallback_correctness(_no_native):
    """无 native 库时 32-bit / 压缩指令均正确执行 (复用 diff test 编码辅助)。"""
    ram, rf, wf = _make_ram()
    h = Hart(id=0)
    inject_memory_backend(h, rf, wf)

    # ---- 32-bit ----
    h.gprs[5] = 10
    h.exec_instr(_i_type(Opc.opImm.value, 5, 0, 5, 7))  # ADDI x5, 7
    assert h.gprs[5] == 17

    h.pc = 0x80000000
    h.exec_instr(_j_type(Opc.jal.value, 1, 40))  # JAL x1, +40
    assert h.gprs[1] == 0x80000004
    assert h.pc == 0x80000028

    h.pc = 0x80000000
    h.exec_instr(_j_type(Opc.jal.value, 0, -2))  # JAL 负偏移 (sext21)
    assert h.pc == 0x7FFFFFFE

    h.exec_instr(_u_type(Opc.lui.value, 10, 0x12345))  # LUI
    assert h.gprs[10] == 0x12345000

    h.gprs[10] = 0
    h.gprs[11] = 0
    h.pc = 0x80000000
    h.exec_instr(_b_type(Opc.br.value, 0, 10, 11, 24))  # BEQ (taken)
    assert h.pc == 0x80000018

    # ---- 压缩 ----
    h.gprs[10] = 5
    h.exec_instr(0x050D)  # C.ADDI x10, 3
    assert h.gprs[10] == 8

    h.pc = 0x80000000
    h.gprs[8] = 0
    c_beqz = _c1(6, 1, int(  # rs1'=x9, offset=+8
        ((8 >> 8) & 0x1) << 12
        | ((8 >> 3) & 0x3) << 10
        | ((8 >> 7) & 0x3) << 5
        | ((8 >> 1) & 0x3) << 3
        | ((8 >> 5) & 0x1) << 2
    ))
    h.exec_instr(c_beqz)  # C.BEQZ x9, +8 (taken)
    assert h.pc == 0x80000008

    h.gprs[2] = 0x80000
    h.exec_instr(0x717D)  # C.ADDI16SP -16
    assert h.gprs[2] == 0x7FFF0


def test_native_available_true():
    """测试环境中 native 库应已加载。"""
    assert native_available() is True
