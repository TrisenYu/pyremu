#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""验证无 Rust 动态库时纯 Python 路径仍然正确。"""

from __future__ import annotations

from unittest import mock

import pytest

from pyremu._native import native_available
from pyremu.core.decoder import Hart
from pyremu.core.mem_check_aux import inject_memory_backend
from pyremu.utils.disassem import Opc
from tests.test_calc.test_compressed_diff import (
    b_type,
    c1,
    i_type,
    j_type,
    make_ram,
    u_type,
)


@pytest.fixture
def no_native():
    """关闭加速执行, 使指令执行走纯 Python 路径.

    ``handle_alu`` / ``handle_op_imm`` / ``handle_op32`` / ``handle_op_imm32``
    以公开门控 ``native_available()`` 判断是否调用加速执行库, 故替换该函数返回
    False 即退回纯 Python 实现, 无须触及动态库的装载状态。
    """
    with mock.patch("pyremu.core.decoder.native_available", return_value=False) as gate:
        yield gate


def test_fallback_correctness(no_native):
    """无加速执行库时 32-bit / 压缩指令均正确执行 (复用 diff test 编码辅助)。"""
    ram, rf, wf = make_ram()
    h = Hart(id=0)
    inject_memory_backend(h, rf, wf)

    # ---- 32-bit ----
    h.gprs[5] = 10
    h.exec_instr(i_type(Opc.opImm.value, 5, 0, 5, 7))  # ADDI x5, 7
    assert h.gprs[5] == 17

    h.pc = 0x80000000
    h.exec_instr(j_type(Opc.jal.value, 1, 40))  # JAL x1, +40
    assert h.gprs[1] == 0x80000004
    assert h.pc == 0x80000028

    h.pc = 0x80000000
    h.exec_instr(j_type(Opc.jal.value, 0, -2))  # JAL 负偏移 (sext21)
    assert h.pc == 0x7FFFFFFE

    h.exec_instr(u_type(Opc.lui.value, 10, 0x12345))  # LUI
    assert h.gprs[10] == 0x12345000

    h.gprs[10] = 0
    h.gprs[11] = 0
    h.pc = 0x80000000
    h.exec_instr(b_type(Opc.br.value, 0, 10, 11, 24))  # BEQ (taken)
    assert h.pc == 0x80000018

    # ---- 压缩 ----
    h.gprs[10] = 5
    h.exec_instr(0x050D)  # C.ADDI x10, 3
    assert h.gprs[10] == 8

    h.pc = 0x80000000
    h.gprs[8] = 0
    c_beqz = c1(6, 1, int(  # rs1'=x9, offset=+8
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

    # ADDI 与 LUI 经纯 Python 算术路径, 门控必须确实被查询过
    assert no_native.called, "纯 Python 路径未被走到: native_available 门控未被查询"


def test_native_available_true():
    """测试环境中加速执行库应已装载。"""
    assert native_available() is True
