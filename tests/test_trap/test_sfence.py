#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""SFENCE.VMA 的编码、全部 hart 的 TLB 刷新与地址翻译模式切换."""

from pyremu.core.decoder import Hart
from pyremu.utils.disassem import Opc


class TestSfenceVma:
    """验证 SFENCE.VMA 指令的 TLB 刷新."""

    def test_sfence_vma_flushes_tlbs(self):
        h = Hart(id=0)
        h.itlb.insert(vpn=0x100, ppn=0x200, perm=0xF)
        h.dtlb.insert(vpn=0x300, ppn=0x400, perm=0xF)
        assert len(h.itlb) == 1 and len(h.dtlb) == 1

        # SFENCE.VMA 正确编码:
        #   funct7=0b0001001 (bits 31:25), rs2=0, rs1=0
        #   -> imm[31:20] = (0b0001001 << 5) | 0 = 0x120
        instr_val = (0x120 << 20) | Opc.sys.value
        h.exec_instr(instr_val)

        assert len(h.itlb) == 0, "SFENCE.VMA 应刷新 itlb"
        assert len(h.dtlb) == 0, "SFENCE.VMA 应刷新 dtlb"

    def test_sfence_vma_rejects_wrong_funct12(self):
        """错误的 funct12 编码 (如旧的 0x104) 应触发 IllInstr (mcause=2)."""

        h = Hart(id=0)
        # 0x104 是旧代码中错误的 funct12 — 不是合法的特权指令编码
        instr_val = (0x104 << 20) | Opc.sys.value
        h.exec_instr(instr_val)
        assert h.mcause_val == 2, (
            f"0x104 不是合法的 SFENCE.VMA 编码, 应触发 IllInstr (mcause=2),"
            f" 实际 mc={h.mcause_val}"
        )
