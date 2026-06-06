#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 20:50:31
# Last modified at 2026/06/06 星期六 21:18:00

"""
指令解码
"""

"""
R 类型: func7 rs2 rs1 func3 rd opcode
I 类型:   imm12   rs1 func3 rd opcode
S 类型: imm7  rs2 rs1 func3 rd opcode
U 类型:      imm20    func3 rd opcode
	J 类型:

Mem排序指令:  fm4 PSxIORW rs1 func3 rd opcode
ecall/ebreak:    func12   rs1 func3 rd opcode

"""
parse_opcode = lambda x: x & 0b111_1111
parse_rd = lambda x: (x >> 6) & 0b1_1111
parse_func3 = lambda x: (x >> 12) & 0b0111
parse_rs1 = lambda x: (x >> 15) & 0b1_1111
parse_rs2 = lambda x: (x >> 20) & 0b1_1111
parse_imm7 = lambda x: (x >> 25) & 0b111_1111
parse_func7 = lambda x: (x >> 25) & 0b111_1111
parse_imm12 = lambda x: (x >> 20) & 0xFFF    # 4 * 3 = 12
parse_imm20 = lambda x: (x >> 12) & 0xF_FFFF # 4 * 5 = 20

