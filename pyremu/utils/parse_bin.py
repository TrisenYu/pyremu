#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/05 星期五 14:12:25
# Last modified at 2026/06/05 星期五 15:25:00
import lief

from pyremu.utils.wrapper import seize_err_if_any


@seize_err_if_any()
def parse_bin(path: str) -> lief.binary:
    return lief.parse(path)


"""
entry = bin.entrypoint							# 程序入口虚拟地址
text = bytes(bin.get_section(".text").content)  #
code_data = bytes(text.content)					# .text原始机器码
va_base = text.virtual_address  				# 代码加载基址
"""

if __name__ == "__main__":
    pass
