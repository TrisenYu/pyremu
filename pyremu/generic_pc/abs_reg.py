#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 21:58:28
# Last modified at 2026/06/06 星期六 23:18:22
from pydantic import BaseModel


class Reg(BaseModel):
    name: str
    alias: str = ""      # 允许使用别名称呼某个寄存器或指令
    restricts: list = [] # 例如读写还有特权级的检查
    val: int = 0         # 内部存储的值，默认清零
