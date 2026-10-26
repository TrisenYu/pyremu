#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""rvdb 多来源调试符号.

对外导出两个类:

- SymLookupOps  符号解析、内核符号加载、符号模块查找与段查找
- SymCmdOps     sym 命令的分发、帮助文本与固件符号输出

各文件按职责拆分:

- lookup.py      符号查找与内核符号加载
- model.py       已加载模块与候选文件的数据结构, 以及按地址选取模块的规则
- elf_idx.py     流式 ELF 符号索引
- loader.py      符号文件的加载、部署区间校验与内容校验
- registry.py    模块清单的登记、替换、移除与失效刷新
- cmd.py         符号命令的分发与帮助文本
- cmd_files.py   sym chdir / sym ls / sym load
- cmd_modules.py sym detail / sym rm / sym info
"""

from __future__ import annotations

from pyremu.debug.sym.cmd import SymCmdOps
from pyremu.debug.sym.lookup import SymLookupOps

__all__ = [
    "SymCmdOps",
    "SymLookupOps",
]
