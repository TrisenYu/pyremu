#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""调试器共用的异常类型."""

from __future__ import annotations


class DebuggerError(Exception):
    """调试器在使用其功能时抛出的异常基类."""


class UsageError(DebuggerError):
    """命令参数不满足用法; 文本面向使用者, 由命令层报告后继续."""


class ElfFormatError(DebuggerError):
    """ELF 文件的结构不满足符号加载的要求."""


class DeployError(DebuggerError):
    """符号文件无法部署到目标内存, 或部署后的内容校验不一致."""
