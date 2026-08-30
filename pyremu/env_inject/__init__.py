#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
环境注入包 — 将外部 shellcode 注入 RAM 作为目标程序的预加载运行时.

snippets.py  提供可复用的 RISC-V 指令片段积木块 (用户自行 import 组装),
preload.py   负责将外部提供的 shellcode 字节流写入 RAM 并返回入口地址。

Usage:
    from pyremu.env_inject import Preloader
    from pyremu.env_inject.snippets import set_sp, set_gp

    # 用户自行组装 shellcode:
    shellcode = b"".join(w.to_bytes(4, "little") for w in set_sp(0x7FFF000).words)
    shellcode += ...  # 更多片段 + 跳转到 main 的指令

    # 注入:
    p = Preloader(emu)
    entry = p.inject(shellcode)
"""

from pyremu.env_inject.preload import Preloader
from pyremu.env_inject.snippets import AsmSnippet, hosted_bootstrap

__all__ = ["AsmSnippet", "Preloader", "hosted_bootstrap"]
