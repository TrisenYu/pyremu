#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
预加载注入器 — 将外部提供的 shellcode / 二进制机器码写入 RAM.

shellcode 来源:
- 外部文件 (裸 RISC-V 指令流, 类似 objcopy -O binary 产物)
- 或用户用 snippets 模块手动组装的字节序列

注入后 Emulator 将从 shellcode 入口开始执行, shellcode 负责
设置环境并跳转到目标程序入口。

默认内存布局 (以 ram_base=0x80000000 + ram_size=128 MiB 为例):

  OpenSBI:   0x80000000 - 0x8003ebb0  (PIE 搬迁后, ~251 KiB)
  DTB:       0x87ff0000 - 0x87ff046c  (ram_base + ram_size - 64KiB)
  Preload:   0x87fe0000 - 0x87fe008c  (ram_base + ram_size - 128KiB)
"""

from pathlib import Path

from pyremu.emulator import Emulator


class Preloader:
    """将外部 shellcode 注入 RAM 并返回入口地址.

    Usage:
        p = Preloader(emu)
        entry = p.inject(shellcode_bytes)           # 从 bytes
        entry = p.inject_file("/path/to/crt0.bin")  # 从文件
        emu.harts[0].pc = entry
    """

    # 默认注入位置: RAM 顶端往下 128 KiB (避免和 DTB @ RAM-64K 撞地址)
    DEFAULT_ADDR_OFFSET = 0x20000

    def __init__(
        self,
        emu: Emulator,
    ) -> None:
        self._emu = emu

    def inject(
        self,
        code: bytes,
        addr: int | None = None,
    ) -> int:
        """将 shellcode 字节写入 RAM.

        Args:
            code: 原始 RISC-V 机器码字节.
            addr: 写入地址 (None = 自动选择 RAM 顶端安全区域).

        Returns:
            shellcode 在 RAM 中的起始地址 (即入口地址).
        """
        if addr is None:
            addr = (
                self._emu.bus.ram_base + self._emu.bus.ram_size
                - self.DEFAULT_ADDR_OFFSET
            ) & ~0xF

        self._emu.bus.write(addr, code)
        return addr

    def inject_file(
        self,
        path: str,
        addr: int | None = None,
    ) -> int:
        """从文件读取 shellcode 并注入 RAM.

        Args:
            path: shellcode 文件路径 (裸二进制).
            addr: 写入地址 (None = 自动选择).

        Returns:
            shellcode 入口地址.
        """
        code = Path(path).read_bytes()
        return self.inject(code, addr)
