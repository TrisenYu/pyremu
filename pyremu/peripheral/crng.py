#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""模拟硬件随机数生成器 (CRNG) — ``pyremu,crng``.

为 guest 提供运行期熵源, 与 ``/chosen/rng-seed`` 的引导期种子互补:
前者只在开机时初始化内核 CRNG, 本设备则可在任意时刻被固件/内核读取,
持续获得随机字节 (对照真实硬件的 hwrng 角色)。

寄存器布局 (每寄存器 4 字节宽):
  偏移    名称     读写    描述
  ──────────────────────────────────────
  0x00    DATA     R      读取返回 ``size`` 字节随机数 (1/2/4/8)
  0x04    STATUS   R      bit0 = ready (恒 1, 熵源永不耗尽)
  0x08    (reserved)
  0x0C    ID       R      设备 ID (只读, 0x43524E47 = "CRNG")

熵源: 默认使用宿主 ``os.urandom`` 提供真随机字节; 传入 ``seed`` 则切换为
确定性的 ``random.Random`` 流 (供测试复现, 两个相同 seed 的实例产生相同序列)。

DTB 绑定:
  compatible = "pyremu,crng"
  reg = <base size>
"""

from __future__ import annotations

import os
import random
import struct

from pyremu.memory.bus import Device

# 寄存器偏移
REG_DATA = 0x00    # 读取返回 size 字节随机数
REG_STATUS = 0x04  # bit0 = ready
REG_ID = 0x0C      # 设备 ID (只读)

STATUS_READY = 1 << 0
DEVICE_ID = 0x4352_4E47  # "CRNG" (ASCII)


class CRNG(Device):
    """模拟硬件随机数生成器 — 纯被动 MMIO 设备."""

    def __init__(
        self,
        base: int = 0x1000_6000,
        size: int = 0x1000,
        *,
        seed: int | None = None,
    ) -> None:
        self.base_addr = base
        self.size = size
        # seed=None -> os.urandom (真随机); 否则 random.Random(seed) (确定性).
        self._rand: random.Random | None = None if seed is None else random.Random(seed)

    def _entropy(
        self,
        n: int,
    ) -> bytes:
        """返回 ``n`` 字节随机数."""
        if self._rand is None:
            return os.urandom(n)
        return self._rand.getrandbits(n * 8).to_bytes(n, "little")

    # ---- Device 接口 ----

    def read(
        self,
        offset: int,
        size: int,
    ) -> bytes:
        if offset == REG_DATA:
            return self._entropy(size)
        if offset == REG_STATUS:
            return struct.pack("<I", STATUS_READY)
        if offset == REG_ID:
            return struct.pack("<I", DEVICE_ID)
        return b"\x00" * size

    def write(
        self,
        offset: int,
        data: bytes,
    ) -> None:
        # 纯被动设备: 所有写入忽略 (无副作用).
        return None
