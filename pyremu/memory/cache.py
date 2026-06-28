#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 16:53:52
# Last modified at 2026/06/06 星期六 17:07:36

"""
一想到要手动定义驱逐缓存行的逻辑就想笑

缓存结构具有范围，像是Dcache和Icache算作是L1缓存，在某个hart内部
然后L2是共享的，通过访存和驱逐写入的策略来实现

对于锁机制的实现，多核是怎么保证上锁的？
"""

TLB_SIZE: int = 256


class CacheEntry:
    def __init__(self) -> None:
        # rwx权限位
        self.perm = 0
        self.level = 0
        self.mdid = 0  # 暂时不想解释
        self.vpn = 0
        self.ppn = 0
        # 有效位
        self.valid = 0
        # 脏位
        self.dirty = 0


# 定义缓存行结构和存储于其中的各种数据
