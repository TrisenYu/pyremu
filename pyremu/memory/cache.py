#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 16:53:52
# Last modified at 2026/06/06 星期六 17:07:36
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
