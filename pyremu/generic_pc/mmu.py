#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 17:07:57
# Last modified at 2026/06/06 星期六 23:30:30

# 访存控制结构
# https://github.com/takahirox/riscv-rust/blob/master/src/mmu.rs
from enum import Enum

MemAccessMode = Enum(
    "MemAccessMode", (
        "None",
        "SV32",
        "SV39",
        "SV48"
    )
)

