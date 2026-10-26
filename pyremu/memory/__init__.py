#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""内存子系统: MMU, TLB, Cache, Bus, ROM."""

from pyremu.configs_gen import TLB_ENTRIES
from pyremu.memory.bus import Bus, Device
from pyremu.memory.cache_base import CacheBase, CacheLineBase, ReplacementPolicy
from pyremu.memory.l2cache import L2Cache, L2CacheLine, MESIState
from pyremu.memory.mmu import PTE, sv39_walk, translate_va
from pyremu.memory.tlb import TLB, TLBLine

TLB_SIZE = int(TLB_ENTRIES)

__all__ = [
    "Bus",
    "CacheBase",
    "CacheLineBase",
    "Device",
    "L2Cache",
    "L2CacheLine",
    "MESIState",
    "PTE",
    "ReplacementPolicy",
    "TLB",
    "TLBLine",
    "TLB_SIZE",
    "sv39_walk",
    "translate_va",
]
