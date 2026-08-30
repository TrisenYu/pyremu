#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 17:07:42
# Last modified at 2026/06/09 星期二

"""
翻译后备缓冲器 (Translation Lookaside Buffer, TLB).

TLB 缓存最近使用的虚拟页号 (VPN) -> 物理页号 (PPN) 映射,
加速从虚拟地址到物理地址的转换, 避免每次内存访问都遍历页表。

重构为继承 CacheBase 抽象基类, 统一缓存管理框架。
"""

from dataclasses import dataclass

from pyremu.memory.cache_base import CacheBase, CacheLineBase, ReplacementPolicy


@dataclass
class TLBLine(CacheLineBase):
    """TLB 缓存行 — 在 CacheLineBase 基础上增加地址翻译专用字段.

    tag  = VPN (虚拟页号)
    ppn  = 物理页号
    perm = 权限位 (R|W|X|U 的组合)
    level = 页表级数 (0=4 KiB, 1=2 MiB, 2=1 GiB)
    asid = 地址空间 ID (0=全局/Bare, 非零时 lookup 需匹配)
    gen  = 插入时的 tlb_gen; lookup 时与 TLB._tlb_gen 比较,
           不匹配表示该条目已被 SFENCE.VMA 失效
    """

    ppn: int = 0
    perm: int = 0
    level: int = 0
    asid: int = 0
    gen: int = 0


class TLB(CacheBase):
    """全相联 TLB, 继承 CacheBase 提供 VPN->PPN 映射缓存.

    默认 FIFO 替换策略, 支持按 VPN 查找/插入/刷新。
    保持与旧版兼容的 lookup/insert/flush 接口。

    Usage:
        tlb = TLB(size=256)
        hit, ppn, perm = tlb.lookup(vpn)
        if not hit:
            # ... 页表遍历 ...
            tlb.insert(vpn, ppn, perm, level=0)
    """

    def __init__(
        self,
        size: int = 256,
        policy: ReplacementPolicy = ReplacementPolicy.FIFO,
    ) -> None:
        # CacheBase 需要 name 参数; 这里用 "TLB" 作为默认名称
        super().__init__(name="TLB", num_entries=size, policy=policy)

    # ----------------------------------------------------------
    #  CacheBase 抽象方法实现
    # ----------------------------------------------------------

    def _make_line(self) -> TLBLine:
        """创建空的 TLB 缓存行."""
        return TLBLine()

    def _match(self, entry: CacheLineBase, key) -> bool:
        """标签匹配: entry.tag (VPN) 等于查找 key."""
        return entry.tag == key

    def _on_evict(self, entry: CacheLineBase) -> None:
        """TLB 逐出无额外操作 (无需回写)."""
        pass

    # ----------------------------------------------------------
    #  公共接口 (保持向后兼容)
    # ----------------------------------------------------------

    def lookup(self, vpn: int, asid: int = 0) -> tuple:
        """在 TLB 中查找 vpn.

        ASID 非零时仅匹配相同 ASID 的条目 — 不同的地址空间不共享映射,
        进程切换换 ASID 后无需 SFENCE.VMA (ASID-tagged TLB 语义).
        """
        entry = self._find_by_tag(vpn)
        if entry is not None:
            e: TLBLine = entry  # type: ignore
            if asid != 0 and e.asid not in (0, asid):
                return False, 0, 0
            return True, e.ppn, e.perm
        return False, 0, 0

    def insert(
        self,
        vpn: int,
        ppn: int,
        perm: int,
        level: int = 0,
        mdid: int = 0,
        asid: int = 0,
    ) -> None:
        """将一条映射插入 TLB.  若 vpn 已存在则原地更新."""
        self._clock += 1

        existing_idx = self._tag_to_idx.get(vpn)
        if existing_idx is not None:
            e: TLBLine = self._entries[existing_idx]  # type: ignore
            if e.valid and e.tag == vpn:
                e.ppn = ppn
                e.perm = perm
                e.level = level
                e.mdid = mdid
                e.asid = asid
                e.last_access = self._clock
                return

        idx = self._pick_victim()
        victim: TLBLine = self._entries[idx]  # type: ignore
        if victim.valid:
            self._tag_to_idx.pop(victim.tag, None)
            self._on_evict(victim)

        victim.tag = vpn
        victim.ppn = ppn
        victim.perm = perm
        victim.level = level
        victim.mdid = mdid
        victim.asid = asid
        victim.valid = True
        victim.dirty = False
        victim.last_access = self._clock
        self._tag_to_idx[vpn] = idx

    def flush(self, vpn: int = 0, asid: int = 0) -> None:
        """刷新 TLB.  单 VPN 刷新为 O(1) via _tag_to_idx.

        Args:
            vpn: 若为 0 则刷新全部; 否则仅刷新匹配该 VPN 的条目.
            asid: 暂未使用 (预留, 配合 ASID 做按地址空间刷新).
        """
        if vpn == 0:
            self.flush_all()
            return

        idx = self._tag_to_idx.pop(vpn, None)
        if idx is not None:
            e: TLBLine = self._entries[idx]  # type: ignore
            if e.valid and e.tag == vpn:
                e.valid = False
                e.tag = 0
                e.ppn = 0
                e.perm = 0
                e.level = 0

    # ----------------------------------------------------------
    #  条目访问
    # ----------------------------------------------------------

    @property
    def entries(self) -> "list[TLBLine]":
        """返回所有 TLBLine 条目 (含无效条目)."""
        return self._entries  # type: ignore[return-type]

    # ----------------------------------------------------------
    #  兼容属性
    # ----------------------------------------------------------

    @property
    def size(self) -> int:
        """返回 TLB 容量 (总槽位数)."""
        return self._num_entries

def decode_tlb_perm(perm: int) -> str:
    """TLB 权限位 -> 可读字符串: 0b1111 -> 'RWXU'."""
    r = "R" if perm & 1 else "-"
    w = "W" if perm & 2 else "-"
    x = "X" if perm & 4 else "-"
    u = "U" if perm & 8 else "S"
    return f"{r}{w}{x}{u}"
