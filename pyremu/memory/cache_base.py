#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/09 星期二

"""
缓存抽象基类。

TLB 和各类 Cache (L1 I-Cache, L1 D-Cache, L2 共享缓存) 在结构上相似:
- 全相联或组相联组织
- 按 tag 匹配查找条目
- 命中返回数据, 未命中则逐出旧条目、插入新条目
- FIFO / LRU / 随机替换策略
- 刷新 (单个或全部)

本模块提供:
- CacheLineBase:   缓存行基类 (tag, valid, dirty, last_access)
- CacheBase:       缓存抽象基类 (条目管理, 查找/分配/刷新/逐出框架)
- ReplacementPolicy: 替换策略枚举

子类只需实现 _make_line(), _match(), _on_evict() 即可定制行为.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ReplacementPolicy(Enum):
    """缓存替换策略."""

    FIFO = "fifo"
    LRU = "lru"


@dataclass
class CacheLineBase:
    """缓存行基类 — TLB 条目和各级缓存行共用.

    子类可添加自己的字段 (如 TLB 的 level/perm, L2 的 mesi/line_data).
    """

    tag: int = 0
    valid: bool = False
    dirty: bool = False
    last_access: int = 0  # 全局时钟值, LRU 替换时比较
    mdid: int = 0  # 内存域 ID (供 TEE 飞地隔离, mfence.did 按域刷新)


class CacheBase(ABC):
    """缓存抽象基类.

    管理固定数量的 CacheLineBase 条目, 提供:
    - 按 tag 查找 (_find_by_tag)
    - 条目分配 (_alloc_entry, FIFO/LRU)
    - 逐出回调 (_on_evict)
    - 刷新 (flush / flush_all)

    子类职责:
    - _make_line() -> 创建子类缓存行实例
    - _match(entry, key) -> 定义 tag 匹配规则
    - _on_evict(entry) -> 逐出时的自定义行为 (如回写)
    - 可覆写 _pick_victim() 自定义替换策略
    """

    def __init__(
        self,
        name: str,
        num_entries: int,
        policy: ReplacementPolicy = ReplacementPolicy.FIFO,
    ) -> None:
        self.name = name
        self._num_entries = num_entries
        self._policy = policy
        self._entries: list[CacheLineBase] = [self._make_line() for _ in range(num_entries)]
        self._clock: int = 0  # 全局访问计数 — LRU 用
        self._fifo_ptr: int = 0  # FIFO 写入指针
        self._hits: int = 0
        self._misses: int = 0
        self._tag_to_idx: dict[int, int] = {}  # O(1) tag->index 快速查找

    # ----------------------------------------------------------
    #  子类必须实现的抽象方法
    # ----------------------------------------------------------

    @abstractmethod
    def _make_line(self) -> CacheLineBase:
        """创建一个空的缓存行 (子类返回自己的 CacheLineBase 子类)."""

    @abstractmethod
    def _match(self, entry: CacheLineBase, key) -> bool:
        """标签匹配规则: entry.tag 是否等于 key."""

    def _on_evict(self, entry: CacheLineBase) -> None:
        """逐出条目时的回调 (子类可选覆写, 如 L2 脏行回写)."""

    # ----------------------------------------------------------
    #  查找
    # ----------------------------------------------------------

    def _find_index(self, key) -> int:
        """按 key 查找条目, 返回索引; 未命中返回 -1. 命中时更新 last_access.

        O(1) 快速路径: _tag_to_idx dict 避免 O(n) 线性扫描.
        """
        self._clock += 1
        idx = self._tag_to_idx.get(key)
        if idx is not None:
            entry = self._entries[idx]
            if entry.valid and self._match(entry, key):
                entry.last_access = self._clock
                self._hits += 1
                return idx
            # 脏条目 (不应发生): 逐出时未清理 dict, 容错清理
            del self._tag_to_idx[key]
        self._misses += 1
        return -1

    def _find_by_tag(self, tag: int) -> Optional[CacheLineBase]:
        """按 tag 查找并返回条目; 未命中返回 None. 命中时更新 last_access."""
        idx = self._find_index(tag)
        return self._entries[idx] if idx >= 0 else None

    # ----------------------------------------------------------
    #  分配 / 替换
    # ----------------------------------------------------------

    def _pick_victim(self) -> int:
        """选择要逐出的槽位索引.

        默认 FIFO: 返回 _fifo_ptr 并推进.
        LRU: 返回 last_access 最小的有效条目; 若无有效条目则用 FIFO.
        """
        if self._policy == ReplacementPolicy.LRU:
            # 优先选无效条目
            for i, entry in enumerate(self._entries):
                if not entry.valid:
                    return i
            # 全部有效 — 找最久未使用的
            oldest_idx = 0
            oldest_time = self._entries[0].last_access
            for i, entry in enumerate(self._entries):
                if entry.last_access < oldest_time:
                    oldest_time = entry.last_access
                    oldest_idx = i
            return oldest_idx

        # FIFO (默认)
        idx = self._fifo_ptr
        self._fifo_ptr = (self._fifo_ptr + 1) % self._num_entries
        return idx

    def _alloc_entry(self, tag: int, **kwargs) -> CacheLineBase:
        """分配一个条目: 若 tag 已存在则原地更新; 否则选择 victim 并逐出.

        Returns:
            分配的 CacheLineBase 条目 (子类应填充自定义字段).
        """
        # 查重 — 原地更新 (O(1) via _tag_to_idx)
        idx = self._find_index(tag)
        if idx >= 0:
            return self._entries[idx]

        # 需要逐出
        idx = self._pick_victim()
        victim = self._entries[idx]
        if victim.valid:
            self._tag_to_idx.pop(victim.tag, None)  # 清理旧 tag->index 映射
            self._on_evict(victim)
        victim.valid = False
        victim.dirty = False
        victim.tag = tag
        victim.last_access = self._clock
        self._tag_to_idx[tag] = idx  # 注册新映射
        return victim

    # ----------------------------------------------------------
    #  刷新
    # ----------------------------------------------------------

    def flush(self, tag: int = 0) -> None:
        """刷新缓存条目.

        Args:
            tag: 若为 0 则刷新全部; 否则仅刷新匹配 tag 的条目.
        """
        if tag == 0:
            self.flush_all()
            return

        idx = self._tag_to_idx.pop(tag, None)
        if idx is not None:
            entry = self._entries[idx]
            if entry.valid and self._match(entry, tag):
                self._on_evict(entry)
                entry.valid = False
                entry.tag = 0
                entry.dirty = False

    def flush_all(self) -> None:
        """刷新全部条目 (回写脏行 + 失效)."""
        for entry in self._entries:
            if entry.valid:
                self._on_evict(entry)
            entry.valid = False
            entry.tag = 0
            entry.dirty = False
        self._tag_to_idx.clear()

    def flush_by_mdid(self, mdid: int) -> int:
        """按内存域 ID 刷新缓存条目 — 抗侧信道.

        遍历全部条目, 将 mdid 匹配的有效条目回写并失效.
        同步维护 _tag_to_idx 快速查找索引.
        返回被刷新的条目数.
        """
        count = 0
        for entry in self._entries:
            if not (entry.valid and entry.mdid == mdid):
                continue
            self._on_evict(entry)
            self._tag_to_idx.pop(entry.tag, None)
            entry.valid = False
            entry.tag = 0
            entry.dirty = False
            count += 1
        return count

    # ----------------------------------------------------------
    #  统计
    # ----------------------------------------------------------

    @property
    def num_entries(self) -> int:
        return self._num_entries

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        """返回当前有效条目数."""
        return sum(1 for e in self._entries if e.valid)

    def __iter__(self):
        """迭代所有有效条目 (供调试/诊断)."""
        return iter(e for e in self._entries if e.valid)

    def __repr__(self) -> str:
        return (
            f"{self.name}(entries={len(self)}/{self._num_entries}, "
            f"policy={self._policy.value}, hit_rate={self.hit_rate:.3f})"
        )
