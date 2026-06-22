#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/09 星期二

"""
L2 共享缓存 — 多 hart 之间的统一二级缓存。

继承 CacheBase, 使用 MESI 一致性协议:
- M (Modified):  脏数据, 仅此缓存拥有, 逐出时需回写 RAM
- E (Exclusive): 干净数据, 仅此缓存拥有, 与 RAM 一致
- S (Shared):    干净数据, 可能多个缓存拥有 (模拟器中仅 L2 自己)
- I (Invalid):   无效条目

组相联组织, 默认 4 路组相联 + LRU 替换。
"""

from dataclasses import dataclass, field
from enum import Enum

from pyremu.memory.cache_base import CacheBase, CacheLineBase, ReplacementPolicy


class MESIState(Enum):
    """MESI 一致性协议状态."""
    MODIFIED = "M"
    EXCLUSIVE = "E"
    SHARED = "S"
    INVALID = "I"


# 默认 L2 缓存配置
DEFAULT_L2_SIZE = 256 * 1024  # 256 KiB
DEFAULT_LINE_SIZE = 64  # 64 字节缓存行
DEFAULT_WAYS = 4  # 4 路组相联


@dataclass
class L2CacheLine(CacheLineBase):
    """L2 缓存行 — 在 CacheLineBase 基础上增加 MESI 状态和数据块.

    tag = 物理地址高位 (addr >> line_shift)
    每行存储 64 字节数据 (bytearray).
    """

    mesi: MESIState = MESIState.INVALID
    data: bytearray = field(default_factory=lambda: bytearray(DEFAULT_LINE_SIZE))


class L2Cache(CacheBase):
    """共享 L2 缓存, 组相联 + MESI 协议.

    位于 Bus 和 RAM 之间, 所有 hart 共享。
    物理地址经过 L2 缓存后再访问 RAM。
    """

    def __init__(
        self,
        size: int = DEFAULT_L2_SIZE,
        line_size: int = DEFAULT_LINE_SIZE,
        ways: int = DEFAULT_WAYS,
        ram_read_fn=None,  # 回调: read(addr, size) -> bytes
        ram_write_fn=None,  # 回调: write(addr, data) -> None
    ) -> None:
        self._line_size = line_size
        self._ways = ways
        self._num_sets = size // (line_size * ways)
        self._line_shift = (line_size - 1).bit_length()  # 64 → 6 bits

        # RAM 后端回调 (用于未命中加载和回写)
        self._ram_read = ram_read_fn
        self._ram_write = ram_write_fn

        super().__init__(
            name="L2Cache",
            num_entries=self._num_sets * ways,
            policy=ReplacementPolicy.LRU,
        )

        # 初始化: 每路一条空行
        self._entries = [self._make_line() for _ in range(self._num_sets * ways)]
        self._current_mdid: int = 0

    # ----------------------------------------------------------
    #  CacheBase 抽象方法实现
    # ----------------------------------------------------------

    def _make_line(self) -> L2CacheLine:
        return L2CacheLine()

    @property
    def current_mdid(self) -> int:
        return self._current_mdid

    @current_mdid.setter
    def current_mdid(self, val: int) -> None:
        self._current_mdid = val

    def _match(self, entry: CacheLineBase, key: int) -> bool:
        """key = 完整物理地址, 匹配 tag (地址高位)."""
        l2e: L2CacheLine = entry  # type: ignore
        return l2e.tag == (key >> self._line_shift)

    def _on_evict(self, entry: CacheLineBase) -> None:
        """M 状态脏行逐出时回写 RAM."""
        l2e: L2CacheLine = entry  # type: ignore
        if l2e.mesi == MESIState.MODIFIED and self._ram_write:
            # 计算物理地址: tag << line_shift → 对齐到缓存行
            pa = l2e.tag << self._line_shift
            self._ram_write(pa, bytes(l2e.data))

    # ----------------------------------------------------------
    #  地址解析
    # ----------------------------------------------------------

    def _addr_fields(self, addr: int) -> tuple:
        """将物理地址分解为 (tag, set_index, offset).

        offset: 行内偏移 (0..63)
        set_index: 组索引
        tag: 地址高位
        """
        offset = addr & (self._line_size - 1)
        set_index = (addr >> self._line_shift) & (self._num_sets - 1)
        tag = addr >> self._line_shift
        return tag, set_index, offset

    def _get_set_entries(self, set_index: int) -> list[L2CacheLine]:
        """返回指定组的所有路条目."""
        base = set_index * self._ways
        return self._entries[base : base + self._ways]  # type: ignore

    # ----------------------------------------------------------
    #  缓存读写
    # ----------------------------------------------------------

    def read(self, addr: int, size: int) -> bytes:
        """从 L2 缓存读取数据。未命中则从 RAM 加载整行。

        自动处理跨缓存行的读取。

        Args:
            addr: 物理地址.
            size: 读取字节数 (1/2/4/8).

        Returns:
            读取的数据 (bytes, 长度 = size).
        """
        result = bytearray()
        cur_addr = addr
        remaining = size

        while remaining:
            tag, set_index, offset = self._addr_fields(cur_addr)
            way_entries = self._get_set_entries(set_index)
            self._clock += 1

            # 当前缓存行还能读取的字节数
            chunk_sz = min(remaining, self._line_size - offset)

            hit = False
            for e in way_entries:
                if not e.valid or e.tag != tag:
                    continue
                e.last_access = self._clock
                e.mdid = self._current_mdid
                self._hits += 1
                result.extend(e.data[offset : offset + chunk_sz])
                hit = True
                break

            if not hit:
                self._misses += 1
                chunk = self._load_line_and_read(
                    cur_addr, chunk_sz, set_index, way_entries, tag, offset,
                )
                result.extend(chunk)

            cur_addr += chunk_sz
            remaining -= chunk_sz

        return bytes(result)

    def _load_line_and_read(
        self,
        addr: int,
        size: int,
        set_index: int,
        way_entries: list[L2CacheLine],
        tag: int,
        offset: int,
    ) -> bytes:
        """未命中时: 选择 victim → 逐出 → 从 RAM 加载整行 → 返回数据."""
        # 选择 victim (同组内的 LRU)
        victim_idx = self._pick_victim_in_set(way_entries)
        if victim_idx < 0:
            # 全部有效但不应发生; 回退到第一个
            victim_idx = 0

        victim = way_entries[victim_idx]

        # 逐出旧行 (M 状态回写)
        if victim.valid and victim.mesi == MESIState.MODIFIED:
            if self._ram_write:
                pa = victim.tag << self._line_shift
                self._ram_write(pa, bytes(victim.data))

        # 从 RAM 加载整行
        line_addr = (addr >> self._line_shift) << self._line_shift
        if self._ram_read:
            line_data = self._ram_read(line_addr, self._line_size)
        else:
            line_data = b"\x00" * self._line_size

        # 写入新行
        victim.tag = tag
        victim.data[:] = line_data
        victim.valid = True
        victim.dirty = False
        victim.mesi = MESIState.EXCLUSIVE
        victim.mdid = self._current_mdid
        victim.last_access = self._clock

        return bytes(victim.data[offset : offset + size])

    def write(self, addr: int, data: bytes) -> None:
        """向 L2 缓存写入数据。写命中时更新行并标记 M。

        自动处理跨缓存行的写入: 将 data 按缓存行边界切分,
        逐段查找/分配并写入。

        Args:
            addr: 物理地址.
            data: 写入数据 (bytes).
        """
        remaining = data
        cur_addr = addr

        while remaining:
            tag, set_index, offset = self._addr_fields(cur_addr)
            way_entries = self._get_set_entries(set_index)
            self._clock += 1

            # 当前缓存行还能容纳的字节数
            chunk_sz = min(len(remaining), self._line_size - offset)
            chunk = remaining[:chunk_sz]

            # 查找命中
            hit = False
            for e in way_entries:
                if not e.valid or e.tag != tag:
                    continue
                for i, b in enumerate(chunk):
                    e.data[offset + i] = b
                e.dirty = True
                e.mesi = MESIState.MODIFIED
                e.mdid = self._current_mdid
                e.last_access = self._clock
                self._hits += 1
                hit = True
                break

            if not hit:
                self._misses += 1
                self._write_allocate(cur_addr, chunk, way_entries, tag, offset)

            remaining = remaining[chunk_sz:]
            cur_addr += chunk_sz

    def _write_allocate(
        self,
        addr: int,
        data: bytes,
        way_entries: list[L2CacheLine],
        tag: int,
        offset: int,
    ) -> None:
        """写未命中时分配新行 (write-allocate 策略)."""
        victim_idx = self._pick_victim_in_set(way_entries)
        victim_idx = max(victim_idx, 0)

        victim = way_entries[victim_idx]

        # 逐出旧行
        if victim.valid and victim.mesi == MESIState.MODIFIED:
            if self._ram_write:
                pa = victim.tag << self._line_shift
                self._ram_write(pa, bytes(victim.data))

        # 从 RAM 加载整行 (或清零)
        line_addr = (addr >> self._line_shift) << self._line_shift
        if self._ram_read:
            line_data = self._ram_read(line_addr, self._line_size)
        else:
            line_data = b"\x00" * self._line_size

        victim.tag = tag
        victim.data[:] = line_data
        victim.valid = True
        victim.mesi = MESIState.EXCLUSIVE
        victim.mdid = self._current_mdid
        victim.last_access = self._clock

        # 现在写入数据
        for i, b in enumerate(data):
            victim.data[offset + i] = b
        victim.dirty = True
        victim.mesi = MESIState.MODIFIED

    def _pick_victim_in_set(self, way_entries: list[L2CacheLine]) -> int:
        """在一组内选择要逐出的路 (优先无效 → LRU)."""
        # 优先选无效的
        for i, e in enumerate(way_entries):
            if not e.valid:
                return i

        # 全部有效 — LRU
        oldest_idx = 0
        oldest_time = way_entries[0].last_access
        for i, e in enumerate(way_entries):
            if e.last_access < oldest_time:
                oldest_time = e.last_access
                oldest_idx = i
        return oldest_idx

    # ----------------------------------------------------------
    #  总线接口 — 供 Bus 调用
    # ----------------------------------------------------------

    def bus_read(self, addr: int, size: int) -> bytes:
        """Bus 读接口: L2 缓存读取, 未命中时透传 RAM."""
        return self.read(addr, size)

    def bus_write(self, addr: int, data: bytes) -> None:
        """Bus 写接口: 写入 L2 缓存, 同时维护一致性."""
        self.write(addr, data)

    def invalidate(self, addr: int) -> None:
        """使某个物理地址对应的缓存行失效 (其他 hart 写入导致)."""
        tag, set_index, offset = self._addr_fields(addr)
        way_entries = self._get_set_entries(set_index)
        for e in way_entries:
            if e.valid and e.tag == tag:
                # M 状态需回写
                if e.mesi == MESIState.MODIFIED and self._ram_write:
                    pa = e.tag << self._line_shift
                    self._ram_write(pa, bytes(e.data))
                e.valid = False
                e.mesi = MESIState.INVALID
                return

    def set_ram_backend(self, read_fn, write_fn) -> None:
        """注入 RAM 后端回调 (Bus 初始化时调用)."""
        self._ram_read = read_fn
        self._ram_write = write_fn

    @property
    def line_size(self) -> int:
        return self._line_size

    @property
    def num_sets(self) -> int:
        return self._num_sets

    @property
    def ways(self) -> int:
        return self._ways

    @property
    def entries(self) -> "list[L2CacheLine]":
        """返回所有 L2CacheLine 条目 (含无效条目)."""
        return self._entries  # type: ignore[return-type]
