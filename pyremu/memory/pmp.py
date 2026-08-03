#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
Physical Memory Protection (PMP) — RISC-V Privileged Spec §3.7.

PMP 在物理地址翻译完成后检查每次访存的合法性.
支持三种地址匹配模式: OFF (禁用), TOR (地址范围), NA4 (4 字节), NAPOT (2 的幂).

入口数可配置 (默认 16, 最大 64).
RV64 编码: pmpcfgN 覆盖 8 个条目, 仅偶数编号的 pmpcfg 可用.
"""

from __future__ import annotations

from array import array as _array
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from pyremu.configs_aux import cfg_bool
from pyremu._native import (
    native_available,
    pmp_check as _native_pmp_check,
)
from pyremu.utils.mask import mask64

# PMP 配置位 (每 8-bit 条目中的位偏移)
PMP_R = 0b0000_0001
PMP_W = 0b0000_0010
PMP_X = 0b0000_0100
PMP_A_MASK = 0b0001_1000  # A[1:0]
PMP_A_OFF = 0b0000_0000  # 禁用
PMP_A_TOR = 0b0000_1000  # Top of Range
PMP_A_NA4 = 0b0001_0000  # Naturally Aligned 4-byte
PMP_A_NAPOT = 0b0001_1000  # Naturally Aligned Power-of-Two
PMP_L = 0b1000_0000  # 锁定位

# RiscvMode 值 -> PMP 检查时的特权级逻辑:
# M=3 且 MPRV=0 -> 跳过 PMP; 否则按 MPP 特权级检查
_MODE_M = 3


def _decode_pmpcfg(cfg_val: int, entry_idx: int) -> int:
    """从 pmpcfgN 的值中提取第 *entry_idx* 个条目的 8-bit 配置."""
    shift = (entry_idx & 0x7) * 8
    return (cfg_val >> shift) & 0xFF


def decode_napot(pmpaddr_val: int) -> tuple[int, int]:
    """解码 NAPOT 格式的 pmpaddr -> (base, size).

    算法: 统计 pmpaddr 中从 LSB 开始的连续 1 的个数 k,
        则 size = 2^(k+3), base = (pmpaddr & ~mask_k) << 2.

    pmpaddr 全 1 时覆盖整个地址空间 (size=0, 作无穷大处理).
    """
    # 计算末尾连续 1 的个数
    val = pmpaddr_val & 0x003F_FFFF_FFFF_FFFF  # 54-bit PMP 地址域
    trailing = 0
    temp = val
    while temp & 1:
        trailing += 1
        temp >>= 1
        if trailing >= 54:
            break

    if trailing >= 54:
        # 全 1: 覆盖整个地址空间
        return 0, 0x1_0000_0000_0000_0000  # 2^64 (实际不限)
    if trailing == 0:
        # 无末尾 1: 8 字节区域
        return (val >> 0) << 2, 8
    size = 1 << (trailing + 3)
    mask = (1 << trailing) - 1  # trailing 个 1
    base = mask64(((val & ~mask) << 2))
    return base, size


def _decode_tor(
    pmpaddr_prev: int,
    pmpaddr_cur: int,
    entry_idx: int,
) -> tuple[int, int]:
    """TOR: 范围为 [prev << 2, cur << 2). 条目 0 的下界为 0."""
    lo = 0 if entry_idx == 0 else (pmpaddr_prev << 2)
    hi = pmpaddr_cur << 2
    return lo, hi - lo


def _addr_in_range(addr: int, size: int, base: int, rsize: int) -> bool:
    """检查 [addr, addr+size) 是否完全落在 [base, base+rsize) 内."""
    if rsize == 0:
        return False
    end = mask64((addr + size - 1))
    rend = mask64((base + rsize - 1))
    return addr >= base and end <= rend


def _check_perm(cfg: int, is_write: bool, is_execute: bool) -> bool:
    """检查单条目权限: R 必须置位; 写入需 W; 执行需 X."""
    if not (cfg & PMP_R):
        return False
    if is_write and not (cfg & PMP_W):
        return False
    if is_execute and not (cfg & PMP_X):
        return False
    return True


@dataclass
class PmpAccessInfo:
    """PMP 检查所需的全部访问信息.

    pa / size 描述访存操作的物理地址和大小;
    mode_val / mstatus_val 提供当前 hart 的特权级和状态;
    is_write / is_execute 区分读/写/执行;
    pmpsplit / mdid 提供 TEE 飞地 PMP 虚拟化参数.
    """

    pa: int
    size: int
    mode_val: int
    mstatus_val: int
    is_write: bool = False
    is_execute: bool = False
    pmpsplit: int = 0
    mdid: int = 0


class Pmp:
    """Physical Memory Protection 检查器.

    直接读取 hart CSRs, 每次 check() 调用时重新计算.
    支持 Rust native 加速路径 (零状态共享, 纯函数 FFI).
    """

    def __init__(
        self,
        csrs: MutableMapping[str, Any],
        num_entries: int = 16,
    ) -> None:
        self._csrs = csrs
        if num_entries < 0 or num_entries > 64:
            raise ValueError("num_entries must be 0..64")
        self._num_entries = num_entries

        # Rust 加速: 扁平化 PMP 条目缓存 (64 u8 cfg + 64 u64 addr)
        self._flat_cfg: bytearray = bytearray(64)
        self._flat_addr: _array = _array("Q", [0] * 64)
        self._cache_dirty: bool = True
        self._use_native: bool = native_available()

    # ---- 属性 ----

    @property
    def num_entries(self) -> int:
        return self._num_entries

    # ---- 缓存管理 ----

    def invalidate_cache(self) -> None:
        """PMP CSR 写入后标记缓存失效."""
        self._cache_dirty = True

    def _rebuild_cache(self) -> None:
        """从 CSR dict 重建扁平化 PMP 数组 (entries -> flat)."""
        for i in range(self._num_entries):
            self._flat_cfg[i] = self._read_cfg(i)
            self._flat_addr[i] = self._read_addr(i)
        # 清除剩余条目
        for i in range(self._num_entries, 64):
            self._flat_cfg[i] = 0
            self._flat_addr[i] = 0
        self._cache_dirty = False

    def _sync_from_flat(self) -> None:
        """将 Rust batch 修改后的 flat 数组同步回 CSR entries (flat -> entries).

        逐字节写回 pmpcfg 寄存器: 先读 CSR 当前值, 替换目标字节,
        写回, 确保同一寄存器内其他条目不受影响.
        """
        # Phase 1: write config bytes (pmpcfg registers).
        for i in range(self._num_entries):
            cfg_reg = (i // 8) * 2
            reg_name = f"pmpcfg{cfg_reg}"
            if reg_name not in self._csrs:
                continue
            cfg_val = self._csrs[reg_name].val
            byte_idx = i & 0x7
            shift = byte_idx * 8
            byte_mask = 0xFF << shift
            cfg_val = (cfg_val & ~byte_mask) | (self._flat_cfg[i] << shift)
            self._csrs[reg_name].val = cfg_val
        # Phase 2: write address values (pmpaddr registers).
        for i in range(self._num_entries):
            reg_name = f"pmpaddr{i}"
            if reg_name in self._csrs:
                self._csrs[reg_name].val = self._flat_addr[i]
        self._cache_dirty = False

    # ---- 权限检查入口 ----

    def check(self, info: PmpAccessInfo) -> bool:
        """检查物理地址访问是否允许. 返回 True 表示通过."""
        pa, size = info.pa, info.size
        mode_val, mstatus_val = info.mode_val, info.mstatus_val
        is_write, is_execute = info.is_write, info.is_execute
        pmpsplit, mdid = info.pmpsplit, info.mdid

        # M-mode MPRV=0 — PMP 不检查 (RISC-V spec §3.7.1)
        if mode_val == _MODE_M and ((mstatus_val >> 17) & 1) == 0:
            return True

        # 飞地 split 超限预检 (在所有路径前, 避免无谓的缓存重建)
        if mdid != 0 and pmpsplit > 0 and pmpsplit >= self._num_entries:
            if cfg_bool("PYREMU_TRACE_PMP"):
                logger.debug(
                    f"[pmp] DENY enclave mdid={mdid} "
                    f"pmpsplit={pmpsplit} >= num_entries={self._num_entries}"
                )
            return False

        # Rust native 加速路径 (处理全部 PMP 逻辑: M-mode bypass, MPRV, pmpsplit, etc.)
        if self._use_native:
            if self._cache_dirty:
                self._rebuild_cache()
            return _native_pmp_check(
                bytes(self._flat_cfg),
                self._flat_addr,
                self._num_entries,
                pa,
                size,
                is_write=is_write,
                is_execute=is_execute,
                mode_val=mode_val,
                mstatus_val=mstatus_val,
                pmpsplit=pmpsplit,
                mdid=mdid,
            )

        # 纯 Python fallback
        return self._check_py(info)

    def _check_py(self, info: PmpAccessInfo) -> bool:
        """纯 Python PMP 检查 (native 库缺失时的 fallback)."""
        pa, size = info.pa, info.size
        mode_val, mstatus_val = info.mode_val, info.mstatus_val
        is_write, is_execute = info.is_write, info.is_execute
        pmpsplit, mdid = info.pmpsplit, info.mdid

        # 确定有效特权级
        eff_mode = mode_val
        if mode_val == _MODE_M:
            mprv = (mstatus_val >> 17) & 1
            if not mprv:
                return True
            mpp = (mstatus_val >> 11) & 3
            mpp_map = {0: 0, 1: 1, 3: 3}
            eff_mode = mpp_map.get(mpp, 3)

        if self._num_entries == 0:
            return False

        enclave_mode = mdid != 0

        # 按优先级遍历 PMP 条目 (低编号优先)
        for i in range(self._num_entries):
            if enclave_mode and pmpsplit > 0 and i < pmpsplit:
                continue

            cfg = self._read_cfg(i)
            if not (cfg & PMP_A_MASK):
                continue

            addr_field = self._read_addr(i)
            if not self._match(i, cfg, addr_field, pa, size):
                continue

            ok = _check_perm(cfg, is_write, is_execute)
            if not ok and cfg_bool("PYREMU_TRACE_PMP"):
                logger.debug(
                    f"[pmp] DENY pa={pa:#018x} size={size} "
                    f"eff_mode={eff_mode} is_write={is_write} "
                    f"matched_entry={i} cfg={cfg:#04x} addr_field={addr_field:#018x}"
                )
            return ok

        return eff_mode == _MODE_M

    # ---- 内部 ----

    def _read_cfg(self, idx: int) -> int:
        """读取第 *idx* 个 PMP 条目的 8-bit 配置."""
        # 每个 pmpcfg 存 4 个条目 (RV32), 或 8 个 (RV64) <-> idx // 4
        # RV64: 仅偶数编号的 pmpcfg 有效 (存储 8 条目)
        # 简化: 统一用 RV64 模型 — 按 8 条目编组
        cfg_reg = (idx // 8) * 2  # pmpcfg0, pmpcfg2, pmpcfg4, ...
        reg_name = f"pmpcfg{cfg_reg}"
        if reg_name not in self._csrs:
            return PMP_A_OFF
        cfg_val = self._csrs[reg_name].val
        return _decode_pmpcfg(cfg_val, idx & 0x7)

    def _read_addr(self, idx: int) -> int:
        """读取 pmpaddr[idx] 的值."""
        reg_name = f"pmpaddr{idx}"
        if reg_name not in self._csrs:
            return 0
        return mask64(self._csrs[reg_name].val)

    def _match(
        self,
        idx: int,
        cfg: int,
        addr_field: int,
        pa: int,
        size: int,
    ) -> bool:
        """检查地址是否匹配条目 *idx*."""
        a_mode = cfg & PMP_A_MASK
        if a_mode == PMP_A_TOR:
            prev = self._read_addr(idx - 1) if idx > 0 else 0
            base, rsize = _decode_tor(prev, addr_field, idx)
        elif a_mode == PMP_A_NA4:
            base = addr_field << 2
            rsize = 4
        elif a_mode == PMP_A_NAPOT:
            base, rsize = decode_napot(addr_field)
        else:
            return False  # OFF 或其他非法值

        return _addr_in_range(pa, size, base, rsize)
