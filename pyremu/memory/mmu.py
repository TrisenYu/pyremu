#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 17:07:57
# Last modified at 2026/06/08 星期一

"""
RISC-V 内存管理单元 (MMU): 页表项 (PTE) 定义与 Sv39 页表遍历。

参考: RISC-V Privileged Specification §4.3-§4.5
"""

from enum import Enum
from typing import Callable

from pyremu._native import (
    native_available,
    pte_assemble_pa as _native_pte_assemble_pa,
    pte_parse as _native_pte_parse,
    sv39_decompose_va as _native_sv39_decompose_va,
)

from pyremu.utils.mask import mask64

# ============================================================
#  地址翻译模式 (satp.MODE)
# ============================================================

MemAccessMode = Enum(
    "MemAccessMode",
    (
        "None",  # 未启用地址翻译
        "SV32",  # 32 位 Sv32
        "SV39",  # 39 位 Sv39
        "SV48",  # 48 位 Sv48
    ),
)

PAGE_SHIFT = 12
PAGE_SIZE = 1 << PAGE_SHIFT  # 4 KiB


# ============================================================
#  PTE — 页表项 (64-bit)
# ============================================================
# Sv39/Sv48 PTE 布局 (bits):
#   [0]     V    — 有效位
#   [1]     R    — 可读
#   [2]     W    — 可写
#   [3]     X    — 可执行
#   [4]     U    — 用户模式可访问
#   [5]     G    — 全局映射
#   [6]     A    — 已访问 (由硬件自动置位)
#   [7]     D    — 已修改 (由硬件自动置位)
#   [9:8]   RSW  — 保留给 S 模式
#   [19:10] PPN0 — 物理页号 [19:10]
#   [28:20] PPN1 — 物理页号 [28:20]
#   [53:29] PPN2 — 物理页号 [53:29] (Sv39 仅用 [37:29])
#   [63:54]      — 保留

PTE_V = 1 << 0
PTE_R = 1 << 1
PTE_W = 1 << 2
PTE_X = 1 << 3
PTE_U = 1 << 4
PTE_G = 1 << 5
PTE_A = 1 << 6
PTE_D = 1 << 7

PTE_RSW = 0b11 << 8
PTE_PPN0 = 0x3FF << 10  # bits 19:10
PTE_PPN1 = 0x1FF << 20  # bits 28:20
PTE_PPN2 = 0x1FFFFFFF << 29  # bits 53:29 (Sv39 只用到 37:29)

# PPN 掩码 (按级数)
PTE_PPN_MASK = (1 << 44) - 1  # 44 位物理页号

# satp MODE 字段的值
SATP_MODE_BARE = 0
SATP_MODE_SV39 = 8
SATP_MODE_SV48 = 9


class PTE:
    """RISC-V 64 位页表项。

    封装原始整数值并提供按字段的读写以及常见判断 (是否叶节点、权限检查等).
    """

    __slots__ = ("raw",)

    def __init__(self, raw: int = 0) -> None:
        self.raw = mask64(raw)

    @classmethod
    def from_int(cls, val: int) -> "PTE":
        return cls(val)

    def to_int(self) -> int:
        return self.raw

    # -- 标志位 --

    @property
    def v(self) -> bool:
        return bool(self.raw & PTE_V)

    @v.setter
    def v(self, x: bool):
        self.raw = (self.raw & ~PTE_V) | (PTE_V if x else 0)

    def _set_flag(self, mask: int, x: bool):
        """通用标志位写入."""
        self.raw = (self.raw & ~mask) | (mask if x else 0)

    @property
    def r(self) -> bool:
        return bool(self.raw & PTE_R)

    @r.setter
    def r(self, x: bool):
        self._set_flag(PTE_R, x)

    @property
    def w(self) -> bool:
        return bool(self.raw & PTE_W)

    @w.setter
    def w(self, x: bool):
        self._set_flag(PTE_W, x)

    @property
    def x(self) -> bool:
        return bool(self.raw & PTE_X)

    @x.setter
    def x(self, x_val: bool):
        self._set_flag(PTE_X, x_val)

    @property
    def u(self) -> bool:
        return bool(self.raw & PTE_U)

    @u.setter
    def u(self, x: bool):
        self._set_flag(PTE_U, x)

    @property
    def g(self) -> bool:
        return bool(self.raw & PTE_G)

    @g.setter
    def g(self, x: bool):
        self._set_flag(PTE_G, x)

    @property
    def a(self) -> bool:
        return bool(self.raw & PTE_A)

    @a.setter
    def a(self, x: bool):
        self._set_flag(PTE_A, x)

    @property
    def d(self) -> bool:
        return bool(self.raw & PTE_D)

    @d.setter
    def d(self, x: bool):
        self._set_flag(PTE_D, x)

    # -- PPN 分解 --

    @property
    def ppn0(self) -> int:
        return (self.raw & PTE_PPN0) >> 10  # 10 bits

    @property
    def ppn1(self) -> int:
        return (self.raw & PTE_PPN1) >> 20  # 9 bits

    @property
    def ppn2(self) -> int:
        return (self.raw & PTE_PPN2) >> 29  # 25 bits (Sv39 仅低 9 位有效)

    @property
    def ppn(self) -> int:
        """组合 PPN[2:0] 为完整的 44 位物理页号.

        RISC-V Sv39 编码: PTE[53:10] = PPN[43:0] (44-bit contiguous).
        PPN[2]=PTE[53:29], PPN[1]=PTE[28:20], PPN[0]=PTE[19:10].
        复原: (PPN[2] << 19) | (PPN[1] << 10) | PPN[0].
        """
        return (self.ppn2 << 19) | (self.ppn1 << 10) | self.ppn0

    @ppn.setter
    def ppn(self, val: int):
        """将 44 位物理页号拆分写入 PPN 字段 (RISC-V 标准连续编码)."""
        val &= PTE_PPN_MASK
        self.raw = (
            (self.raw & ~(PTE_PPN0 | PTE_PPN1 | PTE_PPN2))
            | ((val & 0x3FF) << 10)
            | (((val >> 10) & 0x1FF) << 20)
            | (((val >> 19) & 0x1FFFFFFF) << 29)
        )

    # -- 叶节点判断 --

    def is_leaf(self) -> bool:
        """非叶 PTE: 仅 V 置位, R/W/X 均为 0 (指向下一级页表).

        叶 PTE: R 或 X 至少有一个置位 (也可同时有 W).
        """
        if not self.v:
            return False
        return self.r or self.x

    def is_ptr(self) -> bool:
        """是否为指向下一级页表的指针 (非叶, 非无效)."""
        return self.v and not self.r and not self.w and not self.x

    def is_valid(self) -> bool:
        return self.v

    # -- 权限检查辅助 --

    def check_perm(self, want_r: bool, want_w: bool, want_x: bool, is_user: bool) -> bool:
        """检查 PTE 是否满足所需的访问权限.

        规则:
        - 请求 R 或 W 时, PTE.R 必须置位
        - 请求 X 时, PTE.X 必须置位
        - 用户模式访问时 PTE.U 必须置位
        - 若未置 A (已访问), 触发 page fault 时应由软件设置
        """
        if not self.v:
            return False
        if want_r and not self.r:
            return False
        if want_w and not self.w:
            return False
        if want_x and not self.x:
            return False
        if is_user and not self.u:
            return False
        # 注: A/D 的检查通常由硬件自动完成并置位;
        # 简单实现中跳过 A/D 检查, 允许软件处理
        return True


# ============================================================
#  VPN 分解 (Sv39 / Sv48)
# ============================================================
# Sv39: VA[38:12] 共 27 位 -> vpn[2] = VA[38:30] (9b),
#                           vpn[1] = VA[29:21] (9b),
#                           vpn[0] = VA[20:12] (9b)
# Sv48: VA[47:12] 共 36 位 -> vpn[3] = VA[47:39],
#                           vpn[2] = VA[38:30],
#                           vpn[1] = VA[29:21],
#                           vpn[0] = VA[20:12]


_NATIVE_MMU = native_available()


def _sv39_vpn(va: int) -> tuple:
    """将 39 位虚拟地址分解为 (vpn2, vpn1, vpn0)."""
    if _NATIVE_MMU:
        v = _native_sv39_decompose_va(va)
        return v.vpn2, v.vpn1, v.vpn0
    return (
        (va >> 30) & 0x1FF,
        (va >> 21) & 0x1FF,
        (va >> 12) & 0x1FF,
    )


def _sv48_vpn(va: int) -> tuple:
    """将 48 位虚拟地址分解为 (vpn3, vpn2, vpn1, vpn0)."""
    vpn0 = (va >> 12) & 0x1FF
    vpn1 = (va >> 21) & 0x1FF
    vpn2 = (va >> 30) & 0x1FF
    vpn3 = (va >> 39) & 0x1FF
    return vpn3, vpn2, vpn1, vpn0


# ============================================================
#  Sv39 页表遍历
# ============================================================
# 根页表物理基址 = satp.PPN << PAGE_SHIFT (4 KiB 对齐)
# 每一级页表有 512 个 8 字节 PTE
# 索引 = 对应级的 VPN, PTE 地址 = base + vpn * 8


def _read_pte(pa: int, mem_read_phy: Callable[[int, int], bytes]) -> int:
    """Read a 64-bit PTE from physical memory, return raw int."""
    return int.from_bytes(mem_read_phy(pa, 8), "little", signed=False)


def sv39_walk(root_ppn: int, va: int, mem_read_phy: Callable[[int, int], bytes]) -> tuple:
    """Sv39 三级页表遍历.

    Args:
        root_ppn: satp.PPN 字段 (根页表物理页号).
        va: 39 位虚拟地址.
        mem_read_phy: 物理内存读取回调, 签名 (addr: int, size: int) -> bytes.

    Returns:
        (success: bool, ppn: int, perm_flags: int, page_size: int)
    """
    vpn = _sv39_vpn(va)  # (vpn2, vpn1, vpn0)

    # 一级页表: 基址 = root_ppn << PAGE_SHIFT, 索引 vpn[2]
    table_addr = mask64(root_ppn << PAGE_SHIFT)
    raw = _read_pte(table_addr + vpn[0] * 8, mem_read_phy)

    if _NATIVE_MMU:
        l1 = _native_pte_parse(raw)
        if not l1.is_ptr:
            return False, 0, 0, 0
    else:
        pte = PTE.from_int(raw)
        if not pte.v or pte.r or pte.x:
            return False, 0, 0, 0

    # 二级页表
    l1_ppn = l1.ppn if _NATIVE_MMU else pte.ppn
    table_addr = mask64(l1_ppn << PAGE_SHIFT)
    raw = _read_pte(table_addr + vpn[1] * 8, mem_read_phy)

    if _NATIVE_MMU:
        l2 = _native_pte_parse(raw)
        if not l2.v:
            return False, 0, 0, 0
        if l2.is_leaf:
            # 2 MiB 大页 — Rust 计算合并 PPN
            ppn = _native_pte_assemble_pa(l2.ppn, va, level=1) >> 12
            return True, ppn, l2.perm, 2 * 1024 * 1024
        if not l2.is_ptr:
            return False, 0, 0, 0
    else:
        pte = PTE.from_int(raw)
        if not pte.v:
            return False, 0, 0, 0
        if pte.r or pte.x:
            ppn = pte.ppn
            ppn = (ppn & 0xFFFF_FFFF_FFFF_FE00) | vpn[2]
            return True, ppn, _pte_perm_flags(pte), 2 * 1024 * 1024

    # 三级页表 (4 KiB 普通页)
    l2_ppn = l2.ppn if _NATIVE_MMU else pte.ppn
    table_addr = mask64(l2_ppn << PAGE_SHIFT)
    raw = _read_pte(table_addr + vpn[2] * 8, mem_read_phy)

    if _NATIVE_MMU:
        l3 = _native_pte_parse(raw)
        if not l3.v or not l3.is_leaf:
            return False, 0, 0, 0
        return True, l3.ppn, l3.perm, PAGE_SIZE

    pte = PTE.from_int(raw)
    if not pte.v or not (pte.r or pte.x):
        return False, 0, 0, 0
    return True, pte.ppn, _pte_perm_flags(pte), PAGE_SIZE


def _pte_perm_flags(pte: PTE) -> int:
    """提取 PTE 的权限标志位 (R|W|X|U)."""
    return pte.raw & (PTE_R | PTE_W | PTE_X | PTE_U)


# ============================================================
#  通用地址翻译入口
# ============================================================


def translate_va(va: int, satp_val: int, mem_read_phy: Callable[[int, int], bytes]) -> tuple:
    """根据 satp 配置进行虚拟地址 -> 物理地址翻译.

    当前仅支持 Bare (直接等同物理地址) 和 Sv39.

    Args:
        va: 虚拟地址.
        satp_val: satp CSR 的原始值.
        mem_read_phy: 物理内存读取回调.

    Returns:
        (success: bool, pa: int, perm: int)
        perm 为 PTE 权限位 (R|W|X|U), Bare 模式下固定返回 0xF (全权限).
    """
    mode = satp_val >> 60

    if mode == SATP_MODE_BARE:
        # 无地址翻译: VA 即 PA; 全权限 (物理地址无页级保护)
        return True, mask64(va), 0xF

    if mode == SATP_MODE_SV39:
        root_ppn = satp_val & ((1 << 44) - 1)
        ok, ppn, perm, _ = sv39_walk(root_ppn, va, mem_read_phy)
        if not ok:
            return False, 0, 0
        # 物理地址 = PPN << 12 | offset (VA[11:0])
        offset = va & (PAGE_SIZE - 1)
        pa = (ppn << PAGE_SHIFT) | offset
        return True, mask64(pa), perm

    # Sv48 等其他模式暂未实现
    return False, 0, 0


# ============================================================
#  PTE 标志位 -> 可读字符串
# ============================================================

# Sv39 PTE 标志位掩码
PTE_V = 1 << 0
PTE_R = 1 << 1
PTE_W = 1 << 2
PTE_X = 1 << 3
PTE_U = 1 << 4
PTE_G = 1 << 5
PTE_A = 1 << 6
PTE_D = 1 << 7

_FLAG_NAMES: list[tuple[int, str, bool]] = [
    # (mask, name, leaf_only) — leaf_only=True 仅对叶子 PTE 显示
    (PTE_V, "V", False),
    (PTE_R, "R", True),
    (PTE_W, "W", True),
    (PTE_X, "X", True),
    (PTE_U, "U", True),
    (PTE_G, "G", True),
    (PTE_A, "A", True),
    (PTE_D, "D", True),
]


def pte_flags_str(pte_val: int, *, is_leaf: bool = False) -> str:
    """将 Sv39 PTE 值渲染为权限标志字符串 (纯文本, 无 Rich 标记).

    Args:
        pte_val: 8 字节 PTE 的整数值.
        is_leaf: True 时额外标出叶子页特有标志位 (R/W/X/U/G/A/D).

    Returns:
        以空格分隔的标志名, 如 ``"V R W X A D"``. 无效时返回 ``"-"``.
    """
    parts: list[str] = []
    for mask, name, leaf_only in _FLAG_NAMES:
        if leaf_only and not is_leaf:
            continue
        if pte_val & mask:
            parts.append(name)
    return " ".join(parts) if parts else "-"


def sv39_decompose_va(va: int) -> tuple[int, int, int, int]:
    """将 39 位虚拟地址分解为 (vpn2, vpn1, vpn0, page_offset).

    可用于调试器页表遍历展示.
    """
    vpn2 = (va >> 30) & 0x1FF
    vpn1 = (va >> 21) & 0x1FF
    vpn0 = (va >> 12) & 0x1FF
    offset = va & 0xFFF
    return vpn2, vpn1, vpn0, offset


# ============================================================
#  satp / VA 辅助
# ============================================================


def satp_root_ppn(satp_val: int) -> int:
    """从 satp CSR 值中提取根页表物理页号 (44-bit PPN)."""
    return satp_val & ((1 << 44) - 1)


def sv39_canonical_va(va: int) -> int | None:
    """验证 Sv39 虚拟地址的规范形式.

    RISC-V Sv39 要求 VA[63:39] 全部等于 VA[38] (符号扩展).
    若 *va* 满足此要求, 返回规范形式; 否则返回 None.
    """
    sign_bit = (va >> 38) & 1
    if sign_bit:
        canonical = va | 0xFFFFFF80_00000000  # 高 25 位补 1
    else:
        canonical = va & 0x7F_FFFFFFFF  # 高 25 位清零
    return canonical if va == canonical else None

