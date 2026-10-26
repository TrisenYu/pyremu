#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""MmuViewOps — PMP/SATP/页表遍历显示.

依赖 DebuggerBase + MemoryOps._resolve_addr.
"""

from rich.table import Table

from pyremu.debug.attrs import SharedAttrs
from pyremu.debug.utils import fmt_size, hex_addr
from pyremu.memory.mmu import pte_flags_str, sv39_decompose_va
from pyremu.memory.pmp import (
    decode_napot,
    PMP_A_MASK,
    PMP_A_NA4,
    PMP_A_NAPOT,
    PMP_A_OFF,
    PMP_A_TOR,
    PMP_R,
    PMP_W,
    PMP_X,
)
from pyremu.utils.mask import mask64
from pyremu.utils.wrapper import seize_val_err

# 页表级别到该级叶子页在 VA 中保留的页内偏移掩码
_PT_OFFSET_MASKS = {1: 0x3FFF_FFFF, 2: 0x1F_FFFF, 3: 0xFFF}

# PMP 地址匹配模式到表格中的显示名
_PMP_MODE_NAMES = {
    PMP_A_OFF: "OFF",
    PMP_A_TOR: "TOR",
    PMP_A_NA4: "NA4",
    PMP_A_NAPOT: "NAPOT",
}


class MmuViewOps(SharedAttrs):
    """PMP、页表、SATP 显示."""

    # ----------------------------------------------------------
    #  PMP 配置解码
    # ----------------------------------------------------------

    @staticmethod
    def _decode_pmp_cfg_byte(cfg_val: int, entry_idx: int) -> int:
        """从 RV64 pmpcfgN 值中提取第 *entry_idx* 个条目的 8-bit 配置."""
        shift = (entry_idx & 0x7) * 8
        return (cfg_val >> shift) & 0xFF

    @staticmethod
    def _pmp_rwx_cell(cfg: int) -> str:
        """把权限位编成 RWX 单元格: 依序为 R、W、X, 未置位的位置写 ``-``."""
        return "".join(
            f"[{style}]{char}[/]" if cfg & mask else "[dim]-[/]"
            for mask, char, style in (
                (PMP_R, "r", "green"),
                (PMP_W, "w", "yellow"),
                (PMP_X, "x", "red"),
            )
        )

    # ----------------------------------------------------------
    #  PMP 命令
    # ----------------------------------------------------------

    def cmd_pmp(self) -> None:
        """显示所有 PMP 条目的保护地址范围与权限 (含原始寄存器值).

        RWX 列的三个字符依序对应 R、W、X, 未置位的位置写 ``-``; 锁定位可由 cfg
        列的原始字节直接读出.
        """
        h = self.hart
        num = h._pmp.num_entries if h._pmp else 0

        if num == 0:
            self._console.print("[dim]PMP num_entries=0[/]")
            return

        tbl = Table(
            title=f"Hart {self._hart_id} PMP({num})配置情况",
            border_style="blue",
        )
        tbl.add_column("#", style="cyan", justify="right")
        tbl.add_column("RWX", width=3)
        tbl.add_column("Mode", style="cyan", width=6)
        tbl.add_column("Base", style="green")
        tbl.add_column("End", style="yellow")
        tbl.add_column("Size")
        tbl.add_column("pmpaddr", style="dim yellow")
        tbl.add_column("cfg", style="dim cyan", width=5)

        active_count = 0
        for i in range(num):
            row, is_active = self._pmp_entry_row(h, i)
            active_count += is_active
            tbl.add_row(*row)

        self._console.print(tbl)
        if active_count == 0:
            self._console.print("  [dim]所有条目均未激活 (OFF)[/]")

    def _pmp_entry_row(self, h, i: int) -> tuple[tuple[str, ...], bool]:
        """算出第 i 个 PMP 条目的表格单元格; 第二个返回值表示该条目是否激活."""
        cfg_reg_idx = (i // 8) * 2
        reg_name = f"pmpcfg{cfg_reg_idx}"
        cfg_val = h.csrs[reg_name].val if reg_name in h.csrs else 0
        cfg = self._decode_pmp_cfg_byte(cfg_val, i)

        a_mode = cfg & PMP_A_MASK
        addr_name = f"pmpaddr{i}"
        addr_field = mask64(h.csrs[addr_name].val if addr_name in h.csrs else 0)

        base, size = self._pmp_range(h, i, a_mode, addr_field)
        base_s, end_s, size_s = self._pmp_range_cells(a_mode, base, size)
        cells = (
            str(i),
            self._pmp_rwx_cell(cfg),
            _PMP_MODE_NAMES.get(a_mode, f"?{a_mode >> 3}?"),
            base_s,
            end_s,
            size_s,
            f"0x{addr_field:016x}",
            f"0x{cfg:02x}",
        )
        return cells, a_mode != PMP_A_OFF

    @staticmethod
    def _pmp_range(h, i: int, a_mode: int, addr_field: int) -> tuple[int, int]:
        """按匹配模式算出条目的起始地址与长度; 模式为 OFF 时两者都是 0."""
        if a_mode == PMP_A_TOR:
            prev = mask64(
                h.csrs[f"pmpaddr{i - 1}"].val if i > 0 and f"pmpaddr{i - 1}" in h.csrs else 0
            )
            lo = 0 if i == 0 else (prev << 2)
            hi = addr_field << 2
            return lo, mask64(hi - lo)
        if a_mode == PMP_A_NA4:
            return addr_field << 2, 4
        if a_mode == PMP_A_NAPOT:
            return decode_napot(addr_field)
        return 0, 0

    @staticmethod
    def _pmp_range_cells(a_mode: int, base: int, size: int) -> tuple[str, str, str]:
        """按匹配模式与长度给出起始地址、结束地址与长度三列."""
        if a_mode == PMP_A_OFF:
            return "-", "-", "-"
        if size > 0:
            return (
                f"0x{base:016x}",
                f"0x{mask64(base + size):016x}",
                fmt_size(size),
            )
        return f"0x{base:016x}", "0x0000000000000000", "-"

    # ----------------------------------------------------------
    #  satp 命令
    # ----------------------------------------------------------

    def cmd_satp(self) -> None:
        """显示当前 satp 的 MODE / ASID / PPN 解码."""
        h = self.hart
        mode_names = {0: "Bare", 8: "Sv39", 9: "Sv48", 10: "Sv57"}
        v = h.satp_val
        mode = (v >> 60) & 0xF
        asid = (v >> 44) & 0xFFFF
        ppn = v & ((1 << 44) - 1)
        mn = mode_names.get(mode, f"未知({mode})")
        out = [
            f"satp = [bold]{hex_addr(v)}[/]",
            f"  MODE  = [cyan]{mode}[/] ([green]{mn}[/])",
            f"  ASID  = 0x{asid:04x} ({asid})",
            f"  PPN   = 0x{ppn:011x}",
        ]
        if mode != 0:
            out.append(f"  根页表 PA = [yellow]0x{(ppn << 12):016x}[/]")
        self._console.print("\n".join(out))

    @seize_val_err("无效 VA")
    def cmd_pt(self, arg: str | None = None) -> None:
        """页表遍历: 显示指定 VA 经 Sv39 三级页表逐级翻译的完整路径.

        用法: pt [va]
          va 接受 0x...、十进制、或 sepc/mepc 寄存器名. 默认取当前 sepc.
        """
        h = self.hart
        if h.mmu_mode == 0:
            self._warn("satp.MODE = Bare, 页表遍历不可用")
            return

        if arg is None or arg == "":
            if h.sepc_val != 0:
                arg = f"0x{h.sepc_val:x}"
            elif h.mepc_val != 0:
                arg = f"0x{h.mepc_val:x}"
            else:
                self._warn("没有可用的默认 VA, 请显式指定")
                return
        va_raw = self._resolve_addr(arg)
        if va_raw is None or va_raw < 0 or va_raw > 0xFFFF_FFFF_FFFF_FFFF:
            self._warn(f"无效 VA: {arg}")
            return
        va: int = va_raw

        v = h.satp_val
        root_ppn = v & ((1 << 44) - 1)
        root_pa = root_ppn << 12
        mode_s = (v >> 60) & 0xF
        mode_names = {8: "Sv39"}
        mn = mode_names.get(mode_s, f"MODE={mode_s}")

        vpn2, vpn1, vpn0, page_off = sv39_decompose_va(va)
        vpn2 <<= 3
        out: list[str] = [
            f"[bold]VA[/] [cyan]0x{va:016x}[/]",
            f"  VPN[2]={vpn2 >> 3:#05x}  VPN[1]={vpn1:#05x}  VPN[0]={vpn0:#05x}  "
            f"offset={page_off:#05x}",
            f"  satp = {mn}, root PPN=0x{root_ppn:09x} -> PA=0x{root_pa:016x}",
        ]

        indices = (vpn2 >> 3, vpn1, vpn0)
        next_pa = root_pa
        for level in (1, 2, 3):
            pte = self._read_pt_level(out, level, next_pa, indices[level - 1])
            if pte is None or not (pte & 1):
                self._console.print("\n".join(out))
                return
            if pte & 0xE or level == 3:
                ppn = ((pte >> 10) & 0xF_FFFF_FFFF) << 12
                final_pa = ppn | (va & _PT_OFFSET_MASKS[level])
                out.append(f"  最终 PA = [bold yellow]0x{final_pa:016x}[/]")
                self._console.print("\n".join(out))
                return
            next_pa = ((pte >> 10) & 0xF_FFFF_FFFF) << 12

    def _read_pt_level(
        self,
        out: list[str],
        level: int,
        table_pa: int,
        index: int,
    ) -> int | None:
        """读取一级页表项并追加输出; 该级页表不可读时返回 None."""
        raw = self._emu.bus.try_read(table_pa, 4096)
        if raw is None:
            if level == 1:
                out.append("[red]根页表不可读[/]")
            else:
                out.append(f"[red]L{level} 页表 (PA=0x{table_pa:016x}) 不可读[/]")
            return None
        pte = int.from_bytes(raw[index * 8 : index * 8 + 8], "little", signed=False)
        self._pt_append_level(out, level, index, pte)
        return pte

    def _pt_append_level(
        self,
        out: list[str],
        level: int,
        idx: int,
        pte: int,
    ) -> None:
        """向 *out* 列表追加一级页表遍历的格式化输出行."""
        flags = pte_flags_str(pte, is_leaf=bool(pte & 0xE))
        ppn = (pte >> 10) & 0xF_FFFF_FFFF
        pa = ppn << 12
        status = "[red]无效[/]" if not (pte & 1) else flags
        out.append(f"  L{level}[{idx:#05x}] = 0x{pte:016x}  PPN=0x{ppn:09x}  -> {status}")
        if pte & 1 and not (pte & 0xE) and level < 3:
            out.append(f"        └─ 下一级页表 PA = 0x{pa:016x}")
