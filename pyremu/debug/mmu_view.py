#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""MmuViewMixin — PMP/SATP/页表遍历显示.

依赖 DebuggerBase + MemoryMixin._resolve_addr.
"""

from rich.table import Table

from pyremu.debug._attrs import SharedMixinAttrs
from pyremu.debug.utils import fmt_size, hex_addr
from pyremu.memory.mmu import pte_flags_str, sv39_decompose_va
from pyremu.utils.mask import mask64
from pyremu.memory.pmp import (
    PMP_A_MASK,
    PMP_A_NA4,
    PMP_A_NAPOT,
    PMP_A_OFF,
    PMP_A_TOR,
    PMP_L,
    PMP_R,
    PMP_W,
    PMP_X,
    decode_napot,
)
from pyremu.utils.wrapper import seize_val_err


class MmuViewMixin(SharedMixinAttrs):
    """PMP、页表、SATP 显示."""

    # ----------------------------------------------------------
    #  PMP 配置解码
    # ----------------------------------------------------------

    @staticmethod
    def _decode_pmp_cfg_byte(cfg_val: int, entry_idx: int) -> int:
        """从 RV64 pmpcfgN 值中提取第 *entry_idx* 个条目的 8-bit 配置."""
        shift = (entry_idx & 0x7) * 8
        return (cfg_val >> shift) & 0xFF

    # ----------------------------------------------------------
    #  PMP 命令
    # ----------------------------------------------------------

    def cmd_pmp(self) -> None:
        """显示所有 PMP 条目的保护地址范围与权限 (含原始寄存器值)."""
        h = self.hart
        num = h._pmp.num_entries if h._pmp is not None else 0

        if num == 0:
            self._console.print("[dim]PMP 未配置 (num_entries=0)[/]")
            return

        mode_names = {
            PMP_A_OFF: "OFF",
            PMP_A_TOR: "TOR",
            PMP_A_NA4: "NA4",
            PMP_A_NAPOT: "NAPOT",
        }

        tbl = Table(
            title=f"Hart {self._hart_id}  PMP 条目 ({num} total)",
            border_style="blue",
        )
        tbl.add_column("#", style="cyan", justify="right")
        tbl.add_column("L", style="magenta", width=2)
        tbl.add_column("R", style="green", width=2)
        tbl.add_column("W", style="yellow", width=2)
        tbl.add_column("X", style="red", width=2)
        tbl.add_column("Mode", style="cyan", width=6)
        tbl.add_column("Base", style="green")
        tbl.add_column("End", style="yellow")
        tbl.add_column("Size")
        tbl.add_column("pmpaddr", style="dim yellow")
        tbl.add_column("cfg", style="dim cyan", width=5)

        active_count = 0

        for i in range(num):
            cfg_reg_idx = (i // 8) * 2
            reg_name = f"pmpcfg{cfg_reg_idx}"
            cfg_val = h.csrs[reg_name].val if reg_name in h.csrs else 0
            cfg = self._decode_pmp_cfg_byte(cfg_val, i)

            a_mode = cfg & PMP_A_MASK
            addr_name = f"pmpaddr{i}"
            addr_field = mask64(
                h.csrs[addr_name].val if addr_name in h.csrs else 0
            )

            base, size = 0, 0
            if a_mode == PMP_A_TOR:
                prev = mask64(
                    h.csrs[f"pmpaddr{i - 1}"].val
                    if i > 0 and f"pmpaddr{i - 1}" in h.csrs
                    else 0
                )
                lo = 0 if i == 0 else (prev << 2)
                hi = addr_field << 2
                base, size = lo, mask64((hi - lo))
            elif a_mode == PMP_A_NA4:
                base = addr_field << 2
                size = 4
            elif a_mode == PMP_A_NAPOT:
                base, size = decode_napot(addr_field)

            locked = "l" if cfg & PMP_L else "-"
            r = "r" if cfg & PMP_R else "-"
            w = "w" if cfg & PMP_W else "-"
            x = "x" if cfg & PMP_X else "-"
            mode = mode_names.get(a_mode, f"?{a_mode >> 3}?")
            addr_s = f"0x{addr_field:016x}"
            cfg_s = f"0x{cfg:02x}"

            base_s, end_s, size_s = "-", "-", "-"
            if a_mode != PMP_A_OFF:
                active_count += 1
                base_s = f"0x{base:016x}"
                end_s = "0x0000000000000000"
                size_s = "-"
                if size > 0:
                    end_s = f"0x{mask64(base + size):016x}"
                    size_s = fmt_size(size)

            tbl.add_row(
                str(i), locked, r, w, x, mode,
                base_s, end_s, size_s, addr_s, cfg_s,
            )

        self._console.print(tbl)
        if active_count == 0:
            self._console.print("  [dim]所有条目均未激活 (OFF)[/]")

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

    # ----------------------------------------------------------
    #  页表遍历 (pt)
    # ----------------------------------------------------------

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

        out: list[str] = [
            f"[bold]VA[/] [cyan]0x{va:016x}[/]",
            f"  VPN[2]={vpn2:#05x}  VPN[1]={vpn1:#05x}  VPN[0]={vpn0:#05x}  "
            f"offset={page_off:#05x}",
            f"  satp = {mn}, root PPN=0x{root_ppn:09x} -> PA=0x{root_pa:016x}",
            "",
        ]

        # L1 (根表)
        l1_raw = self._emu.bus.try_read(root_pa, 4096)
        if l1_raw is None:
            out.append("[red]根页表不可读[/]")
            self._console.print("\n".join(out))
            return
        l1_pte = int.from_bytes(
            l1_raw[vpn2 * 8:vpn2 * 8 + 8], "little", signed=False
        )
        self._pt_append_level(out, 1, vpn2, l1_pte)
        if not (l1_pte & 1):
            self._console.print("\n".join(out))
            return
        if l1_pte & 0xE:
            l1_pa = ((l1_pte >> 10) & 0xF_FFFF_FFFF) << 12
            final_pa = l1_pa | (va & 0x3FFF_FFFF)
            out.append(f"  最终 PA = [bold yellow]0x{final_pa:016x}[/]")
            self._console.print("\n".join(out))
            return

        # L2
        l2_pa = ((l1_pte >> 10) & 0xF_FFFF_FFFF) << 12
        l2_raw = self._emu.bus.try_read(l2_pa, 4096)
        if l2_raw is None:
            out.append(f"[red]L2 页表 (PA=0x{l2_pa:016x}) 不可读[/]")
            self._console.print("\n".join(out))
            return
        l2_pte = int.from_bytes(
            l2_raw[vpn1 * 8:vpn1 * 8 + 8], "little", signed=False
        )
        self._pt_append_level(out, 2, vpn1, l2_pte)
        if not (l2_pte & 1):
            self._console.print("\n".join(out))
            return
        if l2_pte & 0xE:
            l2_pa_leaf = ((l2_pte >> 10) & 0xF_FFFF_FFFF) << 12
            final_pa = l2_pa_leaf | (va & 0x1F_FFFF)
            out.append(f"  最终 PA = [bold yellow]0x{final_pa:016x}[/]")
            self._console.print("\n".join(out))
            return

        # L3
        l3_pa = ((l2_pte >> 10) & 0xF_FFFF_FFFF) << 12
        l3_raw = self._emu.bus.try_read(l3_pa, 4096)
        if l3_raw is None:
            out.append(f"[red]L3 页表 (PA=0x{l3_pa:016x}) 不可读[/]")
            self._console.print("\n".join(out))
            return
        l3_pte = int.from_bytes(
            l3_raw[vpn0 * 8:vpn0 * 8 + 8], "little", signed=False
        )
        self._pt_append_level(out, 3, vpn0, l3_pte)
        if not (l3_pte & 1):
            self._console.print("\n".join(out))
            return

        final_pa = (((l3_pte >> 10) & 0xF_FFFF_FFFF) << 12) | page_off
        out.append(f"  最终 PA = [bold yellow]0x{final_pa:016x}[/]")
        self._console.print("\n".join(out))

    def _pt_append_level(
        self, out: list[str], level: int, idx: int, pte: int,
    ) -> None:
        """向 *out* 列表追加一级页表遍历的格式化输出行."""
        flags = pte_flags_str(pte, is_leaf=bool(pte & 0xE))
        ppn = (pte >> 10) & 0xF_FFFF_FFFF
        pa = ppn << 12
        status = "[red]无效[/]" if not (pte & 1) else flags
        out.append(
            f"  L{level}[{idx:#05x}] = 0x{pte:016x}"
            f"  PPN=0x{ppn:09x}"
            f"  -> {status}"
        )
        if pte & 1 and not (pte & 0xE) and level < 3:
            out.append(f"        └─ 下一级页表 PA = 0x{pa:016x}")
