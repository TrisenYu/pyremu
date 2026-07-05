#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""TlbCacheMixin — TLB/Cache 状态显示和刷新.

依赖 DebuggerBase.
"""

from collections import Counter

from rich.table import Table

from pyremu.debug._attrs import SharedMixinAttrs
from pyremu.debug.utils import check_rv64_addr
from pyremu.memory.l2cache import L2Cache
from pyremu.memory.tlb import decode_tlb_perm
from pyremu.utils.wrapper import seize_val_err


class TlbCacheMixin(SharedMixinAttrs):
    """TLB 和 L2 缓存显示."""

    _DEFAULT_TLB_LINES = 32
    _DEFAULT_CACHE_LINES = 64

    # ----------------------------------------------------------
    #  TLB 辅助
    # ----------------------------------------------------------

    @staticmethod
    def _tlb_page_size(level: int) -> str:
        """TLB level -> 页大小标签."""
        return {0: "4K", 1: "2M", 2: "1G"}.get(level, f"Lv{level}")

    # ----------------------------------------------------------
    #  TLB 显示
    # ----------------------------------------------------------

    def _show_tlb(
        self,
        name: str,
        tlb,
        start_entry: int = 0,
        end_entry: int | None = _DEFAULT_TLB_LINES,
    ) -> None:
        """显示单个 TLB 的有效条目及命中率.

        *start_entry* / *end_entry* 限制显示的条目索引范围 (左闭右开);
        *end_entry* 为 None 时显示全部.
        """
        total = tlb._hits + tlb._misses
        rate = tlb._hits / total if total > 0 else 0.0

        title = (
            f"[bold]{name}[/]: {len(tlb)}/{tlb.size} entries, "
            f"hits={tlb._hits}, misses={tlb._misses}, "
            f"rate=[{'green' if rate > 0.9 else 'yellow'}]{rate:.2%}[/]"
        )

        if len(tlb) == 0:
            hint = ""
            if total == 0 and self.hart.mmu_mode == 0:
                hint = "\n[dim]# MMU 处于 Bare 模式 — TLB 不会被填充[/]"
            self._console.print(f"{title}\n  (empty){hint}")
            return

        has_mdid = any(e.mdid != 0 for e in tlb.entries if e.valid)

        tbl = Table(title=title, border_style="blue")
        tbl.add_column("#", style="dim", justify="right")
        tbl.add_column("VPN", style="cyan")
        tbl.add_column("PPN", style="green")
        tbl.add_column("Perm", style="yellow")
        tbl.add_column("Size")
        if has_mdid:
            tbl.add_column("mdid", style="magenta")

        shown = 0
        total_valid = sum(1 for e in tlb.entries if e.valid)
        for i, e in enumerate(tlb.entries):
            if i < start_entry:
                continue
            if end_entry is not None and i >= end_entry:
                break
            if not e.valid:
                continue
            row = [
                str(i),
                f"0x{e.tag:09x}",
                f"0x{e.ppn:09x}",
                decode_tlb_perm(e.perm),
                self._tlb_page_size(e.level),
            ]
            if has_mdid:
                row.append(str(e.mdid))
            tbl.add_row(*row)
            shown += 1

        if (
            end_entry is not None
            and shown < total_valid
            and start_entry + shown < total_valid
        ):
            remaining = total_valid - start_entry - shown
            hint = (
                f"[dim]… 还有 {remaining} 条, "
                f"用 tlb {start_entry + shown}-{tlb.size} 查看更多[/]"
            )
        elif end_entry is not None and start_entry > 0 and shown == 0:
            hint = f"[dim]索引 {start_entry}-{end_entry} 无有效条目[/]"
        else:
            hint = ""
        self._console.print(tbl)
        if hint:
            self._console.print(hint)

    # ----------------------------------------------------------
    #  TLB 搜索
    # ----------------------------------------------------------

    def _tlb_search(self, vpn: int) -> None:
        """在 ITLB / DTLB 中查找指定 VPN 并输出结果."""
        h = self.hart
        self._console.print(f"TLB 查找 [bold]VPN=0x{vpn:09x}[/]:")
        for name, tlb in [("ITLB", h.itlb), ("DTLB", h.dtlb)]:
            found = False
            for e in tlb.entries:
                if not e.valid or e.tag != vpn:
                    continue
                mdid_str = f" mdid={e.mdid}" if e.mdid != 0 else ""
                self._console.print(
                    f"  [[cyan]{name}[/]] VPN=0x{e.tag:09x} -> PPN=0x{e.ppn:09x}  "
                    f"perm={decode_tlb_perm(e.perm)}  "
                    f"size={self._tlb_page_size(e.level)}{mdid_str}"
                )
                found = True
            if not found:
                self._console.print(f"  [[cyan]{name}[/]] [dim]未命中[/]")

    # ----------------------------------------------------------
    #  TLB 命令
    # ----------------------------------------------------------

    @seize_val_err("无效参数")
    def cmd_tlb(self, arg: str | None = None) -> None:
        """显示或查找 TLB 条目.

        tlb             — 预览: 头 32 条 ITLB + DTLB
        tlb -a          — 显示全部条目
        tlb <vpn>       — 查找指定 VPN (hex: 0x...)
        tlb <start>-<end> — 显示条目索引范围
        """
        h = self.hart
        if arg is None:
            self._show_tlb("ITLB", h.itlb)
            self._show_tlb("DTLB", h.dtlb)
            return
        if arg == "-a":
            self._show_tlb("ITLB", h.itlb, end_entry=None)
            self._show_tlb("DTLB", h.dtlb, end_entry=None)
            return
        if "-" in arg and not arg.startswith("-"):
            st_str, _, ed_str = arg.partition("-")
            st, ed = int(st_str, 0), int(ed_str, 0)
            if st < 0 or ed <= st:
                self._err(f"范围需满足 0 ≤ start < end, 得到 {st}-{ed}")
                return
            self._show_tlb("ITLB", h.itlb, start_entry=st, end_entry=ed)
            self._show_tlb("DTLB", h.dtlb, start_entry=st, end_entry=ed)
            return
        vpn = int(arg, 0)
        if not check_rv64_addr(vpn):
            self._err(f"VPN {vpn:#x} 超出 RV64 范围")
            return
        self._tlb_search(vpn)

    @seize_val_err("无效 VPN")
    def cmd_tlbflush(self, vpn_str: str | None = None) -> None:
        """刷新 TLB: tlbflush [vpn] (无参数 = 全部)."""
        h = self.hart
        if vpn_str is None:
            h.itlb.flush_all()
            h.dtlb.flush_all()
            self._console.print("[dim]已刷新全部 ITLB + DTLB[/]")
            return
        vpn = int(vpn_str, 0)
        if not check_rv64_addr(vpn):
            self._err(f"VPN {vpn:#x} 超出 RV64 范围")
            return
        h.itlb.flush(vpn)
        h.dtlb.flush(vpn)
        self._console.print(f"[dim]已刷新 ITLB + DTLB 中 VPN=0x{vpn:09x}[/]")

    # ----------------------------------------------------------
    #  缓存行格式化
    # ----------------------------------------------------------

    @staticmethod
    def _hexdump_bytes(data: bytes, indent: str = "") -> str:
        """字节数据 -> hexdump 多行字符串."""
        rows = []
        for off in range(0, len(data), 16):
            chunk = data[off : off + 16]
            hex_s = " ".join(f"{b:02x}" for b in chunk)
            asc_s = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            rows.append(f"{indent}{off:04x}  {hex_s:<48s}  |{asc_s}|")
        return "\n".join(rows)

    def _fmt_cache_line(self, e, set_idx: int, way_idx: int, full: bool) -> str:
        """格式化单条 L2 缓存行: 元数据 + 可选完整 hexdump."""
        meta = (
            f"  s={set_idx:3d} w={way_idx}  tag=0x{e.tag:09x}  "
            f"st={e.mesi.value:<1}  dirty={e.dirty!s:<5}  "
            f"last={e.last_access}"
        )
        if full:
            return meta + "\n" + self._hexdump_bytes(bytes(e.data), indent="    ")
        preview = bytes(e.data[:16]).hex(" ")
        return meta + f"  data[:16]={preview}"

    # ----------------------------------------------------------
    #  缓存命令
    # ----------------------------------------------------------

    def _parse_cache_args(
        self,
        cache_args: str | None,
        sets: int,
        ways: int,
    ) -> tuple[bool, int | None, int | None, int | None, int | None]:
        """解析 cache 命令参数.

        返回 (error, target_set, target_way, range_start, range_end).
        """
        if cache_args is None:
            return False, None, None, None, None
        parts = cache_args.split()
        first = parts[0]
        if "-" in first and not first.startswith("-"):
            st_str, _, ed_str = first.partition("-")
            st, ed = int(st_str, 0), int(ed_str, 0)
            if not (0 <= st <= ed < sets):
                self._err(f"set 范围需在 [0, {sets - 1}] 内")
                return True, None, None, None, None
            return False, None, None, st, ed
        target_set = int(first, 0)
        if not (0 <= target_set < sets):
            self._err(f"set 索引超出范围 [0, {sets - 1}]")
            return True, None, None, None, None
        target_way = None
        if len(parts) > 1:
            target_way = int(parts[1], 0)
            if not (0 <= target_way < ways):
                self._err(f"way 索引超出范围 [0, {ways - 1}]")
                return True, None, None, None, None
        return False, target_set, target_way, None, None

    @seize_val_err("set/way 索引需为整数")
    def cmd_cache(self, arg: str | None = None) -> None:
        """显示 L2 缓存状态.

        cache               — 概览 + 前 64 条 valid 行 (预览)
        cache <set>         — 指定 set 全部 valid 行 + 完整 hexdump
        cache <set> <way>   — 指定 set/way, 含完整 64 B hexdump
        cache <start>-<end> — set 范围, 预览模式
        """
        l2 = self._emu.bus.l2
        if not isinstance(l2, L2Cache):
            self._console.print("[dim]L2 缓存未启用[/]")
            return

        entries, ways, num_sets = l2.entries, l2.ways, l2.num_sets
        err, target_set, target_way, range_start, range_end = (
            self._parse_cache_args(arg, num_sets, ways)
        )
        if err:
            return

        mesi_counts: Counter[str] = Counter()
        valid_count, dirty_count = 0, 0
        for e in entries:
            if not e.valid:
                continue
            valid_count += 1
            mesi_counts[e.mesi.name] += 1
            dirty_count += 1 if e.dirty else 0

        header = (
            f"[bold]L2 Cache[/]: {valid_count}/{len(entries)} valid, "
            f"{dirty_count} dirty, "
            f"{ways}-way * {num_sets} sets, "
            f"line={l2.line_size} B, "
            f"hit_rate={l2.hit_rate:.3f}\n"
            "MESI: " + " ".join(
                f"{s}={mesi_counts.get(s, 0)}"
                for s in ("MODIFIED", "EXCLUSIVE", "SHARED", "INVALID")
            )
        )

        set_range: range | list[int] = range(num_sets)
        full_dump = False
        limit: int | None = self._DEFAULT_CACHE_LINES
        if range_start is not None and range_end is not None:
            set_range = range(range_start, range_end + 1)
            full_dump = False
            limit = None
        elif target_set is not None:
            set_range = [target_set]
            full_dump = True
            limit = None

        lines: list[str] = []
        for set_idx in set_range:
            for way_idx in range(ways):
                if target_way is not None and way_idx != target_way:
                    continue
                e = entries[set_idx * ways + way_idx]
                if not e.valid:
                    continue
                lines.append(
                    self._fmt_cache_line(e, set_idx, way_idx, full_dump)
                )
                if limit is not None and len(lines) >= limit:
                    break
            if limit is not None and len(lines) >= limit:
                break

        if limit is not None and len(lines) >= limit and valid_count > len(lines):
            lines.append(
                f"[dim]... 还有 {valid_count - len(lines)} 条 valid 行, "
                f"用 'cache <start>-<end>' 查看范围[/]"
            )

        if lines:
            self._console.print(header + "\n" + "\n".join(lines))
        else:
            self._console.print(header + "\n[dim](无 valid 行)[/]")
