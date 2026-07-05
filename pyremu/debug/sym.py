#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""SymbolMixin — 符号解析、内核符号加载、段查找."""

import bisect
from pathlib import Path as _Path

from rich.table import Table

from pyremu.core.mem_check_aux import translate_addr
from pyremu.core.registers import csr_addr_from_name, gpr_idx_from_name
from pyremu.debug._attrs import SharedMixinAttrs
from pyremu.debug.utils import group_order
from pyremu.memory.mmu import sv39_canonical_va
from pyremu.utils.parse_bin import FirmwareSegment, parse_firmware


class SymbolMixin(SharedMixinAttrs):
    """符号表和地址段查找.

    依赖 DebuggerBase 提供的:
      self._image, self._load_offset, self._console,
      self._emu._cfg.ram_base, self._emu.bus,
      self._sym_symbols, self._sym_symbols_pa,
      self._sym_ranges, self._sym_ranges_pa,
      self._sym_load_offset, self._sym_path
    """

    # ----------------------------------------------------------
    #  GPR / CSR 名称查找 (供 _resolve_addr 等使用)
    # ----------------------------------------------------------

    def _read_gpr_by_name(self, name: str) -> int:
        """按名称读取 GPR (例: x12, t0, a0, sp)."""
        idx = gpr_idx_from_name(name)
        return 0 if idx is None else self.hart.read_gpr(idx)

    def _read_csr_by_name(self, name: str) -> int:
        """按名称读取 CSR (例: mtvec, mstatus, mepc)."""
        addr = csr_addr_from_name(name)
        return 0 if addr is None else self.hart.read_csr(addr)

    # ----------------------------------------------------------
    #  符号解析
    # ----------------------------------------------------------

    @staticmethod
    def _resolve_symbol(
        symbols: dict[str, int],
        addr: int,
        ranges: list[tuple[int, int, str]] | None = None,
    ) -> str | None:
        """在符号表中查找包含 *addr* 的函数符号.

        优先使用地址范围做精确包含匹配 (start ≤ addr < end);
        无范围信息时回退到最近前驱符号 (相差 > 64 KiB 视为不匹配).
        """
        # 1) 范围查找: 二分搜索
        if ranges:
            idx = bisect.bisect_right(ranges, addr, key=lambda r: r[0])
            if idx > 0:
                start, end, name = ranges[idx - 1]
                if start <= addr < end:
                    return name

        # 2) 精确匹配
        for name, a in symbols.items():
            if a == addr:
                return name

        # 3) ranges 未命中 → 不猜测
        if ranges:
            return None

        # 4) 无 ranges: 最近前驱
        best_name, best_dist = None, 0xFFFF_FFFF_FFFF_FFFF
        for name, a in symbols.items():
            if name.startswith("$") or name.startswith(".L"):
                continue
            if a <= addr and (addr - a) < best_dist:
                best_dist, best_name = addr - a, name
        return best_name if best_dist <= 0x10000 else None

    def load_kernel_symbols(self, path: str, base_pa: int | None = None) -> bool:
        """从 vmlinux ELF 加载调试符号 (不加载段到 RAM)."""
        if not _Path(path).exists():
            self._err(f"符号文件不存在: {path}")
            return False

        sym_image = parse_firmware(path, base_addr=0)
        if sym_image is None or sym_image.format != "elf":
            self._err(f"无法解析符号文件 (需为 ELF): {path}")
            return False

        if not sym_image.symbols:
            self._warn(f"符号文件中未找到符号表: {path}")
            return False

        if base_pa is None:
            base_pa = self._emu._cfg.ram_base + 0x200000

        min_vaddr = sym_image.entry_point
        load_offset = (base_pa - min_vaddr) & 0xFFFF_FFFF_FFFF_FFFF

        self._sym_symbols = sym_image.symbols
        self._sym_ranges = sym_image.symbol_ranges
        self._sym_load_offset = load_offset
        self._sym_path = path

        self._sym_symbols_pa.clear()
        self._sym_ranges_pa.clear()
        for name, va in self._sym_symbols.items():
            pa = (va + load_offset) & 0xFFFF_FFFF_FFFF_FFFF
            self._sym_symbols_pa[name] = pa
        for start, end, name in self._sym_ranges:
            pa_start = (start + load_offset) & 0xFFFF_FFFF_FFFF_FFFF
            pa_end = (end + load_offset) & 0xFFFF_FFFF_FFFF_FFFF
            self._sym_ranges_pa.append((pa_start, pa_end, name))
        self._sym_ranges_pa.sort(key=lambda r: r[0])

        self._console.print(
            f"[bold]sym[/] 已加载 [green]{_Path(path).name}[/]"
            f" ({len(self._sym_symbols):,} 符号,"
            f" VA 0x{min_vaddr:x} -> PA 0x{base_pa:x},"
            f" load_offset=0x{load_offset:x})"
        )
        return True

    def _resolve_sym_addr(self, va_like: int) -> int | None:
        """将类 VA 地址转换为运行时地址."""
        if not self._sym_symbols:
            return None
        hart = self.hart
        if hart.mmu_mode == 0:
            return (va_like + self._sym_load_offset) & 0xFFFF_FFFF_FFFF_FFFF
        ok, pa = translate_addr(hart, va_like)
        return pa if ok else va_like

    def _sym_lookup_name(self, addr: int) -> str | None:
        """在内核符号表中按运行时地址查找符号名."""
        if not self._sym_symbols:
            return None
        hart = self.hart
        payload1, payload2 = self._sym_symbols, self._sym_ranges
        if hart.mmu_mode == 0 and self._sym_symbols_pa:
            payload1, payload2 = self._sym_symbols_pa, self._sym_ranges_pa
        return self._resolve_symbol(payload1, addr, payload2)

    def _resolve_any_symbol_name(self, runtime_addr: int) -> str | None:
        """在固件 + 内核符号表中查找 *runtime_addr* 的符号名."""
        if not (self._image and self._image.symbols):
            return self._sym_lookup_name(runtime_addr)
        fw_va = runtime_addr - self._load_offset
        fw_ranges = self._image.symbol_ranges
        name = self._resolve_symbol(self._image.symbols, fw_va, fw_ranges)
        if name is not None:
            return name
        return self._sym_lookup_name(runtime_addr)

    # ----------------------------------------------------------
    #  段查找
    # ----------------------------------------------------------

    def _find_segment(self, addr: int) -> "FirmwareSegment | None":
        """返回包含 *addr* 的固件段; 无精确匹配返回 None."""
        if self._image is None:
            return None
        for seg in self._image.segments:
            if seg.vaddr <= addr < seg.vaddr + seg.memsz:
                return seg
        return None

    # ----------------------------------------------------------
    #  代码地址校验
    # ----------------------------------------------------------

    def _is_valid_code_va(self, addr: int) -> bool:
        """检查 *addr* 是否为合法代码地址 (canonical VA 或有效 PA).

        Sv39 模式下, 用户态 (VA[38]=0) 和内核态 (VA[38]=1) 的规范地址
        互不交叉. 当前特权级在 S/M 时仅接受内核态地址, U 时仅接受用户态地址,
        避免将 blake2s G 宏等借用 x1 的临时数据值误判为合法返回地址.
        """
        h = self.hart
        if h.mmu_mode != 0:
            canonical = sv39_canonical_va(addr)
            if canonical is None:
                return False
            # 核态 (S/M/D): 仅接受 VA[38]=1 的内核规范地址
            if h.mode.value >= 1:  # S=1, M=3, D=8
                return bool(addr & (1 << 38))
            # 用户态 (U): 仅接受 VA[38]=0
            return not bool(addr & (1 << 38))
        return self._emu.bus.is_valid_addr(addr)

    # ----------------------------------------------------------
    #  符号命令
    # ----------------------------------------------------------

    def cmd_symbols(self, filter_str: str = "") -> None:
        """列出固件符号表."""
        if self._image is None:
            self._warn("无可用的符号表 (非 ELF 文件)")
            return

        syms = self._image.symbols
        if not syms:
            self._console.print("[dim](符号表为空)[/]")
            return

        if not filter_str:
            groups: dict[str, int] = {}
            for name in syms:
                key = name[0] if name else "?"
                groups[key] = groups.get(key, 0) + 1
            sorted_keys = sorted(groups.keys(), key=group_order)

            tbl = Table(
                title=f"符号分组 ({len(syms)} 项, {len(sorted_keys)} 组)",
                border_style="blue", expand=True,
            )
            tbl.add_column("Prefix", style="cyan")
            tbl.add_column("Count", style="yellow", justify="right")
            for key in sorted_keys:
                tbl.add_row(f"{key}...", str(groups[key]))
            self._console.print(tbl)
            self._console.print(
                "[dim]输入 sym <prefix> 查看具体符号 (如 sym a, sym f, sym _)[/]"
            )
            return

        entries = [
            (name, addr) for name, addr in syms.items()
            if name.lower().startswith(filter_str.lower())
        ]
        entries.sort(key=lambda x: x[1])

        if not entries:
            self._console.print(f"[dim]无以前缀 '{filter_str}' 开头的符号[/]")
            return

        _sym_subgroup_threshold = 50
        if len(entries) > _sym_subgroup_threshold:
            prefix_len = len(filter_str)
            subgroups: dict[str, int] = {}
            for name, _ in entries:
                next_key = name if len(name) <= prefix_len else name[:prefix_len + 1]
                subgroups[next_key] = subgroups.get(next_key, 0) + 1

            def _calc_order(x: str) -> tuple[int, str]:
                if not x:
                    return (5, x)
                c = x[-1].lower() if len(x) > len(filter_str) else x.lower()
                return group_order(c)

            sorted_keys = sorted(subgroups.keys(), key=_calc_order)
            tbl = Table(
                title=f"符号: '{filter_str}' ({len(entries)} 项, {len(sorted_keys)} 子组)",
                border_style="blue", expand=True,
            )
            tbl.add_column("Prefix", style="cyan")
            tbl.add_column("Count", style="yellow", justify="right")
            for key in sorted_keys:
                if not key:
                    continue
                suffix = key[prefix_len:]
                tbl.add_row(f"{filter_str}[bold]{suffix}[/]...", str(subgroups[key]))
            self._console.print(tbl)
            self._console.print(
                "[dim]输入 sym <prefix> 继续深入 (如 sym "
                + sorted_keys[0] + ", sym "
                + (sorted_keys[1] if len(sorted_keys) > 1 else sorted_keys[0])
                + ")[/]"
            )
            return

        tbl = Table(
            title=f"符号: '{filter_str}' ({len(entries)} 项)",
            border_style="blue", expand=True,
        )
        tbl.add_column("Address", style="yellow")
        tbl.add_column("Name", style="cyan")
        for name, addr in entries:
            tbl.add_row(f"0x{addr:016x}", name)
        self._console.print(tbl)

    def cmd_sym(self, args: str) -> None:
        """管理外部调试符号 (vmlinux 等).

        sym load <path> [base]  — 加载调试符号文件
        sym list [filt]         — 列出符号
        sym info                — 显示当前已加载的符号文件信息
        """
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts and parts[0] else ""
        rest = parts[1] if len(parts) > 1 else ""

        if sub == "load":
            load_parts = rest.split(maxsplit=1) if rest else []
            path = load_parts[0] if load_parts else ""
            base_str = load_parts[1] if len(load_parts) > 1 else ""
            if not path:
                self._err("用法: sym load <path> [base_addr]")
                return
            base_pa = int(base_str, 0) if base_str else None
            self.load_kernel_symbols(path, base_pa)
        elif sub == "info":
            if not self._sym_symbols:
                self._console.print("[dim]未加载外部调试符号[/]")
                return
            entry_va = min(self._sym_symbols.values())
            self._console.print(
                f"[bold]sym info[/]\n"
                f"  文件: [green]{self._sym_path}[/]\n"
                f"  符号数: {len(self._sym_symbols):,}\n"
                f"  入口 VA: 0x{entry_va:x}\n"
                f"  load_offset: 0x{self._sym_load_offset:x}"
            )
        elif sub == "list" or not sub:
            self.cmd_symbols(rest)
        else:
            self.cmd_symbols(args)
