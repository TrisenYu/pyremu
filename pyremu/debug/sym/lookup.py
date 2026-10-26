#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""SymLookupOps — 符号解析、内核符号加载、符号模块查找与段查找.

load_kernel_symbols 与 _resolve_sym_addr 与 translate_addr 定义在同一模块:
既有测试替换本模块的 parse_firmware 与 translate_addr, 只有同一模块内的
调用会用到替换后的名字.
"""

from __future__ import annotations

import bisect
from pathlib import Path

from pyremu import configs_gen
from pyremu.core.hart import RiscvMode
from pyremu.core.mem_check_aux import translate_addr
from pyremu.core.registers import csr_addr_from_name, gpr_idx_from_name
from pyremu.debug.attrs import SharedAttrs
from pyremu.debug.sym import model, registry
from pyremu.memory.mmu import sv39_canonical_va
from pyremu.utils.mask import MASK64, mask64
from pyremu.utils.parse_bin import FirmwareSegment, parse_firmware

# 最近前驱符号与查询地址的最大距离
PREDECESSOR_LIMIT = 0x10000
# 最近前驱查找时跳过的编译器生成名称前缀
LOCAL_NAME_PREFIXES = ("$", ".L")


def _symbol_in_ranges(
    ranges: list[tuple[int, int, str]],
    addr: int,
) -> str | None:
    """返回包含 *addr* 的范围符号名; 没有范围包含该地址时返回 None."""
    idx = bisect.bisect_right(ranges, addr, key=lambda r: r[0])
    if idx == 0:
        return None
    start, end, name = ranges[idx - 1]
    if start <= addr < end:
        return name
    return None


def _is_within_ranges(ranges: list[tuple[int, int, str]], addr: int) -> bool:
    """判定 *addr* 是否落在全部范围的覆盖区间内."""
    return ranges[0][0] <= addr < ranges[-1][1]


def _symbol_exact(symbols: dict[str, int], addr: int) -> str | None:
    for name, value in symbols.items():
        if value == addr:
            return name
    return None


def _symbol_predecessor(symbols: dict[str, int], addr: int) -> str | None:
    """返回 64 KiB 内地址不大于 *addr* 的最近符号."""
    best_name, best_dist = None, MASK64
    for name, value in symbols.items():
        if name.startswith(LOCAL_NAME_PREFIXES):
            continue
        if value <= addr and (addr - value) < best_dist:
            best_dist, best_name = addr - value, name
    return best_name if best_dist <= PREDECESSOR_LIMIT else None


class SymLookupOps(SharedAttrs):
    """符号解析、内核符号加载、符号模块查找与段查找."""

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

        优先使用地址范围做精确包含匹配, 即 start <= addr < end;
        无范围信息时回退到最近前驱符号, 相差 > 64 KiB 视为不匹配.
        地址落在范围表中已覆盖的区间内却没有命中任何范围时视为无符号,
        例如 NOTYPE 标签与下一个函数之间属于已覆盖区间.
        """
        if ranges:
            name = _symbol_in_ranges(ranges, addr)
            if name is not None:
                return name

        name = _symbol_exact(symbols, addr)
        if name is not None:
            return name

        if ranges and _is_within_ranges(ranges, addr):
            return None
        return _symbol_predecessor(symbols, addr)

    def load_kernel_symbols(self, path: str, base_pa: int | None = None) -> bool:
        """从 vmlinux ELF 加载调试符号 (不加载段到 RAM)."""
        if not Path(path).exists():
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
            base_pa = self._emu._cfg.ram_base + configs_gen.FW_RESERVED_MEM_SIZE

        min_vaddr = sym_image.entry_point
        load_offset = mask64((base_pa - min_vaddr))

        registry.remove_legacy_modules(self)
        self._sym_symbols = sym_image.symbols
        self._sym_ranges = sym_image.symbol_ranges
        self._sym_load_offset = load_offset
        self._sym_path = path

        self._sym_symbols_pa.clear()
        self._sym_ranges_pa.clear()
        self._pa_to_va = {}
        for name, va in self._sym_symbols.items():
            pa = mask64((va + load_offset))
            self._sym_symbols_pa[name] = pa
            self._pa_to_va[pa] = va
        for start, end, name in self._sym_ranges:
            pa_start = mask64((start + load_offset))
            pa_end = mask64((end + load_offset))
            self._sym_ranges_pa.append((pa_start, pa_end, name))
        self._sym_ranges_pa.sort(key=lambda r: r[0])

        registry.register_module(self, self._legacy_module(base_pa))
        self._console.print(
            f"[bold]sym[/] 已加载 [green]{Path(path).name}[/]"
            f" ({len(self._sym_symbols):,} 符号,"
            f" VA 0x{min_vaddr:x} -> PA 0x{base_pa:x},"
            f" load_offset=0x{load_offset:x})"
        )
        return True

    def _legacy_module(self, base_pa: int) -> model.SymMod:
        """为 load_kernel_symbols 的符号表登记模块条目.

        该条目只供 sym detail 与 sym rm 使用; 查找仍经 _sym_* 表完成,
        状态记为未校验.
        """
        values = list(self._sym_symbols.values())
        cover_start = min(values)
        cover_end = max(values) + 1
        if self._sym_ranges:
            cover_end = max(cover_end, *(end for _start, end, _name in self._sym_ranges))
        mod = model.SymMod(
            file_name=Path(self._sym_path).name if self._sym_path else "",
            path=self._sym_path or "",
            pa_base=base_pa,
            va_base=cover_start,
            cover_start=cover_start,
            cover_end=cover_end,
            sym_count=len(self._sym_symbols),
            is_verified=False,
            load_order=0,
            verify_epoch=self._instr_count,
        )
        registry.assign_load_order(self, mod)
        return mod

    def _resolve_sym_addr(self, va_like: int) -> int | None:
        """将类 VA 地址转换为运行时地址."""
        if not self._sym_symbols:
            return None
        hart = self.hart
        if hart.mmu_mode == 0:
            return mask64((va_like + self._sym_load_offset))
        ok, pa = translate_addr(hart, va_like, is_hart_access=False)
        return pa if ok else va_like

    def _sym_lookup_name(self, addr: int) -> str | None:
        """在内核符号表中按运行时地址查找符号名."""
        if not self._sym_symbols:
            return None
        hart = self.hart
        symbols, ranges = self._sym_symbols, self._sym_ranges
        if hart.mmu_mode == 0 and self._sym_symbols_pa:
            symbols, ranges = self._sym_symbols_pa, self._sym_ranges_pa
        return self._resolve_symbol(symbols, addr, ranges)

    # ----------------------------------------------------------
    #  符号模块查找
    # ----------------------------------------------------------

    def _module_name_at(self, runtime_addr: int) -> str | None:
        """在已加载的符号模块中查找 *runtime_addr* 的符号名.

        先按原值在模块的物理区间与虚拟区间中查找. 未命中而当前特权级不是 M 时,
        把 *runtime_addr* 当作虚拟地址经页表翻译为物理地址, 再按物理地址查找一次.
        只在 pa: 部署的模块没有虚拟区间, S 模式与 U 模式的虚拟地址只能经这次查找命中.
        """
        mods = registry.lookup_modules(self)
        hit = model.select_module(mods, runtime_addr)
        if hit is not None:
            return hit[2]
        hart = self.hart
        if hart.mmu_mode == 0 or hart.mode == RiscvMode.M:
            return None
        ok, pa = translate_addr(hart, runtime_addr, is_hart_access=False)
        if not ok:
            return None
        hit = model.select_module(mods, pa)
        return hit[2] if hit is not None else None

    def _module_runtime_addr_of_name(self, name: str) -> int | None:
        """在已加载的符号模块中按名称查找运行时地址."""
        hit = model.name_to_runtime_addr(registry.lookup_modules(self), name)
        if hit is None:
            return None
        pa, va = hit
        if pa is None:
            return va
        # 判定用 None 比较: 虚拟地址 0 是合法映射, 用真值判定会漏掉该反查项.
        if va is not None:
            self._pa_to_va[pa] = va
        return pa

    def _resolve_any_symbol_name(self, runtime_addr: int) -> str | None:
        """在固件、符号模块与内核符号表中查找 *runtime_addr* 的符号名."""
        if self._image and self._image.symbols:
            fw_va = runtime_addr - self._load_offset
            fw_ranges = self._image.symbol_ranges
            name = self._resolve_symbol(self._image.symbols, fw_va, fw_ranges)
            if name:
                return name
        name = self._module_name_at(runtime_addr)
        if name:
            return name
        return self._sym_lookup_name(runtime_addr)

    # ----------------------------------------------------------
    #  段查找
    # ----------------------------------------------------------

    def _find_segment(self, addr: int) -> FirmwareSegment | None:
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
        # M-mode 始终使用 Bare 翻译 (RISC-V spec §3.7.1), 即使 satp.MODE≠0.
        # M-mode 下的所有地址均为物理地址, 不经过 Sv39 规范地址检查.
        if h.mmu_mode != 0 and h.mode != RiscvMode.M:
            canonical = sv39_canonical_va(addr)
            if canonical is None:
                return False
            # 核态 (S/D): 仅接受 VA[38]=1 的内核规范地址
            if h.mode.value >= 1:  # S=1, D=8
                return bool(addr & (1 << 38))
            # 用户态 (U): 仅接受 VA[38]=0
            return not bool(addr & (1 << 38))
        return self._emu.bus.is_valid_addr(addr)
