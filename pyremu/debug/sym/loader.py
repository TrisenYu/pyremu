#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""符号文件的加载、部署区间校验与内容校验."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyremu.core.mem_check_aux import translate_addr
from pyremu.debug.exception import DeployError, ElfFormatError
from pyremu.debug.sym import elf_idx, model
from pyremu.utils.mask import mask64
from pyremu.utils.parse_bin import ELF64_SYM_SIZE

# 校验样本的最大字节数
SAMPLE_MAX_BYTES = 256
# 校验样本的最小字节数
SAMPLE_MIN_BYTES = 4


@dataclass
class _Sample:
    """校验样本在文件与内存中的位置."""

    cover_start: int
    cover_end: int
    anchor: int
    file_offset: int
    data: bytes
    runtime_pa: int = 0


def load_symbol_module(
    dbg,
    path: str,
    space: str | None,
    base: int | None,
) -> "model.SymMod | None":
    """解析 *path* 指向的 ELF, 按 *space* 与 *base* 部署并校验, 通过后返回模块."""
    try:
        mod = _build_module(dbg, path, space, base)
    except DeployError as exc:
        dbg._err(str(exc))
        return None
    return mod


def verify_module(dbg, mod: "model.SymMod") -> bool:
    """按模块记录的校验样本重新读取内存并比较."""
    if mod.index is None or mod.pa_base is None:
        return True
    sample = _Sample(
        cover_start=mod.cover_start,
        cover_end=mod.cover_end,
        anchor=mod.anchor_link,
        file_offset=mod.sample_offset,
        data=mod.sample_data,
        runtime_pa=mask64(mod.pa_base + (mod.anchor_link - mod.cover_start)),
    )
    is_same, _mis_offset = _compare_sample(dbg, sample)
    return is_same


def _base_text(mod: "model.SymMod") -> str:
    parts: list[str] = []
    if mod.pa_base is not None:
        parts.append(f"PA 0x{mod.pa_base:x}")
    if mod.va_base is not None:
        parts.append(f"VA 0x{mod.va_base:x}")
    return ", ".join(parts)


def _build_module(
    dbg,
    path: str,
    space: str | None,
    base: int | None,
) -> "model.SymMod":
    elf = _open_elf(path)
    index = _build_index(dbg, elf, path)
    try:
        mod = _deploy_and_verify(dbg, path, index, space, base)
    except DeployError:
        index.close()
        raise
    return mod


def _open_elf(path: str) -> "elf_idx.ElfFile":
    try:
        return elf_idx.ElfFile(path)
    except OSError as exc:
        raise DeployError(f"无法读取符号文件 {path}: {exc.strerror or exc}") from exc
    except ElfFormatError as exc:
        raise DeployError(f"无法加载符号文件 {path}: {exc}") from exc


def _build_index(
    dbg,
    elf: "elf_idx.ElfFile",
    path: str,
) -> "elf_idx.ElfSymbolIndex":
    try:
        index = _create_index(elf, path)
    except ElfFormatError as exc:
        elf.close()
        raise DeployError(f"无法加载符号文件 {path}: {exc}") from exc
    except DeployError:
        elf.close()
        raise

    if index.entry_count >= elf_idx.WARN_SYMBOL_ENTRIES:
        address_mb = _index_mb(
            index.entry_count,
            elf_idx.ADDRESS_INDEX_BYTES_PER_SYMBOL,
        )
        name_mb = _index_mb(index.entry_count, elf_idx.NAME_INDEX_BYTES_PER_SYMBOL)
        dbg._warn(
            f"符号表条目 {index.entry_count:,} 条, 地址索引约 {address_mb} MB,"
            f" 名称索引建立后另加约 {name_mb} MB;"
            f" 不再需要的模块可用 sym rm 移除"
        )
    return index


def _create_index(
    elf: "elf_idx.ElfFile",
    path: str,
) -> "elf_idx.ElfSymbolIndex":
    found = elf.find_symbol_table()
    if found is None:
        raise DeployError(f"符号文件中未找到 .symtab 或 .dynsym: {path}")
    symtab, strtab = found
    entry_count = symtab.size // ELF64_SYM_SIZE
    if entry_count > elf_idx.MAX_SYMBOL_ENTRIES:
        address_mb = _index_mb(
            entry_count,
            elf_idx.ADDRESS_INDEX_BYTES_PER_SYMBOL,
        )
        name_mb = _index_mb(entry_count, elf_idx.NAME_INDEX_BYTES_PER_SYMBOL)
        raise DeployError(
            f"符号表条目 {entry_count:,} 条, 超过上限"
            f" {elf_idx.MAX_SYMBOL_ENTRIES:,}; 地址索引约 {address_mb} MB,"
            f" 名称索引另加约 {name_mb} MB"
        )
    return elf_idx.ElfSymbolIndex(elf, symtab, strtab)


def _index_mb(entry_count: int, bytes_per_symbol: int) -> int:
    return entry_count * bytes_per_symbol // (1024 * 1024)


def _deploy_and_verify(
    dbg,
    path: str,
    index: "elf_idx.ElfSymbolIndex",
    space: str | None,
    base: int | None,
) -> "model.SymMod":
    sample = _prepare_sample(index.elf, index, path)
    pa_base, va_base, sample.runtime_pa = _resolve_bases(dbg, sample, space, base)

    is_same, mis_offset = _compare_sample(dbg, sample)
    if not is_same:
        raise DeployError(_mismatch_text(dbg, path, sample, mis_offset))

    return model.SymMod(
        file_name=Path(path).name,
        path=path,
        pa_base=pa_base,
        va_base=va_base,
        cover_start=sample.cover_start,
        cover_end=sample.cover_end,
        sym_count=index.sym_count,
        is_verified=True,
        load_order=0,
        verify_epoch=dbg._instr_count,
        index=index,
        anchor_link=sample.anchor,
        sample_offset=sample.file_offset,
        sample_len=len(sample.data),
        sample_data=sample.data,
    )


def _prepare_sample(
    elf: "elf_idx.ElfFile",
    index: "elf_idx.ElfSymbolIndex",
    path: str,
) -> _Sample:
    """确定覆盖范围、校验样本的起点与样本内容."""
    cover = _coverage(elf, index)
    if cover is None:
        raise DeployError(f"符号文件中没有可用的地址范围: {path}")
    cover_start, cover_end = cover

    anchor = _find_anchor(elf, cover_start, cover_end)
    if anchor is None:
        raise DeployError(f"无法在文件中定位校验样本的起点: {path}")

    file_offset = _locate_sample(elf, anchor)
    if file_offset is None:
        raise DeployError(f"校验样本的起点不在任何可读的节或段内: {path}")

    remaining = elf.file_size - file_offset
    if remaining < SAMPLE_MIN_BYTES:
        raise DeployError(f"校验样本不足 {SAMPLE_MIN_BYTES} 字节: {path}")
    sample_len = min(SAMPLE_MAX_BYTES, remaining)
    data = elf.pread(file_offset, sample_len)
    if len(data) < sample_len:
        raise DeployError(f"读取校验样本失败: {path}")
    return _Sample(cover_start, cover_end, anchor, file_offset, data)


def _coverage(
    elf: "elf_idx.ElfFile",
    index: "elf_idx.ElfSymbolIndex",
) -> tuple[int, int] | None:
    alloc = [s for s in elf.sections if s.flags & elf_idx.SHF_ALLOC and s.size > 0]
    if alloc:
        start = min(s.addr for s in alloc)
        end = max(s.addr + s.size for s in alloc)
        return start, end
    if index.value_min is None:
        return None
    return index.value_min, index.value_max_end


def _find_anchor(elf: "elf_idx.ElfFile", cover_start: int, cover_end: int) -> int | None:
    entry = elf.entry
    if cover_start <= entry < cover_end and _locate_sample(elf, entry) is not None:
        return entry
    exec_start = _lowest_exec_start(elf)
    if exec_start is None or not (cover_start <= exec_start < cover_end):
        return None
    return exec_start


def _lowest_exec_start(elf: "elf_idx.ElfFile") -> int | None:
    starts = [
        s.addr
        for s in elf.sections
        if s.size > 0 and s.flags & elf_idx.SHF_ALLOC and s.flags & elf_idx.SHF_EXECINSTR
    ]
    return min(starts) if starts else None


def _locate_sample(elf: "elf_idx.ElfFile", addr: int) -> int | None:
    for sec in elf.sections:
        if sec.sh_type == elf_idx.SHT_NOBITS or sec.size == 0:
            continue
        if sec.addr <= addr < sec.addr + sec.size:
            return sec.offset + (addr - sec.addr)
    for prog in elf.programs:
        if prog.p_type != elf_idx.PT_LOAD:
            continue
        if prog.vaddr <= addr < prog.vaddr + prog.filesz:
            return prog.offset + (addr - prog.vaddr)
    return None


def _resolve_bases(
    dbg,
    sample: _Sample,
    space: str | None,
    base: int | None,
) -> tuple[int, int | None, int]:
    """确定物理基址、虚拟基址与校验样本的运行时物理地址."""
    span = sample.cover_end - sample.cover_start
    offset = sample.anchor - sample.cover_start
    bus = dbg._emu.bus
    ram_lo = bus.ram_base
    ram_hi = ram_lo + bus.ram_size

    if space == model.SPACE_PA:
        pa_base = mask64(base)
        if not _is_window_contained(pa_base, span, ram_lo, ram_hi):
            raise DeployError(_window_text(pa_base, span, ram_lo, ram_hi))
        return pa_base, None, mask64(pa_base + offset)

    if space == model.SPACE_VA:
        va_base = mask64(base)
        pa_first = _translate(dbg, va_base)
        pa_last = _translate(dbg, mask64(va_base + span - 1))
        if pa_first is None or pa_last is None:
            raise DeployError(f"虚拟地址区间 [0x{va_base:x}, 0x{va_base + span:x}) 无法翻译")
        if not (
            _is_in_window(pa_first, ram_lo, ram_hi) and _is_in_window(pa_last, ram_lo, ram_hi)
        ):
            raise DeployError(_window_text(pa_first, pa_last - pa_first + 1, ram_lo, ram_hi))
        pa_sample = _translate(dbg, mask64(va_base + offset))
        if pa_sample is None:
            raise DeployError(f"校验样本的虚拟地址 0x{va_base + offset:x} 无法翻译")
        return mask64(pa_sample - offset), va_base, pa_sample

    va_base = sample.cover_start
    pa_sample = _translate(dbg, mask64(va_base + offset))
    if pa_sample is None:
        raise DeployError(f"校验样本的虚拟地址 0x{va_base + offset:x} 无法翻译")
    pa_base = mask64(pa_sample - offset)
    if not _is_window_contained(pa_base, span, ram_lo, ram_hi):
        raise DeployError(_window_text(pa_base, span, ram_lo, ram_hi))
    return pa_base, va_base, pa_sample


def _translate(dbg, va: int) -> int | None:
    ok, pa = translate_addr(dbg.hart, va, is_hart_access=False)
    return mask64(pa) if ok else None


def _is_in_window(addr: int, ram_lo: int, ram_hi: int) -> bool:
    return ram_lo <= addr < ram_hi


def _is_window_contained(base: int, size: int, ram_lo: int, ram_hi: int) -> bool:
    return ram_lo <= base and base + size <= ram_hi


def _window_text(base: int, size: int, ram_lo: int, ram_hi: int) -> str:
    return (
        f"部署区间 [0x{base:x}, 0x{base + size:x}) 超出 RAM 窗口 [0x{ram_lo:x}, 0x{ram_hi:x})"
    )


def _compare_sample(dbg, sample: _Sample) -> tuple[bool, int]:
    """比较内存与文件中的样本, 返回 (是否一致, 首个不一致字节的偏移)."""
    bus = dbg._emu.bus
    if not bus.is_ram_addr(sample.runtime_pa):
        return False, 0
    data = bus.try_read(sample.runtime_pa, len(sample.data))
    if data == sample.data:
        return True, -1
    limit = min(len(data), len(sample.data))
    for i in range(limit):
        if data[i] != sample.data[i]:
            return False, i
    return False, limit


def _mismatch_text(dbg, path: str, sample: _Sample, mis_offset: int) -> str:
    text = (
        f"校验不一致: {path} 的样本起点位于文件偏移 0x{sample.file_offset:x}"
        f" (链接地址 0x{sample.anchor:x}, 运行时地址 0x{sample.runtime_pa:x});"
        f" 首个不一致字节的偏移为 {mis_offset}"
    )
    if mis_offset < len(sample.data):
        text += f", 文件中的字节为 0x{sample.data[mis_offset]:02x}"
    data = dbg._emu.bus.try_read(sample.runtime_pa + mis_offset, 1)
    if data:
        text += f", 内存中的字节为 0x{data[0]:02x}"
    return text
