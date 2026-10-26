#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""流式 ELF 符号索引.

只读取 ELF 头、节头表与程序头表; 符号表按块扫描, 每个符号只保留地址与序号;
符号名经 os.pread 按需读取. 名称到地址的索引在第一次按名称查找时建立.
"""

from __future__ import annotations

from array import array
import bisect
from dataclasses import dataclass
import os
import struct

from pyremu.debug.exception import ElfFormatError
from pyremu.utils.parse_bin import (
    ELF64_SYM_FMT,
    ELF64_SYM_SIZE,
    STB_LOCAL,
    STT_FUNC,
)

ELF_HEADER_SIZE = 64
SECTION_HEADER_SIZE = 64
PROGRAM_HEADER_SIZE = 56

SHT_NOBITS = 8
SHT_SYMTAB = 2
SHT_DYNSYM = 11

SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4

PT_LOAD = 1

# 符号表的扫描分块与名称读取的规模
SCAN_CHUNK_BYTES = 4 * 1024 * 1024
NAME_READ_LIMIT = 256
NAME_CACHE_SIZE = 4096

# 超过该条目数时跳过 LOCAL 的非函数符号
LOCAL_SYMBOL_SKIP_THRESHOLD = 500_000

# 索引规模的上限与提示阈值
WARN_SYMBOL_ENTRIES = 2_000_000
MAX_SYMBOL_ENTRIES = 16_000_000

# 地址索引每个符号占用的字节数: array('Q') 与 array('I') 各一项
ADDRESS_INDEX_BYTES_PER_SYMBOL = 12
# 名称索引每个符号占用的字节数: 单个 array('Q')
NAME_INDEX_BYTES_PER_SYMBOL = 8

_PREDECESSOR_LIMIT = 0x10000
_LOCAL_NAME_PREFIXES = ("$", ".L")


@dataclass
class ElfSection:
    """ELF 节头中与符号加载有关的字段."""

    index: int
    sh_type: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int
    entsize: int


@dataclass
class ElfProgram:
    """ELF 程序头中与校验样本定位有关的字段."""

    p_type: int
    offset: int
    vaddr: int
    filesz: int
    memsz: int


class ElfFile:
    """已解析头部表的 ELF 文件, 按偏移读取任意区间."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fd = os.open(path, os.O_RDONLY)
        try:
            self._load_headers()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def pread(self, offset: int, size: int) -> bytes:
        if size <= 0:
            return b""
        return os.pread(self._fd, size, offset)

    def _load_headers(self) -> None:
        self.file_size = os.fstat(self._fd).st_size
        head = self.pread(0, ELF_HEADER_SIZE)
        if len(head) < ELF_HEADER_SIZE:
            raise ElfFormatError("文件长度不足一个 ELF 头")
        if head[:4] != b"\x7fELF":
            raise ElfFormatError("不是 ELF 文件")
        if head[4] != 2:
            raise ElfFormatError("不是 64 位 ELF")
        if head[5] != 1:
            raise ElfFormatError("不是小端 ELF")

        self.entry = struct.unpack_from("<Q", head, 24)[0]
        phoff = struct.unpack_from("<Q", head, 32)[0]
        shoff = struct.unpack_from("<Q", head, 40)[0]
        phentsize = struct.unpack_from("<H", head, 54)[0]
        phnum = struct.unpack_from("<H", head, 56)[0]
        shentsize = struct.unpack_from("<H", head, 58)[0]
        shnum = struct.unpack_from("<H", head, 60)[0]

        if shnum == 0:
            raise ElfFormatError("没有节头表")
        if shentsize != SECTION_HEADER_SIZE:
            raise ElfFormatError(f"节头长度非法: {shentsize}")
        if shoff + shnum * SECTION_HEADER_SIZE > self.file_size:
            raise ElfFormatError("节头表越出文件")
        if phnum and phentsize != PROGRAM_HEADER_SIZE:
            raise ElfFormatError(f"程序头长度非法: {phentsize}")
        if phnum and phoff + phnum * PROGRAM_HEADER_SIZE > self.file_size:
            raise ElfFormatError("程序头表越出文件")

        self.sections = self._read_sections(shoff, shnum)
        self.programs = self._read_programs(phoff, phnum)

    def _read_sections(self, shoff: int, shnum: int) -> list[ElfSection]:
        data = self.pread(shoff, shnum * SECTION_HEADER_SIZE)
        sections: list[ElfSection] = []
        for i in range(shnum):
            off = i * SECTION_HEADER_SIZE
            sh_type = struct.unpack_from("<I", data, off + 4)[0]
            flags = struct.unpack_from("<Q", data, off + 8)[0]
            addr = struct.unpack_from("<Q", data, off + 16)[0]
            offset = struct.unpack_from("<Q", data, off + 24)[0]
            size = struct.unpack_from("<Q", data, off + 32)[0]
            link = struct.unpack_from("<I", data, off + 40)[0]
            entsize = struct.unpack_from("<Q", data, off + 56)[0]
            sections.append(
                ElfSection(
                    index=i,
                    sh_type=sh_type,
                    flags=flags,
                    addr=addr,
                    offset=offset,
                    size=size,
                    link=link,
                    entsize=entsize,
                )
            )
        return sections

    def _read_programs(self, phoff: int, phnum: int) -> list[ElfProgram]:
        if phnum == 0:
            return []
        data = self.pread(phoff, phnum * PROGRAM_HEADER_SIZE)
        programs: list[ElfProgram] = []
        for i in range(phnum):
            off = i * PROGRAM_HEADER_SIZE
            p_type = struct.unpack_from("<I", data, off)[0]
            offset = struct.unpack_from("<Q", data, off + 8)[0]
            vaddr = struct.unpack_from("<Q", data, off + 16)[0]
            filesz = struct.unpack_from("<Q", data, off + 32)[0]
            memsz = struct.unpack_from("<Q", data, off + 40)[0]
            programs.append(
                ElfProgram(
                    p_type=p_type,
                    offset=offset,
                    vaddr=vaddr,
                    filesz=filesz,
                    memsz=memsz,
                )
            )
        return programs

    def find_symbol_table(self) -> tuple[ElfSection, ElfSection] | None:
        """返回符号表节与其字符串表节; 优先 .symtab, 退回 .dynsym."""
        symtab = None
        for sec in self.sections:
            if sec.sh_type == SHT_SYMTAB:
                symtab = sec
                break
        if symtab is None:
            for sec in self.sections:
                if sec.sh_type == SHT_DYNSYM:
                    symtab = sec
                    break
        if symtab is None:
            return None
        if symtab.link >= len(self.sections):
            raise ElfFormatError(f"符号表的字符串表索引非法: {symtab.link}")
        if symtab.offset + symtab.size > self.file_size:
            raise ElfFormatError("符号表越出文件")
        strtab = self.sections[symtab.link]
        if strtab.offset + strtab.size > self.file_size:
            raise ElfFormatError("符号表的字符串表越出文件")
        return symtab, strtab


def _is_local_symbol_skipped(st_info: int, skip_local: bool) -> bool:
    """跳过 LOCAL 的非函数符号."""
    if not skip_local:
        return False
    return (st_info >> 4) == STB_LOCAL and (st_info & 0xF) != STT_FUNC


def _name_hash(raw: bytes) -> int:
    """FNV-1a 32 位散列, 供名称索引使用."""
    value = 0x811C9DC5
    for byte in raw:
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


class ElfSymbolIndex:
    """紧凑符号索引 — 地址到名称与名称到地址.

    地址索引由排序后的 st_value 与符号序号构成; 符号记录与名称按需读取.
    名称索引在第一次按名称查找时建立.
    """

    def __init__(
        self,
        elf: ElfFile,
        symtab: ElfSection,
        strtab: ElfSection,
    ) -> None:
        self._elf = elf
        self._symtab_offset = symtab.offset
        self._strtab_offset = strtab.offset
        self._strtab_size = strtab.size
        self.entry_count = symtab.size // ELF64_SYM_SIZE
        if self.entry_count == 0:
            raise ElfFormatError("符号表为空")
        if symtab.entsize not in (0, ELF64_SYM_SIZE):
            raise ElfFormatError(f"符号表条目长度非法: {symtab.entsize}")
        self._name_cache: dict[int, str] = {}
        self._name_index: array | None = None
        self.value_min: int | None = None
        self.value_max_end = 0
        self._build_address_index()

    @property
    def elf(self) -> ElfFile:
        return self._elf

    # ----------------------------------------------------------
    #  建立索引
    # ----------------------------------------------------------

    def _iter_symtab_chunks(self):
        """按块产出符号表数据, 每项为 (起始序号, 条目数, 字节)."""
        per_chunk = max(1, SCAN_CHUNK_BYTES // ELF64_SYM_SIZE)
        done = 0
        while done < self.entry_count:
            take = min(per_chunk, self.entry_count - done)
            data = self._elf.pread(
                self._symtab_offset + done * ELF64_SYM_SIZE,
                take * ELF64_SYM_SIZE,
            )
            if len(data) < take * ELF64_SYM_SIZE:
                raise ElfFormatError("符号表越出文件")
            yield done, take, data
            done += take

    def _iter_accepted_records(self, skip_local: bool):
        """产出通过过滤条件的 (符号序号, 名称偏移, 值, 大小)."""
        for done, take, data in self._iter_symtab_chunks():
            for k in range(take):
                values = self._read_record(data, k * ELF64_SYM_SIZE)
                if values is None:
                    continue
                st_name, st_info, st_value, st_size = values
                if _is_local_symbol_skipped(st_info, skip_local):
                    continue
                yield done + k, st_name, st_value, st_size

    def _build_address_index(self) -> None:
        skip_local = self.entry_count > LOCAL_SYMBOL_SKIP_THRESHOLD
        packed: list[int] = []
        for sym_index, _st_name, st_value, st_size in self._iter_accepted_records(skip_local):
            packed.append((st_value << 32) | sym_index)
            self._track_span(st_value, st_size)

        packed.sort()
        self._values = array("Q", (item >> 32 for item in packed))
        self._indices = array("I", (item & 0xFFFFFFFF for item in packed))
        self.sym_count = len(self._values)

    def _track_span(self, st_value: int, st_size: int) -> None:
        if self.value_min is None or st_value < self.value_min:
            self.value_min = st_value
        end = st_value + (st_size if st_size > 0 else 1)
        self.value_max_end = max(self.value_max_end, end)

    def _read_record(self, data: bytes, offset: int) -> tuple[int, int, int, int] | None:
        """从已读入的符号表数据中取出一条记录; 不满足过滤条件时返回 None."""
        st_name, st_info, _st_other, st_shndx, st_value, st_size = struct.unpack_from(
            ELF64_SYM_FMT, data, offset
        )
        if st_shndx == 0 or st_value == 0:
            return None
        if st_name == 0 or st_name >= self._strtab_size:
            return None
        return st_name, st_info, st_value, st_size

    # ----------------------------------------------------------
    #  符号记录与名称
    # ----------------------------------------------------------

    def _symbol_record(self, sym_index: int) -> tuple[int, int, int] | None:
        """读取一个符号的 (名称偏移, 值, 大小)."""
        data = self._elf.pread(
            self._symtab_offset + sym_index * ELF64_SYM_SIZE,
            ELF64_SYM_SIZE,
        )
        if len(data) < ELF64_SYM_SIZE:
            return None
        st_name, _info, _other, _shndx, st_value, st_size = struct.unpack_from(
            ELF64_SYM_FMT, data, 0
        )
        return st_name, st_value, st_size

    def _name_at(self, st_name: int) -> str | None:
        cached = self._name_cache.get(st_name)
        if cached is not None:
            return cached
        avail = self._strtab_size - st_name
        if avail <= 0:
            return None
        data = self._elf.pread(self._strtab_offset + st_name, min(NAME_READ_LIMIT, avail))
        end = data.find(b"\x00")
        if end <= 0:
            return None
        name = data[:end].decode("utf-8", errors="replace")
        if len(self._name_cache) >= NAME_CACHE_SIZE:
            self._name_cache.clear()
        self._name_cache[st_name] = name
        return name

    # ----------------------------------------------------------
    #  地址到名称
    # ----------------------------------------------------------

    def resolve_addr(self, addr: int) -> str | None:
        """在链接地址 *addr* 上解析符号名.

        取最近的 st_value 不大于 *addr* 的符号: 精确命中或落在其 st_size 之内时
        取该符号; 否则在 64 KiB 之内回退到最近前驱, 跳过 $ 与 .L 开头的名称.
        """
        pos = bisect.bisect_right(self._values, addr) - 1
        while pos >= 0:
            value = self._values[pos]
            distance = addr - value
            if distance > _PREDECESSOR_LIMIT:
                return None
            record = self._symbol_record(self._indices[pos])
            if record is None:
                # 记录读不出来时退到前一个符号, 否则同一序号会被反复处理
                pos -= 1
                continue
            st_name, st_value, st_size = record
            name = self._name_at(st_name)
            if name is None or len(name) == 0:
                pos -= 1
                continue
            is_inside_symbol = st_value == addr or (st_size > 0 and addr < st_value + st_size)
            if is_inside_symbol or not name.startswith(_LOCAL_NAME_PREFIXES):
                return name
            pos -= 1
        return None

    # ----------------------------------------------------------
    #  名称到地址
    # ----------------------------------------------------------

    def _ensure_name_index(self) -> array:
        if self._name_index is not None:
            return self._name_index

        skip_local = self.entry_count > LOCAL_SYMBOL_SKIP_THRESHOLD
        packed = [
            (st_name << 32) | sym_index
            for sym_index, st_name, _value, _size in self._iter_accepted_records(skip_local)
        ]
        packed.sort()

        entries: list[int] = []
        pos = 0
        for chunk_start, chunk in self._iter_strtab_chunks():
            if pos >= len(packed):
                break
            pos = self._consume_strtab_chunk(packed, pos, chunk_start, chunk, entries)

        entries.sort()
        self._name_index = array("Q", entries)
        return self._name_index

    def _iter_strtab_chunks(self):
        """按块产出字符串表, 每项为 (起始偏移, 字节)."""
        for chunk_start in range(0, self._strtab_size, SCAN_CHUNK_BYTES):
            take = min(SCAN_CHUNK_BYTES, self._strtab_size - chunk_start)
            yield chunk_start, self._elf.pread(self._strtab_offset + chunk_start, take)

    def _consume_strtab_chunk(
        self,
        packed: list[int],
        pos: int,
        chunk_start: int,
        chunk: bytes,
        entries: list[int],
    ) -> int:
        """把一个字符串表分块内的名称加入名称索引, 返回下一个待处理的序号."""
        while pos < len(packed):
            item = packed[pos]
            st_name = item >> 32
            local = st_name - chunk_start
            if local >= len(chunk):
                break
            if local >= 0:
                raw = self._name_bytes(chunk, local, st_name)
                if raw:
                    entries.append((_name_hash(raw) << 32) | (item & 0xFFFFFFFF))
            # 释放临时列表中的大整数, 保留列表槽位供后续遍历
            packed[pos] = 0
            pos += 1
        return pos

    def _name_bytes(self, chunk: bytes, local: int, st_name: int) -> bytes:
        """从一个字符串表分块中取名称字节; 名称跨过分块边界时按需读取."""
        end = chunk.find(b"\x00", local)
        if end > local:
            return chunk[local:end]
        name = self._name_at(st_name)
        return name.encode("utf-8") if name else b""

    def lookup_name(self, name: str) -> int | None:
        """按名称查找符号值; 同名符号取符号表中靠后的一个."""
        index = self._ensure_name_index()
        raw = name.encode("utf-8", errors="replace")
        target = _name_hash(raw)
        lo = bisect.bisect_left(index, target << 32)
        hi = bisect.bisect_left(index, (target + 1) << 32)
        found: int | None = None
        for pos in range(lo, hi):
            record = self._symbol_record(index[pos] & 0xFFFFFFFF)
            if record is None:
                continue
            st_name, st_value, _st_size = record
            if self._name_at(st_name) == name:
                found = st_value
        return found

    # ----------------------------------------------------------
    #  生命周期
    # ----------------------------------------------------------

    def close(self) -> None:
        self._elf.close()
        self._name_cache.clear()
        self._name_index = None
