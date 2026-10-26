#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""合成 ELF 文件的构造辅助, 供符号索引与符号加载用例使用."""

from __future__ import annotations

from dataclasses import dataclass, replace
import struct

ELF_HEADER_SIZE = 64
SECTION_HEADER_SIZE = 64
PROGRAM_HEADER_SIZE = 56

ELF64_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")
PROGRAM_HEADER = struct.Struct("<IIQQQQQQ")
SYMBOL_RECORD = struct.Struct("<IBBHQQ")

SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_NOBITS = 8
SHT_DYNSYM = 11

SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4

PT_LOAD = 1

STB_LOCAL = 0
STB_GLOBAL = 1
STT_NOTYPE = 0
STT_OBJECT = 1
STT_FUNC = 2

ET_EXEC = 2
EM_RISCV = 243

NOP_BYTES = b"\x13\x00\x00\x00"


@dataclass
class SectionSpec:
    """一个节头及其在文件中的内容."""

    sh_type: int = SHT_PROGBITS
    flags: int = 0
    addr: int = 0
    data: bytes | None = b""
    size: int = 0
    link: int = 0
    entsize: int = 0
    offset: int | None = None


@dataclass
class ProgramSpec:
    """一个程序头."""

    p_type: int = PT_LOAD
    offset: int = 0
    vaddr: int = 0
    filesz: int = 0
    memsz: int = 0


@dataclass
class HeaderSpec:
    """写进 ELF 头的覆盖项; 未给出的字段取构造器设定的默认值."""

    entry: int | None = None
    ident: bytes | None = None
    ehsize: int = ELF_HEADER_SIZE
    phentsize: int | None = None
    phnum: int | None = None
    phoff: int | None = None
    shentsize: int | None = None
    shnum: int | None = None
    shoff: int | None = None


@dataclass
class TextSpec:
    """代码节的覆盖项."""

    addr: int = 0x1000
    data: bytes = NOP_BYTES * 4
    sh_type: int = SHT_PROGBITS
    flags: int = SHF_ALLOC | SHF_EXECINSTR
    offset: int | None = None


@dataclass
class SymTabSpec:
    """符号表节的覆盖项."""

    is_dynsym: bool = False
    entsize: int = 24


@dataclass
class LayoutSpec:
    """程序头设置与节头表之后追加的零字节."""

    programs: list[ProgramSpec] | None = None
    extra_size: int = 0


def align_up(value: int, align: int = 8) -> int:
    return (value + align - 1) // align * align


def pack_symbols(
    symbols: list[tuple[str | None, int, int, int, int]],
) -> tuple[bytes, bytes]:
    """把 (名称, 值, 大小, 信息字节, 节索引) 序列打包为符号表与字符串表."""
    strtab = bytearray(b"\x00")
    offsets: dict[str, int] = {}
    records = bytearray()
    for name, value, size, info, shndx in symbols:
        if name is None:
            st_name = 0
        else:
            st_name = offsets.get(name, -1)
            if st_name < 0:
                st_name = len(strtab)
                offsets[name] = st_name
                strtab += name.encode("utf-8") + b"\x00"
        records += SYMBOL_RECORD.pack(st_name, info, 0, shndx, value, size)
    return bytes(records), bytes(strtab)


def build_elf(
    *,
    header: HeaderSpec | None = None,
    sections: list[SectionSpec] | None = None,
    layout: LayoutSpec | None = None,
) -> bytes:
    """按给出的节与程序头生成 ELF64 文件字节.

    *header* 的覆盖项直接写进 ELF 头, 用于构造头部字段非法的用例;
    各表的实际位置仍按布局计算, *layout* 的 *extra_size* 追加在节头表之后.
    """
    head = HeaderSpec() if header is None else header
    file_layout = LayoutSpec() if layout is None else layout
    entry = 0 if head.entry is None else head.entry
    specs = list(sections or [])
    progs = list(file_layout.programs or [])

    cursor = ELF_HEADER_SIZE
    real_phoff = cursor if progs else 0
    cursor += len(progs) * PROGRAM_HEADER_SIZE
    cursor = align_up(cursor)

    payloads: list[tuple[int, bytes]] = []
    section_headers: list[bytes] = []
    for spec in specs:
        offset = cursor if spec.offset is None else spec.offset
        if spec.data is None:
            size = spec.size
        else:
            size = spec.size if spec.size else len(spec.data)
            payloads.append((offset, spec.data[:size]))
        cursor = max(cursor, align_up(offset + size))
        section_headers.append(
            SECTION_HEADER.pack(
                0,
                spec.sh_type,
                spec.flags,
                spec.addr,
                offset,
                size,
                spec.link,
                0,
                0,
                spec.entsize,
            )
        )

    real_shoff = align_up(cursor)
    total = real_shoff + len(specs) * SECTION_HEADER_SIZE + file_layout.extra_size
    buf = bytearray(total)
    for offset, data in payloads:
        buf[offset : offset + len(data)] = data

    for i, prog in enumerate(progs):
        base = real_phoff + i * PROGRAM_HEADER_SIZE
        PROGRAM_HEADER.pack_into(
            buf,
            base,
            prog.p_type,
            0,
            prog.offset,
            prog.vaddr,
            prog.vaddr,
            prog.filesz,
            prog.memsz,
            0,
        )

    for i, shdr in enumerate(section_headers):
        base = real_shoff + i * SECTION_HEADER_SIZE
        buf[base : base + SECTION_HEADER_SIZE] = shdr

    default_ident = b"\x7fELF\x02\x01\x01" + b"\x00" * 9
    header_ident = head.ident if head.ident is not None else default_ident
    ELF64_HEADER.pack_into(
        buf,
        0,
        header_ident,
        ET_EXEC,
        EM_RISCV,
        1,
        entry,
        real_phoff if head.phoff is None else head.phoff,
        real_shoff if head.shoff is None else head.shoff,
        0,
        head.ehsize,
        PROGRAM_HEADER_SIZE if head.phentsize is None else head.phentsize,
        len(progs) if head.phnum is None else head.phnum,
        SECTION_HEADER_SIZE if head.shentsize is None else head.shentsize,
        len(specs) if head.shnum is None else head.shnum,
        0,
    )
    return bytes(buf)


def build_symbol_elf(
    *,
    symbols: list[tuple[str | None, int, int, int, int]] | None = None,
    header: HeaderSpec | None = None,
    text: TextSpec | None = None,
    symtab: SymTabSpec | None = None,
    layout: LayoutSpec | None = None,
) -> bytes:
    """生成由空节、代码节、符号表节与字符串表节构成的 ELF.

    *symbols* 为 None 时不生成符号表节; *header* 未给出 *entry* 时入口地址取 0x1000.
    *layout* 的 *extra_size* 在节头表之后追加零字节, 用于让校验样本落在文件末尾之前.
    """
    head = HeaderSpec() if header is None else header
    if head.entry is None:
        head = replace(head, entry=0x1000)
    text_spec = TextSpec() if text is None else text
    symtab_spec = SymTabSpec() if symtab is None else symtab
    file_layout = LayoutSpec() if layout is None else layout

    sections = [
        SectionSpec(sh_type=SHT_NULL),
        SectionSpec(
            sh_type=text_spec.sh_type,
            flags=text_spec.flags,
            addr=text_spec.addr,
            data=text_spec.data,
            offset=text_spec.offset,
        ),
    ]
    if symbols is not None:
        symtab_data, strtab = pack_symbols(symbols)
        sections.append(
            SectionSpec(
                sh_type=SHT_DYNSYM if symtab_spec.is_dynsym else SHT_SYMTAB,
                data=symtab_data,
                link=3,
                entsize=symtab_spec.entsize,
            )
        )
        sections.append(SectionSpec(sh_type=SHT_STRTAB, data=strtab))
    return build_elf(header=head, sections=sections, layout=file_layout)


def build_symtab_from_records(
    records: list[tuple[int, int, int, int, int, int]],
    strtab: bytes,
    *,
    text_addr: int = 0x1000,
) -> bytes:
    """用原始的 (名称偏移, 信息, 其他, 节索引, 值, 大小) 构造 ELF."""
    symtab = b"".join(SYMBOL_RECORD.pack(*record) for record in records)
    sections = [
        SectionSpec(sh_type=SHT_NULL),
        SectionSpec(
            sh_type=SHT_PROGBITS,
            flags=SHF_ALLOC | SHF_EXECINSTR,
            addr=text_addr,
            data=NOP_BYTES,
        ),
        SectionSpec(sh_type=SHT_SYMTAB, data=symtab, link=3, entsize=24),
        SectionSpec(sh_type=SHT_STRTAB, data=strtab),
    ]
    return build_elf(header=HeaderSpec(entry=text_addr), sections=sections)
