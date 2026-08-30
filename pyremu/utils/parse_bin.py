#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
跨平台固件解析器 — 从 ELF / PE / 二进制固件中提取机器码。

支持格式:
- ELF (Linux 可执行文件, RISC-V firmware)
- PE  (Windows 可执行文件)
- Raw binary (llvm-objcopy -O binary 等工具生成的纯机器码)

格式识别通过文件头魔数自动完成:
    ELF: \\x7fELF
    PE:  MZ (0x4D 0x5A)
    其他: 视为 raw binary
"""

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
import struct

import lief

from pyremu.utils.wrapper import seize_err_if_any

# ============================================================
#  文件格式魔数
# ============================================================

ELF_MAGIC = b"\x7fELF"
PE_MAGIC = b"MZ"

# ============================================================
#  数据模型
# ============================================================


@dataclass
class FirmwareSegment:
    """从固件中提取的一段连续内存区域。

    Attributes:
        vaddr: 虚拟地址 (裸金属场景下即为物理地址)。
        data: 需要加载到该地址的原始字节。
        memsz: 内存中的大小 (可能大于 len(data), 超出部分零填充)。
        name: 段名 (ELF 为所含节名逗号拼接, PE 为节名, raw 为空).
    """

    vaddr: int
    data: bytes
    memsz: int
    name: str = ""


@dataclass
class FirmwareImage:
    """已解析的固件镜像, 可直接加载到模拟器中。

    Attributes:
        entry_point: 程序入口虚拟地址。
        segments: 需要加载到内存的各段 (至少包含 .text 代码段)。
        format: 原始文件格式 ("elf", "pe", "raw")。
        symbols: 符号名 -> 地址映射。ELF 从 .symtab 提取已定义的非零符号;
            PE 从导出表提取; raw binary 为空字典。
        symbol_ranges: 符号地址范围列表, 按起始地址升序排列.
            (start, end, name) — 供二分查找做 PC->名称 解析.
    """

    entry_point: int
    segments: list[FirmwareSegment]
    format: str
    symbols: dict[str, int] = field(default_factory=dict)
    symbol_ranges: list[tuple[int, int, str]] = field(default_factory=list)


# ============================================================
#  格式检测
# ============================================================


def detect_format(
    path: PathLike | str,
) -> str:
    """通过文件头魔数检测固件格式。

    Returns:
        "elf", "pe", 或 "raw".
    """
    with open(path, "rb") as f:
        header = f.read(4)

    if header[:4] == ELF_MAGIC:
        return "elf"
    if header[:2] == PE_MAGIC:
        return "pe"
    return "raw"


# ============================================================
#  ELF 符号表原始解析 (绕过 LIEF 1M 符号上限)
# ============================================================

# Elf64_Sym: 24 bytes
#   st_name  (u32, offset 0)
#   st_info  (u8,  offset 4)
#   st_other (u8,  offset 5)
#   st_shndx (u16, offset 6)
#   st_value (u64, offset 8)
#   st_size  (u64, offset 16)
_ELF64_SYM_FMT = "<IBBHQQ"  # little-endian: u32, u8, u8, u16, u64, u64
_ELF64_SYM_SIZE = struct.calcsize(_ELF64_SYM_FMT)  # 24

# 符号绑定 (bits 7:4 of st_info)
_STB_LOCAL = 0

# 符号类型 (bits 3:0 of st_info) — ELF64 STT_* 常量
_STT_NOTYPE = 0
_STT_OBJECT = 1
_STT_FUNC = 2
_STT_SECTION = 3
_STT_FILE = 4


def _parse_symtab_raw(
    path: str,
    symtab_offset: int,
    symtab_size: int,
    strtab_offset: int,
    strtab_size: int,
    *,
    skip_local_noise: bool = False,
) -> tuple[dict[str, int], list[tuple[int, int, str]]]:
    """直接从 ELF 文件解析 .symtab + .strtab, 不依赖 LIEF.

    绕过 LIEF (≤0.17) 的 1,000,000 符号默认上限.
    大文件 (~1.8M 符号) 约 0.5s.

    Args:
        path: ELF 文件路径.
        symtab_offset: .symtab 节的文件偏移.
        symtab_size: .symtab 节大小 (字节).
        strtab_offset: .strtab 节文件偏移.
        strtab_size: .strtab 节大小 (字节).
        skip_local_noise: True 时跳过 LOCAL 非 FUNC 符号 (FILE/SECTION/NOTYPE 等).
            LOCAL 的 STT_FUNC 始终保留 — 它们在大型 ELF (如 Linux vmlinux)
            中占比约 50%, 缺少会导致栈回溯函数名解析错误.

    Returns:
        (symbols, ranges) 二元组:
        - symbols: 符号名 -> VA 映射
        - ranges:  (start, end, name) 列表, 按 start 升序排列
    """
    count = symtab_size // _ELF64_SYM_SIZE
    symbols: dict[str, int] = {}
    ranges: list[tuple[int, int, str]] = []
    with open(path, "rb") as fh:
        fh.seek(strtab_offset)
        strtab = fh.read(strtab_size)
        fh.seek(symtab_offset)
        symtab_data = fh.read(symtab_size)

    for i in range(count):
        off = i * _ELF64_SYM_SIZE
        st_name, st_info, _st_other, st_shndx, st_value, st_size = struct.unpack_from(
            _ELF64_SYM_FMT, symtab_data, off
        )
        # 跳过未定义/特殊节索引 (shndx==0)
        if st_shndx == 0 or st_value == 0:
            continue
        bind = st_info >> 4
        stype = st_info & 0xF
        # 大型 ELF (如 Linux vmlinux): 跳过 LOCAL 的非函数符号
        # (FILE/SECTION/NOTYPE/OBJECT) 以减少噪声, 但 STT_FUNC
        # 始终保留 — 无论 binding 是 LOCAL 还是 GLOBAL.
        if skip_local_noise and bind == _STB_LOCAL and stype != _STT_FUNC:
            continue
        # 从 strtab 提取名称
        end = strtab.find(b"\x00", st_name)
        if end == -1 or end <= st_name:
            continue
        name = strtab[st_name:end].decode("utf-8", errors="replace")
        if not name:
            continue
        symbols[name] = st_value
        if st_size > 0:
            ranges.append((st_value, st_value + st_size, name))

    ranges.sort(key=lambda r: r[0])
    return symbols, ranges


# ============================================================
#  ELF 解析
# ============================================================


def _parse_elf(
    path: str,
) -> "FirmwareImage":
    """解析 ELF 文件, 提取 LOAD 段和入口地址。

    使用 LIEF 的 ELF 解析器读取 program headers,
    提取类型为 PT_LOAD 的段 (内核/固件加载器实际使用的映射方式)。
    入口地址取自 ELF header 的 e_entry 字段。
    """
    binary = lief.ELF.parse(path)
    if binary is None:
        raise ValueError(f"LIEF failed to parse ELF file: {path}")

    # ELF header 的 e_entry 即为入口虚拟地址
    entry_point = binary.entrypoint

    segments: list[FirmwareSegment] = []
    for seg in binary.segments:
        if seg.type != lief.ELF.Segment.TYPE.LOAD:
            continue
        # 将每个节拆为独立的 FirmwareSegment, 便于逐节展示
        for sec in seg.sections:
            if not sec.name:
                continue
            # SHT_NOBITS (.bss) 的 file offset 可能指向无关数据;
            # 必须用空 data + memsz 触发零填充, 不可直接取 content
            is_nobits = sec.type == lief.ELF.Section.TYPE.NOBITS
            sec_data = b"" if is_nobits else bytes(sec.content)
            # 跳过零尺寸的空节 (链接器生成的无内容标记节)
            if sec.size == 0 and len(sec_data) == 0:
                continue
            sec_name: str = (
                sec.name.decode("utf-8", errors="replace")
                if isinstance(sec.name, bytes)
                else str(sec.name)
            )
            segments.append(
                FirmwareSegment(
                    vaddr=sec.virtual_address,
                    data=sec_data,
                    memsz=sec.size,
                    name=sec_name,
                )
            )

    # 提取已定义的具名符号 (函数/变量名 -> 地址)
    # 优先用原始解析绕过 LIEF 的 1M 符号上限 (Linux vmlinux 有 1.8M+ 符号)
    symbols: dict[str, int] = {}
    symbol_ranges: list[tuple[int, int, str]] = []

    # 定位 .symtab 和 .strtab 节
    symtab_sec = binary.get_section(".symtab")
    strtab_sec = binary.get_section(".strtab")
    if symtab_sec is not None and strtab_sec is not None:
        # 大型 ELF (>500K 符号, 如 Linux vmlinux) 跳过 LOCAL 非函数符号
        # (FILE/SECTION/NOTYPE 等) 以减少噪声; STT_FUNC 始终保留,
        # 无论是 LOCAL 还是 GLOBAL — 否则栈回溯会因缺少静态函数而解析错误.
        sym_count = symtab_sec.size // _ELF64_SYM_SIZE
        symbols, symbol_ranges = _parse_symtab_raw(
            path,
            symtab_sec.file_offset,
            symtab_sec.size,
            strtab_sec.file_offset,
            strtab_sec.size,
            skip_local_noise=(sym_count > 500_000),
        )
    else:
        # 降级: 无节表时使用 LIEF 迭代器 (PE / 损坏的 ELF)
        for exp in binary.symtab_symbols:
            tmp = str(exp.name)
            if len(tmp) > 0:
                symbols[tmp] = exp.value

    for exp in binary.dynamic_symbols:
        tmp = str(exp.name)
        if len(tmp) > 0:
            symbols[tmp] = exp.value

    return FirmwareImage(
        entry_point=entry_point,
        segments=segments,
        format="elf",
        symbols=symbols,
        symbol_ranges=symbol_ranges,
    )


# ============================================================
#  PE 解析
# ============================================================


def _parse_pe(
    path: str,
) -> "FirmwareImage":
    """解析 PE 文件, 提取各节的机器码和入口地址。

    使用 LIEF 的 PE 解析器读取 section headers,
    提取所有包含原始数据的节 (.text / .rdata / .data 等)。
    PE 入口地址 = optional_header.addressof_entrypoint (RVA) + imagebase。
    """
    binary = lief.PE.parse(path)
    if binary is None:
        raise ValueError(f"LIEF failed to parse PE file: {path}")

    # PE 真实入口 = RVA + ImageBase
    image_base = binary.optional_header.imagebase
    entry_rva = binary.optional_header.addressof_entrypoint
    entry_point = image_base + entry_rva

    segments: list[FirmwareSegment] = []
    for sec in binary.sections:
        data = bytes(sec.content)
        if len(data) == 0:
            continue
        segments.append(FirmwareSegment(
            vaddr=image_base + sec.virtual_address,
            data=data,
            memsz=sec.virtual_size,
            name=(
                sec.name.decode("utf-8", errors="replace")
                if isinstance(sec.name, bytes)
                else str(sec.name)
            ),
        ))

    # 提取导出符号 (函数/变量名 -> 地址)
    symbols: dict[str, int] = {}

    exported = binary.get_export()
    if exported is not None:
        for exp in exported.entries:
            tmp = str(exp.name)
            if len(tmp) > 0:
                symbols[tmp] = exp.value

    return FirmwareImage(
        entry_point=entry_point,
        segments=segments,
        format="pe",
        symbols=symbols,
    )


# ============================================================
#  Raw binary 解析
# ============================================================


def _parse_raw(
    path: str,
    base_addr: int = 0,
) -> "FirmwareImage":
    """加载纯二进制固件文件, 全部字节作为一个段映射到 base_addr。

    llvm-objcopy -O binary 等工具生成的 .bin 文件无元数据,
    调用者必须提供正确的加载基址。
    """
    data = Path(path).read_bytes()
    return FirmwareImage(
        entry_point=base_addr,
        segments=[
            FirmwareSegment(
                vaddr=base_addr,
                data=data,
                memsz=len(data),
            )
        ],
        format="raw",
        symbols={},
    )


# ============================================================
#  统一入口
# ============================================================


@seize_err_if_any()
def parse_firmware(
    path: str | PathLike,
    base_addr: int = 0,
) -> "FirmwareImage | None":
    """跨平台固件解析入口 — 自动识别格式并提取机器码。

    格式识别:
        ELF (\\x7fELF 魔数) -> 提取 LOAD 段 + entrypoint
        PE  (MZ 魔数)       -> 提取节内容 + entrypoint (RVA + ImageBase)
        其他                -> 作为 raw binary, 全部字节映射到 base_addr

    Args:
        path: 固件文件路径。
        base_addr: raw binary 时的加载基址 (ELF/PE 时忽略)。

    Returns:
        FirmwareImage 包含入口地址、内存段列表和格式标记; 解析失败返回 None。
    """
    fmt = detect_format(path)
    if fmt == "elf":
        return _parse_elf(path)
    if fmt == "pe":
        return _parse_pe(path)
    return _parse_raw(path, base_addr)
