#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
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
from copy import deepcopy
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
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
        symbols: 符号名 → 地址映射。ELF 从 .symtab 提取已定义的非零符号;
            PE 从导出表提取; raw binary 为空字典。
    """

    entry_point: int
    segments: list[FirmwareSegment]
    format: str
    symbols: dict[str, int] = field(default_factory=dict)


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
        data = bytes(seg.content)
        # 从所含节名组合段名
        sec_names = [s.name for s in seg.sections if s.name]
        seg_name = ",".join(sec_names) if sec_names else "LOAD"
        segments.append(
            FirmwareSegment(
                vaddr=seg.virtual_address,
                data=data,
                memsz=seg.virtual_size,
                name=seg_name,
            )
        )

    # 提取已定义的具名符号 (函数/变量名 → 地址)
    symbols: dict[str, int] = {}
    
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
        segments.append(
            FirmwareSegment(
                vaddr=image_base + sec.virtual_address,
                data=data,
                memsz=sec.virtual_size,
                name=sec.name,
            )
        )

    # 提取导出符号 (函数/变量名 → 地址)
    symbols: dict[str, int] = {}
    
    for exp in binary.get_export().entries:
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
    path: str,
    base_addr: int = 0,
) -> "FirmwareImage | None":
    """跨平台固件解析入口 — 自动识别格式并提取机器码。

    格式识别:
        ELF (\\x7fELF 魔数) → 提取 LOAD 段 + entrypoint
        PE  (MZ 魔数)       → 提取节内容 + entrypoint (RVA + ImageBase)
        其他                → 作为 raw binary, 全部字节映射到 base_addr

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
