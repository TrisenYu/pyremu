#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""统计二进制文件中基本块终结指令 (控制流转移指令) 的静态条数。

以基本块描述控制流口径更明确: 一个基本块是一段只能从首条进入、除末条以外不会
转移控制流的最大指令序列, 其末条必然是控制流转移指令。因此 "基本块终结指令的
条数" 就等于静态基本块的个数, 不受 "跳转" 一词歧义的影响。

终结指令按控制流语义分五类:
  条件分支  beq/bne/blt/bge/bltu/bgeu/c.beqz/c.bnez —— 条件成立时跳转, 可顺延
  过程调用  jal ra, ... / jalr ra, ... / c.jalr —— 写 x1 (ra), 在函数之间转移控制
  跳转      jal x0, ... / jalr x0, off(rs1) / c.j / c.jr rs≠ra —— 函数内跳转
  返回      jalr x0, 0(ra) / c.jr ra
  陷入      ecall / ebreak / c.ebreak / mret / sret

分类依据是 RISC-V 调用约定 (写 x1 为过程调用, jalr x0, 0(x1) 为函数返回),
不依赖助记符别名 —— ret 一律按展开后的 jalr x0, 0(ra) 识别。

同一批终结指令另按转移目标的来源统计:
  寄存器跳转  jalr / c.jr / c.jalr —— 目标取自寄存器, 译码时无法确定
  直接跳转    条件分支 / jal / c.j —— 目标是当前 pc 加立即数
  陷入        ecall / ebreak / c.ebreak / mret / sret

报告同时给出来源与类别的对应条数: 寄存器跳转中的过程调用即经函数指针发起的调用,
其中的返回与跳转分别是函数返回与跳转表分派; 直接跳转中的条件分支即循环与条件语句。

输入文件的容器格式自动判定, 不作特例:
  - ELF (经 LIEF 解析): 逐个扫描带 SHF_EXECINSTR 属性的节, 并排除节内属于数据的
    字节区间, 即符号表标注为数据对象的区间, 以及节起始处的 PE/COFF 头
    (EFI 引导桩把头部放在可执行节里, 按 MZ 魔数与 SizeOfHeaders 识别);
  - 其余情况: 按裸二进制处理, 从 --raw-offset 起扫描到文件末尾。

指令流按 RISC-V 压缩指令的判定规则顺序推进 (低两位不等于 0b11 视为 16 位指令,
否则视为 32 位指令), 逐条交给项目自身的反汇编器 pyremu.utils.disassem.disasm
解释, 再按上述五类归类 —— 不另写解码表, 反汇编口径与调试器一致。

用法::

    uv run python tools/jmp_instr_counter.py <二进制> [<二进制> ...]
    uv run python tools/jmp_instr_counter.py --list <二进制>       # 附各节明细
    uv run python tools/jmp_instr_counter.py --json <二进制>       # 机器可读输出
    uv run python tools/jmp_instr_counter.py --no-data-filter <二进制>
    uv run python tools/jmp_instr_counter.py --exclude-trap <二进制>
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import struct
import sys

import lief

# 反汇编器的 GPR/CSR 名称由 core.registers 在模块初始化时注入, 不导入该模块
# 则反汇编结果的寄存器一律退化为裸编号 (x0 显示成 ?x0). 此处仅为触发注入.
import pyremu.core.registers  # noqa: F401
from pyremu.utils.disassem import disasm, parse_compressed

# ELF 节头属性: 含该属性的节才承载指令.
SHF_EXECINSTR = 0x4
# ELF e_machine 中的 RISC-V.
EM_RISCV = 243

# PE/COFF 头: MZ 魔数、PE 签名, 以及 e_lfanew 与 SizeOfHeaders 两个字段的偏移.
_MZ_MAGIC = b"MZ"
_PE_SIGNATURE = b"PE\x00\x00"
_MZ_LFANEW_OFF = 0x3C
_COFF_HEADER_SIZE = 20
_PE_SIZEOFHEADERS_OFF = 0x3C

KIND_COND = "条件分支"
KIND_CALL = "过程调用"
KIND_JUMP = "跳转"
KIND_RET = "返回"
KIND_TRAP = "陷入"
_KIND_ORDER = (KIND_COND, KIND_CALL, KIND_JUMP, KIND_RET, KIND_TRAP)
# 只在函数内部转移控制流的三类 (不含函数间调用与陷入).
_INTRA_KINDS = (KIND_COND, KIND_JUMP, KIND_RET)

# 转移目标取自何处:
#   寄存器跳转  jalr / c.jr / c.jalr —— 目标取自寄存器, 译码时无法确定
#   直接跳转    条件分支 / jal / c.j —— 目标是当前 pc 加立即数
#   陷入        ecall / ebreak / c.ebreak / mret / sret
MECH_INDIRECT = "寄存器跳转"
MECH_DIRECT = "直接跳转"
MECH_TRAP = "陷入"
_MECH_ORDER = (MECH_INDIRECT, MECH_DIRECT, MECH_TRAP)

_COND_MNEMONICS = frozenset(("beq", "bne", "blt", "bge", "bltu", "bgeu", "c.beqz", "c.bnez"))
_TRAP_MNEMONICS = frozenset(("ecall", "ebreak", "c.ebreak", "mret", "sret"))
# 调用约定中的返回地址寄存器 x1 (ra) 与零寄存器 x0; 两种写法都接受.
_RA_NAMES = frozenset(("x1", "ra"))
_ZERO_NAMES = frozenset(("x0", "zero"))
# 函数返回的访存操作数: 偏移为 0 且基址为 ra.
_RET_OPERAND_RE = re.compile(r"^0\((?:x1|ra)\)$")
# 助记符本身即可确定类别的部分 (不需要看目标寄存器).
_FIXED_KIND = {
    **{mnemonic: KIND_COND for mnemonic in _COND_MNEMONICS},
    **{mnemonic: KIND_TRAP for mnemonic in _TRAP_MNEMONICS},
    "c.jalr": KIND_CALL,
}

# 转移目标取自何处, 只由助记符决定.
_MECHANISM = {
    **{mnemonic: MECH_DIRECT for mnemonic in _COND_MNEMONICS},
    **{mnemonic: MECH_TRAP for mnemonic in _TRAP_MNEMONICS},
    "jal": MECH_DIRECT,
    "c.j": MECH_DIRECT,
    "jalr": MECH_INDIRECT,
    "c.jr": MECH_INDIRECT,
    "c.jalr": MECH_INDIRECT,
}


def mnemonic_of(instr_text: str) -> str:
    """从反汇编文本中取出助记符 (首段空白前的部分)."""
    parts = instr_text.split()
    return parts[0] if parts else ""


def operands_of(instr_text: str) -> list[str]:
    """从反汇编文本中取出操作数列表; 无操作数时返回空列表."""
    parts = instr_text.split(None, 1)
    if len(parts) < 2:
        return []
    return [operand.strip() for operand in parts[1].split(",")]


def classify(mnemonic: str, operands: list[str]) -> str | None:
    """判定终结指令的类别; 不是终结指令时返回 None.

    依据 RISC-V 调用约定区分函数间与函数内转移: 写 x1 (ra) 的是过程调用,
    jalr x0, 0(x1) 是函数返回, 其余 jal/jalr/c.jr 是函数内跳转。
    """
    fixed = _FIXED_KIND.get(mnemonic)
    if fixed is not None:
        return fixed
    first = operands[0] if operands else ""
    if mnemonic in ("jal", "c.j"):
        return KIND_CALL if first in _RA_NAMES else KIND_JUMP
    if mnemonic == "jalr":
        second = operands[1] if len(operands) > 1 else ""
        if first in _ZERO_NAMES and _RET_OPERAND_RE.match(second):
            return KIND_RET
        return KIND_CALL if first in _RA_NAMES else KIND_JUMP
    if mnemonic == "c.jr":
        return KIND_RET if first in _RA_NAMES else KIND_JUMP
    return None


def count_mechanisms(mnemonics: Counter[str]) -> Counter[str]:
    """把终结指令的助记符计数按转移目标的来源汇总."""
    mechanisms: Counter[str] = Counter()
    for mnemonic, count in mnemonics.items():
        mechanism = _MECHANISM.get(mnemonic)
        if mechanism is not None:
            mechanisms[mechanism] += count
    return mechanisms


def pe_header_size(content: bytes) -> int:
    """节起始处 PE/COFF 头占用的字节数; 不是 PE 头则返回 0.

    EFI 引导桩会把 PE/COFF 头放在可执行节里 (以 MZ 魔数开头), 这段是数据而非
    指令。长度取可选头中的 SizeOfHeaders, 即整个头部占用的大小。
    """
    if len(content) < _MZ_LFANEW_OFF + 4 or content[:2] != _MZ_MAGIC:
        return 0
    pe_offset = struct.unpack_from("<I", content, _MZ_LFANEW_OFF)[0]
    if content[pe_offset : pe_offset + 4] != _PE_SIGNATURE:
        return 0
    opt_offset = pe_offset + 4 + _COFF_HEADER_SIZE
    if opt_offset + _PE_SIZEOFHEADERS_OFF + 4 > len(content):
        return 0
    size = struct.unpack_from("<I", content, opt_offset + _PE_SIZEOFHEADERS_OFF)[0]
    return min(size, len(content))


def object_spans(binary: object) -> list[tuple[int, int]]:
    """符号表中标注为数据对象 (OBJECT) 的字节区间, 形如 (起始地址, 长度)."""
    return [
        (sym.value, sym.size)
        for sym in binary.symbols
        if sym.size and sym.type == lief.ELF.Symbol.TYPE.OBJECT
    ]


def clip_spans(
    spans: list[tuple[int, int]],
    low: int,
    high: int,
) -> list[tuple[int, int]]:
    """把绝对地址区间裁剪到 [low, high) 并转为相对 low 的偏移区间."""
    out: list[tuple[int, int]] = []
    for start, length in spans:
        begin, end = max(start, low), min(start + length, high)
        if begin < end:
            out.append((begin - low, end - begin))
    return out


def scan_code(
    data: bytes,
    base_pc: int,
    excluded: bytearray,
) -> tuple[Counter[str], Counter[str], Counter[tuple[str, str]], int]:
    """顺序反汇编一段代码.

    返回 (类别计数, 终结指令助记符计数, 来源与类别的对应计数, 指令条数)。
    excluded 按字节标记数据区间; 指令既不得起始于其中, 也不得跨越它。
    """
    kinds: Counter[str] = Counter()
    mnemonics: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    instructions = 0
    offset = 0
    size = len(data)
    while offset + 2 <= size:
        if excluded[offset]:
            offset += 2
            continue
        half = data[offset] | (data[offset + 1] << 8)
        if parse_compressed(half):
            text = disasm(half, base_pc + offset)
            offset += 2
        else:
            if offset + 4 > size or excluded[offset + 2]:
                offset += 2
                continue
            word = half | (data[offset + 2] << 16) | (data[offset + 3] << 24)
            text = disasm(word, base_pc + offset)
            offset += 4
        instructions += 1
        mnemonic = mnemonic_of(text)
        kind = classify(mnemonic, operands_of(text))
        if kind is not None:
            kinds[kind] += 1
            mnemonics[mnemonic] += 1
            mechanism = _MECHANISM.get(mnemonic)
            if mechanism is not None:
                pairs[(mechanism, kind)] += 1
    return kinds, mnemonics, pairs, instructions


def scan_phantom(
    data: bytes,
    base_pc: int,
    spans: list[tuple[int, int]],
) -> tuple[int, int]:
    """统计被排除的数据区间若当指令解码会产生多少条指令与多少条终结指令.

    用于量化数据过滤的影响: 这些是线性扫描本会误判出来的条数。
    """
    phantom_instrs = 0
    phantom_terms = 0
    for start, length in spans:
        blob = data[start : start + length]
        kinds, _mnemonics, _pairs, decoded = scan_code(
            blob, base_pc + start, bytearray(len(blob))
        )
        phantom_instrs += decoded
        phantom_terms += sum(kinds.values())
    return phantom_instrs, phantom_terms


def elf_exec_sections(path: Path) -> object | None:
    """解析 ELF, 返回 LIEF Binary; 非 ELF 返回 None."""
    try:
        binary = lief.parse(str(path))
    except Exception:  # LIEF 对非 ELF 输入抛出的异常种类不定, 一并退回裸扫描
        return None
    if binary is None or not isinstance(binary, lief.ELF.Binary):
        return None
    machine = binary.header.machine_type
    if int(machine) != EM_RISCV:
        raise SystemExit(f"{path}: 目标架构为 {machine}, 不是 RISC-V, 拒绝统计")
    return binary


def count_file(path: Path, raw_offset: int, include_trap: bool, filter_data: bool) -> dict:
    """统计单个文件的基本块终结指令条数."""
    data = path.read_bytes()
    binary = elf_exec_sections(path)
    if binary is None:
        print(
            f"[warn] {path}: 不是可解析的 ELF, 按裸二进制从偏移 0x{raw_offset:x} 扫描",
            file=sys.stderr,
        )
        sections = [(f"<raw @0x{raw_offset:x}>", 0, data[raw_offset:], [])]
    else:
        spans = object_spans(binary) if filter_data else []
        sections = []
        for section in binary.sections:
            if not section.flags & SHF_EXECINSTR:
                continue
            content = bytes(section.content)
            if not content:
                continue
            low = section.virtual_address
            own = clip_spans(spans, low, low + len(content))
            if filter_data:
                header = pe_header_size(content)
                if header:
                    own.append((0, header))
            sections.append((section.name or "<unnamed>", low, content, own))

    kinds: Counter[str] = Counter()
    mnemonics: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    per_section: list[dict[str, object]] = []
    instructions = 0
    excluded_bytes = 0
    phantom_instrs = 0
    phantom_terms = 0
    for name, vaddr, content, spans in sections:
        excluded = bytearray(len(content))
        for start, length in spans:
            excluded[start : start + length] = b"\x01" * length
            excluded_bytes += length
        section_kinds, section_mnemonics, section_pairs, section_instrs = scan_code(
            content, vaddr, excluded
        )
        kinds.update(section_kinds)
        mnemonics.update(section_mnemonics)
        pairs.update(section_pairs)
        instructions += section_instrs
        if spans:
            pin, pterm = scan_phantom(content, vaddr, spans)
            phantom_instrs += pin
            phantom_terms += pterm
        per_section.append(
            {
                "name": name,
                "bytes": len(content),
                "excluded_bytes": sum(length for _, length in spans),
                "terminators": sum(section_kinds.values()),
            }
        )

    total_terms = sum(kinds[kind] for kind in _KIND_ORDER)
    if not include_trap:
        total_terms -= kinds[KIND_TRAP]
    intra_terms = sum(kinds[kind] for kind in _INTRA_KINDS)
    mechanisms = count_mechanisms(mnemonics)
    return {
        "file": str(path),
        "size": len(data),
        "instructions": instructions,
        "terminators": total_terms,
        "terminator_ratio": total_terms / instructions if instructions else 0.0,
        "intra_terminators": intra_terms,
        "excluded_bytes": excluded_bytes,
        "phantom_instructions": phantom_instrs,
        "phantom_terminators": phantom_terms,
        "by_kind": {kind: kinds[kind] for kind in _KIND_ORDER},
        "by_mechanism": {mech: mechanisms[mech] for mech in _MECH_ORDER},
        "by_mechanism_kind": {
            mech: {kind: pairs[(mech, kind)] for kind in _KIND_ORDER} for mech in _MECH_ORDER
        },
        "by_mnemonic": dict(mnemonics),
        "sections": per_section,
    }


def print_report(result: dict, show_sections: bool) -> None:
    """打印单个文件的统计报告."""
    print(f"{result['file']}")
    print(
        f"  指令总数 {result['instructions']}, 基本块终结指令 {result['terminators']} 条, "
        f"占比 {result['terminator_ratio'] * 100:.3f}%"
    )
    print(f"  其中函数内转移控制流 (条件分支 + 跳转 + 返回) {result['intra_terminators']} 条")
    if result["excluded_bytes"]:
        print(
            f"  已排除可执行节内的数据区间 {result['excluded_bytes']} 字节; "
            f"若把这些字节当指令, 会多算 {result['phantom_instructions']} 条指令"
            f"与 {result['phantom_terminators']} 条终结指令"
        )
    print("  按类别:")
    for kind in _KIND_ORDER:
        print(f"    {kind}  {result['by_kind'].get(kind, 0)}")
    print("  按转移目标的来源:")
    for mech in _MECH_ORDER:
        count = result["by_mechanism"].get(mech, 0)
        parts = [
            f"{kind} {result['by_mechanism_kind'][mech].get(kind, 0)}"
            for kind in _KIND_ORDER
            if result["by_mechanism_kind"][mech].get(kind, 0)
        ]
        print(f"    {mech}  {count:<8} {'  '.join(parts)}")
    print("  按助记符:")
    for mnemonic, count in sorted(
        result["by_mnemonic"].items(), key=lambda kv: (-kv[1], kv[0])
    ):
        print(f"    {mnemonic:<8} {count}")
    if show_sections:
        print("  按节:")
        for section in result["sections"]:
            print(
                f"    {section['name']:<20} {section['bytes']:>10} 字节  "
                f"排除 {section['excluded_bytes']:>6}  终结指令 {section['terminators']:>8}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="统计二进制文件中基本块终结指令的静态条数")
    parser.add_argument("binaries", nargs="+", help="待统计的二进制文件")
    parser.add_argument(
        "--raw-offset",
        type=lambda text: int(text, 0),
        default=0,
        help="非 ELF 输入时跳过的文件头字节数 (默认 0)",
    )
    parser.add_argument("--list", action="store_true", help="附各节明细")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument(
        "--no-data-filter",
        action="store_true",
        help="不排除可执行节内的数据区间, 退回纯线性扫描",
    )
    parser.add_argument(
        "--exclude-trap",
        action="store_true",
        help="不计入 ecall/ebreak/c.ebreak/mret/sret 这类陷入终结指令",
    )
    args = parser.parse_args()

    results = []
    for name in args.binaries:
        path = Path(name)
        if not path.is_file():
            print(f"[error] 找不到文件: {path}", file=sys.stderr)
            return 2
        results.append(
            count_file(
                path,
                args.raw_offset,
                not args.exclude_trap,
                not args.no_data_filter,
            )
        )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for index, result in enumerate(results):
        if index:
            print()
        print_report(result, args.list)

    if len(results) > 1:
        print(f"\n合计终结指令条数: {sum(int(r['terminators']) for r in results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
