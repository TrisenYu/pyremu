#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析飞地调试统计日志并生成 CSV。

日志来自 ``CONFIG_DBG_ENCLAVE_STATS`` 之下的三项统计:

  ``encl_stat<cycle>``       ``CONFIG_DBG_CYCLE_COUNT``     飞地执行区间的累计周期数
  ``encl_stat<alloc-info>``  ``CONFIG_DBG_MEM_INTERACT``    内存申请、被拒绝的申请与释放
  ``encl_stat<pmp-...>``     ``CONFIG_DBG_PMP_CACHE``       PMP 条目组的淘汰、装入与故障

每个飞地输出一行。飞地编号即一次测试项的执行; ``test`` 列取该飞地全部统计行之后
(``--name-position after``, 默认) 或之前 (``before``) 的第一条 ``regress-check`` 行。
默认取 ``after``, 与现有回归脚本先打印统计、再打印检查结果的次序一致。找不到时该列留空。

PMP 列按类别 (``data`` 或 ``instr``) 分开:

  ``evict``  被替换出窗口的条目数, 含访问故障装入时按策略淘汰的条目与整组重装、终止之前
             窗口内的占用条目; 后两者按条目所属的飞地计入
  ``load``   写入本组条目的次数, 含切入飞地时整组重装写下的条目与访问故障装入的条目
  ``fault``  506 号调用的往返次数, 一次调用计一次; 取指故障归 ``instr``, 读与写故障归
             ``data``, 与装入的节点所属的类别无关

兼容两种较早的日志格式: ``[dbg-mem-interact] ...`` 内存报告与
``encl_stat<pmp>: (eid,attr): ...`` 的 PMP 报告; 后者带有一项命中计数, 本工具不再输出该列。

``auto`` 模式下只输出日志中实际出现的统计组; 某项统计整份日志都未出现时, 对应列留空,
与"该统计已启用但本测试项的计数为零"区分开。

用法:

    uv run python tools/parse_dbg_stats.py output/rec.log -o build/dbg_stats.csv
    uv run python tools/parse_dbg_stats.py output/*.log --split --out-dir build/stats
    uv run python tools/parse_dbg_stats.py output/rec.log --features cycle,mem,pmp
"""

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ENCL_STAT_HINT = re.compile(r"encl_stat<")

CYCLE_PATTERNS = (
    re.compile(r"encl_stat<cycle>:\s*\(eid,cycles\):\s*(\d+),(\d+)"),
    re.compile(r"encl_stat<cycle>:\s*\(eid\):\s*(\d+)\s+mcycle:\s*(\d+)"),
)

ALLOC_PATTERN = re.compile(
    r"encl_stat<alloc-info>:\s*\(eid,typ,exit-code\):\s*(\d+),(\d+),(\d+)\s+"
    r"alloc\(req,siz(?:;denum,desiz)?\):\s*(\d+),(\d+)(?:,(\d+),(\d+))?\s+"
    r"free\(req,siz\):\s*(\d+),(\d+)"
)

LEGACY_ALLOC_PATTERN = re.compile(
    r"\[dbg-mem-interact\]\s+enclave_id=(\d+)\s+term=([\w-]+)\s+exit_code=(\d+)\s+"
    r"alloc_req=(\d+)\s+alloc_bytes=(\d+)\s+alloc_denied_req=(\d+)\s+"
    r"alloc_denied_bytes=(\d+)\s+free_req=(\d+)\s+free_bytes=(\d+)"
)

PMP_PATTERN = re.compile(
    r"encl_stat<pmp-(data|instr)-(\d+)>\(evic,load,flt\):\s*"
    r"(\d+),(\d+),(\d+)"
)

# 该项统计曾带一项命中计数, 其值恒为零, 已从输出中移除; 早期日志仍按四元组解析。
PMP_PATTERN_WITH_HIT = re.compile(
    r"encl_stat<pmp-(data|instr)-(\d+)>\(hit,evic,load,flt\):\s*"
    r"\d+,(\d+),(\d+),(\d+)"
)

LEGACY_PMP_PATTERN = re.compile(
    r"encl_stat<pmp>:\s*\(eid,attr\):\s*(\d+),(data|instr)\s+"
    r"entry\(hit,evic,load,flt\):\s*(\d+),(\d+),(\d+),(\d+)"
)

DEFAULT_NAME_PATTERNS = (
    r">>\s*regress-check\s+load\s+(?P<name>\S+)\s+\(exited\)\s*:\s*(?P<status>PASS|FAIL)",
    r">>\s*regress-check\s+(?P<name>[^:]+?)\s*:\s*(?P<status>PASS|FAIL)",
    r"\[regress\]\s+(?P<name>[^:]+?)\s*:\s*(?P<status>PASS|FAIL)",
)

TERM_NAMES = {
    0: "SELF_EXIT",
    1: "HOST_SHUTDOWN",
}

BASE_COLUMNS = ("test", "enclave_id", "status", "term", "exit_code", "source", "first_line")

FEATURE_GROUPS = ("cycle", "mem", "pmp")

GROUP_COLUMNS = {
    "cycle": ("cycles",),
    "mem": (
        "alloc_req",
        "alloc_bytes",
        "alloc_denied_req",
        "alloc_denied_bytes",
        "free_req",
        "free_bytes",
    ),
    "pmp": (
        "pmp_data_evict",
        "pmp_data_load",
        "pmp_data_fault",
        "pmp_instr_evict",
        "pmp_instr_load",
        "pmp_instr_fault",
    ),
}


@dataclass
class Record:
    """日志中的一条统计记录, 对应某飞地的某一项统计。"""

    line_no: int
    eid: int
    group: str
    values: dict[str, int]
    term: str = ""
    exit_code: int | None = None


@dataclass
class NameHit:
    """日志中的一条测试项标记。"""

    line_no: int
    name: str
    status: str


@dataclass
class Item:
    """一个飞地的合并结果, 对应输出 CSV 的一行。"""

    eid: int
    first_line: int
    last_line: int
    groups: set[str] = field(default_factory=set)
    values: dict[str, int] = field(default_factory=dict)
    term: str = ""
    exit_code: int | None = None
    name: str = ""
    status: str = ""


@dataclass
class ParsedLog:
    """一份日志的解析结果。"""

    path: Path
    items: list[Item]
    skipped: int


def term_name(value: int) -> str:
    """把 ``enum enclave_term_kind`` 的取值折成名称。"""
    return TERM_NAMES.get(value, f"TERM_{value}")


def normalize_term(value: str) -> str:
    """把旧格式中的终止类别名称折成同一形态。"""
    return value.replace("-", "_").upper()


def match_cycle(line: str, line_no: int) -> Record | None:
    for pattern in CYCLE_PATTERNS:
        match = pattern.search(line)
        if match:
            values = {"cycles": int(match.group(2))}
            return Record(line_no, int(match.group(1)), "cycle", values)
    return None


def match_alloc(line: str, line_no: int) -> Record | None:
    match = ALLOC_PATTERN.search(line)
    if not match:
        return None
    denied_req = int(match.group(6)) if match.group(6) is not None else 0
    denied_bytes = int(match.group(7)) if match.group(7) is not None else 0
    values = {
        "alloc_req": int(match.group(4)),
        "alloc_bytes": int(match.group(5)),
        "alloc_denied_req": denied_req,
        "alloc_denied_bytes": denied_bytes,
        "free_req": int(match.group(8)),
        "free_bytes": int(match.group(9)),
    }
    return Record(
        line_no,
        int(match.group(1)),
        "mem",
        values,
        term=term_name(int(match.group(2))),
        exit_code=int(match.group(3)),
    )


def match_legacy_alloc(line: str, line_no: int) -> Record | None:
    match = LEGACY_ALLOC_PATTERN.search(line)
    if not match:
        return None
    values = {
        "alloc_req": int(match.group(4)),
        "alloc_bytes": int(match.group(5)),
        "alloc_denied_req": int(match.group(6)),
        "alloc_denied_bytes": int(match.group(7)),
        "free_req": int(match.group(8)),
        "free_bytes": int(match.group(9)),
    }
    return Record(
        line_no,
        int(match.group(1)),
        "mem",
        values,
        term=normalize_term(match.group(2)),
        exit_code=int(match.group(3)),
    )


def match_pmp(line: str, line_no: int) -> Record | None:
    for pattern in (PMP_PATTERN, PMP_PATTERN_WITH_HIT):
        match = pattern.search(line)
        if match:
            eid = int(match.group(2))
            values = pmp_values(match.group(1), match.groups()[2:])
            return Record(line_no, eid, "pmp", values)

    match = LEGACY_PMP_PATTERN.search(line)
    if match:
        eid = int(match.group(1))
        values = pmp_values(match.group(2), match.groups()[3:])
        return Record(line_no, eid, "pmp", values)
    return None


def pmp_values(attr: str, raw: tuple[str, ...]) -> dict[str, int]:
    """按类别展开 ``(evic, load, flt)`` 三个字段。"""
    evict, load, fault = (int(value) for value in raw)
    return {
        f"pmp_{attr}_evict": evict,
        f"pmp_{attr}_load": load,
        f"pmp_{attr}_fault": fault,
    }


def parse_record(line: str, line_no: int) -> Record | None:
    """依次尝试各类统计行的格式。"""
    for matcher in (match_cycle, match_alloc, match_legacy_alloc, match_pmp):
        record = matcher(line, line_no)
        if record is not None:
            return record
    return None


def match_name(line: str, patterns: tuple[re.Pattern[str], ...]) -> tuple[str, str] | None:
    for pattern in patterns:
        match = pattern.search(line)
        if match:
            status = match.groupdict().get("status") or ""
            return match.group("name").strip(), status
    return None


def parse_log(
    path: Path,
    patterns: tuple[re.Pattern[str], ...],
    name_position: str,
) -> ParsedLog:
    """解析一份日志, 按飞地编号合并各统计行。"""
    items: dict[int, Item] = {}
    names: list[NameHit] = []
    skipped = 0

    text = path.read_text(encoding="utf-8", errors="replace")
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = ANSI_ESCAPE.sub("", raw)

        hit = match_name(line, patterns)
        if hit is not None:
            names.append(NameHit(line_no, hit[0], hit[1]))

        record = parse_record(line, line_no)
        if record is None:
            if ENCL_STAT_HINT.search(line):
                skipped += 1
            continue

        item = items.get(record.eid)
        if item is None:
            item = Item(record.eid, record.line_no, record.line_no)
            items[record.eid] = item
        item.last_line = record.line_no
        item.groups.add(record.group)
        item.values.update(record.values)
        if record.term:
            item.term = record.term
        if record.exit_code is not None:
            item.exit_code = record.exit_code

    attach_names(items.values(), names, name_position)
    return ParsedLog(path, list(items.values()), skipped)


def attach_names(items, names: list[NameHit], name_position: str) -> None:
    """按行序把测试项标记关联到飞地。

    ``after`` 取该飞地最后一条统计行之后的第一条标记; ``before`` 取第一条统计行之前
    的最后一条标记。同一份日志内飞地按终止次序打印, 故该关联与串行执行的测试次序一致。
    """
    for item in items:
        if name_position == "before":
            candidates = [hit for hit in names if hit.line_no <= item.first_line]
            chosen = candidates[-1] if candidates else None
        else:
            candidates = [hit for hit in names if hit.line_no > item.last_line]
            chosen = candidates[0] if candidates else None
        if chosen is not None:
            item.name = chosen.name
            item.status = chosen.status


def expand_inputs(raw_paths: list[str]) -> list[Path]:
    """展开目录中的 ``*.log``, 保留输入次序并去重。"""
    paths: list[Path] = []
    seen: set[Path] = set()

    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            candidates = sorted(path.glob("*.log"))
        elif path.is_file():
            candidates = [path]
        else:
            print(f"警告: 跳过不存在的路径: {raw}", file=sys.stderr)
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(candidate)
    return paths


def compile_patterns(raw_patterns: list[str] | None) -> tuple[re.Pattern[str], ...]:
    """编译名称正则; 每条须含 ``name`` 组, ``status`` 组可省略。"""
    source = raw_patterns or DEFAULT_NAME_PATTERNS
    compiled: list[re.Pattern[str]] = []
    for text in source:
        try:
            pattern = re.compile(text)
        except re.error as exc:
            raise ValueError(f"名称正则无效: {text}: {exc}") from exc
        if "name" not in pattern.groupindex:
            raise ValueError(f"名称正则缺少 name 组: {text}")
        compiled.append(pattern)
    return tuple(compiled)


def resolve_enabled(requested: str, present: set[str]) -> set[str]:
    """决定输出哪些统计组。"""
    if requested == "auto":
        return set(present)
    chosen = {part.strip() for part in requested.split(",") if part.strip()}
    unknown = chosen - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"未知的统计组: {', '.join(sorted(unknown))}")
    return chosen


def build_columns(enabled: set[str]) -> list[str]:
    columns = list(BASE_COLUMNS)
    for group in FEATURE_GROUPS:
        if group in enabled:
            columns.extend(GROUP_COLUMNS[group])
    return columns


def build_rows(
    parsed: ParsedLog,
    enabled: set[str],
    forced: set[str],
) -> list[dict[str, object]]:
    """生成一份日志的行; 未启用的统计组留空, 已启用但本项为零时写 0。"""
    file_groups = {group for item in parsed.items for group in item.groups}
    rows: list[dict[str, object]] = []

    for item in sorted(parsed.items, key=lambda it: it.first_line):
        row: dict[str, object] = {
            "test": item.name,
            "enclave_id": item.eid,
            "status": item.status,
            "term": item.term,
            "exit_code": "" if item.exit_code is None else item.exit_code,
            "source": str(parsed.path),
            "first_line": item.first_line,
        }
        for group in FEATURE_GROUPS:
            if group not in enabled:
                continue
            for column in GROUP_COLUMNS[group]:
                if group in file_groups or group in forced:
                    row[column] = item.values.get(column, 0)
                else:
                    row[column] = ""
        rows.append(row)
    return rows


def write_csv(stream, rows: list[dict[str, object]], columns: list[str]) -> None:
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="解析飞地调试统计日志并生成 CSV, 每个飞地一行",
    )
    parser.add_argument("logs", nargs="+", help="日志文件或目录 (目录取其中的 *.log)")
    parser.add_argument("-o", "--out", help="合并输出的 CSV 路径; 省略时写到标准输出")
    parser.add_argument(
        "--split",
        action="store_true",
        help="每份日志各生成一份 CSV, 文件名为 <日志名>_stats.csv",
    )
    parser.add_argument("--out-dir", help="--split 的输出目录; 默认为各日志所在目录")
    parser.add_argument(
        "--features",
        default="auto",
        help="输出哪些统计组: auto (默认, 取日志中出现的组) 或 cycle,mem,pmp 的组合",
    )
    parser.add_argument(
        "--name-pattern",
        action="append",
        help="测试项名称正则, 须含 name 组; 可重复. 省略时使用内置的 regress-check 规则",
    )
    parser.add_argument(
        "--name-position",
        choices=("after", "before"),
        default="after",
        help="测试项标记相对统计行的位置 (默认 after)",
    )
    parser.add_argument("--no-names", action="store_true", help="不解析测试项名称")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="存在无法识别的 encl_stat 行时以非零状态退出",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.split and args.out:
        parser.error("--split 与 --out 不能同时使用")
    if args.out_dir and not args.split:
        parser.error("--out-dir 只与 --split 一起使用")

    paths = expand_inputs(args.logs)
    if not paths:
        print("错误: 没有可解析的日志文件", file=sys.stderr)
        return 2

    try:
        patterns = () if args.no_names else compile_patterns(args.name_pattern)
        probe = [parse_log(path, patterns, args.name_position) for path in paths]
        present = {group for parsed in probe for item in parsed.items for group in item.groups}
        enabled = resolve_enabled(args.features, present)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    columns = build_columns(enabled)
    forced = set() if args.features == "auto" else enabled

    if args.split:
        out_dir = Path(args.out_dir) if args.out_dir else None
        for parsed in probe:
            destination = (out_dir or parsed.path.parent) / f"{parsed.path.stem}_stats.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8", newline="") as stream:
                write_csv(stream, build_rows(parsed, enabled, forced), columns)
            print(f"  -> {destination}", file=sys.stderr)
    else:
        rows = [row for parsed in probe for row in build_rows(parsed, enabled, forced)]
        if args.out:
            destination = Path(args.out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8", newline="") as stream:
                write_csv(stream, rows, columns)
            print(f"  -> {destination}", file=sys.stderr)
        else:
            write_csv(sys.stdout, rows, columns)

    items = sum(len(parsed.items) for parsed in probe)
    skipped = sum(parsed.skipped for parsed in probe)
    print(
        f"已解析 {items} 个飞地, {len(probe)} 份日志; 未识别的统计行 {skipped}",
        file=sys.stderr,
    )
    if skipped:
        print("提示: 未识别的行可能是格式变更, 用 --strict 可将其视为错误", file=sys.stderr)
    return 1 if args.strict and skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
