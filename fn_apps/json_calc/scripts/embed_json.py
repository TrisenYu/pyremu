#!/usr/bin/env python3
"""将多个输入 JSON 文件无损编码为 C 字符串数组, 供载荷内嵌解析使用.

生成的 .inc 形如:

    static const char *const json_inputs[] = {
        "...",   // 第一个输入文件
        "...",   // 第二个输入文件
        "...",   // 第一份语法错误文档 (追加在最末)
    };

    /* 与 json_inputs 逐项对应 */
    static const unsigned char json_expect_ok[] = {
        1, 1, 0,
    };

每个文件按物理行逐行转义并拼接, 引号/反斜杠等任意合法 JSON 字符均可内嵌;
换行以 \\n 保留, 各数组元素由 C 编译器自动以 NUL 结尾. 载荷遍历 json_inputs
即可对每份文档反复 Parse/Delete, 构成 churn 压力; 新增输入只须在命令行追加
一个文件, 无需改动载荷代码.

语法错误的文档放在输入目录下的 bad/ 子目录, 由本脚本自动收录并追加在合法输入
之后 (既有下标不变, 沿用原有下标语义): 它们无法参与任何解析类操作 (压成单行、
往返比对都要求文档合法), 故不经 minify_json.py, 只按原文读为字符串. 两类文档
同处一个 json_inputs 数组, 由逐项对应的 json_expect_ok 区分: 载荷据此对合法
文档断言解析成功, 对语法错误文档断言解析被拒 —— 后者同样压测失败路径上的分配
与回收 (cJSON 释放部分构造的树 / 抛出的异常).

用法: embed_json.py <output.inc> <input1.json> [<input2.json> ...]
"""

from pathlib import Path
import sys


def c_escape(line: str) -> str:
    """转义一行文本为 C 字符串字面量内容 (保留除引号/反斜杠/制表符外的字符)."""
    out: list[str] = []
    for ch in line:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    return "".join(out)


def emit_element(content: str) -> str:
    """将一个文件的完整内容编码为单个数组元素 (相邻字面量由 C 拼接)."""
    lines = content.splitlines()
    return "\n".join('        "' + c_escape(line) + '\\n"' for line in lines)


def read_element(path: Path) -> str:
    """按原文读取一个文件 (不作解析), 编码为单个数组元素."""
    return emit_element(path.read_text(encoding="utf-8"))


def main() -> int:
    if len(sys.argv) < 3:
        print(
            f"用法: {sys.argv[0]} <output.inc> <input1.json> [<input2.json> ...]",
            file=sys.stderr,
        )
        return 2
    dst_path = Path(sys.argv[1])
    src_paths = [Path(p) for p in sys.argv[2:]]

    # 语法错误的文档: 与合法输入同目录下的 bad/ 子目录, 按文件名排序收录.
    # 原样读取, 不参与 minify 与往返比对。
    bad_paths = sorted((src_paths[0].parent / "bad").glob("*.json"))
    elements = [read_element(p) for p in src_paths + bad_paths]
    expect_ok = [1] * len(src_paths) + [0] * len(bad_paths)

    with dst_path.open("w", encoding="utf-8") as dst_file:
        dst_file.write("static const char *const json_inputs[] = {\n")
        dst_file.write(",\n".join(elements))
        dst_file.write(",\n};\n\n")
        dst_file.write(
            "/* 与 json_inputs 逐项对应: 1 = 合法文档 (解析必须成功), "
            "0 = 语法错误文档 (解析必须被拒) */\n"
        )
        dst_file.write("static const unsigned char json_expect_ok[] = {\n")
        dst_file.write("        " + ", ".join(str(f) for f in expect_ok))
        dst_file.write(",\n};\n")
    print(f"内嵌 {len(src_paths)} 份合法输入 + {len(bad_paths)} 份语法错误文档 -> {dst_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
