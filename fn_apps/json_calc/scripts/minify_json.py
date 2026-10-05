#!/usr/bin/env python3
"""将缩进/换行格式化的 JSON 压缩为单行 (浏览器与 API 实际接收的 minified 形态).

病态输入场景: 机器间传输的 JSON 常被整体压成一行、不含任何字符串外的空白字符,
解析器须连续处理长串 token 而无换行锚点。本脚本用 json 模块解析后以
separators=(',', ':') 重排, 抹去所有字符串外的空白, 输出单行文本; 往返解析所得结果与
原始文档相等, 但文本从多行缩进变为单行, 供 cJSON
解析 churn 以浏览器形态压测。

用法: minify_json.py <input.json> <output.min.json>
"""

import json
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(f"用法: {sys.argv[0]} <input.json> <output.min.json>", file=sys.stderr)
        return 2
    src_path, dst_path = sys.argv[1], sys.argv[2]
    with open(src_path, "r", encoding="utf-8") as src_file:
        data = json.load(src_file)
    minified = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    with open(dst_path, "w", encoding="utf-8") as dst_file:
        dst_file.write(minified)
        dst_file.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
