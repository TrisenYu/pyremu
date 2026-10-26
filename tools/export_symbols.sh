#!/usr/bin/env bash
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
#
# 把带符号表的 ELF 汇集到一个目录, 供 rvdb 的 sym chdir / sym ls / sym load 使用。
#
# 用法:
#   tools/export_symbols.sh <输出目录> <源>[:<名称>] ...
#
# 源为普通文件时建立一项, 名称省略时取文件名。
# 源为目录时链接目录内每个普通文件, 名称为 <名称><文件名>, 名称省略时取文件名。
# 建立的是指向构建产物的相对符号链接: 产物重新链接后, 目录中的内容随之更新。
# 源不存在时跳过并报告, 不视为错误。

set -euo pipefail

out_dir=${1:?did not properly set the output directory}
shift

mkdir -p "$out_dir"

linked=0
skipped=0

link_one() {
	local src=$1
	local dst_name=$2
	if [ ! -f "$src" ]; then
		echo "  skip ${dst_name} because ${src} is not built"
		skipped=$((skipped + 1))
		return
	fi
	local rel
	rel=$(realpath --relative-to="$out_dir" -- "$src")
	ln -sfn "$rel" "$out_dir/$dst_name"
	linked=$((linked + 1))
}

for spec in "$@"; do
	src=${spec%%:*}
	name=${spec#*:}
	if [ "$src" = "$name" ]; then
		name=""
	fi
	if [ -d "$src" ]; then
		for f in "$src"/*; do
			[ -f "$f" ] || continue
			link_one "$f" "${name}$(basename "$f")"
		done
	else
		link_one "$src" "${name:-$(basename "$src")}"
	fi
done

echo "export [${linked}] symbols file(s) to  ${out_dir}, skip ${skipped}"
