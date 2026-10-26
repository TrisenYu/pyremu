#!/usr/bin/env bash
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
#
# 把 rootfs 目录打包为 gzip 压缩的 newc cpio, 供调试器以 --initrd 加载。
#
# 用法:
#   tools/build_initramfs.sh <rootfs 目录> <输出文件>
#
# 输出文件取相对路径时按调用者的工作目录换算, 打包过程改换工作目录不影响结果;
# 输出文件的上级目录不存在时创建。fakeroot 可用时经它打包, 使归档中的文件所有者
# 与权限取 rootfs 目录当前的值; 不可用时直接打包并报告。

set -euo pipefail

rootfs_dir=${1:?缺少 rootfs 目录}
out_file=${2:?缺少输出文件}

if [ ! -d "$rootfs_dir" ]; then
	echo "缺少 rootfs 目录: ${rootfs_dir} (先运行 debootstrap)" >&2
	exit 1
fi

for tool in cpio gzip; do
	if ! command -v "$tool" >/dev/null 2>&1; then
		echo "缺少 ${tool}" >&2
		exit 1
	fi
done

out_file=$(realpath --canonicalize-missing -- "$out_file")
mkdir -p "$(dirname "$out_file")"

fake=""
if command -v fakeroot >/dev/null 2>&1; then
	fake=fakeroot
else
	echo "[warn] 无 fakeroot"
fi

(
	cd "$rootfs_dir"
	find . -print0 | $fake cpio --null -o --format=newc 2>/dev/null
) | gzip -9 > "$out_file"

echo "initramfs 已生成: ${out_file} ($(du -h "$out_file" | cut -f1))"
