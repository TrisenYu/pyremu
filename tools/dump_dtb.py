#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成平台设备树 DTB。

用指定平台预设构造 Emulator, 调用 build_dtb() 得到完整 DTB blob, 写入输出路径
(默认 build/emu.dtb)。配合根 makefile 的 `dtb` 目标: 先生成 DTB, 再用
`dtc -I dtb -O dts` 还原为人类可读的 DTS。

本脚本由 makefile 以 `uv run python tools/dump_dtb.py` 的方式调用。
"""

import argparse
from pathlib import Path

from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="生成平台设备树 DTB 并落盘")
    parser.add_argument(
        "out",
        nargs="?",
        default="build/emu.dtb",
        help="输出 DTB 路径 (默认 build/emu.dtb)",
    )
    parser.add_argument(
        "--preset",
        default="qemu_virt",
        help="平台预设名 (qemu_virt / sifive_u54 / minimal / qemu_virt_aia)",
    )
    args = parser.parse_args()

    factory = getattr(PlatformConfig, args.preset, None)
    if factory is None or not callable(factory):
        parser.error(f"未知平台预设: {args.preset}")

    emu = Emulator(factory())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = emu.build_dtb()
    out.write_bytes(blob)
    print(f"  -> {out} ({len(blob)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
