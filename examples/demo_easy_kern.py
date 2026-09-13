#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建:
    make -C examples/easy_kern     # 生成 examples/easy_kern/easy_kern.bin
    make -C bsp/custom-opensbi     # 生成 build/elf/custom_opensbi_fw_jump.elf

运行:
    ulimit -v 4194304 && timeout 300 env PYTHONPATH=. \
        uv run python examples/demo_easy_kern.py

预期输出关键行 (全部经 UART 输出到 stdout):
    [easy_kern] create returned id = 1
    [enclave] entry=0x... sp=0x...   (运行时加载载荷的提示)
    cfrac 打印的因子分解结果
    [easy_kern] demo done, entering wfi
"""

from pathlib import Path
import sys

from pyremu.emulator import Emulator, TimeoutError
from pyremu.env_inject import Preloader
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils.parse_bin import parse_firmware

# 与 bsp 侧 FW_JUMP_ADDR / --ram-base 一致
RAM_BASE = 0x8000_0000
RAM_SIZE = 2 * 1024**3
FW_JUMP_ADDR = 0x8020_0000


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fw_path = root / "build" / "elf" / "custom_opensbi_fw_jump.elf"
    bin_path = root / "examples" / "easy_kern" / "easy_kern.bin"
    for p in (fw_path, bin_path):
        if not p.exists():
            print(f"缺少构建产物: {p}\n请先执行 make 构建 (见文件头部说明)")
            sys.exit(1)

    # 解析 fw_jump (静态 PIE, 段链接于 0): 段整体搬移到 RAM 起始处
    image = parse_firmware(str(fw_path))
    if image is None:
        print(f"无法解析固件: {fw_path}")
        sys.exit(1)
    min_vaddr = min(seg.vaddr for seg in image.segments)
    load_offset = RAM_BASE if min_vaddr < RAM_BASE else 0

    plat_cfg = PlatformConfig(
        num_harts=1,
        ram_size=RAM_SIZE,
        ram_base=RAM_BASE,
        periph=PeripheralConfig(),
    )
    emu = Emulator(plat_cfg)
    emu.load_firmware(image, load_offset=load_offset)

    # easy_kern 预载到 fw_jump 的跳转目标 (FW_JUMP_ADDR)
    Preloader(emu).inject_file(str(bin_path), addr=FW_JUMP_ADDR)

    # DTB 加载到 RAM 顶端, 地址写入 a1, 供 OpenSBI 固件平台解析
    emu.load_dtb(RAM_BASE + RAM_SIZE - 0x10000)

    print("=== 冷启动 fw_jump, 等待 easy_kern 驱动飞地生命周期 ===", flush=True)
    try:
        emu.run(timeout=300.0, yield_every=0)
    except TimeoutError:
        print("... 超时 (300 s)", flush=True)
    print("stop_reason:", getattr(emu, "_run_stop_reason", None), flush=True)
    print("=== 运行结束 ===", flush=True)


if __name__ == "__main__":
    main()
