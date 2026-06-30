#!/usr/bin/env python3
"""cProfile 基准: 与 makefile `emu` 目标完全一致的启动路径。

用法:
  python tools/profile_emu.py              # 运行 cProfile → tools/profile_emu.prof
  snakeviz tools/profile_emu.prof          # 浏览器可视化
  python tools/profile_emu.py --tottime    # 按自身时间排序 (默认 cumulative)
"""
import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyremu.emulator import Emulator  # noqa: E402
from pyremu.env_inject.preload import Preloader  # noqa: E402
from pyremu.platform import PeripheralConfig, PlatformConfig  # noqa: E402
from pyremu.utils.parse_bin import parse_firmware  # noqa: E402

# 与 makefile 中 `emu` 目标一致的参数
EMU_ARGS = [
    "--ram-base=0x80000000",
    "--preload=tests/bins/firm-bin/zsbl_fsbl_stub_cold_asm.bin",
    "--hart=1",
    "tests/bins/elf/custom_opensbi_fw_payload.elf",
]
CYCLES = 1_000_000
OUTPUT = ROOT / "tools" / "profile_emu.prof"


def run_emu_once() -> None:
    """仿真 CYCLES 步，不做 REPL 交互。"""
    # 直接调用 Emulator 而非 Debugger REPL，保持初始化完全一致
    # -- 参数解析 (与 debugger.main 相同) --
    parser = argparse.ArgumentParser()
    parser.add_argument("firmware")
    parser.add_argument("--ram-base", type=lambda x: int(x, 0), default=0x80000000)
    parser.add_argument("--hart", type=int, default=1)
    parser.add_argument("--preload", type=str, default=None)
    ns = parser.parse_args(EMU_ARGS)

    ram_size = 128 * 1024 * 1024  # 128 MiB (debugger 默认)

    # -- 加载固件 (与 debugger 完全一致) --
    image = parse_firmware(ns.firmware)
    assert image is not None, f"无法解析固件: {ns.firmware}"

    # PIE 固件搬迁
    load_offset = 0
    if image.format == "elf":
        min_vaddr = min(seg.vaddr for seg in image.segments)
        if min_vaddr < ns.ram_base:
            load_offset = ns.ram_base

    effective_entry = image.entry_point + load_offset
    prog_cnt = effective_entry

    # -- 创建 Emulator --
    plat_cfg = PlatformConfig(
        num_harts=ns.hart,
        ram_size=ram_size,
        ram_base=ns.ram_base,
        prog_cnt=prog_cnt,
        periph=PeripheralConfig(),
    )
    emu = Emulator(plat_cfg)
    emu.load_firmware(image, load_offset=load_offset)
    for h in emu.harts:
        h.pc = prog_cnt

    # -- FDT --
    fdt_addr = ns.ram_base + ram_size - 0x10000
    emu.load_dtb(fdt_addr)

    # -- preload --
    preload_entry = None
    if ns.preload is not None:
        preloader = Preloader(emu)
        preload_entry = preloader.inject_file(ns.preload)
        for h in emu.harts:
            h.csrs["misa"].val = (2 << 62) | (1 << 18) | (1 << 20)  # MXL=RV64 + U-bit
        emu.load_dtb_blob(ns.ram_base + 0x2200000, emu.build_dtb())

    if preload_entry is not None:
        for h in emu.harts:
            h.pc = preload_entry

    # -- 执行 (不进入 REPL) --
    emu.run(CYCLES)


def main() -> None:
    prof = cProfile.Profile()
    prof.enable()
    try:
        run_emu_once()
    finally:
        prof.disable()

    # 写出完整 profile 供可视化
    prof.dump_stats(str(OUTPUT))

    sio = io.StringIO()
    st = pstats.Stats(prof, stream=sio)
    st.sort_stats("cumulative")
    st.print_stats(40)
    print(sio.getvalue())
    print(f"\n完整 profile: {OUTPUT}")
    print("可视化: snakeviz tools/profile_emu.prof")


if __name__ == "__main__":
    main()
