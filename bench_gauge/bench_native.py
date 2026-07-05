#!/usr/bin/env python3
"""Benchmark Rust native acceleration vs pure Python fallback.

Mimics ``make emu``: ZSBL cold-boot → OpenSBI fw_payload → WFI idle.
Measures instructions executed until WFI (or max instructions), reports
instructions/second.

Usage:
    python bench_gauge/bench_native.py [--max-instrs N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import parse_firmware

FW_PAYLOAD = str(_PROJECT / "tests/bins/elf/custom_opensbi_fw_payload.elf")
ZSBL_BIN = str(_PROJECT / "tests/bins/firm-bin/zsbl_fsbl_stub_cold_asm.bin")
RAM_BASE = 0x8000_0000
RAM_SIZE = 128 * 1024 * 1024
MAX_INSTRS = 2_000_000


def setup_emulator(elf_path: str) -> Emulator:
    """Configure Emulator — mirrors ``make emu`` setup."""
    cfg = PlatformConfig.qemu_virt()
    cfg.num_harts = 1
    cfg.ram_base = RAM_BASE
    cfg.ram_size = RAM_SIZE
    emu = Emulator(cfg)

    # Parse firmware
    image = parse_firmware(elf_path, base_addr=RAM_BASE)
    if image is None:
        raise SystemExit(f"Failed to parse: {elf_path}")

    load_offset = 0
    if image.format == "elf":
        min_vaddr = min(seg.vaddr for seg in image.segments)
        if min_vaddr < RAM_BASE:
            load_offset = RAM_BASE
    emu.load_firmware(image, load_offset=load_offset)

    # Preload ZSBL stub for cold boot
    preload_data = Path(ZSBL_BIN).read_bytes()
    preload_addr = RAM_BASE + RAM_SIZE - 0x10000  # top of RAM - 64K
    emu.bus.write(preload_addr, preload_data)
    for h in emu.harts:
        h.csrs["misa"].val = (2 << 62) | (1 << 18) | (1 << 20)
        h.pc = preload_addr

    return emu


def run_bench(elf_path: str, max_instrs: int, label: str) -> dict:
    """Run until WFI/halt or *max_instrs* reached."""
    emu = setup_emulator(elf_path)
    h = emu.harts[0]

    t0 = time.perf_counter()
    executed = 0
    for _ in range(max_instrs):
        # After WFI, step() blocks in _wfi_sleep_if_idle — detect idle
        # BEFORE calling step() to avoid blocking forever.
        if h._waiting or h._halted:
            break
        emu.step()
        executed += 1
    elapsed = time.perf_counter() - t0

    ips = executed / elapsed if elapsed > 0 else 0
    status = "WFI" if h._waiting else ("halted" if h._halted else f"max ({executed})")
    print(
        f"  [{label}] {executed:,} instrs  {elapsed:.1f}s  →  "
        f"{ips:,.0f} instr/s  ({status}, pc={h.pc:#010x})"
    )
    return {"label": label, "executed": executed, "elapsed_s": elapsed, "instr_per_sec": ips}


def main():
    ap = argparse.ArgumentParser(description="Benchmark Rust native vs Python fallback")
    ap.add_argument("--max-instrs", type=int, default=MAX_INSTRS)
    ap.add_argument("--elf", type=str, default=FW_PAYLOAD)
    args = ap.parse_args()

    print(f"ELF       : {args.elf}")
    print(f"ZSBL      : {ZSBL_BIN}")
    print(f"ram_base  : {RAM_BASE:#x}")
    print(f"max instrs: {args.max_instrs:,}")
    print()

    results = []

    # ---- Native ----
    print("--- Native (Rust .so) ---")
    r = run_bench(args.elf, args.max_instrs, "native")
    results.append(r)

    # ---- Pure Python ----
    print()
    print("--- Pure Python (.so hidden) ---")
    so = _PROJECT / "pyremu/_native/libdecode.so"
    backup = so.with_name("libdecode.so.bench_off")
    if so.exists():
        so.rename(backup)
        try:
            for k in list(sys.modules):
                if any(
                    k.startswith(pfx)
                    for pfx in (
                        "pyremu._native",
                        "pyremu.core",
                        "pyremu.memory",
                        "pyremu.emulator",
                        "pyremu.peripheral",
                        "pyremu.interrupt",
                    )
                ):
                    del sys.modules[k]
            r = run_bench(args.elf, args.max_instrs, "python")
            results.append(r)
        finally:
            backup.rename(so)
    else:
        print("  WARNING: .so not found — native run already pure Python!")

    # ---- Summary ----
    if len(results) == 2:
        n, p = results
        if n["executed"] > 0 and p["executed"] > 0:
            ratio = n["instr_per_sec"] / p["instr_per_sec"]
            print()
            print("=" * 55)
            print(f"  Native : {n['instr_per_sec']:>12,.0f} instr/s")
            print(f"  Python : {p['instr_per_sec']:>12,.0f} instr/s")
            print(f"  Speedup: {ratio:>12.2f}x")
            print("=" * 55)


if __name__ == "__main__":
    main()
