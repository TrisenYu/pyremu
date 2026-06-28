#!/usr/bin/env python3
"""Fast stack-write tracer — patches exec_instr on hart, uses emu.run().

Usage: uv run python tools/trace_stack_corruption.py
"""

import sys
import types
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))

from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import parse_firmware
from pyremu.env_inject.preload import Preloader


def main():
    ram_base = 0x8000_0000
    ram_size = 0x1000_0000

    cfg = PlatformConfig.qemu_virt()
    cfg.ram_base = ram_base
    cfg.ram_size = ram_size
    cfg.num_harts = 1

    emu = Emulator(cfg)

    preload_path = _PROJ / "tests/bins/firm-bin/zsbl_fsbl_stub_cold_asm.bin"
    preloader = Preloader(emu)
    preload_entry = preloader.inject_file(str(preload_path))

    fw_path = _PROJ / "tests/bins/elf/custom_opensbi_fw_payload.elf"
    image = parse_firmware(str(fw_path))
    min_vaddr = min(seg.vaddr for seg in image.segments)
    load_offset = ram_base if min_vaddr < ram_base else 0
    emu.load_firmware(image, load_offset=load_offset)

    for h in emu.harts:
        h.pc = preload_entry

    emu.load_dtb_blob(ram_base + 0x2200000, emu.build_dtb())

    hart = emu.harts[0]

    # Key firmware addresses (physical)
    PC_SW_LENP = ram_base + 0x187C4
    PC_FDT_DRIVER_INIT = ram_base + 0x1983E
    PC_LW_PROPLEN_1 = ram_base + 0x1989C
    PC_LW_PROPLEN_2 = ram_base + 0x198AA
    PC_SW_PROPLEN = ram_base + 0x198FA

    state = {
        "fdt_driver_init_count": 0,
        "loop_guard": 0,
        "last_sw_lenp_val": 0,
        "last_sw_lenp_addr": 0,
        "running": True,
    }

    original_exec = hart.exec_instr

    def traced_exec(self, instr):
        pc = self.pc

        if pc == PC_SW_LENP:
            val = self.gprs[11]
            addr = self.gprs[18]
            state["last_sw_lenp_val"] = val
            state["last_sw_lenp_addr"] = addr
            if state["fdt_driver_init_count"] > 0:
                print(f"[c={emu._cycle:5d}] SW *lenp: *0x{addr:016x} = 0x{val:08x} ({val:>12d})")

        if pc == PC_FDT_DRIVER_INIT:
            state["fdt_driver_init_count"] += 1

        if pc == PC_LW_PROPLEN_1:
            s0 = self.gprs[8]
            proplen_addr = (s0 - 0x6C) & 0xFFFF_FFFF_FFFF_FFFF
            data = emu.bus.try_read(proplen_addr, 4)
            if data is not None:
                proplen = int.from_bytes(data, "little")
            else:
                proplen = -1
            print(f"[c={emu._cycle:5d}] LW proplen(@0x1989C): s0=0x{s0:016x}  "
                  f"*0x{proplen_addr:016x} = 0x{proplen:08x} ({proplen})  "
                  f"call=#{state['fdt_driver_init_count']}")

        if pc == PC_LW_PROPLEN_2:
            state["loop_guard"] += 1
            if state["loop_guard"] > 300:
                s0 = self.gprs[8]
                proplen_addr = (s0 - 0x6C) & 0xFFFF_FFFF_FFFF_FFFF
                data = emu.bus.try_read(proplen_addr, 4)
                if data is not None:
                    proplen = int.from_bytes(data, "little")
                else:
                    proplen = -1
                print(f"\n[cycle {emu._cycle}] LOOP GUARD: "
                      f"proplen=0x{proplen:08x} ({proplen})  s0=0x{s0:016x}")
                print(f"  Stack frame [s0-0x80, s0]:")
                for off in range(0, 0x80, 8):
                    addr = (s0 - 0x80 + off) & 0xFFFF_FFFF_FFFF_FFFF
                    d = emu.bus.try_read(addr, 8)
                    if d is not None:
                        v = int.from_bytes(d, "little")
                        m = " <-- proplen" if addr == proplen_addr else ""
                        print(f"    0x{addr:016x}: 0x{v:016x}{m}")
                state["running"] = False

        elif pc not in (PC_LW_PROPLEN_2,):
            state["loop_guard"] = 0

        if pc == PC_SW_PROPLEN:
            val = self.gprs[11]
            s0 = self.gprs[8]
            proplen_addr = (s0 - 0x6C) & 0xFFFF_FFFF_FFFF_FFFF
            print(f"[c={emu._cycle:5d}] SW proplen: s0=0x{s0:016x}  "
                  f"*0x{proplen_addr:016x} = 0x{val:08x} ({val})")

        # Check if we need to stop early
        if not state["running"]:
            return 0

        return original_exec(instr)

    hart.exec_instr = types.MethodType(traced_exec, hart)

    print("Fast tracing with emu.run()...")
    try:
        emu.run(100000, timeout=120.0, yield_every=0)
    except Emulator.TimeoutError as e:
        print(f"Timeout at cycle {emu._cycle}, pc=0x{e.pc:x}" if e.pc else f"Timeout at cycle {emu._cycle}")
    except Exception as e:
        print(f"Exception: {e}")

    print(f"\nDone. {emu._cycle} cycles, fdt_driver_init called {state['fdt_driver_init_count']} times")


if __name__ == "__main__":
    main()
