from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import parse_firmware
import time, sys

RAM_BASE = 0x80000000; RAM_SIZE = 128 * 1024 * 1024
BASE = "/home/hammer/projects/pyremu"
cfg = PlatformConfig(num_harts=1, ram_base=RAM_BASE, ram_size=RAM_SIZE)
emu = Emulator(cfg)
fw = parse_firmware(f"{BASE}/tests/bins/elf/custom_opensbi_fw_payload.elf")
emu.load_firmware(fw, load_offset=RAM_BASE)
emu.load_dtb(RAM_BASE + RAM_SIZE - 0x10000)
with open(f"{BASE}/tests/bins/firm-bin/zsbl_fsbl_stub_cold_asm.bin", "rb") as f:
    emu.bus.write(RAM_BASE + RAM_SIZE - 0x20000, f.read())
for h in emu.harts:
    h.csrs["misa"].val = (2 << 62) | (1 << 18) | (1 << 20)
emu.load_dtb_blob(RAM_BASE + 0x2200000, emu.build_dtb())
for h in emu.harts:
    h.pc = RAM_BASE + RAM_SIZE - 0x20000
for _ in range(16): emu.step()
hart = emu.harts[0]
start = time.time()
last_txt = b""
for i in range(10000000):
    emu.step()
    if i % 500000 == 0:
        uart = hart._bus._devices.get(0x10000000)
        txt = uart.tx_data() if uart else b""
        t = time.time() - start
        if txt != last_txt:
            print(f"[{i//1000000}M {t:.0f}s] {repr(txt[-300:])}", flush=True)
            last_txt = txt
            if b"OpenSBI" in txt:
                print(f"\nSUCCESS! No FPU, with zicsr/zifencei. {i} steps ({t:.0f}s)")
                sys.exit(0)
        else:
            print(f"[{i//1000000}M {t:.0f}s] PC={hex(hart.pc)}", flush=True)
        if hart._halted:
            print(f"HALTED: mcause={hart.mcause_val}"); break
print(f"Final: PC={hex(hart.pc)}")
