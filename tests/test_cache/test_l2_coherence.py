"""Regression: L2 cache ->bytearray coherence for native batch.

Verifies that data written through the Python L2 cache path is correctly
visible to the Rust native batch engine, and vice versa.
"""

from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig


def _make_emu(ram_size_mb: int = 32):
    cfg = PlatformConfig(num_harts=1, ram_size=ram_size_mb * 1024 * 1024)
    return Emulator(cfg)


def test_python_write_visible_to_native_read():
    """Data written via bus.write() (L2 path) must be readable by native batch."""
    emu = _make_emu()

    # Write a 64-bit value through the L2 cache
    test_pa = emu.bus.ram_base + 0x1000
    test_val = 0xDEAD_BEEF_CAFE_BABE
    emu.bus.write(test_pa, test_val.to_bytes(8, byteorder="little"))

    # Flush L2 so Rust batch sees it
    emu.bus.flush_l2()

    # Verify via Python direct read (bypasses L2)
    direct = emu.bus._ram_read_direct(test_pa, 8)
    assert direct == test_val.to_bytes(8, byteorder="little"), (
        f"Direct read after flush: expected {test_val:#018x},"+
        f"got {int.from_bytes(direct, 'little'):#018x}"
    )

    # Verify via L2 read
    l2_read = emu.bus.read(test_pa, 8)
    assert l2_read == test_val.to_bytes(8, byteorder="little"), (
        f"L2 read after write: expected {test_val:#018x}"
    )


def test_l2_flush_preserves_full_64bit_pointer():
    """A 64-bit pointer with upper bits set must survive L2 flush/invalidate."""
    emu = _make_emu()

    # Simulate a typical 64-bit pointer (like ld-linux / libc addresses)
    pointers = [
        0x0000_003F_F7FD_FAE8,  # high pointer (full 64-bit)
        0x0000_003F_F7EC_B76A,  # libc crash address from actual session
        0x0000_0000_8104_AAE8,  # PA-style pointer
        0xFFFF_FFFF_FFFF_FFFF,  # all-ones sentinel
        0x0000_0000_0000_0001,  # low pointer
    ]

    for pa_offset, ptr in enumerate(pointers):
        pa = emu.bus.ram_base + 0x2000 + pa_offset * 8
        data = ptr.to_bytes(8, byteorder="little")

        # Write through L2
        emu.bus.write(pa, data)

        # Flush ->invalidate cycle (simulates one batch)
        emu.bus.flush_l2()
        emu.bus.invalidate_l2()

        # Read back directly from bytearray
        direct = emu.bus._ram_read_direct(pa, 8)
        assert direct == data, (
            f"Ptr {ptr:#018x} corrupted after flush/invalidate: "
            f"expected {data.hex()}, got {direct.hex()}"
        )


def test_multiple_flush_invalidate_cycles():
    """Repeated flush/invalidate cycles must not corrupt RAM."""
    emu = _make_emu()

    pa = emu.bus.ram_base + 0x3000
    ptr = 0x3FF7_FD00_1234  # realistic high pointer

    for cycle in range(20):
        # Simulate Python writing data
        data = (ptr + cycle).to_bytes(8, byteorder="little")
        emu.bus.write(pa, data)
        emu.bus.flush_l2()

        # Verify direct read
        direct = emu.bus._ram_read_direct(pa, 8)
        assert direct == data, f"Cycle {cycle}: flush corrupted data"

        emu.bus.invalidate_l2()

        # Verify after invalidate (should still be correct)
        direct = emu.bus._ram_read_direct(pa, 8)
        assert direct == data, f"Cycle {cycle}: invalidate corrupted data"


def test_l2_write_straddling_cache_lines():
    """An 8-byte write spanning two L2 cache lines must be correctly flushed."""
    emu = _make_emu()

    # Place write near a 64-byte boundary
    # L2 line size is typically 64 bytes
    pa = emu.bus.ram_base + 0x1000 + 60  # 4 bytes from line boundary
    data = b"\x11\x22\x33\x44\x55\x66\x77\x88"

    emu.bus.write(pa, data)
    emu.bus.flush_l2()

    direct = emu.bus._ram_read_direct(pa, 8)
    assert direct == data, f"Cross-line write corrupted: got {direct.hex()}"
