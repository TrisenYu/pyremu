#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""L2 cache stress tests — simulate firmware access patterns (DTB + BSS memset)."""


from pyremu.memory.l2cache import L2Cache


def _make_ram(size=256 * 1024):
    """Construct a simulated physical RAM bytearray with read/write callbacks."""
    ram = bytearray(size)

    def read_fn(addr, size):
        # addr is relative to the RAM array
        if addr < 0 or addr + size > len(ram):
            return b"\x00" * size
        return bytes(ram[addr : addr + size])

    def write_fn(addr, data):
        if addr < 0 or addr + len(data) > len(ram):
            return  # silently drop out-of-range writes
        for i, b in enumerate(data):
            ram[addr + i] = b

    return ram, read_fn, write_fn


# ---------------------------------------------------------------
# Tests use address-offset into the RAM array (not full PA).
# Cache sees the offset value as the address; _ram_read/_ram_write
# treat the offset directly as the RAM index.
# This is equivalent to ram_base=0, ram_size=256 KiB.
# ---------------------------------------------------------------


class TestCacheWriteAllocateThenRead:
    """Write-allocate: write to an address (cache miss), then read it back."""

    def test_single_write_allocate_read_back(self):
        """Write data to a new address → should be readable immediately."""
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)

        l2.write(0x1000, b"\xDE\xAD\xBE\xEF")
        data = l2.read(0x1000, 4)
        assert data == b"\xDE\xAD\xBE\xEF", f"Expected 0xDEADBEEF, got {data.hex()}"

    def test_write_then_read_partial(self):
        """Write 8 bytes, then read 4 at offset 2."""
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)

        l2.write(0x1000, b"\x01\x02\x03\x04\x05\x06\x07\x08")
        data = l2.read(0x1002, 4)
        assert data == b"\x03\x04\x05\x06", f"Expected 03040506, got {data.hex()}"

    def test_write_multiple_same_line(self):
        """Multiple writes to the same cache line (at different offsets)."""
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)

        l2.write(0x1000, b"\xAA" * 8)
        l2.write(0x1010, b"\xBB" * 8)
        l2.write(0x1030, b"\xCC" * 8)

        assert l2.read(0x1000, 8) == b"\xAA" * 8
        assert l2.read(0x1010, 8) == b"\xBB" * 8
        assert l2.read(0x1030, 8) == b"\xCC" * 8


class TestCacheEvictionIntegrity:
    """Verify data integrity after eviction and re-load."""

    def test_dirty_line_survives_eviction(self):
        """Write to address A, evict it by filling the same set, re-read A."""
        ram, rf, wf = _make_ram()
        # 1 set × 2 ways → easy to force eviction
        l2 = L2Cache(size=128, line_size=64, ways=2)
        l2.set_ram_backend(rf, wf)

        # Write data at 0x0000 (set 0, way 0)
        l2.write(0x0000, b"\xFE\xED\xFA\xCE\xCA\xFE\xBE\xEF")
        # Write data at 0x0040 (same set, way 1) — fills both ways
        l2.write(0x0040, b"\x11" * 8)
        # Now write at 0x0080 (same set) — must evict way 0 (or 1)
        l2.write(0x0080, b"\x22" * 8)
        # Write at 0x00C0 — must evict the other way
        l2.write(0x00C0, b"\x33" * 8)

        # Now re-read 0x0000 — should be a miss, load from RAM (writeback data)
        data = l2.read(0x0000, 8)
        assert data == b"\xFE\xED\xFA\xCE\xCA\xFE\xBE\xEF", (
            f"Dirty eviction failed: expected FEEDFACECAFEBEEF, got {data.hex()}"
        )

    def test_many_writes_dont_corrupt_earlier_data(self):
        """Thrash the cache with many writes, verify earlier data survives."""
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=4)  # 16 sets × 4 ways
        l2.set_ram_backend(rf, wf)

        # Write known data at a specific address
        marker_addr = 0x5000
        l2.write(marker_addr, b"\xAB" * 64)

        # Thrash: write to many addresses (simulates BSS memset),
        # skipping the marker address so we test cache eviction+reload, not overwrite.
        marker_end = marker_addr + 64
        for addr in range(0x0000, 0x10000, 64):
            if marker_addr <= addr < marker_end:
                continue
            l2.write(addr, b"\x00" * 64)

        # Verify marker data is still correct (should survive via writeback+reload)
        data = l2.read(marker_addr, 64)
        assert data == b"\xAB" * 64, (
            f"Data corrupted after thrashing at {marker_addr:#x}: "
            f"first 8 bytes = {data[:8].hex()}"
        )

    def test_dtb_like_pattern(self):
        """Simulate DTB + BSS memset pattern: write small data, then thrash cache."""
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)

        # Simulate DTB data at the "top" of memory
        dtb_addr = 0x3F000
        dtb_data = bytes([i & 0xFF for i in range(256)])  # 256 bytes of known pattern
        for off in range(0, len(dtb_data), 8):
            chunk = dtb_data[off : off + 8]
            l2.write(dtb_addr + off, chunk)

        # Simulate BSS memset: write zeros to a large area
        for addr in range(0x00000, 0x20000, 8):
            l2.write(addr, b"\x00" * 8)

        # Verify DTB data is still intact
        for off in range(0, len(dtb_data), 4):
            expected = dtb_data[off : off + 4]
            actual = l2.read(dtb_addr + off, 4)
            assert actual == expected, (
                f"DTB data corrupted at offset {off:#x}: "
                f"expected {expected.hex()}, got {actual.hex()}"
            )

    def test_interleaved_read_write_integrity(self):
        """Interleave reads from 'DTB' area with writes to 'BSS' area."""
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)

        # Setup: write DTB data
        dtb_addr = 0x30000
        l2.write(dtb_addr, b"\x03\x00\x00\x00\x04\x00\x00\x00")  # FDT_PROP tag + len=4

        # Interleave: read DTB, write BSS, repeat
        for addr in range(0x00000, 0x10000, 8):
            # Write BSS
            l2.write(addr, b"\x00" * 8)
            # Read DTB (should survive or be reloaded correctly)
            tag_data = l2.read(dtb_addr, 4)
            assert tag_data == b"\x03\x00\x00\x00", (
                f"DTB corrupted at iteration {addr:#x}: tag = {tag_data.hex()}"
            )


class TestCrossCacheLineAccess:
    """Accesses that cross cache line boundaries."""

    def test_write_crossing_cache_line(self):
        """Write 16 bytes at offset 56 of a cache line (crosses boundary)."""
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)

        # offset 56 → 8 bytes in line 0, 8 bytes in line 1
        addr = 0x1038  # 0x1038 & 63 = 56
        data = bytes([0xA0 + i for i in range(16)])
        l2.write(addr, data)

        # Read back
        result = l2.read(addr, 16)
        assert result == data, f"Cross-line write failed: {result.hex()} != {data.hex()}"

    def test_read_crossing_cache_line(self):
        """Read 8 bytes at offset 60 (4 bytes in line 0, 4 bytes in line 1)."""
        ram, rf, wf = _make_ram()
        # Pre-fill RAM
        for i in range(128):
            ram[i] = i & 0xFF
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)

        # Read at offset 60 in line 0 → crosses into line 1
        addr = 0x003C  # 0x3C = 60
        data = l2.read(addr, 8)
        expected = bytes([0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x41, 0x42, 0x43])
        assert data == expected, f"Cross-line read failed: {data.hex()} != {expected.hex()}"

    def test_cross_line_write_then_read_back(self):
        """Write across cache line boundary, thrash, then read back."""
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)

        # Write 32 bytes at offset 48 → crosses cache line
        addr = 0x2030  # offset 48, 16 bytes in line N, 16 bytes in line N+1
        original = bytes([0xE0 + i for i in range(32)])
        l2.write(addr, original)

        # Thrash the cache, but skip the lines containing our data
        # The 32-byte write at 0x2030 spans lines [0x2000, 0x2040) and [0x2040, 0x2080)
        skip_lines = {0x2000, 0x2040}
        for a in range(0x0000, 0x8000, 64):
            if a in skip_lines:
                continue
            l2.write(a, b"\xFF" * 64)

        # Read back
        result = l2.read(addr, 32)
        assert result == original, (
            f"Cross-line data corrupted: first 16 = {result[:16].hex()}, "
            f"expected {original[:16].hex()}"
        )


class TestCacheSetAliasing:
    """Test that addresses mapping to the same set don't corrupt each other."""

    def _get_set_index(self, l2, addr):
        return (addr >> l2._line_shift) & l2._set_mask

    def test_same_set_different_tags_no_corruption(self):
        """Two addresses in the same set with different tags don't alias."""
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)

        # Find two addresses that map to the same set
        set_idx = 3
        line_shift = l2._line_shift
        addr_a = (0x100 << line_shift) | (set_idx << line_shift)
        addr_b = (0x200 << line_shift) | (set_idx << line_shift)
        assert self._get_set_index(l2, addr_a) == set_idx
        assert self._get_set_index(l2, addr_b) == set_idx
        assert (addr_a >> line_shift) != (addr_b >> line_shift)  # different tags

        # Write data to both
        l2.write(addr_a, b"\xAA" * 8)
        l2.write(addr_b, b"\xBB" * 8)

        # Both should be readable (2 ways ≥ 2 addresses)
        assert l2.read(addr_a, 8) == b"\xAA" * 8
        assert l2.read(addr_b, 8) == b"\xBB" * 8

    def test_set_overflow_evicts_lru(self):
        """When more addresses than ways map to the same set, LRU evicts oldest."""
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=2)  # 2 ways
        l2.set_ram_backend(rf, wf)

        set_idx = 5
        shift = l2._line_shift

        # Write 3 addresses mapping to the same set (2 ways → 3rd evicts oldest)
        addrs = [
            (0x100 << shift) | (set_idx << shift),
            (0x200 << shift) | (set_idx << shift),
            (0x300 << shift) | (set_idx << shift),
        ]

        l2.write(addrs[0], b"\x11" * 8)
        l2.write(addrs[1], b"\x22" * 8)
        # This should evict addrs[0] (LRU, since it was accessed longest ago)
        l2.write(addrs[2], b"\x33" * 8)

        # addrs[0] should be evicted and re-read from RAM (written back as M)
        data = l2.read(addrs[0], 8)
        assert data == b"\x11" * 8, (
            f"Evicted LRU data corrupted: got {data.hex()}"
        )


class TestCacheWriteSizeVariants:
    """Test different write sizes that the emulator issues."""

    def test_write_1_byte(self):
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)
        l2.write(0x1000, b"\x42")
        assert l2.read(0x1000, 1) == b"\x42"

    def test_write_2_bytes(self):
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)
        l2.write(0x1000, b"\x34\x12")
        assert l2.read(0x1000, 2) == b"\x34\x12"

    def test_write_4_bytes(self):
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)
        l2.write(0x1000, b"\x78\x56\x34\x12")
        assert l2.read(0x1000, 4) == b"\x78\x56\x34\x12"

    def test_write_8_bytes(self):
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)
        l2.write(0x1000, b"\xEF\xCD\xAB\x89\x67\x45\x23\x01")
        assert l2.read(0x1000, 8) == b"\xEF\xCD\xAB\x89\x67\x45\x23\x01"


class TestCacheMfenceDid:
    """mfence.did: flush cache lines by domain ID."""

    def test_flush_by_mdid_does_not_affect_other_domains(self):
        ram, rf, wf = _make_ram()
        l2 = L2Cache(size=4 * 1024, line_size=64, ways=4)
        l2.set_ram_backend(rf, wf)

        # Write data with mdid=0
        l2.current_mdid = 0
        l2.write(0x1000, b"\xAA" * 8)

        # Write data with mdid=1
        l2.current_mdid = 1
        l2.write(0x2000, b"\xBB" * 8)

        # Flush mdid=1
        flushed = l2.flush_by_mdid(1)
        assert flushed >= 1

        # mdid=0 data should still be in cache
        assert l2.read(0x1000, 8) == b"\xAA" * 8

        # mdid=1 data should be evicted and re-read from RAM
        assert l2.read(0x2000, 8) == b"\xBB" * 8


class TestCacheLargeScale:
    """Larger scale test simulating firmware-level access patterns."""

    def test_full_cache_thrash(self):
        """Fill the entire cache multiple times, verify data integrity at checkpoints."""
        ram, rf, wf = _make_ram(1024 * 1024)  # 1 MiB RAM
        l2 = L2Cache(size=16 * 1024, line_size=64, ways=4)  # 64 sets × 4 ways
        l2.set_ram_backend(rf, wf)

        # Pre-write checkpoints with known data
        checkpoints = {
            0x10000: b"\xCA\xFE\xBA\xBE\x00\x00\x00\x00",
            0x20000: b"\xDE\xAD\xC0\xDE\x00\x00\x00\x00",
            0x50000: b"\x01\x23\x45\x67\x89\xAB\xCD\xEF",
        }
        for addr, data in checkpoints.items():
            l2.write(addr, data)

        # Build set of checkpoint cache-line-aligned addresses to skip
        cp_lines = set()
        for cp_addr in checkpoints:
            cp_lines.add(cp_addr & ~63)  # cache-line-aligned
        # Fill cache many times (simulating BSS memset of ~800 KiB),
        # skipping checkpoint lines to test writeback + reload, not overwrite.
        for iteration in range(3):
            for addr in range(0x00000, 0xC0000, 64):
                if addr in cp_lines:
                    continue
                l2.write(addr, b"\x00" * 64)

            # Verify checkpoints survive each iteration
            for cp_addr, cp_data in checkpoints.items():
                actual = l2.read(cp_addr, len(cp_data))
                assert actual == cp_data, (
                    f"Iteration {iteration}, checkpoint {cp_addr:#x} corrupted: "
                    f"expected {cp_data.hex()}, got {actual.hex()}"
                )
