#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""栈溢出在各特权级的表现: 保护页命中与陷态信息."""

import pytest

from pyremu.configs_aux import examed_elf_dir, path_join
from pyremu.core.decoder import Hart
from pyremu.core.hart import (
    RiscvMode,
)
from pyremu.core.mem_check_aux import (
    inject_memory_backend,
    mem_read,
    mem_write,
    MemoryAccessError,
)
from pyremu.emulator import Emulator
from pyremu.memory.bus import Bus
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import parse_firmware
from tests.test_trap.support import (
    make_pte,
    PTE_R,
    PTE_V,
    PTE_W,
)

#  栈溢出防护 -- 覆盖 M / S / U 三种特权级

# 测试共用常量
_SO_RAM_BASE = 0x8000_0000
_SO_RAM_SIZE = 64 * 1024  # 64 KiB

# S-mode Sv39 guard page 物理地址 (使用低地址以适配 2 MiB bytearray)
_SO_SMODE_STACK_VA = 0x2000  # VPN all zero, only vpn0 differs
_SO_SMODE_GUARD_VA = 0x1000  # guard page, unmapped
_SO_SMODE_STACK_PA = 0x5000  # stack physical page


class TestStackOverflowMmode:
    """M-mode stack overflow: PMA physical memory boundary detection."""

    RAM_BASE = _SO_RAM_BASE
    RAM_SIZE = _SO_RAM_SIZE
    RAM_TOP = _SO_RAM_BASE + _SO_RAM_SIZE

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        bus = Bus(ram_size=self.RAM_SIZE, ram_base=self.RAM_BASE)
        inject_memory_backend(h, bus.read, bus.write)
        h.bus = bus
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x80000000
        return h

    # fmt: off
    @pytest.mark.parametrize("addr,size,is_write,expected_cause", [
        # stack overflow below RAM base
        (RAM_BASE - 8,  8, True,  7),   # store below base -> StAccessFault
        (RAM_BASE - 4,  4, False, 5),   # load  below base -> LdAccessFault
        # stack overflow above RAM top
        (RAM_TOP,       4, True,  7),   # store above top  -> StAccessFault
        (RAM_TOP,       4, False, 5),   # load  above top  -> LdAccessFault
        # valid stack range
        (RAM_BASE + 0x1000, 4, False, 0),  # valid load
        (RAM_BASE + 0x1000, 4, True,  0),  # valid store
        # boundary access -- exactly within valid range
        (RAM_BASE,            4, False, 0),  # bottom edge load
        (RAM_TOP - 4,         4, False, 0),  # top edge load
        # 8-byte aligned store at top edge (all bytes in RAM, natural alignment OK)
        (RAM_TOP - 8,         8, True,  0),  # 8-byte at top edge
    ])
    # fmt: on
    def test_stack_boundary_access(self, hart, addr, size, is_write, expected_cause):
        """Stack boundary access: covers overflow direction, op, and edge positions."""
        if not is_write and expected_cause == 0:
            hart._mem_write_phy(addr, b"\x42" * size)

        if is_write:
            try:
                mem_write(hart, addr, b"\x00" * size)
            except MemoryAccessError:
                pass
        else:
            try:
                result = mem_read(hart, addr, size)
            except MemoryAccessError:
                result = None
            if expected_cause == 0:
                assert result is not None

        assert hart.mcause_val == expected_cause, (
            f"addr=0x{addr:x} size={size} {'write' if is_write else 'read'}: "
            f"expected cause={expected_cause}, actual={hart.mcause_val}"
        )


class TestStackOverflowSmode:
    """S-mode stack overflow: Sv39 guard page -> PageFault -> M-mode.

    guard page (0x1000) just below S stack page (0x2000), unmapped.
    Page tables and data all within 2 MiB bytearray.
    PMP NAPOT full address space injected so S-mode accesses pass PMP.
    """

    PAGE_SHIFT = 12
    SATP_MODE_SV39 = 8

    # page table physical layout (low bytearray addresses)
    L1_BASE = 0x10000
    L2_BASE = 0x11000
    L3_BASE = 0x12000

    # VA / PA constants
    STACK_VA = _SO_SMODE_STACK_VA  # 0x2000
    GUARD_VA = _SO_SMODE_GUARD_VA  # 0x1000
    STACK_PA = _SO_SMODE_STACK_PA  # 0x5000

    @pytest.fixture
    def ram_ctx(self):
        """2 MiB bytearray as physical memory."""
        ram = bytearray(2 * 1024 * 1024)

        def read_fn(addr: int, size: int) -> bytes:
            return bytes(ram[addr : addr + size])

        def write_fn(addr: int, data: bytes):
            for i, b in enumerate(data):
                ram[addr + i] = b

        return ram, read_fn, write_fn

    @pytest.fixture
    def hart(self, ram_ctx):
        _ram, read_fn, write_fn = ram_ctx
        h = Hart(id=0)
        inject_memory_backend(h, read_fn, write_fn)
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        return h

    # -- helpers --

    @staticmethod
    def _write_pte(ram: bytearray, table_base: int, index: int, pte_val: int):
        addr = table_base + index * 8
        ram[addr : addr + 8] = pte_val.to_bytes(8, "little")

    def _setup_sv39_stack_only(self, ram: bytearray):
        """Map only S-mode stack page (R+W), guard page stays V=0.

        VA=0x2000: vpn2=0, vpn1=0, vpn0=2
        VA=0x1000: vpn2=0, vpn1=0, vpn0=1
        """
        # L1[0] -> L2
        self._write_pte(
            ram,
            self.L1_BASE,
            0,
            make_pte(PTE_V, self.L2_BASE >> self.PAGE_SHIFT),
        )
        # L2[0] -> L3
        self._write_pte(
            ram,
            self.L2_BASE,
            0,
            make_pte(PTE_V, self.L3_BASE >> self.PAGE_SHIFT),
        )
        # L3[2] -> stack physical page (R+W)
        self._write_pte(
            ram,
            self.L3_BASE,
            2,
            make_pte(PTE_V | PTE_R | PTE_W, self.STACK_PA >> self.PAGE_SHIFT),
        )
        # L3[1] stays V=0 -> guard page

    def _enable_sv39(self, hart: Hart):
        hart.satp_val = (self.SATP_MODE_SV39 << 60) | (self.L1_BASE >> self.PAGE_SHIFT)

    def _prep_smode(self, hart: Hart, ram: bytearray):
        """Full S-mode + Sv39 + PMP NAPOT full address space setup."""
        self._setup_sv39_stack_only(ram)
        self._enable_sv39(hart)
        # PMP NAPOT full address space R+W+X, otherwise S-mode accesses are denied
        hart.csrs["pmpcfg0"].val = 0x1F  # NAPOT, R+W+X
        hart.csrs["pmpaddr0"].val = 0x003F_FFFF_FFFF_FFFF  # full address space
        hart.mode = RiscvMode.S

    # fmt: off
    @pytest.mark.parametrize("case", [
        # guard page (unmapped) -> PageFault
        (_SO_SMODE_GUARD_VA, 8, True,  15),  # store guard   -> StPageFault
        (_SO_SMODE_GUARD_VA, 4, False, 13),  # load  guard   -> LdPageFault
        # valid stack page -> success
        (_SO_SMODE_STACK_VA + 0xE0, 4, True,   0),  # store valid
        (_SO_SMODE_STACK_VA,        4, False,  0),  # load  valid
        # cross-page: starts in guard page, extends into stack page
        (_SO_SMODE_GUARD_VA + 0xFF8, 8, True,  15),  # cross-page store
        (_SO_SMODE_GUARD_VA + 0xFF8, 8, False, 13),  # cross-page load
        # last byte of stack page (single-byte, no cross)
        (_SO_SMODE_STACK_VA + 0xFFF, 1, True,   0),  # byte at page end
    ])
    # fmt: on
    def test_guard_page_access(self, hart, ram_ctx, case):
        """Sv39 guard page: parametrized over access direction/boundary/cross-page."""
        va, size, is_write, expected_cause = case
        ram, _read_fn, _write_fn = ram_ctx
        self._prep_smode(hart, ram)

        # pre-fill physical data for read tests
        if not is_write:
            pa_offset = va - self.STACK_VA
            if 0 <= pa_offset < 0x1000:
                fill_sz = min(size, 0x1000 - pa_offset)
                ram[self.STACK_PA + pa_offset : self.STACK_PA + pa_offset + fill_sz] = (
                    b"\x99" * fill_sz
                )

        if is_write:
            try:
                mem_write(hart, va, b"\x00" * size)
            except MemoryAccessError:
                pass
        else:
            try:
                mem_read(hart, va, size)
            except MemoryAccessError:
                pass

        assert hart.mcause_val == expected_cause, (
            f"va=0x{va:x} size={size} {'write' if is_write else 'read'}: "
            f"expected cause={expected_cause}, actual={hart.mcause_val}"
        )
        # PageFault should enter M-mode (medeleg not set for page faults)
        if expected_cause in (13, 15):
            assert hart.mode == RiscvMode.M, (
                f"PageFault should enter M-mode, actual={hart.mode.name}"
            )

    def test_mmode_bare_no_page_fault(self, hart, ram_ctx):
        """M-mode Bare: guard VA treated as PA, no PageFault."""
        ram, _read_fn, _write_fn = ram_ctx
        self._setup_sv39_stack_only(ram)
        # no Sv39 -> Bare; M-mode MPRV=0 bypasses PMP
        hart.mode = RiscvMode.M

        ram[self.GUARD_VA : self.GUARD_VA + 4] = b"\xAB\xCD\xEF\x01"
        data = mem_read(hart, self.GUARD_VA, 4)
        assert data == b"\xAB\xCD\xEF\x01"
        assert hart.mcause_val == 0


class TestStackOverflowUmode:
    """U-mode stack overflow: Sv39 guard page -> StPageFault delegated to S-mode.

    Uses pre-built firmware u_mode_run_fib.elf:
      M: PMP / medeleg(ECALL+PageFault->S) / MRET->S
      S: Sv39 page tables (U stack 1 page + guard) / stvec / sscratch
      U: well-behaved (bounded input fib) or pathological (stack_bomb)
    """

    ELF_PATH = path_join(examed_elf_dir(), "u_mode_run_fib.elf")

    @pytest.fixture
    def emu_and_fw(self):
        fw = parse_firmware(self.ELF_PATH)
        if fw is None:
            pytest.skip(f"{self.ELF_PATH} not found or failed to parse")
        emu = Emulator(PlatformConfig.qemu_virt())
        emu.load_firmware(fw)
        assert emu.uart is not None
        return emu, fw

    @staticmethod
    def _boot_to_umode(emu: Emulator):
        """Run firmware until S->U mode transition occurs."""
        h = emu.harts[0]
        for _ in range(3000):
            prev = h.mode
            emu.step()
            if prev == RiscvMode.S and h.mode == RiscvMode.U:
                return

    def _run_from(self, emu: Emulator, fw, entry_sym: str, uart_input: bytes, max_steps=3000):
        """Boot to U-mode, redirect to entry, inject input, run until termination."""
        hart = emu.harts[0]
        uart = emu.uart
        assert uart is not None
        entry = fw.symbols.get(entry_sym)
        assert entry is not None, f"symbol {entry_sym} not found"

        self._boot_to_umode(emu)
        hart.pc = entry
        uart.preload(uart_input)

        for _ in range(max_steps):
            emu.step()
            if b"Process terminated" in uart.tx_data():
                break
        return uart.tx_data().decode("latin-1", errors="replace")

    # fmt: off
    @pytest.mark.parametrize("entry_sym,uart_input,expect_fault,expect_output", [
        # pathological: input 0 -> stack_bomb -> StPageFault -> S terminates
        ("u_mode_bad", b"0\n",  True,  "Fault caught"),
        # well-behaved: input 5 -> fib(5)=5, no fault
        ("u_mode_main", b"5\n", False, "fib(5)=5"),
    ])
    # fmt: on
    def test_stack_guard(self, emu_and_fw, entry_sym, uart_input, expect_fault, expect_output):
        """U-mode guard page: overflow caught by S-mode; normal input passes."""
        emu, fw = emu_and_fw
        out = self._run_from(emu, fw, entry_sym, uart_input)

        if expect_fault:
            assert "Fault caught" in out, (
                f"PageFault should have been caught by S-mode: {out!r}"
            )
        else:
            assert "Fault caught" not in out, f"Should not fault: {out!r}"
            assert "scause" not in out, f"No scause expected: {out!r}"

        assert expect_output in out, f"Expected output {expect_output!r}: {out!r}"

    def test_sscratch_preserved_after_fault(self, emu_and_fw):
        """After stack_bomb fault, S-mode stays alive and sscratch is valid."""
        emu, fw = emu_and_fw
        hart = emu.harts[0]
        uart = emu.uart
        assert uart is not None

        self._boot_to_umode(emu)
        sscratch_before = hart.csrs["sscratch"].val
        assert sscratch_before != 0, "sscratch should be initialized before U-mode"

        hart.pc = fw.symbols["u_mode_bad"]
        uart.preload(b"0\n")

        for _ in range(3000):
            emu.step()
            if b"Process terminated" in uart.tx_data():
                break

        # After fault handling, hart should be in S-mode idle, alive
        assert not hart._halted, "S-mode should remain alive after fault"
        assert hart.mode == RiscvMode.S, (
            f"Should return to S-mode idle, actual={hart.mode.name}"
        )
        # sscratch should be a valid non-zero stack pointer
        assert hart.csrs["sscratch"].val != 0, "sscratch should be non-zero"
