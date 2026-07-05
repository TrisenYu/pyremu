#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native acceleration layer — Rust cdylib via ctypes, with pure Python fallback.

Loads the pre-compiled Rust shared library.  If it is missing or the platform
is unsupported, ``loguru`` emits a warning and the module degrades to pure
Python equivalents transparently.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from loguru import logger

# ============================================================
#  Platform detection
# ============================================================

_EXT = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
_NATIVE_DIR = Path(__file__).resolve().parent
_LIB_PATH = _NATIVE_DIR / f"libdecode{_EXT}"

# ============================================================
#  ctypes type definitions (must match Rust #[repr(C)] layout)
# ============================================================

from pyremu._native._ctypes import (  # noqa: E402, F401 — re-export
    Alu64,
    CompressedFields,
    DecodedFields,
    PteFields,
    Sv39Vpn,
)

# ============================================================
#  Library loading
# ============================================================

_lib = None
try:
    _lib = ctypes.CDLL(str(_LIB_PATH))

    # Phase 1: instruction decode
    _lib.decode_fields.argtypes = [ctypes.c_uint32]
    _lib.decode_fields.restype = DecodedFields
    _lib.decode_compressed.argtypes = [ctypes.c_uint16]
    _lib.decode_compressed.restype = CompressedFields

    # Phase 2a: PMP check
    _lib.pmp_check.argtypes = [
        ctypes.c_void_p,  # cfg: *const u8
        ctypes.c_void_p,  # addr: *const u64
        ctypes.c_uint8,  # num_entries
        ctypes.c_uint64,  # pa
        ctypes.c_uint32,  # size
        ctypes.c_uint8,  # is_write
        ctypes.c_uint8,  # is_execute
        ctypes.c_uint8,  # mode_val
        ctypes.c_uint64,  # mstatus_val
        ctypes.c_uint8,  # pmpsplit
        ctypes.c_uint8,  # mdid
    ]
    _lib.pmp_check.restype = ctypes.c_uint8

    # Phase 2b: Sv39 MMU helpers
    _lib.sv39_decompose_va.argtypes = [ctypes.c_uint64]
    _lib.sv39_decompose_va.restype = Sv39Vpn
    _lib.pte_parse.argtypes = [ctypes.c_uint64]
    _lib.pte_parse.restype = PteFields
    _lib.pte_assemble_pa.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint8,
    ]
    _lib.pte_assemble_pa.restype = ctypes.c_uint64
    _lib.sv39_page_size.argtypes = [ctypes.c_uint8]
    _lib.sv39_page_size.restype = ctypes.c_uint64

    # Phase 2c: ALU pure-compute ops (return Alu64 with trap flag)
    _lib.exec_alu_op.argtypes = [
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    _lib.exec_alu_op.restype = Alu64
    _lib.exec_op_imm.argtypes = [
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    _lib.exec_op_imm.restype = Alu64
    _lib.exec_op32.argtypes = [
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    _lib.exec_op32.restype = Alu64
    _lib.exec_op_imm32.argtypes = [
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    _lib.exec_op_imm32.restype = Alu64

    # Phase 3: RAM direct read/write (zero-allocation fast path)
    _lib.bus_read_ram.argtypes = [
        ctypes.c_void_p,  # ram_ptr: *const u8
        ctypes.c_uint64,  # ram_size
        ctypes.c_uint64,  # ram_base
        ctypes.c_uint64,  # shadow_base
        ctypes.c_uint64,  # shadow_size
        ctypes.c_uint64,  # pa
        ctypes.c_uint32,  # size
        ctypes.c_void_p,  # out_buf: *mut u8
    ]
    _lib.bus_read_ram.restype = ctypes.c_uint8
    _lib.bus_write_ram.argtypes = [
        ctypes.c_void_p,  # ram_ptr: *mut u8
        ctypes.c_uint64,  # ram_size
        ctypes.c_uint64,  # ram_base
        ctypes.c_uint64,  # shadow_base
        ctypes.c_uint64,  # shadow_size
        ctypes.c_uint64,  # pa
        ctypes.c_void_p,  # data_ptr: *const u8
        ctypes.c_uint32,  # size
    ]
    _lib.bus_write_ram.restype = ctypes.c_uint8

    # Phase 4: batch execution engine (Phases A-E: full dispatch)
    _lib.run_batch.argtypes = [
        ctypes.c_void_p,   # states: *mut HartState
        ctypes.c_uint32,   # num_harts
        ctypes.c_void_p,   # ram: *mut u8
        ctypes.c_uint64,   # ram_size
        ctypes.c_uint64,   # ram_base
        ctypes.c_uint64,   # shadow_base
        ctypes.c_uint64,   # shadow_size
        ctypes.c_uint64,   # max_instrs
        ctypes.c_void_p,   # result: *mut BatchResult
        # PMP state
        ctypes.c_void_p,   # pmp_cfg: *const u8
        ctypes.c_void_p,   # pmp_addr: *const u64
        ctypes.c_uint8,    # pmp_num
        ctypes.c_uint8,    # pmpsplit_val
        # CLINT state
        ctypes.c_uint64,   # mtime
        ctypes.c_void_p,   # mtimecmp: *const u64
        ctypes.c_void_p,   # msip: *const u8
        # Device MMIO ranges
        ctypes.c_void_p,   # dev_bases: *const u64
        ctypes.c_void_p,   # dev_ends: *const u64
        ctypes.c_uint8,    # num_devices
    ]
    _lib.run_batch.restype = None

except OSError as exc:
    logger.warning(
        "无法加载 native 加速库 ({}): {} — 降级为纯 Python 实现, 性能会下降",
        _LIB_PATH.name,
        exc,
    )


# ============================================================
#  Public API
# ============================================================


def native_available() -> bool:
    """Return ``True`` if the native acceleration library is loaded."""
    return _lib is not None


# ============================================================
#  Pure Python fallback
# ============================================================


def _sext12(val: int) -> int:
    """Sign-extend from 12 bits -> canonical 64-bit unsigned."""
    return (val & 0x7FF) - (val & 0x800) & 0xFFFF_FFFF_FFFF_FFFF


def _decode_fields_py(instr: int) -> DecodedFields:
    """Pure-Python equivalent of Rust ``decode_fields()``."""
    f = DecodedFields()
    instr = instr & 0xFFFF_FFFF

    # Fixed fields
    f.opcode = instr & 0x7F
    f.rd = (instr >> 7) & 0x1F
    f.func3 = (instr >> 12) & 0x7
    f.rs1 = (instr >> 15) & 0x1F
    f.rs2 = (instr >> 20) & 0x1F
    f.func7 = (instr >> 25) & 0x7F
    f.func12 = (instr >> 20) & 0xFFF
    f.is_compressed = 1 if (instr & 0x3) != 3 else 0

    # Immediates
    f.imm12_se = _sext12((instr >> 20) & 0xFFF)
    f.imm20_raw = (instr >> 12) & 0xF_FFFF

    imm_s = ((instr >> 25) & 0x7F) << 5 | ((instr >> 7) & 0x1F)
    f.imm_s = _sext12(imm_s)

    imm_b = ((instr >> 31) & 1) << 12
    imm_b |= ((instr >> 25) & 0x3F) << 5
    imm_b |= ((instr >> 8) & 0xF) << 1
    imm_b |= ((instr >> 7) & 1) << 11
    # _sext(imm_b, 13)
    f.imm_b = (imm_b & 0xFFF) - (imm_b & 0x1000) & 0xFFFF_FFFF_FFFF_FFFF

    imm_j = ((instr >> 31) & 1) << 20
    imm_j |= ((instr >> 21) & 0x3FF) << 1
    imm_j |= ((instr >> 20) & 1) << 11
    imm_j |= ((instr >> 12) & 0xFF) << 12
    # _sext(imm_j, 21)
    f.imm_j = (imm_j & 0xF_FFFF) - (imm_j & 0x10_0000) & 0xFFFF_FFFF_FFFF_FFFF

    return f


# ============================================================
#  Public decode function
# ============================================================


def decode_fields(instr: int) -> DecodedFields:
    """Extract all standard RISC-V instruction fields.

    Uses the native Rust library if available; otherwise falls back to
    pure Python.
    """
    if _lib is not None:
        return _lib.decode_fields(instr)
    return _decode_fields_py(instr)


def _creg(raw: int) -> int:
    """3-bit compressed register -> full register (x8–x15)."""
    return (raw & 0x7) + 8


def _decode_compressed_py(half: int) -> CompressedFields:
    """Pure-Python equivalent of Rust ``decode_compressed()``."""
    cf = CompressedFields()
    h = half & 0xFFFF

    cf.quadrant = h & 0x3
    cf.funct3 = (h >> 13) & 0x7

    bit12 = (h >> 12) & 0x1
    cf.bit12 = bit12
    cf.bit65 = (h >> 5) & 0x3
    cf.sf = (h >> 10) & 0x3

    bits_6_2 = (h >> 2) & 0x1F
    bits_11_7 = (h >> 7) & 0x1F

    # ---- register fields ----
    cf.rdp = _creg((h >> 2) & 0x7)  # C0 rd/rs2, C1-ALU rs2
    cf.rs1p = _creg((h >> 7) & 0x7)  # C0 rs1, C1-ALU rd

    if cf.quadrant == 0:
        cf.rd = cf.rdp
        cf.rs1 = cf.rs1p
        cf.rs2 = cf.rdp
    else:
        cf.rd = bits_11_7
        cf.rs1 = bits_11_7
        cf.rs2 = bits_6_2

    # ---- immediates ----
    imm: int = 0
    imm2: int = 0

    if cf.quadrant == 0:
        if cf.funct3 == 0:
            # C.ADDI4SPN: nzuimm[5:4|9:6|2|3]
            imm = ((h >> 6) & 0x1) << 2
            imm |= ((h >> 5) & 0x1) << 3
            imm |= ((h >> 11) & 0x1) << 4
            imm |= ((h >> 12) & 0x1) << 5
            imm |= ((h >> 7) & 0xF) << 6
        elif cf.funct3 in (2, 6):
            # C.LW / C.SW: uimm[5:3|2|6]
            imm = ((h >> 6) & 0x1) << 2 | ((h >> 10) & 0x7) << 3 | ((h >> 5) & 0x1) << 6
        elif cf.funct3 in (3, 7):
            # C.LD / C.SD: uimm[5:3|6|7]
            imm = ((h >> 5) & 0x3) << 6 | ((h >> 10) & 0x7) << 3

    elif cf.quadrant == 1:
        if cf.funct3 in (0, 1, 2):
            # C.ADDI / C.ADDIW / C.LI: 6-bit signed imm
            raw = (bit12 << 5) | bits_6_2
            imm = (raw & 0x1F) - (raw & 0x20) & 0xFFFF_FFFF_FFFF_FFFF
        elif cf.funct3 == 3:
            if bits_11_7 == 2:
                # C.ADDI16SP: 10-bit signed nzimm
                nz = ((h >> 6) & 0x1) << 4
                nz |= ((h >> 2) & 0x1) << 5
                nz |= ((h >> 5) & 0x1) << 6
                nz |= ((h >> 3) & 0x3) << 7
                nz |= ((h >> 12) & 0x1) << 9
                imm = (nz & 0x1FF) - (nz & 0x200) & 0xFFFF_FFFF_FFFF_FFFF
            else:
                # C.LUI: 6-bit signed nzimm (shift-left-12 in caller)
                nz = (bit12 << 5) | bits_6_2
                imm = (nz & 0x1F) - (nz & 0x20) & 0xFFFF_FFFF_FFFF_FFFF
        elif cf.funct3 == 5:
            # C.J: 11-bit signed offset
            off = bit12 << 11
            off |= ((h >> 8) & 0x1) << 10
            off |= ((h >> 9) & 0x3) << 8
            off |= ((h >> 6) & 0x1) << 7
            off |= ((h >> 7) & 0x1) << 6
            off |= ((h >> 2) & 0x1) << 5
            off |= ((h >> 11) & 0x1) << 4
            off |= ((h >> 3) & 0x7) << 1
            imm = (off & 0x7FF) - (off & 0x800) & 0xFFFF_FFFF_FFFF_FFFF
        elif cf.funct3 in (6, 7):
            # C.BEQZ / C.BNEZ: 9-bit signed offset
            off = bit12 << 8
            off |= ((h >> 10) & 0x3) << 3  # offset[4:3]
            off |= ((h >> 5) & 0x3) << 6  # offset[7:6]
            off |= ((h >> 2) & 0x1) << 5  # offset[5]
            off |= ((h >> 3) & 0x3) << 1  # offset[2:1]
            imm = (off & 0xFF) - (off & 0x100) & 0xFFFF_FFFF_FFFF_FFFF

    elif cf.quadrant == 2:
        if cf.funct3 == 0:
            # C.SLLI: shamt[5:0] = {bit12, bits[6:2]}
            imm = (bit12 << 5) | bits_6_2
        elif cf.funct3 == 2:
            # C.LWSP: uimm[5|4:2|7:6]
            imm = ((h >> 5) & 0x3) << 6 | ((h >> 12) & 0x1) << 5 | ((h >> 2) & 0x7) << 2
        elif cf.funct3 == 3:
            # C.LDSP: uimm[5|4:3|8:6]
            imm = ((h >> 2) & 0x7) << 6 | ((h >> 12) & 0x1) << 5 | ((h >> 5) & 0x3) << 3
        elif cf.funct3 == 6:
            # C.SWSP: uimm[5:2|7:6]
            imm2 = ((h >> 9) & 0xF) << 2 | ((h >> 7) & 0x3) << 6
        elif cf.funct3 == 7:
            # C.SDSP: uimm[5:3|8:6]
            imm2 = ((h >> 7) & 0x7) << 6 | ((h >> 10) & 0x7) << 3

    cf.imm = imm & 0xFFFF_FFFF_FFFF_FFFF
    cf.imm2 = imm2 & 0xFFFF_FFFF_FFFF_FFFF
    return cf


def decode_compressed(half: int) -> CompressedFields:
    """Extract all 16-bit compressed instruction fields.

    Uses the native Rust library if available; otherwise falls back to
    pure Python.
    """
    if _lib is not None:
        return _lib.decode_compressed(half)
    return _decode_compressed_py(half)


# ============================================================
#  Phase 2a: PMP check (Rust wrapper — no pure-Python fallback;
#            the existing pmp.py::Pmp.check() IS the fallback)
# ============================================================


def pmp_check(
    cfg_bytes: bytes,
    addr_array,
    num_entries: int,
    pa: int,
    size: int,
    *,
    is_write: bool = False,
    is_execute: bool = False,
    mode_val: int = 1,
    mstatus_val: int = 0,
    pmpsplit: int = 0,
    mdid: int = 0,
) -> bool:
    """Check PMP access via Rust — all state passed as parameters.

    *cfg_bytes* is a ``bytes`` object (64 bytes, one per entry).
    *addr_array* is an ``array('Q')`` or list of 64 u64 raw pmpaddr values.

    Returns True if the access is allowed.
    """
    if _lib is None:
        # Fallback: caller should use Pmp.check() instead
        raise RuntimeError("native library not available — use Pmp.check()")
    cfg_buf = (ctypes.c_uint8 * len(cfg_bytes)).from_buffer_copy(cfg_bytes)
    addr_buf = (ctypes.c_uint64 * len(addr_array))(*addr_array)
    return bool(
        _lib.pmp_check(
            ctypes.cast(cfg_buf, ctypes.c_void_p),
            ctypes.cast(addr_buf, ctypes.c_void_p),
            num_entries,
            pa,
            size,
            1 if is_write else 0,
            1 if is_execute else 0,
            mode_val,
            mstatus_val,
            pmpsplit,
            mdid,
        )
    )


# ============================================================
#  Phase 2b: Sv39 MMU helpers (with pure Python fallback)
# ============================================================


def sv39_decompose_va(va: int) -> Sv39Vpn:
    """Decompose a 39-bit virtual address into VPN[2:0] and page offset."""
    if _lib is not None:
        return _lib.sv39_decompose_va(va)
    vpn = Sv39Vpn()
    va &= 0xFFFF_FFFF_FFFF_FFFF
    vpn.vpn2 = (va >> 30) & 0x1FF
    vpn.vpn1 = (va >> 21) & 0x1FF
    vpn.vpn0 = (va >> 12) & 0x1FF
    vpn.offset = va & 0xFFF
    return vpn


def pte_parse(raw: int) -> PteFields:
    """Parse a raw 64-bit Sv39 PTE into its component fields."""
    if _lib is not None:
        return _lib.pte_parse(raw)
    raw &= 0xFFFF_FFFF_FFFF_FFFF
    PTE_V = 1 << 0
    PTE_R = 1 << 1
    PTE_X = 1 << 3
    ppn = (raw >> 10) & 0xFFF_FFFF_FFFF
    v = 1 if raw & PTE_V else 0
    r_or_x = raw & (PTE_R | PTE_X)
    f = PteFields()
    f.ppn = ppn
    f.perm = raw & 0xF  # R|W|X|U = bits 1-4
    f.v = v
    f.is_leaf = 1 if v and r_or_x else 0
    f.is_ptr = 1 if v and not r_or_x else 0
    return f


def pte_assemble_pa(ppn: int, va: int, level: int) -> int:
    """Compute physical address from leaf PTE PPN, original VA, and page level."""
    if _lib is not None:
        return _lib.pte_assemble_pa(ppn, va, level)
    if level == 1:
        # 2 MiB superpage: PPN[8:0] from VA[20:12]
        vpn0 = (va >> 12) & 0x1FF
        ppn = (ppn & 0xFFFF_FFFF_FFFF_FE00) | vpn0
    offset = va & 0xFFF
    return ((ppn << 12) | offset) & 0xFFFF_FFFF_FFFF_FFFF


def sv39_page_size(level: int) -> int:
    """Return page size in bytes for a given Sv39 leaf level."""
    if _lib is not None:
        return _lib.sv39_page_size(level)
    return 2 * 1024 * 1024 if level == 1 else 4096


# ============================================================
#  Phase 2c: ALU pure-compute ops (with pure Python fallback)
# ============================================================

_sint64 = ctypes.c_int64
_uint64 = ctypes.c_uint64


def _trunc_div(a: int, b: int) -> int:
    """Truncated division (RISC-V semantics)."""
    return int(a / b)


def _trunc_rem(a: int, b: int) -> int:
    """Truncated remainder (RISC-V semantics)."""
    return int(a - b * int(a / b))


def _alu_ok(value: int) -> Alu64:
    r = Alu64()
    r.value = value
    r.trap = 0
    return r


def _alu_ill() -> Alu64:
    r = Alu64()
    r.trap = 1
    return r


def exec_alu_op(funct3: int, funct7: int, v1: int, v2: int) -> Alu64:
    """RISC-V R-type ALU: returns Alu64(value, trap)."""
    if _lib is not None:
        return _lib.exec_alu_op(funct3, funct7, v1, v2)
    # Pure Python fallback
    if funct3 == 0b000:
        if funct7 == 0:
            return _alu_ok((v1 + v2) & 0xFFFF_FFFF_FFFF_FFFF)
        elif funct7 == 1:
            return _alu_ok((v1 * v2) & 0xFFFF_FFFF_FFFF_FFFF)
        elif funct7 == 0x20:
            return _alu_ok((v1 - v2) & 0xFFFF_FFFF_FFFF_FFFF)
        else:
            return _alu_ill()
    elif funct3 == 0b001:
        if funct7 == 0:
            return _alu_ok((v1 << (v2 & 0x3F)) & 0xFFFF_FFFF_FFFF_FFFF)
        elif funct7 == 1:
            s1 = _sint64(v1).value
            s2 = _sint64(v2).value
            return _alu_ok(_sint64((s1 * s2) >> 64).value & 0xFFFF_FFFF_FFFF_FFFF)
        else:
            return _alu_ill()
    elif funct3 == 0b010:
        if funct7 == 0:
            return _alu_ok(1 if _sint64(v1).value < _sint64(v2).value else 0)
        elif funct7 == 1:
            s1 = _sint64(v1).value
            u2 = _uint64(v2).value
            return _alu_ok(_sint64((s1 * u2) >> 64).value & 0xFFFF_FFFF_FFFF_FFFF)
        else:
            return _alu_ill()
    elif funct3 == 0b011:
        if funct7 == 0:
            return _alu_ok(1 if _uint64(v1).value < _uint64(v2).value else 0)
        elif funct7 == 1:
            u1 = _uint64(v1).value
            u2 = _uint64(v2).value
            return _alu_ok((u1 * u2) >> 64)
        else:
            return _alu_ill()
    elif funct3 == 0b100:
        if funct7 == 0:
            return _alu_ok(v1 ^ v2)
        elif funct7 == 1:
            return _alu_ok(
                _trunc_div(_sint64(v1).value, _sint64(v2).value) & 0xFFFF_FFFF_FFFF_FFFF
            )
        else:
            return _alu_ill()
    elif funct3 == 0b101:
        if funct7 == 0:
            return _alu_ok(_uint64(v1).value >> (v2 & 0x3F))
        elif funct7 == 1:
            return _alu_ok(_trunc_div(_uint64(v1).value, _uint64(v2).value))
        elif funct7 == 0x20:
            return _alu_ok(
                _sint64(_sint64(v1).value >> (v2 & 0x3F)).value & 0xFFFF_FFFF_FFFF_FFFF
            )
        else:
            return _alu_ill()
    elif funct3 == 0b110:
        if funct7 == 0:
            return _alu_ok(v1 | v2)
        elif funct7 == 1:
            return _alu_ok(
                _trunc_rem(_sint64(v1).value, _sint64(v2).value) & 0xFFFF_FFFF_FFFF_FFFF
            )
        else:
            return _alu_ill()
    elif funct3 == 0b111:
        if funct7 == 0:
            return _alu_ok(v1 & v2)
        elif funct7 == 1:
            return _alu_ok(
                _trunc_rem(_uint64(v1).value, _uint64(v2).value) & 0xFFFF_FFFF_FFFF_FFFF
            )
        else:
            return _alu_ill()
    return _alu_ill()


def exec_op_imm(funct3: int, funct7: int, v1: int, imm: int) -> Alu64:
    """RISC-V I-type ALU: returns Alu64(value, trap)."""
    if _lib is not None:
        return _lib.exec_op_imm(funct3, funct7, v1, imm)
    shamt = imm & 0x3F
    funct6 = funct7 >> 1
    if funct3 == 0b000:
        return _alu_ok((v1 + imm) & 0xFFFF_FFFF_FFFF_FFFF)
    elif funct3 == 0b001:
        if funct6 != 0:
            return _alu_ill()
        return _alu_ok((v1 << shamt) & 0xFFFF_FFFF_FFFF_FFFF)
    elif funct3 == 0b010:
        return _alu_ok(1 if _sint64(v1).value < imm else 0)
    elif funct3 == 0b011:
        return _alu_ok(1 if _uint64(v1).value < _uint64(imm).value else 0)
    elif funct3 == 0b100:
        return _alu_ok(v1 ^ imm)
    elif funct3 == 0b101:
        if funct6 == 0:
            return _alu_ok(_uint64(v1).value >> shamt)
        elif funct6 == 0x10:
            return _alu_ok(_sint64(_sint64(v1).value >> shamt).value & 0xFFFF_FFFF_FFFF_FFFF)
        else:
            return _alu_ill()
    elif funct3 == 0b110:
        return _alu_ok(v1 | imm)
    elif funct3 == 0b111:
        return _alu_ok(v1 & imm)
    return _alu_ill()


def exec_op32(funct3: int, funct7: int, v1: int, v2: int) -> Alu64:
    """RISC-V RV64 32-bit word ALU: returns Alu64(value, trap)."""
    if _lib is not None:
        return _lib.exec_op32(funct3, funct7, v1, v2)
    w1 = v1 & 0xFFFF_FFFF
    w2 = v2 & 0xFFFF_FFFF
    se32 = lambda x: (
        _sint64(_sint64(x & 0xFFFF_FFFF).value << 32 >> 32).value & 0xFFFF_FFFF_FFFF_FFFF
    )
    if funct3 == 0b000:
        if funct7 == 0:
            return _alu_ok(se32((w1 + w2) & 0xFFFF_FFFF))
        elif funct7 == 1:
            return _alu_ok(se32((w1 * w2) & 0xFFFF_FFFF))
        elif funct7 == 0x20:
            return _alu_ok(se32((w1 - w2) & 0xFFFF_FFFF))
        else:
            return _alu_ill()
    elif funct3 == 0b001:
        if funct7 != 0:
            return _alu_ill()
        return _alu_ok(se32((w1 << (w2 & 0x1F)) & 0xFFFF_FFFF))
    elif funct3 == 0b100:
        if funct7 != 1:
            return _alu_ill()
        return _alu_ok(se32(_trunc_div(_sint64(w1).value, _sint64(w2).value) & 0xFFFF_FFFF))
    elif funct3 == 0b101:
        if funct7 == 0:
            return _alu_ok(se32((w1 >> (w2 & 0x1F)) & 0xFFFF_FFFF))
        elif funct7 == 1:
            return _alu_ok(se32(_trunc_div(w1, w2) & 0xFFFF_FFFF))
        elif funct7 == 0x20:
            return _alu_ok(se32(_sint64(_sint64(w1).value >> (w2 & 0x1F)).value & 0xFFFF_FFFF))
        else:
            return _alu_ill()
    elif funct3 == 0b110:
        if funct7 != 1:
            return _alu_ill()
        return _alu_ok(se32(_trunc_rem(_sint64(w1).value, _sint64(w2).value) & 0xFFFF_FFFF))
    elif funct3 == 0b111:
        if funct7 != 1:
            return _alu_ill()
        return _alu_ok(se32(_trunc_rem(w1, w2) & 0xFFFF_FFFF))
    return _alu_ill()


def exec_op_imm32(funct3: int, funct7: int, v1: int, imm: int) -> Alu64:
    """RISC-V RV64 32-bit immediate ALU: returns Alu64(value, trap)."""
    if _lib is not None:
        return _lib.exec_op_imm32(funct3, funct7, v1, imm)
    w1 = v1 & 0xFFFF_FFFF
    shamt = imm & 0x1F
    se32 = lambda x: (
        _sint64(_sint64(x & 0xFFFF_FFFF).value << 32 >> 32).value & 0xFFFF_FFFF_FFFF_FFFF
    )
    if funct3 == 0b000:
        return _alu_ok(se32((w1 + (imm & 0xFFFF_FFFF)) & 0xFFFF_FFFF))
    elif funct3 == 0b001:
        if funct7 != 0:
            return _alu_ill()
        return _alu_ok(se32((w1 << shamt) & 0xFFFF_FFFF))
    elif funct3 == 0b101:
        funct6 = funct7 >> 1
        if funct6 == 0:
            return _alu_ok(se32((w1 >> shamt) & 0xFFFF_FFFF))
        elif funct6 == 0x10:
            return _alu_ok(se32(_sint64(_sint64(w1).value >> shamt).value & 0xFFFF_FFFF))
        else:
            return _alu_ill()
    return _alu_ill()


# ============================================================
#  Phase 3: RAM direct read/write (zero-allocation fast path)
# ============================================================


def bus_read_ram(
    ram_buf,  # ctypes array from_buffer(bytearray)
    ram_size: int,
    ram_base: int,
    shadow_base: int,
    shadow_size: int,
    pa: int,
    size: int,
    out_buf,  # ctypes array for output (pre-allocated 8 bytes)
) -> int:
    """Read *size* bytes from physical address *pa* directly into *out_buf*.

    Returns 1 if the address falls within RAM (primary or shadow); 0 otherwise.
    Caller falls back to device / L2-cache / error logic on failure.
    """
    if _lib is not None:
        return _lib.bus_read_ram(
            ram_buf,
            ram_size,
            ram_base,
            shadow_base,
            shadow_size,
            pa,
            size,
            out_buf,
        )
    return 0  # no native lib — caller uses pure-Python path


def bus_write_ram(
    ram_buf,  # ctypes array from_buffer(bytearray)
    ram_size: int,
    ram_base: int,
    shadow_base: int,
    shadow_size: int,
    pa: int,
    data,  # bytes to write
    size: int,
) -> int:
    """Write *size* bytes from *data* to physical address *pa*.

    Returns 1 if the address falls within RAM; 0 otherwise.
    """
    if _lib is not None:
        return _lib.bus_write_ram(
            ram_buf,
            ram_size,
            ram_base,
            shadow_base,
            shadow_size,
            pa,
            data,
            size,
        )
    return 0  # no native lib — caller uses pure-Python path


# ============================================================
#  Phase 4: batch execution bridge
# ============================================================


def run_batch(
    states,  # ctypes array of HartState
    num_harts: int,
    ram_buf,
    ram_size: int,
    ram_base: int,
    shadow_base: int,
    shadow_size: int,
    max_instrs: int,
    result,
    pmp_cfg=b"",
    pmp_addr=None,          # list of int or array('Q') or ctypes array
    pmpsplit: int = 0,
    mtime: int = 0,
    mtimecmp=None,           # list of int or ctypes array
    msip=None,               # list of int or ctypes array
    dev_bases=None,          # ctypes array of uint64
    dev_ends=None,           # ctypes array of uint64
) -> None:
    """Execute up to *max_instrs* instructions across all harts in Rust."""
    if _lib is None:
        return

    # --- PMP conversion ---
    pmp_num = 0
    _pmp_cfg_buf = None
    _pmp_addr_buf = None
    if pmp_cfg and pmp_addr is not None:
        pmp_num = min(len(pmp_cfg), len(pmp_addr))
        if pmp_num > 0:
            # Convert bytes → c_uint8 array
            _pmp_cfg_buf = (ctypes.c_uint8 * len(pmp_cfg)).from_buffer_copy(pmp_cfg)
            # Convert list/array → c_uint64 array
            _pmp_addr_buf = (ctypes.c_uint64 * len(pmp_addr))()
            for i, v in enumerate(pmp_addr):
                _pmp_addr_buf[i] = int(v)

    # --- CLINT conversion ---
    nh = max(int(num_harts), 1)
    _mtc_buf = (ctypes.c_uint64 * nh)()
    if mtimecmp is not None:
        for i in range(min(nh, len(mtimecmp))):
            _mtc_buf[i] = int(mtimecmp[i])
    else:
        for i in range(nh):
            _mtc_buf[i] = 0xFFFF_FFFF_FFFF_FFFF

    _msip_buf = (ctypes.c_uint8 * nh)()
    if msip is not None:
        for i in range(min(nh, len(msip))):
            _msip_buf[i] = int(msip[i])

    # --- Device MMIO: pass-through if ctypes arrays ---
    ndev = 0
    if dev_bases is not None and dev_ends is not None:
        ndev = min(len(dev_bases), len(dev_ends))

    _lib.run_batch(
        states, num_harts,
        ram_buf, ram_size, ram_base, shadow_base, shadow_size,
        max_instrs,
        ctypes.byref(result),
        ctypes.cast(_pmp_cfg_buf, ctypes.c_void_p) if _pmp_cfg_buf else ctypes.c_void_p(0),
        ctypes.cast(_pmp_addr_buf, ctypes.c_void_p) if _pmp_addr_buf else ctypes.c_void_p(0),
        pmp_num, pmpsplit,
        mtime,
        ctypes.cast(_mtc_buf, ctypes.c_void_p),
        ctypes.cast(_msip_buf, ctypes.c_void_p),
        ctypes.cast(dev_bases, ctypes.c_void_p) if dev_bases is not None else ctypes.c_void_p(0),
        ctypes.cast(dev_ends, ctypes.c_void_p) if dev_ends is not None else ctypes.c_void_p(0),
        ndev,
    )
