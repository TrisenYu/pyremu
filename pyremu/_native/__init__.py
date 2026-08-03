#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native acceleration layer — Rust cdylib via ctypes, with pure Python fallback.

Loads the pre-compiled Rust shared library.  If it is missing or the platform
is unsupported, ``loguru`` emits a warning and the module degrades to pure
Python equivalents transparently.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import sys
import tempfile
from pathlib import Path

from loguru import logger

# ============================================================
#  Platform detection
# ============================================================

_EXT = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
_NATIVE_DIR = Path(__file__).resolve().parent
_LIB_PATH = _NATIVE_DIR / f"libdecode{_EXT}" # 单独留在外部，好减少SIGBUS/SIGSEGM一类的错误

# ============================================================
#  ctypes type definitions (must match Rust #[repr(C)] layout)
# ============================================================

from pyremu._native._ctypes import (  # noqa: E402, F401 — re-export
    Alu64,
    CompressedFields,
    DecodedFields,
    FpOut,
    PteFields,
    Sv39Vpn,
)

# ============================================================
#  Library loading
# ============================================================

_lib = None
_tmp_so_path: str | None = None

def _load_lib_safe(lib_path: Path) -> ctypes.CDLL | None:
    """Load the native shared library from a temporary copy.

    ctypes.CDLL uses dlopen() which mmap's the file.  Rebuilding (cargo build)
    overwrites the original .so, and with cp the old inode is truncated,
    causing SIGBUS in the running process.  Copying to a temp file first
    isolates the running process from subsequent rebuilds.
    """
    global _tmp_so_path  # noqa: PLW0603 — intentional module-level tracking
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".so", prefix="libdecode_")
        os.close(fd)
        shutil.copy2(str(lib_path), tmp_path)
        _tmp_so_path = tmp_path
        atexit.register(_cleanup_tmp_so)
        return ctypes.CDLL(tmp_path)
    except (OSError, IOError) as exc:
        logger.warning("Failed to create temp copy of {}: {}", lib_path, exc)
        # Fall back to loading directly (may crash on rebuild)
        try:
            return ctypes.CDLL(str(lib_path))
        except OSError:
            return None


def _cleanup_tmp_so() -> None:
    """Remove the temporary .so copy, ignoring errors (best-effort)."""
    if _tmp_so_path is None:
        return
    try:
        os.unlink(_tmp_so_path)
    except OSError:
        pass


_lib: ctypes.CDLL | None = None

try:
    _lib = _load_lib_safe(_LIB_PATH)
    if _lib is None:
        raise OSError(f"无法加载 {_LIB_PATH.name}")

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

    # F/D floating point compute (return FpOut)
    _lib.fp_exec_op.argtypes = [
        ctypes.c_uint8,   # funct7
        ctypes.c_uint8,   # funct3
        ctypes.c_uint8,   # rs2
        ctypes.c_uint64,  # rs1_bits
        ctypes.c_uint64,  # rs2_bits
        ctypes.c_uint8,   # frm
    ]
    _lib.fp_exec_op.restype = FpOut
    _lib.fp_exec_fma.argtypes = [
        ctypes.c_uint8,   # opcode
        ctypes.c_uint8,   # rm (funct3)
        ctypes.c_uint8,   # fmt
        ctypes.c_uint64,  # rs1
        ctypes.c_uint64,  # rs2
        ctypes.c_uint64,  # rs3
        ctypes.c_uint8,   # frm
    ]
    _lib.fp_exec_fma.restype = FpOut

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

    # Phase 5: concurrent thread-per-hart execution engine
    _lib.run_parallel.argtypes = [
        ctypes.c_void_p,   # hart: *const FfiHartCtx
        ctypes.c_void_p,   # ffi: *const FfiPeriphCtx
        ctypes.c_void_p,   # bp: *const FfiBpCtx
        ctypes.c_void_p,   # tlb: *mut FfiTlbCtx
    ]
    _lib.run_parallel.restype = None

except OSError as exc:
    logger.warning(
        "无法加载 native 加速库 ({}): {} — 降级为性能较低的纯 Python 实现",
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
    # F/D: rs3 = bits[31:27], fmt = bits[26:25]
    f.rs3 = (instr >> 27) & 0x1F
    f.fmt = (instr >> 25) & 0x3

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
    """3-bit compressed register -> full register (x8-x15)."""
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
#  F/D floating point compute (native-only — no pure-Python FPU)
# ============================================================


def fp_exec_op(
    funct7: int, funct3: int, rs2: int, rs1_bits: int, rs2_bits: int, frm: int
) -> FpOut:
    """OP-FP compute via native softfloat.  Requires the native library.

    Raises NotImplementedError if ``.so`` is unavailable (caller converts
    to IllInstr) — there is no pure-Python floating-point fallback.
    """
    if _lib is None:
        raise NotImplementedError("FPU requires native library (libdecode.so)")
    return _lib.fp_exec_op(funct7, funct3, rs2, rs1_bits, rs2_bits, frm)


def fp_exec_fma(
    opcode: int, rm: int, fmt: int, rs1: int, rs2: int, rs3: int, frm: int
) -> FpOut:
    """FMA compute via native softfloat.  Requires the native library."""
    if _lib is None:
        raise NotImplementedError("FPU requires native library (libdecode.so)")
    return _lib.fp_exec_fma(opcode, rm, fmt, rs1, rs2, rs3, frm)


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
#  FFI context structs — ctypes layouts matching Rust #[repr(C)]
# ============================================================


class MemCtx(ctypes.Structure):
    """Memory context — matches Rust ``MemCtx``."""
    _fields_ = [
        ("ram", ctypes.c_void_p),
        ("ram_size", ctypes.c_uint64),
        ("ram_base", ctypes.c_uint64),
        ("shadow_base", ctypes.c_uint64),
        ("shadow_size", ctypes.c_uint64),
    ]


class FfiPmpCtx(ctypes.Structure):
    """PMP context — matches Rust ``FfiPmpCtx``."""
    _fields_ = [
        ("cfg", ctypes.c_void_p),
        ("addr", ctypes.c_void_p),
        ("num", ctypes.c_uint8),
        ("pmpsplit", ctypes.c_uint8),
    ]


class FfiClintCtx(ctypes.Structure):
    """CLINT context — matches Rust ``FfiClintCtx``."""
    _fields_ = [
        ("mtime", ctypes.c_void_p),
        ("mtimecmp", ctypes.c_void_p),
        ("msip", ctypes.c_void_p),
        ("base", ctypes.c_uint64),
    ]


class FfiDevCtx(ctypes.Structure):
    """Device MMIO context — matches Rust ``FfiDevCtx``."""
    _fields_ = [
        ("bases", ctypes.c_void_p),
        ("ends", ctypes.c_void_p),
        ("num", ctypes.c_uint8),
    ]


class FfiUartCtx(ctypes.Structure):
    """UART context — matches Rust ``FfiUartCtx``."""
    _fields_ = [
        ("base", ctypes.c_uint64),
        ("tx_buf", ctypes.c_void_p),
        ("tx_cap", ctypes.c_uint32),
        ("tx_wr", ctypes.c_void_p),
        ("ie", ctypes.c_uint32),          # IE 寄存器影子 (offset 0x10)
        ("txctrl", ctypes.c_uint32),      # TXCTRL 影子 (offset 0x08)
        ("rxctrl", ctypes.c_uint32),      # RXCTRL 影子 (offset 0x0C)
        ("rx_fifo_len", ctypes.c_uint32), # RX FIFO 近似填充量
        ("tx_notify_fd", ctypes.c_int32), # pipe write-end: Rust 通知 TX 线程
        ("no_stdout", ctypes.c_uint8),    # 1=Rust 不写 stdout, Python TX 统一输出
        ("rx_notify", ctypes.c_void_p),   # *mut u8 — TermIO 有新 RX 数据标志
    ]


class FfiVirtIOCtx(ctypes.Structure):
    """virtio-blk MMIO inline context — matches Rust ``FfiVirtIoCtx``."""
    _fields_ = [
        ("base", ctypes.c_uint64),
        ("capacity", ctypes.c_uint64),
        ("queue_num_max", ctypes.c_uint32),
        ("device_features_sel", ctypes.c_uint32),
        ("driver_features_sel", ctypes.c_uint32),
        ("driver_features", ctypes.c_uint64),
        ("queue_sel", ctypes.c_uint32),
        ("queue_num", ctypes.c_uint32),
        ("queue_ready", ctypes.c_uint8),
        ("queue_desc", ctypes.c_uint64),
        ("queue_driver", ctypes.c_uint64),
        ("queue_device", ctypes.c_uint64),
        ("status", ctypes.c_uint32),
        ("interrupt_status", ctypes.c_uint32),
        ("notify_pending", ctypes.c_uint8),
        ("irq_maybe_lower", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 6),
    ]


class FfiExtIrqCtx(ctypes.Structure):
    """Shared external-interrupt context — matches Rust ``FfiExtIrqCtx``."""
    _fields_ = [
        ("pending", ctypes.c_uint8),
        ("sources", ctypes.c_uint32),
        ("max_priority", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 2),
    ]


class FfiHartCtx(ctypes.Structure):
    """Grouped hart execution context — matches Rust ``FfiHartCtx``."""
    _fields_ = [
        ("states", ctypes.c_void_p),
        ("num_harts", ctypes.c_uint32),
        ("max_instrs", ctypes.c_uint64),
        ("result", ctypes.c_void_p),
        ("stop_flag", ctypes.c_void_p),
        ("ext_irq", ctypes.c_void_p),
    ]


class FfiPeriphCtx(ctypes.Structure):
    """Grouped peripheral / memory context — matches Rust ``FfiPeriphCtx``."""
    _fields_ = [
        ("mem", ctypes.c_void_p),
        ("pmp", ctypes.c_void_p),
        ("clint", ctypes.c_void_p),
        ("dev", ctypes.c_void_p),
        ("uart", ctypes.c_void_p),
        ("virtio", ctypes.c_void_p),
    ]


class FfiBpCtx(ctypes.Structure):
    """Grouped breakpoint context — matches Rust ``FfiBpCtx``."""
    _fields_ = [
        ("addrs", ctypes.c_void_p),
        ("count", ctypes.c_uint32),
    ]


class FfiTlbCtx(ctypes.Structure):
    """Grouped TLB generation context — matches Rust ``FfiTlbCtx``."""
    _fields_ = [
        ("gen", ctypes.c_void_p),
        ("gen_per_hart", ctypes.c_void_p),
    ]


# ============================================================
#  Context info classes — Python-side helpers to build FFI structs
# ============================================================


class PmpInfo:
    """PMP configuration for a batch.

    ``cfg`` and ``addr`` accept either Python bytes/list (copied to
    ctypes arrays) or pre-built ctypes arrays (used directly, enabling
    Rust to mutate the underlying memory).
    """
    __slots__ = ("cfg", "addr", "num", "pmpsplit")

    def __init__(self, cfg=None, addr=None, pmpsplit: int = 0, hart_num: int = 1):
        # Normalise cfg to a ctypes array if it isn't one already.
        if cfg is None or isinstance(cfg, (bytes, bytearray)):
            raw = cfg if cfg else b""
            _cfg = (ctypes.c_uint8 * max(len(raw), 1))()
            if raw:
                ctypes.memmove(_cfg, bytes(raw), len(raw))
            self.cfg = _cfg
        else:
            self.cfg = cfg  # already a ctypes array
        # Normalise addr similarly.
        if addr is None:
            self.addr = (ctypes.c_uint64 * 0)()
        elif isinstance(addr, (list, tuple)):
            _addr = (ctypes.c_uint64 * max(len(addr), 1))()
            for i, v in enumerate(addr):
                _addr[i] = int(v)
            self.addr = _addr
        else:
            self.addr = addr  # already a ctypes array (or array.array)
        self.pmpsplit = pmpsplit
        # ``num`` 是 *每 hart* 的 PMP 条目数, 而非扁平缓冲总长。
        # 多 hart 时 cfg/addr 为 64*hart_num 连续数组 (每 hart 独占 64 项切片),
        # 若误把总长写入 FfiPmpCtx.num (c_uint8) 会在 hart_num>=4 时溢出:
        # 64*4=256 ->256 & 0xFF = 0 ->PMP 被静默禁用。故按 hart_num 反算每 hart 值。
        total = (
            min(len(self.cfg), len(self.addr))
            if len(self.cfg) > 0 and len(self.addr) > 0
            else 0
        )
        hn = hart_num if hart_num > 0 else 1
        self.num = total // hn
        if self.num > 0xFF:
            raise ValueError(
                f"per-hart PMP num={self.num} 超出 FfiPmpCtx.num (u8) 范围"
            )


class ClintInfo:
    """CLINT state for a batch."""
    __slots__ = ("mtime", "mtimecmp", "msip", "base")

    def __init__(
        self,
        mtime: int = 0,
        mtimecmp=None,
        msip=None,
        base: int = 0,
    ):
        self.mtime = mtime
        self.mtimecmp = mtimecmp  # list or ctypes array
        self.msip = msip          # list or ctypes array
        self.base = base


class DevInfo:
    """Device MMIO ranges for a batch."""
    __slots__ = ("bases", "ends")

    def __init__(self, bases=None, ends=None):
        self.bases = bases  # ctypes array of uint64
        self.ends = ends    # ctypes array of uint64


class UartInfo:
    """UART context for a batch — lets Rust buffer sbi_printf output inline
    and handle IE/IP/TXCTRL register reads without batch exits."""
    __slots__ = ("base", "tx_buf", "tx_wr", "ie", "txctrl", "rxctrl",
                 "rx_fifo_len", "tx_notify_fd", "no_stdout", "rx_notify")

    def __init__(self, base: int = 0, tx_buf=None, tx_wr=None,
                 ie: int = 0, txctrl: int = 0, rxctrl: int = 0, rx_fifo_len: int = 0,
                 tx_notify_fd: int = -1, no_stdout: int = 0,
                 rx_notify=None):
        self.base = base
        self.tx_buf = tx_buf
        self.tx_wr = tx_wr
        self.ie = ie
        self.txctrl = txctrl
        self.rxctrl = rxctrl
        self.rx_fifo_len = rx_fifo_len
        self.tx_notify_fd = tx_notify_fd
        self.no_stdout = no_stdout
        self.rx_notify = rx_notify


class VirtIOInfo:
    """virtio-blk inline context for a batch — Rust handles all MMIO register
    accesses inline; only QueueNotify exits to Python.

    Carries the full runtime state of the virtio-blk MMIO register file
    across batches.  Without this, dynamic state written by the guest
    (queue descriptors, feature negotiation, InterruptStatus, etc.) is
    silently reset to zero on every batch, and the guest sees a dead device.
    """
    __slots__ = (
        "base", "capacity", "queue_num_max",
        "device_features_sel", "driver_features_sel", "driver_features",
        "queue_sel", "queue_num", "queue_ready",
        "queue_desc", "queue_driver", "queue_device",
        "status", "interrupt_status",
    )

    def __init__(
        self,
        base: int = 0,
        capacity: int = 0,
        queue_num_max: int = 256,
        device_features_sel: int = 0,
        driver_features_sel: int = 0,
        driver_features: int = 0,
        queue_sel: int = 0,
        queue_num: int = 0,
        queue_ready: bool = False,
        queue_desc: int = 0,
        queue_driver: int = 0,
        queue_device: int = 0,
        status: int = 0,
        interrupt_status: int = 0,
    ):
        self.base = base
        self.capacity = capacity
        self.queue_num_max = queue_num_max
        self.device_features_sel = device_features_sel
        self.driver_features_sel = driver_features_sel
        self.driver_features = driver_features
        self.queue_sel = queue_sel
        self.queue_num = queue_num
        self.queue_ready = queue_ready
        self.queue_desc = queue_desc
        self.queue_driver = queue_driver
        self.queue_device = queue_device
        self.status = status
        self.interrupt_status = interrupt_status

def run_parallel(
    states,  # ctypes array of HartState
    num_harts: int,
    ram_buf,
    ram_size: int,
    ram_base: int,
    shadow_base: int,
    shadow_size: int,
    max_instrs: int,
    result,
    pmp: PmpInfo | None = None,
    clint: ClintInfo | None = None,
    dev: DevInfo | None = None,
    uart: UartInfo | None = None,
    virtio: VirtIOInfo | None = None,
    bp_addrs: list[int] | None = None,
    stop_flag=None,  # ctypes.c_uint8 or None — shared stop flag for Ctrl+Q
    ext_irq=None,  # FfiExtIrqCtx or None — external interrupt context
    tlb_gen=None,  # ctypes.c_uint64 — persistent TLB generation counter
    tlb_gen_per_hart=None,  # ctypes array of c_uint64 — per-hart last-seen gen
) -> FfiVirtIOCtx | None:
    """Execute instructions concurrently (thread-per-hart) in Rust.

    Each non-halted hart runs in its own OS thread with a full
    fetch-decode-execute loop.  AMO instructions use real CPU atomics
    (AtomicU32/AtomicU64).  All harts run until a stop condition is hit.

    *bp_addrs* is an optional list of PC addresses that trigger
    ``EXIT_BREAKPOINT`` when matched after instruction execution.
    """
    if _lib is None:
        return

    # --- MemCtx ---
    mem = MemCtx()
    mem.ram = ctypes.cast(ram_buf, ctypes.c_void_p).value or 0
    mem.ram_size = ram_size
    mem.ram_base = ram_base
    mem.shadow_base = shadow_base
    mem.shadow_size = shadow_size

    # --- FfiPmpCtx ---
    _pmp_cfg_buf = None
    _pmp_addr_buf = None
    if pmp is not None and pmp.num > 0:
        _pmp_cfg_buf = pmp.cfg
        _pmp_addr_buf = pmp.addr

    pmp_ffi = FfiPmpCtx()
    pmp_ffi.cfg = ctypes.cast(_pmp_cfg_buf, ctypes.c_void_p).value if _pmp_cfg_buf else 0
    pmp_ffi.addr = ctypes.cast(_pmp_addr_buf, ctypes.c_void_p).value if _pmp_addr_buf else 0
    pmp_ffi.num = pmp.num if pmp is not None else 0
    pmp_ffi.pmpsplit = pmp.pmpsplit if pmp is not None else 0

    # --- FfiClintCtx ---
    nh = max(int(num_harts), 1)
    _mtc_buf = (ctypes.c_uint64 * nh)()
    _msip_buf = (ctypes.c_uint8 * nh)()
    _mtime_val = ctypes.c_uint64(0)
    if clint is not None:
        _mtime_val.value = clint.mtime
        if clint.mtimecmp is not None:
            for i in range(min(nh, len(clint.mtimecmp))):
                _mtc_buf[i] = int(clint.mtimecmp[i])
        else:
            for i in range(nh):
                _mtc_buf[i] = 0xFFFF_FFFF_FFFF_FFFF
        if clint.msip is not None:
            for i in range(min(nh, len(clint.msip))):
                _msip_buf[i] = int(clint.msip[i])
    else:
        for i in range(nh):
            _mtc_buf[i] = 0xFFFF_FFFF_FFFF_FFFF

    clint_ffi = FfiClintCtx()
    clint_ffi.mtime = ctypes.addressof(_mtime_val)
    clint_ffi.mtimecmp = ctypes.cast(_mtc_buf, ctypes.c_void_p).value
    clint_ffi.msip = ctypes.cast(_msip_buf, ctypes.c_void_p).value
    clint_ffi.base = clint.base if clint is not None else 0

    # --- FfiDevCtx ---
    dev_ffi = FfiDevCtx()
    if dev is not None and dev.bases is not None and dev.ends is not None:
        dev_ffi.bases = ctypes.cast(dev.bases, ctypes.c_void_p).value
        dev_ffi.ends = ctypes.cast(dev.ends, ctypes.c_void_p).value
        dev_ffi.num = len(dev.bases)

    # --- FfiUartCtx ---
    uart_ffi = FfiUartCtx()
    _tx_buf = None
    _tx_wr = None
    if uart is not None:
        uart_ffi.base = uart.base
        if uart.tx_buf is not None:
            _tx_buf = uart.tx_buf
            uart_ffi.tx_buf = ctypes.cast(_tx_buf, ctypes.c_void_p).value
            uart_ffi.tx_cap = len(_tx_buf)
        if uart.tx_wr is not None:
            _tx_wr = uart.tx_wr
            uart_ffi.tx_wr = ctypes.addressof(_tx_wr)
        uart_ffi.ie = uart.ie
        uart_ffi.txctrl = uart.txctrl
        uart_ffi.rxctrl = uart.rxctrl
        uart_ffi.rx_fifo_len = uart.rx_fifo_len
        uart_ffi.tx_notify_fd = uart.tx_notify_fd
        uart_ffi.no_stdout = uart.no_stdout
        if uart.rx_notify is not None:
            uart_ffi.rx_notify = ctypes.cast(
                ctypes.pointer(uart.rx_notify), ctypes.c_void_p).value

    # --- FfiVirtIOCtx ---
    _virtio_ffi = FfiVirtIOCtx()
    _virtio_ffi_ptr = None
    if virtio is not None and virtio.base != 0:
        _virtio_ffi.base = virtio.base
        _virtio_ffi.capacity = virtio.capacity
        _virtio_ffi.queue_num_max = virtio.queue_num_max
        _virtio_ffi.device_features_sel = virtio.device_features_sel
        _virtio_ffi.driver_features_sel = virtio.driver_features_sel
        _virtio_ffi.driver_features = virtio.driver_features
        _virtio_ffi.queue_sel = virtio.queue_sel
        _virtio_ffi.queue_num = virtio.queue_num if virtio.queue_num else virtio.queue_num_max
        _virtio_ffi.queue_ready = 1 if virtio.queue_ready else 0
        _virtio_ffi.queue_desc = virtio.queue_desc
        _virtio_ffi.queue_driver = virtio.queue_driver
        _virtio_ffi.queue_device = virtio.queue_device
        _virtio_ffi.status = virtio.status
        _virtio_ffi.interrupt_status = virtio.interrupt_status
        _virtio_ffi_ptr = ctypes.pointer(_virtio_ffi)

    # Breakpoint addresses — build a ctypes array if any provided
    _bp_arr = None
    if bp_addrs:
        _bp_arr = (ctypes.c_uint64 * len(bp_addrs))()
        for i, addr in enumerate(bp_addrs):
            _bp_arr[i] = addr & 0xFFFF_FFFF_FFFF_FFFF

    # --- Build grouped FFI structs ---
    _hart_ctx = FfiHartCtx()
    _hart_ctx.states = ctypes.cast(states, ctypes.c_void_p).value or 0
    _hart_ctx.num_harts = num_harts
    _hart_ctx.max_instrs = max_instrs
    _hart_ctx.result = ctypes.cast(ctypes.byref(result), ctypes.c_void_p).value or 0
    _hart_ctx.stop_flag = (
        ctypes.cast(ctypes.pointer(stop_flag), ctypes.c_void_p).value
        if stop_flag is not None else 0
    )
    _hart_ctx.ext_irq = (
        ctypes.cast(ctypes.pointer(ext_irq), ctypes.c_void_p).value
        if ext_irq is not None else 0
    )

    _periph_ctx = FfiPeriphCtx()
    _periph_ctx.mem = ctypes.addressof(mem)
    _periph_ctx.pmp = ctypes.addressof(pmp_ffi)
    _periph_ctx.clint = ctypes.addressof(clint_ffi)
    _periph_ctx.dev = ctypes.addressof(dev_ffi)
    _periph_ctx.uart = ctypes.addressof(uart_ffi)
    _periph_ctx.virtio = (
        ctypes.cast(_virtio_ffi_ptr, ctypes.c_void_p).value
        if _virtio_ffi_ptr else 0
    )

    _bp_ctx = FfiBpCtx()
    _bp_ctx.addrs = ctypes.cast(_bp_arr, ctypes.c_void_p).value if _bp_arr else 0
    _bp_ctx.count = len(bp_addrs) if bp_addrs else 0

    _tlb_ctx = FfiTlbCtx()
    _tlb_ctx.gen = ctypes.addressof(tlb_gen) if tlb_gen is not None else 0
    _tlb_ctx.gen_per_hart = (
        ctypes.addressof(tlb_gen_per_hart) if tlb_gen_per_hart is not None else 0
    )

    _lib.run_parallel(
        ctypes.byref(_hart_ctx),
        ctypes.byref(_periph_ctx),
        ctypes.byref(_bp_ctx),
        ctypes.byref(_tlb_ctx),
    )

    # Sync mtime back from Rust (advanced by per-instruction AtomicU64 ops).
    # Also sync MSIP and MTIMECMP — concurrent CLINT inline handling may have
    # modified the shared atomic arrays; copy modifications back to the
    # caller's ClintInfo so the emulator's CLINT sync path picks them up.
    if clint is not None:
        clint.mtime = _mtime_val.value
        if clint.msip is not None:
            for i in range(min(nh, len(clint.msip))):
                clint.msip[i] = int(_msip_buf[i])
        if clint.mtimecmp is not None:
            for i in range(min(nh, len(clint.mtimecmp))):
                clint.mtimecmp[i] = int(_mtc_buf[i])

    # Return the virtio FFI struct so the caller can read back changed fields.
    return _virtio_ffi if virtio is not None else None


# ============================================================
#  libtermio.so — Terminal I/O background thread (独立于 CPU 模拟)
# ============================================================

_TERMIO_LIB_PATH = _NATIVE_DIR / f"libtermio{_EXT}"
_termio_lib: ctypes.CDLL | None = None


class TermIoHandle(ctypes.Structure):
    """termio 线程共享状态 — 必须与 Rust ``#[repr(C)] TermIoHandle`` 布局一致.

    Rust 侧 ``test_handle_layout_locked`` 锁定 sizeof=104 及各字段偏移;
    修改任一侧字段必须同步另一侧。

    由 :class:`pyremu.peripheral.termio.TerminalIO` 构建并持有: 指针字段指向
    其 ctypes 缓冲区。``terminal_io_start`` 在 Rust 侧复制全部字段后立即返回,
    结构体本身无需长期存活, 但指针指向的缓冲区必须存活至线程退出。
    """
    _fields_ = [
        ("stdin_fd", ctypes.c_int32),      # RawFd for stdin
        ("stdout_fd", ctypes.c_int32),     # RawFd for stdout
        ("rx_buf", ctypes.c_void_p),       # *mut u8 — shared RX ring buffer
        ("rx_cap", ctypes.c_uint32),       # capacity of rx_buf (power of two)
        ("rx_wr", ctypes.c_void_p),        # *mut AtomicU32 — write index (I/O thread)
        ("rx_rd", ctypes.c_void_p),        # *mut AtomicU32 — read index (Python, 容量控制)
        ("tx_buf", ctypes.c_void_p),       # *mut u8 — shared TX ring buffer
        ("tx_cap", ctypes.c_uint32),       # entry capacity (= byte capacity / 2)
        ("tx_wr", ctypes.c_void_p),        # *mut AtomicU32 — write index (hart threads)
        ("tx_drain", ctypes.c_void_p),     # *mut AtomicU32 — drain index (I/O thread 独占)
        ("stop_flag", ctypes.c_void_p),    # *mut AtomicU8 — stop request flag
        ("pause_flag", ctypes.c_void_p),   # *mut AtomicU8 — pause (debugger break)
        ("rx_notify", ctypes.c_void_p),    # *mut AtomicU8 — new RX data notification
        ("rx_notify_fd", ctypes.c_int32),  # fd: Rust termio 写 1B 唤醒 RX daemon
    ]


try:
    _termio_lib = _load_lib_safe(_TERMIO_LIB_PATH)
    if _termio_lib is None:
        raise OSError(f"无法加载 {_TERMIO_LIB_PATH.name}")

    _termio_lib.terminal_io_start.argtypes = [ctypes.POINTER(TermIoHandle)]
    _termio_lib.terminal_io_start.restype = ctypes.c_int32

    _termio_lib.terminal_io_stop.argtypes = []
    _termio_lib.terminal_io_stop.restype = None

    _termio_lib.terminal_io_is_running.argtypes = []
    _termio_lib.terminal_io_is_running.restype = ctypes.c_int32

    _termio_lib.terminal_io_attach.argtypes = [ctypes.POINTER(TermIoHandle)]
    _termio_lib.terminal_io_attach.restype = ctypes.c_int32

except OSError as exc:
    logger.warning(
        "无法加载 terminal I/O 加速库 ({}): {} — 降级为 Python 轮询 stdin",
        _TERMIO_LIB_PATH.name,
        exc,
    )


def termio_available() -> bool:
    """Return ``True`` if the terminal I/O library (libtermio.so) is loaded."""
    return _termio_lib is not None


def termio_start(handle: TermIoHandle) -> int:
    """Start the terminal I/O background thread in Rust (raw mode).

    *handle* 由调用方 (TerminalIO) 构建; 其指针字段指向的缓冲区必须
    存活至线程退出。

    Returns 0 on success, -1 on error (thread already running or stdin
    is not a TTY), -2 if libtermio.so is not available.
    """
    if _termio_lib is None:
        return -2  # not available
    return _termio_lib.terminal_io_start(ctypes.byref(handle))


def termio_stop() -> None:
    """Signal the I/O thread to stop and wait for it to exit."""
    if _termio_lib is not None:
        _termio_lib.terminal_io_stop()


def termio_attach(handle: TermIoHandle) -> int:
    """Attach I/O thread to already-configured fds (no termios changes)."""
    if _termio_lib is None:
        return -2
    return _termio_lib.terminal_io_attach(ctypes.byref(handle))


def termio_is_running() -> bool:
    """Return True if the terminal I/O thread is currently running."""
    if _termio_lib is not None:
        return _termio_lib.terminal_io_is_running() != 0
    return False

