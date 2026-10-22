#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acceleration layer by pre-compiled dynamic library.
Rust cdylib via ctypes, with pure Python fallback

Loads the pre-compiled Rust shared library.  If it is missing or the platform
is unsupported, ``loguru`` emits a warning and the module degrades to pure
Python equivalents transparently.
"""

from __future__ import annotations

import atexit
from collections.abc import Callable
import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pyremu.memory.pmp import PmpAccessInfo

# ============================================================
#  Platform detection
# ============================================================

_EXT = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
_NATIVE_DIR = Path(__file__).resolve().parent
_LIB_PATH = _NATIVE_DIR / f"libdecode{_EXT}"  # 单独留在外部，好减少SIGBUS/SIGSEGM一类的错误

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

    # instruction decode
    _lib.decode_fields.argtypes = [ctypes.c_uint32]
    _lib.decode_fields.restype = DecodedFields
    _lib.decode_compressed.argtypes = [ctypes.c_uint16]
    _lib.decode_compressed.restype = CompressedFields

    # PMP check
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
        ctypes.c_uint64,  # mdid
    ]
    _lib.pmp_check.restype = ctypes.c_uint8

    # Sv39 MMU helpers
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

    # ALU pure-compute ops (return Alu64 with trap flag)
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
        ctypes.c_uint8,  # funct7
        ctypes.c_uint8,  # funct3
        ctypes.c_uint8,  # rs2
        ctypes.c_uint64,  # rs1_bits
        ctypes.c_uint64,  # rs2_bits
        ctypes.c_uint8,  # frm
    ]
    _lib.fp_exec_op.restype = FpOut
    _lib.fp_exec_fma.argtypes = [
        ctypes.c_uint8,  # opcode
        ctypes.c_uint8,  # rm (funct3)
        ctypes.c_uint8,  # fmt
        ctypes.c_uint64,  # rs1
        ctypes.c_uint64,  # rs2
        ctypes.c_uint64,  # rs3
        ctypes.c_uint8,  # frm
    ]
    _lib.fp_exec_fma.restype = FpOut

    # RAM direct read/write (zero-allocation fast path)
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

    # concurrent thread-per-hart execution engine
    _lib.run_parallel.argtypes = [
        ctypes.c_void_p,  # hart: *const FfiHartCtx
        ctypes.c_void_p,  # mem_ctx: *const FfiMemCtx
        ctypes.c_void_p,  # intr_ctx: *const FfiInterruptCtx
        ctypes.c_void_p,  # dev_ctx: *const FfiDevicesCtx
        ctypes.c_void_p,  # bp: *const FfiBpCtx
        ctypes.c_void_p,  # tlb: *mut FfiTlbCtx
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
    """Return ``True`` if the acceleration library is loaded."""
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
    f.imm_b = (imm_b & 0xFFF) - (imm_b & 0x1000) & 0xFFFF_FFFF_FFFF_FFFF

    imm_j = ((instr >> 31) & 1) << 20
    imm_j |= ((instr >> 21) & 0x3FF) << 1
    imm_j |= ((instr >> 20) & 1) << 11
    imm_j |= ((instr >> 12) & 0xFF) << 12
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
    if _lib:
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
    imm, imm2 = _dc_immediates(h, cf.funct3, cf.quadrant)

    cf.imm = imm & 0xFFFF_FFFF_FFFF_FFFF
    cf.imm2 = imm2 & 0xFFFF_FFFF_FFFF_FFFF
    return cf


def _dc_immediates(h: int, funct3: int, quadrant: int) -> tuple[int, int]:
    """按象限解码立即数; 第二个分量只用于象限 2 的存储类指令."""
    if quadrant == 0:
        return _dc_imm_q0(h, funct3), 0
    if quadrant == 1:
        return _dc_imm_q1(h, funct3), 0
    if quadrant == 2:
        return _dc_imm_q2(h, funct3)
    return 0, 0


def _dc_imm_q0(h: int, funct3: int) -> int:
    """象限 0 的立即数: C.ADDI4SPN / C.LW / C.SW / C.FLD / C.LD / C.FSD / C.SD."""
    if funct3 == 0:
        return (
            ((h >> 6) & 0x1) << 2
            | ((h >> 5) & 0x1) << 3
            | ((h >> 11) & 0x1) << 4
            | ((h >> 12) & 0x1) << 5
            | ((h >> 7) & 0xF) << 6
        )
    if funct3 in (2, 6):
        # C.LW / C.SW: uimm[5:3|2|6]
        return ((h >> 6) & 0x1) << 2 | ((h >> 10) & 0x7) << 3 | ((h >> 5) & 0x1) << 6
    if funct3 in (1, 3, 5, 7):
        # C.FLD / C.LD / C.FSD / C.SD: uimm[5:3|6|7]
        return ((h >> 5) & 0x3) << 6 | ((h >> 10) & 0x7) << 3
    return 0


def _dc_imm_q1_lui(h: int, bit12: int, bits_6_2: int) -> int:
    """funct3=011: C.ADDI16SP (rd=2) 的 10 位 nzimm, 否则 C.LUI 的 6 位 nzimm."""
    bits_11_7 = (h >> 7) & 0x1F
    if bits_11_7 == 2:
        # C.ADDI16SP: 10-bit signed nzimm
        nz = ((h >> 6) & 0x1) << 4
        nz |= ((h >> 2) & 0x1) << 5
        nz |= ((h >> 5) & 0x1) << 6
        nz |= ((h >> 3) & 0x3) << 7
        nz |= bit12 << 9
        return (nz & 0x1FF) - (nz & 0x200) & 0xFFFF_FFFF_FFFF_FFFF
    # C.LUI: 6-bit signed nzimm (shift-left-12 in caller)
    nz = (bit12 << 5) | bits_6_2
    return (nz & 0x1F) - (nz & 0x20) & 0xFFFF_FFFF_FFFF_FFFF


def _dc_imm_q1(h: int, funct3: int) -> int:
    """象限 1 的立即数: 立即数算术、C.ADDI16SP / C.LUI、跳转与分支."""
    bit12 = (h >> 12) & 0x1
    bits_6_2 = (h >> 2) & 0x1F
    if funct3 in (0, 1, 2):
        # C.ADDI / C.ADDIW / C.LI: 6-bit signed imm
        raw = (bit12 << 5) | bits_6_2
        return (raw & 0x1F) - (raw & 0x20) & 0xFFFF_FFFF_FFFF_FFFF
    if funct3 == 3:
        return _dc_imm_q1_lui(h, bit12, bits_6_2)
    if funct3 == 5:
        # C.J: 11-bit signed offset
        off = bit12 << 11
        off |= ((h >> 8) & 0x1) << 10
        off |= ((h >> 9) & 0x3) << 8
        off |= ((h >> 6) & 0x1) << 7
        off |= ((h >> 7) & 0x1) << 6
        off |= ((h >> 2) & 0x1) << 5
        off |= ((h >> 11) & 0x1) << 4
        off |= ((h >> 3) & 0x7) << 1
        return (off & 0x7FF) - (off & 0x800) & 0xFFFF_FFFF_FFFF_FFFF
    if funct3 in (6, 7):
        # C.BEQZ / C.BNEZ: 9-bit signed offset
        off = bit12 << 8
        off |= ((h >> 10) & 0x3) << 3  # offset[4:3]
        off |= ((h >> 5) & 0x3) << 6  # offset[7:6]
        off |= ((h >> 2) & 0x1) << 5  # offset[5]
        off |= ((h >> 3) & 0x3) << 1  # offset[2:1]
        return (off & 0xFF) - (off & 0x100) & 0xFFFF_FFFF_FFFF_FFFF
    return 0


def _dc_imm_q2(h: int, funct3: int) -> tuple[int, int]:
    """象限 2 的立即数: 载入类的偏移写入第一个分量, 存储类的偏移写入第二个."""
    bit12 = (h >> 12) & 0x1
    bits_6_2 = (h >> 2) & 0x1F
    if funct3 == 0:
        return (bit12 << 5) | bits_6_2, 0
    if funct3 == 2:
        return ((h >> 2) & 0x3) << 6 | bit12 << 5 | ((h >> 4) & 0x7) << 2, 0
    if funct3 in (1, 3):
        # C.FLDSP / C.LDSP: uimm[5|4:3|8:6]
        return ((h >> 2) & 0x7) << 6 | bit12 << 5 | ((h >> 5) & 0x3) << 3, 0
    if funct3 == 6:
        return 0, ((h >> 9) & 0xF) << 2 | ((h >> 7) & 0x3) << 6
    if funct3 in (5, 7):
        # C.FSDSP / C.SDSP: uimm[5:3|8:6]
        return 0, ((h >> 7) & 0x7) << 6 | ((h >> 10) & 0x7) << 3
    return 0, 0


def decode_compressed(half: int) -> CompressedFields:
    """Extract all 16-bit compressed instruction fields.

    Uses the native Rust library if available; otherwise falls back to
    pure Python.
    """
    if _lib:
        return _lib.decode_compressed(half)
    return _decode_compressed_py(half)


# ============================================================
# PMP check
# ============================================================


def pmp_check(
    cfg_bytes: bytes,
    addr_array,
    num_entries: int,
    access: PmpAccessInfo,
) -> bool:
    """Check PMP access via Rust — all state passed as parameters.

    *cfg_bytes* is a ``bytes`` object (64 bytes, one per entry).
    *addr_array* is an ``array('Q')`` or list of 64 u64 raw pmpaddr values.
    *access* 给出物理地址、访问宽度与访问属性.

    Returns True if the access is allowed.
    """
    if _lib is None:
        # Fallback: caller should use Pmp.check() instead
        raise RuntimeError("native library not available — use Pmp.check()")
    cfg_buf = (ctypes.c_uint8 * len(cfg_bytes)).from_buffer_copy(cfg_bytes)
    addr_buf = (ctypes.c_uint64 * len(addr_array))(*addr_array)
    return bool(
        _lib.pmp_check(
            ctypes.addressof(cfg_buf),
            ctypes.addressof(addr_buf),
            num_entries,
            access.pa,
            access.size,
            1 if access.is_write else 0,
            1 if access.is_execute else 0,
            access.mode_val,
            access.mstatus_val,
            access.pmpsplit,
            access.mdid,
        )
    )


# ============================================================
# Sv39 MMU helpers
# ============================================================


def sv39_decompose_va(va: int) -> Sv39Vpn:
    """Decompose a 39-bit virtual address into VPN[2:0] and page offset."""
    if _lib:
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
    if _lib:
        return _lib.pte_parse(raw)
    raw &= 0xFFFF_FFFF_FFFF_FFFF
    pte_v = 1 << 0
    pte_r = 1 << 1
    pte_x = 1 << 3
    ppn = (raw >> 10) & 0xFFF_FFFF_FFFF
    v = 1 if raw & pte_v else 0
    r_or_x = raw & (pte_r | pte_x)
    f = PteFields()
    f.ppn = ppn
    f.perm = raw & 0x1E  # R|W|X|U = bits 4:1
    f.v = v
    f.is_leaf = 1 if v and r_or_x else 0
    f.is_ptr = 1 if v and not r_or_x else 0
    return f


def pte_assemble_pa(ppn: int, va: int, level: int) -> int:
    """Compute physical address from leaf PTE PPN, original VA, and page level."""
    if _lib:
        return _lib.pte_assemble_pa(ppn, va, level)
    if level == 1:
        # 2 MiB superpage: PPN[8:0] from VA[20:12]
        vpn0 = (va >> 12) & 0x1FF
        ppn = (ppn & 0xFFFF_FFFF_FFFF_FE00) | vpn0
    offset = va & 0xFFF
    return ((ppn << 12) | offset) & 0xFFFF_FFFF_FFFF_FFFF


def sv39_page_size(level: int) -> int:
    """Return page size in bytes for a given Sv39 leaf level."""
    if _lib:
        return _lib.sv39_page_size(level)
    return 2 * 1024 * 1024 if level == 1 else 4096


# ============================================================
# ALU pure-compute ops
# ============================================================

_sint64 = ctypes.c_int64
_uint64 = ctypes.c_uint64


def _trunc_div(a: int, b: int) -> int:
    """截断除法: 商向零取整, 除数为零时返回全 1.

    不使用 ``int(a / b)``: 真除法先把商舍入到双精度的 53 位有效位, 商的绝对值
    达到 2**53 之后该舍入改变结果.
    """
    if b == 0:
        return 0xFFFF_FFFF_FFFF_FFFF
    quotient = abs(a) // abs(b)
    return -quotient if (a < 0) != (b < 0) else quotient


def _trunc_rem(a: int, b: int) -> int:
    """截断取余: 除数为零时返回被除数, 与被舍入的商无关."""
    if b == 0:
        return a
    return a - b * _trunc_div(a, b)


def _sint32(value: int) -> int:
    """把低 32 位按有符号数解释, 取值范围 [-2**31, 2**31)."""
    low = value & 0xFFFF_FFFF
    return low - 0x1_0000_0000 if low & 0x8000_0000 else low


def _sext32(value: int) -> int:
    """把 32 位结果按符号扩展到 64 位, 返回值规范到 [0, 2**64)."""
    return _sint32(value) & 0xFFFF_FFFF_FFFF_FFFF


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
    if _lib:
        return _lib.exec_alu_op(funct3, funct7, v1, v2)
    # Pure Python fallback
    handler = _ALU_FUNCT3.get(funct3)
    if handler is None:
        return _alu_ill()
    return handler(funct7, v1, v2)


def _alu_add_sub_mul(funct7: int, v1: int, v2: int) -> Alu64:
    """funct3=000: add / mul / sub."""
    if funct7 == 0:
        return _alu_ok((v1 + v2) & 0xFFFF_FFFF_FFFF_FFFF)
    if funct7 == 1:
        return _alu_ok((v1 * v2) & 0xFFFF_FFFF_FFFF_FFFF)
    if funct7 == 0x20:
        return _alu_ok((v1 - v2) & 0xFFFF_FFFF_FFFF_FFFF)
    return _alu_ill()


def _alu_sll_mulh(funct7: int, v1: int, v2: int) -> Alu64:
    """funct3=001: sll / mulh."""
    if funct7 == 0:
        return _alu_ok((v1 << (v2 & 0x3F)) & 0xFFFF_FFFF_FFFF_FFFF)
    if funct7 == 1:
        s1 = _sint64(v1).value
        s2 = _sint64(v2).value
        return _alu_ok(_sint64((s1 * s2) >> 64).value & 0xFFFF_FFFF_FFFF_FFFF)
    return _alu_ill()


def _alu_slt_mulhsu(funct7: int, v1: int, v2: int) -> Alu64:
    """funct3=010: slt / mulhsu."""
    if funct7 == 0:
        return _alu_ok(1 if _sint64(v1).value < _sint64(v2).value else 0)
    if funct7 == 1:
        s1 = _sint64(v1).value
        u2 = _uint64(v2).value
        return _alu_ok(_sint64((s1 * u2) >> 64).value & 0xFFFF_FFFF_FFFF_FFFF)
    return _alu_ill()


def _alu_sltu_mulhu(funct7: int, v1: int, v2: int) -> Alu64:
    """funct3=011: sltu / mulhu."""
    if funct7 == 0:
        return _alu_ok(1 if _uint64(v1).value < _uint64(v2).value else 0)
    if funct7 == 1:
        u1 = _uint64(v1).value
        u2 = _uint64(v2).value
        return _alu_ok((u1 * u2) >> 64)
    return _alu_ill()


def _alu_xor_div(funct7: int, v1: int, v2: int) -> Alu64:
    """funct3=100: xor / div."""
    if funct7 == 0:
        return _alu_ok(v1 ^ v2)
    if funct7 == 1:
        return _alu_ok(
            _trunc_div(_sint64(v1).value, _sint64(v2).value) & 0xFFFF_FFFF_FFFF_FFFF
        )
    return _alu_ill()


def _alu_srl_sra_divu(funct7: int, v1: int, v2: int) -> Alu64:
    """funct3=101: srl / divu / sra."""
    if funct7 == 0:
        return _alu_ok(_uint64(v1).value >> (v2 & 0x3F))
    if funct7 == 1:
        return _alu_ok(_trunc_div(_uint64(v1).value, _uint64(v2).value))
    if funct7 == 0x20:
        return _alu_ok(_sint64(_sint64(v1).value >> (v2 & 0x3F)).value & 0xFFFF_FFFF_FFFF_FFFF)
    return _alu_ill()


def _alu_or_rem(funct7: int, v1: int, v2: int) -> Alu64:
    """funct3=110: or / rem."""
    if funct7 == 0:
        return _alu_ok(v1 | v2)
    if funct7 == 1:
        return _alu_ok(
            _trunc_rem(_sint64(v1).value, _sint64(v2).value) & 0xFFFF_FFFF_FFFF_FFFF
        )
    return _alu_ill()


def _alu_and_remu(funct7: int, v1: int, v2: int) -> Alu64:
    """funct3=111: and / remu."""
    if funct7 == 0:
        return _alu_ok(v1 & v2)
    if funct7 == 1:
        return _alu_ok(
            _trunc_rem(_uint64(v1).value, _uint64(v2).value) & 0xFFFF_FFFF_FFFF_FFFF
        )
    return _alu_ill()


_ALU_FUNCT3: dict[int, Callable[[int, int, int], Alu64]] = {
    0b000: _alu_add_sub_mul,
    0b001: _alu_sll_mulh,
    0b010: _alu_slt_mulhsu,
    0b011: _alu_sltu_mulhu,
    0b100: _alu_xor_div,
    0b101: _alu_srl_sra_divu,
    0b110: _alu_or_rem,
    0b111: _alu_and_remu,
}


def exec_op_imm(funct3: int, funct7: int, v1: int, imm: int) -> Alu64:
    """RISC-V I-type ALU: returns Alu64(value, trap)."""
    if _lib:
        return _lib.exec_op_imm(funct3, funct7, v1, imm)
    handler = _OP_IMM_FUNCT3.get(funct3)
    if handler is None:
        return _alu_ill()
    return handler(funct7, v1, imm)


def _opimm_addi(funct7: int, v1: int, imm: int) -> Alu64:
    return _alu_ok((v1 + imm) & 0xFFFF_FFFF_FFFF_FFFF)


def _opimm_slli(funct7: int, v1: int, imm: int) -> Alu64:
    if funct7 >> 1 != 0:
        return _alu_ill()
    return _alu_ok((v1 << (imm & 0x3F)) & 0xFFFF_FFFF_FFFF_FFFF)


def _opimm_slti(funct7: int, v1: int, imm: int) -> Alu64:
    # SLTI 的两个操作数都按有符号数比较: imm 已规范到 [0, 2**64), 不能直接参与比较
    return _alu_ok(1 if _sint64(v1).value < _sint64(imm).value else 0)


def _opimm_sltiu(funct7: int, v1: int, imm: int) -> Alu64:
    return _alu_ok(1 if _uint64(v1).value < _uint64(imm).value else 0)


def _opimm_xori(funct7: int, v1: int, imm: int) -> Alu64:
    return _alu_ok(v1 ^ imm)


def _opimm_srli_srai(funct7: int, v1: int, imm: int) -> Alu64:
    """funct6 区分 SRLI 与 SRAI, 移位量为 imm 的低 6 位."""
    shamt = imm & 0x3F
    funct6 = funct7 >> 1
    if funct6 == 0:
        return _alu_ok(_uint64(v1).value >> shamt)
    if funct6 == 0x10:
        return _alu_ok(_sint64(_sint64(v1).value >> shamt).value & 0xFFFF_FFFF_FFFF_FFFF)
    return _alu_ill()


def _opimm_ori(funct7: int, v1: int, imm: int) -> Alu64:
    return _alu_ok(v1 | imm)


def _opimm_andi(funct7: int, v1: int, imm: int) -> Alu64:
    return _alu_ok(v1 & imm)


_OP_IMM_FUNCT3: dict[int, Callable[[int, int, int], Alu64]] = {
    0b000: _opimm_addi,
    0b001: _opimm_slli,
    0b010: _opimm_slti,
    0b011: _opimm_sltiu,
    0b100: _opimm_xori,
    0b101: _opimm_srli_srai,
    0b110: _opimm_ori,
    0b111: _opimm_andi,
}


def exec_op32(funct3: int, funct7: int, v1: int, v2: int) -> Alu64:
    """RISC-V RV64 32-bit word ALU: returns Alu64(value, trap)."""
    if _lib:
        return _lib.exec_op32(funct3, funct7, v1, v2)
    handler = _OP32_FUNCT3.get(funct3)
    if handler is None:
        return _alu_ill()
    return handler(funct7, v1, v2)


def _op32_add_sub_mul(funct7: int, v1: int, v2: int) -> Alu64:
    """funct3=000: addw / mulw / subw."""
    w1 = v1 & 0xFFFF_FFFF
    w2 = v2 & 0xFFFF_FFFF
    if funct7 == 0:
        return _alu_ok(_sext32((w1 + w2) & 0xFFFF_FFFF))
    if funct7 == 1:
        return _alu_ok(_sext32((w1 * w2) & 0xFFFF_FFFF))
    if funct7 == 0x20:
        return _alu_ok(_sext32((w1 - w2) & 0xFFFF_FFFF))
    return _alu_ill()


def _op32_sllw(funct7: int, v1: int, v2: int) -> Alu64:
    if funct7 != 0:
        return _alu_ill()
    return _alu_ok(_sext32(((v1 & 0xFFFF_FFFF) << (v2 & 0x1F)) & 0xFFFF_FFFF))


def _op32_divw(funct7: int, v1: int, v2: int) -> Alu64:
    if funct7 != 1:
        return _alu_ill()
    s1 = _sint32(v1 & 0xFFFF_FFFF)
    s2 = _sint32(v2 & 0xFFFF_FFFF)
    return _alu_ok(_sext32(_trunc_div(s1, s2) & 0xFFFF_FFFF))


def _op32_srl_sra_divuw(funct7: int, v1: int, v2: int) -> Alu64:
    """funct3=101: srlw / divuw / sraw, 操作数一律按有符号数解释."""
    w1 = v1 & 0xFFFF_FFFF
    w2 = v2 & 0xFFFF_FFFF
    if funct7 == 0:
        return _alu_ok(_sext32((w1 >> (w2 & 0x1F)) & 0xFFFF_FFFF))
    if funct7 == 1:
        return _alu_ok(_sext32(_trunc_div(w1, w2) & 0xFFFF_FFFF))
    if funct7 == 0x20:
        return _alu_ok(_sext32((_sint32(w1) >> (w2 & 0x1F)) & 0xFFFF_FFFF))
    return _alu_ill()


def _op32_remw(funct7: int, v1: int, v2: int) -> Alu64:
    if funct7 != 1:
        return _alu_ill()
    s1 = _sint32(v1 & 0xFFFF_FFFF)
    s2 = _sint32(v2 & 0xFFFF_FFFF)
    return _alu_ok(_sext32(_trunc_rem(s1, s2) & 0xFFFF_FFFF))


def _op32_remuw(funct7: int, v1: int, v2: int) -> Alu64:
    if funct7 != 1:
        return _alu_ill()
    w1 = v1 & 0xFFFF_FFFF
    w2 = v2 & 0xFFFF_FFFF
    return _alu_ok(_sext32(_trunc_rem(w1, w2) & 0xFFFF_FFFF))


_OP32_FUNCT3: dict[int, Callable[[int, int, int], Alu64]] = {
    0b000: _op32_add_sub_mul,
    0b001: _op32_sllw,
    0b100: _op32_divw,
    0b101: _op32_srl_sra_divuw,
    0b110: _op32_remw,
    0b111: _op32_remuw,
}


def exec_op_imm32(funct3: int, funct7: int, v1: int, imm: int) -> Alu64:
    """RISC-V RV64 32-bit immediate ALU: returns Alu64(value, trap)."""
    if _lib:
        return _lib.exec_op_imm32(funct3, funct7, v1, imm)
    handler = _OP_IMM32_FUNCT3.get(funct3)
    if handler is None:
        return _alu_ill()
    return handler(funct7, v1, imm)


def _opimm32_addiw(funct7: int, v1: int, imm: int) -> Alu64:
    return _alu_ok(_sext32(((v1 & 0xFFFF_FFFF) + (imm & 0xFFFF_FFFF)) & 0xFFFF_FFFF))


def _opimm32_slliw(funct7: int, v1: int, imm: int) -> Alu64:
    if funct7 != 0:
        return _alu_ill()
    return _alu_ok(_sext32(((v1 & 0xFFFF_FFFF) << (imm & 0x1F)) & 0xFFFF_FFFF))


def _opimm32_srliw_sraiw(funct7: int, v1: int, imm: int) -> Alu64:
    """funct6 区分 SRLIW 与 SRAIW."""
    w1 = v1 & 0xFFFF_FFFF
    shamt = imm & 0x1F
    funct6 = funct7 >> 1
    if funct6 == 0:
        return _alu_ok(_sext32((w1 >> shamt) & 0xFFFF_FFFF))
    if funct6 == 0x10:
        return _alu_ok(_sext32((_sint32(w1) >> shamt) & 0xFFFF_FFFF))
    return _alu_ill()


_OP_IMM32_FUNCT3: dict[int, Callable[[int, int, int], Alu64]] = {
    0b000: _opimm32_addiw,
    0b001: _opimm32_slliw,
    0b101: _opimm32_srliw_sraiw,
}


# ============================================================
#  F/D floating point compute (native-only — no pure-Python FPU)
# ============================================================


def fp_exec_op(
    funct7: int,
    funct3: int,
    rs2: int,
    src_bits: tuple[int, int],
    frm: int,
) -> FpOut:
    """OP-FP compute via native softfloat.  Requires the native library.

    *src_bits* 为 (rs1 的位模式, rs2 的位模式); 整数源与浮点源都由调用方按
    指令语义取好.

    Raises NotImplementedError if ``.so`` is unavailable (caller converts
    to IllInstr) — there is no pure-Python floating-point fallback.
    """
    if _lib is None:
        raise NotImplementedError("FPU requires native library (libdecode.so)")
    rs1_bits, rs2_bits = src_bits
    return _lib.fp_exec_op(funct7, funct3, rs2, rs1_bits, rs2_bits, frm)


def fp_exec_fma(
    opcode: int,
    rm: int,
    fmt: int,
    src_bits: tuple[int, int, int],
    frm: int,
) -> FpOut:
    """FMA compute via native softfloat.  Requires the native library.

    *src_bits* 为 (rs1, rs2, rs3) 三个浮点源操作数的位模式.
    """
    if _lib is None:
        raise NotImplementedError("FPU requires native library (libdecode.so)")
    rs1, rs2, rs3 = src_bits
    return _lib.fp_exec_fma(opcode, rm, fmt, rs1, rs2, rs3, frm)


# ============================================================
# RAM direct read/write (zero-allocation fast path)
# ============================================================


def bus_read_ram(
    ram_win: tuple[int, int, int, int, int],
    pa: int,
    size: int,
    out_buf,  # ctypes array for output (pre-allocated 8 bytes)
) -> int:
    """Read *size* bytes from physical address *pa* directly into *out_buf*.

    *ram_win* 为 RAM 直读窗口: (RAM 缓冲区首地址, 字节数, 基址, 影子窗口基址,
    影子窗口字节数).

    Returns 1 if the address falls within RAM (primary or shadow); 0 otherwise.
    Caller falls back to device / L2-cache / error logic on failure.
    """
    if _lib:
        return _lib.bus_read_ram(*ram_win, pa, size, out_buf)
    return 0  # no native lib — caller uses pure-Python path


def bus_write_ram(
    ram_win: tuple[int, int, int, int, int],
    pa: int,
    data,  # bytes to write
    size: int,
) -> int:
    """Write *size* bytes from *data* to physical address *pa*.

    *ram_win* 的含义同 :func:`bus_read_ram`.

    Returns 1 if the address falls within RAM; 0 otherwise.
    """
    if _lib:
        return _lib.bus_write_ram(*ram_win, pa, data, size)
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
        ("timebase_hz", ctypes.c_uint64),  # mtime 时钟源 Hz; 置0即禁用mtime的推进
    ]


class DevMMIOAddrInfo(ctypes.Structure):
    """Device MMIO address ranges — matches Rust ``DevMMIOAddrInfo``."""

    _fields_ = [
        ("bases", ctypes.c_void_p),
        ("ends", ctypes.c_void_p),
        ("num", ctypes.c_uint8),
    ]


class FfiPlicCtx(ctypes.Structure):
    """PLIC state — matches Rust ``FfiPlicCtx``.

    指针字段指向 Emulator 持有的持久 ctypes 数组, Rust 在加速执行内联读写
    (claim/complete/priority/enable/threshold), 设备中断投递目标经 ``set_irq``
    直接 write-through pending/level, 保证 batch 内新设备中断对 Rust 可见.
    """

    _fields_ = [
        ("base", ctypes.c_uint64),
        ("num_sources", ctypes.c_uint32),
        ("num_contexts", ctypes.c_uint32),
        ("priority", ctypes.c_void_p),
        ("pending", ctypes.c_void_p),
        ("level", ctypes.c_void_p),
        ("enable", ctypes.c_void_p),
        ("threshold", ctypes.c_void_p),
        ("claimed", ctypes.c_void_p),
    ]


class FfiWatchdogCtx(ctypes.Structure):
    """Host-side execution watchdog — matches Rust ``FfiWatchdogCtx``.

    携带一次加速执行的时钟源超时时间 ``timeout_ns`` (纳秒, 0 = 禁用).
    """

    _fields_ = [
        ("timeout_ns", ctypes.c_uint64),
    ]


class FfiUartCtx(ctypes.Structure):
    """UART context — matches Rust ``FfiUartCtx``."""

    _fields_ = [
        ("base", ctypes.c_uint64),
        ("tx_buf", ctypes.c_void_p),
        ("tx_cap", ctypes.c_uint32),
        ("tx_wr", ctypes.c_void_p),
        ("ie", ctypes.c_uint32),  # IE 寄存器影子 (offset 0x10)
        ("txctrl", ctypes.c_uint32),  # TXCTRL 影子 (offset 0x08)
        ("rxctrl", ctypes.c_uint32),  # RXCTRL 影子 (offset 0x0C)
        ("rx_fifo_len", ctypes.c_uint32),  # RX FIFO 近似填充量
        ("tx_notify_fd", ctypes.c_int32),  # pipe write-end: Rust 通知 TX 线程
        ("no_stdout", ctypes.c_uint8),  # 1=Rust 不写 stdout, Python TX 统一输出
        ("rx_notify", ctypes.c_void_p),  # *mut u8 — TermIO 有新 RX 数据标志
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
        ("result", ctypes.c_void_p),
        ("stop_flag", ctypes.c_void_p),
        ("ext_irq", ctypes.c_void_p),
    ]


class FfiMemCtx(ctypes.Structure):
    """内存与保护上下文 — matches Rust ``FfiMemCtx``."""

    _fields_ = [
        ("mem", ctypes.c_void_p),
        ("pmp", ctypes.c_void_p),
    ]


class FfiInterruptCtx(ctypes.Structure):
    """中断控制器上下文 — matches Rust ``FfiInterruptCtx``."""

    _fields_ = [
        ("clint", ctypes.c_void_p),
        ("plic", ctypes.c_void_p),
    ]


class FfiDevicesCtx(ctypes.Structure):
    """设备上下文 — matches Rust ``FfiDevicesCtx``."""

    _fields_ = [
        ("dev", ctypes.c_void_p),
        ("uart", ctypes.c_void_p),
        ("virtio", ctypes.c_void_p),
        ("watchdog", ctypes.c_void_p),
    ]


class FfiBpCtx(ctypes.Structure):
    """Grouped breakpoint context — matches Rust ``FfiBpCtx``."""

    _fields_ = [
        ("addrs", ctypes.c_void_p),
        ("count", ctypes.c_uint32),
    ]


class FfiTlbCtx(ctypes.Structure):
    """TLB 刷新计数跨 FFI 边界传递的两处存放位置, 与 Rust ``FfiTlbCtx`` 一致."""

    _fields_ = [
        ("flush_cnt", ctypes.c_void_p),
        ("flush_cnt_per_hart", ctypes.c_void_p),
    ]


# ============================================================
#  Context info classes — Python-side helpers to build FFI structs
# ============================================================


class PmpInfo:
    """PMP configuration.

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
            raise ValueError(f"per-hart PMP num={self.num} 超出 FfiPmpCtx.num (u8) 范围")


@dataclass(slots=True)
class ClintInfo:
    """CLINT state"""

    mtime: int = 0
    mtimecmp: Any = None  # list or ctypes array
    msip: Any = None  # list or ctypes array (level + edge counter)
    base: int = 0
    timebase_hz: int = 10_000_000  # mtime 时钟源频率


@dataclass(slots=True)
class DevInfo:
    """Device MMIO ranges"""

    bases: Any = None  # ctypes array of uint64
    ends: Any = None  # ctypes array of uint64


@dataclass(slots=True)
class PlicInfo:
    """PLIC inline context — 持久 ctypes 数组, Rust 在加速执行内联读写.

    与 ClintInfo 每轮 marshal 重新分配不同: PLIC 状态必须跨 batch 持久, 因为
    设备中断投递目标需 write-through 到同一批数组, 使 batch 内
    新到达的设备中断对 Rust inline 仲裁可见. Emulator 持有这些数组, marshal
    仅复制 Python PLIC 状态入数组, unmarshal 反向复制.
    """

    base: int = 0
    num_sources: int = 0
    num_contexts: int = 0
    priority: Any = None  # ctypes array of uint8 (num_sources+1)
    pending: Any = None  # ctypes array of uint8 (num_sources+1)
    level: Any = None  # ctypes array of uint8 (num_sources+1)
    enable: Any = None  # ctypes array of uint32 (num_contexts * num_words)
    threshold: Any = None  # ctypes array of uint8 (num_contexts)
    claimed: Any = None  # ctypes array of uint32 (num_contexts)


@dataclass(slots=True)
class UartInfo:
    """UART context — lets Rust buffer sbi_printf output inline
    and handle IE/IP/TXCTRL register reads without exiting from speedup lib"""

    base: int = 0
    tx_buf: Any = None
    tx_wr: Any = None
    ie: int = 0
    txctrl: int = 0
    rxctrl: int = 0
    rx_fifo_len: int = 0
    tx_notify_fd: int = -1
    no_stdout: int = 0
    rx_notify: Any = None
    # 持久 ctypes FfiUartCtx — 由 Emulator 持有并注入 UART 设备, 设备侧
    # 状态变化时直接改写其寄存器字段; 提供时不再每轮重建快照.
    ffi: Any = None


@dataclass(slots=True)
class VirtIOInfo:
    """virtio-blk inline context — Rust handles all MMIO register
    accesses inline; only QueueNotify exits to Python.

    Carries the full runtime state of the virtio-blk MMIO register file
    across speedup process.  Without this, dynamic state written by the guest
    (queue descriptors, feature negotiation, InterruptStatus, etc.) is
    silently reset to zero, and the guest sees a dead device.
    """

    base: int = 0
    capacity: int = 0
    queue_num_max: int = 256
    device_features_sel: int = 0
    driver_features_sel: int = 0
    driver_features: int = 0
    queue_sel: int = 0
    queue_num: int = 0
    queue_ready: bool = False
    queue_desc: int = 0
    queue_driver: int = 0
    queue_device: int = 0
    status: int = 0
    interrupt_status: int = 0


@dataclass(slots=True)
class RunConfig:
    """一次 run_parallel 调用的输入: 必给的执行上下文与可选的设备状态."""

    states: Any  # ctypes array of HartState
    num_harts: int
    ram_ptr: int
    ram_size: int
    ram_base: int
    shadow_base: int
    shadow_size: int
    result: Any
    pmp: PmpInfo | None = None
    clint: ClintInfo | None = None
    dev: DevInfo | None = None
    uart: UartInfo | None = None
    virtio: VirtIOInfo | None = None
    plic: PlicInfo | None = None
    watchdog_timeout_ns: int = 0  # 时钟源超时时间 (纳秒); 0 = 禁用
    bp_addrs: list[int] | None = None
    stop_flag: Any = None  # ctypes.c_uint8 — Ctrl+Q 与设备共用的停止标志
    ext_irq: Any = None  # FfiExtIrqCtx — 外部中断上下文
    flush_cnt: Any = None  # ctypes.c_uint64 — 已发生的 TLB 刷新次数
    flush_cnt_per_hart: Any = None  # ctypes array of c_uint64 — 各 hart 上次见到的取值


@dataclass(slots=True)
class FfiParts:
    """一个 FFI 结构与它指向的 ctypes 缓冲区.

    FFI 结构内只存裸地址, 缓冲区由本对象的 buffers 持有, 存活至 run_parallel
    返回; 缺少该引用时引擎会读到已释放的内存.
    """

    ctx: Any
    buffers: tuple[Any, ...] = ()
    is_present: bool = True


def _addr_or_zero(obj) -> int:
    """对象为 None 时返回 0, 否则返回其地址.

    按"是否提供"判定, 不能用真值判定: ctypes 标量的真值取自它的当前取值
    (c_uint8(0) 为假), 真值判定会把取值为 0 的停止标志与计数器当作未提供,
    从而向引擎传空指针.
    """
    return ctypes.addressof(obj) if obj is not None else 0


class ClintMarshal:
    """CLINT 状态在调用期间的 ctypes 副本: 调用前写入, 返回后回读."""

    def __init__(self, clint: ClintInfo | None, num_harts: int) -> None:
        nh = self.num_harts = max(int(num_harts), 1)
        self.mtime = ctypes.c_uint64(0)
        self.mtimecmp = (ctypes.c_uint64 * nh)()
        self.msip = (ctypes.c_uint8 * nh)()
        if clint is None:
            for i in range(nh):
                self.mtimecmp[i] = 0xFFFF_FFFF_FFFF_FFFF
        else:
            self.mtime.value = clint.mtime
            if clint.mtimecmp:
                for i in range(min(nh, len(clint.mtimecmp))):
                    self.mtimecmp[i] = int(clint.mtimecmp[i])
            else:
                for i in range(nh):
                    self.mtimecmp[i] = 0xFFFF_FFFF_FFFF_FFFF
            if clint.msip:
                for i in range(min(nh, len(clint.msip))):
                    self.msip[i] = int(clint.msip[i])

        self.ctx = FfiClintCtx()
        self.ctx.mtime = ctypes.addressof(self.mtime)
        self.ctx.mtimecmp = ctypes.addressof(self.mtimecmp)
        self.ctx.msip = ctypes.addressof(self.msip)
        self.ctx.base = clint.base if clint is not None else 0
        self.ctx.timebase_hz = clint.timebase_hz if clint is not None else 0

    def sync_back(self, clint: ClintInfo | None) -> None:
        """把引擎改写后的 mtime、mtimecmp 与 msip 复制回调用方的 ClintInfo.

        mtime 由引擎逐指令按原子操作推进; mtimecmp 与 msip 可能被引擎内的
        内联 CLINT 处理改写, 两者都要回读.
        """
        if clint is None:
            return
        clint.mtime = self.mtime.value
        if clint.msip:
            for i in range(min(self.num_harts, len(clint.msip))):
                clint.msip[i] = int(self.msip[i])
        if clint.mtimecmp:
            for i in range(min(self.num_harts, len(clint.mtimecmp))):
                clint.mtimecmp[i] = int(self.mtimecmp[i])


def _build_pmp_ffi(pmp: PmpInfo | None) -> FfiParts:
    ffi = FfiPmpCtx()
    if pmp is None:
        return FfiParts(ffi)
    ffi.num = pmp.num
    ffi.pmpsplit = pmp.pmpsplit
    if pmp.num <= 0:
        return FfiParts(ffi)
    ffi.cfg = ctypes.addressof(pmp.cfg)
    ffi.addr = ctypes.addressof(pmp.addr)
    return FfiParts(ffi, (pmp.cfg, pmp.addr))


def _build_dev_ffi(dev: DevInfo | None) -> DevMMIOAddrInfo:
    ffi = DevMMIOAddrInfo()
    if dev is not None and dev.bases and dev.ends:
        ffi.bases = ctypes.addressof(dev.bases)
        ffi.ends = ctypes.addressof(dev.ends)
        ffi.num = len(dev.bases)
    return ffi


def _build_uart_ffi(uart: UartInfo | None) -> FfiParts:
    # 调用方提供持久上下文时直接复用 (其寄存器字段由设备侧持续改写, 见
    # UART._publish_native_regs); 否则按参数值构建一次性快照.
    if uart is None:
        return FfiParts(FfiUartCtx())
    if uart.ffi is not None:
        return FfiParts(uart.ffi)

    ffi = FfiUartCtx()
    ffi.base = uart.base
    buffers: list[Any] = []
    # 这三项按"是否提供"判定, 不能用真值判定: 它们可以是 ctypes 标量, 其真值
    # 取自当前取值 (c_uint32(0) 与 c_uint8(0) 均为假), 真值判定会把取值为 0 的
    # 写游标与接收通知标志当作未提供, 从而向引擎传空指针.
    if uart.tx_buf is not None:
        ffi.tx_buf = ctypes.addressof(uart.tx_buf)
        ffi.tx_cap = len(uart.tx_buf)
        buffers.append(uart.tx_buf)
    if uart.tx_wr is not None:
        ffi.tx_wr = ctypes.addressof(uart.tx_wr)
        buffers.append(uart.tx_wr)
    ffi.ie = uart.ie
    ffi.txctrl = uart.txctrl
    ffi.rxctrl = uart.rxctrl
    ffi.rx_fifo_len = uart.rx_fifo_len
    ffi.tx_notify_fd = uart.tx_notify_fd
    ffi.no_stdout = uart.no_stdout
    if uart.rx_notify is not None:
        ffi.rx_notify = ctypes.addressof(uart.rx_notify)
        buffers.append(uart.rx_notify)
    return FfiParts(ffi, tuple(buffers))


def _build_virtio_ffi(virtio: VirtIOInfo | None) -> FfiParts:
    ffi = FfiVirtIOCtx()
    if virtio is None or virtio.base == 0:
        return FfiParts(ffi, is_present=False)
    ffi.base = virtio.base
    ffi.capacity = virtio.capacity
    ffi.queue_num_max = virtio.queue_num_max
    ffi.device_features_sel = virtio.device_features_sel
    ffi.driver_features_sel = virtio.driver_features_sel
    ffi.driver_features = virtio.driver_features
    ffi.queue_sel = virtio.queue_sel
    ffi.queue_num = virtio.queue_num if virtio.queue_num else virtio.queue_num_max
    ffi.queue_ready = 1 if virtio.queue_ready else 0
    ffi.queue_desc = virtio.queue_desc
    ffi.queue_driver = virtio.queue_driver
    ffi.queue_device = virtio.queue_device
    ffi.status = virtio.status
    ffi.interrupt_status = virtio.interrupt_status
    return FfiParts(ffi)


def _build_plic_ffi(plic: PlicInfo | None) -> FfiParts:
    """持久数组指针直传, Rust 在连续执行内读写, 设备线程按同一数组写入."""
    ffi = FfiPlicCtx()
    if plic is None or plic.base == 0:
        return FfiParts(ffi, is_present=False)
    ffi.base = plic.base
    ffi.num_sources = plic.num_sources
    ffi.num_contexts = plic.num_contexts
    ffi.priority = ctypes.addressof(plic.priority)
    ffi.pending = ctypes.addressof(plic.pending)
    ffi.level = ctypes.addressof(plic.level)
    ffi.enable = ctypes.addressof(plic.enable)
    ffi.threshold = ctypes.addressof(plic.threshold)
    ffi.claimed = ctypes.addressof(plic.claimed)
    return FfiParts(ffi)


def _build_bp_ffi(bp_addrs: list[int] | None) -> FfiParts:
    ffi = FfiBpCtx()
    if not bp_addrs:
        return FfiParts(ffi)
    arr = (ctypes.c_uint64 * len(bp_addrs))()
    for i, addr in enumerate(bp_addrs):
        arr[i] = addr & 0xFFFF_FFFF_FFFF_FFFF
    ffi.addrs = ctypes.addressof(arr)
    ffi.count = len(bp_addrs)
    return FfiParts(ffi, (arr,))


def _build_watchdog_ffi(timeout_ns: int) -> tuple[FfiWatchdogCtx, bool]:
    """时钟源超时时间的 FFI 结构与"是否提供"标志; 0 表示禁用, 不传指针."""
    ffi = FfiWatchdogCtx()
    if not timeout_ns:
        return ffi, False
    ffi.timeout_ns = int(timeout_ns) & 0xFFFF_FFFF_FFFF_FFFF
    return ffi, True


def run_parallel(cfg: RunConfig) -> FfiVirtIOCtx | None:
    """Execute instructions concurrently (thread-per-hart) in Rust.

    Each non-halted hart runs in its own OS thread with a full
    fetch-decode-execute loop.  AMO instructions use real CPU atomics
    (AtomicU32/AtomicU64).  All harts run until a stop condition is hit
    (stop_flag / all-idle WFI / halted / trap loop / breakpoint / MMIO).
    No instruction quota — execution is continuous.

    *cfg.bp_addrs* is an optional list of PC addresses that trigger
    ``EXIT_BREAKPOINT`` when matched after instruction execution.
    """
    if _lib is None:
        return None

    mem = MemCtx()
    mem.ram = int(cfg.ram_ptr) if cfg.ram_ptr else 0
    mem.ram_size = cfg.ram_size
    mem.ram_base = cfg.ram_base
    mem.shadow_base = cfg.shadow_base
    mem.shadow_size = cfg.shadow_size

    clint_state = ClintMarshal(cfg.clint, cfg.num_harts)
    pmp_parts = _build_pmp_ffi(cfg.pmp)
    dev_ffi = _build_dev_ffi(cfg.dev)
    uart_parts = _build_uart_ffi(cfg.uart)
    virtio_parts = _build_virtio_ffi(cfg.virtio)
    plic_parts = _build_plic_ffi(cfg.plic)
    watchdog_ffi, has_watchdog = _build_watchdog_ffi(cfg.watchdog_timeout_ns)
    bp_parts = _build_bp_ffi(cfg.bp_addrs)

    hart_ctx = FfiHartCtx()
    hart_ctx.states = ctypes.addressof(cfg.states) if cfg.states is not None else 0
    hart_ctx.num_harts = cfg.num_harts
    hart_ctx.result = ctypes.addressof(cfg.result)
    hart_ctx.stop_flag = _addr_or_zero(cfg.stop_flag)
    hart_ctx.ext_irq = _addr_or_zero(cfg.ext_irq)

    mem_ctx = FfiMemCtx()
    mem_ctx.mem = ctypes.addressof(mem)
    mem_ctx.pmp = ctypes.addressof(pmp_parts.ctx)

    intr_ctx = FfiInterruptCtx()
    intr_ctx.clint = ctypes.addressof(clint_state.ctx)
    intr_ctx.plic = _addr_or_zero(plic_parts.ctx) if plic_parts.is_present else 0

    dev_ctx = FfiDevicesCtx()
    dev_ctx.dev = ctypes.addressof(dev_ffi)
    dev_ctx.uart = ctypes.addressof(uart_parts.ctx)
    dev_ctx.virtio = _addr_or_zero(virtio_parts.ctx) if virtio_parts.is_present else 0
    dev_ctx.watchdog = _addr_or_zero(watchdog_ffi) if has_watchdog else 0

    tlb_ctx = FfiTlbCtx()
    tlb_ctx.flush_cnt = _addr_or_zero(cfg.flush_cnt)
    tlb_ctx.flush_cnt_per_hart = _addr_or_zero(cfg.flush_cnt_per_hart)

    _lib.run_parallel(
        ctypes.byref(hart_ctx),
        ctypes.byref(mem_ctx),
        ctypes.byref(intr_ctx),
        ctypes.byref(dev_ctx),
        ctypes.byref(bp_parts.ctx),
        ctypes.byref(tlb_ctx),
    )

    clint_state.sync_back(cfg.clint)

    # Return the virtio FFI struct so the caller can read back changed fields.
    return virtio_parts.ctx if cfg.virtio else None


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
        ("stdin_fd", ctypes.c_int32),  # RawFd for stdin
        ("stdout_fd", ctypes.c_int32),  # RawFd for stdout
        ("rx_buf", ctypes.c_void_p),  # *mut u8 — shared RX ring buffer
        ("rx_cap", ctypes.c_uint32),  # capacity of rx_buf (power of two)
        ("rx_wr", ctypes.c_void_p),  # *mut AtomicU32 — write index (I/O thread)
        ("rx_rd", ctypes.c_void_p),  # *mut AtomicU32 — read index (Python, 容量控制)
        ("tx_buf", ctypes.c_void_p),  # *mut u8 — shared TX ring buffer
        ("tx_cap", ctypes.c_uint32),  # entry capacity (= byte capacity / 2)
        ("tx_wr", ctypes.c_void_p),  # *mut AtomicU32 — write index (hart threads)
        ("tx_drain", ctypes.c_void_p),  # *mut AtomicU32 — drain index (I/O thread 独占)
        ("stop_flag", ctypes.c_void_p),  # *mut AtomicU8 — stop request flag
        ("pause_flag", ctypes.c_void_p),  # *mut AtomicU8 — pause (debugger break)
        ("rx_notify", ctypes.c_void_p),  # *mut AtomicU8 — new RX data notification
        ("rx_notify_fd", ctypes.c_int32),  # fd: Rust termio 写 1B 唤醒 RX daemon
        ("rx_drain_fd", ctypes.c_int32),  # fd: Python 消费 ring 后写 1B 唤醒 termio 续传
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

    _termio_lib.terminal_io_set_emu_stop_flag.argtypes = [ctypes.c_void_p]
    _termio_lib.terminal_io_set_emu_stop_flag.restype = None

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
    if _termio_lib:
        _termio_lib.terminal_io_stop()


def termio_attach(handle: TermIoHandle) -> int:
    """Attach I/O thread to already-configured fds (no termios changes)."""
    if _termio_lib is None:
        return -2
    return _termio_lib.terminal_io_attach(ctypes.byref(handle))


def termio_is_running() -> bool:
    """Return True if the terminal I/O thread is currently running."""
    if _termio_lib:
        return _termio_lib.terminal_io_is_running() != 0
    return False


def termio_set_emu_stop_flag(ptr: int) -> None:
    """把 CPU 引擎停止标志地址注入 termio 线程, 供 Ctrl+Q 直连置位.

    *ptr* 为 ``ctypes.addressof(emu._native_stop_flag)``. termio 线程检测到
    Ctrl+Q 时直接写该标志, 绕过 SIGINT -> Python handler 往返 (主线程阻塞在
    run_parallel 时信号处理器不执行, 直连路径保证 Rust 引擎仍能即时停止).
    """
    if _termio_lib:
        _termio_lib.terminal_io_set_emu_stop_flag(ptr)
