# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
"""位宽掩码 — 将任意精度 Python int 截断为硬件位宽.

Python 的任意精度整数在右移等运算中可能产生超大值,
与 64-bit / 32-bit 硬件语义不一致。
本模块提供 ``mask64`` / ``mask32`` 作为统一截断点,
替代散落各处的 ``& 0xFFFF_FFFF_FFFF_FFFF`` / ``& 0xFFFF_FFFF``.

Usage::

    from pyremu.utils.mask import mask64, mask32

    result = mask64(v1 + v2)   # 等价于 (v1 + v2) & 0xFFFF_FFFF_FFFF_FFFF
    lo = mask32(value)         # 等价于 value & 0xFFFF_FFFF
"""

MASK08: int = 0xFF
MASK16: int = (MASK08 << 8) | MASK08
MASK32: int = (MASK16 << 16) | MASK16
MASK64: int = (MASK32 << 32) | MASK32


def mask64(val: int) -> int:
    """将 *val* 截断为 64-bit 无符号整数 ``[0, 2^64)``."""
    return val & MASK64

def mask32(val: int) -> int:
    """将 *val* 截断为 32-bit 无符号整数 ``[0, 2^32)``."""
    return val & MASK32

def mask16(val: int) -> int:
    """将 *val* 截断为 16-bit 无符号整数 ``[0, 2^16)``."""
    return val & MASK16

def mask08(val: int) -> int:
    return val & MASK08

# ---------------------------------------------------------------
#  Sign-extend helpers (moved from disassem.py to break circular import)
# ---------------------------------------------------------------


def sext(val: int, bits: int) -> int:
    """Sign-extend *val* from *bits* width to a canonical 64-bit unsigned Python int.

    Python's arbitrary-precision integers behave differently from finite-width
    hardware in bitwise operations (|, &, ^, <<, >>) when values are negative.
    Canonicalizing to [0, 2^64) ensures consistent behaviour regardless of
    whether the value was built via sign-extend, zero-extend, or arithmetic.
    """
    sign_bit = 1 << (bits - 1)
    result = (val & (sign_bit - 1)) - (val & sign_bit)
    if bits <= 64:
        result = mask64(result)
    return result


def sext8(val: int) -> int:
    """Sign-extend from 8 bits -> canonical 64-bit unsigned."""
    return mask64((val & 0x7F) - (val & 0x80))


def sext12(val: int) -> int:
    """Sign-extend from 12 bits -> canonical 64-bit unsigned."""
    return mask64((val & 0x7FF) - (val & 0x800))


def sext16(val: int) -> int:
    """Sign-extend from 16 bits -> canonical 64-bit unsigned."""
    return mask64((val & 0x7FFF) - (val & 0x8000))


def sext32(val: int) -> int:
    """Sign-extend from 32 bits -> canonical 64-bit unsigned."""
    return mask64((val & 0x7FFF_FFFF) - (val & 0x8000_0000))
