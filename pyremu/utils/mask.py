# SPDX-LICENSE-IDENTIFIER: MIT
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

MASK04: int = 0xF
MASK08: int = (MASK04 << 4) | MASK04
MASK12: int = (MASK08 << 4) | MASK04
MASK16: int = (MASK12 << 4) | MASK04
MASK32: int = (MASK16 << 16) | MASK16
MASK64: int = (MASK32 << 32) | MASK32


def mask64(val: int) -> int:
    return val & MASK64


def mask32(val: int) -> int:
    return val & MASK32


def mask16(val: int) -> int:
    return val & MASK16


def mask12(val: int) -> int:
    return val & MASK12


def mask08(val: int) -> int:
    return val & MASK08


def mask04(val: int) -> int:
    return val & MASK04


def is_power_of_two_in(value: int, minimum: int, maximum: int) -> bool:
    """判定 *value* 是 2 的幂且落在 [minimum, maximum] 内."""
    return minimum <= value <= maximum and (value & (value - 1)) == 0


# ---------------------------------------------------------------
#  循环移位 — 移位量取以位宽为模的余数
# ---------------------------------------------------------------


def rol64(val: int, shamt: int) -> int:
    """64 位循环左移, 移位量为 shamt 以 64 为模的余数."""
    v = mask64(val)
    s = shamt & 0x3F
    return mask64((v << s) | (v >> (64 - s))) if s else v


def ror64(val: int, shamt: int) -> int:
    """64 位循环右移, 移位量为 shamt 以 64 为模的余数."""
    v = mask64(val)
    s = shamt & 0x3F
    return mask64((v >> s) | (v << (64 - s))) if s else v


def rol32(val: int, shamt: int) -> int:
    """32 位循环左移, 移位量为 shamt 以 32 为模的余数, 结果截断为 32 位."""
    v = mask32(val)
    s = shamt & 0x1F
    return mask32((v << s) | (v >> (32 - s))) if s else v


def ror32(val: int, shamt: int) -> int:
    """32 位循环右移, 移位量为 shamt 以 32 为模的余数, 结果截断为 32 位."""
    v = mask32(val)
    s = shamt & 0x1F
    return mask32((v >> s) | (v << (32 - s))) if s else v


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
