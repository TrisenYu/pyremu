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

MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
MASK32: int = 0xFFFF_FFFF


def mask64(val: int) -> int:
    """将 *val* 截断为 64-bit 无符号整数 ``[0, 2^64)``."""
    return val & MASK64


def mask32(val: int) -> int:
    """将 *val* 截断为 32-bit 无符号整数 ``[0, 2^32)``."""
    return val & MASK32
