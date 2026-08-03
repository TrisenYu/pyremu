# -*- coding: utf-8 -*-
"""GPR / CSR 名称查找 — 独立于 core.registers 以避免循环导入.

由 ``core.registers`` 的工厂函数在模块初始化后调用 ``_init_gpr_map`` /
``_init_csr_map`` 注入名称映射。 ``utils.disassem`` 仅依赖本模块,
不依赖 ``core.*``, 从而断开 core.decoder ↔ utils.disassem 循环导入链。
"""

from __future__ import annotations

# 默认占位: 模块加载时 register 数据尚未注入, 回退到裸编号.
_gpr_names: list[str] = []
_gpr_aliases: list[str] = []
_csr_names: dict[int, str] = {}


def init_gpr_map(names: list[str], aliases: list[str]) -> None:
    """由 ``core.registers`` 调用以注入 GPR 名称/别名映射."""
    global _gpr_names, _gpr_aliases
    _gpr_names = names
    _gpr_aliases = aliases


def init_csr_map(csr_names: dict[int, str]) -> None:
    """由 ``core.registers`` 调用以注入 CSR 名称映射."""
    global _csr_names
    _csr_names = csr_names


def gpr_name(idx: int) -> str:
    """返回第 idx 号 GPR 的名称 (例: x0, x10)."""
    if 0 <= idx < len(_gpr_names):
        return _gpr_names[idx]
    return f"?x{idx}"


def gpr_alias(idx: int) -> str:
    """返回第 idx 号 GPR 的 ABI 别名 (例: zero, a0, sp)."""
    if 0 <= idx < len(_gpr_aliases):
        return _gpr_aliases[idx]
    return ""


def check_csr(csr_id: int) -> tuple[bool, str]:
    """检查 CSR 地址是否有效, 返回 (valid, name)."""
    csr_id &= 0xFFF
    name = _csr_names.get(csr_id, "")
    return (name != "", name)
