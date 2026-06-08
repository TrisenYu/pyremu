#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
RISC-V 寄存器定义 — 合并 abs_reg + gpr_fpr + csr.

包含:
- Reg / FPR: 通用寄存器基类
- GPR / FPR 预构建列表及工厂函数 (register_gpr, register_fpr)
- CSR / CsrAccess: 控制和状态寄存器
- CSR 地址库 (_csr_bank) 及查询/注册函数 (check_csr, register_csr)
"""

from copy import deepcopy
from enum import Enum
from functools import partial

from pydantic import BaseModel

# ============================================================
#  通用寄存器基类
# ============================================================


class Reg(BaseModel):
    """通用整数寄存器."""
    name: str
    alias: str = ""
    restricts: list = []
    val: int = 0


class FPR(BaseModel):
    """浮点寄存器."""
    name: str
    alias: str = ""
    restricts: list = []
    val: float = 0.0


# ============================================================
#  GPR / FPR 预构建列表 & 工厂函数
# ============================================================

_rst_reg = partial(Reg, val=0)
_gpr = [
    _rst_reg(name="x0", alias="zero"),
    _rst_reg(name="x1", alias="ra"),
    _rst_reg(name="x2", alias="sp"),
    _rst_reg(name="x3", alias="gp"),
    _rst_reg(name="x4", alias="tp"),
    _rst_reg(name="x5", alias="t0"),
    _rst_reg(name="x6", alias="t1"),
    _rst_reg(name="x7", alias="t2"),
    _rst_reg(name="x8", alias="s0"),
    _rst_reg(name="x9", alias="s1"),
    _rst_reg(name="x10", alias="a0"),
    _rst_reg(name="x11", alias="a1"),
    _rst_reg(name="x12", alias="a2"),
    _rst_reg(name="x13", alias="a3"),
    _rst_reg(name="x14", alias="a4"),
    _rst_reg(name="x15", alias="a5"),
    _rst_reg(name="x16", alias="a6"),
    _rst_reg(name="x17", alias="a7"),
    _rst_reg(name="x18", alias="s2"),
    _rst_reg(name="x19", alias="s3"),
    _rst_reg(name="x20", alias="s4"),
    _rst_reg(name="x21", alias="s5"),
    _rst_reg(name="x22", alias="s6"),
    _rst_reg(name="x23", alias="s7"),
    _rst_reg(name="x24", alias="s8"),
    _rst_reg(name="x25", alias="s9"),
    _rst_reg(name="x26", alias="s10"),
    _rst_reg(name="x27", alias="s11"),
    _rst_reg(name="x28", alias="t3"),
    _rst_reg(name="x29", alias="t4"),
    _rst_reg(name="x30", alias="t5"),
    _rst_reg(name="x31", alias="t6"),
]

_rst_fpr = partial(FPR, val=0.0)
_fpr = [
    _rst_fpr(name="f0", alias="ft0"),
    _rst_fpr(name="f1", alias="ft1"),
    _rst_fpr(name="f2", alias="ft2"),
    _rst_fpr(name="f3", alias="ft3"),
    _rst_fpr(name="f4", alias="ft4"),
    _rst_fpr(name="f5", alias="ft5"),
    _rst_fpr(name="f6", alias="ft6"),
    _rst_fpr(name="f7", alias="ft7"),
    _rst_fpr(name="f8", alias="fs0"),
    _rst_fpr(name="f9", alias="fs1"),
    _rst_fpr(name="f10", alias="fa0"),
    _rst_fpr(name="f11", alias="fa1"),
    _rst_fpr(name="f12", alias="fa2"),
    _rst_fpr(name="f13", alias="fa3"),
    _rst_fpr(name="f14", alias="fa4"),
    _rst_fpr(name="f15", alias="fa5"),
    _rst_fpr(name="f16", alias="fa6"),
    _rst_fpr(name="f17", alias="fa7"),
    _rst_fpr(name="f18", alias="fs2"),
    _rst_fpr(name="f19", alias="fs3"),
    _rst_fpr(name="f20", alias="fs4"),
    _rst_fpr(name="f21", alias="fs5"),
    _rst_fpr(name="f22", alias="fs6"),
    _rst_fpr(name="f23", alias="fs7"),
    _rst_fpr(name="f24", alias="fs8"),
    _rst_fpr(name="f25", alias="fs9"),
    _rst_fpr(name="f26", alias="fs10"),
    _rst_fpr(name="f27", alias="fs11"),
    _rst_fpr(name="f28", alias="ft8"),
    _rst_fpr(name="f29", alias="ft9"),
    _rst_fpr(name="f30", alias="ft10"),
    _rst_fpr(name="f31", alias="ft11"),
]


def register_gpr():
    """返回 32 个 GPR 的独立副本."""
    return deepcopy(_gpr)


def register_fpr():
    """返回 32 个 FPR 的独立副本."""
    return deepcopy(_fpr)


# ============================================================
#  CSR 访问权限 & 特权级控制
# ============================================================


class CsrAccess(Enum):
    """CSR 按特权级的读写权限编码.

    高 4 位: 特权级 (U=1, S=2, H=4, M=8, D=16 组合)
    低 2 位: 10=只读, 11=读写
    """

    u_ro = 0b0000_10
    u_rw = 0b0000_11
    s_ro = 0b0001_10
    s_rw = 0b0001_11
    h_ro = 0b0010_10
    h_rw = 0b0010_11
    m_ro = 0b0100_10
    m_rw = 0b0100_11
    d_ro = 0b1000_10
    d_rw = 0b1000_11
    ds_ro = 0b1001_10
    ds_rw = 0b1001_11
    dm_ro = 0b1100_10
    dm_rw = 0b1100_11


_csr_access_mask = 0b1111_11


class CSR(Reg):
    """控制和状态寄存器."""
    access: CsrAccess
    xlen: int = 64

    def strip_w(self) -> "CSR":
        """设为只读 (低 2 位 = 10)."""
        val = self.access.value
        val &= _csr_access_mask
        self.access = CsrAccess(val)
        return self

    def strip_mmode(self) -> "CSR":
        """移除 M 模式权限位."""
        val = self.access.value
        val &= 0b1000_11
        self.access = CsrAccess(val)
        return self

    def add_dmode(self) -> "CSR":
        """添加 D 模式权限位."""
        val = self.access.value
        val |= 0b1000_00
        self.access = CsrAccess(val)
        return self


# ============================================================
#  CSR 构建器 & 完整地址库
# ============================================================

_UmodeCSR = partial(CSR, access=CsrAccess.u_rw)
_SmodeCSR = partial(CSR, access=CsrAccess.s_rw)
_HmodeCSR = partial(CSR, access=CsrAccess.h_rw)
_MmodeCSR = partial(CSR, access=CsrAccess.m_rw)
_DmodeCSR = partial(CSR, access=CsrAccess.d_rw)
_DMmodeCSR = partial(CSR, access=CsrAccess.dm_rw)

_csr_bank: dict[int, CSR] = {
    # 000 status
    0x001: _UmodeCSR(name="fflags", xlen=32).strip_w(),
    0x002: _UmodeCSR(name="frm", xlen=32),
    0x003: _UmodeCSR(name="fcsr", xlen=32),
    # 向量扩展 CSR
    0x008: _UmodeCSR(name="vstart").strip_w(),
    0x009: _UmodeCSR(name="vxsat").strip_w(),
    0x00a: _UmodeCSR(name="vxrm").strip_w(),
    0x00F: _UmodeCSR(name="vcsr").strip_w(),
    # Supervisor 模式 CSR
    0x100: _SmodeCSR(name="sstaus"),
    0x104: _SmodeCSR(name="sie"),
    0x105: _SmodeCSR(name="stvec"),
    0x106: _SmodeCSR(name="scounteren", xlen=32),
    0x10A: _SmodeCSR(name="senvcfg"),
    0x114: _SmodeCSR(name="sieh", xlen=32),
    0x120: _SmodeCSR(name="scountinhibit"),
    0x140: _SmodeCSR(name="sscratch"),
    0x141: _SmodeCSR(name="sepc"),
    0x142: _SmodeCSR(name="scause"),
    0x143: _SmodeCSR(name="stval"),
    0x144: _SmodeCSR(name="sip"),
    0x14D: _SmodeCSR(name="stimecmp", xlen=64),
    0x14E: _SmodeCSR(name="sctrctl"),
    0x14F: _SmodeCSR(name="sctrstatus", xlen=32),
    0x150: _SmodeCSR(name="siselect"),
    0x151: _SmodeCSR(name="sireg"),
    0x152: _SmodeCSR(name="sireg2"),
    0x153: _SmodeCSR(name="sireg3"),
    0x154: _SmodeCSR(name="siph", xlen=32),
    0x155: _SmodeCSR(name="sireg4"),
    0x156: _SmodeCSR(name="sireg5"),
    0x157: _SmodeCSR(name="sireg6"),
    0x15C: _SmodeCSR(name="stopei"),
    0x15D: _SmodeCSR(name="stimecmph", xlen=32),
    0x15F: _SmodeCSR(name="sctrdepth", xlen=32),
    0x180: _SmodeCSR(name="satp"),
    0x181: _SmodeCSR(name="srmcfg"),
    0x183: _SmodeCSR(name="spmpen"),
    0x193: _SmodeCSR(name="spmpenh", xlen=32),
    # Hypervisor 模式 CSR
    0x200: _HmodeCSR(name="vsstatus"),
    0x204: _HmodeCSR(name="vsie"),
    0x205: _HmodeCSR(name="vstvec"),
    0x214: _HmodeCSR(name="vsieh", xlen=32),
    0x240: _HmodeCSR(name="vsscratch", xlen=32),
    0x241: _HmodeCSR(name="vsepc"),
    0x242: _HmodeCSR(name="vscause"),
    0x243: _HmodeCSR(name="vstval"),
    0x244: _HmodeCSR(name="vsip", xlen=64),
    0x24D: _HmodeCSR(name="vstimecmp", xlen=64),
    0x24E: _HmodeCSR(name="vsctrctl"),
    0x250: _HmodeCSR(name="vsiselect"),
    0x251: _HmodeCSR(name="vsireg"),
    0x252: _HmodeCSR(name="vsireg2"),
    0x253: _HmodeCSR(name="vsireg3"),
    0x254: _HmodeCSR(name="vsiph", xlen=32),
    0x255: _HmodeCSR(name="vsireg4"),
    0x256: _HmodeCSR(name="vsireg5"),
    0x257: _HmodeCSR(name="vsireg6"),
    0x25C: _HmodeCSR(name="vstopei"),
    0x25D: _HmodeCSR(name="vstimecmph"),
    0x280: _HmodeCSR(name="vsatp"),
    # Machine 模式 CSR
    0x300: _MmodeCSR(name="mstatus"),
    0x301: _MmodeCSR(name="misa"),
    0x302: _MmodeCSR(name="medeleg"),
    0x303: _MmodeCSR(name="mideleg"),
    0x304: _MmodeCSR(name="mie"),
    0x305: _MmodeCSR(name="mtvec"),
    0x306: _MmodeCSR(name="mcounteren", xlen=32),
    0x308: _MmodeCSR(name="mvien", xlen=64),
    0x309: _MmodeCSR(name="mvip", xlen=64),
    0x30A: _MmodeCSR(name="menvcfg"),
    0x310: _MmodeCSR(name="mstatush", xlen=32),
    0x312: _MmodeCSR(name="medelegh", xlen=32),
    0x313: _MmodeCSR(name="midelegh", xlen=32),
    0x314: _MmodeCSR(name="mieh", xlen=32),
    0x316: _MmodeCSR(name="mpmpdeleg"),
    0x318: _MmodeCSR(name="mvienh", xlen=32),
    0x319: _MmodeCSR(name="mviph", xlen=32),
    0x31A: _MmodeCSR(name="menvcfgh", xlen=32),
    0x320: _MmodeCSR(name="mcountinhibit", xlen=32),
    0x321: _MmodeCSR(name="mcyclecfg", xlen=64),
    0x322: _MmodeCSR(name="minstretcfg", xlen=64),
    0x340: _MmodeCSR(name="mscratch"),
    0x341: _MmodeCSR(name="mepc"),
    0x342: _MmodeCSR(name="mcause"),
    0x343: _MmodeCSR(name="mtval"),
    0x344: _MmodeCSR(name="mip"),
    0x34A: _MmodeCSR(name="mtinst"),
    0x34B: _MmodeCSR(name="mtval2"),
    0x34E: _MmodeCSR(name="mctrctl"),
    0x350: _MmodeCSR(name="miselect"),
    0x351: _MmodeCSR(name="mireg"),
    0x352: _MmodeCSR(name="mireg2"),
    0x353: _MmodeCSR(name="mireg3"),
    0x354: _MmodeCSR(name="miph", xlen=32),
    0x355: _MmodeCSR(name="mireg4"),
    0x356: _MmodeCSR(name="mireg5"),
    0x357: _MmodeCSR(name="mireg6"),
    0x35C: _MmodeCSR(name="mtopei"),
    0x5A8: _SmodeCSR(name="scontext").add_dmode(),
    0x600: _HmodeCSR(name="hstatus"),
    0x602: _HmodeCSR(name="hedeleg"),
    0x603: _HmodeCSR(name="hideleg"),
    0x604: _HmodeCSR(name="hie"),
    0x605: _HmodeCSR(name="htimedelta"),
    0x606: _HmodeCSR(name="hcounteren", xlen=32),
    0x607: _HmodeCSR(name="hgeie"),
    0x608: _HmodeCSR(name="hvien"),
    0x609: _HmodeCSR(name="hvictl"),
    0x60a: _HmodeCSR(name="henvcfg"),
    0x612: _HmodeCSR(name="hedelegh", xlen=32),
    0x613: _HmodeCSR(name="hidelegh", xlen=32),
    0x615: _HmodeCSR(name="htimedeltah", xlen=32),
    0x618: _HmodeCSR(name="hvienh", xlen=32),
    0x61a: _HmodeCSR(name="henvcfgh", xlen=32),
    0x643: _HmodeCSR(name="htval"),
    0x644: _HmodeCSR(name="hip"),
    0x645: _HmodeCSR(name="hvip", xlen=64),
    0x646: _HmodeCSR(name="hviprio1", xlen=64),
    0x647: _HmodeCSR(name="hviprio2", xlen=64),
    0x64a: _HmodeCSR(name="htinst"),
    0x64D: _HmodeCSR(name="htimecmp"),
    0x64E: _HmodeCSR(name="hctrctl"),
    0x655: _HmodeCSR(name="hviph", xlen=32),
    0x656: _HmodeCSR(name="hviprio1h", xlen=32),
    0x657: _HmodeCSR(name="hviprio2h", xlen=32),
    0x680: _HmodeCSR(name="hgatp"),
    0x6A8: _HmodeCSR(name="hcontext"),
    0x721: _MmodeCSR(name="mcyclecfgh", xlen=32),
    0x722: _MmodeCSR(name="minstretcfgh", xlen=32),
    # 调试 / 触发 CSR
    0x7A0: _DMmodeCSR(name="tselect"),
    0x7A1: _DMmodeCSR(name="tdata1"),
    0x7A2: _DMmodeCSR(name="tdata2"),
    0x7A3: _DMmodeCSR(name="tdata3"),
    0x7A4: _DMmodeCSR(name="tinfo").strip_w(),
    0x7A5: _DMmodeCSR(name="tcontrol"),
    0x7A8: _MmodeCSR(name="mcontext"),
    0x7b0: _DmodeCSR(name="dcsr"),
    0x7b1: _DmodeCSR(name="dpc"),
    0x7b2: _DmodeCSR(name="dscratch0"),
    0x7b3: _DmodeCSR(name="dscratch1"),
    # 计数器
    0xB00: _MmodeCSR(name="mcycle", xlen=64),
    0xB02: _MmodeCSR(name="minstret", xlen=64),
    0xB80: _MmodeCSR(name="mcycleh", xlen=32),
    0xB81: _MmodeCSR(name="mtimeh", xlen=32),
    0xB82: _MmodeCSR(name="minstreth", xlen=32),
    # 用户级计数器 (只读)
    0xC00: _UmodeCSR(name="cycle").strip_w(),
    0xC01: _UmodeCSR(name="time").strip_w(),
    0xC02: _UmodeCSR(name="instret").strip_w(),
    0xC20: _UmodeCSR(name="vl").strip_w(),
    0xC21: _UmodeCSR(name="vtype").strip_w(),
    0xC22: _UmodeCSR(name="vlenb").strip_w(),
    0xC80: _UmodeCSR(name="cycleh").strip_w(),
    0xC81: _UmodeCSR(name="timeh").strip_w(),
    0xC82: _UmodeCSR(name="instreth").strip_w(),
    0xDA0: _SmodeCSR(name="scountovf").strip_w(),
    0xE12: _HmodeCSR(name="hgeip").strip_w(),
    0xEB0: _HmodeCSR(name="vstopi").strip_w(),
    # 机器信息 (只读)
    0xF11: _MmodeCSR(name="mvendorid", xlen=32).strip_w(),
    0xF12: _MmodeCSR(name="marchid").strip_w(),
    0xF13: _MmodeCSR(name="mimpid").strip_w(),
    0xF14: _MmodeCSR(name="mhartid").strip_w(),
    0xF15: _MmodeCSR(name="mconfigptr").strip_w(),
    0xFB0: _MmodeCSR(name="mtopi").strip_w(),
}

# 批量生成编号连续的 CSR
for i in range(0, 4):
    _csr_bank[0x10C + i] = _SmodeCSR(name=f"sstateen{i}")
    _csr_bank[0x30C + i] = _MmodeCSR(name=f"mstateen{i}")
    _csr_bank[0x31C + i] = _MmodeCSR(name=f"mstateen{i}h", xlen=32)
    _csr_bank[0x60C + i] = _HmodeCSR(name=f"hstateen{i}")
    _csr_bank[0x61C + i] = _HmodeCSR(name=f"hstateen{i}h", xlen=32)

for i in range(0, 16):
    _csr_bank[0x3A0 + i] = _MmodeCSR(name=f"pmpcfg{i}")

for i in range(0, 64):
    _csr_bank[0x3B0 + i] = _MmodeCSR(name=f"pmpaddr{i}")

for i in range(3, 32):
    _csr_bank[0x320 + i] = _MmodeCSR(name=f"mhpmevent{i}", xlen=64)
    _csr_bank[0x720 + i] = _MmodeCSR(name=f"mhpmevent{i}h", xlen=32)
    _csr_bank[0xB00 + i] = _MmodeCSR(name=f"mhpmcounter{i}", xlen=64)
    _csr_bank[0xB80 + i] = _MmodeCSR(name=f"mhpmcounter{i}h", xlen=32)
    _csr_bank[0xC00 + i] = _UmodeCSR(name=f"hpmcounter{i}").strip_w()
    _csr_bank[0xC80 + i] = _UmodeCSR(name=f"hpmcounter{i}h", xlen=32).strip_w()


def check_csr(csr_id: int) -> tuple[bool, str]:
    """检查 CSR 地址是否有效, 返回 (valid, name)."""
    csr_id &= 0xFFF
    if csr_id not in _csr_bank:
        return False, ""
    return True, _csr_bank[csr_id].name


def register_csr():
    """返回所有 CSR 的独立副本 (name → CSR 对象)."""
    ret = {v.name: v for _, v in _csr_bank.items()}
    return deepcopy(ret)
