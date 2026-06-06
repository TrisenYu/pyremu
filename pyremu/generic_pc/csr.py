#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 16:59:34
# Last modified at 2026/06/06 星期六 23:13:21

"""
特权级寄存器
有点希望用数值指代会对应的csr寄存器对象，
从而能包含其名称、访问时发生的属性校验
"""
from pyremu.generic_pc.abs_reg import Reg


class CSR(Reg):
    privilege: bool = True

csr_bank: dict[int, CSR] = {
    # 000 status
    0x001: CSR(name="fflags"), # 浮点数异常标志寄存器
    0x002: CSR(name="frm"),    # 指定浮点动态舍入模式 float-rounding-mode
    0x003: CSR(name="fcsr"),   # 以上两者的组合
    # 004 uie
    # 005 utvec
    # utvt 007
    0x008: CSR(name="vstart"), # 保存向量指令开始或恢复执行的元素索引
    0x009: CSR(name="vxsat"),
    0x00a: CSR(name="vxrm"),
    0x00F: CSR(name="vcsr"),
    0x011: CSR(name="ssp"),   # supervisor shadow stack
    0x015: CSR(name="seed"),
    # jvt 用户级csr
    # 045 unxti
    # 046 uintstatus
    # 048 uscratchcsw
    # 049 uscratchcswl

    0x100: CSR(name="sstaus"),   # S模式状态寄存器
    # 0x102: CSR(name="sedeleg"),  # S模式异常派发寄存器
    # 0x103: CSR(name="sideleg"),  # S模式中断派发寄存器
    0x104: CSR(name="sie"),      # S模式中断暂停寄存器
    0x105: CSR(name="stvec"),    # S模式中断基地址寄存器
    0x106: CSR(name="scounteren"), # S模式计数器使能寄存器
    0x10A: CSR(name="senvcfg"), # 环境配置寄存器

    0x114: CSR(name="sieh"), # aia为rv32提供的s模式中断使能高位CSR，用于访问SIE的高32位
    0x120: CSR(name="scountinhibit"),

    0x140: CSR(name="sscratch"),
    0x141: CSR(name="sepc"),
    0x142: CSR(name="scause"),
    0x143: CSR(name="stval"),
    0x144: CSR(name="sip"),
    0x14D: CSR(name="stimecmp"),
    0x14E: CSR(name="sctrctl"), # aia相关
    0x14F: CSR(name="sctrstatus"), # aia相关
    0x150: CSR(name="siselect"), # aia相关
    0x151: CSR(name="sireg"), # aia相关
    0x152: CSR(name="sireg2"), # aia相关
    0x153: CSR(name="sireg3"), # aia相关
    0x154: CSR(name="siph"), # aia相关
    0x155: CSR(name="sireg4"), # aia相关
    0x156: CSR(name="sireg5"), # aia相关
    0x157: CSR(name="sireg6"), # aia相关
    0x15C: CSR(name="stopei"),
    0x15D: CSR(name="stimecmph"),
    0x15F: CSR(name="sctrdepth"),

    0x180: CSR(name="satp"),
    0x181: CSR(name="srmcfg"),
    0x183: CSR(name="spmpen"),
    0x193: CSR(name="spmpenh"),

    0x200: CSR(name="vsstatus"),
    0x204: CSR(name="vsie"),
    0x205: CSR(name="vstvec"),
    0x214: CSR(name="vsieh"),
    0x240: CSR(name="vsscratch"),
    0x241: CSR(name="vsepc"),
    0x242: CSR(name="vscause"),
    0x243: CSR(name="vstval"),
    0x244: CSR(name="vsip"),
    0x24D: CSR(name="vstimecmp"),
    0x24E: CSR(name="vsctrctl"),
    0x250: CSR(name="vsiselect"), # 虚拟机&aia相关
    0x251: CSR(name="vsireg"), # 虚拟机&aia相关
    0x252: CSR(name="vsireg2"), # 虚拟机&aia相关
    0x253: CSR(name="vsireg3"), # 虚拟机&aia相关
    0x254: CSR(name="vsiph"), # 虚拟机&aia相关
    0x255: CSR(name="vsireg4"), # 虚拟机&aia相关
    0x256: CSR(name="vsireg5"), # 虚拟机&aia相关
    0x257: CSR(name="vsireg6"), # 虚拟机&aia相关
    0x25C: CSR(name="vstopei"), # 虚拟机&只有实现了imsic时才有这个
    0x25D: CSR(name="vstimecmph"),
    0x280: CSR(name="vsatp"),

    0x301: CSR(name="misa"), # 报告当前hart支持的ISA扩展
    0x302: CSR(name="medeleg"),
    0x303: CSR(name="mideleg"),
    0x304: CSR(name="mie"),
    0x306: CSR(name="mcounteren"),
    0x308: CSR(name="mvien"), # aia中断使能CSR
    0x309: CSR(name="mvip"), # aia中断挂起CSR
    0x30A: CSR(name="menvcfg"),

    0x312: CSR(name="medelegh"),
    0x313: CSR(name="midelegh"),
    0x314: CSR(name="mieh"),
    0x316: CSR(name="mpmpdeleg"),
    0x318: CSR(name="mvienh"), # aia中断使能CSR
    0x319: CSR(name="mviph"), # aia中断挂起CSR
    0x31A: CSR(name="menvcfgh"),

    0x320: CSR(name="mcountinhibit"),
    0x321: CSR(name="mcyclecfg"),
    0x322: CSR(name="minstretcfg"),

    0x340: CSR(name="mscratch"),
    0x341: CSR(name="mepc"),
    0x342: CSR(name="mcause"),
    0x343: CSR(name="mtval"),
    0x344: CSR(name="mip"),
    # 345: mnxti, 未标准化
    # 346: mintstatus, 未标准化
    # 348: mscratchcsw, 未标准化
    # 349: mscratchcswl, 未标准化
    0x34A: CSR(name="mtinst"),
    0x34B: CSR(name="mtval2"),
    0x34E: CSR(name="mctrctl"),
    0x34F: CSR(name="mctrstatus"), # aia相关
    0x350: CSR(name="miselect"), # aia相关
    0x351: CSR(name="mireg"), # aia相关
    0x352: CSR(name="mireg2"), # aia相关
    0x353: CSR(name="mireg3"), # aia相关
    0x354: CSR(name="miph"), # aia相关
    0x355: CSR(name="mireg4"), # aia相关
    0x356: CSR(name="mireg5"), # aia相关
    0x357: CSR(name="mireg6"), # aia相关
    0x35C: CSR(name="mtopei"), # 只有实现了imsic时才有这个

    0x5A8: CSR(name="scontext"),

    0x600: CSR(name="hstatus"),
    0x602: CSR(name="hedeleg"),
    0x603: CSR(name="hideleg"),
    0x604: CSR(name="hie"),
    0x605: CSR(name="htimedelta"),
    0x606: CSR(name="hcounteren"),
    0x607: CSR(name="hgeie"),
    0x608: CSR(name="hvien"),
    0x609: CSR(name="hvictl"),
    0x60a: CSR(name="henvcfg"),
    0x612: CSR(name="hedelegh"),
    0x613: CSR(name="hidelegh"),
    0x615: CSR(name="htimedeltah"),
    0x618: CSR(name="hvienh"),
    0x61a: CSR(name="henvcfgh"),

    0x643: CSR(name="htval"),
    0x644: CSR(name="hip"),
    0x645: CSR(name="hvip"),
    0x646: CSR(name="hviprio1"),
    0x647: CSR(name="hviprio2"),
    0x64a: CSR(name="htinst"),
    0x64D: CSR(name="htimecmp"),
    0x64E: CSR(name="hctrctl"),

    0x655: CSR(name="hviph"), # 虚拟机&aia相关
    0x656: CSR(name="hviprio1h"), # 虚拟机&aia相关
    0x657: CSR(name="hviprio2h"), # 虚拟机&aia相关

    0x680: CSR(name="hgatp"),
    0x6A8: CSR(name="hcontext"),

    0x721: CSR(name="mcyclecfgh"),
    0x722: CSR(name="minstretcfgh"),

    # 未获批定义
    # 0x740: CSR(name="mnscratch"), # RNMI暂存寄存器，一般存放上下文地址
    # 0x741: CSR(name="mnepc"),
    # 0x742: CSR(name="mncause"),
    # 0x744: CSR(name="mnstatus"),

    0x7A0: CSR(name="tselect"),
    0x7A1: CSR(name="tdata1"),
    0x7A2: CSR(name="tdata2"),
    0x7A3: CSR(name="tdata3"),
    0x7A4: CSR(name="tinfo"),
    0x7A5: CSR(name="tcontrol"),

    0x7b0: CSR(name="dcsr"),
    0x7b1: CSR(name="dpc"),
    0x7b2: CSR(name="dscratch0"),
    0x7b3: CSR(name="dscratch1"),

    0xB00: CSR(name="mcycle"),
    0xB02: CSR(name="minstret"),

    0xB80: CSR(name="mcycleh"),
    0xB81: CSR(name="mtimeh"),
    0xB82: CSR(name="minstreth"), # 记录hart已经完成的指令数

    0xC00: CSR(name="cycle"), # 用户级的
    0xC01: CSR(name="time"),
    0xC02: CSR(name="instret"),

    0xDA0: CSR(name="scountovf"), # 计数器溢出状态

    0xE12: CSR(name="hgeip"),
    0xEB0: CSR(name="vstopi"),

    0xF11: CSR(name="mvendorid"),
    0xF12: CSR(name="marchid"),
    0xF13: CSR(name="mimpid"),
    0xF14: CSR(name="mhartid"),   # 当前正在执行指令的hart ID
    0xF15: CSR(name="mconfigptr"),
    0xFB0: CSR(name="mtopi"),
}

for i in range(0, 4):
    csr_bank[0x10C+i] = CSR(name=f"sstateen{i}")
    csr_bank[0x31C+i] = CSR(name=f"mstateen{i}h")
    csr_bank[0x60C+i] = CSR(name=f"hstateen{i}")
    csr_bank[0x61C+i] = CSR(name=f"hstateen{i}h")

for i in range(0, 16):
    csr_bank[0x3A0+i] = CSR(name=f"pmpcfg{i}")

for i in range(0, 64):
    csr_bank[0x3B0+i] = CSR(name=f"pmpaddr{i}")

for i in range(3, 32):
    csr_bank[0x320+i] = CSR(name=f"mhpmevent{i}")
    csr_bank[0x720+i] = CSR(name=f"mhpmevent{i}h")
    csr_bank[0xB00+i] = CSR(name=f"mhpmcounter{i}")
    csr_bank[0xB80+i] = CSR(name=f"mhpmcounter{i}h")
    csr_bank[0xC00+i] = CSR(name=f"hpmcounter{i}")
    csr_bank[0xC80+i] = CSR(name=f"hpmcounter{i}h")

if __name__ == "__main__":
    print(len(csr_bank))
