#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
Hart (硬件线程) 的寄存器文件定义，包含:
- 通用整数寄存器 (GPR x0–x31)
- 浮点寄存器 (FPR f0–f31)
- 控制和状态寄存器 (CSR)
- 特权级模式 (RiscvMode)
- mstatus 等关键 CSR 的位字段定义及快捷属性
"""

from enum import Enum

from pyremu.core.registers import check_csr, register_csr, register_fpr, register_gpr
from pyremu.core.trap import TrapType
from pyremu.memory.cache import TLB_SIZE
from pyremu.memory.tlb import TLB


class RiscvMode(Enum):
    """RISC-V 特权级模式。

    使用位编码 (1 << N) 以便将来做权限掩码比较:
        U=0 (用户), S=1 (监管), H=2 ( hypervisor ),
        M=4 (机器), D=8 (调试).
    """

    U = 0
    S = 1
    H = 2
    M = 4
    D = 8


# ============================================================
#  mstatus / sstatus 位字段定义 (RV64)
# ============================================================
# 这些常量用于按位读写 mstatus CSR 中的关键控制位。

MSTATUS_SIE = 1 << 1  # Supervisor 中断使能
MSTATUS_MIE = 1 << 3  # Machine 中断使能
MSTATUS_SPIE = 1 << 5  # Supervisor 先前中断使能 (进入 trap 前)
MSTATUS_UBE = 1 << 6  # 用户模式大端 (通常为 0)
MSTATUS_MPIE = 1 << 7  # Machine 先前中断使能 (进入 trap 前)
MSTATUS_SPP = 1 << 8  # Supervisor 先前特权级 (0=U, 1=S)
MSTATUS_VS = 0b11 << 9  # 虚拟化状态 (H 扩展)
MSTATUS_MPP = 0b11 << 11  # Machine 先前特权级 (0=U, 1=S, 3=M)
MSTATUS_FS = 0b11 << 13  # 浮点单元状态
MSTATUS_XS = 0b11 << 15  # 用户扩展状态
MSTATUS_MPRV = 1 << 17  # 修改特权级下的内存访问特权
MSTATUS_SUM = 1 << 18  # 允许 S 模式访问 U 模式页
MSTATUS_MXR = 1 << 19  # 使能可执行内存的可读
MSTATUS_TVM = 1 << 20  # 陷入 S 模式访问 satp
MSTATUS_TW = 1 << 21  # 陷入 WFI
MSTATUS_TSR = 1 << 22  # 陷入 SRET
MSTATUS_SD = 1 << 63  # 状态脏位 (FS 或 XS 为脏时置 1)

# SPP / MPP 编码 → RiscvMode 的映射
_SPP_TO_MODE = {0: RiscvMode.U, 1: RiscvMode.S}
_MPP_TO_MODE = {0: RiscvMode.U, 1: RiscvMode.S, 3: RiscvMode.M}
_MODE_TO_MPP = {RiscvMode.U: 0, RiscvMode.S: 1, RiscvMode.M: 3}
_MODE_TO_SPP = {RiscvMode.U: 0, RiscvMode.S: 1}


class HartWithRegs:
    """硬件线程的寄存器文件。

    每个 hart 拥有独立的 GPR、FPR、CSR 以及指令/数据 TLB。
    本类提供寄存器读写方法以及 mstatus/mtvec 等关键 CSR 的
    快捷属性访问。
    """

    def __init__(self, id: int):
        self.id = id
        self.gprs = register_gpr()
        self.fprs = register_fpr()
        self.csrs = register_csr()
        self.csrs["mhartid"].val = id
        self.pc = 0
        self.mode = RiscvMode.M

        # TLB: 指令 TLB 和数据 TLB 分开
        # MMIO 地址禁止 CPU 缓存 (读 MMIO 可能改变硬件状态)
        self.itlb = TLB(size=TLB_SIZE)
        self.dtlb = TLB(size=TLB_SIZE)

        # 物理内存后端 — 由子类或外部注入
        # 调用约定: mem_read_phy(addr: int, size: int) -> bytes
        #           mem_write_phy(addr: int, data: bytes) -> None
        self._mem_read_phy = None
        self._mem_write_phy = None

        # 共享总线引用 — 用于判断 MMIO 地址 (不可缓存) 和预留失效
        self._bus = None

        # 中断控制器引用 — 每条指令执行后在指令边界检查是否有待处理中断
        self._interrupt_ctrl = None

        # LR/SC 预留 (A-extension 原子指令)
        # 执行 LR 时记录预留地址; 任何 hart 向该地址写入时清除预留;
        # SC 仅在预留有效时成功, 否则失败; trap 发生时也清除预留
        self._reservation_addr: int = 0
        self._reservation_valid: bool = False

        # 页表遍历模式 — 由 satp CSR 的 MODE 字段决定
        # Bare=0, Sv39=8, Sv48=9, Sv57=10, Sv64=11
        self._mmu_mode = 0

    # ----------------------------------------------------------
    #  寄存器读写
    # ----------------------------------------------------------

    def read_csr(self, csr_id: int) -> int:
        check, csr_name = check_csr(csr_id)
        if not check:
            return -1
        return self.csrs[csr_name].val

    def write_csr(self, csr_id: int, val: int):
        check, csr_name = check_csr(csr_id)
        if not check:
            return
        self.csrs[csr_name].val = val

    def read_gpr(self, reg_id: int) -> int:
        return self.gprs[reg_id & 0x1F].val

    def write_gpr(self, reg_id: int, val: int):
        self.gprs[reg_id & 0x1F].val = val

    def read_fpr(self, reg_id: int) -> float:
        return self.fprs[reg_id & 0x1F].val

    def write_fpr(self, reg_id: int, val: float):
        self.fprs[reg_id & 0x1F].val = val

    # ----------------------------------------------------------
    #  mstatus 字段快捷属性 (直接读写 CSR 中的 mstatus 值)
    # ----------------------------------------------------------

    @property
    def mstatus_val(self) -> int:
        """读取完整 mstatus CSR 值."""
        return self.csrs["mstatus"].val

    @mstatus_val.setter
    def mstatus_val(self, v: int):
        self.csrs["mstatus"].val = v

    # -- MIE / MPIE / MPP (Machine 级) --

    @property
    def mie(self) -> bool:
        return bool(self.mstatus_val & MSTATUS_MIE)

    @mie.setter
    def mie(self, en: bool):
        """设置 Machine 中断使能."""
        if en:
            self.mstatus_val |= MSTATUS_MIE
        else:
            self.mstatus_val &= ~MSTATUS_MIE

    @property
    def mpie(self) -> bool:
        return bool(self.mstatus_val & MSTATUS_MPIE)

    @mpie.setter
    def mpie(self, en: bool):
        if en:
            self.mstatus_val |= MSTATUS_MPIE
        else:
            self.mstatus_val &= ~MSTATUS_MPIE

    @property
    def mpp(self) -> RiscvMode:
        """从 mstatus.MPP 字段解码先前的特权级."""
        raw = (self.mstatus_val & MSTATUS_MPP) >> 11
        return _MPP_TO_MODE.get(raw, RiscvMode.U)

    @mpp.setter
    def mpp(self, mode: RiscvMode):
        """将当前特权级编码写入 mstatus.MPP."""
        raw = _MODE_TO_MPP.get(mode, 0)
        self.mstatus_val = (self.mstatus_val & ~MSTATUS_MPP) | (raw << 11)

    # -- SIE / SPIE / SPP (Supervisor 级) --

    @property
    def sie(self) -> bool:
        return bool(self.mstatus_val & MSTATUS_SIE)

    @sie.setter
    def sie(self, en: bool):
        if en:
            self.mstatus_val |= MSTATUS_SIE
        else:
            self.mstatus_val &= ~MSTATUS_SIE

    @property
    def spie(self) -> bool:
        return bool(self.mstatus_val & MSTATUS_SPIE)

    @spie.setter
    def spie(self, en: bool):
        if en:
            self.mstatus_val |= MSTATUS_SPIE
        else:
            self.mstatus_val &= ~MSTATUS_SPIE

    @property
    def spp(self) -> RiscvMode:
        """从 mstatus.SPP 字段解码 Supervisor 先前的特权级."""
        raw = (self.mstatus_val & MSTATUS_SPP) >> 8
        return _SPP_TO_MODE.get(raw, RiscvMode.U)

    @spp.setter
    def spp(self, mode: RiscvMode):
        raw = _MODE_TO_SPP.get(mode, 0)
        self.mstatus_val = (self.mstatus_val & ~MSTATUS_SPP) | (raw << 8)

    # ----------------------------------------------------------
    #  其他关键 CSR 的快捷属性
    # ----------------------------------------------------------

    @property
    def mtvec_val(self) -> int:
        """M 模式 trap 向量基址 (低 2 位为 MODE)."""
        return self.csrs["mtvec"].val

    @property
    def stvec_val(self) -> int:
        """S 模式 trap 向量基址."""
        return self.csrs["stvec"].val

    @property
    def mepc_val(self) -> int:
        return self.csrs["mepc"].val

    @mepc_val.setter
    def mepc_val(self, v: int):
        self.csrs["mepc"].val = v

    @property
    def sepc_val(self) -> int:
        return self.csrs["sepc"].val

    @sepc_val.setter
    def sepc_val(self, v: int):
        self.csrs["sepc"].val = v

    @property
    def mcause_val(self) -> int:
        return self.csrs["mcause"].val

    @mcause_val.setter
    def mcause_val(self, v: int):
        self.csrs["mcause"].val = v

    @property
    def scause_val(self) -> int:
        return self.csrs["scause"].val

    @scause_val.setter
    def scause_val(self, v: int):
        self.csrs["scause"].val = v

    @property
    def mtval_val(self) -> int:
        return self.csrs["mtval"].val

    @mtval_val.setter
    def mtval_val(self, v: int):
        self.csrs["mtval"].val = v

    @property
    def stval_val(self) -> int:
        return self.csrs["stval"].val

    @stval_val.setter
    def stval_val(self, v: int):
        self.csrs["stval"].val = v

    @property
    def satp_val(self) -> int:
        """satp CSR: MODE(4) | ASID(16) | PPN(44)."""
        return self.csrs["satp"].val

    # ----------------------------------------------------------
    #  页表遍历模式 (从 satp.MODE 解析)
    # ----------------------------------------------------------

    @property
    def mmu_mode(self) -> int:
        """返回 satp.MODE 字段 (Bare=0, Sv39=8, Sv48=9)."""
        return self.satp_val >> 60

    @property
    def satp_ppn(self) -> int:
        """返回 satp 中的根页表物理页号 (PPN)."""
        return self.satp_val & ((1 << 44) - 1)

    # ----------------------------------------------------------
    #  总线 / 中断控制器引用 (Emulator 初始化时注入)
    # ----------------------------------------------------------

    @property
    def bus(self):
        """共享总线 (用于 MMIO 地址检测)."""
        return self._bus

    @bus.setter
    def bus(self, b):
        self._bus = b

    @property
    def interrupt_ctrl(self):
        """中断控制器 (CLINT / 未来 IMSIC 等)."""
        return self._interrupt_ctrl

    @interrupt_ctrl.setter
    def interrupt_ctrl(self, ctrl):
        self._interrupt_ctrl = ctrl

    # ----------------------------------------------------------
    #  mip 快捷属性 (中断挂起位)
    # ----------------------------------------------------------

    @property
    def mip_val(self) -> int:
        """读取 mip CSR (中断挂起寄存器)."""
        return self.csrs["mip"].val

    @mip_val.setter
    def mip_val(self, v: int):
        self.csrs["mip"].val = v

    @property
    def mie_val(self) -> int:
        """读取 mie CSR (中断使能寄存器)."""
        return self.csrs["mie"].val

    # ----------------------------------------------------------
    #  LR/SC 预留管理 (A-extension)
    # ----------------------------------------------------------

    def set_reservation(self, addr: int) -> None:
        """设置 LR 预留地址 (原子 load-reserved 成功时调用)."""
        self._reservation_addr = addr
        self._reservation_valid = True

    def clear_reservation(self) -> None:
        """清除 LR/SC 预留 (SC 失败 / 其他 hart 写入 / trap 时调用)."""
        self._reservation_valid = False
        self._reservation_addr = 0

    @property
    def reservation_valid(self) -> bool:
        return self._reservation_valid

    @property
    def reservation_addr(self) -> int:
        return self._reservation_addr

    # ----------------------------------------------------------
    #  中断检查 (指令边界)
    # ----------------------------------------------------------

    def check_pending_interrupts(self) -> bool:
        """在指令边界检查是否有待处理且使能的中断。

        若有, 则通过 _take_trap 注入中断并返回 True;
        否则返回 False.

        优先级: MEI > MSI > MTI > SEI > SSI > STI
        (RISC-V Privileged Spec §3.1.9)
        """
        if self._interrupt_ctrl is None:
            return False

        has_pending, mip_bits, highest_src = self._interrupt_ctrl.check_interrupt(
            self.id
        )
        if not has_pending:
            return False

        # 更新 mip CSR: 合并硬件中断源和软件写入的 mip 位
        current_mip = self.mip_val
        self.mip_val = current_mip | mip_bits

        # 检查是否有 M 模式使能的中断 (mip & mie & M-mode bits)
        mie = self.mie_val
        m_enabled = (self.mode == RiscvMode.M) or (
            self.mode in (RiscvMode.U, RiscvMode.S)
        )
        if not m_enabled:
            # S 模式或更低: M 模式中断始终可以抢占
            pass

        # 若当前在 M 模式且 MIE=0, 不响应中断
        if self.mode == RiscvMode.M and not self.mie:
            return False

        masked = self.mip_val & mie
        if masked == 0:
            return False

        # 按优先级找最高优先级的使能中断
        int_priority = [
            (1 << 11, TrapType.MmodeExternInterrupt),  # MEI
            (1 << 3, TrapType.MmodeSoftInterrupt),  # MSI
            (1 << 7, TrapType.MmodeTimerInterrupt),  # MTI
        ]
        for mask, trap_type in int_priority:
            if masked & mask:
                # 中断已响应 — 清除对应的 mip 位
                # (CLINT 中 MSIP 需软件清除, 但这里清除内部表示)
                self._take_trap(trap_type, tval=0, is_interrupt=True)
                return True

        return False
