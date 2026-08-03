#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
Hart (硬件线程) 的寄存器文件定义，包含:
- 通用整数寄存器 (GPR x0-x31)
- 浮点寄存器 (FPR f0-f31)
- 控制和状态寄存器 (CSR)
- 特权级模式 (RiscvMode)
- mstatus 等关键 CSR 的位字段定义及快捷属性
"""

from __future__ import annotations

from collections.abc import Callable
import ctypes
from enum import Enum
from typing import TYPE_CHECKING

from pyremu.core.diag import HartDiag
from pyremu.core.registers import (
    check_csr,
    csr_addr_from_name,
    gpr_idx_from_name,
    register_csr,
    register_fpr,
)
from pyremu.memory.cache import TLB_SIZE
from pyremu.memory.pmp import Pmp
from pyremu.memory.tlb import TLB
from pyremu.utils.mask import mask64, mask32

if TYPE_CHECKING:
    from pyremu.interrupt.controller import InterruptController
    from pyremu.interrupt.plic import PLIC
    from pyremu.memory.bus import Bus


class GprFile:
    """GPR 寄存器文件 — 32 个 64-bit 整数, x0 硬连线为 0.

    替代 pydantic ``list[Reg]`` 作为指令执行热路径上的寄存器后端,
    消除每条指令 ~150ns 的 pydantic 模型验证开销.

    用法与 ``list[int]`` 一致: ``gprs[10]`` 读, ``gprs[10] = v`` 写.
    x0 (索引 0) 写入被静默丢弃, 读取恒返回 0.
    """

    __slots__ = ("_r",)

    def __init__(self) -> None:
        self._r = [0] * 32

    def __getitem__(self, idx: int) -> int:
        return self._r[idx & 0x1F]

    def __setitem__(self, idx: int, val: int) -> None:
        i = idx & 0x1F
        if i != 0:
            self._r[i] = mask64(val)

    def __iter__(self):
        return iter(self._r)

    def __len__(self) -> int:
        return 32

    def as_list(self) -> list[int]:
        """返回底层 32 元素的副本, 供快照等外部使用."""
        return self._r.copy()


class RiscvMode(Enum):
    """RISC-V 特权级模式。

    数值按标准 RISC-V 特权级编码:
        U=0 (用户), S=1 (监管), H=2 ( hypervisor ),
        M=3 (机器), D=8 (调试).
    """

    U = 0
    S = 1
    H = 2
    M = 3
    D = 8


# u8 privilege value -> RiscvMode (for unmarshalling from FFI / wire formats).
_MODE_FROM_U8: dict[int, RiscvMode] = {0: RiscvMode.U, 1: RiscvMode.S, 3: RiscvMode.M}


def mode_from_u8(raw: int) -> RiscvMode:
    """Convert the wire-level u8 privilege encoding to a ``RiscvMode``.

    Unknown values fall back to M-mode.
    """
    return _MODE_FROM_U8.get(raw, RiscvMode.M)


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

# sstatus 是 mstatus 的受限视图 (RISC-V 特权规范 §4.1.1).
# sstatus 的读写必须通过 mstatus 的对应位, 两者不可独立.
_SSTATUS_MASK = (
    MSTATUS_SIE
    | MSTATUS_SPIE
    | MSTATUS_UBE
    | MSTATUS_SPP
    | MSTATUS_VS
    | MSTATUS_FS
    | MSTATUS_XS
    | MSTATUS_MPRV  # SUM 复用 MPRV 位? 不, SUM=bit18, bit17=MPRV
    | (1 << 18)  # SUM: permit Supervisor User Memory access
    | (1 << 19)  # MXR: Make eXecutable Readable
    | (1 << 63)  # SD: state dirty (read-only, = FS==3 || XS==3)
)
# writable sstatus mask 排除 VS (bits 9-10) 和 SD (bit 63)
_SSTATUS_WRITABLE_MASK = _SSTATUS_MASK & ~(MSTATUS_VS | (1 << 63))
MSTATUS_SUM = 1 << 18  # 允许 S 模式访问 U 模式页
MSTATUS_MXR = 1 << 19  # 使能可执行内存的可读
MSTATUS_TVM = 1 << 20  # 陷入 S 模式访问 satp
MSTATUS_TW = 1 << 21  # 陷入 WFI
MSTATUS_TSR = 1 << 22  # 陷入 SRET
MSTATUS_SD = 1 << 63  # 状态脏位 (FS 或 XS 为脏时置 1)

# SPP / MPP 编码 -> RiscvMode 的映射
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

    def __init__(self, id: int, pmp_entries: int = 64):
        self.id = id
        self.gprs = GprFile()
        self.fprs = register_fpr()
        # 浮点寄存器原始 bits (NaN-boxed u64) — FFI marshal 与位精确运算的真值源。
        # self.fprs (FPR float 对象) 供调试器展示; _fpr_bits 为权威存储。
        self._fpr_bits = [0] * 32
        self.csrs = register_csr()

        # 机器信息寄存器 — 只读, 复位时写入
        self.csrs["mhartid"].val = id
        self.csrs["mvendorid"].val = 0  # 非商业实现
        self.csrs["marchid"].val = 0  # 未指定架构 ID
        self.csrs["mimpid"].val = 1  # 实现版本
        # misa: MXL=2 (RV64) | I | M | A | F | D | C | S | U
        #    (与 DTB riscv,isa 字段一致: rv64imafdc, 加 S/U 支持)
        self.csrs["misa"].val = (
            (2 << 62)               # MXL=2 (RV64)
            | (1 << 8)              # I — base integer
            | (1 << 12)             # M — integer multiply/divide
            | (1 << 0)              # A — atomic
            | (1 << 5)              # F — single-precision float
            | (1 << 3)              # D — double-precision float
            | (1 << 2)              # C — compressed
            | (1 << 18)             # S — supervisor mode
            | (1 << 20)             # U — user mode
        )

        self.pc = 0
        self.mode = RiscvMode.M

        # TLB: 指令 TLB 和数据 TLB 分开
        # MMIO 地址禁止 CPU 缓存 (读 MMIO 可能改变硬件状态)
        self.itlb = TLB(size=TLB_SIZE)
        self.dtlb = TLB(size=TLB_SIZE)

        # PMP: 物理内存保护 — 条目数为运行时只读的平台约束
        # 超范围 CSR 访问在 _check_csr 中通过 _pmp_csr_valid 触发 IllInstr
        self._pmp_entries = pmp_entries
        self._pmp = Pmp(self.csrs, num_entries=pmp_entries)

        # 物理内存后端 — 由子类或外部注入
        # 调用约定: mem_read_phy(addr: int, size: int) -> bytes
        #           mem_write_phy(addr: int, data: bytes) -> None
        self._mem_read_phy: Callable[[int, int], bytes] | None = None
        self._mem_write_phy: Callable[[int, bytes], None] | None = None

        # 共享总线引用 — 用于判断 MMIO 地址 (不可缓存) 和预留失效
        self._bus: Bus | None = None

        # 全 hart 引用 — 用于 mfence.did 等需要广播到所有 hart 的操作
        self._all_harts: list[HartWithRegs] | None = None

        # 中断控制器引用 — 每条指令执行后在指令边界检查是否有待处理中断
        self._interrupt_ctrl: InterruptController | None = None

        # PLIC 引用 — 外部中断 (MEIP/SEIP), 与 CLINT 互补
        self._plic: PLIC | None = None

        # LR/SC 预留 (A-extension 原子指令)
        # 执行 LR 时记录预留地址; 任何 hart 向该地址写入时清除预留;
        # SC 仅在预留有效时成功, 否则失败; trap 发生时也清除预留
        self._reservation_addr: int = 0
        self._reservation_valid: bool = False
        self._reservation_value: int = 0  # value loaded by LR, passed to SC for CAS

        # 页表遍历模式 — 由 satp CSR 的 MODE 字段决定
        # Bare=0, Sv39=8, Sv48=9, Sv57=10, Sv64=11
        self._mmu_mode = 0

        # 异常/陷态追踪
        self._halted: bool = False  # 进入不可恢复陷态后置位
        self._consecutive_traps: int = 0  # 连续 trap 计数 (正常执行时清零)

        # mdid 缓存 — 避免每周期通过 pydantic dict 读取 (热路径, ~1M 次/基准测试)
        self._mdid_val: int = 0

        # pmpsplit 缓存 — PMP 条目拆分点: 低 [0, split) 归 host, 高 [split, N) 归飞地
        self._pmpsplit_val: int = 0

        # WFI 低功耗等待状态
        # 当 hart 执行 WFI 且无可处理中断时置位; 中断挂起且使能时硬件唤醒
        self._waiting: bool = False

        # WFI 唤醒标记 — 从 WFI 被中断唤醒后置位, mret/sret 或重新进入 WFI 时清零.
        # 此期间的指令 (中断 handler + 返回路径) 不计入 _total_instrs,
        # 以保证指令计数器反映的是固件实际执行的非中断上下文指令.
        self._wfi_woken: bool = False

        # 每 hart 指令计数 — 该 hart 实际执行的指令数.
        # 跨 batch 累加; native batch 中由 Rust 通过 state.total_instrs 更新,
        # 纯 Python 路径中由 sync_counters 递增.
        self._total_instrs: int = 0

        # CLINT MSIP 边沿诊断 (并发路径 sync_msip 更新)
        self.diag = HartDiag()

        # 中断状态缓存 — 避免每条指令都做完整的 CLINT+CSR+PLIC 遍历 (~1272 ns).
        # _int_state_version 在软件写 CSR / CLINT 变化 / 特权级切换时递增.
        # check_pending_interrupts 发现版本未变且 timer 未到期时直接返回 False.
        self._int_state_version: int = 0
        self._int_cache_version: int = -1  # -1 = 首次调用强制全量检查
        self._int_cache_next_timer: int = 0  # 最近 timer 唤醒时间 (0 = 无 timer 使能)

    # ----------------------------------------------------------
    #  寄存器读写
    # ----------------------------------------------------------

    def _csr_read_raw(self, csr_name: str) -> int:
        """Read CSR value bypassing pydantic model (hot path, ~10% faster)."""
        csr = self.csrs[csr_name]
        assert csr is not None, f"CSR {csr_name!r} not found in hart {self.id}"
        return csr.__dict__["val"]

    def read_csr(self, csr_id: int) -> int:
        check, csr_name = check_csr(csr_id)
        if not check:
            return -1
        # 硬件计数器 — 从 CLINT 动态读取 (time / timeh)
        # RISC-V 规范: time / timeh 是内存映射 CLINT mtime 的只读 CSR 镜像
        if csr_name == "time" and self._interrupt_ctrl is not None:
            return self._interrupt_ctrl.get_mtime()
        if csr_name == "timeh" and self._interrupt_ctrl is not None:
            return mask32(self._interrupt_ctrl.get_mtime() >> 32)
        # sstatus 是 mstatus 的受限视图, 读取 sstatus 时返回 mstatus 的对应位
        if csr_name == "sstatus":
            mval = self._csr_read_raw("mstatus")
            return mval & _SSTATUS_MASK
        # sie 是 mie 的受限视图 — 只有 mideleg 委派的位在 S 模式可见
        if csr_name == "sie":
            return self._csr_read_raw("mie") & self._csr_read_raw("mideleg")
        # sip 是 mip 的受限视图 — 只有 mideleg 委派的位在 S 模式可见
        if csr_name == "sip":
            return self._csr_read_raw("mip") & self._csr_read_raw("mideleg")
        return self._csr_read_raw(csr_name)

    # CSR names whose writes affect interrupt state and must invalidate the
    # check_pending_interrupts cache.
    _INT_SENSITIVE_CSRS: frozenset[str] = frozenset({
        "mip", "mie", "mideleg", "mstatus", "stimecmp",
    })

    def _csr_write_raw(self, csr_name: str, val: int) -> None:
        """Write CSR value bypassing pydantic ``BaseModel.__setattr__``.

        Hot-path optimization: ``h.csrs[name].val = v`` costs ~120ns per write
        in pydantic field-set machinery; ``__dict__['val']`` is ~40ns (3× faster).
        Callers MUST ensure *csr_name* is valid and no side-effects are required.
        """
        csr = self.csrs[csr_name]
        assert csr is not None, f"CSR {csr_name!r} not found in hart {self.id}"
        csr.__dict__["val"] = mask64(val)# type: ignore[index]  # pydantic MappingProxyType vs runtime dict
        if csr_name in self._INT_SENSITIVE_CSRS:
            self._int_state_version += 1

    def write_csr(self, csr_id: int, val: int):
        check, csr_name = check_csr(csr_id)
        if not check:
            return
        # 有副作用的 CSR 必须通过 property setter 写入以同步缓存
        if csr_name == "satp":
            self.satp_val = val
        elif csr_name == "mdid":
            self.mdid_val = val
        elif csr_name == "pmpsplit":
            self.pmpsplit_val = val
        elif csr_name.startswith(("pmpcfg", "pmpaddr")):
            # PMP CSR 写入 -> 使 Rust 扁平缓存失效, 并同步到所有 hart
            self._csr_write_raw(csr_name, val)
            self._pmp.invalidate_cache()
            # 同步 PMP 到其他 hart — OpenSBI 冷启动 hart 可能不是 hart 0,
            # 而 _step_native 始终使用 active[0]._pmp 构建传给 Rust 的扁平数组
            for h in (self._all_harts or ()):
                if h is not self:
                    h._csr_write_raw(csr_name, val)
                    h._pmp.invalidate_cache()
            return
        elif csr_name == "stimecmp":
            # SSTC: S-mode 直接写 stimecmp -> 同步到 CLINT mtimecmp
            # 用于 timer 比较 (mtime >= stimecmp 时触发 STIP)
            self._csr_write_raw(csr_name, val)
            if self._interrupt_ctrl is not None:
                self._interrupt_ctrl.set_mtimecmp(self.id, val)
        elif csr_name == "stimecmph":
            # RV32 only: stimecmp 高 32 位 (RV64 上 stimecmp 已是 64-bit)
            cur = mask32(self._csr_read_raw("stimecmp"))
            merged = cur | (mask32(val) << 32)
            self._csr_write_raw("stimecmp", merged)
            self._csr_write_raw(csr_name, mask32(val))
            if self._interrupt_ctrl is not None:
                self._interrupt_ctrl.set_mtimecmp(self.id, merged)
        elif csr_name == "sstatus":
            # sstatus 是 mstatus 的受限视图: 写入 sstatus 时更新 mstatus 对应位
            mstatus = self._csr_read_raw("mstatus")
            mstatus = (mstatus & ~_SSTATUS_WRITABLE_MASK) | (val & _SSTATUS_WRITABLE_MASK)
            self._csr_write_raw("mstatus", mstatus)
            self._csr_write_raw("sstatus", mstatus & _SSTATUS_MASK)
        elif csr_name == "sie":
            # sie 是 mie 的受限视图 — 只有 mideleg 委派的位可通过 S 模式写
            mideleg = self._csr_read_raw("mideleg")
            old_mie = self._csr_read_raw("mie")
            new_mie = (old_mie & ~mideleg) | (val & mideleg)
            self._csr_write_raw("mie", new_mie)
        elif csr_name == "sip":
            # sip 是 mip 的受限视图 — 只有 mideleg 委派的位可通过 S 模式写
            # (大多数中断是只读的, 但 SSIP 可被 S 模式软件置位/清除)
            mideleg = self._csr_read_raw("mideleg")
            old_mip = self._csr_read_raw("mip")
            new_mip = (old_mip & ~mideleg) | (val & mideleg)
            self._csr_write_raw("mip", new_mip)
        else:
            self._csr_write_raw(csr_name, val)

    def read_gpr(self, reg_id: int) -> int:
        return self.gprs[reg_id & 0x1F]

    def write_gpr(self, reg_id: int, val: int):
        self.gprs[reg_id & 0x1F] = val

    def read_fpr(self, reg_id: int) -> float:
        return self.fprs[reg_id & 0x1F].val

    def write_fpr(self, reg_id: int, val: float):
        self.fprs[reg_id & 0x1F].val = val

    def read_gpr_by_name(self, name: str) -> int:
        """按名称读取 GPR (例: x12, t0, a0, sp, zero). 未找到返回 0."""
        idx = gpr_idx_from_name(name)
        if idx is None:
            return 0
        return self.read_gpr(idx)

    def read_csr_by_name(self, name: str) -> int:
        """按名称读取 CSR (例: mtvec, mstatus, mepc, misa). 未找到返回 0."""
        addr = csr_addr_from_name(name)
        if addr is None:
            return 0
        return self.read_csr(addr)

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

    @satp_val.setter
    def satp_val(self, v: int) -> None:
        """写入 satp 时同步更新缓存的 MMU 模式.

        ASID 字段 (bits[59:44]) 按 WARL 硬连线为 0: 本实现的 TLB (Python 与
        Rust 批量引擎两侧) 查找均不带 ASID 标签。若允许 ASID 读回非零,
        Linux 探测到 ASID 支持后会启用 ASID 分配器, 上下文切换时仅改写
        satp.ASID 而不执行 sfence.vma — 前一地址空间的 TLB 表项残留命中,
        用户进程读到脏数据随机 SIGSEGV (ld.so 崩溃)。读回 0 则内核走
        no-ASID 路径, 每次 mm 切换显式 local_flush_tlb_all().

        每次 satp 写入均刷新 TLB — 不仅限 MODE 字段变化.
        约束: 飞地上下文切换 (alter_hart_ctx_for_enclave) 恢复 host
        satp 时, host 与飞地均为 Sv39 -> MODE 相同 -> 旧逻辑跳过 flush ->
        飞地的 TLB 残留 (VPN2=0x180, 与内核 VA 重叠) 毒化内核地址空间 ->
        缺页异常 + 栈溢出.
        """
        v &= ~(0xFFFF << 44)
        self.csrs["satp"].val = mask64(v)
        # 必须无条件刷新: 两个不同 Sv39 页表之间切换时 MODE 不变,
        # 但 TLB 中的旧映射 (VPN->PPN) 已失效.
        self.itlb.flush_all()
        self.dtlb.flush_all()
        self._mmu_mode = (v >> 60) & 0xF

    # ----------------------------------------------------------
    #  页表遍历模式 (由 satp.MODE 字段驱动)
    # ----------------------------------------------------------

    @property
    def mmu_mode(self) -> int:
        """返回当前 MMU 翻译模式 (Bare=0, Sv39=8, Sv48=9).

        值由 satp_val setter 在每次写入 satp CSR 时同步更新.
        """
        return self._mmu_mode

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
        # 自动注册中断状态变化回调 — 即使外部替换 CLINT 也不漏通知
        if ctrl is not None:
            ctrl.set_int_state_change_callback(self.notify_int_state_change)

    @property
    def plic(self):
        """PLIC 中断控制器 (外部中断 MEIP/SEIP)."""
        return self._plic

    @plic.setter
    def plic(self, p):
        self._plic = p

    def notify_int_state_change(self) -> None:
        """通知中断状态可能已改变 (CSR 写入 / CLINT 更新 / 特权级切换).

        递增版本号使 check_pending_interrupts 的缓存失效,
        触发下一条指令的全量中断检查.
        """
        self._int_state_version += 1

    @property
    def all_harts(self):
        """所有 hart 的引用 (用于广播操作, 如 mfence.did)."""
        return self._all_harts

    @all_harts.setter
    def all_harts(self, harts):
        self._all_harts = harts

    # ----------------------------------------------------------
    #  mdid — 内存域 ID (TEE 飞地/服务标识, M 模式管理器维护)
    # ----------------------------------------------------------

    @property
    def mdid_val(self) -> int:
        """读取 mdid CSR (内存域 ID) — 缓存避免每周期 pydantic dict 查找."""
        return self._mdid_val

    @mdid_val.setter
    def mdid_val(self, v: int):
        val = mask64(v)
        self._mdid_val = val
        self.csrs["mdid"].val = val

    # ----------------------------------------------------------
    #  pmpsplit — PMP 条目拆分 (TEE 飞地 PMP 虚拟化)
    # ----------------------------------------------------------

    @property
    def pmpsplit_val(self) -> int:
        """读取 pmpsplit CSR — PMP 条目拆分点.

        PMP 条目 [0, split) 归 host (mdid==0) 使用,
        [split, N) 归飞地 (mdid!=0) 使用.
        值为 0 时全部条目归 host (无飞地 PMP 隔离).
        """
        return self._pmpsplit_val

    @pmpsplit_val.setter
    def pmpsplit_val(self, v: int):
        val = mask64(v)
        self._pmpsplit_val = val
        self.csrs["pmpsplit"].val = val

    # ----------------------------------------------------------
    #  mip 快捷属性 (中断挂起位)
    # ----------------------------------------------------------

    @property
    def mip_val(self) -> int:
        """读取 mip CSR (中断挂起寄存器)."""
        return self.csrs["mip"].val

    @mip_val.setter
    def mip_val(self, v: int):
        self._csr_write_raw("mip", v)

    @property
    def mie_val(self) -> int:
        """读取 mie CSR (中断使能寄存器)."""
        return self.csrs["mie"].val

    # ----------------------------------------------------------
    #  LR/SC 预留管理 (A-extension)
    # ----------------------------------------------------------

    def set_reservation(self, addr: int, value: int = 0) -> None:
        """设置 LR 预留地址和加载值 (原子 load-reserved 成功时调用)."""
        self._reservation_addr = addr
        self._reservation_valid = True
        self._reservation_value = value

    def clear_reservation(self) -> None:
        """清除 LR/SC 预留 (SC 失败 / 其他 hart 写入 / trap 时调用)."""
        self._reservation_valid = False
        self._reservation_addr = 0
        self._reservation_value = 0

    @property
    def reservation_valid(self) -> bool:
        return self._reservation_valid

    @property
    def reservation_addr(self) -> int:
        return self._reservation_addr

    # ----------------------------------------------------------
    #  trap 处理 — 由 core/trap_handler.py 中的独立函数提供
    #  deliver_trap / trap_ecall / trap_ebreak / trap_mret / trap_sret /
    #  handle_wfi / check_pending_interrupts
    # ----------------------------------------------------------


# ============================================================
#  FFI state structs — ctypes mirror of Rust repr(C) layouts
# ============================================================



class TlbEntry(ctypes.Structure):
    """Single TLB entry — must match Rust ``state::TlbEntry`` exactly (32 bytes)."""
    _fields_ = [
        ("vpn", ctypes.c_uint64),
        ("ppn", ctypes.c_uint64),
        ("perm", ctypes.c_uint8),
        ("level", ctypes.c_uint8),
        ("valid", ctypes.c_uint8),
        ("mdid", ctypes.c_uint8),
        ("tlb_epoch", ctypes.c_uint32),
        ("dirty", ctypes.c_uint8),
        ("accessed", ctypes.c_uint8),
        ("asid", ctypes.c_uint16),
    ]


class HartDiagC(ctypes.Structure):
    """Diagnostic counters — matches Rust ``HartDiag`` exactly."""
    _fields_ = [
        ("clint_msip_set", ctypes.c_uint64),
        ("clint_msip_clr", ctypes.c_uint64),
        ("clint_mtc_wr", ctypes.c_uint64),
        ("clint_msip_wr0", ctypes.c_uint64),
        ("clint_msip_wr1", ctypes.c_uint64),
        ("clint_wr1_remote", ctypes.c_uint64),
        ("clint_wr1_self", ctypes.c_uint64),
        ("cooldown_start", ctypes.c_uint64),
        ("wfi_wake_msip", ctypes.c_uint64),
        ("wfi_wake_mtip", ctypes.c_uint64),
        ("wfi_wake_other", ctypes.c_uint64),
        ("trap_msip_total", ctypes.c_uint64),
        ("trap_msip_delegated", ctypes.c_uint64),
        ("msip_last_seen", ctypes.c_uint64),
        ("msip_masked_by_msie", ctypes.c_uint64),
        ("msip_pending_no_trap", ctypes.c_uint64),
        ("nt_mip_snapshot", ctypes.c_uint64),
        ("nt_mie_snapshot", ctypes.c_uint64),
        ("msie_cleared_at_pc", ctypes.c_uint64),
        ("wfi_wake_no_msip_trap", ctypes.c_uint64),
        ("nt_mode", ctypes.c_uint8),
        ("nt_clint_raw", ctypes.c_uint8),
        ("_pad2", ctypes.c_uint8 * 6),
        ("nt_pending", ctypes.c_uint64),
    ]


class HartState(ctypes.Structure):
    """Per-hart state marshaled to/from the Rust batch execution loop.

    Field order and types must exactly match ``state::HartState`` in Rust.
    """

    _fields_ = [
        # GPRs: 32 × u64
        ("gprs", ctypes.c_uint64 * 32),
        # Key CSRs (u64)
        ("mstatus", ctypes.c_uint64),
        ("mtvec", ctypes.c_uint64),
        ("stvec", ctypes.c_uint64),
        ("mepc", ctypes.c_uint64),
        ("sepc", ctypes.c_uint64),
        ("mcause", ctypes.c_uint64),
        ("scause", ctypes.c_uint64),
        ("mtval", ctypes.c_uint64),
        ("stval", ctypes.c_uint64),
        ("satp", ctypes.c_uint64),
        ("mie", ctypes.c_uint64),
        ("mip", ctypes.c_uint64),
        ("medeleg", ctypes.c_uint64),
        ("mideleg", ctypes.c_uint64),
        # PC
        ("pc", ctypes.c_uint64),
        # Reservation (LR/SC)
        ("reservation_addr", ctypes.c_uint64),
        ("reservation_value", ctypes.c_uint64),
        # Single-byte fields (packed after u64s)
        ("reservation_valid", ctypes.c_uint8),
        ("mode", ctypes.c_uint8),
        ("mmu_mode", ctypes.c_uint8),
        ("waiting", ctypes.c_uint8),
        ("wfi_woken", ctypes.c_uint8),
        ("halted", ctypes.c_uint8),
        ("consecutive_traps", ctypes.c_uint8),
        ("mdid", ctypes.c_uint8),
        ("pmpsplit", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 5),
        # ---- Phase B: TLB entries (array-of-structs, 32 × 2) ----
        ("itlb", TlbEntry * TLB_SIZE),
        ("dtlb", TlbEntry * TLB_SIZE),
        # ---- Phase C: Extra CSRs ----
        ("mscratch", ctypes.c_uint64),
        ("sscratch", ctypes.c_uint64),
        ("mhartid", ctypes.c_uint64),
        ("mcounteren", ctypes.c_uint64),
        ("scounteren", ctypes.c_uint64),
        # ---- Phase D: Sstc stimecmp ----
        ("stimecmp", ctypes.c_uint64),
        # ---- Phase E: cache ----
        ("_mmu_mode_pad", ctypes.c_uint64),
        # ---- Phase F: per-hart instruction counter ----
        ("total_instrs", ctypes.c_uint64),
        # ---- Diagnostic sub-struct (matches Rust HartDiag) ----
        ("diag", HartDiagC),
        # ---- Phase G: F/D floating point ----
        ("fprs", ctypes.c_uint64 * 32),
        ("fcsr", ctypes.c_uint32),
        ("_fpad", ctypes.c_uint8 * 4),
    ]


class BatchResult(ctypes.Structure):
    """Result returned by ``run_batch`` describing why the batch stopped."""

    _fields_ = [
        ("total_instrs", ctypes.c_uint64),
        ("exit_reason", ctypes.c_uint8),
        ("exit_hart_id", ctypes.c_uint8),
        ("exit_pc", ctypes.c_uint64),
        ("exit_instr", ctypes.c_uint32),
        ("trap_cause", ctypes.c_uint32),
        ("trap_tval", ctypes.c_uint64),
        ("trap_is_interrupt", ctypes.c_uint8),
        ("trap_delegated", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 6),
    ]


# Exit reason constants (must match state.rs)
EXIT_NORMAL = 0
EXIT_TRAP = 1
EXIT_MMIO = 2
EXIT_ECALL = 3
EXIT_EBREAK = 4
EXIT_WFI_WAIT = 5
EXIT_ERROR = 6
EXIT_BREAKPOINT = 7


# ============================================================
#  Marshal / unmarshal
# ============================================================


def marshal_hart(hart: HartWithRegs, state: HartState) -> None:
    """Copy hart state from Python ``HartWithRegs`` into a Rust ``HartState``."""
    for i in range(32):
        state.gprs[i] = hart.gprs[i]

    state.mstatus = hart.mstatus_val
    state.mtvec = hart.csrs["mtvec"].val
    state.stvec = hart.csrs["stvec"].val
    state.mepc = hart.mepc_val
    state.sepc = hart.sepc_val
    state.mcause = hart.mcause_val
    state.scause = hart.scause_val
    state.mtval = hart.mtval_val
    state.stval = hart.stval_val
    state.satp = hart.satp_val
    state.mie = hart.mie_val
    state.mip = hart.mip_val
    state.medeleg = hart.csrs["medeleg"].val
    state.mideleg = hart.csrs["mideleg"].val

    state.pc = hart.pc
    state.mode = hart.mode.value
    state.mmu_mode = hart.mmu_mode

    state.reservation_valid = 1 if hart.reservation_valid else 0
    state.reservation_addr = hart.reservation_addr
    state.reservation_value = hart._reservation_value

    state.waiting = 1 if hart._waiting else 0
    state.wfi_woken = 1 if hart._wfi_woken else 0
    state.halted = 1 if hart._halted else 0
    state.consecutive_traps = hart._consecutive_traps

    state.mdid = hart.mdid_val
    state.pmpsplit = hart.pmpsplit_val

    # ---- Phase B: TLB entries ----
    # Skip TLB marshalling for now — Rust handles TLB internally during batch.
    # Python TLB stays as ground truth; before each batch, Rust TLB is cold
    # but refills from page walks.

    # ---- Phase C: Extra CSRs ----
    state.mscratch = hart.csrs["mscratch"].val
    state.sscratch = hart.csrs["sscratch"].val
    state.mhartid = hart.csrs["mhartid"].val
    state.mcounteren = hart.csrs["mcounteren"].val
    state.scounteren = hart.csrs["scounteren"].val if "scounteren" in hart.csrs else 0
    state.stimecmp = hart.csrs["stimecmp"].val if "stimecmp" in hart.csrs else 0

    # ---- Phase F: per-hart instruction counter ----
    state.total_instrs = hart._total_instrs

    # ---- Phase F2: MSIP edge counter — must survive round-trips so
    # Rust sync_msip doesn't re-detect stale edges at batch start ----
    state.diag.msip_last_seen = hart.diag.msip_last_seen

    # ---- Phase G: F/D floating point ----
    for i in range(32):
        state.fprs[i] = hart._fpr_bits[i]
    state.fcsr = hart.csrs["fcsr"].val & 0xFF


def unmarshal_hart(state: HartState, hart: HartWithRegs) -> None:
    """Copy Rust ``HartState`` back into a Python ``HartWithRegs``."""
    for i in range(32):
        hart.gprs[i] = state.gprs[i]

    hart.mstatus_val = state.mstatus
    hart.csrs["mtvec"].val = state.mtvec
    hart.csrs["stvec"].val = state.stvec
    hart.mepc_val = state.mepc
    hart.sepc_val = state.sepc
    hart.mcause_val = state.mcause
    hart.scause_val = state.scause
    hart.mtval_val = state.mtval
    hart.stval_val = state.stval
    hart.satp_val = state.satp
    hart._csr_write_raw("mie", state.mie)
    hart._csr_write_raw("mip", state.mip)
    hart.csrs["medeleg"].val = state.medeleg
    hart.csrs["mideleg"].val = state.mideleg

    hart.pc = state.pc
    hart.mode = mode_from_u8(state.mode)
    hart._mmu_mode = state.mmu_mode

    hart._reservation_valid = state.reservation_valid != 0
    hart._reservation_addr = state.reservation_addr
    hart._reservation_value = state.reservation_value

    hart._waiting = state.waiting != 0
    hart._wfi_woken = state.wfi_woken != 0
    hart._halted = state.halted != 0
    hart._consecutive_traps = state.consecutive_traps

    hart._mdid_val = state.mdid
    hart._pmpsplit_val = state.pmpsplit

    # ---- Phase B: TLB — Rust TLB is cold after batch, don't write back ----
    # Python TLB is ground truth.

    # ---- Phase C: Extra CSRs ----
    hart.csrs["mscratch"].val = state.mscratch
    hart.csrs["sscratch"].val = state.sscratch
    hart.csrs["mhartid"].val = state.mhartid
    hart.csrs["mcounteren"].val = state.mcounteren
    if "scounteren" in hart.csrs:
        hart.csrs["scounteren"].val = state.scounteren
    hart._csr_write_raw("stimecmp", state.stimecmp)

    # ---- Phase F: per-hart instruction counter ----
    hart._total_instrs = state.total_instrs

    # ---- Diagnostic: CLINT MSIP edge counters ----
    hart.diag.load_ctypes(state.diag)

    # ---- Phase G: F/D floating point ----
    for i in range(32):
        hart._fpr_bits[i] = state.fprs[i]
    hart.csrs["fcsr"].val = state.fcsr & 0xFF
    hart.csrs["frm"].val = (state.fcsr >> 5) & 0x7
    hart.csrs["fflags"].val = state.fcsr & 0x1F
