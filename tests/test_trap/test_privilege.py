#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""特权级恢复与 CSR 权限: MRET/SRET, U 模式特权指令, CSR 读写."""

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.hart import (
    MSTATUS_MIE,
    MSTATUS_MPIE,
    MSTATUS_MPP,
    MSTATUS_SIE,
    MSTATUS_SPIE,
    MSTATUS_SPP,
    MSTATUS_TW,
    RiscvMode,
)
from pyremu.core.mem_check_aux import (
    inject_memory_backend,
)
from pyremu.core.trap_def import TrapType
from pyremu.core.trap_handler import (
    deliver_trap,
    trap_mret,
    trap_sret,
)
from pyremu.memory.bus import Bus
from pyremu.utils.disassem import Opc
from tests.test_trap.support import (
    make_hart,
    MTVEC,
)


class TestMretSret:
    """验证从 trap 返回 (MRET/SRET) 的上下文恢复."""

    @pytest.fixture
    def hart_after_m_trap(self) -> Hart:
        """模拟: 从 M 模式进入 trap 后的 hart 状态."""
        h = Hart(id=0)
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        h.mode = RiscvMode.M
        h.mstatus_val = MSTATUS_MIE  # 初始 MIE=1
        deliver_trap(h, TrapType.IllInstr, is_interrupt=False)
        # 现在 hart 在 trap handler 中, MPP=M, MPIE=1, MIE=0
        return h

    def test_mret_restores_pc_and_mode(self, hart_after_m_trap):
        h = hart_after_m_trap
        assert h.mepc_val == 0x1000
        trap_mret(h)
        assert h.pc == 0x1000, "MRET 应恢复到 mepc"
        assert h.mode == RiscvMode.M, "MPP=M 应恢复到 M 模式"

    def test_mret_restores_mie_from_mpie(self, hart_after_m_trap):
        h = hart_after_m_trap
        trap_mret(h)
        assert h.mie is True, "MPIE=1 -> MIE 应恢复为 1"

    def test_mret_clears_mpp_to_u(self, hart_after_m_trap):
        h = hart_after_m_trap
        trap_mret(h)
        assert h.mpp == RiscvMode.U, "MRET 后 MPP 应重置为 U"

    def test_mret_preserves_mpie(self, hart_after_m_trap):
        h = hart_after_m_trap
        trap_mret(h)
        assert h.mpie is True, "MRET 后 MPIE 应为 1"

    @pytest.fixture
    def hart_after_s_trap(self) -> Hart:
        """通过 medeleg 委派获得 S 模式 trap 后的状态."""
        h = Hart(id=0)
        h.csrs["stvec"].val = 0x80004000
        h.pc = 0x3000
        h.mode = RiscvMode.S
        h.mstatus_val = MSTATUS_SIE  # SIE=1 before trap
        # 委派 S 模式 ecall: medeleg[9]=1 -> trap 留在 S 模式
        h.csrs["medeleg"].val = 1 << 9
        deliver_trap(h, TrapType.EcallFromSmode, tval=0, is_interrupt=False)
        return h

    def test_sret_restores_pc_and_mode(self, hart_after_s_trap):
        h = hart_after_s_trap
        trap_sret(h)
        assert h.pc == 0x3000, "SRET 应恢复到 sepc"
        assert h.mode == RiscvMode.S, "SPP=S -> 应恢复到 S 模式"

    def test_sret_restores_sie_from_spie(self, hart_after_s_trap):
        h = hart_after_s_trap
        trap_sret(h)
        assert h.sie is True, "SPIE=1 -> SIE 应恢复为 1"

    def test_sret_clears_spp_to_u(self, hart_after_s_trap):
        h = hart_after_s_trap
        trap_sret(h)
        assert h.spp == RiscvMode.U, "SRET 后 SPP 应重置为 U"


class TestSretBranches:
    def test_sret_in_umode_traps_ill_instr(self):
        h = make_hart(RiscvMode.U)
        trap_sret(h)
        assert h.mcause_val == 2
        assert h.mtval_val == 0x10200073
        assert h.pc == MTVEC

    def test_sret_with_spie_clear_disables_sie(self):
        h = make_hart(RiscvMode.S)
        h.mstatus_val = MSTATUS_SIE | MSTATUS_SPP
        h.sepc_val = 0x8000_5000
        trap_sret(h)
        assert h.mode is RiscvMode.S
        assert not h.sie
        assert h.mstatus_val & MSTATUS_SPIE
        assert h.pc == 0x8000_5000

    def test_mret_with_mpie_clear_disables_mie(self):
        h = make_hart()
        h.mstatus_val = MSTATUS_MIE
        h.mepc_val = 0x8000_6000
        trap_mret(h)
        assert not (h.mstatus_val & MSTATUS_MIE)
        assert h.mstatus_val & MSTATUS_MPIE
        assert h.pc == 0x8000_6000


class TestUmodePrivilegedInstructionTraps:
    """U 模式执行特权指令/CSR -> 陷态, medeleg 委派到 S 模式."""

    @pytest.fixture
    def h(self) -> Hart:
        bus = Bus(ram_size=0x10000, ram_base=0x80000000)
        hart = Hart(id=0)
        inject_memory_backend(hart, bus.read, bus.write)
        hart.bus = bus
        hart.pc = 0x80000000
        hart.csrs["mtvec"].val = 0x80001000
        hart.csrs["stvec"].val = 0x80002000
        hart.mstatus_val = MSTATUS_MIE
        return hart

    # -- helpers --

    @staticmethod
    def _ecall() -> int:
        return 0x73  # ecall

    @staticmethod
    def _csrrw(rd: int, csr: int, rs1: int) -> int:
        return (csr << 20) | (rs1 << 15) | (0b001 << 12) | (rd << 7) | 0b1110011

    @staticmethod
    def _wfi() -> int:
        return 0x10500073

    # -- U -> S delegation of ECALL --

    def test_u_ecall_delegated_to_s(self, h):
        """U 模式 ecall -> medeleg 委派 -> S 模式 trap."""
        h.csrs["medeleg"].val = 1 << 8
        h.mode = RiscvMode.U
        h.exec_instr(self._ecall())
        assert h.mode == RiscvMode.S
        assert h.pc == 0x80002000, f"应跳转 stvec, 实际 PC={h.pc:#x}"
        assert h.scause_val == 8, f"scause 应为 8, 实际 {h.scause_val:#x}"

    def test_u_ecall_no_delegation_stays_m(self, h):
        """未委派: U 模式 ecall -> M 模式 trap."""
        h.mode = RiscvMode.U
        h.exec_instr(self._ecall())
        assert h.mode == RiscvMode.M
        assert h.pc == 0x80001000

    # -- U -> S delegation of CSR access --

    def test_u_csr_read_mstatus_traps(self, h):
        """U 模式读 mstatus (M-mode CSR) -> IllInstr -> 委派到 S."""
        h.csrs["medeleg"].val = 1 << 2  # 委派 IllInstr
        h.mode = RiscvMode.U
        instr = self._csrrw(5, 0x300, 0)  # csrrw t0, mstatus, x0
        h.exec_instr(instr)
        assert h.scause_val == 2, f"IllInstr=2, 实际 scause={h.scause_val:#x}"
        assert h.mode == RiscvMode.S

    def test_u_csr_write_satp_traps(self, h):
        """U 模式写 satp (S-mode CSR) -> 也应陷态 (U 模式不能直接写 S CSR)."""
        h.csrs["medeleg"].val = 1 << 2
        h.mode = RiscvMode.U
        instr = self._csrrw(0, 0x180, 5)  # csrrw x0, satp, t0
        h.exec_instr(instr)
        assert h.scause_val == 2
        assert h.mode == RiscvMode.S

    # -- U -> S delegation of WFI (mstatus.TW) --

    def test_u_wfi_with_tw_traps(self, h):
        """mstatus.TW=1 + U 模式 WFI -> IllInstr -> 委派到 S."""
        h.csrs["medeleg"].val = 1 << 2
        h.mode = RiscvMode.U
        h.mstatus_val = MSTATUS_TW  # TW=1
        h.exec_instr(self._wfi())
        assert h.scause_val == 2
        assert h.mode == RiscvMode.S

    # -- U -> S delegation of MRET --

    def test_u_mret_traps(self, h):
        """U 模式 mret -> IllInstr (U 模式不可执行 mret)."""
        h.csrs["medeleg"].val = 1 << 2
        h.mode = RiscvMode.U
        h.exec_instr(0x30200073)  # mret
        assert h.scause_val == 2, f"应为 IllInstr, scause={h.scause_val:#x}"
        assert h.mode == RiscvMode.S

    # -- 未委派时进入 M 模式 --

    def test_u_ill_instr_no_delegation_stays_m(self, h):
        """medeleg=0: U 模式 mret -> M 模式 trap."""
        h.mode = RiscvMode.U
        h.exec_instr(0x30200073)  # mret
        assert h.mode == RiscvMode.M
        assert h.pc == 0x80001000, f"应跳转 mtvec, PC={h.pc:#x}"


class TestCsrrwRdRs1:
    """csrrw rd, csr, rs1 在 rd==rs1 时必须先读 rs1 再写 rd."""

    def test_csrrw_rd_eq_rs1_preserves_new_value(self):
        """csrrw sp, sscratch, sp 应正确交换 sp 和 sscratch."""

        h = Hart(id=0)
        # 用 mscratch (M-mode 可访问) 代替 sscratch 测试
        old_sp = 0x80101000
        old_scratch = 0x8000A000
        h.gprs[2] = old_sp
        h.write_csr(0x340, old_scratch)  # mscratch = old_scratch

        # 执行 csrrw sp, mscratch, sp
        instr_val = (0x340 << 20) | (2 << 15) | (1 << 12) | (2 << 7) | Opc.sys.value
        h.exec_instr(instr_val)

        assert h.gprs[2] == old_scratch, (
            f"csrrw 后 sp 应为旧 mscratch 值 0x{old_scratch:08x}, 实际 0x{h.gprs[2]:08x}"
        )
        assert h.csrs["mscratch"].val == old_sp, (
            f"csrrw 后 mscratch 应为旧 sp 值 0x{old_sp:08x}, "
            f"实际 0x{h.csrs['mscratch'].val:08x}"
        )

    def test_csrrw_rd_neq_rs1_still_works(self):
        """csrrw t0, mscratch, sp 在 rd!=rs1 时仍应正常工作."""

        h = Hart(id=0)
        old_sp = 0x80101000
        old_scratch = 0x8000A000
        h.gprs[2] = old_sp
        h.write_csr(0x340, old_scratch)

        # csrrw t0, mscratch, sp  (rd=5, rs1=2)
        instr_val = (0x340 << 20) | (2 << 15) | (1 << 12) | (5 << 7) | Opc.sys.value
        h.exec_instr(instr_val)

        assert h.gprs[5] == old_scratch, "rd(t0) 应为旧 CSR 值"
        assert h.csrs["mscratch"].val == old_sp, "CSR 应写入 rs1(sp) 的原始值"

    def test_csrrs_rd_eq_rs1_preserves_rs1(self):
        """csrrs t0, mscratch, t0: rd==rs1 时 SET 位应使用原始 rs1 值."""

        h = Hart(id=0)
        old_csr = 0x00000000
        rs1_orig = 0x0000000F  # t0 = 0xF (要 SET 的位)
        h.gprs[5] = rs1_orig  # t0
        h.write_csr(0x340, old_csr)

        # csrrs t0, mscratch, t0  (rd=5, rs1=5)
        instr_val = (0x340 << 20) | (5 << 15) | (2 << 12) | (5 << 7) | Opc.sys.value
        h.exec_instr(instr_val)

        assert h.gprs[5] == old_csr, f"rd 应为旧 CSR 值 0, 实际 0x{h.gprs[5]:x}"
        assert h.csrs["mscratch"].val == rs1_orig, (
            f"CSR 应为 old|rs1=0x{rs1_orig:x}, 实际 0x{h.csrs['mscratch'].val:x}"
        )

    def test_csrrc_rd_eq_rs1_preserves_rs1(self):
        """csrrc t0, mscratch, t0: rd==rs1 时 CLEAR 位应使用原始 rs1 值."""

        h = Hart(id=0)
        old_csr = 0x000000FF
        rs1_orig = 0x0000000F  # t0 = 0xF (要 CLEAR 的位)
        h.gprs[5] = rs1_orig  # t0
        h.write_csr(0x340, old_csr)

        # csrrc t0, mscratch, t0  (rd=5, rs1=5)
        instr_val = (0x340 << 20) | (5 << 15) | (3 << 12) | (5 << 7) | Opc.sys.value
        h.exec_instr(instr_val)

        expected_csr = old_csr & ~rs1_orig  # 0xFF & ~0x0F = 0xF0
        assert h.gprs[5] == old_csr, f"rd 应为旧 CSR 值 0xFF, 实际 0x{h.gprs[5]:x}"
        assert h.csrs["mscratch"].val == expected_csr, (
            f"CSR 应为 0x{expected_csr:x}, 实际 0x{h.csrs['mscratch'].val:x}"
        )


class TestCsrPrivilege:
    """低特权级访问高特权 CSR 应触发 IllInstr 陷态."""

    SYS = Opc.sys.value  # 0x73

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        h.csrs["mtvec"].val = 0x80000000
        h.pc = 0x1000
        return h

    # ---- 辅助: 构建 CSR 指令编码 ----

    @staticmethod
    def _csrrw(rd: int, rs1: int, csr: int) -> int:
        """CSRRW rd, csr, rs1."""
        return (csr << 20) | (rs1 << 15) | (1 << 12) | (rd << 7) | Opc.sys.value

    @staticmethod
    def _csrrwi(rd: int, uimm: int, csr: int) -> int:
        """CSRRWI rd, csr, uimm."""
        return (csr << 20) | (uimm << 15) | (5 << 12) | (rd << 7) | Opc.sys.value

    @staticmethod
    def _csrrs(rd: int, rs1: int, csr: int) -> int:
        """CSRRS rd, csr, rs1."""
        return (csr << 20) | (rs1 << 15) | (2 << 12) | (rd << 7) | Opc.sys.value

    @staticmethod
    def _csrrc(rd: int, rs1: int, csr: int) -> int:
        """CSRRC rd, csr, rs1."""
        return (csr << 20) | (rs1 << 15) | (3 << 12) | (rd << 7) | Opc.sys.value

    # ---- U 模式访问高特权 CSR ----

    def test_umode_read_mstatus_traps(self, hart):
        """U 模式 CSRRW 读取 mstatus -> IllInstr."""
        hart.mode = RiscvMode.U
        # CSRRW x5, mstatus, x0  (只读)
        instr = self._csrrw(rd=5, rs1=0, csr=0x300)
        # exec_instr 不抛异常, 通过 _take_trap 注入
        hart.exec_instr(instr)
        assert hart.mcause_val == 2, "应触发 IllInstr"
        assert hart.mepc_val == 0x1000

    def test_umode_write_mstatus_traps(self, hart):
        """U 模式 CSRRW 写入 mstatus -> IllInstr."""
        hart.mode = RiscvMode.U
        # CSRRW x0, mstatus, x10
        instr = self._csrrw(rd=0, rs1=10, csr=0x300)
        hart.gprs[10] = 0xDEAD
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_umode_csrrwi_mepc_traps(self, hart):
        """U 模式 CSRRWI 写 mepc -> IllInstr."""
        hart.mode = RiscvMode.U
        instr = self._csrrwi(rd=0, uimm=7, csr=0x341)  # mepc
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_umode_csrrs_mie_traps(self, hart):
        """U 模式 CSRRS mie -> IllInstr."""
        hart.mode = RiscvMode.U
        instr = self._csrrs(rd=5, rs1=0, csr=0x304)  # mie
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    # ---- S 模式访问 M 模式 CSR ----

    def test_smode_read_mstatus_traps(self, hart):
        """S 模式访问 M-only CSR -> IllInstr."""
        hart.mode = RiscvMode.S
        instr = self._csrrw(rd=5, rs1=0, csr=0x300)  # mstatus
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_smode_write_mtvec_traps(self, hart):
        """S 模式写入 mtvec (M-only) -> IllInstr."""
        hart.mode = RiscvMode.S
        instr = self._csrrw(rd=0, rs1=10, csr=0x305)  # mtvec
        hart.gprs[10] = 0x8888
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_smode_can_access_stvec(self, hart):
        """S 模式可以访问 S-mode CSR stvec."""
        hart.mode = RiscvMode.S
        hart.csrs["stvec"].val = 0xABCD0000
        # CSRRW x5, stvec, x0 (只读)
        instr = self._csrrw(rd=5, rs1=0, csr=0x105)  # stvec
        advance = hart.exec_instr(instr)
        assert advance == 4, "合法 CSR 应正常推进 PC"
        assert hart.gprs[5] == 0xABCD0000

    # ---- 写入只读 CSR ----

    def test_umode_write_readonly_csr_traps(self, hart):
        """U 模式写入只读 CSR (cycle) -> IllInstr."""
        hart.mode = RiscvMode.U
        # CSRRW x0, cycle, x10
        instr = self._csrrw(rd=0, rs1=10, csr=0xC00)  # cycle (u_ro)
        hart.gprs[10] = 42
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_mmode_write_readonly_csr_traps(self, hart):
        """M 模式写入只读 CSR (mvendorid) -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=0, rs1=10, csr=0xF11)  # mvendorid (m_ro)
        hart.gprs[10] = 99
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    # ---- CSRRW rs1=x0 写只读 CSR (CSRRW 始终为写操作, 即使 rs1=x0) ----

    def test_csrrw_rs1_zero_to_readonly_traps(self, hart):
        """CSRRW x0, cycle, x0: rs1=x0 仍写入 CSR -> IllInstr.

        RISC-V spec: CSRRW 始终是写操作, 即使 rs1=x0 也会把 0 写入 CSR.
        写入只读 CSR 必然触发非法指令陷态. 此处复现用户调试器中的
        mtval=0xc0001073 IllInstr 场景."""
        hart.mode = RiscvMode.M
        # csrrw x0, cycle, x0 — funct3=001(CSRRW), rd=0, rs1=0, csr=0xC00
        instr = self._csrrw(rd=0, rs1=0, csr=0xC00)  # cycle (u_ro)
        hart.exec_instr(instr)
        assert hart.mcause_val == 2, "CSRRW rs1=x0 仍为写操作, 应触发 IllInstr"

    def test_csrrw_rs1_zero_to_mvendorid_traps(self, hart):
        """CSRRW x0, mvendorid, x0: M 模式 CSRRS 正确, 但 CSRRW 仍非法."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=0, rs1=0, csr=0xF11)  # mvendorid (m_ro)
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    # ---- CSRRWI 写只读 CSR ----

    def test_csrrwi_to_readonly_cycle_traps(self, hart):
        """CSRRWI x0, cycle, 7: 立即数形式写入只读 CSR -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrwi(rd=0, uimm=7, csr=0xC00)  # cycle (u_ro)
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_csrrwi_uimm_zero_to_readonly_traps(self, hart):
        """CSRRWI x0, cycle, 0: uimm=0 仍为写操作 -> IllInstr.

        CSRRWI 与 CSRRW 一样始终为写操作, uimm=0 时写 0 仍然非法."""
        hart.mode = RiscvMode.M
        instr = self._csrrwi(rd=0, uimm=0, csr=0xC00)  # cycle (u_ro)
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    # ---- CSRRS/CSRRC rs1≠0 写只读 CSR ----

    def test_csrrs_write_to_readonly_traps(self, hart):
        """CSRRS x0, cycle, x10: rs1≠0 时 SET 位是写操作 -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrs(rd=0, rs1=10, csr=0xC00)  # cycle (u_ro)
        hart.gprs[10] = 0x1
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_csrrc_write_to_readonly_traps(self, hart):
        """CSRRC x0, cycle, x10: rs1≠0 时 CLEAR 位是写操作 -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrc(rd=0, rs1=10, csr=0xC00)  # cycle (u_ro)
        hart.gprs[10] = 0x1
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    # ---- CSRRSI/CSRRCI uimm≠0 写只读 CSR ----

    def test_csrrsi_to_readonly_traps(self, hart):
        """CSRRSI x0, cycle, 3: uimm≠0 的 SET 为写操作 -> IllInstr."""
        hart.mode = RiscvMode.M
        # CSRRSI funct3=110
        instr = (0xC00 << 20) | (3 << 15) | (6 << 12) | (0 << 7) | Opc.sys.value
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_csrrci_to_readonly_traps(self, hart):
        """CSRRCI x0, cycle, 3: uimm≠0 的 CLEAR 为写操作 -> IllInstr."""
        hart.mode = RiscvMode.M
        # CSRRCI funct3=111
        instr = (0xC00 << 20) | (3 << 15) | (7 << 12) | (0 << 7) | Opc.sys.value
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    # ---- 合法访问不受影响 ----

    def test_umode_can_read_umode_csr(self, hart):
        """U 模式在 mcounteren 与 scounteren 使能后可以读取 cycle."""
        hart.mode = RiscvMode.U
        hart.csrs["mcounteren"].val = 1 << 0  # CY
        hart.csrs["scounteren"].val = 1 << 0  # CY
        # CSRRS x5, cycle, x0: rs1=0 表示只读 (不写 CSR)
        instr = self._csrrs(rd=5, rs1=0, csr=0xC00)  # cycle
        advance = hart.exec_instr(instr)
        assert advance == 4, "合法 CSR 读应正常推进 PC"

    def test_umode_cycle_read_without_scounteren_traps(self, hart):
        """U 模式读取 cycle: scounteren 未置位时 -> IllInstr."""
        hart.mode = RiscvMode.U
        hart.csrs["mcounteren"].val = 0b111
        hart.csrs["scounteren"].val = 0
        hart.exec_instr(self._csrrs(rd=5, rs1=0, csr=0xC00))
        assert hart.mcause_val == 2, "scounteren 未使能时读取 cycle 应触发 IllInstr"

    def test_smode_cycle_read_without_mcounteren_traps(self, hart):
        """S 模式读取 cycle: mcounteren 未置位时 -> IllInstr."""
        hart.mode = RiscvMode.S
        hart.csrs["mcounteren"].val = 0
        hart.csrs["scounteren"].val = 0b111
        hart.exec_instr(self._csrrs(rd=5, rs1=0, csr=0xC00))
        assert hart.mcause_val == 2, "mcounteren 未使能时读取 cycle 应触发 IllInstr"

    def test_mmode_can_read_write_mstatus(self, hart):
        """M 模式可以读写 mstatus."""
        hart.mode = RiscvMode.M
        hart.mstatus_val = 0xA0000000
        # CSRRW x0, mstatus, x10
        instr = self._csrrw(rd=0, rs1=10, csr=0x300)
        hart.gprs[10] = 0xB0000000
        advance = hart.exec_instr(instr)
        assert advance == 4
        assert hart.mstatus_val == 0xB0000000

    # ---- 机器信息只读寄存器 (mvendorid/marchid/mimpid/mhartid) ----

    def test_write_mvendorid_traps(self, hart):
        """写入只读 mvendorid -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=0, rs1=10, csr=0xF11)
        hart.gprs[10] = 99
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_write_marchid_traps(self, hart):
        """写入只读 marchid -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=0, rs1=10, csr=0xF12)
        hart.gprs[10] = 99
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_write_mimpid_traps(self, hart):
        """写入只读 mimpid -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=0, rs1=10, csr=0xF13)
        hart.gprs[10] = 99
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_write_mhartid_traps(self, hart):
        """写入只读 mhartid -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=0, rs1=10, csr=0xF14)
        hart.gprs[10] = 99
        hart.exec_instr(instr)
        assert hart.mcause_val == 2

    def test_read_mvendorid_succeeds(self, hart):
        """读取 mvendorid 返回平台预设值."""
        hart.mode = RiscvMode.M
        instr = self._csrrs(rd=5, rs1=0, csr=0xF11)
        advance = hart.exec_instr(instr)
        assert advance == 4
        assert hart.gprs[5] == hart.csrs["mvendorid"].val

    # ---- 非法 CSR 地址 ----

    def test_unknown_csr_traps(self, hart):
        """访问不存在的 CSR 地址 -> IllInstr."""
        hart.mode = RiscvMode.M
        instr = self._csrrw(rd=5, rs1=0, csr=0xFFF)  # 未定义
        hart.exec_instr(instr)
        assert hart.mcause_val == 2


class TestSstatusMstatusLinkage:
    """验证 sstatus (0x100) 是 mstatus (0x300) 的受限视图.

    RISC-V 规范: sstatus 读写必须通过 mstatus 对应位, 两者不可独立.
    """

    @pytest.fixture
    def hart(self) -> Hart:
        h = Hart(id=0)
        h.pc = 0x1000
        h.mode = RiscvMode.S
        return h

    # ---- SPP 位联动 ----

    def test_sstatus_read_spp_reflects_mstatus(self, hart):
        """读 sstatus.SPP 应返回 mstatus.SPP 的值."""
        # mstatus 初始 SPP=0
        assert (hart.read_csr(0x100) & MSTATUS_SPP) == 0
        # 写 mstatus.SPP=1
        hart.mstatus_val |= MSTATUS_SPP
        assert (hart.read_csr(0x100) & MSTATUS_SPP) != 0, "sstatus.SPP 应反映 mstatus.SPP 的值"

    def test_sstatus_write_spp_updates_mstatus(self, hart):
        """写 sstatus.SPP 应更新 mstatus.SPP."""
        hart.mstatus_val &= ~MSTATUS_SPP  # 清零
        # 通过 write_csr 写 sstatus = SPP
        hart.write_csr(0x100, MSTATUS_SPP)
        assert hart.mstatus_val & MSTATUS_SPP, "写 sstatus.SPP=1 应更新 mstatus.SPP"

    def test_sstatus_read_sie_reflects_mstatus(self, hart):
        """读 sstatus.SIE 应返回 mstatus.SIE 的值."""
        hart.mstatus_val |= MSTATUS_SIE
        assert (hart.read_csr(0x100) & MSTATUS_SIE) != 0, "sstatus.SIE 应反映 mstatus.SIE 的值"
        hart.mstatus_val &= ~MSTATUS_SIE
        assert (hart.read_csr(0x100) & MSTATUS_SIE) == 0

    def test_sstatus_write_sie_updates_mstatus(self, hart):
        """写 sstatus.SIE 应更新 mstatus.SIE."""
        hart.mstatus_val &= ~MSTATUS_SIE
        hart.write_csr(0x100, MSTATUS_SIE)
        assert hart.mstatus_val & MSTATUS_SIE, "写 sstatus.SIE=1 应更新 mstatus.SIE"

    def test_sstatus_write_does_not_leak_to_higher_bits(self, hart):
        """sstatus 写只影响受限视图内的位, 不影响 mstatus 高位 (如 MPP)."""
        original_mstatus = hart.mstatus_val
        # 写 sstatus 为全 1
        hart.write_csr(0x100, 0xFFFF_FFFF_FFFF_FFFF)
        # MPP (bits 11-12) 是 mstatus 独有的, 不应被 sstatus 写改变
        assert (hart.mstatus_val & MSTATUS_MPP) == (original_mstatus & MSTATUS_MPP), (
            "sstatus 写不应影响 mstatus.MPP 等高位字段"
        )

    def test_sstatus_read_excludes_mstatus_only_bits(self, hart):
        """读 sstatus 不应返回 MPP/MPIE 等 mstatus 专有位."""
        hart.mstatus_val = MSTATUS_MPP | MSTATUS_MPIE | MSTATUS_SPP | MSTATUS_SIE
        sstatus_val = hart.read_csr(0x100)
        assert (sstatus_val & MSTATUS_MPP) == 0, "sstatus 不应暴露 mstatus.MPP"
        assert (sstatus_val & MSTATUS_MPIE) == 0, "sstatus 不应暴露 mstatus.MPIE"
        assert (sstatus_val & MSTATUS_SPP) != 0, "sstatus 应包含 SPP"
        assert (sstatus_val & MSTATUS_SIE) != 0, "sstatus 应包含 SIE"

    # ---- 与 trap handler 集成 ----

    def test_spp_set_after_smode_trap(self, hart):
        """S 模式触发 ebreak -> sstatus.SPP 应 = 1 (来自 S 模式)."""
        hart.mode = RiscvMode.S
        hart.csrs["medeleg"].val = 1 << 3  # 委派 breakpoint
        hart.csrs["stvec"].val = 0x80004000
        hart.pc = 0x2000
        hart.mstatus_val = 0  # 清零 mstatus

        deliver_trap(hart, TrapType.Breakpoint, hart.pc, is_interrupt=False)

        # 陷阱后 mstatus.SPP 应为 1 (来自 S 模式)
        assert hart.mstatus_val & MSTATUS_SPP, "S 模式 ebreak 陷阱后 mstatus.SPP 必须为 1"
        # sstatus 读应反映 SPP=1
        assert hart.read_csr(0x100) & MSTATUS_SPP, "sstatus.SPP 应反映 mstatus.SPP=1"
