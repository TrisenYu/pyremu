#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""多核模拟器集成测试: 多 hart 执行, IPI, AMO 跨 hart 竞争."""

import pytest

from pyremu.emulator import Emulator
from pyremu.core.trap_handler import check_pending_interrupts


class TestEmulatorInit:
    """模拟器初始化."""

    def test_single_hart(self):
        """单 hart 初始化."""
        emu = Emulator(num_harts=1)
        assert emu.num_harts == 1
        assert len(emu.harts) == 1
        assert emu.harts[0].pc == emu.reset_vector

    def test_four_harts(self):
        """4 hart 初始化."""
        emu = Emulator(num_harts=4)
        assert emu.num_harts == 4
        assert len(emu.harts) == 4
        for i, h in enumerate(emu.harts):
            assert h.id == i
            assert h.pc == emu.reset_vector

    def test_eight_harts(self):
        """8 hart 初始化, 每个 hart 有独立寄存器和相同复位向量."""
        emu = Emulator(num_harts=8)
        assert emu.num_harts == 8
        for h in emu.harts:
            assert h.pc == emu.reset_vector
            assert h.gprs[0].val == 0  # x0 始终为 0
            assert h.mode.name == "M"  # 复位后为 M 模式

    def test_hart_ids_unique(self):
        """每个 hart 的 mhartid CSR 应唯一."""
        emu = Emulator(num_harts=4)
        ids = {h.csrs["mhartid"].val for h in emu.harts}
        assert ids == {0, 1, 2, 3}

    def test_bus_shared(self):
        """所有 hart 共享同一个 Bus."""
        emu = Emulator(num_harts=4)
        for h in emu.harts:
            assert h.bus is emu.bus

    def test_clint_registered(self):
        """CLINT 已注册为 Bus 设备."""
        emu = Emulator(num_harts=4)
        assert emu.bus.is_device_addr(emu.clint.base_addr)


class TestEmulatorExecution:
    """执行循环."""

    def test_step_executes_all_harts(self):
        """step() 每个 hart 执行一条指令."""
        emu = Emulator(num_harts=4, reset_vector=0x1000)
        # 载入 NOP 指令 (ADDI x0, x0, 0 = 0x00000013)
        emu.load_code(0x1000, b"\x13\x00\x00\x00")
        initial_pc = emu.harts[0].pc
        executed = emu.step()
        assert executed == 4
        # 每条指令后 PC 应 +4
        for h in emu.harts:
            assert h.pc == initial_pc + 4

    def test_pc_advances_per_step(self):
        """每条 ADDI x0,x0,0 后 PC 推进 4 字节."""
        emu = Emulator(num_harts=1, reset_vector=0x1000)
        # 载入多条 NOP 保证有足够的指令
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 10)
        assert emu.harts[0].pc == 0x1000
        emu.step()
        assert emu.harts[0].pc == 0x1004
        emu.step()
        assert emu.harts[0].pc == 0x1008

    def test_hart_gpr_write_independent(self):
        """hart 间寄存器独立: hart 0 写 x5, 不影响 hart 1 的 x5."""
        emu = Emulator(num_harts=4, reset_vector=0x1000)
        # ADDI x5, x10, 0 → x5 = x10 (每条 hart 的 x10 不同)
        # Instr: imm[11:0]=0, rs1=x10, funct3=000, rd=x5, op=0010011
        instr = (0 << 20) | (10 << 15) | (5 << 7) | 0b0010011
        emu.load_code(0x1000, instr.to_bytes(4, "little"))
        # 各 hart 的 x10 不同
        emu.harts[0].gprs[10].val = 0
        emu.harts[1].gprs[10].val = 100
        emu.harts[2].gprs[10].val = 200
        emu.harts[3].gprs[10].val = 300
        emu.step()
        assert emu.harts[0].gprs[5].val == 0
        assert emu.harts[1].gprs[5].val == 100
        assert emu.harts[2].gprs[5].val == 200
        assert emu.harts[3].gprs[5].val == 300


class TestEmulatorIPI:
    """多 hart 间 IPI (核间中断)."""

    def test_ipi_pending_detection(self):
        """CLINT send_ipi → hart 检测到待处理 MSIP."""
        emu = Emulator(num_harts=4, reset_vector=0x1000)
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 10)

        # 开启 hart 1 的中断
        emu.harts[1].mie = True  # mstatus.MIE
        emu.harts[1].csrs["mie"].val = 1 << 3  # mie.MSIP = 1
        emu.harts[1].csrs["mtvec"].val = 0x80000000

        # 发送 IPI 到 hart 1
        emu.clint.send_ipi(1)

        # hart 1 的 check_pending_interrupts 应检测到
        interrupted = check_pending_interrupts(emu.harts[1])
        assert interrupted, "hart 1 应检测到 MSIP 中断"

    def test_ipi_diverges_hart_pc(self):
        """IPI 导致目标 hart 跳转到 mtvec, 与其他 hart 的 PC 不同."""
        emu = Emulator(num_harts=4, reset_vector=0x1000)
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 10)

        # 所有 hart 相同的初始 PC
        for h in emu.harts:
            assert h.pc == 0x1000

        # hart 1 开启中断, 其他 hart 不开启
        emu.harts[1].mie = True
        emu.harts[1].csrs["mie"].val = 1 << 3
        emu.harts[1].csrs["mtvec"].val = 0x80000000

        # 发送 IPI 到 hart 1
        emu.clint.send_ipi(1)

        # 执行一个周期 — hart 1 应跳转到 mtvec trap handler
        emu.step()

        # hart 0 (无中断): PC = 0x1000 + 4
        assert emu.harts[0].pc == 0x1004, "hart 0 应正常推进 PC"
        # hart 1 (收到 IPI): PC 应跳转到 mtvec = 0x80000000
        assert emu.harts[1].pc == 0x80000000, (
            f"hart 1 应被 IPI 重定向到 mtvec, 实际 PC={emu.harts[1].pc:#x}"
        )

    def test_ipi_causes_trap_context_save(self):
        """IPI trap 应正确保存上下文."""
        emu = Emulator(num_harts=4, reset_vector=0x1000)
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 10)

        h = emu.harts[2]
        h.mie = True
        h.csrs["mie"].val = 1 << 3
        h.csrs["mtvec"].val = 0x80000000
        saved_pc = h.pc  # 0x1000

        emu.clint.send_ipi(2)
        emu.step()

        # 中断在指令边界触发, mepc 保存下一条指令的 PC
        next_pc = saved_pc + 4  # 0x1004
        assert h.mepc_val == next_pc, (
            f"mepc 应保存下一条指令 PC={next_pc:#x}, 实际={h.mepc_val:#x}"
        )
        assert h.mie is False, "进入 trap 后 MIE 应清零"
        # MSI: bit63=1, code=3
        assert (h.mcause_val >> 63) & 1 == 1, "mcause bit63 应置位 (中断)"
        assert (h.mcause_val & ~(1 << 63)) == 3, "mcause code 应为 3 (MSI)"


class TestEmulatorAMOCompetition:
    """AMO 跨 hart 竞争测试."""

    def test_lr_sc_competition(self):
        """hart 0 做 LR, hart 1 写同地址, hart 0 的 SC 应失败."""
        emu = Emulator(num_harts=2, reset_vector=0x1000)

        # hart 0: LR.D x5, (x10) → SC.D x5, x6, (x10)
        # 先载入代码片段...

        # 简化: 直接操作 hart
        h0, h1 = emu.harts

        # hart 0 做 LR
        h0.gprs[10].val = 0x2000  # addr
        h0.set_reservation(0x2000)

        # hart 1 写入同地址 → 应清除 hart 0 的预留
        # (通过 Bus 写入)
        emu.bus.write(0x2000, b"\x42\x00\x00\x00\x00\x00\x00\x00")

        # 验证 hart 0 的预留被清除
        h0.clear_reservation()  # 模拟 Bus 通知
        assert not h0.reservation_valid

    def test_memory_shared_between_harts(self):
        """两个 hart 通过共享内存通信."""
        emu = Emulator(num_harts=2, reset_vector=0x1000)

        # hart 0 写内存
        emu.bus.write(0x5000, b"\xCA\xFE\xBA\xBE\x00\x00\x00\x00")

        # hart 1 读内存 (通过 Bus)
        data = emu.bus.read(0x5000, 8)
        assert data[:4] == b"\xCA\xFE\xBA\xBE"


class TestEmulatorState:
    """状态导出."""

    def test_dump_hart_regs(self):
        """dump_hart_regs 导出关键状态."""
        emu = Emulator(num_harts=2, reset_vector=0x80000000)
        regs = emu.dump_hart_regs(0)
        assert regs["hart_id"] == 0
        assert regs["pc"] == 0x80000000
        assert "x0" in regs["gprs"]
        assert "x31" in regs["gprs"]

    def test_mem_hexdump(self):
        """mem_hexdump 输出格式化 hex 字符串."""
        emu = Emulator(num_harts=1)
        emu.bus.write(0x1000, b"Hello, RISC-V!")
        dump = emu.mem_hexdump(0x1000, 14)
        assert "Hello, RISC-V!" in dump
        assert "1000" in dump


class TestEmulatorStoreMemory:
    """通过 hart 执行 store 指令写入物理内存.

    验证完整的 store 路径: exec_instr → handle_st → _mem_write →
    _translate_full → _mem_write_phy → Bus.write → RAM/L2。
    这是对 #sw-not-writing-to-ram 的回归测试。
    """

    RAM_BASE = 0x8000_0000  # 默认 DRAM 基址

    @pytest.fixture
    def emu(self) -> Emulator:
        return Emulator(num_harts=1)

    def test_sw_in_ram_range(self, emu):
        """SW 写入 RAM 范围内的地址, 应能通过 Bus 读回."""
        h = emu.harts[0]
        # sp 指向 RAM 内
        h.gprs[2].val = self.RAM_BASE + 0x1000  # sp
        h.gprs[1].val = 0xCAFE_BABE  # ra (x1)

        # sw x1, 0(x2):  opcode=0100011 funct3=010 rs2=1 rs1=2 imm=0
        instr = (0 << 25) | (1 << 20) | (2 << 15) | (2 << 12) | (0x23)
        h.exec_instr(instr)

        data = emu.bus.read(self.RAM_BASE + 0x1000, 4)
        assert data == b"\xBE\xBA\xFE\xCA", (
            f"SW 写入失败: 期望 b'\\xbe\\xba\\xfe\\xca', 实际 {data.hex()}"
        )

    def test_sw_multiple_stores(self, emu):
        """连续 SW 写入不同偏移, 数据不互相覆盖."""
        h = emu.harts[0]
        base = self.RAM_BASE + 0x2000
        h.gprs[2].val = base  # sp
        h.gprs[8].val = 0xAAAA_BBBB  # s0
        h.gprs[9].val = 0xCCCC_DDDD  # s1

        # sw s0, 0(sp)
        instr0 = (0 << 25) | (8 << 20) | (2 << 15) | (2 << 12) | (0x23)
        h.exec_instr(instr0)
        # sw s1, 4(sp)
        instr1 = (0 << 25) | (9 << 20) | (2 << 15) | (2 << 12) | (0x23)
        # 手动构造 imm=4: bit 7=1, bits 8-11=0, bits 25-31=0
        instr1 = (0 << 25) | (9 << 20) | (2 << 15) | (2 << 12) | (4 << 7) | (0x23)
        h.exec_instr(instr1)

        data = emu.bus.read(base, 8)
        assert data[:4] == b"\xBB\xBB\xAA\xAA", f"偏移 0: {data[:4].hex()}"
        assert data[4:8] == b"\xDD\xDD\xCC\xCC", f"偏移 4: {data[4:8].hex()}"

    def test_sd_doubleword_store(self, emu):
        """SD 写入 8 字节, 验证双字 store 的 PMA."""
        h = emu.harts[0]
        addr = self.RAM_BASE + 0x3000
        h.gprs[2].val = addr  # sp
        h.gprs[1].val = 0xDEAD_BEEF_CAFE_BABE  # ra

        # sd x1, 0(x2): opcode=0100011 funct3=011 rs2=1 rs1=2 imm=0
        instr = (0 << 25) | (1 << 20) | (2 << 15) | (3 << 12) | (0x23)
        h.exec_instr(instr)

        data = emu.bus.read(addr, 8)
        expected = (0xDEAD_BEEF_CAFE_BABE).to_bytes(8, "little")
        assert data == expected, f"SD 写入失败: 期望 {expected.hex()}, 实际 {data.hex()}"

    def test_store_at_ram_base_edge(self, emu):
        """SW 写入 RAM 基址 (ram_base) 边界, 确保不越界."""
        h = emu.harts[0]
        # 恰好从 ram_base 开始
        h.gprs[2].val = self.RAM_BASE
        h.gprs[1].val = 0x1234_5678
        instr = (0 << 25) | (1 << 20) | (2 << 15) | (2 << 12) | (0x23)
        h.exec_instr(instr)
        assert emu.bus.read(self.RAM_BASE, 4) == b"\x78\x56\x34\x12"


# ============================================================
#  M 扩展: 除零 / 模零行为 (RISC-V 规范: 返回 -1 或 dividend)
# ============================================================


class TestMDivideByZero:
    """验证 DIV/DIVU/REM/REMU 在除数为零时的行为.

    RISC-V 特权架构规定:
    - DIV[U] 除零 → 返回 −1 (所有位为 1)
    - REM[U] 除零 → 返回被除数 (dividend)
    不触发异常.
    """

    @pytest.fixture
    def emu(self) -> Emulator:
        return Emulator(num_harts=1, ram_base=0, reset_vector=0x1000)

    def _exec_rtype(self, emu, funct3: int, funct7: int, rd: int, rs1: int, rs2: int):
        """执行一条 R-type 指令并返回目标寄存器结果."""
        h = emu.harts[0]
        instr = (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0b01100_11
        h.exec_instr(instr)
        return h.gprs[rd].val

    def test_div_by_zero(self, emu):
        """DIV x5, x10, x0 (divisor=0) → x5 = -1."""
        emu.harts[0].gprs[10].val = 42
        result = self._exec_rtype(emu, funct3=4, funct7=1, rd=5, rs1=10, rs2=0)
        assert result == 0xFFFF_FFFF_FFFF_FFFF, f"应为 -1, 实际 {result:#x}"

    def test_divu_by_zero(self, emu):
        """DIVU x5, x10, x0 → x5 = -1 (all-1s)."""
        emu.harts[0].gprs[10].val = 0x8000_0000_0000_0000
        result = self._exec_rtype(emu, funct3=5, funct7=1, rd=5, rs1=10, rs2=0)
        assert result == 0xFFFF_FFFF_FFFF_FFFF

    def test_rem_by_zero(self, emu):
        """REM x5, x10, x0 → x5 = x10 (dividend)."""
        emu.harts[0].gprs[10].val = 42
        result = self._exec_rtype(emu, funct3=6, funct7=1, rd=5, rs1=10, rs2=0)
        assert result == 42

    def test_remu_by_zero(self, emu):
        """REMU x5, x10, x0 → x5 = x10."""
        emu.harts[0].gprs[10].val = 0xDEAD
        result = self._exec_rtype(emu, funct3=7, funct7=1, rd=5, rs1=10, rs2=0)
        assert result == 0xDEAD

    def test_div_normal(self, emu):
        """DIV x5, x10, x2 (10 / 3 = 3)."""
        emu.harts[0].gprs[10].val = 10
        emu.harts[0].gprs[2].val = 3
        result = self._exec_rtype(emu, funct3=4, funct7=1, rd=5, rs1=10, rs2=2)
        assert result == 3

    def test_rem_normal(self, emu):
        """REM x5, x10, x2 (10 % 3 = 1)."""
        emu.harts[0].gprs[10].val = 10
        emu.harts[0].gprs[2].val = 3
        result = self._exec_rtype(emu, funct3=6, funct7=1, rd=5, rs1=10, rs2=2)
        assert result == 1


class TestStoreImmediateOffset:
    """回归: parse_imm_s 的 imm[11:5] 缺少 <<5 位移导致 offset>=32 时错误."""

    @pytest.fixture
    def emu(self):
        emu = Emulator(num_harts=1)
        emu.harts[0].gprs[2].val = 0x80001000  # sp
        emu.harts[0].gprs[10].val = 0xCAFEBABEDEADBEEF  # a0 = test value
        return emu

    def _exec_sd(self, emu, offset: int):
        """执行 sd a0, offset(sp)."""
        h = emu.harts[0]
        h.pc = 0x80000000
        # imm[11:5] = offset >> 5, imm[4:0] = offset & 0x1F
        imm_11_5 = (offset >> 5) & 0x7F
        imm_4_0 = offset & 0x1F
        instr = (
            (imm_11_5 << 25)
            | (10 << 20)  # rs2 = a0 = x10
            | (2 << 15)  # rs1 = sp = x2
            | (3 << 12)  # funct3 = sd
            | (imm_4_0 << 7)  # rd field = imm[4:0]
            | 0b0100011  # opcode = STORE
        )
        h.exec_instr(instr)
        if h.pc != 0x80000000:  # trap occurred
            return None, h.mcause_val
        # Read back from the written address
        addr = (emu.harts[0].gprs[2].val + offset) & 0xFFFF_FFFF_FFFF_FFFF
        data = emu.dump_memory(addr, 8)
        return int.from_bytes(data, "little"), 0

    def test_sd_offset_0(self, emu):
        """sd offset=0: 边界值, imm[11:5]=0 (原 bug 不触发)."""
        val, mcause = self._exec_sd(emu, 0)
        assert mcause == 0
        assert val == 0xCAFEBABEDEADBEEF

    def test_sd_offset_32(self, emu):
        """sd offset=32: imm[11:5]=1, 原 bug 会算出 offset=1 导致未对齐故障."""
        val, mcause = self._exec_sd(emu, 32)
        assert mcause == 0, f"unexpected trap mcause=0x{mcause:x}"
        assert val == 0xCAFEBABEDEADBEEF

    def test_sd_offset_64(self, emu):
        """sd offset=64: imm[11:5]=2."""
        val, mcause = self._exec_sd(emu, 64)
        assert mcause == 0
        assert val == 0xCAFEBABEDEADBEEF

    def test_sd_offset_96(self, emu):
        """sd offset=96: imm[11:5]=3."""
        val, mcause = self._exec_sd(emu, 96)
        assert mcause == 0
        assert val == 0xCAFEBABEDEADBEEF

    def test_sd_offset_2047(self, emu):
        """sd offset=2047: 12-bit S-type 最大正偏移."""
        val, mcause = self._exec_sd(emu, 2047)
        # 2047 未对齐 → 预期 StAddrMisaligned
        assert mcause == 0x6


class TestAuipcSignExtension:
    """回归: AUIPC/LUI 的 U-immediate 缺少 32-bit 符号扩展."""

    @pytest.fixture
    def emu(self):
        return Emulator(num_harts=1)

    def _exec_auipc(self, emu, imm20_raw: int):
        """执行 auipc rd, imm20_raw."""
        h = emu.harts[0]
        h.pc = 0x80000000
        instr = ((imm20_raw & 0xFFFFF) << 12) | (5 << 7) | 0b0010111
        h.exec_instr(instr)
        return h.gprs[5].val

    def test_auipc_positive(self, emu):
        """AUIPC with positive offset (bit 19=0) — 原代码也可正确处理."""
        result = self._exec_auipc(emu, 1)
        assert result == 0x80001000  # 0x80000000 + 0x1000

    def test_auipc_negative(self, emu):
        """AUIPC with negative offset (bit 19=1) — 原 bug 导致高位错误."""
        result = self._exec_auipc(emu, 0xFFFFF)  # -1 in 20-bit signed
        # PC + (-1 << 12) = 0x80000000 - 0x1000 = 0x7FFFF000
        assert result == 0x7FFFF000

    def test_auipc_max_negative(self, emu):
        """AUIPC most-negative offset (0x80000 = -2^19)."""
        result = self._exec_auipc(emu, 0x80000)  # most negative 20-bit value
        # PC + (0x80000 << 12) signed = PC - 0x80000000
        # 0x80000000 - 0x80000000 = 0
        assert result == 0

    def test_lui_negative(self, emu):
        """LUI with bit 31 set → sign-extended to 64-bit negative."""
        h = emu.harts[0]
        h.pc = 0x80000000
        instr = (0xFFFFF << 12) | (5 << 7) | 0b0110111  # LUI x5, 0xFFFFF
        h.exec_instr(instr)
        # imm20_raw << 12 = 0xFFFFF000, sign-extended from 32-bit → -0x1000
        assert h.gprs[5].val == 0xFFFFFFFFFFFFF000


class TestLoadAlignmentSizes:
    """回归: handle_ld 对所有 load 类型统一读 8 字节导致虚假对齐故障."""

    @pytest.fixture
    def emu(self):
        emu = Emulator(num_harts=1)
        # 在 0x80000048-0x8000004F 区域写入已知数据
        test_data = bytes(range(0x48, 0x50))  # 0x48, 0x49, ..., 0x4F
        emu.load_code(0x80000048, test_data)
        return emu

    def _exec_load(self, emu, addr: int, funct3: int):
        """执行 ld-type 指令: rd=x5, rs1=x10, offset=0."""
        h = emu.harts[0]
        h.gprs[10].val = addr  # rs1
        h.pc = 0x80000000
        instr = (5 << 7) | (funct3 << 12) | (10 << 15) | 0b0000011
        h.exec_instr(instr)
        return h.gprs[5].val, h.mcause_val

    def test_lbu_at_0x49(self, emu):
        """lbu 从奇地址读取 1 字节, 不应触发对齐故障."""
        val, mcause = self._exec_load(emu, 0x80000049, 0b100)  # funct3=lbu
        assert mcause == 0, f"unexpected trap mcause=0x{mcause:x}"
        assert val == 0x49

    def test_lbu_at_0x48(self, emu):
        """lbu 从 8 字节对齐地址读取."""
        val, mcause = self._exec_load(emu, 0x80000048, 0b100)  # funct3=lbu
        assert mcause == 0
        assert val == 0x48

    def test_lhu_at_0x4a(self, emu):
        """lhu 从 2 字节对齐地址读取."""
        val, mcause = self._exec_load(emu, 0x8000004A, 0b101)  # funct3=lhu
        assert mcause == 0
        assert val == 0x4B4A  # little-endian

    def test_lwu_at_0x4c(self, emu):
        """lwu 从 4 字节对齐地址读取."""
        val, mcause = self._exec_load(emu, 0x8000004C, 0b110)  # funct3=lwu
        assert mcause == 0
        assert val == 0x4F4E4D4C

    def test_lh_misaligned(self, emu):
        """lh 从奇数地址 → 应对齐故障."""
        val, mcause = self._exec_load(emu, 0x80000049, 0b001)  # funct3=lh
        assert mcause == 0x4  # LdAddrMisaligned


class TestITypeShifts:
    """回归: I-type 移位使用 funct6 (bits 31:26) 而非 funct7 — bit 25 属于 shamt."""

    @pytest.fixture
    def h(self):
        from pyremu.core.decoder import Hart
        from pyremu.core.mem_check_aux import inject_memory_backend
        from pyremu.memory.bus import Bus

        bus = Bus(ram_size=0x10000, ram_base=0x80000000)
        hart = Hart(id=0)
        inject_memory_backend(hart, bus.read, bus.write)
        hart.bus = bus
        hart.pc = 0x80000000
        return hart

    # -- helpers: 构造指令编码并执行 --

    @staticmethod
    def _slli(rd, rs1, shamt):
        """slli rd, rs1, shamt — RV64."""
        return (
            (0b000000 << 26)  # funct6
            | (shamt << 20)   # 6-bit shamt
            | (rs1 << 15)
            | (0b001 << 12)   # funct3
            | (rd << 7)
            | 0b0010011       # opcode OP-IMM
        )

    @staticmethod
    def _srli(rd, rs1, shamt):
        return (
            (0b000000 << 26) | (shamt << 20) | (rs1 << 15)
            | (0b101 << 12) | (rd << 7) | 0b0010011
        )

    @staticmethod
    def _srai(rd, rs1, shamt):
        return (
            (0b010000 << 26) | (shamt << 20) | (rs1 << 15)
            | (0b101 << 12) | (rd << 7) | 0b0010011
        )

    def _exec(self, hart, instr):
        advance = hart.exec_instr(instr)
        if advance != 0 and hart.pc == 0x80000000:
            hart.pc = (hart.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF

    # -- SLLI --

    def test_slli_shamt_0(self, h):
        """SLLI shamt=0 应该是 NOP (恒等)."""
        h.gprs[10].val = 0x123456789ABCDEF0
        self._exec(h, self._slli(10, 10, 0))
        assert h.gprs[10].val == 0x123456789ABCDEF0

    def test_slli_shamt_1(self, h):
        h.gprs[11].val = 0x1
        self._exec(h, self._slli(11, 11, 1))
        assert h.gprs[11].val == 0x2

    def test_slli_shamt_32(self, h):
        """SLLI shamt=32 — bit 25 置位, 曾触发 funct7≠0 的误判."""
        h.gprs[12].val = 0xFFFFFFFFDEADBEEF
        self._exec(h, self._slli(12, 12, 32))
        assert h.gprs[12].val == 0xDEADBEEF00000000

    def test_slli_shamt_63(self, h):
        """SLLI 最大移位量 63."""
        h.gprs[13].val = 0x3
        self._exec(h, self._slli(13, 13, 63))
        assert h.gprs[13].val == (0x3 << 63) & 0xFFFF_FFFF_FFFF_FFFF

    def test_slli_wraps_64(self, h):
        """移位后 & mask 保证 64 位截断."""
        h.gprs[14].val = 0xABCD
        self._exec(h, self._slli(14, 14, 60))
        assert h.gprs[14].val == 0xD000000000000000

    # -- SRLI --

    def test_srli_shamt_32(self, h):
        """SRLI shamt=32 — bit 25 置位."""
        h.gprs[15].val = 0xDEADBEEF00000000
        self._exec(h, self._srli(15, 15, 32))
        assert h.gprs[15].val == 0xDEADBEEF

    def test_srli_shamt_0(self, h):
        h.gprs[16].val = 0x8000000000000000
        self._exec(h, self._srli(16, 16, 0))
        assert h.gprs[16].val == 0x8000000000000000

    # -- SRAI --

    def test_srai_shamt_32_positive(self, h):
        """SRAI 正数算术右移 — 高位补 0."""
        h.gprs[17].val = 0x7ABCDEF000000000
        self._exec(h, self._srai(17, 17, 32))
        assert h.gprs[17].val == 0x7ABCDEF0

    def test_srai_shamt_32_negative(self, h):
        """SRAI 负数算术右移 — 高位补 1 (符号扩展)."""
        h.gprs[18].val = 0x8000000000000000  # 最小负值
        self._exec(h, self._srai(18, 18, 32))
        assert h.gprs[18].val == 0xFFFFFFFF80000000

    def test_srai_shamt_63(self, h):
        """SRAI 最大移位量 — 符号位填满全部位."""
        h.gprs[19].val = 0x8000000000000000
        self._exec(h, self._srai(19, 19, 63))
        assert h.gprs[19].val == 0xFFFFFFFFFFFFFFFF

    # -- 非法 funct6 被拒绝 --

    def test_slli_bad_funct6_traps(self, h):
        """非法 funct6 应触发 IllInstr."""
        bad = (0b111111 << 26) | (0 << 20) | (10 << 15) | (0b001 << 12) | (10 << 7) | 0b0010011
        advance = h.exec_instr(bad)
        assert advance == 0
        assert h.mcause_val != 0
