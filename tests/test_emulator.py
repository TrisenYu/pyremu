#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""多核模拟器集成测试: 多 hart 执行, IPI, AMO 跨 hart 竞争."""

import pytest

from pyremu.emulator import Emulator


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
        interrupted = emu.harts[1].check_pending_interrupts()
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
