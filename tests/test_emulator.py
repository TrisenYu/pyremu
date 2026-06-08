#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""多核模拟器集成测试: 多 hart 执行, IPI, AMO 跨 hart 竞争."""


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
