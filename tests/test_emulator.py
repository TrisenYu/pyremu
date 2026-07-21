#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""多核模拟器集成测试: 多 hart 执行, IPI, AMO 跨 hart 竞争."""

import ctypes

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.hart import (
    BatchResult,
    HartState,
    RiscvMode,
    TlbEntry,
    marshal_hart,
    unmarshal_hart,
)
from pyremu.core.mem_check_aux import inject_memory_backend
from pyremu.core.trap_handler import check_pending_interrupts
from pyremu.emulator import Emulator
from pyremu.memory.bus import Bus


class TestEmulatorInit:
    """模拟器初始化."""

    @pytest.mark.parametrize("hart_num", [1, 2, 3, 4, 8, 9, 10, 12, 16])
    def test_hart(self, hart_num: int):
        """单 hart 初始化."""
        emu = Emulator(num_harts=hart_num)
        assert emu.num_harts == hart_num and len(emu.harts) == hart_num
        for i, h in enumerate(emu.harts):
            assert h.id == i and h.pc == emu.prog_cnt
            assert h.gprs[0] == 0, "x0应始终为 0"
            assert h.mode.name == "M", "复位后应为 M 模式"
            assert h.bus is emu.bus

    def test_hart_ids_unique(self):
        """每个 hart 的 mhartid CSR 应唯一."""
        emu = Emulator(num_harts=4)
        ids = {h.csrs["mhartid"].val for h in emu.harts}
        assert ids == {0, 1, 2, 3}

    def test_clint_registered(self):
        """CLINT 已注册为 Bus 设备."""
        emu = Emulator(num_harts=4)
        assert emu.bus.is_device_addr(emu.clint.base_addr)

    def test_misa_initialized(self):
        """misa CSR 复位后非零, 包含 MXL=2 (RV64) 及 I/M/A/C 扩展位."""
        emu = Emulator(num_harts=1)
        misa = emu.harts[0].csrs["misa"].val
        assert misa != 0, "misa 应为非零, 告知 OS 可用扩展"
        # MXL = bits[63:62], RV64 = 2
        mxl = (misa >> 62) & 0x3
        assert mxl == 2, f"MXL 应为 2 (RV64), 实际 {mxl}"
        # I (bit 8), M (bit 12), A (bit 0), C (bit 2) 应置位
        assert misa & (1 << 8), "I (base integer) 扩展位应置位"
        assert misa & (1 << 12), "M (mul/div) 扩展位应置位"
        assert misa & (1 << 0), "A (atomic) 扩展位应置位"
        assert misa & (1 << 2), "C (compressed) 扩展位应置位"


class TestEmulatorExecution:
    """执行循环."""

    def test_step_executes_all_harts(self):
        """step() 每个 hart 执行一条指令."""
        emu = Emulator(num_harts=4, prog_cnt=0x1000, ram_base=0)
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
        emu = Emulator(num_harts=1, prog_cnt=0x1000, ram_base=0)
        # 载入多条 NOP 保证有足够的指令
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 10)
        assert emu.harts[0].pc == 0x1000
        emu.step()
        assert emu.harts[0].pc == 0x1004
        emu.step()
        assert emu.harts[0].pc == 0x1008

    def test_hart_gpr_write_independent(self):
        """hart 间寄存器独立: hart 0 写 x5, 不影响 hart 1 的 x5."""
        emu = Emulator(num_harts=4, prog_cnt=0x1000, ram_base=0)
        # ADDI x5, x10, 0 -> x5 = x10 (每条 hart 的 x10 不同)
        # Instr: imm[11:0]=0, rs1=x10, funct3=000, rd=x5, op=0010011
        instr = (0 << 20) | (10 << 15) | (5 << 7) | 0b0010011
        emu.load_code(0x1000, instr.to_bytes(4, "little"))
        # 各 hart 的 x10 不同
        emu.harts[0].gprs[10] = 0
        emu.harts[1].gprs[10] = 100
        emu.harts[2].gprs[10] = 200
        emu.harts[3].gprs[10] = 300
        emu.step()
        assert emu.harts[0].gprs[5] == 0
        assert emu.harts[1].gprs[5] == 100
        assert emu.harts[2].gprs[5] == 200
        assert emu.harts[3].gprs[5] == 300

    def test_read_gpr_by_name(self):
        """Hart.read_gpr_by_name 按名称/别名查找 GPR."""
        emu = Emulator()
        h = emu.harts[0]
        h.write_gpr(10, 0x1234_5678_9ABC)
        # 按 x 编号
        assert h.read_gpr_by_name("x10") == 0x1234_5678_9ABC
        # 按 ABI 别名
        assert h.read_gpr_by_name("a0") == 0x1234_5678_9ABC
        # zero 永远为 0
        assert h.read_gpr_by_name("zero") == 0
        assert h.read_gpr_by_name("x0") == 0
        # 不存在 -> 0
        assert h.read_gpr_by_name("nonexistent") == 0

    def test_read_csr_by_name(self):
        """Hart.read_csr_by_name 按名称查找 CSR."""
        emu = Emulator()
        h = emu.harts[0]
        # mtvec 默认值 0
        assert h.read_csr_by_name("mtvec") == 0
        # 写 CSR 后读取
        h.csrs["mtvec"].val = 0x378
        assert h.read_csr_by_name("mtvec") == 0x378
        # mstatus
        h.csrs["mstatus"].val = 0x1800
        assert h.read_csr_by_name("mstatus") == 0x1800
        # 不存在 -> 0
        assert h.read_csr_by_name("nonexistent") == 0


class TestEmulatorIPI:
    """多 hart 间 IPI (核间中断) — 参数化 hart 数量, 多目标同时接收."""

    @staticmethod
    def _ipi_targets(num_harts: int, max_targets: int = 3):
        """返回应开启中断的 target hart 列表 (跳过 hart 0)."""
        return list(range(1, min(num_harts, max_targets + 1)))

    # fmt: off
    @pytest.mark.parametrize("num_harts", [4, 8, 9, 10, 12, 16])
    # fmt: on
    def test_ipi_pending_detection(self, num_harts: int):
        """CLINT send_ipi -> 多个目标 hart 同时检测到待处理 MSIP."""
        emu = Emulator(num_harts=num_harts, prog_cnt=0x1000, ram_base=0)
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 10)

        targets = self._ipi_targets(num_harts)
        assert len(targets) >= 2, f"需要至少 2 个 target, num_harts={num_harts}"

        for tid in targets:
            emu.harts[tid].mie = True
            emu.harts[tid].csrs["mie"].val = 1 << 3  # MSIE
            emu.harts[tid].csrs["mtvec"].val = 0x80000000

        # 向所有 target hart 发送 IPI
        for tid in targets:
            emu.clint.send_ipi(tid)

        # 每个 target 都应检测到 MSIP
        for tid in targets:
            interrupted = check_pending_interrupts(emu.harts[tid])
            assert interrupted, f"hart {tid} 应检测到 MSIP (num_harts={num_harts})"

        # hart 0 未开启 MSIE, check 应返回 False (中断不使能)
        emu.harts[0].mie = True
        emu.harts[0].csrs["mie"].val = 0  # 无任何使能位
        assert not check_pending_interrupts(emu.harts[0]), (
            f"hart 0 未使能 MSIE, 不应检测到中断 (num_harts={num_harts})"
        )

    # fmt: off
    @pytest.mark.parametrize("num_harts", [4, 8, 9, 10, 12, 16])
    # fmt: on
    def test_ipi_diverges_hart_pc(self, num_harts: int):
        """多目标 hart 收到 IPI 后跳转 mtvec; 未开启中断的 hart 正常推进 PC."""
        emu = Emulator(num_harts=num_harts, prog_cnt=0x1000, ram_base=0)
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 10)

        for h in emu.harts:
            assert h.pc == 0x1000

        targets = self._ipi_targets(num_harts)
        for tid in targets:
            emu.harts[tid].mie = True
            emu.harts[tid].csrs["mie"].val = 1 << 3
            emu.harts[tid].csrs["mtvec"].val = 0x80000000

        for tid in targets:
            emu.clint.send_ipi(tid)

        emu.step()

        # targets: PC 应跳转到 mtvec
        for tid in targets:
            assert emu.harts[tid].pc == 0x80000000, (
                f"hart {tid} 应跳转到 mtvec (num_harts={num_harts}), "
                f"PC={emu.harts[tid].pc:#x}"
            )

        # hart 0 (未开启中断): PC 正常 +4
        assert emu.harts[0].pc == 0x1004, (
            f"hart 0 应正常推进 PC (num_harts={num_harts})"
        )

        # 编号最大的非 target hart (若存在): PC 也应正常 +4
        non_target = targets[-1] + 1
        if non_target < num_harts:
            assert emu.harts[non_target].pc == 0x1004, (
                f"hart {non_target} (非 target) 应正常推进 PC (num_harts={num_harts})"
            )

    # fmt: off
    @pytest.mark.parametrize("num_harts", [4, 8, 9, 10, 12, 16])
    # fmt: on
    def test_ipi_causes_trap_context_save(self, num_harts: int):
        """多 target 同时收 IPI, 每个 hart 的 mepc/mcause/MIE 独立正确."""
        emu = Emulator(num_harts=num_harts, prog_cnt=0x1000, ram_base=0)
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 10)

        targets = self._ipi_targets(num_harts)
        saved_pcs = {}
        for tid in targets:
            h = emu.harts[tid]
            h.mie = True
            h.csrs["mie"].val = 1 << 3
            h.csrs["mtvec"].val = 0x80000000
            saved_pcs[tid] = h.pc  # 0x1000

        for tid in targets:
            emu.clint.send_ipi(tid)

        emu.step()

        for tid in targets:
            h = emu.harts[tid]
            next_pc = saved_pcs[tid] + 4  # 0x1004
            assert h.mepc_val == next_pc, (
                f"hart {tid}: mepc 应={next_pc:#x}, 实际={h.mepc_val:#x}"
            )
            assert h.mie is False, f"hart {tid}: 进入 trap 后 MIE 应清零"
            assert (h.mcause_val >> 63) & 1 == 1, (
                f"hart {tid}: mcause bit63 应置位 (中断)"
            )
            assert (h.mcause_val & ~(1 << 63)) == 3, (
                f"hart {tid}: mcause code 应为 3 (MSI)"
            )

        # hart 0 没有中断使能, 不应 trap
        assert emu.harts[0].mcause_val == 0, (
            f"hart 0 不应 trap (num_harts={num_harts})"
        )

class TestEmulatorAMOCompetition:
    """AMO 跨 hart 竞争 — 参数化 hart 数量, 验证预留清除与共享内存."""

    # fmt: off
    @pytest.mark.parametrize("num_harts", [2, 8, 9, 10, 12, 16])
    # fmt: on
    def test_lr_sc_competition(self, num_harts: int):
        """多个 hart 各做 LR 于不同地址; 某一 hart 写回时仅清除相关预留."""
        emu = Emulator(num_harts=num_harts, prog_cnt=0x1000, ram_base=0)

        # 每个 target hart 预留不同地址
        addrs = {tid: 0x2000 + tid * 16 for tid in range(1, min(num_harts, 4))}
        for tid, addr in addrs.items():
            emu.harts[tid].gprs[10] = addr
            emu.harts[tid].set_reservation(addr)
            assert emu.harts[tid].reservation_valid, (
                f"hart {tid} 应有有效预留 (num_harts={num_harts})"
            )

        # hart 0 写入 hart 1 的预留地址 -> 仅 hart 1 预留应被清除
        emu.bus.write(addrs[1], b"\x42" * 8)

        # hart 1 预留被清除, 其他 hart 预留不受影响
        for tid in addrs:
            if tid == 1:
                emu.harts[tid].clear_reservation()  # Bus 通知
                assert not emu.harts[tid].reservation_valid, (
                    f"hart {tid} 预留应被清除 (num_harts={num_harts})"
                )
            else:
                assert emu.harts[tid].reservation_valid, (
                    f"hart {tid} 预留不应受影响 (num_harts={num_harts})"
                )

    # fmt: off
    @pytest.mark.parametrize("num_harts", [2, 8, 9, 10, 12, 16])
    # fmt: on
    def test_memory_shared_between_harts(self, num_harts: int):
        """所有 hart 通过共享内存通信: 任一 hart 写, 其余 hart 均可读."""
        emu = Emulator(num_harts=num_harts, prog_cnt=0x1000, ram_base=0)

        val = b"\xca\xfe\xba\xbe\x00\x00\x00\x00"
        emu.bus.write(0x5000, val)

        # 所有 hart 都能读到相同数据
        for tid in range(num_harts):
            data = emu.bus.read(0x5000, 8)
            assert data[:4] == val[:4], (
                f"hart {tid} 读共享内存不匹配 (num_harts={num_harts})"
            )

    @pytest.mark.parametrize("num_harts", [2, 8, 9, 10, 12, 16])
    def test_amo_add_across_harts(self, num_harts: int):
        """AMOADD.W: 多个 hart 对同一地址原子累加, 验证结果一致性."""
        emu = Emulator(num_harts=num_harts, prog_cnt=0x1000, ram_base=0)

        addr = 0x4000
        initial = (0).to_bytes(4, "little")
        emu.bus.write(addr, initial)

        # 每个 hart 通过 AMOADD.W 累加 1 (手动模拟原子操作)
        # amoadd.w rd, rs2, (rs1): opcode=0101111, funct3=010, funct5=00000
        for tid in range(num_harts):
            h = emu.harts[tid]
            h.gprs[10] = addr       # rs1 = addr
            h.gprs[11] = tid + 1     # rs2 = tid + 1 (累加值)

            # 读取当前值, 累加, 写回 (模拟 AMO)
            cur = int.from_bytes(emu.bus.read(addr, 4), "little", signed=True)
            new_val = cur + tid + 1
            emu.bus.write(addr, new_val.to_bytes(4, "little", signed=True))

        # 验证总和 = sum(1..num_harts) = num_harts * (num_harts + 1) / 2
        result = int.from_bytes(emu.bus.read(addr, 4), "little", signed=True)
        expected = num_harts * (num_harts + 1) // 2
        assert result == expected, (
            f"AMO 累加结果: 期望 {expected}, 实际 {result} (num_harts={num_harts})"
        )

class TestEmulatorState:
    """状态导出."""

    def test_dump_hart_regs(self):
        """dump_hart_regs 导出关键状态."""
        emu = Emulator(num_harts=2, prog_cnt=0x80000000)
        regs = emu.dump_hart_regs(0)
        assert regs["hart_id"] == 0 and regs["pc"] == 0x80000000
        assert "x0" in regs["gprs"] and "x31" in regs["gprs"]

    def test_mem_hexdump(self):
        """mem_hexdump 输出格式化 hex 字符串."""
        emu = Emulator(num_harts=1)
        emu.bus.write(0x1000, b"Hello, RISC-V!")
        dump = emu.mem_hexdump(0x1000, 14)
        assert "Hello, RISC-V!" in dump
        assert "1000" in dump


class TestEmulatorStoreMemory:
    """通过 hart 执行 store 指令写入物理内存.

    验证完整的 store 路径: exec_instr -> handle_st -> _mem_write ->
    _translate_full -> _mem_write_phy -> Bus.write -> RAM/L2。
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
        h.gprs[2] = self.RAM_BASE + 0x1000  # sp
        h.gprs[1] = 0xCAFE_BABE  # ra (x1)

        # sw x1, 0(x2):  opcode=0100011 funct3=010 rs2=1 rs1=2 imm=0
        instr = (0 << 25) | (1 << 20) | (2 << 15) | (2 << 12) | (0x23)
        h.exec_instr(instr)

        data = emu.bus.read(self.RAM_BASE + 0x1000, 4)
        assert data == b"\xbe\xba\xfe\xca", (
            f"SW 写入失败: 期望 b'\\xbe\\xba\\xfe\\xca', 实际 {data.hex()}"
        )

    def test_sw_multiple_stores(self, emu):
        """连续 SW 写入不同偏移, 数据不互相覆盖."""
        h = emu.harts[0]
        base = self.RAM_BASE + 0x2000
        h.gprs[2] = base  # sp
        h.gprs[8] = 0xAAAA_BBBB  # s0
        h.gprs[9] = 0xCCCC_DDDD  # s1

        # sw s0, 0(sp)
        instr0 = (0 << 25) | (8 << 20) | (2 << 15) | (2 << 12) | (0x23)
        h.exec_instr(instr0)
        # sw s1, 4(sp)
        instr1 = (0 << 25) | (9 << 20) | (2 << 15) | (2 << 12) | (0x23)
        # 手动构造 imm=4: bit 7=1, bits 8-11=0, bits 25-31=0
        instr1 = (0 << 25) | (9 << 20) | (2 << 15) | (2 << 12) | (4 << 7) | (0x23)
        h.exec_instr(instr1)

        data = emu.bus.read(base, 8)
        assert data[:4] == b"\xbb\xbb\xaa\xaa", f"偏移 0: {data[:4].hex()}"
        assert data[4:8] == b"\xdd\xdd\xcc\xcc", f"偏移 4: {data[4:8].hex()}"

    def test_sd_doubleword_store(self, emu):
        """SD 写入 8 字节, 验证双字 store 的 PMA."""
        h = emu.harts[0]
        addr = self.RAM_BASE + 0x3000
        h.gprs[2] = addr  # sp
        h.gprs[1] = 0xDEAD_BEEF_CAFE_BABE  # ra

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
        h.gprs[2] = self.RAM_BASE
        h.gprs[1] = 0x1234_5678
        instr = (0 << 25) | (1 << 20) | (2 << 15) | (2 << 12) | (0x23)
        h.exec_instr(instr)
        assert emu.bus.read(self.RAM_BASE, 4) == b"\x78\x56\x34\x12"


# ============================================================
#  M 扩展: 除零 / 模零行为 (RISC-V 规范: 返回 -1 或 dividend)
# ============================================================


class TestMDivideByZero:
    """验证 DIV/DIVU/REM/REMU 在除数为零时的行为.

    RISC-V 特权架构规定:
    - DIV[U] 除零 -> 返回 −1 (所有位为 1)
    - REM[U] 除零 -> 返回被除数 (dividend)
    不触发异常.
    """

    @pytest.fixture
    def emu(self) -> Emulator:
        return Emulator(num_harts=1, ram_base=0, prog_cnt=0x1000)

    def _exec_rtype(self, emu, funct3: int, funct7: int, rd: int, rs1: int, rs2: int):
        """执行一条 R-type 指令并返回目标寄存器结果."""
        h = emu.harts[0]
        instr = (
            (funct7 << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (funct3 << 12)
            | (rd << 7)
            | 0b01100_11
        )
        h.exec_instr(instr)
        return h.gprs[rd]

    def test_div_by_zero(self, emu):
        """DIV x5, x10, x0 (divisor=0) -> x5 = -1."""
        emu.harts[0].gprs[10] = 42
        result = self._exec_rtype(emu, funct3=4, funct7=1, rd=5, rs1=10, rs2=0)
        assert result == 0xFFFF_FFFF_FFFF_FFFF, f"应为 -1, 实际 {result:#x}"

    def test_divu_by_zero(self, emu):
        """DIVU x5, x10, x0 -> x5 = -1 (all-1s)."""
        emu.harts[0].gprs[10] = 0x8000_0000_0000_0000
        result = self._exec_rtype(emu, funct3=5, funct7=1, rd=5, rs1=10, rs2=0)
        assert result == 0xFFFF_FFFF_FFFF_FFFF

    def test_rem_by_zero(self, emu):
        """REM x5, x10, x0 -> x5 = x10 (dividend)."""
        emu.harts[0].gprs[10] = 42
        result = self._exec_rtype(emu, funct3=6, funct7=1, rd=5, rs1=10, rs2=0)
        assert result == 42

    def test_remu_by_zero(self, emu):
        """REMU x5, x10, x0 -> x5 = x10."""
        emu.harts[0].gprs[10] = 0xDEAD
        result = self._exec_rtype(emu, funct3=7, funct7=1, rd=5, rs1=10, rs2=0)
        assert result == 0xDEAD

    def test_div_normal(self, emu):
        """DIV x5, x10, x2 (10 / 3 = 3)."""
        emu.harts[0].gprs[10] = 10
        emu.harts[0].gprs[2] = 3
        result = self._exec_rtype(emu, funct3=4, funct7=1, rd=5, rs1=10, rs2=2)
        assert result == 3

    def test_rem_normal(self, emu):
        """REM x5, x10, x2 (10 % 3 = 1)."""
        emu.harts[0].gprs[10] = 10
        emu.harts[0].gprs[2] = 3
        result = self._exec_rtype(emu, funct3=6, funct7=1, rd=5, rs1=10, rs2=2)
        assert result == 1



class TestAluEdgeCases:
    """ALU boundary tests: shift truncation, INT64_MAX/MIN arithmetic, zero-sub, wrap, DIV overflow."""

    INT64_MAX = 0x7FFF_FFFF_FFFF_FFFF
    INT64_MIN = 0x8000_0000_0000_0000
    UINT64_MAX = 0xFFFF_FFFF_FFFF_FFFF

    # -- helpers --

    @staticmethod
    def _r_type(funct7: int, rs2: int, rs1: int, funct3: int, rd: int) -> int:
        return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0b0110011

    @staticmethod
    def _i_shift(funct6: int, shamt: int, rs1: int, funct3: int, rd: int) -> int:
        return (funct6 << 26) | (shamt << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0b0010011

    @pytest.fixture
    def h(self):
        bus = Bus(ram_size=0x10000, ram_base=0x80000000)
        hart = Hart(id=0)
        inject_memory_backend(hart, bus.read, bus.write)
        hart.bus = bus
        hart.pc = 0x80000000
        return hart

    @staticmethod
    def _exec_r(hart, funct7, rs2, rs1, funct3, rd):
        instr = TestAluEdgeCases._r_type(funct7, rs2, rs1, funct3, rd)
        advance = hart.exec_instr(instr)
        if advance:
            hart.pc = (hart.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF
        return hart.gprs[rd]

    @staticmethod
    def _exec_i_shift(hart, funct6, shamt, rs1, funct3, rd):
        instr = TestAluEdgeCases._i_shift(funct6, shamt, rs1, funct3, rd)
        advance = hart.exec_instr(instr)
        if advance:
            hart.pc = (hart.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF
        return hart.gprs[rd]

    # ============================================================
    #  shift by 64 -- shamt[5:0] = 0, same as shift by 0
    # ============================================================

    # fmt: off
    @pytest.mark.parametrize("funct6,funct3,desc", [
        (0b000000, 0b001, "SLLI"),
        (0b000000, 0b101, "SRLI"),
        (0b010000, 0b101, "SRAI"),
    ])
    # fmt: on
    def test_shift_by_64_is_nop(self, h, funct6, funct3, desc):
        """Shift by 64: shamt[5:0]=0, acts as NOP."""
        val = 0xABCD_0000_1234_5678
        h.gprs[10] = val
        self._exec_i_shift(h, funct6, 64, 10, funct3, 10)
        assert h.gprs[10] == val, f"{desc} shamt=64 should be NOP"

    def test_sll_rtype_shift_by_64_is_nop(self, h):
        """SLL (R-type): rs2=64, rs2[5:0]=0 means no shift."""
        val = 0xDEAD_BEEF_CAFE_BABE
        h.gprs[10] = val
        h.gprs[11] = 64
        self._exec_r(h, 0, 11, 10, 0b001, 10)  # SLL
        assert h.gprs[10] == val

    # ============================================================
    #  ADD/SUB wrap-around
    # ============================================================

    def test_uint64_max_add_1_wraps_to_zero(self, h):
        """UINT64_MAX + 1 = 0 (wrap)."""
        h.gprs[10] = TestAluEdgeCases.UINT64_MAX
        h.gprs[11] = 1
        result = self._exec_r(h, 0, 11, 10, 0b000, 10)  # ADD
        assert result == 0, f"UINT64_MAX + 1 should be 0, got 0x{result:x}"

    def test_zero_sub_1_wraps_to_uint64_max(self, h):
        """0 - 1 = UINT64_MAX (SUB wrap)."""
        h.gprs[10] = 0
        h.gprs[11] = 1
        result = self._exec_r(h, 0b0100000, 11, 10, 0b000, 10)  # SUB
        assert result == TestAluEdgeCases.UINT64_MAX, (
            f"0 - 1 should be UINT64_MAX, got 0x{result:x}"
        )

    # ============================================================
    #  INT64_MAX boundary
    # ============================================================

    def test_int64_max_add_self(self, h):
        """INT64_MAX + INT64_MAX = 0xFFFFFFFFFFFFFFFE (wraps)."""
        val = TestAluEdgeCases.INT64_MAX
        h.gprs[10] = val
        h.gprs[11] = val
        result = self._exec_r(h, 0, 11, 10, 0b000, 10)
        assert result == 0xFFFF_FFFF_FFFF_FFFE, (
            f"INT64_MAX + INT64_MAX expected 0xFFFFFFFFFFFFFFFE, got 0x{result:x}"
        )

    def test_int64_max_add_1_overflows_to_int64_min(self, h):
        """INT64_MAX + 1 = INT64_MIN (signed overflow wrap)."""
        h.gprs[10] = TestAluEdgeCases.INT64_MAX
        h.gprs[11] = 1
        result = self._exec_r(h, 0, 11, 10, 0b000, 10)
        assert result == TestAluEdgeCases.INT64_MIN, (
            f"INT64_MAX + 1 should be INT64_MIN, got 0x{result:x}"
        )

    def test_int64_max_mul_self(self, h):
        """INT64_MAX * INT64_MAX: low 64 = 1, MULH high 64 = 0x3FFFFFFFFFFFFFFF."""
        val = TestAluEdgeCases.INT64_MAX
        h.gprs[10] = val
        h.gprs[11] = val
        result_lo = self._exec_r(h, 1, 11, 10, 0b000, 10)  # MUL
        assert result_lo == 1, f"INT64_MAX^2 lo64 should be 1, got 0x{result_lo:x}"
        h.gprs[10] = val
        result_hi = self._exec_r(h, 1, 11, 10, 0b001, 10)  # MULH
        assert result_hi == 0x3FFF_FFFF_FFFF_FFFF, (
            f"INT64_MAX^2 hi64 should be 0x3FFFFFFFFFFFFFFF, got 0x{result_hi:x}"
        )

    # ============================================================
    #  INT64_MIN boundary
    # ============================================================

    def test_int64_min_add_self_wraps_to_zero(self, h):
        """INT64_MIN + INT64_MIN = 0 (wraps)."""
        val = TestAluEdgeCases.INT64_MIN
        h.gprs[10] = val
        h.gprs[11] = val
        result = self._exec_r(h, 0, 11, 10, 0b000, 10)
        assert result == 0, f"INT64_MIN + INT64_MIN should be 0, got 0x{result:x}"

    def test_int64_min_sub_1(self, h):
        """INT64_MIN - 1 = INT64_MAX."""
        h.gprs[10] = TestAluEdgeCases.INT64_MIN
        h.gprs[11] = 1
        result = self._exec_r(h, 0b0100000, 11, 10, 0b000, 10)
        assert result == TestAluEdgeCases.INT64_MAX, (
            f"INT64_MIN - 1 should be INT64_MAX, got 0x{result:x}"
        )

    def test_int64_min_mul_self(self, h):
        """INT64_MIN * INT64_MIN: low 64 = 0, MULH = 0x4000000000000000."""
        val = TestAluEdgeCases.INT64_MIN
        h.gprs[10] = val
        h.gprs[11] = val
        result_lo = self._exec_r(h, 1, 11, 10, 0b000, 10)
        assert result_lo == 0, f"INT64_MIN^2 lo64 should be 0, got 0x{result_lo:x}"
        h.gprs[10] = val
        result_hi = self._exec_r(h, 1, 11, 10, 0b001, 10)
        assert result_hi == 0x4000_0000_0000_0000, (
            f"INT64_MIN^2 hi64 should be 0x4000000000000000, got 0x{result_hi:x}"
        )

    # ============================================================
    #  DIV overflow -- INT64_MIN / -1
    # ============================================================

    def test_div_overflow_int64_min_div_neg1(self, h):
        """INT64_MIN / -1 overflows; RISC-V spec returns dividend."""
        h.gprs[10] = TestAluEdgeCases.INT64_MIN
        h.gprs[11] = TestAluEdgeCases.UINT64_MAX  # -1
        result = self._exec_r(h, 1, 11, 10, 0b100, 10)  # DIV
        assert result == TestAluEdgeCases.INT64_MIN, (
            f"INT64_MIN / -1 overflow should return INT64_MIN, got 0x{result:x}"
        )

    def test_rem_int64_min_rem_neg1(self, h):
        """INT64_MIN % -1 = 0 (remainder of overflow is 0 per spec)."""
        h.gprs[10] = TestAluEdgeCases.INT64_MIN
        h.gprs[11] = TestAluEdgeCases.UINT64_MAX  # -1
        result = self._exec_r(h, 1, 11, 10, 0b110, 10)  # REM
        assert result == 0, f"INT64_MIN % -1 should be 0, got 0x{result:x}"

    # ============================================================
    #  32-bit word ops -- ADDIW / SLLIW truncation + sign-extend
    # ============================================================

    def test_addiw_sign_extends_negative(self, h):
        """ADDIW: 0x7FFFFFFF + 1 = 0xFFFFFFFF80000000 (sign-extended)."""
        h.gprs[10] = 0x7FFF_FFFF
        instr = (1 << 20) | (10 << 15) | (0b000 << 12) | (10 << 7) | 0b0011011
        advance = h.exec_instr(instr)
        if advance:
            h.pc = (h.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF
        assert h.gprs[10] == 0xFFFF_FFFF_8000_0000, (
            f"ADDIW: expected 0xFFFFFFFF80000000, got 0x{h.gprs[10]:016x}"
        )

    def test_slliw_shamt_0_truncates_32bit(self, h):
        """SLLIW shamt=0: truncates to 32-bit, sign-extends to 64."""
        h.gprs[10] = 0xABCD_0000_8765_4321  # bit31=1 -> negative
        instr = (0 << 25) | (0 << 20) | (10 << 15) | (0b001 << 12) | (10 << 7) | 0b0011011
        advance = h.exec_instr(instr)
        if advance:
            h.pc = (h.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF
        expected = 0xFFFF_FFFF_8765_4321
        assert h.gprs[10] == expected, (
            f"SLLIW: expected 0x{expected:016x}, got 0x{h.gprs[10]:016x}"
        )

class TestStoreImmediateOffset:
    """回归: parse_imm_s 的 imm[11:5] 缺少 <<5 位移导致 offset>=32 时错误."""

    @pytest.fixture
    def emu(self):
        emu = Emulator(num_harts=1)
        emu.harts[0].gprs[2] = 0x80001000  # sp
        emu.harts[0].gprs[10] = 0xCAFEBABEDEADBEEF  # a0 = test value
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
        addr = (emu.harts[0].gprs[2] + offset) & 0xFFFF_FFFF_FFFF_FFFF
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
        # 2047 未对齐 -> 预期 StAddrMisaligned
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
        return h.gprs[5]

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
        """LUI with bit 31 set -> sign-extended to 64-bit negative."""
        h = emu.harts[0]
        h.pc = 0x80000000
        instr = (0xFFFFF << 12) | (5 << 7) | 0b0110111  # LUI x5, 0xFFFFF
        h.exec_instr(instr)
        # imm20_raw << 12 = 0xFFFFF000, sign-extended from 32-bit -> -0x1000
        assert h.gprs[5] == 0xFFFFFFFFFFFFF000


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
        h.gprs[10] = addr  # rs1
        h.pc = 0x80000000
        instr = (5 << 7) | (funct3 << 12) | (10 << 15) | 0b0000011
        h.exec_instr(instr)
        return h.gprs[5], h.mcause_val

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
        """lh 从奇数地址 -> 应对齐故障."""
        val, mcause = self._exec_load(emu, 0x80000049, 0b001)  # funct3=lh
        assert mcause == 0x4  # LdAddrMisaligned


class TestITypeShifts:
    """回归: I-type 移位使用 funct6 (bits 31:26) 而非 funct7 — bit 25 属于 shamt."""

    @pytest.fixture
    def h(self):
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
            | (shamt << 20)  # 6-bit shamt
            | (rs1 << 15)
            | (0b001 << 12)  # funct3
            | (rd << 7)
            | 0b0010011  # opcode OP-IMM
        )

    @staticmethod
    def _srli(rd, rs1, shamt):
        return (
            (0b000000 << 26)
            | (shamt << 20)
            | (rs1 << 15)
            | (0b101 << 12)
            | (rd << 7)
            | 0b0010011
        )

    @staticmethod
    def _srai(rd, rs1, shamt):
        return (
            (0b010000 << 26)
            | (shamt << 20)
            | (rs1 << 15)
            | (0b101 << 12)
            | (rd << 7)
            | 0b0010011
        )

    def _exec(self, hart, instr):
        advance = hart.exec_instr(instr)
        if advance != 0 and hart.pc == 0x80000000:
            hart.pc = (hart.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF

    # -- SLLI --

    def test_slli_shamt_0(self, h):
        """SLLI shamt=0 应该是 NOP (恒等)."""
        h.gprs[10] = 0x123456789ABCDEF0
        self._exec(h, self._slli(10, 10, 0))
        assert h.gprs[10] == 0x123456789ABCDEF0

    def test_slli_shamt_1(self, h):
        h.gprs[11] = 0x1
        self._exec(h, self._slli(11, 11, 1))
        assert h.gprs[11] == 0x2

    def test_slli_shamt_32(self, h):
        """SLLI shamt=32 — bit 25 置位, 曾触发 funct7≠0 的误判."""
        h.gprs[12] = 0xFFFFFFFFDEADBEEF
        self._exec(h, self._slli(12, 12, 32))
        assert h.gprs[12] == 0xDEADBEEF00000000

    def test_slli_shamt_63(self, h):
        """SLLI 最大移位量 63."""
        h.gprs[13] = 0x3
        self._exec(h, self._slli(13, 13, 63))
        assert h.gprs[13] == (0x3 << 63) & 0xFFFF_FFFF_FFFF_FFFF

    def test_slli_wraps_64(self, h):
        """移位后 & mask 保证 64 位截断."""
        h.gprs[14] = 0xABCD
        self._exec(h, self._slli(14, 14, 60))
        assert h.gprs[14] == 0xD000000000000000

    # -- SRLI --

    def test_srli_shamt_32(self, h):
        """SRLI shamt=32 — bit 25 置位."""
        h.gprs[15] = 0xDEADBEEF00000000
        self._exec(h, self._srli(15, 15, 32))
        assert h.gprs[15] == 0xDEADBEEF

    def test_srli_shamt_0(self, h):
        h.gprs[16] = 0x8000000000000000
        self._exec(h, self._srli(16, 16, 0))
        assert h.gprs[16] == 0x8000000000000000

    # -- SRAI --

    def test_srai_shamt_32_positive(self, h):
        """SRAI 正数算术右移 — 高位补 0."""
        h.gprs[17] = 0x7ABCDEF000000000
        self._exec(h, self._srai(17, 17, 32))
        assert h.gprs[17] == 0x7ABCDEF0

    def test_srai_shamt_32_negative(self, h):
        """SRAI 负数算术右移 — 高位补 1 (符号扩展)."""
        h.gprs[18] = 0x8000000000000000  # 最小负值
        self._exec(h, self._srai(18, 18, 32))
        assert h.gprs[18] == 0xFFFFFFFF80000000

    def test_srai_shamt_63(self, h):
        """SRAI 最大移位量 — 符号位填满全部位."""
        h.gprs[19] = 0x8000000000000000
        self._exec(h, self._srai(19, 19, 63))
        assert h.gprs[19] == 0xFFFFFFFFFFFFFFFF

    # -- 非法 funct6 被拒绝 --

    def test_slli_bad_funct6_traps(self, h):
        """非法 funct6 应触发 IllInstr."""
        bad = (0b111111 << 26) | (0 << 20) | (10 << 15) | (0b001 << 12) | (10 << 7) | 0b0010011
        advance = h.exec_instr(bad)
        assert advance == 0
        assert h.mcause_val != 0


class TestRegValueCanonicalization:
    """回归: 不同指令路径产生的相同 64-bit 值必须在 Python 中 == 相等.

    SLLIW 通过 _sext 返回负 Python int 时, 与 LUI+ADDI 产生的正 64-bit 值
    在 Python == 比较中不相等, 导致 BEQ/BNE 误判. 此测试验证修复.
    """

    @pytest.fixture
    def h(self):
        bus = Bus(ram_size=0x10000, ram_base=0x80000000)
        hart = Hart(id=0)
        inject_memory_backend(hart, bus.read, bus.write)
        hart.bus = bus
        hart.pc = 0x80000000
        return hart

    # -- helpers --

    @staticmethod
    def _addi(rd, rs1, imm12):
        return (imm12 << 20) | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0b0010011

    @staticmethod
    def _slliw(rd, rs1, shamt):
        """slliw rd, rs1, shamt — RV64 32-bit word shift left."""
        return (
            (0b0000000 << 25)  # funct7
            | (shamt << 20)  # 5-bit shamt
            | (rs1 << 15)
            | (0b001 << 12)  # funct3
            | (rd << 7)
            | 0b0011011  # opcode OP-IMM32
        )

    @staticmethod
    def _slli(rd, rs1, shamt):
        return (
            (0b000000 << 26)
            | (shamt << 20)
            | (rs1 << 15)
            | (0b001 << 12)
            | (rd << 7)
            | 0b0010011
        )

    @staticmethod
    def _or(rd, rs1, rs2):
        return (rs2 << 20) | (rs1 << 15) | (0b110 << 12) | (rd << 7) | 0b0110011

    @staticmethod
    def _lui(rd, imm20):
        return (imm20 << 12) | (rd << 7) | 0b0110111

    @staticmethod
    def _beq(rs1, rs2, offset_imm):
        """beq rs1, rs2, offset — B-type."""
        # B-type immediate encoding
        imm = offset_imm & 0x1FFF  # 13-bit signed
        b_imm = (
            ((imm >> 12) & 1) << 31
            | ((imm >> 5) & 0x3F) << 25
            | (rs2 << 20)
            | (rs1 << 15)
            | (0b000 << 12)
            | ((imm >> 1) & 0xF) << 8
            | ((imm >> 11) & 1) << 7
            | 0b1100011
        )
        return b_imm

    @staticmethod
    def _bne(rs1, rs2, offset_imm):
        imm = offset_imm & 0x1FFF
        b_imm = (
            ((imm >> 12) & 1) << 31
            | ((imm >> 5) & 0x3F) << 25
            | (rs2 << 20)
            | (rs1 << 15)
            | (0b001 << 12)
            | ((imm >> 1) & 0xF) << 8
            | ((imm >> 11) & 1) << 7
            | 0b1100011
        )
        return b_imm

    def _exec(self, hart, instr):
        """Execute one instruction and advance PC if the handler didn't."""
        saved_pc = hart.pc
        advance = hart.exec_instr(instr)
        # If handler returned an advance (not 0), increment PC.
        # The handler may have already changed PC (branches/jumps); if so,
        # advance is 0 and we don't touch PC.
        if advance != 0 and hart.pc == saved_pc:
            hart.pc = (saved_pc + advance) & 0xFFFF_FFFF_FFFF_FFFF

    # -- tests --

    def test_slliw_produces_64bit_canonical(self, h):
        """SLLIW 结果应为规范化的 64-bit 无符号值, 不是负 Python int."""
        # 模拟固件中 slliw a1, a1, 24 的操作 (a1 = 0xd0)
        h.gprs[11] = 0xD0
        self._exec(h, self._slliw(11, 11, 24))
        result = h.gprs[11]
        # 硬件上应为 0xFFFFFFFFD0000000 (64-bit unsigned)
        assert result == 0xFFFFFFFFD0000000, (
            f"slliw result: {result:#018x}, expected 0xffffffffd0000000"
        )
        # 关键回归: 结果必须是规范化的 64-bit 值 (非负 Python int)
        assert result >= 0, f"slliw produced negative value: {result}"
        assert result < (1 << 64), f"slliw produced overflow: {result}"

    def test_slliw_and_lui_produce_equal_values(self, h):
        """通过 SLLIW+OR 组合和 LUI+ADDI 构建相同值, 必须 == 相等."""
        # 路径 1: 模拟固件中构造 0xFFFFFFFFD00DFEED 的 SLLIW 路径
        h.gprs[11] = 0xD0
        self._exec(h, self._slliw(11, 11, 24))  # a1 = 0xFFFFFFFFD0000000
        h.gprs[12] = 0x0D
        self._exec(h, self._slli(12, 12, 16))  # a2 = 0x0D0000
        self._exec(h, self._or(13, 12, 11))  # a3 = a2 | a1 = 0xFFFFFFFFD00D0000
        h.gprs[14] = 0xFE
        self._exec(h, self._slli(14, 14, 8))  # a4 = 0xFE00
        self._exec(h, self._addi(14, 14, 0xED))  # a4 = 0xFEED (a4 = a4 + 0xED)
        self._exec(h, self._or(15, 13, 14))  # a5 = 0xFFFFFFFFD00DFEED
        via_slliw = h.gprs[15]

        # 路径 2: 通过 LUI+ADDI 构建同一值
        self._exec(h, self._lui(16, 0xD00E0))  # a6 = 0xFFFFFFFFD00E0000
        self._exec(h, self._addi(16, 16, -0x113))  # a6 = 0xFFFFFFFFD00DFEED
        via_lui = h.gprs[16]

        assert via_slliw == via_lui, f"via SLLIW: {via_slliw:#018x}, via LUI: {via_lui:#018x}"

    def test_beq_with_values_from_different_paths(self, h):
        """BEQ 应正确判定不同路径构建的相等值."""
        # 构建 0xFFFFFFFFD00DFEED 通过 SLLIW 路径
        h.gprs[11] = 0xD0
        self._exec(h, self._slliw(11, 11, 24))
        h.gprs[12] = 0x0D
        self._exec(h, self._slli(12, 12, 16))
        self._exec(h, self._or(13, 12, 11))
        h.gprs[14] = 0xFE
        self._exec(h, self._slli(14, 14, 8))
        self._exec(h, self._addi(14, 14, 0xED))
        self._exec(h, self._or(15, 13, 14))

        # 构建相同值通过 LUI 路径
        self._exec(h, self._lui(16, 0xD00E0))
        self._exec(h, self._addi(16, 16, -0x113))

        # BEQ x15, x16, +8 (跳过下一条, 即 success)
        h.pc = 0x80000100
        self._exec(h, self._beq(15, 16, 8))
        # 分支应被采用 (相等), PC 跳转 +8
        assert h.pc == 0x80000108, f"BEQ should be taken for equal values, PC={h.pc:#x}"

    def test_bne_with_values_from_different_paths(self, h):
        """BNE 应正确判定不同路径构建的相等值 (不应分支)."""
        # 构建同一值两次 (通过不同路径)
        h.gprs[11] = 0xD0
        self._exec(h, self._slliw(11, 11, 24))
        h.gprs[12] = 0x0D
        self._exec(h, self._slli(12, 12, 16))
        self._exec(h, self._or(13, 12, 11))
        h.gprs[14] = 0xFE
        self._exec(h, self._slli(14, 14, 8))
        self._exec(h, self._addi(14, 14, 0xED))
        self._exec(h, self._or(15, 13, 14))

        self._exec(h, self._lui(16, 0xD00E0))
        self._exec(h, self._addi(16, 16, -0x113))

        # 验证值相等
        assert h.gprs[15] == h.gprs[16]

        # BNE x15, x16, +8 — 值相等, 不应分支
        h.pc = 0x80000200
        self._exec(h, self._bne(15, 16, 8))
        assert h.pc == 0x80000204, f"BNE should NOT be taken for equal values, PC={h.pc:#x}"


class TestUartFlush:
    """UART 行缓冲: 多 hart 输出按 \\n 分列刷新, flush_all 强制刷新."""

    @staticmethod
    def _sb(rs1: int, rs2: int, imm: int) -> int:
        """构造 SB (store byte) 指令."""
        imm12 = imm & 0xFFF
        return (
            ((imm12 >> 5) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (0b000 << 12)
            | ((imm12 & 0x1F) << 7)
            | 0b0100011
        )

    @staticmethod
    def _lui(rd: int, imm20: int) -> int:
        """构造 LUI 指令."""
        return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | 0b0110111

    def test_newline_triggers_immediate_flush(self):
        """写 \\n 时 _write_reg 立即触发刷新, 不依赖外部 flush_all."""
        emu = Emulator(num_harts=1, ram_size=128 * 1024 * 1024)
        captured: list[str] = []

        def cap(text: str) -> None:
            captured.append(text)

        assert emu.uart is not None
        emu.uart._tx_callback = cap
        hart = emu.harts[0]

        # x11 = '\n' (0x0A)
        hart.gprs[11] = 0x0A
        hart.pc = 0x80000000
        code = self._lui(10, 0x10000).to_bytes(4, "little") + self._sb(10, 11, 0).to_bytes(
            4, "little"
        )
        emu.bus.write(0x80000000, code)

        emu.step()  # LUI
        emu.step()  # SB — 写入 '\n'

        # \n 在 _write_reg 中触发即时刷新
        assert len(captured) >= 1, f"写 \\n 应产生输出, 实际: {captured}"

    def test_data_stays_buffered_until_newline_or_flush(self):
        """写非 \\n 字符时数据留在行缓冲中, 直到 \\n 或 flush_all 才输出.

        _write_reg 在 \\n 时即时刷新; 无 \\n 的字节留在行缓冲中,
        由外部周期调用 flush_all (native 批次结束) 或显式调用输出.
        _tx_callback (默认 _uart_tx_flush) 每次 write 后 flush stdout,
        确保部分行一旦被 _flush_hart 刷新即可见, 不再滞留在 Python
        stdout 缓冲区中等待下一行.
        """
        emu = Emulator(num_harts=1, ram_size=128 *1024 * 1024)
        assert emu.uart is not None
        captured: list[str] = []

        def cap(text: str) -> None:
            captured.append(text)

        emu.uart._tx_callback = cap
        hart = emu.harts[0]

        uart_base = 0x10000000
        code_addr = 0x80000000
        lui_imm = (uart_base >> 12) & 0xFFFFF
        code = self._lui(10, lui_imm).to_bytes(4, "little") + self._sb(10, 11, 0).to_bytes(
            4, "little"
        )
        emu.bus.write(code_addr, code)
        hart.gprs[11] = 0x41  # 'A'
        hart.pc = code_addr

        emu.step()  # LUI
        emu.step()  # SB — 写入 'A' 到 UART, 无 \n 不刷新

        # emu.step() 在纯 Python 路径不调 flush_all; 'A' 留在行缓冲中
        assert len(captured) == 0, f"无 \\n 时纯 Python 路径不调 flush_all, 实际: {captured}"

        # flush_all 强制刷新, 'A' 应被输出
        emu.uart.flush_all()
        assert len(captured) > 0, f"flush_all 后应有输出, 实际: {captured}"
        assert "A" in "".join(captured), f"应输出 'A', 实际: {captured}"

    def test_flush_all_clears_all_hart_buffers(self):
        """flush_all() 应清空所有 hart 的行缓冲并产生输出."""
        emu = Emulator(num_harts=2, ram_size=128 * 1024 * 1024)
        assert emu.uart is not None
        captured: list[str] = []

        def cap(text: str) -> None:
            captured.append(text)

        emu.uart._tx_callback = cap

        # 模拟两个 hart 各自写入无 \\n 的字符
        emu.uart.set_writer(0)
        emu.bus.write(0x10000000, b"X")
        emu.uart.set_writer(1)
        emu.bus.write(0x10000000, b"Y")

        assert len(emu.uart._line_bufs.get(0, [])) == 1
        assert len(emu.uart._line_bufs.get(1, [])) == 1

        emu.uart.flush_all()
        assert len(emu.uart._line_bufs.get(0, [])) == 0
        assert len(emu.uart._line_bufs.get(1, [])) == 0
        assert len(captured) >= 2, f"应输出两个 hart 的内容, 实际: {captured}"


# ============================================================
#  step() 边界用例: 已停止 hart, 连续 trap, 非法指令
# ============================================================


class TestStepEdgeCases:
    """step() 的边界行为."""

    def test_halted_hart_is_skipped(self):
        """已停止的 hart 在 step() 中被跳过."""
        emu = Emulator(num_harts=2, prog_cnt=0x1000, ram_base=0)
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 10)
        emu.harts[0]._halted = True
        executed = emu.step()
        # 仅 hart 1 执行了
        assert executed == 1
        assert emu.harts[0].pc == 0x1000  # 跳过的 hart PC 未变
        assert emu.harts[1].pc == 0x1004  # 正常执行的 hart PC+4

    def test_consecutive_traps_halt_hart(self):
        """连续 3 次以上 trap 导致 hart 永久停止."""
        emu = Emulator(num_harts=1, prog_cnt=0x1000, ram_base=0)
        # 写入非法指令 (全零)
        emu.load_code(0x1000, b"\x00\x00\x00\x00" * 10)
        h = emu.harts[0]
        # 执行多步触发连续 trap
        for _ in range(5):
            emu.step()
        assert h._halted, "连续 trap 后 hart 应被停止"
        # 此后 step() 跳过该 hart
        executed = emu.step()
        assert executed == 0, "已停止的 hart 不应再执行"

    def test_not_implemented_error_delivers_ill_instr(self):
        """未实现的操作码触发 IllInstr 陷态, hart 不崩溃."""
        emu = Emulator(num_harts=1, prog_cnt=0x1000, ram_base=0)
        # 构造一个未实现的自定义 opcode (bits[1:0]=11 保证 32-bit 编码)
        bad_instr = 0x7F | (10 << 7) | (10 << 15) | (10 << 20)
        emu.load_code(0x1000, bad_instr.to_bytes(4, "little"))
        h = emu.harts[0]
        h.csrs["mtvec"].val = 0x80000000  # 设置 trap handler
        emu.step()
        # 应投递 IllInstr 后跳转到 mtvec
        assert h.pc == 0x80000000
        assert (h.mcause_val & 0x7FFF_FFFF_FFFF_FFFF) == 2  # IllInstr


# ============================================================
#  load_firmware 边界
# ============================================================


class TestLoadFirmware:
    """load_firmware 边界用例."""

    def test_none_image_raises_valueerror(self):
        """image=None 时抛出 ValueError."""
        emu = Emulator()
        with pytest.raises(ValueError, match="无效"):
            emu.load_firmware(None)


# ============================================================
#  run() 超时
# ============================================================


class TestRunTimeout:
    """run() 的超时."""

    def test_timeout_raises_timeouterror(self):
        """非零 timeout 超时时抛出 TimeoutError."""
        emu = Emulator(num_harts=1, prog_cnt=0x1000, ram_base=0)
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 10000)
        with pytest.raises(emu.TimeoutError) as exc_info:
            emu.run(max_cycles=10**9, timeout=0.001, yield_every=1000)
        err = exc_info.value
        assert "超时" in str(err)

    def test_run_returns_int(self):
        """run() 返回执行的周期数 (int)."""
        emu = Emulator(num_harts=1, prog_cnt=0x1000, ram_base=0)
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 5)
        cycles = emu.run(max_cycles=5, timeout=0)
        assert isinstance(cycles, int)
        assert cycles >= 5

    def test_yield_every_zero_runs_full(self):
        """yield_every=0 时 run() 不限速执行全部周期."""
        emu = Emulator(num_harts=1, prog_cnt=0x1000, ram_base=0)
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 5)
        cycles = emu.run(max_cycles=5, timeout=0, yield_every=0)
        assert cycles >= 5


# ============================================================
#  属性与状态
# ============================================================


class TestEmulatorProperties:
    """Emulator 的属性和状态."""

    def test_peripherals_dict(self):
        """peripherals 属性返回外设字典 (CLINT 为独立设备不在 dict 中)."""
        emu = Emulator()
        periph = emu.peripherals
        assert "uart" in periph
        assert "spi" in periph
        assert "i2c" in periph
        assert "gpio" in periph

    def test_cycle_and_total_instructions(self):
        """cycle 和 total_instructions 初始为 0, step() 后增长."""
        emu = Emulator(num_harts=1, prog_cnt=0x1000, ram_base=0)
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 10)
        assert emu.cycle == 0
        assert emu.total_instructions == 0
        emu.step()
        assert emu.cycle >= 1
        assert emu.total_instructions == 1
        emu.step()
        assert emu.total_instructions == 2

    def test_counter_csr_sync_per_step(self):
        """mcycle/cycle/minstret/instret/time CSR 每条指令后同步更新."""
        emu = Emulator(num_harts=1, prog_cnt=0x1000, ram_base=0)
        emu.load_code(0x1000, b"\x13\x00\x00\x00" * 10)
        h = emu.harts[0]

        # 初始值取决于复位, 允许从 0 开始
        c0 = h.csrs["mcycle"].val
        i0 = h.csrs["minstret"].val
        t0 = h.csrs["time"].val
        emu.step()
        c1 = h.csrs["mcycle"].val
        i1 = h.csrs["minstret"].val
        t1 = h.csrs["time"].val
        assert c1 > c0, f"mcycle 应递增: {c0} -> {c1}"
        assert i1 > i0, f"minstret 应递增: {i0} -> {i1}"
        assert t1 > t0, f"time 应递增 (来自 CLINT mtime): {t0} -> {t1}"
        # time 应等于 CLINT 的 mtime
        assert t1 == emu.clint.get_mtime(), (
            f"time CSR 应与 CLINT mtime 同步: {t1} vs {emu.clint.get_mtime()}"
        )

        # cycle / instret 为只读影子, 应与 mcycle / minstret 同步
        assert h.csrs["cycle"].val == c1, "cycle 应与 mcycle 同步"
        assert h.csrs["instret"].val == i1, "instret 应与 minstret 同步"

        # 执行更多指令, 验证差值
        emu.step()
        emu.step()
        delta_c = h.csrs["mcycle"].val - c1
        delta_i = h.csrs["minstret"].val - i1
        delta_t = h.csrs["time"].val - t1
        assert delta_c == 2, f"3 步后 mcycle 差值应为 2, 实际 {delta_c}"
        assert delta_i == 2, f"3 步后 minstret 差值应为 2, 实际 {delta_i}"
        assert delta_t == 2, f"3 步后 time 差值应为 2, 实际 {delta_t}"


class TestL2SizeZero:
    """l2_size=0 退化场景: L2 内部收敛为最小可用缓存, 不崩溃."""

    def test_l2_size_zero_emulator_runs(self):
        """l2_size=0 的 Emulator 可正常执行指令, 不触发 IndexError."""
        from pyremu.platform import PlatformConfig
        cfg = PlatformConfig.qemu_virt()
        cfg.num_harts = 1
        cfg.l2_size = 0
        cfg.ram_base = 0x80000000
        emu = Emulator(config=cfg)
        # 写入两条 nop 指令
        emu.load_code(0x80000000, b"\x13\x00\x00\x00\x13\x00\x00\x00")
        emu.step()
        assert emu.harts[0].pc == 0x80000004
        emu.step()
        assert emu.harts[0].pc == 0x80000008


# ============================================================
#  SPI / I2C / GPIO 外设初始化 — 默认 qemu_virt 配置
# ============================================================


class TestPeripheralInit:
    """SPI / I2C / GPIO 外设均已在默认配置中注册到总线."""

    def test_spi_registered_on_bus(self):
        """SPI 设备在总线默认地址 0x10001000."""
        emu = Emulator()
        assert emu.spi is not None
        assert emu.bus.is_device_addr(0x10001000)

    def test_i2c_registered_on_bus(self):
        """I2C 设备在总线默认地址 0x10002000."""
        emu = Emulator()
        assert emu.i2c is not None
        assert emu.bus.is_device_addr(0x10002000)

    def test_gpio_registered_on_bus(self):
        """GPIO 设备在总线默认地址 0x10003000."""
        emu = Emulator()
        assert emu.gpio is not None
        assert emu.bus.is_device_addr(0x10003000)


# ============================================================
#  build_dtb / load_dtb
# ============================================================


class TestDeviceTree:
    """设备树生成与加载."""

    def test_build_dtb_returns_valid_dtb(self):
        """build_dtb 返回非空 DTB blob."""
        emu = Emulator()
        dtb = emu.build_dtb()
        assert len(dtb) > 0
        # DTB 应以 FDT magic 开头 (0xD00DFEED big-endian)
        magic = int.from_bytes(dtb[:4], "big")
        assert magic == 0xD00DFEED, f"DTB magic 错误: 0x{magic:08x}"

    def test_build_dtb_contains_memory_node(self):
        """build_dtb DTB 包含 memory 节点信息."""
        emu = Emulator()
        dtb = emu.build_dtb()
        # DTB 应包含 "memory" 字符串 (节点名)
        assert b"memory" in dtb, "DTB 应包含 memory 节点"

    def test_load_dtb_blob_sets_a1(self):
        """load_dtb_blob 写入 DTB 并设置所有 hart 的 a1."""
        emu = Emulator(num_harts=2, ram_size=128 * 1024 * 1024)
        dtb = emu.build_dtb()
        addr = 0x87FF0000
        emu.load_dtb_blob(addr, dtb)
        # 验证写入
        readback = emu.bus.read(addr, len(dtb))
        assert readback == dtb
        # 验证所有 hart 的 a1 已设置
        for h in emu.harts:
            assert h.gprs[11] == addr

    def test_load_dtb_generates_and_loads(self):
        """load_dtb 生成并加载 DTB, a1 指向该地址."""
        emu = Emulator(num_harts=1, ram_size=128 * 1024 * 1024)
        addr = 0x87FF0000
        emu.load_dtb(addr)
        # 验证 a1
        assert emu.harts[0].gprs[11] == addr
        # 验证 DTB 可读
        magic_bytes = emu.bus.read(addr, 4)
        magic = int.from_bytes(magic_bytes, "big")
        assert magic == 0xD00DFEED

    def test_default_bootargs_provides_console(self):
        """未指定 bootargs 时, 默认包含 earlycon=sbi console=ttySIF0."""
        emu = Emulator()
        dtb = emu.build_dtb()
        assert b"earlycon=sbi" in dtb, (
            "默认 bootargs 应包含 earlycon=sbi 以确保内核输出可见"
        )
        assert b"console=ttySIF0" in dtb, (
            "默认 bootargs 应包含 console=ttySIF0 以将 UART 设为首选控制台"
        )
        assert b"random.trust_bootloader=on" in dtb, (
            "默认 bootargs 应包含 random.trust_bootloader=on 以完成 CRNG 初始化"
        )

    def test_explicit_bootargs_overrides_default(self):
        """显式指定 bootargs 时使用用户值而非默认值."""
        emu = Emulator(bootargs="console=ttyS0 debug")
        dtb = emu.build_dtb()
        assert b"console=ttyS0 debug" in dtb
        # 默认值不应出现
        assert b"console=ttySIF0" not in dtb


class TestBusErrorOnUnmappedAccess:
    """通过完整取指-执行流水线触发 Bus error, 验证 PMA 空洞地址访问 -> AccessFault.

    模拟 Linux 内核启动中探访未映射 MMIO 地址时发生的 Bus error 场景.
    与 ``test_trap.py::TestMemoryAccessFaults`` 的区别: 本测试经过
    fetch → decode → execute → translate → PMP → PMA 完整路径,
    而非直接调用 ``mem_read`` / ``mem_write``.
    """

    RAM_BASE = 0x8000_0000
    HOLE_ADDR = 0x4000_0000  # 非 RAM、非设备 MMIO 的空洞地址

    @pytest.fixture
    def emu(self) -> Emulator:
        emu = Emulator(num_harts=1)
        # 使用纯 Python 路径以隔离测试目标 (不受 native batch 影响)
        emu._native_batch = False
        return emu

    def test_load_from_unmapped_addr_traps(self, emu):
        """lw 从空洞地址读取 -> 经完整流水线触发 LdAccessFault."""
        h = emu.harts[0]
        h.pc = self.RAM_BASE
        # 将目标地址放入 x10 (rs1)
        h.gprs[10] = self.HOLE_ADDR

        # lw x5, 0(x10): opcode=0000011 funct3=010 rd=5 rs1=10 imm=0
        instr = (0 << 20) | (10 << 15) | (2 << 12) | (5 << 7) | 0x3
        emu.load_code(self.RAM_BASE, instr.to_bytes(4, "little"))

        emu.step()

        # 应触发 LdAccessFault (mcause = 5)
        assert h.mcause_val == 5, (
            f"应为 LdAccessFault (5), 实际 mcause={h.mcause_val}"
        )
        # PC 应跳转到 mtvec (默认 0x0), 而非停留在加载指令之后
        assert h.pc != self.RAM_BASE + 4, (
            f"PC 不应指向下一条指令, 应已跳转到 mtvec handler, 实际 pc={h.pc:#x}"
        )

    def test_store_to_unmapped_addr_traps(self, emu):
        """sw 到空洞地址 -> 经完整流水线触发 StAccessFault."""
        h = emu.harts[0]
        h.pc = self.RAM_BASE
        h.gprs[10] = self.HOLE_ADDR  # rs1 = 目标地址
        h.gprs[11] = 0xDEAD_BEEF  # rs2 = 写入值

        # sw x11, 0(x10): opcode=0100011 funct3=010 rs2=11 rs1=10 imm=0
        instr = (0 << 25) | (11 << 20) | (10 << 15) | (2 << 12) | 0x23
        emu.load_code(self.RAM_BASE, instr.to_bytes(4, "little"))

        emu.step()

        # 应触发 StAccessFault (mcause = 7)
        assert h.mcause_val == 7, (
            f"应为 StAccessFault (7), 实际 mcause={h.mcause_val}"
        )
        assert h.pc != self.RAM_BASE + 4, (
            f"PC 应已跳转到 mtvec handler, 实际 pc={h.pc:#x}"
        )

    def test_device_addr_load_succeeds(self, emu):
        """从已注册设备 MMIO 地址读取不触发 Bus error (经过设备路径)."""
        h = emu.harts[0]
        h.pc = self.RAM_BASE
        # UART 基址 (默认 0x1000_0000) 已注册为设备
        uart_base = 0x1000_0000
        h.gprs[10] = uart_base

        # lw x5, 0(x10)
        instr = (0 << 20) | (10 << 15) | (2 << 12) | (5 << 7) | 0x3
        emu.load_code(self.RAM_BASE, instr.to_bytes(4, "little"))

        emu.step()

        # UART 是已注册设备, 不应触发 AccessFault
        assert h.mcause_val != 5, (
            f"设备 MMIO 读取不应触发 LdAccessFault, 实际 mcause={h.mcause_val}"
        )

    def test_ram_boundary_crossing_traps(self, emu):
        """跨 RAM 边界的 load -> LdAccessFault (部分字节在空洞中)."""
        h = emu.harts[0]
        h.pc = self.RAM_BASE
        # 定位到 RAM 最后 4 字节, ld 8 字节 → 高 4 字节溢出到空洞
        ram_end = self.RAM_BASE + emu.bus.ram_size
        h.gprs[10] = ram_end - 4

        # ld x5, 0(x10): opcode=0000011 funct3=011 rd=5 rs1=10 imm=0
        instr = (0 << 20) | (10 << 15) | (3 << 12) | (5 << 7) | 0x3
        emu.load_code(self.RAM_BASE, instr.to_bytes(4, "little"))

        emu.step()

        # PMA 检查应拒绝跨边界访问
        assert h.mcause_val in (5, 4), (
            f"跨边界访问应触发 LdAccessFault(5) 或 LdAddrMisaligned(4), "
            f"实际 mcause={h.mcause_val}"
        )


class TestNativeBatchLayout:
    """Python ctypes ↔ Rust repr(C) 结构体布局一致性验证.

    Rust 侧将 TLB 存储为 ``[TlbEntry; 32]`` (array-of-structs).
    若 Python 侧错误地使用分离数组 (如 itlb_vpn, itlb_ppn, ...)
    会造成内存布局不匹配 → SIGBUS (Bus error).
    这些用例在每次构建后锁死布局合约.
    """

    def test_tlb_entry_size_24(self) -> None:
        """TlbEntry 必须恰好 24 字节."""
        assert ctypes.sizeof(TlbEntry) == 24, (
            f"TlbEntry 应为 24 字节, 实际 {ctypes.sizeof(TlbEntry)}B"
        )

    def test_hart_state_8byte_aligned(self) -> None:
        """HartState 必须 8 字节对齐."""
        assert ctypes.sizeof(HartState) % 8 == 0

    def test_hart_state_under_4k(self) -> None:
        """HartState 不应膨胀超过 4096 字节."""
        assert ctypes.sizeof(HartState) < 4096

    def test_batch_result_8byte_aligned(self) -> None:
        """BatchResult 必须 8 字节对齐."""
        assert ctypes.sizeof(BatchResult) % 8 == 0

    def test_itlb_is_array_of_structs(self) -> None:
        """itlb 确保 array-of-structs 而非 struct-of-arrays.

        struct-of-arrays 会导致 Rust 在 offsetof(itlb[i].ppn) 读到垃圾 → SIGBUS.
        """
        hs = HartState()
        assert len(hs.itlb) == 32
        assert hasattr(hs.itlb[0], "vpn")
        assert hasattr(hs.itlb[0], "ppn")
        assert hasattr(hs.itlb[0], "valid")

    def test_dtlb_is_array_of_structs(self) -> None:
        """dtlb 确保 array-of-structs."""
        hs = HartState()
        assert len(hs.dtlb) == 32
        assert hasattr(hs.dtlb[0], "vpn")
        assert hasattr(hs.dtlb[0], "ppn")
        assert hasattr(hs.dtlb[0], "valid")


class TestMarshalUnmarshalRoundtrip:
    """marshal_hart → unmarshal_hart 双向数据完整性验证.

    batch 边界 marshal/unmarshal 是 Python HartWithRegs ↔ Rust HartState
    的唯一数据通道.  若任一字段在往返过程中丢失或损坏, batch 内部的 Rust
    执行会基于错误状态继续运行 — 产生静默错误 (如 ld-linux s4=-1 消失).

    本测试覆盖 HartState 中所有非 pad 字段的往返保真度.
    """

    # ---------- helpers ----------
    @staticmethod
    def _fresh_hart(hart_id: int = 0) -> Hart:
        """创建一个处于 M 模式的 Hart (__init__ 默认)."""
        return Hart(id=hart_id)

    @staticmethod
    def _roundtrip(hart: Hart) -> Hart:
        """marshal → unmarshal 到新 hart, 返回新 hart."""
        state = HartState()
        marshal_hart(hart, state)
        out = Hart(id=hart.id + 1)
        unmarshal_hart(state, out)
        return out

    # ---------- GPR 往返 ----------
    def test_gpr_all_zero_init(self) -> None:
        """所有 GPR 初始为 0 时往返不变."""
        hart = self._fresh_hart()
        out = self._roundtrip(hart)
        for i in range(32):
            assert out.gprs[i] == 0, f"gprs[{i}] = {out.gprs[i]}, expected 0"

    def test_gpr_fills_pattern_roundtrip(self) -> None:
        """每寄存器写入唯一值后往返不变."""
        hart = self._fresh_hart()
        for i in range(32):
            # 用可识别模式: i * 0x1111_1111_1111_1111 + 0xDEAD_BEEF_CAFE_BABE
            hart.gprs[i] = (i * 0x1111_1111_1111_1111 + 0xDEAD_BEEF_CAFE_BABE) & 0xFFFF_FFFF_FFFF_FFFF
        # x0 写保护 — GprFile 应忽略写入
        hart.gprs[0] = 0  # explicitly ensure
        out = self._roundtrip(hart)
        assert out.gprs[0] == 0, "x0 must always be 0"
        for i in range(1, 32):
            expected = (i * 0x1111_1111_1111_1111 + 0xDEAD_BEEF_CAFE_BABE) & 0xFFFF_FFFF_FFFF_FFFF
            assert out.gprs[i] == expected, (
                f"gprs[{i}] = {out.gprs[i]:#018x}, expected {expected:#018x}"
            )

    def test_s4_negative_one_preserved(self) -> None:
        """s4 (x20) = -1 (0xFFFF_FFFF_FFFF_FFFF) 在往返后必须保持.

        ld-linux 用 s4=-1 作为循环退出哨兵.  若 unmarshal 在某处将其清零
        或截断, BNE s2,s4 将永不退出 — 继续迭代到 DT_RELA=7 → LoadPageFault.
        """
        hart = self._fresh_hart()
        hart.gprs[20] = 0xFFFF_FFFF_FFFF_FFFF  # s4 = -1
        out = self._roundtrip(hart)
        assert out.gprs[20] == 0xFFFF_FFFF_FFFF_FFFF, (
            f"s4 = {out.gprs[20]:#018x}, expected -1"
        )

    def test_gpr_edge_values_roundtrip(self) -> None:
        """GPR 极端值 (0, -1, MAX, 符号位, 交替 bits) 往返不变."""
        hart = self._fresh_hart()
        test_values = {
            1: 0,                           # zero
            2: 0xFFFF_FFFF_FFFF_FFFF,       # -1 (ld-linux s4 sentinel)
            3: 0x7FFF_FFFF_FFFF_FFFF,       # max positive signed
            4: 0x8000_0000_0000_0000,       # min negative signed
            5: 0xAAAA_AAAA_AAAA_AAAA,       # alternating bits
            6: 0x5555_5555_5555_5555,       # alternating bits (inverted)
            7: 0x0000_0000_0000_0001,       # 1 (smallest non-zero)
            8: 0x8000_0000_0000_0001,       # sign bit + 1
        }
        for idx, val in test_values.items():
            hart.gprs[idx] = val
        out = self._roundtrip(hart)
        for idx, val in test_values.items():
            assert out.gprs[idx] == val, (
                f"gprs[{idx}] = {out.gprs[idx]:#018x}, expected {val:#018x}"
            )

    # ---------- 关键 CSR 往返 ----------
    @pytest.mark.parametrize("csr_name,val", [
        ("mstatus", 0x0000000A00001800),  # MPP=M, SPP=S, FS=1, XS=1
        ("mtvec", 0x80001000),            # vectored M-mode
        ("stvec", 0x80200000),            # direct S-mode
        ("mepc", 0x80001234),
        ("sepc", 0x3FFF_FFFF_F000),
        ("mcause", 0x8000_0000_0000_000D),  # LoadPageFault interrupt
        ("scause", 0x0000_0000_0000_0008),  # EcallFromUmode
        ("mtval", 0x0000_0033D),
        ("stval", 0x3FFF_FFFF_F000),
        ("satp", 0x8000_0000_0001_2345),  # Sv39, PPN=0x12345
        ("mie", 0x888),                    # MEIE | SEIE | MTIE
        ("mip", 0x080),                    # STIP
        ("medeleg", 0xBEEF),
        ("mideleg", 0xDEAD),
    ])
    def test_csr_roundtrip(self, csr_name: str, val: int) -> None:
        """关键 CSR 往返不变."""
        hart = self._fresh_hart()
        hart.csrs[csr_name].val = val
        out = self._roundtrip(hart)
        actual = out.csrs[csr_name].val
        assert actual == val, f"{csr_name}: {actual:#018x} ≠ expected {val:#018x}"

    # ---------- 状态字段 ----------
    def test_pc_roundtrip(self) -> None:
        hart = self._fresh_hart()
        hart.pc = 0x3FFF_FFFF_3ABA
        out = self._roundtrip(hart)
        assert out.pc == 0x3FFF_FFFF_3ABA

    @pytest.mark.parametrize("mode", [RiscvMode.U, RiscvMode.S, RiscvMode.M])
    def test_mode_roundtrip(self, mode: RiscvMode) -> None:
        hart = Hart(id=0)
        hart.mode = mode
        out = self._roundtrip(hart)
        assert out.mode == mode

    def test_mmu_mode_roundtrip(self) -> None:
        hart = self._fresh_hart()
        hart._mmu_mode = 8  # Sv39
        out = self._roundtrip(hart)
        assert out.mmu_mode == 8

    # ---------- LR/SC reservation ----------
    def test_reservation_roundtrip(self) -> None:
        hart = self._fresh_hart()
        hart._reservation_valid = True
        hart._reservation_addr = 0x8000_1234
        out = self._roundtrip(hart)
        assert out._reservation_valid is True
        assert out._reservation_addr == 0x8000_1234

    def test_reservation_invalid_roundtrip(self) -> None:
        hart = self._fresh_hart()
        hart._reservation_valid = False
        out = self._roundtrip(hart)
        assert out._reservation_valid is False

    # ---------- WFI / halted ----------
    def test_waiting_wfi_woken_roundtrip(self) -> None:
        hart = self._fresh_hart()
        hart._waiting = True
        hart._wfi_woken = True
        out = self._roundtrip(hart)
        assert out._waiting is True
        assert out._wfi_woken is True

    def test_halted_roundtrip(self) -> None:
        hart = self._fresh_hart()
        hart._halted = True
        hart._consecutive_traps = 3
        out = self._roundtrip(hart)
        assert out._halted is True
        assert out._consecutive_traps == 3

    # ---------- TEE CSRs ----------
    def test_mdid_pmpsplit_roundtrip(self) -> None:
        hart = self._fresh_hart()
        hart._mdid_val = 5
        hart._pmpsplit_val = 2
        out = self._roundtrip(hart)
        assert out.mdid_val == 5
        assert out.pmpsplit_val == 2

    # ---------- Extra CSRs ----------
    def test_extra_csrs_roundtrip(self) -> None:
        hart = self._fresh_hart()
        hart.csrs["mscratch"].val = 0xDEAD_BEEF_0000_1111
        hart.csrs["sscratch"].val = 0xCAFE_BABE_2222_3333

        # mhartid is read-only (set in __init__ to hart.id)
        # Write mcounteren directly
        hart.csrs["mcounteren"].val = 0x0000_0000_0000_0007  # CY|TM|IR

        out = self._roundtrip(hart)
        assert out.csrs["mscratch"].val == 0xDEAD_BEEF_0000_1111
        assert out.csrs["sscratch"].val == 0xCAFE_BABE_2222_3333
        assert out.csrs["mhartid"].val == 0  # original hart had id=0
        assert out.csrs["mcounteren"].val == 0x0000_0000_0000_0007

    # ---------- instruction counter ----------
    def test_total_instrs_roundtrip(self) -> None:
        hart = self._fresh_hart()
        hart._total_instrs = 123456789
        out = self._roundtrip(hart)
        assert out._total_instrs == 123456789

    # ---------- float ----------
    def test_fpr_roundtrip(self) -> None:
        hart = self._fresh_hart()
        for i in range(32):
            hart._fpr_bits[i] = (0x3FF0_0000_0000_0000 + i * 0x1000_0000_0000) & 0xFFFF_FFFF_FFFF_FFFF
        out = self._roundtrip(hart)
        for i in range(32):
            expected = (0x3FF0_0000_0000_0000 + i * 0x1000_0000_0000) & 0xFFFF_FFFF_FFFF_FFFF
            assert out._fpr_bits[i] == expected, f"fpr[{i}] mismatch"

    def test_fcsr_roundtrip(self) -> None:
        hart = self._fresh_hart()
        hart.csrs["fcsr"].val = 0x0000_0000_0000_00A0  # FRM=2, NV|DZ flags
        out = self._roundtrip(hart)
        assert out.csrs["fcsr"].val == 0x0000_0000_0000_00A0


class TestLdLinuxAddrChain:
    """AUIPC + slli/add: ld-linux 数组索引地址计算链.

    ld-linux 偏移 0x3aba-0x3ace:
      slli s1,a5,2; add s1,s1,a5; slli s1,s1,5
      addi s1,s1,-0xa0; auipc a5,0x1e; addi a5,a5,0x542; add s1,s1,a5
    若任一指令结果错误则 s1 指向错误元素 → ld a5,0(s1) 读到 DT_RELA=7.
    """

    SLLI_S1_A5_2 = 0x00279493    # 4B: slli s1, a5, 2
    C_ADD_S1_A5 = 0x94BE          # 2B: c.add s1, a5
    C_SLLI_S1_5 = 0x0496          # 2B: c.slli s1, 5
    ADDI_S1_NEG_A0 = 0xF6048493  # 4B: addi s1, s1, -0xa0
    AUIPC_A5_1E = 0x0001E797     # 4B: auipc a5, 0x1e
    ADDI_A5_A5_542 = 0x54278793  # 4B: addi a5, a5, 0x542
    C_ADD_S1_A5_FINAL = 0x94BE   # 2B: c.add s1, a5
    PC_START = 0x3ABA
    PC_AT_AUIPC = 0x3AC6  # 0x3aba + 4 + 2 + 2 + 4

    @staticmethod
    def _make_hart() -> Hart:
        hart = Hart(id=0)
        hart.mode = RiscvMode.M  # M-mode, bare translation
        return hart

    @pytest.mark.parametrize("counter", [0, 1, 2, 3, 5, 10, 256])
    def test_s1_address(self, counter: int) -> None:
        """s1 == (counter-1)*0xa0 + pc_auipc + 0x1e542."""
        hart = self._make_hart()
        hart.pc = self.PC_START
        hart.gprs[15] = counter  # a5

        for instr in (self.SLLI_S1_A5_2, self.C_ADD_S1_A5, self.C_SLLI_S1_5,
                      self.ADDI_S1_NEG_A0, self.AUIPC_A5_1E,
                      self.ADDI_A5_A5_542, self.C_ADD_S1_A5_FINAL):
            hart.pc += hart.exec_instr(instr)

        expected = (counter - 1) * 0xA0 + self.PC_AT_AUIPC + 0x1E000 + 0x542
        assert hart.gprs[9] == expected, (
            f"counter={counter}: s1={hart.gprs[9]:#018x}, expected {expected:#018x}"
        )

    @pytest.mark.parametrize("pc_base", [
        0x3ABA,
        0x3F_F7FD_cABA,  # ld-linux 运行时 VA
    ])
    @pytest.mark.parametrize("counter", [1, 2, 5])
    def test_auipc_pc_relative(self, counter: int, pc_base: int) -> None:
        """AUIPC 使用自身 PC 而非其他值."""
        hart = self._make_hart()
        hart.pc = pc_base
        hart.gprs[15] = counter

        for instr in (self.SLLI_S1_A5_2, self.C_ADD_S1_A5, self.C_SLLI_S1_5,
                      self.ADDI_S1_NEG_A0, self.AUIPC_A5_1E,
                      self.ADDI_A5_A5_542, self.C_ADD_S1_A5_FINAL):
            hart.pc += hart.exec_instr(instr)

        pc_auipc = pc_base + (self.PC_AT_AUIPC - self.PC_START)
        expected = (counter - 1) * 0xA0 + pc_auipc + 0x1E000 + 0x542
        assert hart.gprs[9] == expected
        assert hart.gprs[15] == pc_auipc + 0x1E000 + 0x542


class TestBranchEdgeCases:
    """分支边界值测试 — ld-linux 循环退出条件.

    ld-linux 偏移 0x3ab6:  bltz s2, 0x3b0a  (s2<0 时跳过循环)
    ld-linux 偏移 0x3b06:  bne  s2, s4, 0x3ad0  (s2!=-1 时继续循环)
    """

    def _make_hart(self) -> Hart:
        hart = Hart(id=0)
        hart.mode = RiscvMode.M
        return hart

    # ---- BLTZ (s2 < 0 → branch) ----
    @pytest.mark.parametrize("s2_val,taken", [
        (-1, True),    # s2=-1 < 0 → branch to skip loop
        (0, False),    # s2=0 >= 0 → fall through (enter loop)
        (1, False),    # s2=1 >= 0 → fall through
        (-2, True),    # s2=-2 < 0 → branch
    ])
    def test_bltz_edge(self, s2_val: int, taken: bool) -> None:
        """BLTZ at 0x3ab6: branch taken iff s2 < 0."""
        hart = self._make_hart()
        hart.pc = 0x3AB6
        hart.gprs[18] = s2_val & 0xFFFF_FFFF_FFFF_FFFF  # s2

        # bltz s2, 0x3b0a = 0x04094a63
        advance = hart.exec_instr(0x04094A63)
        hart.pc += advance

        if taken:
            assert hart.pc == 0x3B0A, f"s2={s2_val}: should branch to 0x3b0a"
        else:
            # Fall-through: pc += advance (4 for non-taken 32-bit branch)
            assert hart.pc == 0x3ABA, f"s2={s2_val}: should fall through to 0x3aba"

    # ---- BNE (s2 != s4 → branch back) ----
    @pytest.mark.parametrize("s2_val,s4_val,taken", [
        (-1, -1, False),  # s2 == s4 → exit loop
        (0, -1, True),    # s2 != s4 → continue loop
        (1, -1, True),    # continue
        (-2, -1, True),   # s2=-2 != -1 → continue
    ])
    def test_bne_exit_edge(self, s2_val: int, s4_val: int, taken: bool) -> None:
        """BNE at 0x3b06: branch NOT taken when s2 == s4."""
        hart = self._make_hart()
        hart.pc = 0x3B06
        hart.gprs[18] = s2_val & 0xFFFF_FFFF_FFFF_FFFF  # s2
        hart.gprs[20] = s4_val & 0xFFFF_FFFF_FFFF_FFFF  # s4

        # bne s2, s4, 0x3ad0 = 0xFD4915E3
        advance = hart.exec_instr(0xFD4915E3)
        hart.pc += advance

        if taken:
            assert hart.pc == 0x3AD0, f"s2={s2_val} s4={s4_val}: should loop to 0x3ad0"
        else:
            assert hart.pc == 0x3B0A, f"s2={s2_val} s4={s4_val}: should exit to 0x3b0a"
