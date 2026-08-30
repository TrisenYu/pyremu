#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""CLINT 设备测试: MSIP (IPI), MTIMECMP (定时器), MTIME."""

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.hart import MSTATUS_MIE, RiscvMode
from pyremu.core.mem_check_aux import inject_memory_backend
from pyremu.core.trap_handler import _update_hw_mip
from pyremu.interrupt.clint import (
    CLINT,
    CLINT_BASE,
    MSIP_OFFSET,
    MTIME_OFFSET,
    MTIMECMP_OFFSET,
)
from pyremu.interrupt.controller import INT_SOURCE_MIP_MASK, IntSource
from pyremu.memory.bus import Bus


class TestCLINTRegisters:
    """CLINT 寄存器读写."""

    @pytest.fixture
    def clint(self) -> CLINT:
        return CLINT(num_harts=4)

    def test_read_msip_initial_zero(self, clint):
        """初始 MSIP 应为 0."""
        data = clint.read(MSIP_OFFSET, 4)  # hart 0
        assert data == b"\x00\x00\x00\x00"

    def test_write_msip_triggers_ipi(self, clint):
        """写 MSIP 应设置软件中断挂起."""
        clint.write(MSIP_OFFSET + 4, b"\x01\x00\x00\x00")  # hart 1 msip = 1
        has_pending, mip, src = clint.check_interrupt(1)
        assert has_pending
        assert mip & INT_SOURCE_MIP_MASK[IntSource.MSI]
        assert src == IntSource.MSI

    def test_write_msip_zero_clears_ipi(self, clint):
        """写 0 到 MSIP 应清除中断."""
        clint.write(MSIP_OFFSET, b"\x01\x00\x00\x00")  # hart 0 msip = 1
        has_pending, _, _ = clint.check_interrupt(0)
        assert has_pending
        clint.write(MSIP_OFFSET, b"\x00\x00\x00\x00")  # hart 0 msip = 0
        has_pending, _, _ = clint.check_interrupt(0)
        assert not has_pending

    def test_send_ipi(self, clint):
        """send_ipi 辅助方法."""
        clint.send_ipi(2)  # 向 hart 2 发送 IPI
        has_pending, _, _ = clint.check_interrupt(2)
        assert has_pending

    def test_clear_ipi(self, clint):
        """clear_ipi 辅助方法."""
        clint.send_ipi(3)
        clint.clear_ipi(3)
        has_pending, _, _ = clint.check_interrupt(3)
        assert not has_pending

    def test_mtimecmp_timer_interrupt(self, clint):
        """mtime >= mtimecmp 时应触发定时器中断."""
        clint.tick(100)  # mtime = 100
        # 设置 hart 0 的 mtimecmp = 50 (已过期)
        clint.write(MTIMECMP_OFFSET, (50).to_bytes(8, "little"))
        has_pending, mip, src = clint.check_interrupt(0)
        assert has_pending
        assert mip & INT_SOURCE_MIP_MASK[IntSource.MTI]

    def test_mtimecmp_not_reached(self, clint):
        """mtime < mtimecmp 时不应触发定时器中断."""
        clint.tick(10)  # mtime = 10
        clint.write(MTIMECMP_OFFSET, (100).to_bytes(8, "little"))
        has_pending, _, _ = clint.check_interrupt(0)
        assert not has_pending

    def test_mtime_read(self, clint):
        """MTIME 寄存器读."""
        clint.tick(42)
        data = clint.read(MTIME_OFFSET, 8)
        val = int.from_bytes(data, "little")
        assert val == 42

    def test_mtime_write(self, clint):
        """MTIME 寄存器写."""
        clint.write(MTIME_OFFSET, (0xDEAD).to_bytes(8, "little"))
        assert clint.get_mtime() == 0xDEAD

    def test_interrupt_priority(self, clint):
        """中断优先级: MSI > MTI."""
        clint.send_ipi(0)  # MSI
        clint.tick(100)
        clint.write(MTIMECMP_OFFSET, (50).to_bytes(8, "little"))  # MTI
        _, _, src = clint.check_interrupt(0)
        assert src == IntSource.MSI, "MSI 优先级应高于 MTI"


class TestCLINTBusIntegration:
    """通过 Bus 访问 CLINT."""

    def test_bus_write_to_msip(self):
        """通过总线写 MSIP -> IPI 触发."""
        clint = CLINT(num_harts=4)
        bus = Bus(ram_size=1024 * 1024)
        bus.add_device(CLINT_BASE, clint)

        # hart 0 写 hart 1 的 msip
        bus.write(CLINT_BASE + MSIP_OFFSET + 4, b"\x01\x00\x00\x00")
        has_pending, _, _ = clint.check_interrupt(1)
        assert has_pending

    def test_bus_read_device_addr_detected(self):
        """Bus.is_device_addr 应检测 CLINT 地址."""
        clint = CLINT(num_harts=2)
        bus = Bus(ram_size=1024 * 1024)
        bus.add_device(CLINT_BASE, clint)
        assert bus.is_device_addr(CLINT_BASE)
        assert bus.is_device_addr(CLINT_BASE + MSIP_OFFSET)
        assert bus.is_device_addr(CLINT_BASE + MTIME_OFFSET)


class TestCLINTMmioPath:
    """验证 CLINT MSIP 通过 MMIO 存储指令的完整路径.

    固件 (OpenSBI / Linux) 通过 store 指令写 CLINT MSIP 寄存器
    来发送/清除核间中断。该路径必须正确通过
    Hart -> mem_write -> Bus.write -> CLINT.write。
    """

    # RV64 指令编码
    SW_INSTR = 0x00552023  # sw x5, 0(x10) — 将 x5 的值存入 *x10
    SD_INSTR = 0x00553023  # sd x5, 0(x10) — 同上 (64-bit)
    LW_INSTR = 0x00052283  # lw x5, 0(x10)
    WFI_INSTR = 0x10500073  # wfi

    BASE_REG = 10  # x10 (a0) — 用于 store 指令的基址寄存器

    @staticmethod
    def _make_hart_with_clint(
        hart_id: int = 0,
        num_harts: int = 2,
    ):
        """创建带 Bus + CLINT 的 Hart, M-mode."""
        clint = CLINT(num_harts=num_harts)
        clint.base_addr = CLINT_BASE
        bus = Bus(ram_size=1024 * 1024, ram_base=0x8000_0000)
        bus.add_device(CLINT_BASE, clint)

        h = Hart(id=hart_id)
        h.pc = 0x80000000
        h.mode = RiscvMode.M
        h.mstatus_val = MSTATUS_MIE
        h.csrs["mtvec"].val = 0x80000100
        h.csrs["mie"].val = 1 << 3  # MSIE
        h.interrupt_ctrl = clint
        inject_memory_backend(h, bus.read, bus.write)
        h.bus = bus

        return h, clint, bus

    def test_store_to_msip_sets_hardware_bit(self):
        """Store 指令写入 CLINT MSIP -> _msip[hart_id] 置位."""
        h, clint, bus = self._make_hart_with_clint()

        # 设置 store 目标地址 = CLINT MSIP[hart0]
        msip_addr = CLINT_BASE + MSIP_OFFSET
        h.write_gpr(self.BASE_REG, msip_addr)  # rs1 (base addr reg)
        h.write_gpr(5, 1)  # rs2 (value = 1)

        # 将 SW 指令写入 RAM
        bus.write(0x80000000, self.SW_INSTR.to_bytes(4, "little"))

        # 执行 store
        h.exec_instr(self.SW_INSTR)

        # 验证 CLINT _msip 已置位
        assert clint._msip[0] == 1, (
            f"MMIO store 应置位 _msip[0], 实际={clint._msip[0]}"
        )
        has_pending, mip, src = clint.check_interrupt(0)
        assert has_pending, "check_interrupt 应返回 pending"
        assert src is not None and src.name == "MSI", f"中断源应为 MSI, 实际={src}"

    def test_store_zero_to_msip_clears_hardware_bit(self):
        """Store 指令写 0 到 CLINT MSIP -> _msip[hart_id] 清零."""
        h, clint, bus = self._make_hart_with_clint()

        # 先通过 Python API 置位 MSIP (模拟先前的 IPI)
        clint.send_ipi(0)
        assert clint._msip[0] == 1, "send_ipi 应置位 _msip"

        # 现在通过 store 指令清除 MSIP
        msip_addr = CLINT_BASE + MSIP_OFFSET
        h.write_gpr(self.BASE_REG, msip_addr)  # rs1
        h.write_gpr(5, 0)  # rs2 (value = 0)

        bus.write(0x80000000, self.SW_INSTR.to_bytes(4, "little"))
        h.exec_instr(self.SW_INSTR)

        assert clint._msip[0] == 0, (
            f"MMIO store 写 0 应清零 _msip[0], 实际={clint._msip[0]}"
        )
        has_pending, _, _ = clint.check_interrupt(0)
        assert not has_pending, "check_interrupt 应返回 no pending"

    def test_store_to_other_hart_msip_sets_correct_bit(self):
        """Hart 0 store 到 CLINT MSIP[hart1] -> _msip[1] 置位, _msip[0] 不变."""
        h, clint, bus = self._make_hart_with_clint(hart_id=0, num_harts=4)

        # 目标: CLINT MSIP[hart3] (offset = MSIP_OFFSET + 3 * 4)
        msip_addr = CLINT_BASE + MSIP_OFFSET + 3 * 4
        h.write_gpr(self.BASE_REG, msip_addr)  # rs1
        h.write_gpr(5, 1)  # rs2 (value = 1)

        bus.write(0x80000000, self.SW_INSTR.to_bytes(4, "little"))
        h.exec_instr(self.SW_INSTR)

        # 验证只有 hart 3 收到 MSIP
        for hid in range(4):
            expected = 1 if hid == 3 else 0
            assert clint._msip[hid] == expected, (
                f"_msip[{hid}] 应为 {expected}, 实际={clint._msip[hid]}"
            )
        has_pending, _, _ = clint.check_interrupt(3)
        assert has_pending, "hart 3 应有 pending MSIP"

    def test_msip_clear_via_mmio_and_verify_no_pending(self):
        """MMIO 清零 MSIP 后 mip CSR 通过 _update_hw_mip 反映已清零状态.

        回归: 若只清零 _msip 但 mip CSR 未同步, check_pending_interrupts
        仍会看到过时的 MSIP=1 -> 虚假中断投递 -> MSIP 风暴.
        """
        h, clint, bus = self._make_hart_with_clint(hart_id=0, num_harts=2)

        # Step 1: 通过 MMIO store 置位 MSIP[0]
        msip_addr = CLINT_BASE + MSIP_OFFSET
        h.write_gpr(self.BASE_REG, msip_addr)
        h.write_gpr(5, 1)
        bus.write(0x80000000, self.SW_INSTR.to_bytes(4, "little"))
        h.exec_instr(self.SW_INSTR)
        assert clint._msip[0] == 1

        # Step 2: 同步 mip CSR 从硬件
        has_pending, mip_bits, _ = clint.check_interrupt(0)
        _update_hw_mip(h, mip_bits)
        assert h.mip_val & (1 << 3), "mip.MSIP 应置位"

        # Step 3: 通过 MMIO store 清零 MSIP[0]
        h.pc = 0x80000004
        h.write_gpr(self.BASE_REG, msip_addr)  # 重置基址寄存器
        h.write_gpr(5, 0)  # value = 0
        bus.write(0x80000004, self.SW_INSTR.to_bytes(4, "little"))
        h.exec_instr(self.SW_INSTR)
        assert clint._msip[0] == 0, "MSIP 应已被 MMIO store 清零"

        # Step 4: 重新同步 mip CSR -> MSIP 必须为 0
        has_pending, mip_bits, _ = clint.check_interrupt(0)
        _update_hw_mip(h, mip_bits)
        assert not (h.mip_val & (1 << 3)), (
            f"清零后 mip.MSIP 必须为 0, 实际 mip={h.mip_val:#x}"
        )
        assert not has_pending, "check_interrupt 应返回 no pending"
