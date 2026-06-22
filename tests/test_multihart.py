#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""多 hart 集成测试: 验证多核启动 + 定时器中断."""

import struct

import pytest

from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import parse_firmware


@pytest.fixture
def multi_hart_emu():
    """2-hart emulator with kernel loaded."""
    cfg = PlatformConfig.qemu_virt()
    cfg.num_harts = 2
    emu = Emulator(cfg)
    kernel = parse_firmware("tests/bins/elf/kernel.elf")
    assert kernel is not None, "kernel.elf 解析失败"
    emu.load_firmware(kernel)
    return emu


class TestMultiHartBoot:
    """验证多 hart 基本启动."""

    def test_both_harts_boot_to_smode(self, multi_hart_emu):
        """两个 hart 都应从 M 模式进入 S 模式."""
        emu = multi_hart_emu
        for _ in range(500):
            emu.step()
            if all(h.mode.name == "S" for h in emu.harts):
                break

        for i, h in enumerate(emu.harts):
            assert h.mode.name == "S", f"Hart {i} 仍是 {h.mode.name}, 未进入 S"

    def test_harts_have_separate_csrs(self, multi_hart_emu):
        """每个 hart 有独立的 CSR 状态."""
        emu = multi_hart_emu
        for _ in range(100):
            emu.step()

        # 两个 hart 执行相同代码, satp 应相同 (共享页表)
        satp0 = emu.harts[0].satp_val
        satp1 = emu.harts[1].satp_val
        assert satp0 == satp1, "共享页表, satp 应相同"

        # 独立 PC (可能因时序差异有偏移)
        pc0 = emu.harts[0].pc
        pc1 = emu.harts[1].pc
        assert 0x80000000 <= pc0 <= 0x80010000, f"Hart 0 PC=0x{pc0:08x}"
        assert 0x80000000 <= pc1 <= 0x80010000, f"Hart 1 PC=0x{pc1:08x}"


class TestMultiHartTimer:
    """验证定时器中断."""

    def test_clint_mtime_advances(self, multi_hart_emu):
        """CLINT mtime 应随 step() 推进."""
        emu = multi_hart_emu
        mtime_before = emu.clint.get_mtime()
        for _ in range(100):
            emu.step()
        mtime_after = emu.clint.get_mtime()
        assert mtime_after == mtime_before + 100, (
            f"mtime 应推进 100, 实际 {mtime_after - mtime_before}"
        )

    def test_mtimecmp_writable(self, multi_hart_emu):
        """mtimecmp 可通过总线写入."""
        emu = multi_hart_emu
        # 写入 hart 0 的 mtimecmp (CLINT_BASE + 0x4000)
        val = 50000
        emu.bus.write(0x02004000, struct.pack("<Q", val))
        raw = emu.bus.try_read(0x02004000, 8)
        assert raw is not None, "无法读取 mtimecmp"
        assert int.from_bytes(raw, "little") == val, "mtimecmp 回读不匹配"

    def test_timer_interrupt_signaled(self, multi_hart_emu):
        """当 mtime >= mtimecmp > 0 时 MTIP 应置位."""
        emu = multi_hart_emu
        # 等待内核完成 mtimecmp 初始化 (boot 期间设为 -1)
        for _ in range(300):
            emu.step()
        # 获取当前 mtime, 设置 mtimecmp = mtime + 50
        mtime = emu.clint.get_mtime()
        emu.bus.write(0x02004000, struct.pack("<Q", mtime + 50))
        # 运行超过 mtimecmp
        for _ in range(100):
            emu.step()
        # 检查 hart 0 的 mip[MTIP] (硬件设置, 不受 CSR 写影响)
        # mip 位由 CLINT 硬件直接驱动, 读取 mip CSR 验证
        _ = emu.clint.check_interrupt(0)  # 触发 CLINT 更新
        mip = emu.harts[0].csrs["mip"].val
        assert mip & (1 << 7), f"MTIP 未置位, mip=0x{mip:x}"
