#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""多 hart 集成测试: 验证多核启动 + 定时器中断."""

import os
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
        """CLINT mtime 应随 step() 推进 — 每 hart 每指令 1 tick."""
        emu = multi_hart_emu
        num_harts = len(emu.harts)
        mtime_before = emu.clint.get_mtime()
        for _ in range(100):
            emu.step()
        mtime_after = emu.clint.get_mtime()
        expected = mtime_before + num_harts * 100
        assert mtime_after == expected, (
            f"mtime 应推进 {num_harts * 100}, 实际 {mtime_after - mtime_before}"
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


class TestNativeBatchShortSlice:
    """验证 Rust native batch 的 short-slice 机制防止 IPI 自旋死锁.

    当 Hart 0 通过 CLINT MSIP 向 Hart 1 发送跨核中断后进入自旋等待,
    short-slice 机制限制发送核的指令预算, 使接收核获得更多 CPU 时间
    来完成 IPI 触发的工作.
    """

    def test_native_batch_no_double_halt_on_boot(self):
        """原生 batch 多核启动不会因 MSIP 风暴导致 halted.

        回归: 若 short-slice 机制缺失, Hart 0 发送 MSIP 后自旋消耗全部
        时间片, Hart 1 无法响应 ->MSIP 重复投递 ->连续 trap 检测触发 halted.
        """
        old_val = os.environ.get("PYREMU_NATIVE_BATCH")
        os.environ["PYREMU_NATIVE_BATCH"] = "1"
        try:
            cfg = PlatformConfig.qemu_virt()
            cfg.num_harts = 2
            emu = Emulator(cfg)
            kernel = parse_firmware("tests/bins/elf/kernel.elf")
            assert kernel is not None, "kernel.elf 解析失败"
            emu.load_firmware(kernel)

            # Run enough steps to complete M→S boot transition.
            # kernel.elf is small; 200 steps should suffice.
            for _ in range(200):
                emu.step()
                if any(h._halted for h in emu.harts):
                    hid = next(i for i, h in enumerate(emu.harts) if h._halted)
                    pytest.fail(
                        f"Hart {hid} halted during multi-hart boot "
                        f"(insn: H0={emu.harts[0]._total_instrs} "
                        f"H1={emu.harts[1]._total_instrs})"
                    )

            # Both harts should have made progress.
            in0 = emu.harts[0]._total_instrs
            in1 = emu.harts[1]._total_instrs
            assert in0 > 0, f"Hart 0 未执行指令"
            assert in1 > 0, f"Hart 1 未执行指令"

            # No hart should be stuck in a trap loop (MSIP storm check).
            for i, h in enumerate(emu.harts):
                ct = h._consecutive_traps
                assert ct < 3, (
                    f"Hart {i} consecutive_traps={ct} >= 3 — "
                    f"疑似 MSIP 风暴 (mode={h.mode.name} pc=0x{h.pc:x})"
                )
        finally:
            if old_val is None:
                del os.environ["PYREMU_NATIVE_BATCH"]
            else:
                os.environ["PYREMU_NATIVE_BATCH"] = old_val
