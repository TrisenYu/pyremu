"""WFI 被跨核 MSIP 唤醒的正确性测试 — 最小多核场景."""

import struct

import pytest

from pyremu.core.hart import RiscvMode
from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig

CLINT_BASE = 0x02000000


class TestWFIMsipWakeup:
    """H0 写 MSIP → H1 从 WFI 唤醒."""

    @pytest.fixture
    def emu_two_harts(self):
        cfg = PlatformConfig(num_harts=2, ram_base=0x80000000, ram_size=64 * 1024 * 1024)
        emu = Emulator(cfg, bootargs="")
        return emu

    @pytest.fixture
    def emu_two_harts_native(self):
        """启用 native batch 构造的双 hart Emulator (读取 native diag 计数器的用例需要)。

        测试默认 PYREMU_NATIVE_BATCH=0; 此处构造时临时置 "1" 使 _native_states 就绪,
        构造后由 conftest autouse 复位。"""
        import os
        cfg = PlatformConfig(num_harts=2, ram_base=0x80000000, ram_size=64 * 1024 * 1024)
        prev = os.environ.get("PYREMU_NATIVE_BATCH")
        os.environ["PYREMU_NATIVE_BATCH"] = "1"
        try:
            emu = Emulator(cfg, bootargs="")
        finally:
            os.environ["PYREMU_NATIVE_BATCH"] = prev if prev is not None else "0"
        return emu

    @staticmethod
    def _setup_harts(emu):
        for h in emu.harts:
            h.mode = RiscvMode.M
            h.satp_val = 0
            mtvec = 0x80000000 + h.id * 0x100
            h.csrs["mtvec"].val = mtvec
            emu.bus.write(mtvec, struct.pack("<I", 0x30200073))  # mret
            h.csrs["mie"].val = 1 << 3   # MSIE
            h.csrs["mstatus"].val = 1 << 3  # MIE=1

    def test_msip_wakes_wfi_hart_native(self, emu_two_harts):
        """原生 batch 路径: MSIP 唤醒 WFI hart.

        验证:
        1. H1 进入 WFI → state.waiting = 1
        2. H0 写 CLINT MSIP[1] = 1
        3. 下一个 step() 后 H1 不再 waiting, 且已处理中断
        """
        emu = emu_two_harts
        self._setup_harts(emu)
        h0, h1 = emu.harts[0], emu.harts[1]

        # 在 H1 的 PC 位置写入 WFI 指令 + 自旋
        wfi_loop = struct.pack("<I", 0x10500073)  # wfi
        emu.bus.write(0x80000100, wfi_loop)       # H1 PC
        h1.pc = 0x80000100

        # H0 先运行几步, 让 H1 有机会执行 WFI
        # step() 会让两个 hart 各执行一条指令
        for _ in range(5):
            emu.step()

        # H1 应该已经执行了 WFI 进入 waiting
        assert h1._waiting, f"H1 should be in WFI state, got waiting={h1._waiting}"

        # H0 写 CLINT MSIP[1] = 1
        msip1_pa = CLINT_BASE + 4
        emu.bus.write(msip1_pa, struct.pack("<I", 1))

        # 再 step
        emu.step()

        # H1 应该已被唤醒 (waiting=False)
        assert not h1._waiting, (
            f"H1 should have woken up from WFI after MSIP, "
            f"but waiting={h1._waiting}, mode={h1.mode.name}, "
            f"pc={h1.pc:#010x}, mip={h1.mip_val:#x}, mie={h1.mie_val:#x}"
        )

    def test_msip_wakes_wfi_hart_pure_python(self, emu_two_harts):
        """纯 Python 路径: MSIP 唤醒 WFI hart."""
        import os
        os.environ["PYREMU_NATIVE_BATCH"] = "0"
        try:
            self.test_msip_wakes_wfi_hart_native(emu_two_harts)
        finally:
            os.environ["PYREMU_NATIVE_BATCH"] = "0"

    def test_no_msip_wfi_stays_waiting(self, emu_two_harts):
        """无 MSIP 时 WFI 保持等待."""
        emu = emu_two_harts
        self._setup_harts(emu)
        h0, h1 = emu.harts[0], emu.harts[1]
        # H0 做 NOP 循环 (避免在 PC=0 触发非法指令 trap)
        emu.bus.write(0x80000200, struct.pack("<I", 0x00000013))  # nop
        h0.pc = 0x80000200
        # H1 执行 WFI
        emu.bus.write(0x80000100, struct.pack("<I", 0x10500073))  # wfi
        h1.pc = 0x80000100
        for _ in range(5):
            emu.step()
        assert h1._waiting, f"H1 should still be in WFI (no MSIP sent), got waiting={h1._waiting}"
        assert h1.pc == 0x80000104, f"H1 PC should advance past WFI, got {h1.pc:#010x}"

    @pytest.mark.skip(
        reason="diagnostic 计数器需 Rust 编译时启用 --features diagnostic"
    )
    def test_msip_sets_counter(self, emu_two_harts_native):
        """验证 diag_clint_msip_set 计数器正确递增 (native batch 路径)."""
        emu = emu_two_harts_native
        self._setup_harts(emu)
        h0, h1 = emu.harts[0], emu.harts[1]
        # 直接从 Python 写 CLINT MSIP[1]
        emu.bus.write(CLINT_BASE + 4, struct.pack("<I", 1))
        emu.step()
        # 从 native state 读取计数器
        ns = emu._native_states
        assert ns[1].diag.clint_msip_set >= 1, (
            f"H1 diag_clint_msip_set should be >= 1, got {ns[1].diag.clint_msip_set}"
        )
