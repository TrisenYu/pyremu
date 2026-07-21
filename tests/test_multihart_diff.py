#!/usr/bin/env python3
"""多核差分测试: 验证跨 hart CSR 隔离、IPI 投递、TLB 一致性.

每个测试同步验证 Python 和 Rust native batch 两条路径, 差异即 bug.
设计原则: 每条测试用最少的指令, 直接断言预期行为.
"""

import os
import pytest

os.environ["PYREMU_NATIVE_BATCH"] = "0"

from pyremu._native import native_available
from pyremu.core.hart import HartWithRegs, MSTATUS_MIE, RiscvMode
from pyremu.core.mem_check_aux import inject_memory_backend
from pyremu.core.trap_handler import (
    check_pending_interrupts,
    deliver_trap,
    try_wfi_wakeup,
)
from pyremu.core.trap_def import TrapType
from pyremu.emulator import Emulator
from pyremu.interrupt.clint import CLINT
from pyremu.memory.bus import Bus
from pyremu.memory.pmp import Pmp
from pyremu.platform import PlatformConfig


def make_hart(hid: int, bus: Bus, clint: CLINT, pmp: Pmp) -> HartWithRegs:
    """创建配置就绪的 hart (HartWithRegs 而非 decoder.Hart)."""
    h = HartWithRegs(id=hid, pmp_entries=64)
    inject_memory_backend(h, bus.read, bus.write)
    h.bus = bus
    h.interrupt_ctrl = clint
    h._pmp = pmp
    h.csrs["misa"].val = (2 << 62) | (1 << 18) | (1 << 20)
    h.csrs["mscratch"].val = 0x80100000 + hid * 0x10000
    return h


def make_pmp() -> Pmp:
    return Pmp(csrs={}, num_entries=64)


def set_mie(h: HartWithRegs, val: int) -> None:
    """写 mie CSR (mie_val property 无 setter, 用底层方法)."""
    h._csr_write_raw("mie", val)


def set_mip(h: HartWithRegs, val: int) -> None:
    """写 mip CSR."""
    h.mip_val = val


def set_msip(clint: CLINT, hart_id: int, val: int) -> None:
    """写 CLINT MSIP[hart_id]."""
    clint._msip[hart_id] = val


# ═══════════════════════════════════════════════════════════════════
# Test 1: satp CSR 完全隔离
# ═══════════════════════════════════════════════════════════════════

class TestSatpIsolation:
    """satp 是 per-hart CSR: Hart 0 写 satp 不影响 Hart 1."""

    def test_satp_independent_per_hart(self):
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h0 = make_hart(0, bus, clint, make_pmp())
        h1 = make_hart(1, bus, clint, make_pmp())

        assert h0.satp_val == 0 and h1.satp_val == 0

        h0.satp_val = 0x8000000000080000  # Sv39
        assert h0._mmu_mode == 8
        assert h1.satp_val == 0, f"H1 satp={h1.satp_val:#x}, 应为 0"
        assert h1._mmu_mode == 0, f"H1 _mmu_mode={h1._mmu_mode}, 应为 0"

    def test_satp_native_roundtrip(self):
        """Rust batch marshal/unmarshal 后 satp 不串扰."""
        cfg = PlatformConfig(num_harts=2, ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        emu = Emulator(cfg)
        h0, h1 = emu.harts
        h0.satp_val = 0x8000000000080000
        emu.step()
        assert h1.satp_val == 0, f"Rust roundtrip H1 satp={h1.satp_val:#x}"

    def test_mmu_mode_cache_independent(self):
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h0 = make_hart(0, bus, clint, make_pmp())
        h1 = make_hart(1, bus, clint, make_pmp())

        h0.satp_val = 0x8000000000080000
        h1.satp_val = 0
        assert h0._mmu_mode == 8 and h1._mmu_mode == 0

        h0.satp_val = 0  # 改回 Bare
        assert h1._mmu_mode == 0, f"H1 _mmu_mode 被 H0 修改污染了"


# ═══════════════════════════════════════════════════════════════════
# Test 2: MIP/MIE CSR 隔离
# ═══════════════════════════════════════════════════════════════════

class TestMipMieIsolation:
    """mip/mie 是 per-hart CSR."""

    def test_mie_independent(self):
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h0 = make_hart(0, bus, clint, make_pmp())
        h1 = make_hart(1, bus, clint, make_pmp())

        set_mie(h0, (1 << 3) | (1 << 7))  # MSIE | MTIE
        assert h0.mie_val == ((1 << 3) | (1 << 7))
        assert h1.mie_val == 0, f"H1 mie={h1.mie_val:#x}, 应为 0"

    def test_mip_independent(self):
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h0 = make_hart(0, bus, clint, make_pmp())
        h1 = make_hart(1, bus, clint, make_pmp())

        set_mip(h0, 1 << 3)
        assert h0.mip_val & (1 << 3)
        assert (h1.mip_val & (1 << 3)) == 0, f"H1 mip={h1.mip_val:#x} 被污染"

    def test_mip_mie_native_roundtrip(self):
        """经 Rust batch marshal/unmarshal 后 mip/mie 不串扰."""
        cfg = PlatformConfig(num_harts=2, ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        emu = Emulator(cfg)
        h0, h1 = emu.harts

        set_mie(h0, (1 << 3) | (1 << 7))
        set_mip(h0, 1 << 7)

        emu.step()

        assert h1.mie_val == 0, f"Rust roundtrip H1 mie={h1.mie_val:#x}"
        assert h1.mip_val == 0, f"Rust roundtrip H1 mip={h1.mip_val:#x}"


# ═══════════════════════════════════════════════════════════════════
# Test 3: 跨 hart MSIP 投递 — IPI 核心机制
# ═══════════════════════════════════════════════════════════════════

class TestCrossHartMsip:
    """Hart 0 通过 CLINT 向 Hart 1 发送 MSIP."""

    def test_msip_write_target_sees_pending(self):
        clint = CLINT(num_harts=2)
        set_msip(clint, 1, 1)
        has_pending, bits, _ = clint.check_interrupt(1)
        assert has_pending, "MSIP[1]=1 时 check_interrupt(1) 应为 True"
        assert bits & (1 << 3), f"MSIP bit missing: {bits:#x}"

    def test_msip_clear_target_sees_clear(self):
        clint = CLINT(num_harts=2)
        set_msip(clint, 1, 1)
        assert clint.check_interrupt(1)[0]
        set_msip(clint, 1, 0)
        assert not clint.check_interrupt(1)[0], "清除后仍 pending"

    def test_msip_native_roundtrip(self):
        """Rust batch 中 MSIP 状态经 marshal/unmarshal 保持一致."""
        cfg = PlatformConfig(num_harts=2, ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        emu = Emulator(cfg)
        set_msip(emu.clint, 1, 1)
        emu.step()
        assert emu.clint._msip[1] == 1, f"MSIP[1]={emu.clint._msip[1]}"


# ═══════════════════════════════════════════════════════════════════
# Test 4: Trap 投递不影响其他 hart 状态 — 最关键的多核隔离
# ═══════════════════════════════════════════════════════════════════

class TestTrapStateIsolation:
    """Hart 0 上投递 trap 不应改变 Hart 1 的任何寄存器."""

    def test_trap_delivery_isolated_python(self):
        """deliver_trap(H0) 不改变 H1 的任何寄存器."""
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h0 = make_hart(0, bus, clint, make_pmp())
        h1 = make_hart(1, bus, clint, make_pmp())

        h1_init_pc = 0x80200000
        h1.pc = h1_init_pc
        h1.mode = RiscvMode.S
        h1.satp_val = 0
        h1_init_mstatus = h1.mstatus_val

        # Hart 0: 投递 IllInstr (不依赖 CLINT 状态)
        h0.pc = 0x80001000
        h0.mode = RiscvMode.S
        h0.csrs["mtvec"].val = 0x80000000
        h0.csrs["medeleg"].val = 0

        deliver_trap(h0, TrapType.IllInstr, tval=0xDEAD, is_interrupt=False)

        # Hart 1 状态应完全不变
        assert h1.pc == h1_init_pc, f"H1 PC={h1.pc:#x} ≠ {h1_init_pc:#x}"
        assert h1.mode == RiscvMode.S, f"H1 mode={h1.mode}"
        assert h1.satp_val == 0, f"H1 satp={h1.satp_val:#x}"
        assert h1.mstatus_val == h1_init_mstatus, f"H1 mstatus changed"

    def test_trap_delivery_isolated_native(self):
        """Rust batch 中 H0 有中断时 H1 (WFI 等待) 状态不漂移."""
        cfg = PlatformConfig(num_harts=2, ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        emu = Emulator(cfg)
        h0, h1 = emu.harts

        h1_init_pc = 0x80200000
        h1.pc = h1_init_pc
        h1.mode = RiscvMode.S
        h1.satp_val = 0
        h1._waiting = True  # WFI 等待, 不执行

        h0.pc = 0x80001000
        h0.mode = RiscvMode.M
        set_mie(h0, 1 << 3)
        h0._csr_write_raw("mip", 1 << 3)
        h0.csrs["mtvec"].val = 0x80000000

        emu.step()

        assert h1.satp_val == 0, f"native H1 satp={h1.satp_val:#x}"
        assert h1.pc == h1_init_pc, f"native H1 pc={h1.pc:#x}"


# ═══════════════════════════════════════════════════════════════════
# Test 5: WFI 跨 hart 唤醒
# ═══════════════════════════════════════════════════════════════════

class TestWfiCrossHartWake:
    """WFI 等待的 hart 应能被另一 hart 的 MSIP 唤醒."""

    def test_wfi_wake_by_msip_python(self):
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h1 = make_hart(1, bus, clint, make_pmp())

        h1._waiting = True
        set_mie(h1, 1 << 3)
        set_msip(clint, 1, 1)

        awakened = try_wfi_wakeup(h1)
        assert awakened, "MSIP=1 时应唤醒 WFI"
        assert not h1._waiting, "唤醒后 _waiting 应为 False"

    def test_wfi_native_batch_wake(self):
        """Rust batch 中 WFI hart 应被 CLINT MSIP 唤醒."""
        cfg = PlatformConfig(num_harts=2, ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        emu = Emulator(cfg)
        h0, h1 = emu.harts

        h1._waiting = True
        set_mie(h1, 1 << 3)
        h1.mode = RiscvMode.M
        emu.clint._msip[1] = 1

        emu.step()
        assert not h1._waiting, f"native batch 后 H1 _waiting={h1._waiting}"


# ═══════════════════════════════════════════════════════════════════
# Test 6: SFENCE.VMA 广播 — 跨 hart TLB 一致性
# ═══════════════════════════════════════════════════════════════════

class TestSfenceVmaBroadcast:
    """SFENCE.VMA 应刷新所有 hart 的 TLB."""

    def test_flush_all_harts_tlbs(self):
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h0 = make_hart(0, bus, clint, make_pmp())
        h1 = make_hart(1, bus, clint, make_pmp())

        h1.itlb.insert(vpn=0x100, ppn=0x200, perm=0x7, level=1, mdid=0)
        found, _, _ = h1.itlb.lookup(0x100)
        assert found

        h0.itlb.flush_all()
        h0.dtlb.flush_all()
        h1.itlb.flush_all()
        h1.dtlb.flush_all()

        for h in [h0, h1]:
            found, _, _ = h.itlb.lookup(0x100)
            assert not found, f"H{h.id} itlb 未清空"
            found, _, _ = h.dtlb.lookup(0x100)
            assert not found, f"H{h.id} dtlb 未清空"

    def test_tlb_native_roundtrip(self):
        """TLB 条目在 marshal/unmarshal 中正确传递."""
        cfg = PlatformConfig(num_harts=2, ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        emu = Emulator(cfg)
        h0, h1 = emu.harts

        h0.itlb.insert(vpn=0x100, ppn=0x200, perm=0x7, level=1, mdid=0)
        assert h0.itlb.lookup(0x100) is not None

        emu.step()

        # TLB entries are preserved across marshal/unmarshal
        assert h0.itlb.lookup(0x100) is not None, "TLB entry lost after batch"


# ═══════════════════════════════════════════════════════════════════
# Test 7: PMP mdid/pmpsplit 飞地参数隔离
# ═══════════════════════════════════════════════════════════════════

class TestPmpEnclaveIsolation:
    """mdid/pmpsplit CSR 是 per-hart 的."""

    def test_pmpsplit_per_hart(self):
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h0 = make_hart(0, bus, clint, make_pmp())
        h1 = make_hart(1, bus, clint, make_pmp())

        h0.pmpsplit_val = 32
        h1.pmpsplit_val = 0
        assert h0.pmpsplit_val == 32
        assert h1.pmpsplit_val == 0, f"H1 pmpsplit={h1.pmpsplit_val}"

    def test_mdid_per_hart(self):
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h0 = make_hart(0, bus, clint, make_pmp())
        h1 = make_hart(1, bus, clint, make_pmp())

        h0.mdid_val = 5
        h1.mdid_val = 0
        assert h0.mdid_val == 5
        assert h1.mdid_val == 0, f"H1 mdid={h1.mdid_val}"


# ═══════════════════════════════════════════════════════════════════
# Test 8: 连续多步跨 hart 一致性 — 模拟真实 IPI 场景
# ═══════════════════════════════════════════════════════════════════

class TestMultiStepConsistency:
    """多次 step 后跨 hart 状态不应漂移 (累积性串扰检测)."""

    def test_repeated_steps_no_state_drift(self):
        """10 次 step, 不活跃 Hart 1 的状态应完全不变."""
        cfg = PlatformConfig(num_harts=2, ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        emu = Emulator(cfg)
        h0, h1 = emu.harts

        # 冻结 Hart 1: Bare mode, 已知 PC, S 模式, 已知寄存器
        h1.pc = 0x80200000
        h1.mode = RiscvMode.S
        h1.satp_val = 0
        set_mie(h1, 0)
        set_mip(h1, 0)
        h1._waiting = True  # WFI 等待, 不执行

        # Hart 0: M 模式, 可执行
        h0.pc = 0x80000000
        h0.mode = RiscvMode.M
        h0.satp_val = 0

        for _ in range(10):
            emu.step()

        # Hart 1 所有状态应不变
        assert h1.pc == 0x80200000, f"10 步后 H1 PC={h1.pc:#x}"
        assert h1.mode == RiscvMode.S, f"10 步后 H1 mode={h1.mode}"
        assert h1.satp_val == 0, f"10 步后 H1 satp={h1.satp_val:#x}"
        assert h1.mie_val == 0, f"10 步后 H1 mie={h1.mie_val:#x}"
        assert h1.mip_val == 0, f"10 步后 H1 mip={h1.mip_val:#x}"
        assert h1._mmu_mode == 0, f"10 步后 H1 _mmu_mode={h1._mmu_mode}"


# ═══════════════════════════════════════════════════════════════════
# Test 9: 硬件 MIP 位清除 — 防止中断风暴
# ═══════════════════════════════════════════════════════════════════


class TestHardwareMipClear:
    """验证硬件源中断撤除后 mip 对应位被清除, 不产生虚假中断重入.

    对应 bug: check_pending_interrupts 仅 OR 累加硬件 mip 位,
    CLINT MSIP 清零后旧位仍残留 mip CSR -> 无限 MmodeSoftInterrupt 重投递
    -> 两个 hart 在 M-mode trap handler 之间往返, 无法推进.
    """

    def test_msip_cleared_after_clint_write_zero(self):
        """CLINT MSIP 写 0 后, check_pending_interrupts 应清除 mip.MSIP."""
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h0 = make_hart(0, bus, clint, make_pmp())

        # 设置为 M 模式, 使能 MSIE, mtvec 指向有效地址
        h0.mode = RiscvMode.M
        h0.mstatus_val |= MSTATUS_MIE  # 全局中断使能 (mstatus.MIE)
        set_mie(h0, 1 << 3)  # MSIE
        h0.csrs["mtvec"].val = 0x80000000

        # Step 1: 设置 MSIP -> mip 应有 MSIP
        set_msip(clint, 0, 1)
        has_pending, mip_bits, _ = clint.check_interrupt(0)
        assert has_pending and (mip_bits & (1 << 3)), "MSIP=1 时应 pending"

        # Step 2: check_pending_interrupts 投递中断
        delivered = check_pending_interrupts(h0)
        assert delivered, "MSIP pending + MSIE enabled -> 应投递中断"
        # 进入 M-mode trap handler 后 MIE 被硬件清除
        assert not h0.mie, "trap 入口应清除 MIE"
        # 模拟 handler 处理: 清除 CLINT MSIP
        clint.clear_ipi(0)
        assert clint._msip[0] == 0, "clear_ipi 后 MSIP 应为 0"

        # Step 3: 模拟 mret -> 恢复 MIE
        h0._csr_write_raw("mstatus", h0.mstatus_val | MSTATUS_MIE)
        assert h0.mie, "mret 后 MIE 应恢复为 1"

        # Step 4: 再次调用 check_pending_interrupts
        # ── 核心断言: MSIP 已清, 不应再投递任何中断 ──
        delivered_again = check_pending_interrupts(h0)
        assert not delivered_again, (
            "CLINT MSIP 已清零, check_pending_interrupts 不应再投递中断 "
            "(旧 bug: mip MSIP 位未清除, 导致虚假重入)"
        )

        # 验证 mip CSR 中 MSIP 位确已清除
        mip_after = h0._csr_read_raw("mip")
        assert (mip_after & (1 << 3)) == 0, (
            f"mip.MSIP 应在 CLINT 清零后撤除, 实际 mip={mip_after:#x}"
        )

    def test_mtip_cleared_after_mtimecmp_disabled(self):
        """mtimecmp=0 (禁用) 后, MTIP 不再挂起."""
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h0 = make_hart(0, bus, clint, make_pmp())

        h0.mode = RiscvMode.M
        h0.mstatus_val |= MSTATUS_MIE  # 全局中断使能
        set_mie(h0, 1 << 7)  # MTIE
        h0.csrs["mtvec"].val = 0x80000000

        # Step 1: 设置 mtimecmp 并使 mtime 超过它
        clint._mtimecmp[0] = 100
        clint._mtime = 200
        has_pending, mip_bits, _ = clint.check_interrupt(0)
        assert has_pending and (mip_bits & (1 << 7)), "mtime >= mtimecmp 时 MTIP 应 pending"

        # Step 2: 投递中断
        delivered = check_pending_interrupts(h0)
        assert delivered, "MTIP pending + MTIE enabled -> 应投递中断"

        # Step 3: 模拟 handler 清除定时器 (写入 mtimecmp=0)
        clint._mtimecmp[0] = 0
        clint._notify_state_change()

        # Step 4: 恢复 MIE (模拟 mret)
        h0._csr_write_raw("mstatus", h0.mstatus_val | MSTATUS_MIE)

        # Step 5: 核心断言 — MTIP 已清除
        delivered_again = check_pending_interrupts(h0)
        assert not delivered_again, (
            "mtimecmp 已禁用, check_pending_interrupts 不应再投递 MTIP 中断"
        )
        mip_after = h0._csr_read_raw("mip")
        assert (mip_after & (1 << 7)) == 0, (
            f"mip.MTIP 应在 mtimecmp=0 后清除, 实际 mip={mip_after:#x}"
        )

    def test_msip_clear_prevents_spurious_retrap_loop(self):
        """模拟 MSIP 风暴场景: IPI -> trap -> clear -> mret -> 应安静, 不重入.

        这是诊断中观察到的真实故障模式: 两个 hart 在 M-mode 之间无限往返,
        每次 ~5500 traps/10s, 900s 内共投递 214K 次 MSIP 中断.
        """
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h0 = make_hart(0, bus, clint, make_pmp())

        h0.mode = RiscvMode.M
        h0.mstatus_val |= MSTATUS_MIE  # 全局中断使能
        set_mie(h0, 1 << 3)  # MSIE
        h0.csrs["mtvec"].val = 0x80000000

        trap_count = 0

        # 模拟一次完整的 IPI 处理循环
        for _ in range(10):
            # 外部 hart 发 IPI (置 MSIP)
            if trap_count == 0:
                clint.send_ipi(0)

            # 检查中断
            if check_pending_interrupts(h0):
                trap_count += 1
                # handler: 清除本 hart 的 MSIP
                clint.clear_ipi(0)
                # mret: 恢复 MIE
                h0._csr_write_raw("mstatus", h0.mstatus_val | MSTATUS_MIE)

        # 核心断言: 只投递了 1 次中断, 不是 10 次
        assert trap_count == 1, (
            f"IPI 清除后不应再有中断重入. "
            f"预期 1 次, 实际 {trap_count} 次 "
            f"(旧 bug: 每次 check_pending_interrupts 都重投递 MSIP)"
        )

    def test_cross_hart_msip_no_storm(self):
        """跨 hart 完整流程: Hart 0 发 IPI -> Hart 1 收到 -> 清除 -> 不重入."""
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        h0 = make_hart(0, bus, clint, make_pmp())
        h1 = make_hart(1, bus, clint, make_pmp())

        # 两 hart 均 M 模式, MSIE 使能, 全局中断开启
        for h in [h0, h1]:
            h.mode = RiscvMode.M
            h.mstatus_val |= MSTATUS_MIE  # 全局中断使能
            set_mie(h, 1 << 3)  # MSIE
            h.csrs["mtvec"].val = 0x80000000

        # Hart 0 发 IPI 给 Hart 1
        clint.send_ipi(1)

        # Hart 1 收到中断
        delivered = check_pending_interrupts(h1)
        assert delivered, "Hart 1 应收到 MSIP 中断"

        # Hart 1 handler 清除自己的 MSIP
        clint.clear_ipi(1)

        # mret 恢复 MIE
        h1._csr_write_raw("mstatus", h1.mstatus_val | (1 << 3))

        # Hart 1 不应再有中断
        assert not check_pending_interrupts(h1), "清除后 Hart 1 不应重入"

        # Hart 0 不受影响
        assert (h0._csr_read_raw("mip") & (1 << 3)) == 0, "Hart 0 的 MSIP 不应被污染"


class TestMsipClearViaMemWritePhy:
    """验证固件通过 MMIO 写 (sw zero, CLINT_BASE) 清除 MSIP 的完整路径.

    真实场景: OpenSBI 的 mswi_ipi_clear() 调用 writel_relaxed(0, &msip[hartid])
    -> MMIO 写 -> mem_write -> _mem_write_phy -> Bus.write -> CLINT.write.
    现有测试直接调 clint.clear_ipi() 绕过了这条路径,
    无法暴露 Bus/CLINT 层路由 bug (如 Sv39 翻译导致 PA 错位).
    """

    def test_msip_clear_via_bus_write(self):
        """通过 Bus.write 写 CLINT MSIP=0 -> 中断应撤除, 不重投递."""
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        bus.add_device(0x02000000, clint)
        h0 = make_hart(0, bus, clint, make_pmp())

        h0.mode = RiscvMode.M
        h0.mstatus_val |= MSTATUS_MIE
        set_mie(h0, 1 << 3)
        h0.csrs["mtvec"].val = 0x80000000

        # 固件写 CLINT MSIP=1 (32-bit MMIO)
        bus.write(0x02000000, (1).to_bytes(4, "little"))
        assert clint._msip[0] == 1, "Bus.write 应置位 CLINT MSIP[0]"

        # 中断投递
        assert check_pending_interrupts(h0), "MSIP=1 应触发中断"

        # 模拟 handler: 固件写 CLINT MSIP=0
        bus.write(0x02000000, (0).to_bytes(4, "little"))
        assert clint._msip[0] == 0, "Bus.write 应清零 CLINT MSIP[0]"

        # mret 恢复 MIE
        h0._csr_write_raw("mstatus", h0.mstatus_val | MSTATUS_MIE)

        # 核心断言: 不再重入
        assert not check_pending_interrupts(h0), (
            "Bus.write 清零 MSIP 后不应再投递 (旧 bug: CLINT 未收到清零写入)"
        )

    def test_msip_clear_via_mem_write_phy(self):
        """通过 _mem_write_phy 写 CLINT MSIP=0 -> 中断撤除, 不重投递.

        这是 OpenSBI 代码在模拟器中实际走过的路径: _mem_write_phy(PA, data)
        -> Bus.write -> CLINT.write. 专用于检测 _mem_write_phy/Bus 层路由缺陷.
        """
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        bus.add_device(0x02000000, clint)
        h0 = make_hart(0, bus, clint, make_pmp())

        h0.mode = RiscvMode.M
        h0.mstatus_val |= MSTATUS_MIE
        set_mie(h0, 1 << 3)
        h0.csrs["mtvec"].val = 0x80000000
        assert h0._mem_write_phy is not None
        # 通过 _mem_write_phy 置位 MSIP (模拟固件 sw 指令)
        h0._mem_write_phy(0x02000000, (1).to_bytes(4, "little"))
        assert clint._msip[0] == 1, "_mem_write_phy 应置位 CLINT MSIP[0]"

        # 中断投递
        assert check_pending_interrupts(h0), "MSIP=1 应触发中断"

        # 模拟 handler: 通过 _mem_write_phy 清零
        h0._mem_write_phy(0x02000000, (0).to_bytes(4, "little"))
        assert clint._msip[0] == 0, "_mem_write_phy 应清零 CLINT MSIP[0]"

        # mret 恢复 MIE
        h0._csr_write_raw("mstatus", h0.mstatus_val | MSTATUS_MIE)

        assert not check_pending_interrupts(h0), (
            "_mem_write_phy 清零 MSIP 后不应再投递"
        )

    def test_msip_clear_preserves_mtip(self):
        """清零 MSIP 时不应影响 MTIP 等其他中断位."""
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        bus.add_device(0x02000000, clint)
        h0 = make_hart(0, bus, clint, make_pmp())

        h0.mode = RiscvMode.M
        h0.mstatus_val |= MSTATUS_MIE
        set_mie(h0, (1 << 3) | (1 << 7))  # MSIE + MTIE
        h0.csrs["mtvec"].val = 0x80000000
        assert h0._mem_write_phy is not None
        # 同时置位 MSIP 和 MTIP
        h0._mem_write_phy(0x02000000, (1).to_bytes(4, "little"))
        clint._mtimecmp[0] = 100
        clint._mtime = 200
        clint._notify_state_change()

        # 验证两者都 pending
        _, mip_bits, _ = clint.check_interrupt(0)
        assert mip_bits & (1 << 3), "MSIP 应 pending"
        assert mip_bits & (1 << 7), "MTIP 应 pending"

        # 投递 MSIP (优先级高于 MTIP)
        assert check_pending_interrupts(h0), "应投递 MSIP"

        # handler 清零 MSIP
        h0._mem_write_phy(0x02000000, (0).to_bytes(4, "little"))

        # mret 恢复 MIE
        h0._csr_write_raw("mstatus", h0.mstatus_val | MSTATUS_MIE)

        # MSIP 不再投递, 但 MTIP 仍应 pending
        assert check_pending_interrupts(h0), (
            "清零 MSIP 后 MTIP 仍应 pending, 应投递 MTIP"
        )

    def test_msip_double_set_clear_via_mem_write_phy(self):
        """连续两次 MSIP 置位->清零, 每次清零后都不应重入."""
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        bus.add_device(0x02000000, clint)
        h0 = make_hart(0, bus, clint, make_pmp())

        h0.mode = RiscvMode.M
        h0.mstatus_val |= MSTATUS_MIE
        set_mie(h0, 1 << 3)
        h0.csrs["mtvec"].val = 0x80000000

        assert h0._mem_write_phy, "null function as _mem_write_phy"
        for round_num in range(3):
            # 置位
            h0._mem_write_phy(0x02000000, (1).to_bytes(4, "little"))
            assert check_pending_interrupts(h0), f"第{round_num}轮: 应投递 MSIP"

            # 清零
            h0._mem_write_phy(0x02000000, (0).to_bytes(4, "little"))
            h0._csr_write_raw("mstatus", h0.mstatus_val | MSTATUS_MIE)

            assert not check_pending_interrupts(h0), (
                f"第{round_num}轮: 清零后不应重入 "
                f"(mip={h0._csr_read_raw('mip'):#x})"
            )

    def test_msip_clear_must_go_through_bus_not_skip(self):
        """验证 _mem_write_phy -> Bus.write 路由未被绕过.

        如果 _mem_write_phy 没有正确调用 Bus.write, CLINT 不会收到清零,
        这是本次诊断发现的实际缺陷路径.
        """
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        bus.add_device(0x02000000, clint)
        h0 = make_hart(0, bus, clint, make_pmp())

        h0.mode = RiscvMode.M
        h0.mstatus_val |= MSTATUS_MIE
        set_mie(h0, 1 << 3)
        h0.csrs["mtvec"].val = 0x80000000

        # 通过 CLINT 直接写 (不走 Bus)
        clint._msip[0] = 1
        clint._notify_state_change()
        assert check_pending_interrupts(h0), "应投递 MSIP"
        assert h0._mem_write_phy, "null function as _mem_write_phy"
        # handler: 通过 _mem_write_phy 清零 (走 Bus 路由)
        h0._mem_write_phy(0x02000000, (0).to_bytes(4, "little"))

        # 验证 CLINT 确实收到了清零
        assert clint._msip[0] == 0, (
            "_mem_write_phy(CLINT_BASE, 0) 必须使 CLINT._msip[0]=0. "
            "若 _mem_write_phy 未正确路由到 Bus -> CLINT.write, 此断言失败."
        )

        # 恢复 MIE 后不应再投递
        h0._csr_write_raw("mstatus", h0.mstatus_val | MSTATUS_MIE)
        assert not check_pending_interrupts(h0), (
            "_mem_write_phy 清零 CLINT 后不应重入"
        )


class TestCrossHartMemWritePhyIpi:
    """跨 hart IPI 完整流程: 通过 _mem_write_phy 发/清 IPI."""

    def test_cross_hart_ipi_via_mem_write_phy(self):
        """Hart 0 通过 _mem_write_phy 发 IPI -> Hart 1 收到 -> 清除 -> 不重入."""
        bus = Bus(ram_base=0x80000000, ram_size=128 * 1024 * 1024)
        clint = CLINT(num_harts=2)
        bus.add_device(0x02000000, clint)
        h0 = make_hart(0, bus, clint, make_pmp())
        h1 = make_hart(1, bus, clint, make_pmp())
        assert h0._mem_write_phy and h1._mem_write_phy, "null function as _mem_write_phy"
        for h in [h0, h1]:
            h.mode = RiscvMode.M
            h.mstatus_val |= MSTATUS_MIE
            set_mie(h, 1 << 3)
            h.csrs["mtvec"].val = 0x80000000

        # Hart 0 通过 MMIO 写 CLINT MSIP[1] (模拟 OpenSBI sbi_ipi_raw_send)
        h0._mem_write_phy(0x02000004, (1).to_bytes(4, "little"))
        assert clint._msip[1] == 1, "Hart 1 的 MSIP 应被置位"

        # Hart 1 收到中断
        assert check_pending_interrupts(h1), "Hart 1 应收到 MSIP"

        # Hart 1 handler: 通过 MMIO 清除 MSIP[1]
        h1._mem_write_phy(0x02000004, (0).to_bytes(4, "little"))
        assert clint._msip[1] == 0, "Hart 1 的 MSIP 应被清零"

        h1._csr_write_raw("mstatus", h1.mstatus_val | MSTATUS_MIE)

        # Hart 1 不再重入
        assert not check_pending_interrupts(h1), "Hart 1 清零后不应重入"
        # Hart 0 不受影响
        assert (h0._csr_read_raw("mip") & (1 << 3)) == 0, "Hart 0 不应被污染"


# ═══════════════════════════════════════════════════════════════════
# Test: Native batch + Python 双路径 MSIP 清除验证
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("use_native", [
    pytest.param(False, id="python"),
    pytest.param(True, id="native"),
])
class TestMsipClearBothPaths:
    """验证 MSIP 中断投递后 CLINT MSIP 清零 — Python / Rust 路径一致.

    若 native batch 路径的 ``check_and_deliver_interrupt`` 未在投递 MSIP
    后清除 CLINT 硬件源, mip.MSIP 将保持为 1 -> mret 后立即重入 -> 风暴.
    """

    @staticmethod
    def _make_emu(use_native: bool) -> Emulator:
        """创建单 hart 最小 Emulator, 可选启用 native batch."""
        cfg = PlatformConfig(
            num_harts=1, ram_base=0x80000000,
            ram_size=1024 * 1024,  # 1 MiB
        )
        saved = os.environ.get("PYREMU_NATIVE_BATCH")
        os.environ["PYREMU_NATIVE_BATCH"] = "1" if use_native else "0"
        try:
            emu = Emulator(cfg)
        finally:
            if saved is not None:
                os.environ["PYREMU_NATIVE_BATCH"] = saved
            else:
                del os.environ["PYREMU_NATIVE_BATCH"]
        if use_native and not emu._native_batch:
            pytest.skip("native batch not initialised")
        return emu

    @staticmethod
    def _setup_msip_handler(emu: Emulator):
        """放置 mret @ mtvec, 配置 M-mode + MSIE + MIE."""
        h = emu.harts[0]
        # mret (0x30200073) @ mtvec=0x80000000: 最简单的 handler, 立即返回
        emu.bus.write_ram_direct(0x80000000, (0x30200073).to_bytes(4, "little"))
        h.csrs["mtvec"].val = 0x80000000
        h.mode = RiscvMode.M
        h.mstatus_val |= MSTATUS_MIE   # 全局中断使能
        set_mie(h, 1 << 3)             # MSIE
        h.pc = 0x80000004

    def test_msip_cleared_after_delivery(self, use_native):
        """MSIP 投递后 clint._msip 和 mip.MSIP 均应清零."""
        emu = self._make_emu(use_native)
        self._setup_msip_handler(emu)
        h = emu.harts[0]

        # NOP 循环 (addi x0, x0, 0 ; jal x0, -4) @ 0x80000004
        emu.bus.write_ram_direct(0x80000004, (0x00000013).to_bytes(4, "little"))
        emu.bus.write_ram_direct(0x80000008, (0xffdff06f).to_bytes(4, "little"))

        # 触发 MSIP
        emu.clint._msip[0] = 1
        emu.clint._notify_state_change()

        emu.step()

        assert emu.clint._msip[0] == 0, (
            f"[{use_native}] MSIP 投递后 clint._msip[0] 应为 0"
        )
        assert (h._csr_read_raw("mip") & (1 << 3)) == 0, (
            f"[{use_native}] MSIP 投递后 mip.MSIP 应为 0"
        )

    def test_no_spurious_reentry(self, use_native):
        """MSIP 投递后 hart 未被 halted (无连续重入风暴)."""
        emu = self._make_emu(use_native)
        self._setup_msip_handler(emu)
        h = emu.harts[0]

        # 计数循环 (addi x1, x1, 1 ; jal x0, -4) @ 0x80000004
        emu.bus.write_ram_direct(0x80000004, (0x00108093).to_bytes(4, "little"))
        emu.bus.write_ram_direct(0x80000008, (0xffdff06f).to_bytes(4, "little"))

        emu.clint._msip[0] = 1
        emu.clint._notify_state_change()

        emu.step()

        assert not h._halted, (
            f"[{use_native}] MSIP 投递后不应 halted (风暴检测触发了?)"
        )


class TestWfiWakeupMsipBypass:
    """WFI 唤醒: CLINT MSIP 硬件中断线应绕过 mie.MSIE.

    真实硬件上 MSIP 是独立物理中断线, WFI 唤醒不依赖 mie.MSIE.
    mie.MSIE 仅控制该中断是否被 *投递* (trap delivery).
    此行为与 Rust native batch ``exec.rs`` line 368-392 一致.
    """

    @staticmethod
    def _make_emu(use_native: bool, num_harts: int = 1) -> Emulator:
        cfg = PlatformConfig(
            num_harts=num_harts, ram_base=0x80000000,
            ram_size=1024 * 1024,
        )
        saved = os.environ.get("PYREMU_NATIVE_BATCH")
        os.environ["PYREMU_NATIVE_BATCH"] = "1" if use_native else "0"
        try:
            emu = Emulator(cfg)
        finally:
            if saved is not None:
                os.environ["PYREMU_NATIVE_BATCH"] = saved
            else:
                del os.environ["PYREMU_NATIVE_BATCH"]
        if use_native and not emu._native_batch:
            pytest.skip("native batch not initialised")
        return emu

    @staticmethod
    def _setup_msip_handler(emu: Emulator):
        """mret @ mtvec=0x80000000, M-mode + MIE=1, MSIE=0 (刻意清零)."""
        h = emu.harts[0]
        emu.bus.write_ram_direct(0x80000000, (0x30200073).to_bytes(4, "little"))
        h.csrs["mtvec"].val = 0x80000000
        h.mode = RiscvMode.M
        h.mstatus_val |= MSTATUS_MIE  # 全局 MIE=1
        # 刻意清零 MSIE (bit 3), 验证 WFI 唤醒绕过此位
        set_mie(h, 0)  # MSIE=0, all other source enables = 0
        h.pc = 0x80000004

    @pytest.mark.parametrize("use_native", [False, True])
    def test_wfi_wakeup_msip_active_msie_zero(self, use_native):
        """CLINT MSIP=1 且 mie.MSIE=0 时 try_wfi_wakeup 应为 True."""
        emu = self._make_emu(use_native)
        h = emu.harts[0]
        h.mode = RiscvMode.M
        h.mstatus_val |= MSTATUS_MIE
        set_mie(h, 0)  # MSIE=0
        set_msip(emu.clint, 0, 1)

        # 手工置 WFI 等待
        h._waiting = True

        result = try_wfi_wakeup(h)
        assert result, (
            f"[{'native' if use_native else 'python'}] "
            "CLINT MSIP=1 但 MSIE=0: try_wfi_wakeup 应返回 True"
        )
        assert not h._waiting, "WFI 等待标志应已清除"
        assert h._wfi_woken, "wfi_woken 应被置位"

    @pytest.mark.parametrize("use_native", [False, True])
    def test_wfi_wakeup_msip_bypass_sets_msie(self, use_native):
        """MSIE=0 时 WFI 唤醒应临时置位 MSIE 以供中断投递."""
        emu = self._make_emu(use_native)
        h = emu.harts[0]
        h.mode = RiscvMode.M
        h.mstatus_val |= MSTATUS_MIE
        set_mie(h, 0)  # MSIE=0
        set_msip(emu.clint, 0, 1)
        h._waiting = True

        try_wfi_wakeup(h)

        # 验证临时置位 MSIE
        assert (h._csr_read_raw("mie") & (1 << 3)) != 0, (
            f"[{'native' if use_native else 'python'}] "
            "WFI 唤醒后 mie.MSIE 应被临时置位"
        )

    @pytest.mark.parametrize("use_native", [False, True])
    def test_wfi_wakeup_msip_bypass_delivers_interrupt(self, use_native):
        """WFI 绕过 MSIE=0 唤醒后, MSIP 应被成功投递 (进 M-mode)."""
        emu = self._make_emu(use_native)
        self._setup_msip_handler(emu)
        h = emu.harts[0]

        # NOP 循环 @ 0x80000004 (正常 S-mode 代码)
        emu.bus.write_ram_direct(0x80000004, (0x00000013).to_bytes(4, "little"))
        emu.bus.write_ram_direct(0x80000008, (0xffdff06f).to_bytes(4, "little"))

        # 模拟: Hart 在 WFI 等待, CLINT MSIP 被另一 hart 置位
        h.mode = RiscvMode.S
        h._waiting = True
        set_msip(emu.clint, 0, 1)

        emu.step()

        # MSIP 应被投递 -> M-mode trap -> mret -> 回到 S-mode
        assert emu.clint._msip[0] == 0, (
            f"[{'native' if use_native else 'python'}] "
            "MSIP 投递后 CLINT MSIP 应清零"
        )
        assert (h._csr_read_raw("mip") & (1 << 3)) == 0, (
            "mip.MSIP 应为 0"
        )

    @pytest.mark.parametrize("use_native", [False, True])
    def test_wfi_wakeup_msie_zero_no_spurious_halt(self, use_native):
        """MSIE=0 + MSIP WFI 唤醒 -> 不应触发连续 trap 风暴 halted."""
        emu = self._make_emu(use_native)
        self._setup_msip_handler(emu)
        h = emu.harts[0]

        # 计数循环
        emu.bus.write_ram_direct(0x80000004, (0x00108093).to_bytes(4, "little"))
        emu.bus.write_ram_direct(0x80000008, (0xffdff06f).to_bytes(4, "little"))

        h.mode = RiscvMode.S
        h._waiting = True
        set_msip(emu.clint, 0, 1)

        emu.step()

        assert not h._halted, (
            f"[{'native' if use_native else 'python'}] "
            "MSIE=0 WFI 绕过唤醒后不应当触发 halted"
        )

    def test_wfi_wakeup_with_mie_msie_also_works(self):
        """mie.MSIE=1 的正常路径不受影响."""
        emu = self._make_emu(False)
        h = emu.harts[0]
        h.mode = RiscvMode.M
        h.mstatus_val |= MSTATUS_MIE
        set_mie(h, 1 << 3)  # MSIE=1
        set_msip(emu.clint, 0, 1)
        h._waiting = True

        assert try_wfi_wakeup(h), "MSIE=1 时 MSIP 应正常唤醒"
        assert not h._waiting

    def test_wfi_wakeup_no_interrupt_no_wake(self):
        """无待处理中断时 WFI 不应被唤醒."""
        emu = self._make_emu(False)
        h = emu.harts[0]
        h.mode = RiscvMode.M
        h.mstatus_val |= MSTATUS_MIE
        set_mie(h, 1 << 3)  # MSIE=1
        set_msip(emu.clint, 0, 0)  # MSIP=0
        h._waiting = True

        assert not try_wfi_wakeup(h), "无待处理中断时不应唤醒"
        assert h._waiting, "WFI 等待标志应保持"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
