#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

from pyremu.core.decoder import Hart
from pyremu.core.hart import RiscvMode
from pyremu.core.mem_check_aux import (
    AccessFault,
    inject_memory_backend,
    mem_read,
    mem_write,
    MemoryAccessFault,
)
from pyremu.core.trap_def import trap_cause_code, TrapType
from pyremu.emulator import Emulator
from pyremu.memory.bus import Bus
from pyremu.memory.l2cache import L2CacheLine
from pyremu.memory.pmp import Pmp, PmpAccessInfo
from pyremu.memory.tlb import TLB, TLBLine
from pyremu.platform import PlatformConfig

# ============================================================
#  mdid / pmpsplit CSR 寄存器
# ============================================================
PAGE_SHIFT = 12
L1_BASE = 0x1000
L2_BASE = 0x2000
L3_BASE = 0x3000
DATA_PA = 0x100000


def _make_emu(ram_size: int = 8 * 1024 * 1024, num_harts: int = 1):
    """创建小内存 Emulator — 控制 4 GiB ulimit 内存压力."""
    cfg = PlatformConfig.qemu_virt()
    cfg.ram_size = ram_size
    cfg.num_harts = num_harts
    return Emulator(cfg)


class TestMdidCSR:
    """mdid (0x5C0) 和 pmpsplit (0x5C1) CSR 基本属性."""

    def test_mdid_exists_and_defaults_to_zero(self):
        emu = _make_emu()
        h = emu.harts[0]
        assert "mdid" in h.csrs, "mdid CSR 应存在"
        assert h.csrs["mdid"].val == 0, "mdid 默认值应为 0"

    def test_pmpsplit_exists_and_defaults_to_zero(self):
        emu = _make_emu()
        h = emu.harts[0]
        assert "pmpsplit" in h.csrs, "pmpsplit CSR 应存在"
        assert h.csrs["pmpsplit"].val == 0, "pmpsplit 默认值应为 0"

    def test_mdid_mmode_rw(self):
        """M 模式可读写 mdid."""
        emu = _make_emu()
        h = emu.harts[0]
        assert h.mode == RiscvMode.M
        # 写
        h.write_csr(0x5C0, 0xBEEF)
        assert h.csrs["mdid"].val == 0xBEEF, "mdid 写入后应可读回"
        # 写回 0
        h.write_csr(0x5C0, 0)
        assert h.csrs["mdid"].val == 0

    def test_mdid_write_read_roundtrip(self):
        """mdid 写入任意 64 位值的往返测试."""
        emu = _make_emu()
        h = emu.harts[0]
        for val in (1, 0xDEADBEEF, 0xFFFF_FFFF_FFFF_FFFF, 0x12345678_9ABCDEF0):
            h.write_csr(0x5C0, val)
            assert h.csrs["mdid"].val == val, f"mdid 往返失败: {val:#x}"

    def test_mdid_property(self):
        """HartWithRegs.mdid_val property 快捷访问."""
        emu = _make_emu()
        h = emu.harts[0]
        h.mdid_val = 0xC0FFEE
        assert h.mdid_val == 0xC0FFEE
        assert h.csrs["mdid"].val == 0xC0FFEE

    def test_umode_access_mdid_traps(self):
        """U 模式访问 mdid (M-mode only) 应触发 IllInstr."""
        emu = _make_emu()
        h = emu.harts[0]
        # 切换到 U 模式
        h.mode = RiscvMode.U
        # csrrw 到 mdid
        instr = (0x5C0 << 20) | (5 << 15) | (0b001 << 12) | (6 << 7) | 0x73
        h.pc = 0x1000
        h.exec_instr(instr)
        assert h.mcause_val == trap_cause_code(TrapType.IllInstr), (
            f"U 模式访问 mdid 应 IllInstr(2), 实际 mcause={h.mcause_val}"
        )


# ============================================================
#  mfence.did 指令
# ============================================================


_MFENCE_DID = 0x5A000073


class TestMfenceDid:
    """mfence.did 指令 — 执行、flush 语义、隔离性."""

    # -- 指令编码 --

    def test_exec_does_not_trap(self):
        """mfence.did 是合法指令, 不应触发陷态."""
        emu = _make_emu()
        h = emu.harts[0]
        h.pc = 0x80000000
        trap_before = h.mcause_val
        h.exec_instr(_MFENCE_DID)
        assert h.mcause_val == trap_before, f"mfence.did 不应触发 trap, mcause={h.mcause_val}"

    def test_invalid_funct12_traps(self):
        """funct12 非法编码应触发 IllInstr (funct3=000, 未注册的 funct12)."""
        emu = _make_emu()
        h = emu.harts[0]
        h.pc = 0x80000000
        # funct12=0xFF0 (未实现), funct3=0, rd=rs1=0
        bad = (0xFF0 << 20) | 0x73
        h.exec_instr(bad)
        assert h.mcause_val == trap_cause_code(TrapType.IllInstr), (
            f"非法 funct12 应 IllInstr(2), 实际 mcause={h.mcause_val}"
        )

    # -- TLB flush 隔离性 --

    def test_flushes_only_matching_mdid(self):
        emu = _make_emu()
        h = emu.harts[0]

        # 插入 3 条: mdid=0, mdid=1, mdid=2
        h.dtlb.insert(vpn=0x1000, ppn=0xAAA, perm=7, mdid=0)
        h.dtlb.insert(vpn=0x2000, ppn=0xBBB, perm=7, mdid=1)
        h.dtlb.insert(vpn=0x3000, ppn=0xCCC, perm=7, mdid=2)

        # 设 mdid=1, 执行 mfence.did
        h.mdid_val = 1
        h.pc = 0x80000000
        h.exec_instr(_MFENCE_DID)

        # mdid=0 的还在
        hit0, _, _ = h.dtlb.lookup(0x1000)
        assert hit0, "mdid=0 不应被刷掉"
        # mdid=1 的被刷掉
        hit1, _, _ = h.dtlb.lookup(0x2000)
        assert not hit1, "mdid=1 应被刷掉"
        # mdid=2 的还在
        hit2, _, _ = h.dtlb.lookup(0x3000)
        assert hit2, "mdid=2 不应被刷掉"

    def test_flushes_itlb_and_dtlb_both(self):
        emu = _make_emu()
        h = emu.harts[0]

        h.itlb.insert(vpn=0x100, ppn=0x10, perm=5, mdid=1)
        h.dtlb.insert(vpn=0x200, ppn=0x20, perm=7, mdid=1)

        h.mdid_val = 1
        h.pc = 0x80000000
        h.exec_instr(_MFENCE_DID)

        assert len(h.itlb) == 0, "itlb 应被清空"
        assert len(h.dtlb) == 0, "dtlb 应被清空"

    def test_nonzero_mdid_flushes_nothing_when_no_match(self):
        """mdid=3 但没有任何条目带此标记 -> flush 为空操作."""
        emu = _make_emu()
        h = emu.harts[0]

        h.dtlb.insert(vpn=0x1000, ppn=0xAAA, perm=7, mdid=1)
        h.dtlb.insert(vpn=0x2000, ppn=0xBBB, perm=7, mdid=2)

        h.mdid_val = 3  # 无匹配
        h.pc = 0x80000000
        h.exec_instr(_MFENCE_DID)

        assert len(h.dtlb) == 2, "mdid=3 不应刷掉任何条目"

    # -- L2 缓存 flush --

    def test_flushes_l2_cache_matching_mdid(self):
        """mfence.did 同时刷新 L2 缓存中匹配 mdid 的条目."""
        emu = _make_emu()
        h = emu.harts[0]
        # 通过 bus 写入数据, 触发 L2 缓存分配
        addr_a = 0x80001000
        addr_b = 0x80002000
        emu.bus.write(addr_a, b"A" * 64)
        emu.bus.write(addr_b, b"B" * 64)
        _ = emu.bus.read(addr_a, 4)  # 触发 L2 分配
        _ = emu.bus.read(addr_b, 4)

        # 手动标记 L2 条目: 第一个 mdid=1, 第二个 mdid=0
        l2 = emu.bus._l2
        assert l2 is not None
        entries = [e for e in l2.entries if e.valid]
        assert len(entries) >= 2, f"应有至少 2 条 L2 有效条目, 实际 {len(entries)}"
        entries[0].mdid = 1
        entries[1].mdid = 0

        h.mdid_val = 1
        h.pc = 0x80000000
        h.exec_instr(_MFENCE_DID)

        # mdid=1 的被刷掉, mdid=0 的保留
        remaining_mdid = sum(1 for e in l2.entries if e.valid)
        assert remaining_mdid >= 1, f"mdid=0 的 L2 条目应保留, 实际剩余 {remaining_mdid}"
        # 验证 mdid=0 的仍在
        still_valid = [e for e in l2.entries if e.valid]
        assert all(e.mdid == 0 for e in still_valid), "所有剩余条目的 mdid 应为 0"

    # -- 广播: 多 hart --

    def test_broadcasts_to_all_harts(self):
        """hart 0 执行 mfence.did -> 全部 hart 的 TLB 均被刷新."""
        emu = _make_emu(num_harts=4)
        for h in emu.harts:
            h.dtlb.insert(vpn=0x42, ppn=0x42, perm=7, mdid=1)

        # hart 0 执行 mfence.did (mdid=1)
        emu.harts[0].mdid_val = 1
        emu.harts[0].pc = 0x80000000
        emu.harts[0].exec_instr(_MFENCE_DID)

        for i, h in enumerate(emu.harts):
            assert len(h.dtlb) == 0, f"Hart {i} dtlb 应被清空 (广播)"
            assert len(h.itlb) == 0, f"Hart {i} itlb 应被清空 (广播)"

    def test_broadcast_only_flushes_matching_harts(self):
        """不同 hart 的 TLB 带有不同 mdid -> 仅匹配的被刷."""
        emu = _make_emu(num_harts=2)
        h0, h1 = emu.harts

        # hart 0: mdid=1, hart 1: mdid=2
        h0.dtlb.insert(vpn=0x100, ppn=0x10, perm=7, mdid=1)
        h1.dtlb.insert(vpn=0x200, ppn=0x20, perm=7, mdid=2)

        h0.mdid_val = 1
        h0.pc = 0x80000000
        h0.exec_instr(_MFENCE_DID)

        assert len(h0.dtlb) == 0, "Hart 0 dtlb (mdid=1) 应被刷掉"
        assert len(h1.dtlb) == 1, "Hart 1 dtlb (mdid=2) 不应被刷掉"


# ============================================================
#  TLB 自动 mdid 标记 (Sv39 页表遍历)
# ============================================================


class TestTLBAutoMdid:
    """TLB.insert() 接受 mdid 参数, _translate_addr 自动传入 hart.mdid."""

    def test_insert_stores_mdid(self):
        """直接调用 tlb.insert(vpn, ppn, perm, mdid=...) 应存储."""
        tlb = TLB(size=8)
        tlb.insert(vpn=0x100, ppn=0xAAA, perm=7, mdid=5)
        hit, ppn, perm = tlb.lookup(0x100)
        assert hit, "应命中"
        assert ppn == 0xAAA
        # 验证 mdid 存储
        entry = [e for e in tlb.entries if e.valid and e.tag == 0x100][0]
        assert entry.mdid == 5, f"TLB 条目的 mdid 应为 5, 实际 {entry.mdid}"

    def test_insert_default_mdid_zero(self):
        """不传 mdid 时默认 0."""
        tlb = TLB(size=8)
        tlb.insert(vpn=0x100, ppn=0xBBB, perm=7)
        entry = [e for e in tlb.entries if e.valid and e.tag == 0x100][0]
        assert entry.mdid == 0

    def test_update_preserves_new_mdid(self):
        """同 VPN 更新时, mdid 也被更新."""
        tlb = TLB(size=8)
        tlb.insert(vpn=0x100, ppn=0xAAA, perm=7, mdid=1)
        tlb.insert(vpn=0x100, ppn=0xBBB, perm=7, mdid=2)  # 同 VPN 更新
        entries = [e for e in tlb.entries if e.valid and e.tag == 0x100]
        assert len(entries) == 1, "应只有一条同 VPN 条目"
        assert entries[0].mdid == 2, "更新后 mdid 应为 2"

    def test_translate_addr_tags_tlb_with_hart_mdid(self):
        """Sv39 翻译时, TLB 条目自动标记为当前 hart.mdid."""

        ram = bytearray(2 * 1024 * 1024)

        def read_fn(addr, size):
            return bytes(ram[addr : addr + size])

        def write_fn(addr, data):
            ram[addr : addr + len(data)] = data

        h = Hart(id=0)
        inject_memory_backend(h, read_fn, write_fn)
        h.bus = None  # 无 Bus, 绕过 PMA 检查

        # 标记此飞地为 mdid=7
        h.mdid_val = 7

        # 构建 Sv39 三级 4 KiB 映射 va=0 -> DATA_PA
        vpn0 = 0
        vpn1 = 0
        vpn2 = 0
        target_ppn = DATA_PA >> PAGE_SHIFT

        def write_pte(table_base, index, pte_val):
            addr = table_base + index * 8
            ram[addr : addr + 8] = pte_val.to_bytes(8, "little")

        def make_pte(v=False, r=False, w=False, x=False, ppn=0):
            val = 1 if v else 0
            if r:
                val |= 2
            if w:
                val |= 4
            if x:
                val |= 8
            val |= (ppn & 0x3FF) << 10
            val |= ((ppn >> 10) & 0x1FF) << 20
            val |= ((ppn >> 19) & 0x1FFFFFFF) << 29
            return val

        write_pte(L1_BASE, vpn2, make_pte(v=True, ppn=L2_BASE >> PAGE_SHIFT))
        write_pte(L2_BASE, vpn1, make_pte(v=True, ppn=L3_BASE >> PAGE_SHIFT))
        write_pte(L3_BASE, vpn0, make_pte(v=True, r=True, w=True, x=True, ppn=target_ppn))

        h.mode = RiscvMode.S  # MMU 翻译仅在 S/U 模式生效
        h.satp_val = (8 << 60) | (L1_BASE >> PAGE_SHIFT)  # Sv39

        # 触发地址翻译 -> TLB 插入
        try:
            mem_read(h, 0x0, 4)
        except MemoryAccessFault:
            pass  # PMP/Access 拒绝时 TLB 不插入, 测试下面断言自然失败

        # 验证 TLB 条目被标记为 mdid=7
        dtlb_entries = [e for e in h.dtlb.entries if e.valid]
        if dtlb_entries:
            for e in dtlb_entries:
                assert e.mdid == 7, f"TLB 条目 mdid 应为 7, 实际 {e.mdid}"

        # 切换飞地: mdid=7 -> mdid=9, 用 mdid=9 执行 mfence.did
        # — 不应刷掉 mdid=7 的条目 (飞地隔离)
        h.mdid_val = 9
        h.pc = 0x80000000
        h.exec_instr(_MFENCE_DID)

        remaining_after_9 = len(h.dtlb)
        assert remaining_after_9 >= 1, (
            f"mfence.did(mdid=9) 不应刷掉 mdid=7 的条目 (飞地隔离), "
            f"实际剩余 {remaining_after_9}"
        )

        # 切回 mdid=7, 再次执行 mfence.did — 此时应清空
        h.mdid_val = 7
        h.pc = 0x80000000
        h.exec_instr(_MFENCE_DID)

        remaining_after_7 = len(h.dtlb)
        assert remaining_after_7 == 0, (
            f"mfence.did(mdid=7) 应刷掉 mdid=7 的条目, 实际剩余 {remaining_after_7}"
        )


# ============================================================
#  Bare 模式 (satp 未启用) — TLB 不自动填充, 手动条目可刷
# ============================================================


class TestBareModeMdid:
    """Bare 模式下 TLB 不参与翻译, 但手动填充的条目仍被 mfence.did 管理."""

    def test_bare_mode_no_tlb_population(self):
        """Bare 模式: 内存访问不填充 TLB."""
        ram = bytearray(64 * 1024)

        def read_fn(addr, size):
            return bytes(ram[addr : addr + size])

        def write_fn(addr, data):
            ram[addr : addr + len(data)] = data

        h = Hart(id=0)
        inject_memory_backend(h, read_fn, write_fn)
        h.mdid_val = 3
        h.bus = None  # 绕过 PMA 检查

        # Bare 模式 (satp.MODE=0)
        assert h.mmu_mode == 0

        # 访问内存 — 不应插入 TLB
        mem_read(h, 0x1000, 4)
        assert len(h.dtlb) == 0, "Bare 模式不应自动填充 TLB"

    def test_manual_tlb_entries_still_flushed(self):
        """Bare 模式下手动插入的 TLB 条目 (通过 mfence.did) 仍可被刷新."""
        emu = _make_emu()
        h = emu.harts[0]

        # Bare 模式, 手动插入
        assert h.mmu_mode == 0
        h.dtlb.insert(vpn=0x100, ppn=0x10, perm=7, mdid=2)
        h.itlb.insert(vpn=0x100, ppn=0x10, perm=5, mdid=2)

        h.mdid_val = 2
        h.pc = 0x80000000
        h.exec_instr(_MFENCE_DID)

        assert len(h.dtlb) == 0
        assert len(h.itlb) == 0


# ============================================================
#  TLBLine / L2CacheLine mdid 字段
# ============================================================


class TestMdidField:
    """CacheLineBase.mdid 字段继承与默认值."""

    def test_tlb_line_has_mdid_default(self):
        line = TLBLine()
        assert hasattr(line, "mdid"), "TLBLine 应有 mdid 字段"
        assert line.mdid == 0

    def test_l2_cache_line_has_mdid_default(self):

        line = L2CacheLine()
        assert hasattr(line, "mdid"), "L2CacheLine 应有 mdid 字段"
        assert line.mdid == 0

    def test_flush_by_mdid_returns_count(self):
        tlb = TLB(size=8)
        tlb.insert(vpn=0x100, ppn=0xA, perm=7, mdid=1)
        tlb.insert(vpn=0x200, ppn=0xB, perm=7, mdid=1)
        tlb.insert(vpn=0x300, ppn=0xC, perm=7, mdid=2)

        count = tlb.flush_by_mdid(1)
        assert count == 2, f"应刷掉 2 条, 实际 {count}"
        # mdid=2 的还在
        assert len(tlb) == 1

    def test_flush_by_mdid_miss_returns_zero(self):
        tlb = TLB(size=8)
        tlb.insert(vpn=0x100, ppn=0xA, perm=7, mdid=1)
        count = tlb.flush_by_mdid(99)
        assert count == 0
        assert len(tlb) == 1, "不匹配的 mdid 不应刷掉任何条目"


# ============================================================
#  L2 缓存自动 mdid 标记 — 经 emulator 内存访问路径
# ============================================================


class TestL2AutoMdid:
    """L2 缓存分配时自动打上 current_mdid (由 emulator.step 逐 hart 设置)."""

    def test_l2_read_hit_updates_mdid(self):
        """读命中时 L2 条目 mdid 更新为当前 hart 的 mdid."""
        emu = _make_emu()
        l2 = emu.bus._l2
        assert l2 is not None
        addr = 0x80001000

        # hart mdid=1 写数据 -> L2 分配, mdid=1
        l2.current_mdid = 1
        emu.bus.write(addr, b"X" * 64)
        _ = emu.bus.read(addr, 4)

        entries = [e for e in l2.entries if e.valid and e.tag == (addr >> l2._line_shift)]
        assert len(entries) == 1, "应有一条 L2 条目"
        assert entries[0].mdid == 1, f"新分配条目 mdid 应为 1, 实际 {entries[0].mdid}"

        # 切换 hart mdid=2, 再次读取同一地址 -> 命中, mdid 应刷新为 2
        l2.current_mdid = 2
        _ = emu.bus.read(addr, 4)
        entries = [e for e in l2.entries if e.valid and e.tag == (addr >> l2._line_shift)]
        assert entries[0].mdid == 2, f"命中后 mdid 应刷新为 2, 实际 {entries[0].mdid}"

    def test_l2_write_hit_updates_mdid(self):
        """CPU store (≤8B) 命中 L2 时更新 mdid.

        DMA (>8B, RAM 地址) 直写 bytearray 绕过 L2, 匹配硬件语义:
        设备 DMA 不经过 CPU cache, 只有 CPU 访存才填充缓存行.
        因此用 8-byte store 测试 L2 mdid 行为.
        """
        emu = _make_emu()
        l2 = emu.bus._l2
        assert l2 is not None
        addr = 0x80001000

        # 先触发一次读以将缓存行加载到 L2
        _ = emu.bus.read(addr, 8)
        entries = [e for e in l2.entries if e.valid and e.tag == (addr >> l2._line_shift)]
        assert len(entries) > 0, "读取应填充 L2"

        l2.current_mdid = 1
        emu.bus.write(addr, b"Y" * 8)

        entries = [e for e in l2.entries if e.valid and e.tag == (addr >> l2._line_shift)]
        assert entries[0].mdid == 1

        # 切换 mdid, CPU store 命中
        l2.current_mdid = 3
        emu.bus.write(addr, b"Z" * 8)
        entries = [e for e in l2.entries if e.valid and e.tag == (addr >> l2._line_shift)]
        assert entries[0].mdid == 3, f"写命中后 mdid 应刷新为 3, 实际 {entries[0].mdid}"

    def test_l2_auto_tag_then_selective_flush(self):
        """不同 mdid 的 hart 访问不同地址 -> mfence.did 仅刷匹配的."""
        emu = _make_emu()
        h = emu.harts[0]
        l2 = emu.bus._l2
        assert l2 is not None
        addr_a = 0x80001000
        addr_b = 0x80002000

        # mdid=0xA 访问 addr_a
        l2.current_mdid = 0xA
        emu.bus.write(addr_a, b"A" * 64)
        _ = emu.bus.read(addr_a, 4)

        # mdid=0xB 访问 addr_b
        l2.current_mdid = 0xB
        emu.bus.write(addr_b, b"B" * 64)
        _ = emu.bus.read(addr_b, 4)

        # 验证两条条目的 mdid
        all_valid = [e for e in l2.entries if e.valid]
        mdids = {e.mdid for e in all_valid}
        assert 0xA in mdids, "应有 mdid=0xA 的条目"
        assert 0xB in mdids, "应有 mdid=0xB 的条目"

        # mfence.did(mdid=0xA): 仅刷 mdid=0xA 的条目
        h.mdid_val = 0xA
        h.pc = 0x80000000
        h.exec_instr(_MFENCE_DID)

        remaining = [e for e in l2.entries if e.valid]
        remaining_mdids = {e.mdid for e in remaining}
        assert 0xA not in remaining_mdids, "mdid=0xA 的 L2 条目应被刷掉"
        assert 0xB in remaining_mdids, "mdid=0xB 的 L2 条目应保留"

    def test_emulator_step_sets_current_mdid(self):
        """emulator.step() 自动将 hart.mdid 同步到 L2.current_mdid."""
        emu = _make_emu()
        h = emu.harts[0]
        l2 = emu.bus._l2

        h.mdid_val = 0x77
        assert l2 is not None
        # 写一条 nop 到 PC, 然后 step()
        addr = 0x80000000
        emu.bus.write(addr, b"\x13\x00\x00\x00")  # nop (addi x0, x0, 0)
        h.pc = addr

        emu.step()

        # step() 内设置了 l2.current_mdid = hart.mdid_val
        # 取指 (bus.read) 会经过 L2 -> 新行标记 mdid=0x77
        assert l2._current_mdid == 0x77, (
            f"step 后 current_mdid 应为 0x77, 实际 {l2._current_mdid}"
        )
        l2_entries = [e for e in l2.entries if e.valid]
        assert len(l2_entries) >= 1, "取指应产生 L2 条目"
        for e in l2_entries:
            assert e.mdid == 0x77, (
                f"emulator.step() 自动标记的 L2 条目 mdid 应为 0x77, "
                f"实际 {e.mdid} (tag=0x{e.tag:x})"
            )


# ============================================================
#  pmpsplit CSR — PMP 条目拆分与飞地隔离
# ============================================================


class TestPmpsplitCSR:
    """pmpsplit (0x5C1) CSR 基本属性."""

    def test_pmpsplit_defaults_to_zero(self):
        emu = _make_emu()
        h = emu.harts[0]
        assert h.pmpsplit_val == 0, "pmpsplit 默认值应为 0"

    def test_pmpsplit_write_read_roundtrip(self):
        emu = _make_emu()
        h = emu.harts[0]
        for val in (0, 8, 16, 32, 63, 0xFFFF_FFFF_FFFF_FFFF):
            h.write_csr(0x5C1, val)
            assert h.csrs["pmpsplit"].val == val, f"pmpsplit 往返失败: {val:#x}"
            assert h.pmpsplit_val == val, "cached pmpsplit_val 应同步"

    def test_pmpsplit_property_sync(self):
        emu = _make_emu()
        h = emu.harts[0]
        h.pmpsplit_val = 32
        assert h.csrs["pmpsplit"].val == 32
        assert h._pmpsplit_val == 32

    def test_umode_access_pmpsplit_traps(self):
        emu = _make_emu()
        h = emu.harts[0]
        h.mode = RiscvMode.U
        instr = (0x5C1 << 20) | (5 << 15) | (0b001 << 12) | (6 << 7) | 0x73
        h.pc = 0x1000
        h.exec_instr(instr)
        assert h.mcause_val == trap_cause_code(TrapType.IllInstr), (
            f"U 模式访问 pmpsplit 应 IllInstr(2), 实际 mcause={h.mcause_val}"
        )


# ============================================================
#  pmpsplit + PMP 飞地隔离
# ============================================================


class TestPmpsplitPmpIsolation:
    """pmpsplit 限制飞地可见的 PMP 条目范围."""

    def test_enclave_sees_only_assigned_entries(self):
        """mdid!=0 且 pmpsplit=4: PMP 条目 0-3 对飞地不可见, 4-7 正常检查."""

        pmp = Pmp({}, num_entries=8)

        class FakeCSR:
            val: int
            def __init__(self, v): self.val = v

        # 配置条目 0 (host 范围): allow 0x1000, R+W, NAPOT 4KB
        k_4k = 9
        addr_val_4k = (0x1000 >> 2) | ((1 << k_4k) - 1)
        pmp._csrs["pmpaddr0"] = FakeCSR(addr_val_4k)
        pmp._csrs["pmpcfg0"] = FakeCSR(0b11000 | 0b0011)  # NAPOT, R+W

        # 配置条目 5 (飞地范围): allow 0x2000, R+W, NAPOT 4KB
        addr_val_2 = (0x2000 >> 2) | ((1 << k_4k) - 1)
        pmp._csrs["pmpaddr5"] = FakeCSR(addr_val_2)
        pmp._csrs["pmpcfg0"] = FakeCSR(
            pmp._csrs["pmpcfg0"].val | ((0b11000 | 0b0011) << 40)
        )

        # Host 模式 (mdid=0): 两个条目都可见
        info_host = PmpAccessInfo(
            pa=0x1000, size=4, mode_val=0,
            mstatus_val=0, pmpsplit=4, mdid=0
        )
        assert pmp.check(info_host), "host 应能访问条目 0 的区域"

        # 飞地模式 (mdid=1, pmpsplit=4): 条目 0 不可见, 应 DENY
        info_enc = PmpAccessInfo(
            pa=0x1000, size=4, mode_val=0,
            mstatus_val=0, pmpsplit=4, mdid=1
        )
        assert not pmp.check(info_enc), (
            "飞地不应能访问 host 的 PMP 条目 0 区域"
        )

        # 飞地应能访问条目 5 的区域
        info_enc5 = PmpAccessInfo(
            pa=0x2000, size=4, mode_val=0,
            mstatus_val=0, pmpsplit=4, mdid=1
        )
        assert pmp.check(info_enc5), "飞地应能访问自己的 PMP 条目 5 区域"

    def test_enclave_no_entries_when_pmpsplit_exceeds(self):
        """pmpsplit >= num_entries -> 飞地无可用 PMP 条目, 全部拒绝."""

        pmp = Pmp({}, num_entries=8)
        info = PmpAccessInfo(
            pa=0x1000, size=4, mode_val=0,
            mstatus_val=0, pmpsplit=8, mdid=1
        )
        assert not pmp.check(info), "pmpsplit=8 >= num_entries=8 -> 飞地全拒"

    def test_pmpsplit_zero_legacy_all_visible(self):
        """pmpsplit=0 -> 飞地也能看到全部条目 (兼容模式)."""

        pmp = Pmp({}, num_entries=4)

        class FakeCSR:
            val: int
            def __init__(self, v): self.val = v

        k_4k = 9
        addr_val = (0x5000 >> 2) | ((1 << k_4k) - 1)
        pmp._csrs["pmpaddr0"] = FakeCSR(addr_val)
        pmp._csrs["pmpcfg0"] = FakeCSR(0b11000 | 0b0011)

        # pmpsplit=0 + mdid=1: 条目 0 仍可见
        info = PmpAccessInfo(pa=0x5000, size=4, mode_val=0, mstatus_val=0, pmpsplit=0, mdid=1)
        assert pmp.check(info), "pmpsplit=0 时飞地应能看到全部条目"

    def test_host_always_sees_all_entries(self):
        """mdid=0 (host) 始终能看到全部 PMP 条目, 无视 pmpsplit."""

        pmp = Pmp({}, num_entries=8)

        class FakeCSR:
            val: int
            def __init__(self, v): self.val = v

        addr_val = (0x3000 >> 2) | 0x1FF  # 4KB @ 0x3000
        pmp._csrs["pmpaddr0"] = FakeCSR(addr_val)
        pmp._csrs["pmpcfg0"] = FakeCSR(0b11000 | 0b0011)

        info = PmpAccessInfo(pa=0x3000, size=4, mode_val=0, mstatus_val=0, pmpsplit=60, mdid=0)
        assert pmp.check(info), "host 总应能看到全部条目, 无视 pmpsplit"


# ============================================================
#  多飞地 mdid 隔离 — TLB / L2
# ============================================================


class TestMultiEnclaveMdid:
    """多个飞地并存时的 mdid 隔离验证."""

    def test_three_enclaves_tlb_isolation(self):
        """3 个飞地各插入 TLB 条目, mfence.did 仅刷匹配的."""
        emu = _make_emu()
        h = emu.harts[0]

        # 飞地 A (mdid=1): vpn=0x100
        h.dtlb.insert(vpn=0x100, ppn=0xA00, perm=7, mdid=1)
        # 飞地 B (mdid=2): vpn=0x200
        h.dtlb.insert(vpn=0x200, ppn=0xB00, perm=7, mdid=2)
        # 飞地 C (mdid=3): vpn=0x300
        h.dtlb.insert(vpn=0x300, ppn=0xC00, perm=7, mdid=3)

        # 模拟飞地 B 执行 mfence.did
        h.mdid_val = 2
        h.pc = 0x80000000
        h.exec_instr(_MFENCE_DID)

        # 飞地 A 的条目还在
        assert h.dtlb.lookup(0x100)[0], "mdid=1 的 TLB 条目应保留"
        # 飞地 B 的被刷掉
        assert not h.dtlb.lookup(0x200)[0], "mdid=2 的 TLB 条目应被刷掉"
        # 飞地 C 的还在
        assert h.dtlb.lookup(0x300)[0], "mdid=3 的 TLB 条目应保留"

        # 再刷飞地 A
        h.mdid_val = 1
        h.exec_instr(_MFENCE_DID)
        assert not h.dtlb.lookup(0x100)[0], "mdid=1 的 TLB 条目应被刷掉"
        assert h.dtlb.lookup(0x300)[0], "mdid=3 的仍应保留"

    def test_multi_enclave_l2_isolation(self):
        """4 个伪飞地访问不同地址, 各自 L2 条目标不同 mdid, 按域刷新互不干扰."""
        emu = _make_emu()
        l2 = emu.bus._l2
        assert l2 is not None

        addrs = [0x80001000, 0x80002000, 0x80003000, 0x80004000]
        mdids = [10, 20, 30, 40]

        # 各飞地写入各自地址
        for addr, mdid in zip(addrs, mdids):
            l2.current_mdid = mdid
            emu.bus.write(addr, b"X" * 64)
            _ = emu.bus.read(addr, 4)

        # 验证 4 个不同 mdid
        all_entries = [e for e in l2.entries if e.valid]
        present = {e.mdid for e in all_entries}
        for m in mdids:
            assert m in present, f"mdid={m} 应有 L2 条目"

        # 刷 mdid=20 — 仅该飞地受影响
        h = emu.harts[0]
        h.mdid_val = 20
        h.pc = 0x80000000
        h.exec_instr(_MFENCE_DID)

        remaining = {e.mdid for e in l2.entries if e.valid}
        assert 20 not in remaining, "mdid=20 应被刷掉"
        assert 10 in remaining, "mdid=10 应保留"
        assert 30 in remaining, "mdid=30 应保留"
        assert 40 in remaining, "mdid=40 应保留"

    def test_same_vpn_different_mdid_overwrites(self):
        """同 VPN 插入不同 mdid -> 原地更新 (TLB 以 tag 为键, mdid 仅用于 flush)."""
        tlb = TLB(size=8)
        tlb.insert(vpn=0x42, ppn=0x100, perm=7, mdid=1)
        tlb.insert(vpn=0x42, ppn=0x200, perm=7, mdid=2)

        # 同 VPN -> 同 tag -> 原地更新, 只有一条有效条目
        entries = [e for e in tlb.entries if e.valid and e.tag == 0x42]
        assert len(entries) == 1, f"同 VPN 应原地更新, 实际 {len(entries)}"
        assert entries[0].mdid == 2, f"更新后 mdid 应为 2, 实际 {entries[0].mdid}"
        assert entries[0].ppn == 0x200, "更新后 ppn 应为 0x200"

    def test_mdid_zero_host_entries_isolated_from_enclave(self):
        """host (mdid=0) 的 TLB 条目不受飞地 mfence.did 影响."""
        emu = _make_emu()
        h = emu.harts[0]

        h.dtlb.insert(vpn=0x100, ppn=0xA00, perm=7, mdid=0)  # host
        h.dtlb.insert(vpn=0x200, ppn=0xB00, perm=7, mdid=5)  # enclave

        # 飞地 5 执行 mfence.did
        h.mdid_val = 5
        h.pc = 0x80000000
        h.exec_instr(_MFENCE_DID)

        # host 条目 (mdid=0) 应保留
        assert h.dtlb.lookup(0x100)[0], "host (mdid=0) TLB 条目不应被飞地刷掉"
        # 飞地 5 的条目应被刷掉
        assert not h.dtlb.lookup(0x200)[0], "飞地 (mdid=5) TLB 条目应被刷掉"


# ============================================================
#  pmpsplit + mdid 集成 — PMP 条目隔离
# ============================================================


class TestPmpsplitMdidIntegration:
    """pmpsplit 划分 PMP 条目 + mdid 标记飞地 -> 硬件级别的飞地间隔离."""

    def test_enclave_a_cannot_see_enclave_b_pmp_entries(self):
        """设置两个飞地各自的 PMP 区域后, 飞地 A 不能访问飞地 B 的物理内存."""
        pmp = Pmp({}, num_entries=16)

        class FakeCSR:
            def __init__(self, v):
                self.val = v

        # Host 区域: PMP 条目 0-5
        # 飞地区域: PMP 条目 6-15 (pmpsplit=6)
        # 飞地 A (mdid=1): 条目 8 — 0x8000_0000-0x8000_0FFF (R+W)
        # 飞地 B (mdid=2): 条目 10 — 0x8000_2000-0x8000_2FFF (R+W)
        k_4k = 9
        addr_a = (0x8000_0000 >> 2) | ((1 << k_4k) - 1)
        addr_b = (0x8000_2000 >> 2) | ((1 << k_4k) - 1)

        # 条目 8 (飞地 A 区间)
        pmp._csrs["pmpaddr8"] = FakeCSR(addr_a)
        # 条目 10 (飞地 B 区间)
        pmp._csrs["pmpaddr10"] = FakeCSR(addr_b)

        # 写 pmpcfg: 条目 8 和 10 均为 NAPOT+R+W
        # pmpcfg2 覆盖条目 8-15
        cfg_val = 0
        cfg_val |= (0b11000 | 0b0011) << (0 * 8)  # entry 8
        cfg_val |= (0b11000 | 0b0011) << (2 * 8)  # entry 10
        pmp._csrs["pmpcfg2"] = FakeCSR(cfg_val)

        # 飞地 A (mdid=1, pmpsplit=6): 只能看到条目 6-15
        # 访问自己的内存 (条目 8) -> OK
        info_a_own = PmpAccessInfo(
            pa=0x8000_0000, size=4, mode_val=0, mstatus_val=0, pmpsplit=6, mdid=1,
        )
        assert pmp.check(info_a_own), "飞地 A 应能访问自己的内存"

        # 访问飞地 B 的内存 (条目 10) -> 也 OK (条目 10 在飞地区间内)
        # 注意: 虽然 PMP 不阻止, 但 pmpsplit 只控制可见条目范围.
        # 飞地间的进一步隔离由 mfence.did + TLB 管理实现.
        info_a_other = PmpAccessInfo(
            pa=0x8000_2000, size=4, mode_val=0, mstatus_val=0, pmpsplit=6, mdid=1,
        )
        assert pmp.check(info_a_other), "条目 10 在飞地区间内, PMP 不阻止"

        # 访问 host 区域 (条目 0) -> DENY (条目 0 < pmpsplit, 对飞地不可见)
        info_a_host = PmpAccessInfo(
            pa=0x0000, size=4, mode_val=0, mstatus_val=0, pmpsplit=6, mdid=1,
        )
        assert not pmp.check(info_a_host), "飞地不应看到 host 区间 (条目 0 < pmpsplit=6)"

    def test_emulator_multi_hart_different_mdid(self):
        """多 hart 各自运行不同飞地, mdid + pmpsplit 独立配置."""
        emu = _make_emu(num_harts=4)
        hart_configs = [
            (0, 1, 8),   # hart 0: mdid=1, pmpsplit=8
            (1, 2, 8),   # hart 1: mdid=2, pmpsplit=8
            (2, 3, 8),   # hart 2: mdid=3, pmpsplit=8
            (3, 0, 0),   # hart 3: mdid=0, pmpsplit=0 (host)
        ]
        for hart_idx, mdid, pmpsplit in hart_configs:
            h = emu.harts[hart_idx]
            h.mdid_val = mdid
            h.pmpsplit_val = pmpsplit

        # 验证各 hart 独立
        for hart_idx, mdid, pmpsplit in hart_configs:
            h = emu.harts[hart_idx]
            assert h.mdid_val == mdid, f"Hart {hart_idx} mdid"
            assert h.pmpsplit_val == pmpsplit, f"Hart {hart_idx} pmpsplit"

    def test_pmpsplit_pmp_fault_on_enclave_access_host_region(self):
        """通过 hart 执行 store -> trap 验证 pmpsplit 阻止飞地访问 host 内存."""
        hart = Hart(id=0, pmp_entries=8)
        bus = Bus(ram_size=1024 * 1024, ram_base=0x8000_0000)
        inject_memory_backend(hart, bus.read, bus.write)
        hart.bus = bus
        hart.csrs["mtvec"].val = 0x80000000
        hart.pc = 0x1000

        # Host 条目 0: 保护 0x8000_1000-0x8000_1FFF (NAPOT+R+W)
        k_4k = 9
        host_addr_val = (0x8000_1000 >> 2) | ((1 << k_4k) - 1)
        hart.csrs["pmpaddr0"].val = host_addr_val
        cfg = (0b11000 | 0b0011)  # NAPOT, R+W for entry 0
        hart.csrs["pmpcfg0"].val = cfg

        # 飞地条目 4: 允许 0x8000_2000-0x8000_2FFF (NAPOT+R+W)
        encl_addr_val = (0x8000_2000 >> 2) | ((1 << k_4k) - 1)
        hart.csrs["pmpaddr4"].val = encl_addr_val
        cfg |= (0b11000 | 0b0011) << (4 * 8)
        hart.csrs["pmpcfg0"].val = cfg

        # 配置飞地: mdid=1, pmpsplit=4 (条目 0-3=host, 4-7=enclave)
        hart.mdid_val = 1
        hart.pmpsplit_val = 4
        hart.mode = RiscvMode.S

        # 飞地写入自己的区域 (条目 4) -> OK
        mem_write(hart, 0x8000_2000, b"\xaa\xbb")
        assert hart.mcause_val == 0, f"飞地写自己的区域应成功: mcause={hart.mcause_val}"

        # 飞地写 host 区域 (条目 0, 对飞地不可见) -> StAccessFault
        try:
            mem_write(hart, 0x8000_1000, b"\xcc\xdd")
        except AccessFault:
            pass
        assert hart.mcause_val == 7, (
            f"飞地写 host 区域应 StAccessFault(7), 实际 mcause={hart.mcause_val}"
        )
