#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""mdid CSR + mfence.did 指令测试 — TEE 飞地隔离与抗侧信道刷新."""

import pytest

from pyremu.core.decoder import Hart
from pyremu.core.hart import HartWithRegs, RiscvMode
from pyremu.core.mem_check_aux import inject_memory_backend, mem_read
from pyremu.core.registers import CsrAccessError
from pyremu.core.trap import TrapType, trap_cause_code
from pyremu.emulator import Emulator
from pyremu.memory.tlb import TLB, TLBLine
from pyremu.platform import PlatformConfig


# ============================================================
#  mdid / pmpsplit CSR 寄存器
# ============================================================


class TestMdidCSR:
    """mdid (0x5C0) 和 pmpsplit (0x5C1) CSR 基本属性."""

    def test_mdid_exists_and_defaults_to_zero(self):
        emu = Emulator(PlatformConfig.qemu_virt())
        h = emu.harts[0]
        assert "mdid" in h.csrs, "mdid CSR 应存在"
        assert h.csrs["mdid"].val == 0, "mdid 默认值应为 0"

    def test_pmpsplit_exists_and_defaults_to_zero(self):
        emu = Emulator(PlatformConfig.qemu_virt())
        h = emu.harts[0]
        assert "pmpsplit" in h.csrs, "pmpsplit CSR 应存在"
        assert h.csrs["pmpsplit"].val == 0, "pmpsplit 默认值应为 0"

    def test_mdid_mmode_rw(self):
        """M 模式可读写 mdid."""
        emu = Emulator(PlatformConfig.qemu_virt())
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
        emu = Emulator(PlatformConfig.qemu_virt())
        h = emu.harts[0]
        for val in (1, 0xDEADBEEF, 0xFFFF_FFFF_FFFF_FFFF, 0x12345678_9ABCDEF0):
            h.write_csr(0x5C0, val)
            assert h.csrs["mdid"].val == val, f"mdid 往返失败: {val:#x}"

    def test_mdid_property(self):
        """HartWithRegs.mdid_val property 快捷访问."""
        emu = Emulator(PlatformConfig.qemu_virt())
        h = emu.harts[0]
        h.mdid_val = 0xC0FFEE
        assert h.mdid_val == 0xC0FFEE
        assert h.csrs["mdid"].val == 0xC0FFEE

    def test_umode_access_mdid_traps(self):
        """U 模式访问 mdid (M-mode only) 应触发 IllInstr."""
        emu = Emulator(PlatformConfig.qemu_virt())
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
        emu = Emulator(PlatformConfig.qemu_virt())
        h = emu.harts[0]
        h.pc = 0x80000000
        trap_before = h.mcause_val
        h.exec_instr(_MFENCE_DID)
        assert h.mcause_val == trap_before, f"mfence.did 不应触发 trap, mcause={h.mcause_val}"

    def test_invalid_funct12_traps(self):
        """funct12 非法编码应触发 IllInstr (funct3=000, 未注册的 funct12)."""
        emu = Emulator(PlatformConfig.qemu_virt())
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
        """mfence.did 仅刷新 mdid 匹配的 TLB 条目, 不匹配的保留."""
        emu = Emulator(PlatformConfig.qemu_virt())
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
        """mfence.did 同时刷新 itlb 和 dtlb."""
        emu = Emulator(PlatformConfig.qemu_virt())
        h = emu.harts[0]

        h.itlb.insert(vpn=0x100, ppn=0x10, perm=5, mdid=1)
        h.dtlb.insert(vpn=0x200, ppn=0x20, perm=7, mdid=1)

        h.mdid_val = 1
        h.pc = 0x80000000
        h.exec_instr(_MFENCE_DID)

        assert len(h.itlb) == 0, "itlb 应被清空"
        assert len(h.dtlb) == 0, "dtlb 应被清空"

    def test_nonzero_mdid_flushes_nothing_when_no_match(self):
        """mdid=3 但没有任何条目带此标记 → flush 为空操作."""
        emu = Emulator(PlatformConfig.qemu_virt())
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
        emu = Emulator(PlatformConfig.qemu_virt())
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
        """hart 0 执行 mfence.did → 全部 hart 的 TLB 均被刷新."""
        emu = Emulator(PlatformConfig(num_harts=4, ram_size=128 * 1024 * 1024))
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
        """不同 hart 的 TLB 带有不同 mdid → 仅匹配的被刷."""
        emu = Emulator(PlatformConfig(num_harts=2, ram_size=128 * 1024 * 1024))
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
        PAGE_SHIFT = 12
        L1_BASE = 0x1000
        L2_BASE = 0x2000
        L3_BASE = 0x3000
        DATA_PA = 0x100000

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

        # 构建 Sv39 三级 4 KiB 映射 va=0 → DATA_PA
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
            val |= ((ppn >> 9) & 0x1FF) << 20
            val |= ((ppn >> 18) & 0x1FFFFFFF) << 29
            return val

        write_pte(L1_BASE, vpn2, make_pte(v=True, ppn=L2_BASE >> PAGE_SHIFT))
        write_pte(L2_BASE, vpn1, make_pte(v=True, ppn=L3_BASE >> PAGE_SHIFT))
        write_pte(L3_BASE, vpn0, make_pte(v=True, r=True, w=True, x=True, ppn=target_ppn))

        h.satp_val = (8 << 60) | (L1_BASE >> PAGE_SHIFT)  # Sv39

        # 触发地址翻译 → TLB 插入
        mem_read(h, 0x0, 4)

        # 验证 TLB 条目被标记为 mdid=7
        dtlb_entries = [e for e in h.dtlb.entries if e.valid]
        assert len(dtlb_entries) >= 1, "应有 TLB 条目"
        for e in dtlb_entries:
            assert e.mdid == 7, f"TLB 条目 mdid 应为 7, 实际 {e.mdid}"

        # 切换飞地: mdid=7 → mdid=9, 用 mdid=9 执行 mfence.did
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

        # Bare 模式 (satp.MODE=0)
        assert h.mmu_mode == 0

        # 访问内存 — 不应插入 TLB
        mem_read(h, 0x1000, 4)
        assert len(h.dtlb) == 0, "Bare 模式不应自动填充 TLB"

    def test_manual_tlb_entries_still_flushed(self):
        """Bare 模式下手动插入的 TLB 条目 (通过 mfence.did) 仍可被刷新."""
        emu = Emulator(PlatformConfig.qemu_virt())
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
        from pyremu.memory.l2cache import L2CacheLine

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
        emu = Emulator(PlatformConfig.qemu_virt())
        h = emu.harts[0]
        l2 = emu.bus._l2
        addr = 0x80001000

        # hart mdid=1 写数据 → L2 分配, mdid=1
        l2.current_mdid = 1
        emu.bus.write(addr, b"X" * 64)
        _ = emu.bus.read(addr, 4)

        entries = [e for e in l2.entries if e.valid and e.tag == (addr >> l2._line_shift)]
        assert len(entries) == 1, "应有一条 L2 条目"
        assert entries[0].mdid == 1, f"新分配条目 mdid 应为 1, 实际 {entries[0].mdid}"

        # 切换 hart mdid=2, 再次读取同一地址 → 命中, mdid 应刷新为 2
        l2.current_mdid = 2
        _ = emu.bus.read(addr, 4)
        entries = [e for e in l2.entries if e.valid and e.tag == (addr >> l2._line_shift)]
        assert entries[0].mdid == 2, f"命中后 mdid 应刷新为 2, 实际 {entries[0].mdid}"

    def test_l2_write_hit_updates_mdid(self):
        """写命中时 L2 条目 mdid 更新为当前 hart 的 mdid."""
        emu = Emulator(PlatformConfig.qemu_virt())
        h = emu.harts[0]
        l2 = emu.bus._l2
        addr = 0x80001000

        l2.current_mdid = 1
        emu.bus.write(addr, b"Y" * 64)

        entries = [e for e in l2.entries if e.valid and e.tag == (addr >> l2._line_shift)]
        assert entries[0].mdid == 1

        # 切换 mdid, 写命中
        l2.current_mdid = 3
        emu.bus.write(addr, b"Z" * 64)
        entries = [e for e in l2.entries if e.valid and e.tag == (addr >> l2._line_shift)]
        assert entries[0].mdid == 3, f"写命中后 mdid 应刷新为 3, 实际 {entries[0].mdid}"

    def test_l2_auto_tag_then_selective_flush(self):
        """不同 mdid 的 hart 访问不同地址 → mfence.did 仅刷匹配的."""
        emu = Emulator(PlatformConfig.qemu_virt())
        h = emu.harts[0]
        l2 = emu.bus._l2
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
        emu = Emulator(PlatformConfig.qemu_virt())
        h = emu.harts[0]
        l2 = emu.bus._l2

        h.mdid_val = 0x77

        # 写一条 nop 到 PC, 然后 step()
        addr = 0x80000000
        emu.bus.write(addr, b"\x13\x00\x00\x00")  # nop (addi x0, x0, 0)
        h.pc = addr

        emu.step()

        # step() 内设置了 l2.current_mdid = hart.mdid_val
        # 取指 (bus.read) 会经过 L2 → 新行标记 mdid=0x77
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
