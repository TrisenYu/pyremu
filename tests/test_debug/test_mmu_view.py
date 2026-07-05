#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""测试 pyremu.debug.mmu_view — PMP/SATP/页表."""

from pyremu.debug.mmu_view import MmuViewMixin
from pyremu.emulator import Emulator

# ============================================================
#  Mini 测试类
# ============================================================


class _TestMmuDbg(MmuViewMixin):
    """最小聚合类供 MmuViewMixin 测试."""

    def __init__(self, emu, hart_id=0):
        from rich.console import Console
        self._emu = emu
        self._hart_id = hart_id
        self._console = Console(highlight=False)
        self._warn = lambda msg: None
        self._err = lambda msg: None

    @property
    def hart(self):
        return self._emu.harts[self._hart_id]

    def _resolve_addr(self, arg):
        try:
            return int(arg, 0)
        except (ValueError, TypeError):
            return None


def _make_mmudbg():
    emu = Emulator(prog_cnt=0x1000)
    emu.load_code(0x1000, b"\x13\x00\x00\x00")  # NOP
    return _TestMmuDbg(emu)


# ============================================================
#  PMP cfg 解码
# ============================================================


class TestDecodePmpCfgByte:
    """_decode_pmp_cfg_byte — 从 pmpcfgN 提取条目配置."""

    def test_entry_0(self):
        val = MmuViewMixin._decode_pmp_cfg_byte(0x000000000000001F, 0)
        assert val == 0x1F  # L+R+W+X

    def test_entry_7(self):
        """最后一个条目在 pmpcfg0 中."""
        cfg_val = 0xFF00000000000000  # entry 7 = 0xFF
        val = MmuViewMixin._decode_pmp_cfg_byte(cfg_val, 7)
        assert val == 0xFF

    def test_entry_8(self):
        """第一个条目在 pmpcfg2 中."""
        cfg_val = 0x000000000000001F  # entry 8 = 0x1F
        val = MmuViewMixin._decode_pmp_cfg_byte(cfg_val, 0)  # local idx 0
        assert val == 0x1F

    def test_entry_15(self):
        """最后一个条目在 pmpcfg2 中."""
        cfg_val = 0xFF00000000000000
        val = MmuViewMixin._decode_pmp_cfg_byte(cfg_val, 7)
        assert val == 0xFF


# ============================================================
#  PMP / SATP / PT 命令
# ============================================================


class TestCmdPmp:
    """cmd_pmp — PMP 条目显示."""

    def test_no_pmp(self):
        """无 PMP 条目时显示提示."""
        dbg = _make_mmudbg()
        dbg.hart._pmp = None
        dbg.cmd_pmp()

    def test_with_pmp_entries(self):
        dbg = _make_mmudbg()
        dbg.cmd_pmp()


class TestCmdSatp:
    """cmd_satp — SATP 解码."""

    def test_bare_mode(self):
        dbg = _make_mmudbg()
        dbg.hart.satp_val = 0
        dbg.cmd_satp()

    def test_sv39_mode(self):
        dbg = _make_mmudbg()
        dbg.hart.satp_val = (8 << 60) | 0x80000
        dbg.cmd_satp()


class TestCmdPt:
    """cmd_pt — 页表遍历."""

    def test_bare_mode_warns(self):
        dbg = _make_mmudbg()
        dbg.hart.satp_val = 0  # Bare
        dbg.cmd_pt("0x1000")  # 应不抛异常

    def test_sv39_no_page_table(self):
        """Sv39 模式但根页表不可读."""
        dbg = _make_mmudbg()
        dbg.hart.satp_val = (8 << 60) | 0x100  # Sv39, 根 PPN 指向无效区域
        dbg.cmd_pt("0x1000")  # 应不抛异常

    def test_with_valid_sv39_table(self):
        """构建一个简单的 Sv39 页表并遍历."""
        emu = Emulator(ram_size=0x100000, prog_cnt=0x1300)
        # 构建 identity 映射: VA 0x1000 -> PA 0x1000
        # L1 在 0x8000, L2 在 0x9000, L3 在 0xA000
        L1 = 0x8000
        L2 = 0x9000
        L3 = 0xA000
        l1_raw = bytearray(4096)
        l2_raw = bytearray(4096)
        l3_raw = bytearray(4096)
        # L1[0] -> L2, V=1, R=0,W=0,X=0 (pointer)
        l1_raw[0:8] = ((L2 >> 12) << 10 | 0x01).to_bytes(8, "little")
        # L2[0] -> L3
        l2_raw[0:8] = ((L3 >> 12) << 10 | 0x01).to_bytes(8, "little")
        # L3[1] (VPN[0]=1 for VA 0x1000) -> PA 0x1000, V=1, R=1,W=1,X=1, ...
        l3_raw[8:16] = ((0x1000 >> 12) << 10 | 0x0F).to_bytes(8, "little")
        emu.bus.write_ram_direct(L1, bytes(l1_raw))
        emu.bus.write_ram_direct(L2, bytes(l2_raw))
        emu.bus.write_ram_direct(L3, bytes(l3_raw))
        emu.load_code(0x1300, b"\x13\x00\x00\x00")
        dbg = _TestMmuDbg(emu)
        dbg.hart.satp_val = (8 << 60) | (L1 >> 12)
        dbg.cmd_pt("0x1000")
