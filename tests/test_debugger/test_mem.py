#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""测试 pyremu.debug.mem — 读写语义、地址解析、反汇编与内存显示."""

import io

from rich.console import Console

from pyremu.core.hart import RiscvMode
from pyremu.core.trap_def import trap_cause_name
from pyremu.debug import Debugger
from pyremu.debug.mem import MemoryMixin
from pyremu.debug.utils import MAX_INSTR_COUNT
from pyremu.emulator import Emulator
from pyremu.memory.mmu import PTE
from pyremu.utils.parse_bin import FirmwareImage, FirmwareSegment

RAM_BASE = 0x8000_0000
RAM_SIZE = 0x0040_0000
CODE_PA = RAM_BASE + 0x1000  # 反汇编测试的代码区
DATA_PA = RAM_BASE + 0x8000  # 页映射的目标物理页
L1_BASE = RAM_BASE + 0x0002_0000  # 根页表
L2_BASE = RAM_BASE + 0x0002_1000
L3_BASE = RAM_BASE + 0x0002_2000
SATP_SV39 = 8 << 60
VA_TEXT = 0x1000  # 映射到 CODE_PA 的虚地址
VA_DATA = 0x2000  # 映射到 DATA_PA 的虚地址
HOLE_PA = 0x9000_0000  # 不落入任何 RAM 与设备窗口
NOP = b"\x13\x00\x00\x00"
JAL_ZERO = bytes.fromhex("6f000000")


def _make_dbg(image: FirmwareImage | None = None):
    """建立最小 Emulator 与 Debugger, 输出重定向到内存缓冲区."""
    emu = Emulator(num_harts=1, ram_size=RAM_SIZE, ram_base=RAM_BASE, prog_cnt=CODE_PA)
    emu._termio = None
    dbg = Debugger(emulator=emu, hart_id=0, image=image)
    buf = io.StringIO()
    dbg._console = Console(file=buf, highlight=False, width=200)
    return dbg, emu, buf


def _fw_image(symbols: dict | None = None, segments: list | None = None) -> FirmwareImage:
    return FirmwareImage(
        entry_point=CODE_PA,
        segments=list(segments or []),
        format="elf",
        symbols=dict(symbols or {}),
        symbol_ranges=[],
    )


def _write_pte(bus, pa: int, ppn: int, **flags) -> None:
    pte = PTE()
    pte.ppn = ppn & 0xF_FFFF_FFFF
    for name, value in flags.items():
        setattr(pte, name, value)
    bus.write(pa, pte.to_int().to_bytes(8, "little"))


def _map_page(bus, va: int, pa: int, **flags) -> None:
    """建立 4 KiB 页映射 VA 到 PA, 页表结构为 L1 指向 L2, L2 指向 L3."""
    _write_pte(bus, L1_BASE + ((va >> 30) & 0x1FF) * 8, L2_BASE >> 12, v=True)
    _write_pte(bus, L2_BASE + ((va >> 21) & 0x1FF) * 8, L3_BASE >> 12, v=True)
    _write_pte(bus, L3_BASE + ((va >> 12) & 0x1FF) * 8, pa >> 12, v=True, **flags)


def _enable_sv39(hart) -> None:
    """使 hart 以 S 模式在 Sv39 下运行, 根页表指向 L1_BASE."""
    hart.mode = RiscvMode.S
    hart.satp_val = SATP_SV39 | (L1_BASE >> 12)


def _code_dbg(image: FirmwareImage | None = None):
    """建立代码区已写入若干条 NOP 的调试器."""
    dbg, emu, buf = _make_dbg(image=image)
    emu.bus.write(CODE_PA, NOP * 6)
    dbg.hart.pc = CODE_PA
    return dbg, emu, buf


def _sym_dbg():
    """建立内核符号表只含 k_main 的调试器."""
    dbg, emu, buf = _code_dbg()
    dbg._sym_symbols = {"k_main": CODE_PA}
    dbg._sym_ranges = []
    dbg._sym_symbols_pa = {"k_main": CODE_PA}
    dbg._sym_ranges_pa = []
    return dbg, emu, buf


def _vdasm_dbg(code: bytes = NOP * 4):
    """建立 Sv39 下 VA_TEXT 可读可执行的调试器."""
    dbg, emu, buf = _make_dbg()
    emu.bus.write(DATA_PA, code)
    _map_page(emu.bus, VA_DATA, DATA_PA, r=True, x=True)
    _enable_sv39(dbg.hart)
    dbg.hart.pc = VA_DATA
    return dbg, emu, buf


# ============================================================
#  虚拟地址读取
# ============================================================


class TestTryReadVa:
    def test_size_zero_returns_none(self):
        dbg, _, _ = _make_dbg()
        assert dbg._try_read_va(RAM_BASE, 0) is None

    def test_size_above_page_returns_none(self):
        dbg, _, _ = _make_dbg()
        assert dbg._try_read_va(RAM_BASE, 4097) is None

    def test_bare_mode_treats_va_as_pa(self):
        dbg, emu, _ = _make_dbg()
        emu.bus.write(DATA_PA, b"\x01\x02\x03\x04")
        assert dbg._try_read_va(DATA_PA, 4) == b"\x01\x02\x03\x04"

    def test_m_mode_with_sv39_treats_va_as_pa(self):
        """M 模式绕过 MMU, satp 已配置时仍按物理地址读取."""
        dbg, emu, _ = _make_dbg()
        emu.bus.write(DATA_PA, b"\xaa\xbb\xcc\xdd")
        hart = dbg.hart
        hart.mode = RiscvMode.M
        hart.satp_val = SATP_SV39 | (L1_BASE >> 12)
        assert dbg._try_read_va(DATA_PA, 4) == b"\xaa\xbb\xcc\xdd"

    def test_sv39_translates_before_read(self):
        dbg, emu, _ = _make_dbg()
        emu.bus.write(DATA_PA, b"\x11\x22\x33\x44")
        _map_page(emu.bus, VA_DATA, DATA_PA, r=True, w=True)
        _enable_sv39(dbg.hart)
        assert dbg._try_read_va(VA_DATA, 4) == b"\x11\x22\x33\x44"

    def test_sv39_unmapped_page_returns_none(self):
        dbg, _, _ = _make_dbg()
        _enable_sv39(dbg.hart)
        assert dbg._try_read_va(VA_DATA, 4) is None

    def test_sv39_x_only_page_rejected_for_data_read(self):
        """数据读语义要求 PTE.R; 只可执行的页面在 MXR 为 0 时不可读."""
        dbg, emu, _ = _make_dbg()
        _map_page(emu.bus, VA_DATA, DATA_PA, x=True)
        _enable_sv39(dbg.hart)
        assert dbg._try_read_va(VA_DATA, 4) is None

    def test_read_spanning_two_pages_concatenates(self):
        """跨页读取逐页翻译, 两页的物理地址可以不连续."""
        dbg, emu, _ = _make_dbg()
        next_pa = DATA_PA + 0x2000
        emu.bus.write(DATA_PA + 0xFFC, b"\x01\x02\x03\x04")
        emu.bus.write(next_pa, b"\x05\x06\x07\x08")
        _map_page(emu.bus, VA_DATA, DATA_PA, r=True)
        _map_page(emu.bus, VA_DATA + 0x1000, next_pa, r=True)
        _enable_sv39(dbg.hart)
        assert dbg._try_read_va(VA_DATA + 0xFFC, 8) == b"\x01\x02\x03\x04\x05\x06\x07\x08"


# ============================================================
#  取指语义读取与物理读取
# ============================================================


class TestTryReadCodeVa:
    def test_x_only_page_is_readable_by_fetch_semantics(self):
        dbg, emu, _ = _make_dbg()
        emu.bus.write(DATA_PA, b"\x13\x00\x00\x00")
        _map_page(emu.bus, VA_DATA, DATA_PA, x=True)
        _enable_sv39(dbg.hart)
        assert dbg._try_read_code_va(VA_DATA, 4) == b"\x13\x00\x00\x00"

    def test_unmapped_page_returns_none(self):
        dbg, _, _ = _make_dbg()
        _enable_sv39(dbg.hart)
        assert dbg._try_read_code_va(VA_DATA, 4) is None


class TestTryReadPa:
    def test_reads_physical_bytes(self):
        dbg, emu, _ = _make_dbg()
        emu.bus.write(DATA_PA, b"\xde\xad\xbe\xef")
        assert dbg._try_read_pa(DATA_PA, 4) == b"\xde\xad\xbe\xef"

    def test_size_zero_returns_none(self):
        dbg, _, _ = _make_dbg()
        assert dbg._try_read_pa(DATA_PA, 0) is None

    def test_size_above_page_returns_none(self):
        dbg, _, _ = _make_dbg()
        assert dbg._try_read_pa(DATA_PA, 4097) is None


class TestTryReadVaForced:
    def test_size_zero_returns_none(self):
        dbg, _, _ = _make_dbg()
        assert dbg._try_read_va_forced(RAM_BASE, 0) is None

    def test_bare_mode_returns_va_as_pa(self):
        dbg, emu, _ = _make_dbg()
        emu.bus.write(DATA_PA, b"\x01\x02")
        assert dbg._try_read_va_forced(DATA_PA, 2) == (DATA_PA, b"\x01\x02")

    def test_sv39_returns_translated_pa(self):
        dbg, emu, _ = _make_dbg()
        emu.bus.write(DATA_PA, b"\x09\x08")
        _map_page(emu.bus, VA_DATA, DATA_PA, r=True)
        _enable_sv39(dbg.hart)
        assert dbg._try_read_va_forced(VA_DATA, 2) == (DATA_PA, b"\x09\x08")

    def test_sv39_unmapped_returns_none(self):
        dbg, _, _ = _make_dbg()
        _enable_sv39(dbg.hart)
        assert dbg._try_read_va_forced(VA_DATA, 4) is None


# ============================================================
#  地址解析
# ============================================================


class TestResolveAddr:
    def test_pc_keyword_returns_program_counter(self):
        dbg, _, _ = _make_dbg()
        dbg.hart.pc = DATA_PA
        assert dbg._resolve_addr("pc") == DATA_PA
        assert dbg._resolve_addr("PC") == DATA_PA

    def test_gpr_name_returns_register_value(self):
        dbg, _, _ = _make_dbg()
        dbg.hart.write_gpr(10, 0x1234)
        assert dbg._resolve_addr("a0") == 0x1234

    def test_known_csr_name_with_zero_value_returns_zero(self):
        """CSR 名解析成功而取值为 0 时返回 0, 不退化为数值解析."""
        dbg, _, _ = _make_dbg()
        dbg.hart.csrs["mtvec"].val = 0
        assert dbg._resolve_addr("mtvec") == 0

    def test_known_csr_name_returns_csr_value(self):
        dbg, _, _ = _make_dbg()
        dbg.hart.csrs["mtvec"].val = 0x5678
        assert dbg._resolve_addr("mtvec") == 0x5678

    def test_image_symbol_is_offset_by_load_offset(self):
        dbg, _, _ = _make_dbg(image=_fw_image(symbols={"main": 0x1000}))
        dbg._load_offset = 0x100
        assert dbg._resolve_addr("main") == 0x1100

    def test_kernel_symbol_is_offset_by_symbol_load_offset(self):
        dbg, _, _ = _make_dbg()
        dbg._sym_symbols = {"k_main": 0x1000}
        dbg._sym_load_offset = 0x2000
        assert dbg._resolve_addr("k_main") == 0x3000

    def test_numeric_literal(self):
        dbg, _, _ = _make_dbg()
        assert dbg._resolve_addr("0x8000") == 0x8000

    def test_negative_literal_returns_none(self):
        dbg, _, _ = _make_dbg()
        assert dbg._resolve_addr("-8") is None

    def test_literal_at_two_pow_64_returns_none(self):
        dbg, _, _ = _make_dbg()
        assert dbg._resolve_addr("0x10000000000000000") is None

    def test_unknown_name_returns_none(self):
        dbg, _, _ = _make_dbg()
        assert dbg._resolve_addr("nowhere") is None


# ============================================================
#  取指与反汇编辅助
# ============================================================


class TestFetchAndDisasm:
    def test_compressed_instruction_uses_16_bit_hex(self):
        dbg, emu, _ = _make_dbg()
        emu.bus.write(DATA_PA, b"\x01\x00")  # c.nop
        result = dbg._fetch_and_disasm(DATA_PA)
        assert result is not None
        assert result[0] == "0001"

    def test_full_instruction_uses_32_bit_hex(self):
        dbg, emu, _ = _make_dbg()
        emu.bus.write(DATA_PA, NOP)
        result = dbg._fetch_and_disasm(DATA_PA)
        assert result is not None
        assert result[0] == "00000013"

    def test_unreadable_address_returns_none(self):
        dbg, _, _ = _make_dbg()
        _enable_sv39(dbg.hart)
        assert dbg._fetch_and_disasm(VA_DATA) is None


class TestColorizeAsm:
    def test_empty_text_is_returned_as_is(self):
        assert MemoryMixin._colorize_asm("") == ""

    def test_markup_prefixed_text_is_returned_as_is(self):
        assert MemoryMixin._colorize_asm("[dim]x[/]") == "[dim]x[/]"

    def test_unknown_is_returned_as_is(self):
        assert MemoryMixin._colorize_asm("(unknown)") == "(unknown)"

    def test_load_store_mnemonic_without_operand_is_cyan(self):
        assert MemoryMixin._colorize_asm("ld") == "[cyan]ld[/]"

    def test_load_store_with_operand_is_cyan(self):
        assert MemoryMixin._colorize_asm("sd a0, 8(sp)") == "[cyan]sd[/] a0, [magenta]8[/](sp)"

    def test_branch_mnemonic_is_green(self):
        colored = MemoryMixin._colorize_asm("beq a0, a1, 16")
        assert colored == "[green]beq[/] a0, a1, [magenta]16[/]"

    def test_fence_mnemonic_is_yellow(self):
        assert MemoryMixin._colorize_asm("fence rw, rw") == "[yellow]fence[/] rw, rw"

    def test_privileged_mnemonic_is_red(self):
        assert MemoryMixin._colorize_asm("mret") == "[red]mret[/]"

    def test_plain_mnemonic_keeps_operands_unstyled(self):
        assert MemoryMixin._colorize_asm("mv a0, a1") == "mv a0, a1"


class TestCtrlFlowKind:
    def test_jal(self):
        assert MemoryMixin._ctrl_flow_kind(0x0000006F) == "jal"

    def test_jalr(self):
        assert MemoryMixin._ctrl_flow_kind(0x00000067) == "jalr"

    def test_branch(self):
        assert MemoryMixin._ctrl_flow_kind(0x00000063) == "branch"

    def test_ecall(self):
        assert MemoryMixin._ctrl_flow_kind(0x00000073) == "ecall"

    def test_ebreak(self):
        assert MemoryMixin._ctrl_flow_kind((0x001 << 20) | 0x73) == "ebreak"

    def test_mret(self):
        assert MemoryMixin._ctrl_flow_kind((0x302 << 20) | 0x73) == "mret"

    def test_sret(self):
        assert MemoryMixin._ctrl_flow_kind((0x102 << 20) | 0x73) == "sret"

    def test_wfi(self):
        assert MemoryMixin._ctrl_flow_kind((0x105 << 20) | 0x73) == "wfi"

    def test_other_system_instruction_is_csr(self):
        assert MemoryMixin._ctrl_flow_kind((0xC00 << 20) | 0x73) == "csr"

    def test_fence(self):
        assert MemoryMixin._ctrl_flow_kind(0x0000000F) == "fence"
        assert MemoryMixin._ctrl_flow_kind(0x0000002F) == "fence"

    def test_ordinary_instruction_is_normal(self):
        assert MemoryMixin._ctrl_flow_kind(0x00000013) == "normal"


class TestCtrlFlowKindCompressed:
    def test_c_jal(self):
        assert MemoryMixin._ctrl_flow_kind_compressed((1 << 13) | 1) == "jal"

    def test_c_j(self):
        assert MemoryMixin._ctrl_flow_kind_compressed((5 << 13) | 1) == "jal"

    def test_c_jr(self):
        assert MemoryMixin._ctrl_flow_kind_compressed((4 << 13) | 2) == "jalr"

    def test_c_mv_is_normal(self):
        assert MemoryMixin._ctrl_flow_kind_compressed((4 << 13) | 2 | (1 << 2)) == "normal"

    def test_c_jalr(self):
        assert MemoryMixin._ctrl_flow_kind_compressed((6 << 13) | 2) == "jalr"

    def test_c_jalr_with_nonzero_rs2_is_normal(self):
        assert MemoryMixin._ctrl_flow_kind_compressed((6 << 13) | 2 | (1 << 2)) == "normal"

    def test_c_beqz(self):
        assert MemoryMixin._ctrl_flow_kind_compressed((6 << 13) | 1) == "branch"

    def test_c_bnez(self):
        assert MemoryMixin._ctrl_flow_kind_compressed((7 << 13) | 1) == "branch"

    def test_c_ebreak(self):
        assert MemoryMixin._ctrl_flow_kind_compressed(0x0002) == "ebreak"

    def test_other_compressed_is_normal(self):
        assert MemoryMixin._ctrl_flow_kind_compressed(0x0001) == "normal"


class TestTrapCauseNameStatic:
    def test_matches_module_level_function(self):
        """静态转发与 trap_def 的映射一致."""
        for mcause in (0, 2, 13, (1 << 63) | 7):
            assert MemoryMixin._trap_cause_name_static(mcause) == trap_cause_name(mcause)


class TestWarnPcIfSuspect:
    def _warns(self, dbg):
        messages: list[str] = []
        dbg._warn = messages.append
        return messages

    def test_odd_value_warns_alignment(self):
        dbg, _, _ = _make_dbg()
        messages = self._warns(dbg)
        dbg._warn_pc_if_suspect(dbg.hart, 0x1001)
        assert any("未对齐" in m for m in messages)

    def test_value_beyond_rv64_warns_range(self):
        dbg, _, _ = _make_dbg()
        messages = self._warns(dbg)
        dbg._warn_pc_if_suspect(dbg.hart, 1 << 64)
        assert any("超出 RV64" in m for m in messages)

    def test_aligned_value_in_range_is_silent(self):
        dbg, _, _ = _make_dbg()
        messages = self._warns(dbg)
        dbg._warn_pc_if_suspect(dbg.hart, CODE_PA)
        assert messages == []


# ============================================================
#  pc 命令
# ============================================================


class TestCmdPc:
    def test_unresolvable_address_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_pc("nowhere")
        assert "无法解析地址" in buf.getvalue()

    def test_address_beyond_rv64_reports_error(self):
        """符号加装载偏移越过 2^64 时拒绝设置 PC."""
        dbg, _, buf = _make_dbg()
        dbg._image = _fw_image(symbols={"huge": 0xFFFF_FFFF_FFFF_FFF0})
        dbg._load_offset = 0x20
        dbg.cmd_pc("huge")
        assert "超出 RV64 范围" in buf.getvalue()
        assert dbg.hart.pc == CODE_PA

    def test_odd_address_warns_alignment(self):
        dbg, _, buf = _make_dbg()
        dbg.hart.pc = CODE_PA
        dbg.cmd_pc(hex(CODE_PA + 1))
        assert "未对齐" in buf.getvalue()

    def test_unreadable_pc_prints_placeholder(self):
        dbg, _, buf = _make_dbg()
        _enable_sv39(dbg.hart)
        dbg.hart.pc = VA_DATA
        dbg.cmd_pc()
        assert "无法读取" in buf.getvalue()

    def test_zero_mcause_prints_no_trap_table(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_pc()
        assert "Trap 上下文" not in buf.getvalue()

    def test_trap_table_is_printed_for_s_and_u_mode(self):
        dbg, _, buf = _make_dbg()
        hart = dbg.hart
        hart.csrs["mcause"].val = 2
        for mode in (RiscvMode.S, RiscvMode.U):
            dbg._trap_displayed_mcause = None
            hart.mode = mode
            buf.truncate(0)
            buf.seek(0)
            dbg.cmd_pc()
            assert "Trap 上下文" in buf.getvalue()

    def test_same_mcause_is_reported_only_once(self):
        dbg, _, buf = _make_dbg()
        dbg.hart.csrs["mcause"].val = 2
        dbg.cmd_pc()
        assert "Trap 上下文" in buf.getvalue()
        buf.truncate(0)
        buf.seek(0)
        dbg.cmd_pc()
        assert "Trap 上下文" not in buf.getvalue()

    def test_interrupt_mcause_marks_context_as_interrupt(self):
        dbg, _, buf = _make_dbg()
        dbg.hart.csrs["mcause"].val = (1 << 63) | 7
        dbg.cmd_pc()
        assert "中断" in buf.getvalue()

    def test_nonzero_scause_is_annotated_with_its_name(self):
        dbg, _, buf = _make_dbg()
        dbg.hart.csrs["mcause"].val = 2
        dbg.hart.csrs["scause"].val = 13
        dbg.cmd_pc()
        assert "Trap 上下文" in buf.getvalue()

    def test_nonzero_sstatus_alone_is_annotated(self):
        """scause 为 0 而 sstatus 非 0 时不带异常名, 只报告使能位与特权级."""
        dbg, _, buf = _make_dbg()
        dbg.hart.csrs["mcause"].val = 2
        dbg.hart.csrs["scause"].val = 0
        dbg.hart.csrs["sstatus"].val = 0x100
        dbg.cmd_pc()
        assert "SPP=1" in buf.getvalue()

    def test_waiting_hart_is_marked(self):
        dbg, _, buf = _make_dbg()
        dbg.hart._waiting = True
        dbg.cmd_pc()
        assert "WFI" in buf.getvalue()


# ============================================================
#  反汇编命令
# ============================================================


class TestCountInstrsBetween:
    def test_start_outside_ram_returns_upper_bound(self):
        dbg, _, _ = _make_dbg()
        assert dbg._count_instrs_between(HOLE_PA, RAM_BASE + RAM_SIZE) == MAX_INSTR_COUNT

    def test_counts_mixed_compressed_and_full_instructions(self):
        dbg, emu, _ = _make_dbg()
        emu.bus.write(CODE_PA, b"\x01\x00" + NOP)  # c.nop + nop = 6 字节
        assert dbg._count_instrs_between(CODE_PA, CODE_PA + 6) == 2


class TestCmdDisasm:
    def test_unresolvable_address_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_disasm("nowhere")
        assert "无法解析地址" in buf.getvalue()

    def test_instruction_count_out_of_range_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_disasm("0x1000", "0")
        assert "1-512" in buf.getvalue()

    def test_address_beyond_rv64_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg._image = _fw_image(symbols={"huge": 0xFFFF_FFFF_FFFF_FFF0})
        dbg._load_offset = 0x20
        dbg.cmd_disasm("huge")
        assert "超出 RV64 范围" in buf.getvalue()

    def test_odd_address_is_aligned_with_warning(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_disasm(hex(CODE_PA + 1), "1")
        assert "对齐" in buf.getvalue()

    def test_symbol_scope_line_is_printed_once(self):
        dbg, _, buf = _sym_dbg()
        dbg.cmd_disasm(hex(CODE_PA), "2")
        out = buf.getvalue()
        assert "k_main" in out
        assert out.count("k_main") == 1

    def test_scope_line_carries_segment_name(self):
        segment = FirmwareSegment(vaddr=CODE_PA, data=b"", memsz=0x1000, name=".text")
        image = _fw_image(segments=[segment])
        dbg, _, buf = _sym_dbg()
        dbg._image = image
        dbg.cmd_disasm(hex(CODE_PA), "1")
        out = buf.getvalue()
        assert ".text" in out and "k_main" in out

    def test_terminator_at_reference_pc_ends_step_advance(self):
        """分支指令之后不再给出相对步数, 且基准步数不再前移."""
        dbg, emu, buf = _make_dbg()
        emu.bus.write(CODE_PA, JAL_ZERO + NOP)
        dbg.hart.pc = CODE_PA
        dbg.cmd_disasm(hex(CODE_PA), "2")
        assert "─" * 8 in buf.getvalue()
        assert dbg._disasm_base_step == 0

    def test_forward_count_past_terminator_returns_upper_bound(self):
        """PC 不在 RAM 内且低于反汇编地址: 前向计数达到上限, 步数停止前移."""
        dbg, _, buf = _code_dbg()
        dbg.hart.pc = 0
        dbg.cmd_disasm(hex(CODE_PA), "2")
        assert dbg._disasm_base_step == MAX_INSTR_COUNT
        assert dbg._disasm_past_terminator is True
        assert "+" not in buf.getvalue().split("\n", 1)[1]

    def test_backward_count_past_terminator_returns_upper_bound(self):
        """PC 高于反汇编地址且不在 RAM 内: 反向计数达到上限."""
        dbg, _, _ = _code_dbg()
        dbg.hart.pc = HOLE_PA
        dbg.cmd_disasm(hex(CODE_PA), "1")
        assert dbg._disasm_base_step == -MAX_INSTR_COUNT
        assert dbg._disasm_past_terminator is True

    def test_relative_step_prefix_before_terminator(self):
        dbg, _, buf = _code_dbg()
        dbg.hart.pc = CODE_PA
        dbg.cmd_disasm(hex(CODE_PA + 8), "2")
        assert "+2" in buf.getvalue()

    def test_continue_from_next_address_keeps_step_base(self):
        dbg, _, _ = _code_dbg()
        dbg.cmd_disasm(hex(CODE_PA), "2")
        assert dbg._disasm_next_addr == CODE_PA + 8
        base = dbg._disasm_base_step
        dbg.cmd_disasm(hex(dbg._disasm_next_addr), "2")
        assert dbg._disasm_next_addr == CODE_PA + 16
        assert base == 2

    def test_read_window_is_not_cut_mid_instruction(self):
        """读取窗口长度为请求条数的四倍, 窗口内不会出现半个指令.

        窗口长为 4*inst_count, 而进入循环的每次迭代至多推进 4 字节, 且循环在
        收集到 inst_count 条时结束, 故已消费字节数不超过 4*(inst_count-1),
        剩余字节数不小于 4, 不小于任何一条指令的长度.
        """
        dbg, emu, buf = _make_dbg()
        cut_code = b"\x01\x00" * 3 + JAL_ZERO  # 三条压缩指令 + 一条四字节指令
        emu.bus.write(CODE_PA, cut_code)
        dbg.hart.pc = CODE_PA
        dbg.cmd_disasm(hex(CODE_PA), "2")
        out = buf.getvalue()
        assert "截断" not in out
        assert dbg._disasm_next_addr == CODE_PA + 4

    def test_image_symbol_address_is_disassembled(self):
        dbg, emu, buf = _make_dbg(image=_fw_image(symbols={"main": CODE_PA}))
        emu.bus.write(CODE_PA, NOP)
        dbg.cmd_disasm("main", "1")
        assert "00000013" in buf.getvalue()


# ============================================================
#  虚拟地址反汇编命令
# ============================================================


class TestCmdVdisasm:
    def test_unresolvable_address_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_vdisasm("nowhere")
        assert "无法解析地址" in buf.getvalue()

    def test_instruction_count_out_of_range_reports_error(self):
        dbg, _, buf = _vdasm_dbg()
        dbg.cmd_vdisasm(hex(VA_DATA), "0")
        assert "1-512" in buf.getvalue()

    def test_address_beyond_rv64_reports_error(self):
        dbg, _, buf = _vdasm_dbg()
        dbg._image = _fw_image(symbols={"huge": 0xFFFF_FFFF_FFFF_FFF0})
        dbg._load_offset = 0x20
        dbg.cmd_vdisasm("huge")
        assert "超出 RV64 范围" in buf.getvalue()

    def test_bare_mode_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_vdisasm(hex(DATA_PA), "1")
        assert "Bare 模式" in buf.getvalue()

    def test_pa_to_va_reverse_lookup_restores_virtual_address(self):
        """以物理地址提交时经 _pa_to_va 反查回虚地址再遍历页表."""
        dbg, _, buf = _vdasm_dbg()
        dbg._pa_to_va = {DATA_PA: VA_DATA}
        dbg.cmd_vdisasm(hex(DATA_PA), "1")
        assert "00000013" in buf.getvalue()

    def test_odd_address_is_aligned_with_warning(self):
        dbg, _, buf = _vdasm_dbg()
        dbg.cmd_vdisasm(hex(VA_DATA + 1), "1")
        assert "对齐" in buf.getvalue()

    def test_missing_memory_backend_reports_error(self):
        dbg, _, buf = _vdasm_dbg()
        dbg.hart._mem_read_phy = None
        dbg.cmd_vdisasm(hex(VA_DATA), "1")
        assert "内存后端未挂载" in buf.getvalue()

    def test_unmapped_va_reports_page_fault(self):
        dbg, _, buf = _make_dbg()
        _enable_sv39(dbg.hart)
        dbg.cmd_vdisasm(hex(VA_DATA), "1")
        assert "页表翻译失败" in buf.getvalue()

    def test_symbol_scope_line_is_printed(self):
        dbg, _, buf = _vdasm_dbg()
        dbg._sym_symbols = {"k_main": VA_DATA}
        dbg._sym_ranges = []
        dbg.cmd_vdisasm(hex(VA_DATA), "2")
        assert "k_main" in buf.getvalue()

    def test_terminator_instruction_is_underlined(self):
        dbg, _, buf = _vdasm_dbg(JAL_ZERO + NOP)
        dbg.cmd_vdisasm(hex(VA_DATA), "2")
        assert "─" * 8 in buf.getvalue()

    def test_read_window_is_not_cut_mid_instruction(self):
        """读取窗口长度为请求条数的四倍, 循环在收满条数或读完窗口时结束."""
        dbg, _, buf = _vdasm_dbg(b"\x01\x00" * 3 + JAL_ZERO)
        dbg.cmd_vdisasm(hex(VA_DATA), "2")
        out = buf.getvalue()
        assert "截断" not in out
        assert dbg._vdisasm_next_addr == VA_DATA + 4

    def test_reference_pc_is_marked(self):
        dbg, _, buf = _vdasm_dbg()
        dbg.hart.pc = VA_DATA
        dbg.cmd_vdisasm(hex(VA_DATA), "2")
        assert "pc ->" in buf.getvalue()


# ============================================================
#  VA 到 PA 映射表头
# ============================================================


class TestShowVaMappingHeader:
    def test_walk_on_tlb_miss_then_hit(self):
        """首次遍历填充 TLB, 第二次由 TLB 命中取出 ppn 与权限位."""
        dbg, emu, buf = _make_dbg()
        _map_page(emu.bus, VA_DATA, DATA_PA, r=True)
        _enable_sv39(dbg.hart)
        dbg._show_va_mapping_header(VA_DATA, 4)
        first = buf.getvalue()
        assert f"VA 0x{VA_DATA:016x} -> PA 0x{DATA_PA:016x}" in first
        # 遍历成功后 TLB 命中, 输出内容不变
        dbg._show_va_mapping_header(VA_DATA, 4)
        assert buf.getvalue().count(f"VA 0x{VA_DATA:016x} -> PA 0x{DATA_PA:016x}") == 2

    def test_unmapped_page_is_reported_as_missing(self):
        dbg, _, buf = _make_dbg()
        _enable_sv39(dbg.hart)
        dbg._show_va_mapping_header(VA_DATA, 4)
        assert "缺页" in buf.getvalue()

    def test_missing_memory_backend_is_reported(self):
        dbg, _, buf = _make_dbg()
        dbg.hart._mem_read_phy = None
        dbg._show_va_mapping_header(VA_DATA, 4)
        assert "内存后端未挂载" in buf.getvalue()

    def test_size_spanning_two_pages_reports_both(self):
        """起始地址在页内偏移 0x800 处, 4096 字节的区间跨过页边界."""
        dbg, emu, buf = _make_dbg()
        next_pa = DATA_PA + 0x1000
        _map_page(emu.bus, VA_DATA, DATA_PA, r=True, w=True, x=True, u=True)
        _map_page(emu.bus, VA_DATA + 0x1000, next_pa, r=True)
        _enable_sv39(dbg.hart)
        dbg._show_va_mapping_header(VA_DATA + 0x800, 4096)
        out = buf.getvalue()
        assert f"VA 0x{VA_DATA:016x} -> PA 0x{DATA_PA:016x}" in out
        assert f"VA 0x{VA_DATA + 0x1000:016x} -> PA 0x{next_pa:016x}" in out


# ============================================================
#  物理内存显示命令
# ============================================================


class TestCmdMem:
    def test_hexdump_is_printed(self):
        dbg, emu, buf = _make_dbg()
        emu.bus.write(DATA_PA, b"ABCD")
        dbg.cmd_mem(hex(DATA_PA), "16")
        assert "41 42 43 44" in buf.getvalue()

    def test_unresolvable_address_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_mem("nowhere")
        assert "无法解析地址" in buf.getvalue()

    def test_size_out_of_range_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_mem(hex(DATA_PA), "4097")
        assert "1-4096" in buf.getvalue()

    def test_address_beyond_rv64_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg._image = _fw_image(symbols={"huge": 0xFFFF_FFFF_FFFF_FFF0})
        dbg._load_offset = 0x20
        dbg.cmd_mem("huge")
        assert "超出 RV64 范围" in buf.getvalue()

    def test_non_integer_size_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_mem(hex(DATA_PA), "abc")
        assert "需为整数" in buf.getvalue()


# ============================================================
#  虚拟内存显示命令
# ============================================================


class TestCmdVmem:
    def test_mapping_and_hexdump_are_printed(self):
        dbg, emu, buf = _make_dbg()
        emu.bus.write(DATA_PA, b"ABCD")
        _map_page(emu.bus, VA_DATA, DATA_PA, r=True, w=True)
        _enable_sv39(dbg.hart)
        dbg.cmd_vmem(hex(VA_DATA), "8")
        out = buf.getvalue()
        assert f"VA 0x{VA_DATA:016x} -> PA 0x{DATA_PA:016x}" in out
        assert "41 42 43 44" in out

    def test_unresolvable_address_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_vmem("nowhere")
        assert "无法解析地址" in buf.getvalue()

    def test_size_out_of_range_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_vmem(hex(VA_DATA), "0")
        assert "1-4096" in buf.getvalue()

    def test_address_beyond_rv64_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg._image = _fw_image(symbols={"huge": 0xFFFF_FFFF_FFFF_FFF0})
        dbg._load_offset = 0x20
        dbg.cmd_vmem("huge")
        assert "超出 RV64 范围" in buf.getvalue()

    def test_bare_mode_reports_error(self):
        dbg, _, buf = _make_dbg()
        dbg.cmd_vmem(hex(DATA_PA), "8")
        assert "Bare 模式" in buf.getvalue()

    def test_pa_to_va_reverse_lookup_restores_virtual_address(self):
        dbg, emu, buf = _make_dbg()
        emu.bus.write(DATA_PA, b"ABCD")
        _map_page(emu.bus, VA_DATA, DATA_PA, r=True)
        _enable_sv39(dbg.hart)
        dbg._pa_to_va = {DATA_PA: VA_DATA}
        dbg.cmd_vmem(hex(DATA_PA), "4")
        assert f"PA 0x{DATA_PA:016x}" in buf.getvalue()

    def test_missing_memory_backend_reports_error(self):
        dbg, _, buf = _make_dbg()
        _enable_sv39(dbg.hart)
        dbg.hart._mem_read_phy = None
        dbg.cmd_vmem(hex(VA_DATA), "8")
        assert "内存后端未挂载" in buf.getvalue()

    def test_unmapped_va_reports_page_fault(self):
        dbg, _, buf = _make_dbg()
        _enable_sv39(dbg.hart)
        dbg.cmd_vmem(hex(VA_DATA), "8")
        assert "页表翻译失败" in buf.getvalue()

    def test_x_only_page_fails_data_read_after_successful_walk(self):
        """页表遍历通过但数据读语义被拒: 只可执行的页面不可按数据读显示."""
        dbg, emu, buf = _make_dbg()
        _map_page(emu.bus, VA_DATA, DATA_PA, x=True)
        _enable_sv39(dbg.hart)
        dbg.cmd_vmem(hex(VA_DATA), "8")
        assert "翻译后物理读取失败" in buf.getvalue()
