#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""StackWalkOps — FP 链回溯、跨特权级边界帧、栈帧显示.

依赖 DebuggerBase + MemoryOps._try_read_va + SymLookupOps.
"""

import struct

from rich.table import Table

from pyremu.core.decoder import decode_c_sdsp, decode_sd_sp
from pyremu.core.hart import RiscvMode
from pyremu.debug.attrs import SharedAttrs
from pyremu.debug.types import StackFrame, TrapEntry
from pyremu.debug.utils import hex_addr
from pyremu.utils.str_aux import dye_hexdump
from pyremu.utils.wrapper import seize_val_err

# 跨特权级 trap 上下文中 RA/FP 保存偏移的常见布局.
# 当前指令级解析 (_parse_trap_save_offsets) 失败时,
# 按优先级依次尝试这些 (ra_off, fp_off) 对.  0 表示该偏移未知/不需要.
_FALLBACK_TRAP_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 8),  # 紧凑型: sd x1,0(sp); sd x8,8(sp)
    (8, 64),  # Linux pt_regs: ra=PT_RA(8), s0=PT_S0(64)
    (0, 16),  # 稀疏型: sd x1,0(sp); sd x8,16(sp)
    (8, 16),  # 类型 B: RA 在 gp 之后
    (8, 24),  # 类型 C
)


class StackWalkOps(SharedAttrs):
    """栈帧回溯 (backtrace)."""

    # ----------------------------------------------------------
    #  调用点推断
    # ----------------------------------------------------------

    # ---- call instruction encoding constants (RV64) ----
    # C1 quadrant: bits[1:0]=01, bits[15:13]=funct3
    # C2 quadrant: bits[1:0]=10, bits[15:13]=funct3
    #
    # c.jal  (RV64)  funct3=101  quadrant C1  -> jal  x1, imm     [2 B]
    # c.jalr           funct3=100  quadrant C2  -> jalr x1, rs1, 0  [2 B]
    # jal              opcode=110_1111,  rd=x1                    [4 B]
    # jalr             opcode=110_0111,  funct3=000, rd=x1        [4 B]
    # ----------------------------------------------------------

    _C_JAL_MASK: int = 0xE003  # bits[15:13]=101, bits[1:0]=01
    _C_JAL_MATCH: int = 0x2001

    _C_JALR_MASK: int = 0xF003  # bits[15:13]=100, bit[12]=0, bits[1:0]=10
    _C_JALR_MATCH: int = 0x9002

    @staticmethod
    def _is_plausible_ra(ra: int) -> bool:
        """快速拒绝明显无效的返回地址.

        返回 False 的值:
          - 0, 即空指针
          - 0xFFFFFFFFFFFFFFFF, 表示未初始化或函数覆盖了 x1
          - < 4, 无法安全地取 ra-4
          - 地址落在 64-bit 空间最高 64 KiB 内, 即 > 0xFFFF_FFFF_FFFF_0000.
            内核/用户代码的实际映射地址都远低于此边界.
        """
        return ra not in {0, 0xFFFF_FFFF_FFFF_FFFF} and 4 <= ra <= 0xFFFF_FFFF_FFFF_0000

    def _check_return_addr(self, ra: int) -> bool:
        return ra != 0 and self._is_plausible_ra(ra) and self._is_valid_code_va(ra)

    def _call_site_pc(self, ra: int) -> int:
        """从返回地址推断调用指令的 PC.

        RISC-V 调用指令的返回地址为 PC+2 (c.jal/c.jalr) 或 PC+4
        (jal/jalr). 读取 ra 之前的指令字节以判定调用类型; 若无法
        确定则保守假设 4 字节.
        """
        if not self._is_plausible_ra(ra):
            return 0

        # -- 尝试 ra-2: 16-bit 压缩调用 --
        # 调用点位于被调用函数的代码段内, 该段可能只可执行, 故按取指语义读取
        raw = self._try_read_code_va(ra - 2, 2)
        if raw:
            h = int.from_bytes(raw, "little", signed=False)
            if (h & 0x3) != 3 and (
                (h & self._C_JAL_MASK) == self._C_JAL_MATCH
                or (h & self._C_JALR_MASK) == self._C_JALR_MATCH
            ):
                return ra - 2

        # -- 尝试 ra-4: 32-bit 调用 --
        _ = self._try_read_code_va(ra - 4, 4)
        return ra - 4

    # ----------------------------------------------------------
    #  帧链接读取
    # ----------------------------------------------------------

    def _try_read_frame_link(self, fp: int) -> tuple[int, int] | None:
        """读取 fp 指向的栈帧链接 (saved_ra, saved_fp); 越界则返回 None.

        RISC-V 标准栈帧布局: fp-8 存返回地址, fp-16 存调用者的 fp.
        """
        ra_raw = self._try_read_va(fp - 8, 8)
        if ra_raw is None:
            return None
        ra = int.from_bytes(ra_raw, "little", signed=False)
        fp_raw = self._try_read_va(fp - 16, 8)
        if fp_raw is None:
            return None
        fp = int.from_bytes(fp_raw, "little", signed=False)
        return ra, fp

    # ----------------------------------------------------------
    #  Trap 入口解析
    # ----------------------------------------------------------

    def _parse_trap_save_offsets(
        self, tvec: int, max_instrs: int = 128
    ) -> tuple[int, int] | None:
        """从 trap 入口代码中提取 RA / FP 相对 trap 帧 SP 的保存偏移.

        扫描 *tvec* 起最多 *max_instrs* 条指令, 寻找形如
          sd x1, OFF(sp) / c.sdsp x1, OFF(sp)   -> ra_off = OFF
          sd x8, OFF(sp) / c.sdsp x8, OFF(sp)   -> fp_off = OFF
        的寄存器保存操作.  必须至少找到 RA 偏移才视为成功.
        """
        raw = self._try_read_code_va(tvec, max_instrs * 4)
        if raw is None:
            return None

        ra_off: int | None = None
        fp_off: int | None = None
        pos = 0
        while pos + 2 <= len(raw) and (ra_off is None or fp_off is None):
            half = int.from_bytes(raw[pos : pos + 2], "little", signed=False)
            is_compressed = (half & 0x3) != 3
            choice = decode_c_sdsp
            pos += 2
            if not is_compressed:
                choice = decode_sd_sp
                half = int.from_bytes(raw[pos - 2 : pos + 2], "little", signed=False)
                pos += 2
            # 判定用 None 比较: 保存偏移 0 合法 (sd x1, 0(sp)), 而该函数的失败返回值
            # 是 None. 用真值判定会把偏移 0 当作未匹配, 并在后续指令上被覆写.
            if ra_off is None:
                ra_off = self._try_match_save(half, 1, choice)
            if fp_off is None:
                fp_off = self._try_match_save(half, 8, choice)

        return (ra_off, fp_off or 0) if ra_off is not None else None

    @staticmethod
    def _try_match_save(instr: int, reg: int, decoder) -> int | None:
        """若 *instr* 是 ``sd x{reg}, OFF(sp)`` 则返回 OFF, 否则 None."""
        decoded = decoder(instr)
        if decoded is None:
            return None
        rs2, offset = decoded
        return offset if rs2 == reg else None

    # ----------------------------------------------------------
    #  边界帧解析
    # ----------------------------------------------------------

    def _resolve_boundary_frame(self, trapped_pc: int, prev_name: str) -> tuple[str, str]:
        """解析跨特权级边界帧的 privilege mode."""
        if prev_name == "U" and trapped_pc >= (1 << 63):
            return ("S", "")
        if prev_name != "M" and self._image:
            seg = self._find_segment(trapped_pc - self._load_offset)
            if seg and seg.name and ".text" in seg.name:
                return ("M", "")
        return (prev_name, "")

    def _is_valid_mmode_code(self, pc: int) -> bool:
        """检查 *pc* 是否落在已知的 M 模式代码段内."""
        if not self._image:
            return False
        seg = self._find_segment(pc - self._load_offset)
        return seg and seg.name and ".text" in seg.name

    def _resolve_frame_mode(self, pc: int, default_mode: str) -> str:
        """根据帧 PC 所在段推断真实特权级 (M-mode .text vs S-mode .payload)."""
        if self._image is None:
            return default_mode
        h = self.hart
        if h.mode.name != "S" or h.mpp.value >= h.mode.value:
            return default_mode
        seg = self._find_segment(pc - self._load_offset)
        if seg and seg.name and seg.name == ".text":
            return "M"
        return default_mode

    # ----------------------------------------------------------
    #  FP 链回溯
    # ----------------------------------------------------------

    @staticmethod
    def _is_invalid_link(
        saved_fp: int,
        saved_ra: int,
        current_fp: int,
        *,
        fp0: int,
        visited: set[int],
    ) -> bool:
        """返回 True 表示帧链接无效, 应终止 FP 链回溯.

        无效条件:
          - saved_fp 为零、不回退 (<= current_fp)、或未 8-byte 对齐
          - saved_ra 为零 (无返回地址)
          - saved_fp 已出现在已访问集合中 (含初始帧 fp0), 防死循环
        """
        return (
            saved_fp in ({0, fp0} | visited)
            or saved_fp <= current_fp
            or (saved_fp & 0x7) != 0
            or saved_ra == 0
        )

    def _resolve_prev_ra(self, frames: list[StackFrame], saved_ra: int) -> tuple[int, bool]:
        """解析当前帧的调用者 RA.

        帧 #1 (len(frames)==1): 优先当前 hart 的 live x1 (ra);
          若已被函数破坏则降级为栈上 *saved_ra*; 两者均无效时设
          ra_corrupted=True 并返回最佳猜测值 (供 call_site 推算).
        更深帧: 直接使用栈上 *saved_ra*.

        Returns (prev_ra, ra_corrupted).
        """
        # -- 更深帧: 直接使用栈上 saved_ra --
        if len(frames) != 1:
            ok = self._check_return_addr(saved_ra)
            return saved_ra, (not ok)

        # -- 帧 #1: 优先 live ra, 再 fallback 到栈上 saved_ra --
        live_ra = frames[0].ra
        if self._check_return_addr(live_ra):
            return live_ra, False
        if self._check_return_addr(saved_ra):
            return saved_ra, False
        guess = live_ra if live_ra != 0 else saved_ra
        return guess, True

    def _append_ra_inferred_frames(self, frames: list[StackFrame]) -> None:
        """FP 链终止后, 从最后有效 RA 推断额外调用者帧."""
        choice = frames[-1] if len(frames) >= 1 else None
        if choice is None or not self._check_return_addr(choice.ra):
            return
        frames.append(
            StackFrame(
                idx=len(frames),
                fp=0,
                sp=choice.fp,
                ra=0,
                pc=self._call_site_pc(choice.ra),
                mode=choice.mode,
                note="无显式返回地址",
            )
        )

    def _walk_frame_chain(self) -> list[StackFrame]:
        """沿 FP 链遍历调用栈, 返回 StackFrame 列表."""
        h = self.hart
        current_fp: int = h.gprs[8]  # s0/fp
        cur_mode = h.mode.name
        frames: list[StackFrame] = [
            StackFrame(
                idx=0,
                fp=current_fp,
                sp=h.gprs[2],
                ra=h.gprs[1],
                pc=h.pc,
                mode=cur_mode,
            )
        ]
        visited: set[int] = {current_fp}
        _seen_lower_mode = False

        while True:
            if current_fp == 0:
                break
            link = self._try_read_frame_link(current_fp)
            if link is None:
                break
            saved_ra, saved_fp = link

            if self._is_invalid_link(
                saved_fp,
                saved_ra,
                current_fp,
                fp0=frames[0].fp,
                visited=visited,
            ):
                break
            visited.add(saved_fp)

            prev_ra, ra_corrupted = self._resolve_prev_ra(frames, saved_ra)
            call_site = self._call_site_pc(prev_ra)
            note = "RA 可能已被当前函数覆盖 (非合法代码地址)" if ra_corrupted else ""

            if not _seen_lower_mode:
                fmode = self._resolve_frame_mode(call_site, cur_mode)
                if fmode != cur_mode:
                    _seen_lower_mode = True
            else:
                fmode = frames[-1].mode

            frames.append(
                StackFrame(
                    idx=len(frames),
                    fp=saved_fp,
                    sp=current_fp,
                    ra=saved_ra,
                    pc=call_site,
                    mode=fmode,
                    note=note,
                )
            )
            current_fp = saved_fp

        # FP 链终止后的 RA 推断 + 跨特权级回溯
        self._append_ra_inferred_frames(frames)
        effective_mode = frames[-1].mode if len(frames) > 1 else cur_mode
        self._walk_prev_mode_frames(frames, effective_mode, visited)
        return frames

    def _walk_prev_mode_frames(
        self, frames: list[StackFrame], cur_mode: str, visited: set[int]
    ) -> None:
        """尝试解析 trap 入口, 定位被中断上下文并继续回溯低特权级 FP 链."""
        h = self.hart
        context = self._trap_entry_context(h)
        if context is None:
            return
        entry, s_trapped_pc, s_prev_mode = context
        if entry.trapped_pc == 0 or entry.prev_mode.value >= h.mode.value or entry.tvec == 0:
            return

        # 若 trapped_pc 已与现有序号帧的 PC 重合 (含 ±4),
        # 说明该 trap 上下文已被 FP 链覆盖, 不重复添加.
        for f in frames:
            if abs(f.pc - entry.trapped_pc) <= 4:
                return

        prev_name = entry.prev_mode.name
        is_nested = (
            h.mode in (RiscvMode.M, RiscvMode.D)
            and s_trapped_pc != 0
            and entry.prev_mode.value < RiscvMode.M.value
            and self._is_valid_mmode_code(entry.trapped_pc)
        )

        boundary = self._resolve_boundary_frame(entry.trapped_pc, prev_name)
        offsets = self._parse_trap_save_offsets(entry.tvec)

        if offsets:
            self._add_prev_mode_frame(frames, entry, visited, offsets)
            return
        if entry.prev_mode == RiscvMode.U and not is_nested:
            # S->U 边界: 指令解析失败时, 尝试常见 trap 帧布局以恢复
            # U-mode 的 FP/SP/RA 并继续 U-mode FP 链回溯.
            self._add_prev_mode_frame_fallback(frames, entry, visited, boundary)
            return
        payload = self._boundary_payload(boundary[1], is_nested)
        frames.append(self._pc_frame(frames, entry.trapped_pc, boundary[0], payload))
        if is_nested:
            nested_entry = TrapEntry(h.stvec_val, s_trapped_pc, s_prev_mode)
            self._add_prev_mode_frame(frames, nested_entry, visited, offsets)

    def _trap_entry_context(self, h) -> tuple[TrapEntry, int, RiscvMode] | None:
        """取当前特权级 trap 入口的参数, 以及嵌套回溯所需的 S 侧现场.

        返回 (主 trap 入口, S 侧 trapped_pc, S 侧 prev_mode). 当前模式不在
        M/D/S 之内, 或 S 侧的 sepc 判定为残留值时返回 None.
        """
        if h.mode not in (RiscvMode.M, RiscvMode.D, RiscvMode.S):
            return None
        if h.mode in (RiscvMode.M, RiscvMode.D):
            prev_mode = h.mpp
            s_trapped_pc = h.sepc_val
            s_prev_mode = h.spp
            if s_prev_mode == RiscvMode.U and s_trapped_pc >= (1 << 63):
                s_prev_mode = RiscvMode.S
            entry = TrapEntry(h.mtvec_val, h.mepc_val, prev_mode)
            return entry, s_trapped_pc, s_prev_mode

        trapped_pc = h.sepc_val
        prev_mode = h.spp
        if prev_mode == RiscvMode.U and trapped_pc >= (1 << 63):
            prev_mode = RiscvMode.S
        # 检测残留 sepc: 从 M 模式经 mret 进入 S 模式后 SPP=U 且 sepc 为残留 RAM 地址,
        # 或 Sv39 启用时 sepc 落在 RAM 物理地址范围, 即非规范虚拟地址.
        # 仅当 Sv39 启用或 trapped_pc 高于 RAM 窗口时才认为无效,
        # Bare 翻译下 U 模式的虚拟地址即 RAM 物理地址, 属合法 trap 现场.
        # 直接从 satp CSR 读 MODE 字段, 避免依赖缓存 _mmu_mode.
        satp_mode = (h.satp_val >> 60) & 0xF
        if self._emu.bus.is_ram_addr(trapped_pc):
            if satp_mode != 0 or prev_mode != RiscvMode.U:
                # Sv39 启用则内核与 sepc 应为规范高虚拟地址, 该地址为残留值
                # Bare 翻译下 SPP=S 而 sepc 落在 RAM 窗口: 按残留值处理, 跳过
                return None
        entry = TrapEntry(h.stvec_val, trapped_pc, prev_mode)
        return entry, 0, RiscvMode.U

    @staticmethod
    def _boundary_payload(boundary_note: str, is_nested: bool) -> str:
        """给出边界帧的说明文本."""
        if boundary_note:
            return boundary_note
        if is_nested:
            return "因mepc嵌套保存trap, 无法解析寄存器布局"
        return "trap 入口寄存器保存布局无法解析 (仅 PC)"

    @staticmethod
    def _pc_frame(frames: list[StackFrame], pc: int, mode: str, note: str) -> StackFrame:
        """构造只含 PC 的边界帧."""
        return StackFrame(idx=len(frames), fp=0, sp=0, ra=0, pc=pc, note=note, mode=mode)

    def _add_prev_mode_frame(
        self,
        frames: list[StackFrame],
        entry: TrapEntry,
        visited: set[int],
        offsets: tuple[int, int] | None = None,
    ) -> None:
        """为指定 trap 上下文添加边界帧, 并尝试继续回溯低特权级 FP 链.

        若 *offsets* 为 None, 从 entry.tvec 指令流解析 RA/FP 保存偏移.
        """
        trapped_pc, tvec = entry.trapped_pc, entry.tvec
        if trapped_pc == 0 or tvec == 0:
            return
        prev_name = entry.prev_mode.name
        boundary_mode, boundary_note = self._resolve_boundary_frame(trapped_pc, prev_name)
        if offsets is None:
            offsets = self._parse_trap_save_offsets(tvec)
        if offsets is None:
            note = boundary_note or "trap 入口寄存器保存布局无法解析 (仅 PC)"
            frames.append(self._pc_frame(frames, trapped_pc, boundary_mode, note))
            return
        search_base = self._frame_search_base(frames)
        if search_base == 0:
            note = "仅获知trap现场PC，缺少栈帧回溯参考"
            frames.append(self._pc_frame(frames, trapped_pc, boundary_mode, note))
            return
        ra_off, fp_off = offsets
        trap_sp = self._find_trap_sp(search_base, trapped_pc, ra_off)
        saved_ra, saved_fp, saved_sp = self._load_trap_saved_regs(trap_sp, ra_off, fp_off)
        note = boundary_note
        if saved_fp is None and saved_sp is None and saved_ra is None:
            note = "仅 trap PC"
        frames.append(
            StackFrame(
                idx=len(frames),
                fp=saved_fp or 0,
                sp=saved_sp or 0,
                ra=saved_ra or 0,
                pc=trapped_pc,
                note=note,
                mode=boundary_mode,
            )
        )
        if saved_fp and saved_fp not in visited:
            visited.add(saved_fp)
            self._walk_fp_chain(frames, saved_fp, visited, prev_name)

    @staticmethod
    def _frame_search_base(frames: list[StackFrame]) -> int:
        """取最后一个帧的 FP 或 SP, 对齐到 8 字节作为搜索起点."""
        last_fp = frames[-1].fp if frames else 0
        last_sp = frames[-1].sp if frames else 0
        return min(last_fp or last_sp, last_sp or last_fp) & ~0x7

    def _find_trap_sp(self, search_base: int, trapped_pc: int, ra_off: int) -> int | None:
        """在 search_base 起的 4096 字节内寻找保存了 trapped_pc 的 trap 帧."""
        raw = self._emu.bus.try_read(search_base, 4096)
        if raw is None:
            return None
        mepc_bytes = struct.pack("<Q", trapped_pc)
        pos = len(raw) - 8
        while pos >= 0:
            if raw[pos : pos + 8] != mepc_bytes:
                pos -= 8
                continue
            trial_sp = self._match_trap_sp(search_base + pos, ra_off)
            if trial_sp is not None:
                return trial_sp
            pos -= 8
        return None

    def _match_trap_sp(self, candidate_base: int, ra_off: int) -> int | None:
        """在候选帧基址附近试出 trap 帧的寄存器保存基址."""
        for mepc_guess in range(0, 5):
            trial_sp = candidate_base - (mepc_guess << 6)
            ra_val = self._emu.bus.read_u64(trial_sp + ra_off)
            if ra_val is None or not self._emu.bus.is_valid_addr(ra_val):
                continue
            zero_raw = self._emu.bus.read_u64(trial_sp)
            if zero_raw is None or zero_raw != 0:
                continue
            return trial_sp
        return None

    def _load_trap_saved_regs(
        self,
        trap_sp: int | None,
        ra_off: int,
        fp_off: int,
    ) -> tuple[int | None, int | None, int | None]:
        """读出 trap 帧保存的 (ra, fp, sp); 未找到 trap 帧时三者都是 None."""
        if trap_sp is None:
            return None, None, None
        saved_ra = self._emu.bus.read_u64(trap_sp + ra_off)
        # 此处按"是否需要"判定而非按"是否提供"判定: 偏移 0 表示该寄存器在
        # 本次布局中未被保存, 见 _FALLBACK_TRAP_OFFSETS 的记载.
        saved_fp = self._emu.bus.read_u64(trap_sp + fp_off) if fp_off else None
        saved_sp = self._emu.bus.read_u64(trap_sp + ra_off + 8)
        return saved_ra, saved_fp, saved_sp

    def _walk_fp_chain(
        self,
        frames: list[StackFrame],
        start_fp: int,
        visited: set[int],
        mode_name: str,
    ) -> None:
        """从 start_fp 沿 FP 链追加栈帧, 最多 64 帧."""
        current_fp = start_fp
        for _ in range(64):
            if current_fp == 0:
                break
            link = self._try_read_frame_link(current_fp)
            if link is None or link[0] == 0 or link[1] == 0:
                break
            next_ra, next_fp = link
            if next_fp <= current_fp or (next_fp & 0x7) or next_fp in visited:
                break
            visited.add(next_fp)
            frames.append(
                StackFrame(
                    idx=len(frames),
                    fp=next_fp,
                    sp=current_fp,
                    ra=next_ra,
                    pc=self._call_site_pc(next_ra),
                    mode=mode_name,
                )
            )
            current_fp = next_fp

    def _add_prev_mode_frame_fallback(
        self,
        frames: list[StackFrame],
        entry: TrapEntry,
        visited: set[int],
        boundary: tuple[str, str],
    ) -> None:
        """指令解析失败时, 尝试常见 trap 帧布局以恢复低特权级上下文.

        依次尝试 _FALLBACK_TRAP_OFFSETS 中的 (ra_off, fp_off) 对,
        首个成功恢复 saved_fp 的布局生效并继续 FP 链回溯.
        全部失败则退回到 "仅 trap PC" 帧.
        """
        if entry.trapped_pc == 0 or entry.tvec == 0:
            return
        for offsets in _FALLBACK_TRAP_OFFSETS:
            # 临时试构建 — 捕获是否成功找到 saved_fp
            trial: list[StackFrame] = []
            self._add_prev_mode_frame(trial, entry, visited, offsets)
            if not trial or trial[-1].fp == 0:
                continue
            # 该布局恢复了 saved_fp, 把试构建的帧并入正式帧列表后返回
            frames.extend(trial)
            # 将 trial 中所有有效 FP 加入 visited 以避免 FP 链循环
            for f in trial:
                if f.fp != 0:
                    visited.add(f.fp)
            return
        # 全部 fallback 失败: 添加仅 PC 帧
        note = boundary[1] or "trap 入口寄存器保存布局无法解析 (仅 PC)"
        frames.append(self._pc_frame(frames, entry.trapped_pc, boundary[0], note))

    # ----------------------------------------------------------
    #  帧标注
    # ----------------------------------------------------------

    def _fmt_frame_where(self, pc: int, color: str) -> str:
        """为帧 PC 构造 '段:函数名' 标注字符串."""
        fn = self._resolve_any_symbol_name(pc) or ""
        seg = self._find_segment(pc - self._load_offset)
        seg_name = seg.name if seg and seg.name else ""
        if seg_name and fn:
            return f"[dim]{seg_name}[/]:[{color}]{fn}[/]"
        if fn:
            return f"[{color}]{fn}[/]"
        if seg_name:
            return f"[dim]{seg_name}[/]"
        return ""

    # ----------------------------------------------------------
    #  栈帧命令
    # ----------------------------------------------------------

    @seize_val_err("无效帧号")
    def cmd_frame(self, arg: str | None = None) -> None:
        """栈帧回溯 — #01 起始编号.

        stack / bt   — 显示全部帧 + 当前帧的栈内存
        frame <N>    — 切换到第 N 帧并刷新回溯 (N 为 0-indexed)
        """
        self._stack_frames = self._walk_frame_chain()

        if arg:
            n = int(arg, 0)
            if not (0 <= n < len(self._stack_frames)):
                self._err(f"帧号 {n} 超出范围 [0, {len(self._stack_frames) - 1}]")
                return
            self._current_frame_idx = n

        tbl = Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("frame", style="bold")
        tbl.add_column("where")
        tbl.add_column("regs", style="dim")
        for f in self._stack_frames:
            tag = f"#{f.idx + 1:02d}"
            color = self._MODE_COLORS.get(f.mode, "")
            mode_tag = f"[{color}]{f.mode}[/]" if color else ""
            where = self._fmt_frame_where(f.pc, color)

            # 寄存器字段: pc 始终显示, sp/fp/ra 仅非零时显示
            parts = [f"pc={hex_addr(f.pc)}"]
            if f.sp != 0:
                parts.append(f"sp={hex_addr(f.sp)}")
            if f.fp != 0:
                parts.append(f"fp={hex_addr(f.fp)}")
            if f.ra != 0:
                parts.append(f"ra={hex_addr(f.ra)}")
            if f.note:
                parts.append(f"[bold magenta]{f.note}[/]")
            regs = "  ".join(parts)

            tbl.add_row(f"{tag} {mode_tag}", where, regs)

        self._console.print(
            f"[bold]执行栈帧[/] (深度 {len(self._stack_frames)})",
            tbl,
        )

        # 当前帧栈内存
        if self._current_frame_idx >= len(self._stack_frames):
            return
        cur = self._stack_frames[self._current_frame_idx]
        # sp=0 帧 (边界帧/残留 sepc 帧) 无有效栈内存, 不尝试读取
        if cur.sp == 0:
            self._console.print(
                f"[dim]帧 #{self._current_frame_idx + 1:02d} — "
                "无有效栈指针 (sp=0), 跳过栈内存显示[/]"
            )
            return
        # 经 VA 到 PA 的翻译读取栈内存
        stack_data = self._try_read_va(cur.sp, 64)
        if stack_data is None:
            self._console.print(
                f"[dim]帧 #{self._current_frame_idx + 1:02d} sp={hex_addr(cur.sp)} — "
                "栈内存不可读 (翻译失败或 PA 读失败)[/]"
            )
            return
        self._console.print(
            f"[dim]帧 #{self._current_frame_idx + 1:02d} "
            f"sp={hex_addr(cur.sp)} 栈内存:[/]\n" + dye_hexdump(stack_data, cur.sp)
        )
