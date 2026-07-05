#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""StackWalkMixin — FP 链回溯、跨特权级边界帧、栈帧显示.

依赖 DebuggerBase + MemoryMixin._try_read_va + SymbolMixin.
"""

import struct

from rich.table import Table

from pyremu.core.decoder import decode_c_sdsp, decode_sd_sp
from pyremu.core.hart import RiscvMode
from pyremu.debug._attrs import SharedMixinAttrs
from pyremu.debug.types import StackFrame
from pyremu.debug.utils import hex_addr
from pyremu.utils.wrapper import seize_val_err


class StackWalkMixin(SharedMixinAttrs):
    """栈帧回溯 (backtrace)."""

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
        """解析 trap 入口代码, 提取 RA/FP 相对 trap SP 的保存偏移."""
        raw = self._emu.bus.try_read(tvec, max_instrs * 4)
        if raw is None:
            return None
        ra_off: int | None = None
        fp_off: int | None = None
        pos = 0
        while pos + 2 <= len(raw) and (ra_off is None or fp_off is None):
            half = int.from_bytes(raw[pos:pos + 2], "little", signed=False)
            if (half & 0x3) != 3:  # 16-bit compressed
                decoded = decode_c_sdsp(half)
                if decoded is not None:
                    rs2, uimm = decoded
                    if rs2 == 1 and ra_off is None:
                        ra_off = uimm
                    elif rs2 == 8 and fp_off is None:
                        fp_off = uimm
                pos += 2
                continue
            # 32-bit
            instr = int.from_bytes(raw[pos:pos + 4], "little", signed=False)
            decoded = decode_sd_sp(instr)
            if decoded is not None:
                rs2, imm = decoded
                if rs2 == 1 and ra_off is None:
                    ra_off = imm
                elif rs2 == 8 and fp_off is None:
                    fp_off = imm
            pos += 4

        if ra_off is None:
            return None
        return ra_off, (fp_off if fp_off is not None else 0)

    # ----------------------------------------------------------
    #  边界帧解析
    # ----------------------------------------------------------

    def _resolve_boundary_frame(
        self, trapped_pc: int, prev_name: str
    ) -> tuple[str, str]:
        """解析跨特权级边界帧的 privilege mode."""
        if prev_name == "U" and trapped_pc >= (1 << 63):
            return ("S", "")
        if prev_name != "M" and self._image:
            seg = self._find_segment(trapped_pc - self._load_offset)
            if seg is not None and seg.name and ".text" in seg.name:
                return ("M", "")
        return (prev_name, "")

    def _is_valid_mmode_code(self, pc: int) -> bool:
        """检查 *pc* 是否落在已知的 M 模式代码段内."""
        if not self._image:
            return False
        seg = self._find_segment(pc - self._load_offset)
        return seg is not None and seg.name is not None and ".text" in seg.name

    def _resolve_frame_mode(self, pc: int, default_mode: str) -> str:
        """根据帧 PC 所在段推断真实特权级 (M-mode .text vs S-mode .payload)."""
        if self._image is None:
            return default_mode
        h = self.hart
        if h.mode.name != "S":
            return default_mode
        if h.mpp.value >= h.mode.value:
            return default_mode
        seg = self._find_segment(pc - self._load_offset)
        if seg is not None and seg.name is not None:
            if seg.name == ".text":
                return "M"
        return default_mode

    # ----------------------------------------------------------
    #  FP 链回溯
    # ----------------------------------------------------------

    def _walk_frame_chain(self) -> list[StackFrame]:
        """沿 FP 链遍历调用栈, 返回 StackFrame 列表."""
        h = self.hart
        current_fp: int = h.gprs[8]  # s0/fp
        visited: set[int] = {current_fp}
        cur_mode = h.mode.name
        frames: list[StackFrame] = [StackFrame(
            idx=0, fp=current_fp, sp=h.gprs[2], ra=h.gprs[1],
            pc=h.pc, mode=cur_mode,
        )]
        _seen_lower_mode = False

        while True:
            if current_fp == 0:
                break
            link = self._try_read_frame_link(current_fp)
            if link is None:
                break
            saved_ra, saved_fp = link
            if saved_fp == 0:
                break
            if saved_fp <= current_fp or (saved_fp & 0x7) or saved_fp in visited:
                break
            if saved_fp == frames[0].fp:
                break
            if saved_ra == 0:
                break
            visited.add(saved_fp)
            # 帧 #1 优先使用 live x1; 若 live x1 被当前函数用作临时寄存器
            # (如 blake2s G 宏), 降级使用栈上保存的 RA
            ra_corrupted = False
            if len(frames) == 1:
                prev_ra = frames[0].ra if frames[0].ra != 0 else saved_ra
                if not self._is_valid_code_va(prev_ra):
                    if saved_ra != 0 and self._is_valid_code_va(saved_ra):
                        prev_ra = saved_ra
                    else:
                        ra_corrupted = True
            else:
                prev_ra = saved_ra
                if not self._is_valid_code_va(prev_ra):
                    ra_corrupted = True
            call_site = prev_ra - 4 if prev_ra >= 4 else 0
            note = "RA 可能已被当前函数覆盖 (非合法代码地址)" if ra_corrupted else ""

            if not _seen_lower_mode:
                fmode = self._resolve_frame_mode(call_site, cur_mode)
                if fmode != cur_mode:
                    _seen_lower_mode = True
            else:
                fmode = frames[-1].mode
            frames.append(StackFrame(
                idx=len(frames), fp=saved_fp, sp=current_fp,
                ra=saved_ra, pc=call_site, mode=fmode, note=note,
            ))
            current_fp = saved_fp

        effective_mode = frames[-1].mode if len(frames) > 1 else cur_mode
        self._walk_prev_mode_frames(frames, effective_mode, visited)
        return frames

    def _walk_prev_mode_frames(
        self, frames: list[StackFrame], cur_mode: str, visited: set[int]
    ) -> None:
        """尝试解析 trap 入口, 定位被中断上下文并继续回溯低特权级 FP 链."""
        h = self.hart
        mode_val = h.mode.value

        if h.mode in set((RiscvMode.M, RiscvMode.D)):
            tvec = h.mtvec_val
            trapped_pc = h.mepc_val
            prev_mode = h.mpp
            s_trapped_pc = h.sepc_val
            s_prev_mode = h.spp
            if s_prev_mode == RiscvMode.U and s_trapped_pc >= (1 << 63):
                s_prev_mode = RiscvMode.S
        elif h.mode == RiscvMode.S:
            tvec = h.stvec_val
            trapped_pc = h.sepc_val
            prev_mode = h.spp
            if prev_mode == RiscvMode.U and trapped_pc >= (1 << 63):
                prev_mode = RiscvMode.S
            # M→S mret 后 SPP=U 且 sepc 为裸 RAM 地址 → 残留 sepc
            if (
                prev_mode == RiscvMode.U
                and self._emu.bus.is_ram_addr(trapped_pc)
            ):
                return
            # Sv39 启用时, sepc 不应为裸 RAM 地址 (内核运行在高 VA).
            # 若 sepc 落在 RAM PA 范围且 MMU 开启, 必为残留值.
            # 直接从 satp CSR 读 MODE 字段, 避免依赖缓存 _mmu_mode.
            satp_mode = (h.satp_val >> 60) & 0xF
            if (
                satp_mode != 0
                and self._emu.bus.is_ram_addr(trapped_pc)
            ):
                return
            s_trapped_pc = 0
            s_prev_mode = RiscvMode.U
        else:
            return

        if trapped_pc == 0 or prev_mode.value > mode_val or tvec == 0:
            return

        prev_name = prev_mode.name
        _nested = (
            h.mode in (RiscvMode.M, RiscvMode.D)
            and s_trapped_pc != 0
            and prev_mode.value < RiscvMode.M.value
            and self._is_valid_mmode_code(trapped_pc)
        )

        boundary_mode, boundary_note = self._resolve_boundary_frame(
            trapped_pc, prev_name
        )
        offsets = self._parse_trap_save_offsets(tvec)
        if offsets is None:
            payload = boundary_note
            if not boundary_note and _nested:
                payload = "M-mode 嵌套 trap (mepc), 寄存器保存布局无法解析"
            elif not boundary_note and not _nested:
                payload = "trap 入口寄存器保存布局无法解析 (仅 PC)"
            frames.append(StackFrame(
                idx=len(frames), fp=0, sp=0, ra=0,
                pc=trapped_pc, note=(payload), mode=boundary_mode,
            ))
            if _nested:
                self._add_prev_mode_frame(
                    frames, s_trapped_pc, s_prev_mode, h.stvec_val, visited
                )
            return
        self._add_prev_mode_frame(
            frames, trapped_pc, prev_mode, tvec, visited
        )

    def _add_prev_mode_frame(
        self, frames: list[StackFrame], trapped_pc: int,
        prev_mode: RiscvMode, tvec: int, visited: set[int],
    ) -> None:
        """为指定 trap 上下文添加边界帧, 并尝试继续回溯低特权级 FP 链."""
        if trapped_pc == 0 or tvec == 0:
            return
        prev_name = prev_mode.name
        boundary_mode, boundary_note = self._resolve_boundary_frame(
            trapped_pc, prev_name
        )
        offsets = self._parse_trap_save_offsets(tvec)
        if offsets is None:
            frames.append(StackFrame(
                idx=len(frames), fp=0, sp=0, ra=0,
                pc=trapped_pc,
                note=(boundary_note or "trap 入口寄存器保存布局无法解析 (仅 PC)"),
                mode=boundary_mode,
            ))
            return
        ra_off, fp_off = offsets

        last_fp = frames[-1].fp if frames else 0
        last_sp = frames[-1].sp if frames else 0
        search_base = min(
            last_fp if last_fp else last_sp,
            last_sp if last_sp else last_fp,
        ) & ~0x7
        if search_base == 0:
            frames.append(StackFrame(
                idx=len(frames), fp=0, sp=0, ra=0,
                pc=trapped_pc, note="仅 trap PC (缺少栈帧参考点)",
                mode=boundary_mode,
            ))
            return
        raw = self._emu.bus.try_read(search_base, 4096)
        if raw is None:
            return
        mepc_bytes = struct.pack("<Q", trapped_pc)
        trap_sp: int | None = None
        pos = len(raw) - 8
        while pos >= 0:
            if raw[pos:pos + 8] != mepc_bytes:
                pos -= 8
                continue
            candidate_base = search_base + pos
            for mepc_guess in (256, 0, 128, 64, 192):
                trial_sp = candidate_base - mepc_guess
                ra_addr = trial_sp + ra_off
                ra_raw = self._emu.bus.try_read(ra_addr, 8)
                if ra_raw is None:
                    continue
                ra_val = int.from_bytes(ra_raw, "little", signed=False)
                if not self._emu.bus.is_valid_addr(ra_val):
                    continue
                zero_raw = self._emu.bus.try_read(trial_sp, 8)
                if zero_raw is None:
                    continue
                if int.from_bytes(zero_raw, "little", signed=False) != 0:
                    continue
                trap_sp = trial_sp
                break
            if trap_sp is not None:
                break
            pos -= 8

        saved_ra = saved_fp = saved_sp = None
        if trap_sp is not None:
            saved_ra = self._emu.bus.read_u64(trap_sp + ra_off)
            if fp_off:
                saved_fp = self._emu.bus.read_u64(trap_sp + fp_off)
            saved_sp = self._emu.bus.read_u64(trap_sp + ra_off + 8)

        call_site = trapped_pc - 4 if trapped_pc >= 4 else 0
        note = boundary_note
        if saved_fp is None and saved_sp is None and saved_ra is None:
            note = "仅 trap PC (栈扫描无匹配)"
        frames.append(StackFrame(
            idx=len(frames), fp=saved_fp or 0, sp=saved_sp or 0,
            ra=saved_ra or 0, pc=call_site, note=note, mode=boundary_mode,
        ))

        if not (saved_fp and saved_fp != 0 and saved_fp not in visited):
            return
        visited.add(saved_fp)
        current_fp = saved_fp
        for _ in range(64):
            if current_fp == 0:
                break
            link = self._try_read_frame_link(current_fp)
            if link is None:
                break
            next_ra, next_fp = link
            if next_fp == 0:
                break
            if next_fp <= current_fp or (next_fp & 0x7) or next_fp in visited:
                break
            if next_ra == 0:
                break
            visited.add(next_fp)
            call_site = next_ra - 4 if next_ra >= 4 else 0
            frames.append(StackFrame(
                idx=len(frames), fp=next_fp, sp=current_fp,
                ra=next_ra, pc=call_site, mode=prev_name,
            ))
            current_fp = next_fp

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

        if arg is not None:
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
            if f.note:
                if f.sp != 0:
                    regs = (
                        f"pc={hex_addr(f.pc)}  sp={hex_addr(f.sp)}  "
                        f"fp={hex_addr(f.fp)}  ra={hex_addr(f.ra)}  "
                        f"[bold magenta]{f.note}[/]"
                    )
                else:
                    regs = f"pc={hex_addr(f.pc)}  [bold magenta]{f.note}[/]"
            else:
                regs = (
                    f"pc={hex_addr(f.pc)}  sp={hex_addr(f.sp)}  "
                    f"fp={hex_addr(f.fp)}  ra={hex_addr(f.ra)}"
                )
            tbl.add_row(f"{tag} {mode_tag}", where, regs)

        self._console.print(
            f"[bold]执行栈帧[/] (深度 {len(self._stack_frames)})", tbl,
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
        # 使用 VA→PA 翻译读取栈内存 (Sv39 等 MMU 模式下 VA 非物理地址)
        stack_data = self._try_read_va(cur.sp, 64)
        if stack_data is None:
            self._console.print(
                f"[dim]帧 #{self._current_frame_idx + 1:02d} sp={hex_addr(cur.sp)} — "
                "栈内存不可读 (翻译失败或 PA 读失败)[/]"
            )
            return
        self._console.print(
            f"[dim]帧 #{self._current_frame_idx + 1:02d} "
            f"sp={hex_addr(cur.sp)} 栈内存:[/]\n"
            + _fmt_hexdump(stack_data, cur.sp)
        )


def _fmt_hexdump(data: bytes, base_addr: int = 0) -> str:
    """格式化字节为 hexdump 字符串, 地址列 dim 灰, hex 值默认亮色, ASCII 区 dim 灰."""
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        hex_bytes = " ".join(f"{b:02x}" for b in chunk)
        ascii_chars = "".join(
            chr(b) if 32 <= b < 127 else "." for b in chunk
        )
        lines.append(
            f"  [dim]{hex_addr(base_addr + i)}[/]  "
            f"{hex_bytes:<48s}  "
            f"[dim]|{ascii_chars}|[/]"
        )
    return "\n".join(lines)
