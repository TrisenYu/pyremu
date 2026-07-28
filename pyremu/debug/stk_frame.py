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

# 跨特权级 trap 上下文中 RA/FP 保存偏移的常见布局.
# 当前指令级解析 (_parse_trap_save_offsets) 失败时,
# 按优先级依次尝试这些 (ra_off, fp_off) 对.  0 表示该偏移未知/不需要.
_FALLBACK_TRAP_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 8),    # 紧凑型: sd x1,0(sp); sd x8,8(sp)
    (8, 64),   # Linux pt_regs: ra=PT_RA(8), s0=PT_S0(64)
    (0, 16),   # 稀疏型: sd x1,0(sp); sd x8,16(sp)
    (8, 16),   # 类型 B: RA 在 gp 之后
    (8, 24),   # 类型 C
)

class StackWalkMixin(SharedMixinAttrs):
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
    #                                                  bit[12]=0
    # jal              opcode=110_1111,  rd=x1                    [4 B]
    # jalr             opcode=110_0111,  funct3=000, rd=x1        [4 B]
    # ----------------------------------------------------------

    _C_JAL_MASK: int  = 0xE003   # bits[15:13]=101, bits[1:0]=01
    _C_JAL_MATCH: int = 0x2001

    _C_JALR_MASK: int  = 0xF003  # bits[15:13]=100, bit[12]=0, bits[1:0]=10
    _C_JALR_MATCH: int = 0x9002

    @staticmethod
    def _is_plausible_ra(ra: int) -> bool:
        """快速拒绝明显无效的返回地址.

        返回 False 的值:
          - 0 (空指针)
          - 0xFFFFFFFFFFFFFFFF (未初始化/哨兵/函数破坏 x1)
          - < 4 (无法安全 ra-4)
          - 地址落在 64-bit 空间最高 64 KiB 内 (> 0xFFFF_FFFF_FFFF_0000).
            内核/用户代码的实际映射地址都远低于此边界.
        """
        if ra == 0 or ra == 0xFFFF_FFFF_FFFF_FFFF:
            return False
        if ra < 4:
            return False
        if ra > 0xFFFF_FFFF_FFFF_0000:
            return False
        return True

    def _call_site_pc(self, ra: int) -> int:
        """从返回地址推断调用指令的 PC.

        RISC-V 调用指令的返回地址为 PC+2 (c.jal/c.jalr) 或 PC+4
        (jal/jalr). 读取 ra 之前的指令字节以判定调用类型; 若无法
        确定则保守假设 4 字节.
        """
        if not self._is_plausible_ra(ra):
            return 0

        # -- 尝试 ra-2: 16-bit 压缩调用 --
        raw = self._try_read_va(ra - 2, 2)
        if raw is not None:
            h = int.from_bytes(raw, "little", signed=False)
            if (h & 0x3) != 3:
                if (h & self._C_JAL_MASK) == self._C_JAL_MATCH:
                    return ra - 2
                if (h & self._C_JALR_MASK) == self._C_JALR_MATCH:
                    return ra - 2

        # -- 尝试 ra-4: 32-bit 调用 --
        raw = self._try_read_va(ra - 4, 4)
        if raw is not None:
            w = int.from_bytes(raw, "little", signed=False)
            if ((w >> 7) & 0x1F) != 1:
                return ra - 4
            opc = w & 0x7F
            if opc == 0b1101111:
                return ra - 4
            if opc == 0b1100111 and ((w >> 12) & 0x7) == 0b000:
                return ra - 4

        # -- 无法确定: 假设 4-byte 调用 --
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
        raw = self._try_read_va(tvec, max_instrs * 4)
        if raw is None:
            return None

        ra_off: int | None = None
        fp_off: int | None = None
        pos = 0
        while pos + 2 <= len(raw) and (ra_off is None or fp_off is None):
            half = int.from_bytes(raw[pos:pos + 2], "little", signed=False)
            is_compressed = (half & 0x3) != 3
            choice = decode_c_sdsp
            pos += 2
            if not is_compressed:
                choice = decode_sd_sp
                half = int.from_bytes(raw[pos-2:pos+2], "little", signed=False)
                pos += 2
            ra_off = ra_off or self._try_match_save(half, 1, choice)
            fp_off = fp_off or self._try_match_save(half, 8, choice)

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
        if h.mode.name != "S" or h.mpp.value >= h.mode.value:
            return default_mode
        seg = self._find_segment(pc - self._load_offset)
        if seg is not None and seg.name is not None:
            if seg.name == ".text":
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
        if saved_fp == 0:
            return True
        if saved_fp <= current_fp:
            return True
        if (saved_fp & 0x7) != 0:
            return True
        if saved_ra == 0:
            return True
        if saved_fp == fp0 or saved_fp in visited:
            return True
        return False

    def _resolve_prev_ra(
        self, frames: list[StackFrame], saved_ra: int
    ) -> tuple[int, bool]:
        """解析当前帧的调用者 RA.

        帧 #1 (len(frames)==1): 优先当前 hart 的 live x1 (ra);
          若已被函数破坏则降级为栈上 *saved_ra*; 两者均无效时设
          ra_corrupted=True 并返回最佳猜测值 (供 call_site 推算).
        更深帧: 直接使用栈上 *saved_ra*.

        Returns (prev_ra, ra_corrupted).
        """
        # -- 更深帧: 直接使用栈上 saved_ra --
        if len(frames) != 1:
            ok = (
                saved_ra != 0
                and self._is_plausible_ra(saved_ra)
                and self._is_valid_code_va(saved_ra)
            )
            return saved_ra, (not ok)

        # -- 帧 #1: 优先 live ra, 再 fallback 到栈上 saved_ra --
        live_ra = frames[0].ra
        if live_ra != 0 and self._is_plausible_ra(live_ra) and self._is_valid_code_va(live_ra):
            return live_ra, False
        if saved_ra != 0 and self._is_plausible_ra(saved_ra) and self._is_valid_code_va(saved_ra):
            return saved_ra, False
        # 两者均无效: 用 live_ra (或 saved_ra) 作为最佳猜测
        guess = live_ra if live_ra != 0 else saved_ra
        return guess, True

    def _append_ra_inferred_frames(self, frames: list[StackFrame]) -> None:
        """FP 链终止后, 从最后有效 RA 推断额外调用者帧."""
        if len(frames) == 1:
            ra = frames[0].ra
            if ra != 0 and self._is_plausible_ra(ra) and self._is_valid_code_va(ra):
                frames.append(StackFrame(
                    idx=1, fp=0, sp=frames[0].fp, ra=0,
                    pc=self._call_site_pc(ra), mode=frames[0].mode,
                    note="调用者 (FP 链不可达, 由 RA 推断)",
                ))
        elif len(frames) >= 2:
            last_ra = frames[-1].ra
            if last_ra != 0 and self._is_plausible_ra(last_ra) and self._is_valid_code_va(last_ra):
                frames.append(StackFrame(
                    idx=len(frames), fp=0, sp=frames[-1].fp, ra=0,
                    pc=self._call_site_pc(last_ra), mode=frames[-1].mode,
                    note="调用者 (FP 链终止, 由 RA 推断)",
                ))

    def _walk_frame_chain(self) -> list[StackFrame]:
        """沿 FP 链遍历调用栈, 返回 StackFrame 列表."""
        h = self.hart
        current_fp: int = h.gprs[8]  # s0/fp
        cur_mode = h.mode.name
        frames: list[StackFrame] = [StackFrame(
            idx=0, fp=current_fp, sp=h.gprs[2], ra=h.gprs[1],
            pc=h.pc, mode=cur_mode,
        )]
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
                saved_fp, saved_ra, current_fp,
                fp0=frames[0].fp, visited=visited,
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

            frames.append(StackFrame(
                idx=len(frames), fp=saved_fp, sp=current_fp,
                ra=saved_ra, pc=call_site, mode=fmode, note=note,
            ))
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
            # 检测残留 sepc: M->S mret 后 SPP=U 且 sepc 为过时 RAM 地址,
            # 或 Sv39 启用时 sepc 落在 RAM PA 范围 (非规范 VA).
            # 仅当 MMU 开启或 trapped_pc 高于 RAM 窗口时才认为无效 —
            # Bare 翻译下 U-mode PA 即 RAM 地址, 属合法 trap 现场.
            # 直接从 satp CSR 读 MODE 字段, 避免依赖缓存 _mmu_mode.
            satp_mode = (h.satp_val >> 60) & 0xF
            if self._emu.bus.is_ram_addr(trapped_pc):
                if satp_mode != 0:
                    # Sv39 启用 -> 内核/sepc 应为规范高 VA, 残留值
                    return
                if prev_mode == RiscvMode.U:
                    # Bare 翻译下 U-mode trap: RAM 地址合法, 不跳过
                    pass
                else:
                    # Bare 翻译下非 U 模式 (SPP=S): sepc 落在 RAM
                    # 但 S 模式代码预期在高区 ->残留值
                    return
            s_trapped_pc = 0
            s_prev_mode = RiscvMode.U
        else:
            return

        if trapped_pc == 0 or prev_mode.value >= mode_val or tvec == 0:
            return

        # 若 trapped_pc 已与现有序号帧的 PC 重合 (含 ±4),
        # 说明该 trap 上下文已被 FP 链覆盖, 不重复添加.
        for f in frames:
            if abs(f.pc - trapped_pc) <= 4:
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

        if offsets is not None:
            self._add_prev_mode_frame(
                frames, trapped_pc, prev_mode, tvec, visited, offsets
            )
        elif prev_mode == RiscvMode.U and not _nested:
            # S->U 边界: 指令解析失败时, 尝试常见 trap 帧布局以恢复
            # U-mode 的 FP/SP/RA 并继续 U-mode FP 链回溯.
            self._add_prev_mode_frame_fallback(
                frames, trapped_pc, prev_mode, tvec, visited,
                boundary_mode, boundary_note,
            )
        else:
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
                    frames, s_trapped_pc, s_prev_mode, h.stvec_val, visited, offsets
                )

    def _add_prev_mode_frame(
        self, frames: list[StackFrame], trapped_pc: int,
        prev_mode: RiscvMode, tvec: int, visited: set[int],
        offsets: tuple[int, int] | None = None,
    ) -> None:
        """为指定 trap 上下文添加边界帧, 并尝试继续回溯低特权级 FP 链.

        若 *offsets* 为 None, 从 *tvec* 指令流解析 RA/FP 保存偏移.
        """
        if trapped_pc == 0 or tvec == 0:
            return
        prev_name = prev_mode.name
        boundary_mode, boundary_note = self._resolve_boundary_frame(
            trapped_pc, prev_name
        )
        if offsets is None:
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

        call_site = trapped_pc
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
            call_site = self._call_site_pc(next_ra)
            frames.append(StackFrame(
                idx=len(frames), fp=next_fp, sp=current_fp,
                ra=next_ra, pc=call_site, mode=prev_name,
            ))
            current_fp = next_fp

    def _add_prev_mode_frame_fallback(
        self, frames: list[StackFrame], trapped_pc: int,
        prev_mode: RiscvMode, tvec: int, visited: set[int],
        boundary_mode: str, boundary_note: str,
    ) -> None:
        """指令解析失败时, 尝试常见 trap 帧布局以恢复低特权级上下文.

        依次尝试 _FALLBACK_TRAP_OFFSETS 中的 (ra_off, fp_off) 对,
        首个成功恢复 saved_fp 的布局生效并继续 FP 链回溯.
        全部失败则退回到 "仅 trap PC" 帧.
        """
        if trapped_pc == 0 or tvec == 0:
            return
        for ra_off, fp_off in _FALLBACK_TRAP_OFFSETS:
            offsets: tuple[int, int] = (ra_off, fp_off)
            # 临时试构建 — 捕获是否成功找到 saved_fp
            trial: list[StackFrame] = []
            self._add_prev_mode_frame(
                trial, trapped_pc, prev_mode, tvec, visited, offsets,
            )
            if not trial or trial[-1].fp == 0:
                continue
            # 成功恢复: 将结果帧合并到正式帧列表并返回
            frames.extend(trial)
            # 将 trial 中所有有效 FP 加入 visited 以避免 FP 链循环
            for f in trial:
                if f.fp == 0:
                    continue
                visited.add(f.fp)
            return
        # 全部 fallback 失败: 添加仅 PC 帧
        frames.append(StackFrame(
            idx=len(frames), fp=0, sp=0, ra=0,
            pc=trapped_pc,
            note=(boundary_note or "trap 入口寄存器保存布局无法解析 (仅 PC)"),
            mode=boundary_mode,
        ))

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
        # 使用 VA->PA 翻译读取栈内存 (Sv39 等 MMU 模式下 VA 非物理地址)
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
            + self._emu._fmt_hexdump(cur.sp, stack_data)
        )
