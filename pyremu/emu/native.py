#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""NativeOps — 每轮加速执行前后同步 Python 侧与 Rust 侧的状态, 并处理引擎退回的原因.

hart、PMP、CLINT、PLIC、UART 与 virtio-blk 的状态在每轮加速执行前后于两侧之间
复制.
"""

import ctypes

from pyremu._native import (
    ClintInfo,
    DevInfo,
    PlicInfo,
    PmpInfo,
    run_parallel,
    RunConfig,
    UartInfo,
    VirtIOInfo,
)
from pyremu.configs_aux import cfg_int
from pyremu.core.decoder import Hart
from pyremu.core.hart import (
    EXIT_BREAKPOINT,
    EXIT_EBREAK,
    EXIT_ECALL,
    EXIT_MMIO,
    EXIT_NOT_MAINRAM,
    EXIT_TIMEOUT,
    EXIT_TRAP,
    InstrToBeExec,
    marshal_hart,
    unmarshal_hart,
)
from pyremu.core.mem_check_aux import check_instruction_fetch, MemoryAccessError
from pyremu.core.trap_def import TrapType
from pyremu.core.trap_handler import check_pending_interrupts, deliver_trap, try_wfi_wakeup
from pyremu.emu.run import RunStopReason
from pyremu.utils.mask import mask64


class NativeOps:
    """每轮加速执行前后的状态同步、退出处理与结果写回步骤."""

    @staticmethod
    def _handle_native_sys_exit(hart: Hart, instr: int) -> int:
        """执行加速执行引擎退回 Python 侧的一条指令.

        返回 1 表示该指令计入指令数, 返回 0 表示不计入: instr 为 0, 或 hart 因
        WFI 被唤醒.
        """
        if instr == 0:
            return 0
        saved_pc = hart.pc
        try:
            advance = hart.exec_instr(instr)
        except NotImplementedError, MemoryAccessError:
            deliver_trap(hart, TrapType.IllInstr, tval=instr, is_interrupt=False)
            advance = 0
        if advance != 0 and hart.pc == saved_pc:
            hart.pc = mask64(hart.pc + advance)
        if not hart._halted:
            check_pending_interrupts(hart)
        return 0 if hart._wfi_woken else 1

    def _native_exec_outside_ram_fetch(self, hart: Hart) -> int:
        """取指的目标不属于主存时, 由 Python 取指并执行一条指令.

        加速执行引擎只从主存读取指令字节, 遇到设备窗口与空洞时以
        EXIT_NOT_MAINRAM 退出本轮. 设备窗口的指令字节经总线读取, 读取不到时由
        取指校验投递访问故障. 返回执行的指令数, 取值为 0 或 1.
        """
        if self._bp_addrs and hart.pc in self._bp_addrs:
            self._run_stop_reason = RunStopReason.BREAKPOINT
            return 0
        pc = hart.pc
        ok, fetch_pa = check_instruction_fetch(hart, pc)
        if not ok:
            return 0
        if (pc & 0xFFF) >= 0xFFE:
            ok2, fetch_pa2 = check_instruction_fetch(hart, mask64(pc + 2))
            if not ok2:
                return 0
            instr_bytes = self.bus.read(fetch_pa, 2) + self.bus.read(fetch_pa2, 2)
        else:
            instr_bytes = self.bus.read(fetch_pa, 4)
        instr = int.from_bytes(instr_bytes, "little", signed=False)
        if instr == 0:
            # 全零编码为非法指令, 不能按 0 条指令跳过.
            deliver_trap(hart, TrapType.IllInstr, tval=0, is_interrupt=False)
            return 1
        return self._handle_native_sys_exit(hart, instr)

    def _speedup_for_cmd_step(self, active: list[Hart]) -> int:
        # 将 L2 缓存的脏行写回 bytearray, 使加速执行引擎直接读取 bytearray 中的
        # 指令与数据时, 能看到 Python 侧的全部写入.
        self.bus.flush_l2()

        # 全部 hart 的状态都要复制, 包括已停机的: Rust 侧统计 total_instrs 与
        # active_hart_num 时用得到每个 hart 的状态.
        # 接收守护线程已在后台持续把输入写入 UART FIFO, 不消费 _rx_notify;
        # _rx_notify 由加速执行引擎检测, 检出即退出本轮, 随后在空闲轮询中清零.
        # 进入本轮前先把 PLIC 外部中断的 MEIP 与 SEIP 同步进各 hart 的 mip:
        # 加速执行引擎内部只同步 CLINT 的 MSIP 与 MTIP, 不感知 PLIC, 否则
        # virtio 等外设中断无法进入 hart.
        self._native_sync_plic_mip()
        for i, hart in enumerate(self.harts):
            marshal_hart(hart, self._speedup_hart_states[i])

        dev_info = self._native_marshal_dev()
        pmp_info = self._native_marshal_pmp(active)
        clint_info = self._native_marshal_clint()
        uart_info = self._native_marshal_uart()
        virtio_info = self._native_marshal_virtio()
        plic_info = self._native_marshal_plic()

        # PIE 链接地址区间随本轮回传给加速执行引擎, 引擎按同一换算访问主存.
        pie_ofs_base, pie_ofs_size = self.bus.pie_ofs_pair()

        # 各 hart 由一个线程执行, 统一经 run_parallel 调用.
        virtio_ffi = run_parallel(
            RunConfig(
                states=self._speedup_hart_states,
                num_harts=len(self.harts),
                ram_ptr=self._native_ram_ptr,
                ram_size=self.bus.ram_size,
                ram_base=self.bus.ram_base,
                shadow_base=pie_ofs_base,
                shadow_size=pie_ofs_size,
                result=self._speedup_instr_group,
                pmp=pmp_info,
                clint=clint_info,
                dev=dev_info,
                uart=uart_info,
                virtio=virtio_info,
                plic=plic_info,
                watchdog_timeout_ns=self._watchdog_timeout_ns(),
                bp_addrs=self._bp_addrs if self._bp_addrs else None,
                stop_flag=self._native_stop_flag,
                ext_irq=self._native_ext_irq,
                flush_cnt=self.bus.tlb_flush_cnt,
                flush_cnt_per_hart=self._tlb_flush_cnt_per_hart,
            )
        )

        result = self._speedup_instr_group
        # 外部中断为电平触发: 环形缓冲区与 UART FIFO 均为空时才清零 ext_irq.
        # 否则加速执行期间新到的数据会丢失: 这些数据可能落在排空之后、引擎
        # 检查 ext_irq 之前.
        ring_empty = True
        if self._termio is not None:
            rd = self._termio._rx_rd.value
            wr = self._termio._rx_wr.value
            ring_empty = rd == wr
        fifo_empty = self.uart is not None and len(self.uart._rx_fifo) == 0
        if ring_empty and fifo_empty:
            self._native_ext_irq.pending = 0
        # _rx_notify 的唯一作用是让引擎尽快退出本轮, 退出后 Python 才能把环形
        # 缓冲区的字节写入 UART FIFO. 本轮既已结束, 通知即已兑现, 无条件清零:
        # 若等环形缓冲区与 UART FIFO 皆空才清零, 在 UART FIFO 满或受调试程序
        # 关闭中断的窗口内, 下一轮会在第一条指令之前立即退出, 受调试程序得不到
        # 任何执行机会而活锁, 每轮指令数恒为 0.
        # 尚未写入 UART FIFO 的字节仍留在环形缓冲区, 由每轮 _feed_uart_stdin 与
        # 接收守护线程继续写入, 不丢失; termio 线程写入新字节时会重新置位.
        if self._termio is not None:
            self._termio._rx_notify.value = 0
        for hid in range(len(self.harts)):
            unmarshal_hart(self._speedup_hart_states[hid], self.harts[hid])

        # 本轮引擎内可能执行了使翻译条目失效的动作: 受调试程序执行 SFENCE.VMA
        # 或改写 satp. 计数发生变化时把 Python 侧的两个 TLB 全部作废, 否则沿用
        # 既有条目.
        flush_cnt_after = self.bus.tlb_flush_cnt.value
        if flush_cnt_after != self._tlb_flush_cnt_before:
            for hart in self.harts:
                hart.itlb.flush_all()
                hart.dtlb.flush_all()
        self._tlb_flush_cnt_before = flush_cnt_after

        # 加速执行期间可能直接修改了 bytearray; 使 L2 缓存全部失效,
        # 强制 Python 侧后续读取从 bytearray 重新加载.
        self.bus.invalidate_l2()

        self._native_unmarshal_pmp()
        # 发送环形缓冲区的归档由 _tx_archive 守护线程异步处理,
        # 不在此处同步调用; UART 输入输出与单轮加速执行循环完全解耦.
        self._native_unmarshal_clint(clint_info)
        self._native_unmarshal_plic(plic_info)

        # mtime 已由加速执行引擎在执行期间按指令增量推进, 见
        # advance_clock_source, 此处不再重复推进; _cycle 记录时钟源流逝, 供
        # run() 返回值与 cycle 属性使用, 不写入 mcycle.
        self._advance_mcycle()

        # CLINT._msip 的电平由 _native_unmarshal_clint 复制回来: 受调试程序的
        # 陷态处理程序写 0 清除 MSIP 时, Rust 侧 clint_write_msip_concurrent
        # 同步执行 fetch_and(0xFE) 清零 msip 数组的电平位, unmarshal 据此把
        # CLINT._msip 恢复为单轮加速执行结束时的真实电平, 取 0 表示已确认.
        #
        # 此处不得按本轮开始前的旧电平反向清除 CLINT._msip. 若本轮开始前 MSIP
        # 已挂起, 执行期间旧 IPI 的陷态处理程序写 0 清电平, 随后另一 hart 又
        # 写入新 IPI, 使电平重新置 1, 而目标 hart 的 sync_msip 尚未消费该电平,
        # 即 mip.MSIP 为 0, 则 unmarshal 读到的电平 1 代表一个真实挂起、尚未
        # 确认的 IPI, 必须保留到下一轮投递. 若按旧电平清除, 该新 IPI 被永久
        # 丢弃, 发送方永远自旋在 OpenSBI 的 tlb_sync 等待确认, 造成
        # TLB-shootdown 死锁.

        self._native_unmarshal_virtio(virtio_ffi)

        all_exec_cnt = self._native_handle_exit(result, result.total_instrs)
        # 重放至多执行一条指令, 记录随即失效, 后续命令单步不受影响
        if self.virtio_blk is not None:
            self.virtio_blk.clear_notify_served()
        return self._native_finalize(all_exec_cnt)

    # ---- _speedup_for_cmd_step 的组成: 进入前的状态复制、退出后的结果同步与收尾 ----

    def _native_sync_plic_mip(self) -> None:
        """把各 hart 的 PLIC 挂起外部中断合并进其 mip 的 MEIP 位与 SEIP 位.

        加速执行引擎内部只同步 CLINT 的 MSIP 与 MTIP, 不感知 PLIC. 设备的 MMIO
        访问已退出到 Python 侧处理, 故每次进入单轮加速执行前在此从 PLIC 刷新
        MEIP 与 SEIP, 否则 virtio 完成中断无法进入 hart. 这两位完全由 PLIC
        驱动, 同步时直接替换, 其余软件置位保持不变.
        """
        ext_mask = (1 << 9) | (1 << 11)
        for hart in self.harts:
            if hart._imsic is None and hart._plic is None:
                continue
            ext_mip = 0
            if hart._imsic is not None:
                ext_mip = hart._imsic.get_pending_mip(hart.id)
            if ext_mip == 0 and hart._plic is not None:
                ext_mip = hart._plic.get_pending_mip(hart.id)
            hart.mip_val = (hart.mip_val & ~ext_mask) | ext_mip

    def _native_marshal_plic(self) -> PlicInfo | None:
        """把 Python 的 PLIC 状态写入跨轮保留的 ctypes 数组, 返回 PlicInfo 交给 Rust.

        pending 与 level 由 PLIC 对象持有, 即 _ffi_pending 与 _ffi_level 两个
        数组, set_irq 改变中断线时直接写入这两个数组, 故此处无需复制.
        优先级、使能位、阈值与已认领状态由本函数写入, 这几个数组跨单轮加速执行
        保留, 尺寸不变时不重新分配. 设备中断投递目标经 plic.set_irq 写入的正是
        同一批数组, 单轮内新到达的设备中断因此对 Rust 直接可见. 仅在
        num_sources 或 num_contexts 变化时重新分配.
        """
        plic = self.plic
        if plic is None:
            return None
        ns = plic._num_sources
        nc = plic._num_contexts
        nw = (ns + 31) // 32
        if len(self._plic_priority_arr) != ns + 1 or len(self._plic_enable_arr) != nc * nw:
            self._plic_priority_arr = (ctypes.c_uint8 * (ns + 1))()
            self._plic_enable_arr = (ctypes.c_uint32 * (nc * nw))()
            self._plic_threshold_arr = (ctypes.c_uint8 * nc)()
            self._plic_claimed_arr = (ctypes.c_uint32 * nc)()
        # 把 Python 的 PLIC 状态写入数组: 中断源 0 为保留项, 保持为 0.
        # pending 与 level 由 set_irq 直接写入, 此处无需复制.
        for i in range(1, ns + 1):
            self._plic_priority_arr[i] = plic._priority[i] & 0x7
        for c in range(nc):
            for w in range(nw):
                word = 0
                base = w * 32
                for b in range(32):
                    src = base + b
                    if 0 < src <= ns and plic._enable[c][src]:
                        word |= 1 << b
                self._plic_enable_arr[c * nw + w] = word
            self._plic_threshold_arr[c] = plic._threshold[c] & 0x7
            self._plic_claimed_arr[c] = plic._claimed[c]
        self._plic_base = plic.base_addr
        self._plic_num_sources = ns
        self._plic_num_contexts = nc
        return PlicInfo(
            base=plic.base_addr,
            num_sources=ns,
            num_contexts=nc,
            priority=self._plic_priority_arr,
            pending=plic._ffi_pending,
            level=plic._ffi_level,
            enable=self._plic_enable_arr,
            threshold=self._plic_threshold_arr,
            claimed=self._plic_claimed_arr,
        )

    def _native_unmarshal_plic(self, plic_info: PlicInfo | None) -> None:
        """把 Rust 改写过的数组同步到 Python 的 PLIC 对象.

        优先级、使能位、阈值与已认领状态在单轮加速执行内由 Rust 直接修改数组,
        边界处复制回来, 使下一轮 _native_sync_plic_mip 读到的是本轮结束后的仲裁
        状态, 而非本轮开始前的取值.
        """
        if plic_info is None or self.plic is None:
            return
        plic = self.plic
        ns = plic._num_sources
        nc = plic._num_contexts
        nw = (ns + 31) // 32
        for i in range(1, ns + 1):
            plic._priority[i] = self._plic_priority_arr[i] & 0x7
            plic._pending[i] = bool(plic._ffi_pending[i])
            plic._level[i] = bool(plic._ffi_level[i])
        for c in range(nc):
            for w in range(nw):
                word = self._plic_enable_arr[c * nw + w]
                base = w * 32
                for b in range(32):
                    src = base + b
                    if 0 < src <= ns:
                        plic._enable[c][src] = bool(word & (1 << b))
            plic._threshold[c] = self._plic_threshold_arr[c] & 0x7
            plic._claimed[c] = self._plic_claimed_arr[c]

    def _native_marshal_dev(self) -> DevInfo:
        """把设备 MMIO 地址范围写入 DevInfo, 供 Rust 判定 MMIO 直通."""
        devices = self.bus.devices
        dev_bases = (ctypes.c_uint64 * 0)()
        dev_ends = (ctypes.c_uint64 * 0)()
        if devices:
            dev_bases = (ctypes.c_uint64 * len(devices))()
            dev_ends = (ctypes.c_uint64 * len(devices))()
            for j, (base, dev) in enumerate(sorted(devices.items())):
                dev_bases[j] = base
                dev_ends[j] = base + dev.size
        return DevInfo(bases=dev_bases, ends=dev_ends)

    def _native_marshal_pmp(self, active: list[Hart]) -> PmpInfo:
        """构建每个 hart 各占一个切片的 PMP 缓冲区, 连续存放 num_harts * 64 项.

        每个 hart 有独立的 PMP, hart h 的条目位于 [h*64, h*64+64); 加速执行
        引擎按 hart_id 偏移读取各自的切片.

        多个 hart 不得共用同一 PMP 缓冲区: SMP 启动时各 hart 在各自的
        run_parallel 线程中并发重写自己的 PMP, 共用缓冲区会造成数据竞争, 一个
        hart 取指时可能读到另一个 hart 刚写下的条目, 其地址不在这些条目覆盖的
        范围内, 因而触发访问故障, cause 为 1, 多出现在 handle_exception 入口.
        参见 CHANGELOG 2026-07-14.
        """
        total = len(self.harts)
        ref_pmp = self.harts[0]._pmp
        self._pmp_num = ref_pmp._num_entries if ref_pmp is not None else 0
        self._pmp_cfg_buf = (ctypes.c_uint8 * 0)()
        self._pmp_addr_buf = (ctypes.c_uint64 * 0)()
        if self._pmp_num > 0:
            self._pmp_cfg_buf = (ctypes.c_uint8 * (64 * total))()
            self._pmp_addr_buf = (ctypes.c_uint64 * (64 * total))()
            for hid, h in enumerate(self.harts):
                hp = h._pmp
                if hp is None or hp._num_entries == 0:
                    continue
                if hp._cache_dirty:
                    hp._rebuild_cache()
                base = hid * 64
                fc = hp._flat_cfg
                fa = hp._flat_addr
                for j in range(64):
                    self._pmp_cfg_buf[base + j] = fc[j]
                    self._pmp_addr_buf[base + j] = fa[j]
        return PmpInfo(
            cfg=self._pmp_cfg_buf,
            addr=self._pmp_addr_buf,
            pmpsplit=active[0].pmpsplit_val if active else 0,
            hart_num=total,
        )

    def _native_unmarshal_pmp(self) -> None:
        """把各 hart 的 PMP 切片复制回其独立的 PMP, 仅在实际改动时同步 CSR.

        每个 hart 的 OpenSBI 把 pmpcfg 与 pmpaddr 写入自己的 64 项切片,
        不复制到其它 hart. sync_from_flat 写入 CSR 的开销较大, 绝大多数单轮
        加速执行中 PMP 无变化, 故仅在切片确实被 Rust 改写时才同步.
        """
        if self._pmp_num <= 0:
            return
        for hid, h in enumerate(self.harts):
            hp = h._pmp
            if hp is None or hp._num_entries == 0:
                continue
            base = hid * 64
            fc = hp._flat_cfg
            fa = hp._flat_addr
            changed = False
            for j in range(64):
                cv = self._pmp_cfg_buf[base + j]
                av = self._pmp_addr_buf[base + j]
                if cv != fc[j] or av != fa[j]:
                    fc[j] = cv
                    fa[j] = av
                    changed = True
            if changed:
                hp._sync_from_flat()

    def _native_marshal_clint(self) -> ClintInfo:
        """构建 ClintInfo 数组, Rust 侧可直接更新其中的 MSIP 与 MTIMECMP.

        数组按全部 hart 而非仅运行中的 hart 分配: 某个 hart 可能写入已停机 hart
        的 MSIP 或 MTIMECMP, 故每个 hart 在数组中各占一项.

        每个 hart 的 MSIP 编码为 ``level | (edge << 1)``, 第 0 位是电平, 第 1 至
        7 位是电平由 0 变 1 的次数. 电平由 0 变 1 时按 128 取模把该次数加一:

            edge = (edge + 1) mod 128

        Rust 的 sync_msip 比较本次与上次的次数来判定 MSIP 是否被改写; 只比较
        电平会漏掉 0 变 1 再变回 0 的写法.
        """
        clint = self.clint
        total = len(self.harts)
        self._clint_mtimecmp_arr = (ctypes.c_uint64 * total)()
        self._clint_msip_arr = (ctypes.c_uint8 * total)()
        if clint is None:
            return ClintInfo(
                mtime=0,
                mtimecmp=self._clint_mtimecmp_arr,
                msip=self._clint_msip_arr,
                base=0,
                timebase_hz=0,
            )

        # mtime 由 Rust 的 advance_clock_source 在加速执行期间按指令增量推进,
        # Python 侧不在此处按流逝的时间补足 mtime.

        for hid in range(total):
            if hid < len(clint._mtimecmp):
                self._clint_mtimecmp_arr[hid] = clint._mtimecmp[hid]
            if hid < len(clint._msip):
                level = clint._msip[hid] & 1
                prev = self._clint_msip_prev[hid]
                if level == 1 and prev == 0:
                    self._clint_msip_edge[hid] = (self._clint_msip_edge[hid] + 1) & 0x7F
                self._clint_msip_prev[hid] = level
                self._clint_msip_arr[hid] = level | (self._clint_msip_edge[hid] << 1)
        return ClintInfo(
            mtime=clint.get_mtime(),
            mtimecmp=self._clint_mtimecmp_arr,
            msip=self._clint_msip_arr,
            base=clint.base_addr,
            timebase_hz=self._cfg.timebase_freq,
        )

    def _native_unmarshal_clint(self, clint_info: ClintInfo) -> None:
        """把 Rust 改写的数组同步回 Python 的 CLINT, 并对齐 Sstc 的 stimecmp.

        内核通过 Sstc 写 stimecmp 设定时器, stimecmp 即权威定时器值; 该值始终
        复制到 mtimecmp, 供 SBI_SET_TIMER 与 WFI 空闲优化一致使用.

        MSIP 从 Rust 返回的编码值中提取电平由 0 变 1 的次数, 供下轮写入数组时
        使用; 仅电平位复制到 CLINT 的 msip.
        """
        clint = self.clint
        if clint is None:
            return
        for hid in range(len(self.harts)):
            if hid < len(clint._mtimecmp):
                clint._mtimecmp[hid] = self._clint_mtimecmp_arr[hid]
            if hid < len(clint._msip):
                raw = self._clint_msip_arr[hid]
                self._clint_msip_edge[hid] = (raw >> 1) & 0x7F
                level = raw & 1
                clint._msip[hid] = level
                # 下轮进入本步时不重复计数
                self._clint_msip_prev[hid] = level
        clint._mtime = clint_info.mtime

    def _native_marshal_uart(self) -> UartInfo:
        """构造 UartInfo, 使 Rust 直接处理 TXDATA 与 IE、IP、TXCTRL 寄存器访问.

        发送环形缓冲区与写索引由 TerminalIO 提供. tx_notify_fd 为兼容性字段,
        Rust 从不写入它: 归档线程按 poll 超时读出该缓冲区, 不依赖该通知.
        no_stdout 保持 0, 由 Rust 即时输出到标准输出, 采用非阻塞写, 对端不可写时
        丢弃该字节, 见 uart.rs 的 try_write_fd, 使 CPU 引擎线程不阻塞在控制台
        输出上.
        """
        if self._termio is None or self.uart is None or self._native_uart_ffi is None:
            return UartInfo(base=0)

        # 此处刷新 termio 持有的字段与 UART 基址; 寄存器字段由下一行的
        # uart._publish_native_regs() 写入.
        ctx = self._native_uart_ffi
        ctx.base = self.uart.base_addr
        ctx.tx_buf = ctypes.addressof(self._termio.tx_buf)
        ctx.tx_cap = len(self._termio.tx_buf)
        ctx.tx_wr = ctypes.addressof(self._termio.tx_wr)
        ctx.tx_notify_fd = self._termio.tx_notify_w
        ctx.no_stdout = 0
        ctx.rx_notify = ctypes.addressof(self._termio.rx_notify)
        self.uart._publish_native_regs()
        return UartInfo(base=self.uart.base_addr, ffi=ctx)

    def _native_flush_uart(self) -> None:
        """把发送环形缓冲区的内容归档到 hart 日志文件.

        Rust 侧经 libc::write 已即时输出, 此处仅消费环形缓冲区防止溢出,
        不触发 _tx_callback 向标准输出重复输出.
        """
        if self.uart is None or self._termio is None:
            return
        self._termio.drain_tx_logs_archive_only()

    # ---- virtio-blk 的状态复制与结果同步 ----

    def _native_marshal_virtio(self) -> VirtIOInfo:
        """构造 VirtIOInfo, 内容为完整的运行时 MMIO 寄存器状态.

        Rust 在单轮加速执行内处理全部 virtio MMIO 读写, 单轮结束后由
        _native_unmarshal_virtio 把 Rust 改动过的字段同步回 Python 的 vblk.
        本函数必须在每轮加速执行前把当前 Python 状态完整复制给 Rust, 否则队列
        描述符、中断状态等动态寄存器会在跨轮时归零, 受调试程序看到的设备状态
        回退为未初始化.
        """
        vblk = self.virtio_blk
        if vblk is None:
            return VirtIOInfo(base=0)
        mmio = vblk.mmio
        queue = mmio.queues[0]
        return VirtIOInfo(
            base=vblk.base_addr,
            capacity=vblk.capacity_sectors,
            queue_num_max=mmio.max_queue_num,
            device_features_sel=mmio.device_features_sel,
            driver_features_sel=mmio.driver_features_sel,
            driver_features=mmio.driver_features,
            queue_sel=mmio.queue_sel,
            queue_num=queue.num,
            queue_ready=queue.ready,
            queue_desc=queue.desc,
            queue_driver=queue.driver,
            queue_device=queue.device,
            status=mmio.status,
            interrupt_status=mmio.interrupt_status,
        )

    def _native_unmarshal_virtio(self, virtio_ffi) -> None:
        """把 Rust 改动过的 virtio 状态同步回 Python 的 VirtIOBlock.

        Rust 在单轮加速执行内处理 MMIO 寄存器读写, 结束后按以下次序同步:
        - 寄存器状态复制回 VirtIOBlock 的 mmio 与队列 0;
        - irq_maybe_lower 置位时调用 lower_irq_if_idle;
        - notify_pending 置位时处理队列, 清空后记录该次通知已处理.
        """
        vblk = self.virtio_blk
        if vblk is None or virtio_ffi is None:
            return

        # 把 Rust 在单轮加速执行内改写的寄存器状态复制回设备对象
        mmio = vblk.mmio
        queue = mmio.queues[0]
        mmio.device_features_sel = virtio_ffi.device_features_sel
        mmio.driver_features_sel = virtio_ffi.driver_features_sel
        mmio.driver_features = virtio_ffi.driver_features
        mmio.queue_sel = virtio_ffi.queue_sel
        queue.num = virtio_ffi.queue_num
        queue.ready = bool(virtio_ffi.queue_ready)
        queue.desc = virtio_ffi.queue_desc
        queue.driver = virtio_ffi.queue_driver
        queue.device = virtio_ffi.queue_device
        mmio.status = virtio_ffi.status
        mmio.interrupt_status = virtio_ffi.interrupt_status

        # InterruptACK 清空所有中断位时拉低 PLIC 中断线.
        # 必须在 notify_pending 之前处理: 若两者在同一轮加速执行内触发,
        # 先降低中断线电平再处理新队列, 避免 lower_irq_if_idle 看到
        # process_queue 刚写入的 interrupt_status 而跳过降电平,
        # 使 _do_complete 时电平仍为高, 中断线反复置高.
        if virtio_ffi.irq_maybe_lower:
            virtio_ffi.irq_maybe_lower = 0
            vblk.lower_irq_if_idle()

        # QueueNotify: Rust 置 notify_pending=1, 由 Python 处理 virtqueue.
        # 分批处理, 每批最多 VIRTIO_PROCESS_BATCH 个描述符, 循环直到全部完成.
        # 每批之间 Python 可响应 Ctrl+Q; _max_batches 防止异常情况下的死循环.
        if not virtio_ffi.notify_pending:
            return

        virtio_ffi.notify_pending = 0
        _batch_guard, _max_batches = 0, 256
        is_cleared = False
        while True:
            if not vblk.process_queue(max_descriptors=cfg_int("VIRTIO_PROCESS_BATCH")):
                is_cleared = True
                break
            _batch_guard += 1
            if _batch_guard >= _max_batches:
                if not self._warned_infinite_virtio:
                    self._warned_infinite_virtio = True
                break
        # 队列已清空时记录本次通知, 使随后重放的该次存储指令不再处理队列.
        # 触及批次上限时不记录, 由重放的这次处理继续清空剩余请求.
        if is_cleared:
            vblk.mark_notify_served()
        virtio_ffi.interrupt_status = mmio.interrupt_status

    def _native_handle_exit(self, result: InstrToBeExec, all_exec_cnt: int) -> int:
        """处理单轮加速执行的退出原因, 取值为 ECALL、MMIO、TRAP 或 BREAKPOINT."""
        total_harts = len(self.harts)
        if result.exit_hart_id >= total_harts:
            return all_exec_cnt
        exit_hart = self.harts[result.exit_hart_id]
        if result.exit_reason in (EXIT_ECALL, EXIT_MMIO):
            all_exec_cnt += self._handle_native_sys_exit(exit_hart, result.exit_instr)
        elif result.exit_reason == EXIT_NOT_MAINRAM:
            all_exec_cnt += self._native_exec_outside_ram_fetch(exit_hart)
        elif result.exit_reason == EXIT_TRAP:
            all_exec_cnt += 1
        elif result.exit_reason == EXIT_BREAKPOINT:
            # 断点命中; PC 已核对 bp_addrs. 记录停止原因供 run() 与调试器区分于
            # 设备暂停事件. 若不就此退出, 重入单轮加速执行会立即再命中而空转.
            self._run_stop_reason = RunStopReason.BREAKPOINT
        elif result.exit_reason == EXIT_EBREAK:
            # 固件停机序列 (semihosting SYS_EXIT), 全部 hart 停止.
            self._run_stop_reason = RunStopReason.EBREAK
        elif result.exit_reason == EXIT_TIMEOUT:
            # 时钟源超时: 看门狗线程在单轮加速执行内推进 mtime 越过截止时间时
            # 置停止标志; 记录原因供 run() 抛出 TimeoutError.
            self._run_stop_reason = RunStopReason.TIMEOUT
        return all_exec_cnt

    def _native_finalize(self, all_exec_cnt: int) -> int:
        """同步 CSR 镜像, 唤醒 WFI 等待, 并推进看门狗.

        mtime 由加速执行引擎按指令增量推进, 见 advance_clock_source, 边界处
        不再重复推进; 此处仅同步 mcycle、minstret 与 time 等 CSR 镜像.
        """
        instr_delta = all_exec_cnt - self._prev_all_exec_cnt
        self._prev_all_exec_cnt = all_exec_cnt

        self.sync_counters(max(0, instr_delta))

        wfi_waiting = 0
        for hart in self.harts:
            if not hart._waiting or hart._halted:
                continue
            try_wfi_wakeup(hart)
            if hart._waiting:
                wfi_waiting += 1

        self._wfi_sleep_if_idle(self.harts, wfi_waiting)
        self._native_sync_plic_mip()
        _recheck = 0
        for hart in self.harts:
            if not hart._waiting or hart._halted:
                continue
            if try_wfi_wakeup(hart):
                _recheck += 1
        if _recheck:
            wfi_waiting = max(0, wfi_waiting - _recheck)
            self._wake_event.clear()

        if instr_delta > 0:
            self.watchdog.kick_all()
        else:
            self.watchdog.tick(self.harts)

        for hart in self.harts:
            if not hart._waiting or hart._halted:
                continue
            try_wfi_wakeup(hart)
            if not hart._waiting:
                wfi_waiting = max(0, wfi_waiting - 1)
        return all_exec_cnt
