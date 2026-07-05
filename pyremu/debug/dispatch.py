#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""DispatchMixin — REPL 命令分发、帮助信息、hart 切换."""

import signal

from rich.panel import Panel
from rich.table import Table

from pyremu.debug._attrs import SharedMixinAttrs
from pyremu.debug.utils import fmt_size


class DispatchMixin(SharedMixinAttrs):
    """命令分发与配置."""

    _SHOW_HANDLERS: dict[str, str] = {
        "mstatus": "cmd_mstatus",
        "satp": "cmd_satp",
        "mcause": "cmd_mcause",
        "scause": "cmd_scause",
        "mtvec": "cmd_mtvec",
        "stvec": "cmd_stvec",
        "mip": "cmd_mip",
        "mie": "cmd_mie",
        "sip": "cmd_sip",
        "sie": "cmd_sie",
        "medeleg": "cmd_medeleg",
        "mideleg": "cmd_mideleg",
        "pmps": "cmd_pmp",
        "tlb": "cmd_tlb",
    }

    def _dispatch_show(self, sub: str) -> None:
        if not sub:
            avail = ", ".join(sorted(self._SHOW_HANDLERS.keys()))
            self._console.print(f"[dim]show 可用子命令: {avail}[/]")
            return
        handler_name = self._SHOW_HANDLERS.get(sub)
        if handler_name is None:
            avail = ", ".join(sorted(self._SHOW_HANDLERS.keys()))
            self._warn(f"show: 未知子命令 '{sub}'. 可用: {avail}")
            return
        getattr(self, handler_name)()

    def cmd_hart(self, hart_id: int) -> None:
        """切换活跃 hart."""
        if not (0 <= hart_id < self._emu.num_harts):
            self._err(f"Hart ID 超出范围 [0, {self._emu.num_harts - 1}]")
            return
        self._hart_id = hart_id
        self._snapshot = None
        self._mem_changes = []
        if self._emu.uart is not None:
            self._emu.uart.flush_all()
        self._console.print(f"[dim]切换到 Hart {hart_id}[/]")

    # ----------------------------------------------------------
    #  主分发
    # ----------------------------------------------------------

    def _dispatch(self, parts: list[str]) -> bool:
        """分发 REPL 命令; 返回 False 表示退出."""
        cmd = parts[0].lower()

        if cmd in ("q", "quit", "exit"):
            return False

        # 执行控制
        if cmd in ("b", "bp"):
            self._dispatch_bp(parts[1:])
            return True
        if cmd in ("s", "step"):
            count = int(parts[1]) if len(parts) > 1 else 1
            self.cmd_step(count)
            return True
        if cmd in ("c", "continue"):
            self.cmd_continue()
            return True
        if cmd == "watch":
            addr = parts[1] if len(parts) > 1 else ""
            size = parts[2] if len(parts) > 2 else "8"
            self.cmd_watch(addr, size)
            return True
        if cmd in ("r", "run"):
            n = int(parts[1]) if len(parts) > 1 else 1
            self.cmd_run(n)
            return True
        if cmd in ("undo", "rollback"):
            self.rollback()
            return True
        if cmd == "restart":
            self.cmd_restart()
            return True

        # 寄存器
        if cmd in ("regs", "gpr"):
            self.cmd_regs()
            return True
        if cmd == "reg":
            if len(parts) < 2:
                self._warn("用法: reg <name>  例: reg x10, reg a0")
            else:
                self.cmd_reg(" ".join(parts[1:]))
            return True
        if cmd in ("set", "w"):
            if len(parts) < 3:
                self._warn("用法: set <name> <value>  例: set sp 0x8000")
            else:
                self.cmd_set(parts[1], parts[2])
            return True

        # CSR
        if cmd == "csr":
            if len(parts) >= 2:
                self.cmd_csr(" ".join(parts[1:]))
                return True
            self._warn(
                "用法: csr <name>  或  csr <name1>, <name2>, ...\n"
                "  例: csr mstatus\n"
                "  例: csr mscratch, mepc, pmpcfg0, pmpaddr0\n"
                "  例: csr list"
            )
            return True
        if cmd == "csrw":
            if len(parts) < 3:
                self._warn(
                    "用法: csrw <name> <value>  例: csrw mtvec 0x80000001"
                )
            else:
                self.cmd_csrw(parts[1], parts[2])
            return True

        # 状态/内存
        if cmd == "pc":
            self.cmd_pc(parts[1] if len(parts) > 1 else None)
            return True
        if cmd == "mode":
            self.cmd_mode()
            return True
        if cmd == "mstatus":
            self.cmd_mstatus()
            return True
        if cmd == "show":
            self._dispatch_show(parts[1] if len(parts) > 1 else "")
            return True
        if cmd == "tlb":
            self.cmd_tlb(parts[1] if len(parts) > 1 else None)
            return True
        if cmd == "tlbflush":
            self.cmd_tlbflush(parts[1] if len(parts) > 1 else None)
            return True
        if cmd == "cache":
            self.cmd_cache(parts[1] if len(parts) > 1 else None)
            return True
        if cmd == "mem":
            if len(parts) < 2:
                self._warn("用法: mem <addr> [size]  例: mem 0x80000000 64")
            else:
                self.cmd_mem(parts[1], parts[2] if len(parts) > 2 else "64")
            return True
        if cmd == "vmem":
            if len(parts) < 2:
                self._warn("用法: vmem <vaddr> [size]  例: vmem sepc 256")
            else:
                self.cmd_vmem(parts[1], parts[2] if len(parts) > 2 else "64")
            return True
        if cmd == "disasm":
            if len(parts) < 2:
                self._warn(
                    "用法: disasm <addr> [count]  例: disasm 0x80000000 32"
                )
            else:
                self.cmd_disasm(
                    parts[1], parts[2] if len(parts) > 2 else "16"
                )
            return True
        if cmd == "vdisasm":
            if len(parts) < 2:
                self._warn(
                    "用法: vdisasm <vaddr> [count]  例: vdisasm sepc 16"
                )
            else:
                self.cmd_vdisasm(
                    parts[1], parts[2] if len(parts) > 2 else "16"
                )
            return True
        if cmd == "pt":
            self.cmd_pt(parts[1] if len(parts) > 1 else None)
            return True
        if cmd in ("status", "info"):
            self.cmd_status(parts[1] if len(parts) > 1 else None)
            return True
        if cmd in ("stack", "bt", "frame", "f"):
            self.cmd_frame(parts[1] if len(parts) > 1 else None)
            return True

        # 符号
        if cmd in ("symbols",):
            self.cmd_symbols(parts[1] if len(parts) > 1 else "")
        elif cmd == "sym":
            self.cmd_sym(parts[1] if len(parts) > 1 else "")
            return True

        # 配置
        if cmd == "hart":
            if len(parts) < 2:
                self._warn(f"用法: hart <id>  当前: {self._hart_id}")
            else:
                self.cmd_hart(int(parts[1]))
            return True

        # 帮助
        if cmd in ("h", "help", "?"):
            self._print_help()
            return True

        self._err(f"未知命令: {cmd} (输入 'help' 查看帮助)")
        return True

    # ----------------------------------------------------------
    #  REPL 主循环
    # ----------------------------------------------------------

    def repl(self) -> None:
        """主交互循环."""
        self._enter_repl_mode()
        banner = Panel.fit(
            "\n".join([
                f"Harts: [bold cyan]{self._emu.num_harts}[/]   "
                f"Active hart: [bold cyan]{self._hart_id}[/]",
                f"Prog Cnt: [bold yellow]{self._emu.prog_cnt:#018x}[/]  "
                f"RAM: [dim]{fmt_size(self._emu.bus._ram_size)}"
                f" @ 0x{self._emu.bus.ram_base:x}[/]  "
                f"FDT: [dim]{f'0x{self._fdt_addr:x}' if self._fdt_addr else '—'}[/]",
                "",
                "Ctrl+C [dim]once[/]  -> pause emulation",
                "Type [bold]help/h/?[/] for commands",
            ]),
            title="RISC-V Interactive Debugger [bold green](rvdb)[/]",
            border_style="green",
        )
        self._console.print(banner)

        while True:
            try:
                raw = self._session.prompt(
                    [("class:prompt", f"\nrvdbg[{self._hart_id}] ")]
                ).strip()
            except KeyboardInterrupt:
                self._console.print()
                continue
            except EOFError:
                self._console.print("[dim]退出[/]")
                break

            if not raw and self._last_command is None:
                continue
            elif not raw:
                assert self._last_command is not None
                raw = self._last_command
                if (
                    raw.startswith("disasm ")
                    and self._disasm_next_addr is not None
                ):
                    parts = raw.split()
                    parts[1] = hex(self._disasm_next_addr)
                    raw = " ".join(parts)
                if (
                    raw.startswith("vdisasm ")
                    and self._vdisasm_next_addr is not None
                ):
                    parts = raw.split()
                    parts[1] = hex(self._vdisasm_next_addr)
                    raw = " ".join(parts)
                self._console.print(f"[dim](重复) {raw}[/]")

            parts = raw.split()
            try:
                if not self._dispatch(parts):
                    break
            except Exception:
                self._console.print_exception()
            else:
                self._last_command = raw

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._console.print("[dim]调试器已退出[/]")

    # ----------------------------------------------------------
    #  帮助
    # ----------------------------------------------------------

    def _print_help(self) -> None:
        """显示帮助 — 使用 Rich Table 以保证中英文混排对齐."""

        def _section(title: str, rows: list[tuple[str, str]]) -> Table:
            tab = Table(title=title, border_style="green", show_header=False)
            tab.add_column("cmd", style="cyan", no_wrap=True)
            tab.add_column("desc")
            for cmd, desc in rows:
                tab.add_row(cmd, desc)
            return tab

        sections: list[Table] = [
            _section("执行控制", [
                ("s/step [n]", "单步执行 n 条指令 (默认 1)"),
                ("c/continue", "连续执行 (直到 Ctrl+C 暂停)"),
                ("r/run [n]", "执行 n 条指令 (默认 1)"),
                ("undo/rollback", "回滚上一条指令"),
                ("restart", "重置 hart/CLINT, 重新加载固件"),
                ("b/bp <addr>", "在指定地址设置断点"),
                ("b/bp ecall|ebreak|mret|sret|wfi",
                 "在指定指令类型设置断点"),
                ("b/bp opcode <hex>",
                 "在指定 opcode 设置断点 (如 0x73)"),
                ("b/bp", "列出所有断点"),
                ("bp delete <n>", "删除编号为 n 的断点"),
                ("bp clear", "清除全部断点"),
            ]),
            _section("读寄存器命令", [
                ("regs/gpr", "显示全部 GPR (x0–x31)"),
                ("reg <name>", "显示指定 GPR (例: reg a0, reg x10)"),
                ("csr <name>", "显示指定 CSR (例: csr mstatus)"),
                ("csr list", "列出所有可用 CSR 名称"),
                ("pc", "显示当前 PC 及反汇编"),
                ("mode", "显示当前特权级"),
                ("mstatus", "显示 mstatus 各字段分解"),
                ("tlb [vpn]", "显示 ITLB / DTLB, 或查找指定 VPN"),
                ("cache [set] [way]", "显示 L2 缓存状态及数据"),
                ("satp", "显示 satp 解码 (MODE/ASID/PPN)"),
                ("pt [va]", "Sv39 页表遍历: 逐级显示 PTE 与权限"),
                ("show-pmp", "显示全部 PMP 条目的保护范围与权限"),
            ]),
            _section("写寄存器命令", [
                ("set/w <name> <val>", "写入 GPR (例: set sp 0x8000)"),
                ("csrw <name> <val>",
                 "写入 CSR (例: csrw mtvec 0x80000001)"),
                ("pc <addr>", "设置 PC 并显示反汇编"),
            ]),
            _section("内存/符号", [
                ("mem <addr> [size]",
                 "hexdump 给定物理地址下的内存 (默认 64 字节)"),
                ("vmem <addr> [size]",
                 "hexdump 给定虚拟地址下的内存 (默认 64 字节)"),
                ("disasm <addr> [count]",
                 "对物理地址起的 count 条指令做反汇编 (默认 16)"),
                ("vdisasm <addr> [count]",
                 "对虚拟地址起的 count 条指令做反汇编 (默认 16)"),
                ("sym/symbols [filt]", "列出符号表 (可选过滤)"),
            ]),
            _section("状态", [
                ("status/info [hart]", "全部 hart 概览, 或指定 hart 详情"),
                ("stack/bt", "栈帧情况"),
                ("frame/f <N>", "切换到第 N 帧"),
            ]),
            _section("配置", [("hart <id>", "切换活跃 hart")]),
            _section("其他", [
                ("help/h/?", "显示本帮助"),
                ("quit/q/exit", "退出调试器"),
            ]),
        ]

        for tab in sections:
            self._console.print(tab)
