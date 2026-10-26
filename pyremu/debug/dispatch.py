#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""DispatchOps — REPL 命令分发、帮助信息、hart 切换."""

import signal
from typing import Any, Callable

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from pyremu.debug.attrs import SharedAttrs
from pyremu.debug.exception import UsageError
from pyremu.debug.utils import fmt_size
from pyremu.utils.tick import fmt_now_ms

CSR_USAGE = (
    "用法: csr <name>  或  csr <name1>, <name2>, ...\n"
    "  例: csr mstatus\n"
    "  例: csr mscratch, mepc, pmpcfg0, pmpaddr0\n"
    "  例: csr list"
)


class DispatchOps(SharedAttrs):
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
        "pmp": "cmd_pmp",
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
        self._console.print(f"[dim]切换到 Hart {hart_id}[/]")

    # ----------------------------------------------------------
    #  主分发
    # ----------------------------------------------------------
    @staticmethod
    def _dispatch_aux(cmds: list[str], threshold: int, default_ans: Any = None) -> Any:
        if len(cmds) > threshold:
            return cmds[threshold]
        return default_ans

    # 命令名到处理函数的映射. 调用时传入调试器对象与切分后的参数,
    # 返回的布尔值不使用; 参数个数不足时 _require_args 报告用法, 命令不执行.
    _DISPATCH_HANDLERS: dict[str, Callable[[DispatchOps, list[str]], object]] = {
        # 执行控制
        "b": lambda self, parts: self._dispatch_bp(parts[1:]),
        "bp": lambda self, parts: self._dispatch_bp(parts[1:]),
        "s": lambda self, parts: self.cmd_step(int(self._dispatch_aux(parts, 1, 1))),
        "step": lambda self, parts: self.cmd_step(int(self._dispatch_aux(parts, 1, 1))),
        "c": lambda self, parts: self.cmd_continue(),
        "continue": lambda self, parts: self.cmd_continue(),
        "watch": lambda self, parts: self.cmd_watch(
            self._dispatch_aux(parts, 1, ""), self._dispatch_aux(parts, 2, "8")
        ),
        "r": lambda self, parts: self.cmd_run(int(self._dispatch_aux(parts, 1, 1))),
        "run": lambda self, parts: self.cmd_run(int(self._dispatch_aux(parts, 1, 1))),
        "undo": lambda self, parts: self.rollback(),
        "rollback": lambda self, parts: self.rollback(),
        "restart": lambda self, parts: self.cmd_restart(),
        # 寄存器
        "regs": lambda self, parts: self.cmd_regs(),
        "gpr": lambda self, parts: self.cmd_regs(),
        "reg": lambda self, parts: (
            self._require_args(parts, 2, "用法: reg <name>  例: reg x10, reg a0")
            and self.cmd_reg(" ".join(parts[1:]))
        ),
        "set": lambda self, parts: (
            self._require_args(parts, 3, "用法: set <name> <value>  例: set sp 0x8000")
            and self.cmd_set(parts[1], parts[2])
        ),
        "w": lambda self, parts: (
            self._require_args(parts, 3, "用法: set <name> <value>  例: set sp 0x8000")
            and self.cmd_set(parts[1], parts[2])
        ),
        "csr": lambda self, parts: (
            self._require_args(parts, 2, CSR_USAGE) and self.cmd_csr(" ".join(parts[1:]))
        ),
        "csrw": lambda self, parts: (
            self._require_args(
                parts, 3, "用法: csrw <name> <value>  例: csrw mtvec 0x80000001"
            )
            and self.cmd_csrw(parts[1], parts[2])
        ),
        # 状态与内存
        "pc": lambda self, parts: self.cmd_pc(self._dispatch_aux(parts, 1)),
        "mode": lambda self, parts: self.cmd_mode(),
        "mstatus": lambda self, parts: self.cmd_mstatus(),
        "show": lambda self, parts: self._dispatch_show(self._dispatch_aux(parts, 1, "")),
        "tlb": lambda self, parts: self.cmd_tlb(self._dispatch_aux(parts, 1)),
        "tlbflush": lambda self, parts: self.cmd_tlbflush(self._dispatch_aux(parts, 1)),
        "cache": lambda self, parts: self.cmd_cache(self._dispatch_aux(parts, 1)),
        "mem": lambda self, parts: (
            self._require_args(parts, 2, "用法: mem <addr> [size]  例: mem 0x80000000 64")
            and self.cmd_mem(parts[1], self._dispatch_aux(parts, 2, "64"))
        ),
        "vmem": lambda self, parts: (
            self._require_args(parts, 2, "用法: vmem <vaddr> [size]  例: vmem sepc 256")
            and self.cmd_vmem(parts[1], self._dispatch_aux(parts, 2, "64"))
        ),
        "disasm": lambda self, parts: (
            self._require_args(
                parts, 2, "用法: disasm <addr> [count]  例: disasm 0x80000000 32"
            )
            and self.cmd_disasm(parts[1], self._dispatch_aux(parts, 2, "16"))
        ),
        "vdisasm": lambda self, parts: (
            self._require_args(parts, 2, "用法: vdisasm <vaddr> [count]  例: vdisasm sepc 16")
            and self.cmd_vdisasm(parts[1], self._dispatch_aux(parts, 2, "16"))
        ),
        "pt": lambda self, parts: self.cmd_pt(self._dispatch_aux(parts, 1)),
        "status": lambda self, parts: self.cmd_status(self._dispatch_aux(parts, 1)),
        "info": lambda self, parts: self.cmd_status(self._dispatch_aux(parts, 1)),
        "stack": lambda self, parts: self.cmd_frame(self._dispatch_aux(parts, 1)),
        "bt": lambda self, parts: self.cmd_frame(self._dispatch_aux(parts, 1)),
        "frame": lambda self, parts: self.cmd_frame(self._dispatch_aux(parts, 1)),
        "f": lambda self, parts: self.cmd_frame(self._dispatch_aux(parts, 1)),
        # 符号与配置
        "symbols": lambda self, parts: self.cmd_symbols(self._dispatch_aux(parts, 1, "")),
        "sym": lambda self, parts: self.cmd_sym(self._command_rest),
        # 映像与磁盘
        "bins-info": lambda self, parts: self.cmd_bins_info(self._command_rest),
        "bins": lambda self, parts: self.cmd_bins_info(self._command_rest),
        "disk-info": lambda self, parts: self.cmd_disk_info(self._command_rest),
        "disk": lambda self, parts: self.cmd_disk_info(self._command_rest),
        "hart": lambda self, parts: (
            self._require_args(parts, 2, f"用法: hart <id>  当前: {self._hart_id}")
            and self.cmd_hart(int(parts[1]))
        ),
        # 帮助
        "h": lambda self, parts: self._print_help(),
        "help": lambda self, parts: self._print_help(),
        "?": lambda self, parts: self._print_help(),
    }

    def _dispatch(self, parts: list[str], rest: str = "") -> bool:
        """分发 REPL 命令; 返回 False 表示退出.

        *rest* 为命令名之后的原文, 未给出时由 *parts* 拼接得到; 需要保留
        参数内部空白的命令读取 ``_command_rest``.
        """
        cmd = parts[0].lower()
        if cmd in ("q", "quit", "exit"):
            return False
        handler = self._DISPATCH_HANDLERS.get(cmd)
        if handler is None:
            self._err(f"未知命令: {cmd} (输入 'help' 查看帮助)")
            return True
        self._command_rest = rest or " ".join(parts[1:])
        handler(self, parts)
        return True

    def _require_args(self, parts: list[str], count: int, usage: str) -> bool:
        """参数个数达到 count 时返回 True, 否则报告用法并返回 False."""
        if len(parts) >= count:
            return True
        self._warn(usage)
        return False

    # ----------------------------------------------------------
    #  REPL 主循环
    # ----------------------------------------------------------

    def repl(self) -> None:
        """主交互循环."""
        self._enter_repl_mode()
        banner = Panel.fit(
            "\n".join(
                [
                    f"Harts: [bold cyan]{self._emu.num_harts}[/]   "
                    f"Active hart: [bold cyan]{self._hart_id}[/]",
                    f"Prog Cnt: [bold yellow]{self._emu.prog_cnt:#018x}[/]  "
                    f"RAM: [dim]{fmt_size(self._emu.bus.ram_size)}"
                    f" @ 0x{self._emu.bus.ram_base:x}[/]  "
                    f"FDT: [dim]{f'0x{self._fdt_addr:x}' if self._fdt_addr else '—'}[/]",
                    "",
                    "Ctrl+Q [dim]once[/]  -> pause emulation",
                    "Type [bold]help/h/?[/] for commands",
                ]
            ),
            title="RISC-V Interactive Debugger [bold green](rvdb)[/]",
            border_style="green",
        )
        self._console.print(banner)

        while True:
            try:
                raw = self._session.prompt(
                    [
                        ("class:prompt", "\n"),
                        # 每次渲染提示符时现取系统时间, 插在 rvdbg 之前.
                        ("class:clock", f"{fmt_now_ms()} "),
                        ("class:prompt", f"rvdbg[{self._hart_id}] "),
                    ]
                ).strip()
            except KeyboardInterrupt:
                self._console.print()
                continue
            except OSError:
                self._console.print("[dim]终端断开, 退出[/]")
                break
            except EOFError:
                self._console.print("[dim]退出[/]")
                break

            if not raw and self._last_command is None:
                continue
            elif not raw:
                raw = self._last_command
                if raw.startswith("disasm ") and self._disasm_next_addr is not None:
                    parts = raw.split()
                    parts[1] = hex(self._disasm_next_addr)
                    raw = " ".join(parts)
                if raw.startswith("vdisasm ") and self._vdisasm_next_addr is not None:
                    parts = raw.split()
                    parts[1] = hex(self._vdisasm_next_addr)
                    raw = " ".join(parts)
                self._console.print(f"[dim](重复) {raw}[/]")

            parts = raw.split()
            try:
                if not self._dispatch(parts):
                    break
            except UsageError as exc:
                self._err(escape(str(exc)))
            except Exception:
                self._console.print_exception()
            else:
                self._last_command = raw

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._console.print("[dim]调试器已退出[/]")

    # ----------------------------------------------------------
    #  帮助
    # ----------------------------------------------------------
    @staticmethod
    def _section(title: str, rows: list[tuple[str, str]]) -> Table:
        """构造一节帮助文本; 方括号是用法里的字面字符, 不按标记解析."""
        tab = Table(title=title, border_style="green", show_header=False)
        tab.add_column("cmd", style="cyan", no_wrap=True)
        tab.add_column("desc")
        for cmd, desc in rows:
            tab.add_row(escape(cmd), escape(desc))
        return tab

    def _print_help(self) -> None:
        """显示帮助 — 使用 Rich Table 以保证中英文混排对齐."""
        sections: list[Table] = [
            self._section(
                "执行控制",
                [
                    ("s/step [n]", "单步执行 n 条指令 (默认 1)"),
                    ("c/continue", "连续执行 (直到 Ctrl+Q 暂停)"),
                    ("r/run [n]", "执行 n 条指令 (默认 1)"),
                    ("undo/rollback", "回滚上一条指令"),
                    ("restart", "重置 hart/CLINT, 重新加载固件"),
                    ("b/bp <addr>", "在指定地址设置断点"),
                    ("b/bp ecall|ebreak|mret|sret|wfi", "在指定指令类型设置断点"),
                    ("b/bp opcode <hex>", "在指定 opcode 设置断点 (如 0x73)"),
                    (
                        "b/bp if reg/csr <name> [op] <val>",
                        "条件断点 (比较符可缺省, 如 csr mepc 0x... 视为相等)",
                    ),
                    ("b/bp", "列出所有断点"),
                    ("bp delete <n>", "删除编号为 n 的断点"),
                    ("bp clear", "清除全部断点"),
                ],
            ),
            self._section(
                "读寄存器命令",
                [
                    ("regs/gpr", "显示全部 GPR (x0-x31)"),
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
                    ("show pmps", "显示全部 PMP 条目的保护范围与权限"),
                ],
            ),
            self._section(
                "写寄存器命令",
                [
                    ("set/w <name> <val>", "写入 GPR (例: set sp 0x8000)"),
                    ("csrw <name> <val>", "写入 CSR (例: csrw mtvec 0x80000001)"),
                    ("pc <addr>", "设置 PC 并显示反汇编"),
                ],
            ),
            self._section(
                "内存",
                [
                    ("mem <addr> [size]", "hexdump 给定物理地址下的内存 (默认 64 字节)"),
                    ("vmem <addr> [size]", "hexdump 给定虚拟地址下的内存 (默认 64 字节)"),
                    ("disasm <addr> [count]", "对物理地址起的 count 条指令做反汇编 (默认 16)"),
                    (
                        "vdisasm <addr> [count]",
                        "对虚拟地址起的 count 条指令做反汇编 (默认 16)",
                    ),
                ],
            ),
            self._section(
                "符号命令",
                [
                    ("sym chdir <绝对路径>", "记录候选目录; 省略参数时显示当前目录"),
                    ("sym pwd", "显示所有符号表共用的候选目录"),
                    ("sym ls", "列出候选目录中的普通文件"),
                    ("sym load <#n|文件名> [pa:<addr>|va:<addr>]", "加载符号文件并校验内容"),
                    ("sym detail [地址|pc|符号名]", "查看已加载模块或覆盖某地址的模块"),
                    ("sym rm <序号|文件名>", "移除符号模块"),
                    ("sym info", "显示外部符号文件信息"),
                    ("sym/symbols [filt]", "列出固件符号表 (可选过滤)"),
                    ("sym help", "显示符号子命令的用法"),
                ],
            ),
            self._section(
                "映像与磁盘",
                [
                    ("bins-info", "列出命令行传入的映像及其物理地址范围"),
                    ("disk-info [<映像路径>]", "显示磁盘映像的格式摘要与布局"),
                ],
            ),
            self._section(
                "状态",
                [
                    ("status/info [hart]", "全部 hart 概览, 或指定 hart 详情"),
                    ("stack/bt", "栈帧情况"),
                    ("frame/f <N>", "切换到第 N 帧"),
                ],
            ),
            self._section("配置", [("hart <id>", "切换活跃 hart")]),
            self._section(
                "其他",
                [
                    ("help/h/?", "显示本帮助"),
                    ("quit/q/exit", "退出调试器"),
                ],
            ),
        ]

        for tab in sections:
            self._console.print(tab)
