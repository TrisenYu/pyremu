#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""CLI 入口 — 解析参数、加载固件并启动交互调试器."""

import argparse
import sys
from pathlib import Path

from loguru import logger

from pyremu.debug import Debugger
from pyremu.debug.utils import _RAM_MUL, fmt_size
from pyremu.emulator import Emulator
from pyremu.env_inject import Preloader
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils.parse_bin import FirmwareImage, parse_firmware

# ============================================================
#  Argument parser
# ============================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器."""
    p = argparse.ArgumentParser(
        description="RISC-V Interactive Debugger (rvdb)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  %(prog)s firmware.bin\n"
        "  %(prog)s firmware.elf --entry-symbol main\n"
        "  %(prog)s app.elf --preload crt0.bin --entry-symbol main\n"
        "  %(prog)s firmware.bin --harts 4 --ram 256M",
    )
    p.add_argument("firmware", help="固件文件路径 (ELF / PE / raw binary)")
    p.add_argument(
        "--base-addr", type=lambda x: int(x, 0), default=0x80000000,
        help="raw binary 的加载基址 (ELF/PE 时忽略, 默认 0x80000000)",
    )
    p.add_argument(
        "--prog-cnt", type=lambda x: int(x, 0), default=None,
        help="程序计数器 — 每个 hart 的开始执行地址",
    )
    p.add_argument(
        "--ram-base", type=lambda x: int(x, 0), default=0x8000_0000,
        help="物理内存基址 (默认 0x80000000)",
    )
    p.add_argument(
        "--fdt", type=lambda x: int(x, 0) if x is not None else None,
        default=-1, nargs="?", const=-1,
        help="设备树加载地址 (默认 RAM 顶端 −64 KiB)",
    )
    p.add_argument(
        "--fdt-file", type=str, default=None, metavar="PATH[:ADDR]",
        help="加载预编译 DTB 文件 (可选 :地址)",
    )
    p.add_argument(
        "--no-fdt", action="store_true", default=False,
        help="禁用设备树",
    )
    p.add_argument("--harts", type=int, default=1, help="hart 数量 (默认 1)")
    p.add_argument(
        "--ram", type=str, default="128M",
        help="RAM 大小 (支持 K/M/G 后缀, 默认 128M)",
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
        choices=[
            "TRACE", "DEBUG", "INFO", "SUCCESS",
            "WARNING", "ERROR", "CRITICAL",
        ],
        help="日志级别 (默认 INFO)",
    )
    p.add_argument(
        "--preload", type=str, default=None,
        help="预加载 shellcode 文件路径",
    )
    p.add_argument(
        "--preload-addr", type=lambda x: int(x, 0), default=None,
        help="预加载 shellcode 的加载地址 (默认 RAM 顶端 −64 KiB)",
    )
    p.add_argument(
        "--entry-symbol", type=str, default=None,
        help="目标程序入口符号名 (如 'main')",
    )
    p.add_argument(
        "--sym", type=str, default=None, metavar="PATH",
        help="调试符号文件路径 (如 Linux vmlinux)",
    )
    p.add_argument(
        "--sym-base", type=lambda x: int(x, 0), default=None,
        metavar="ADDR",
        help="符号文件的加载基址 (默认 ram_base + 0x200000)",
    )
    p.add_argument(
        "--kernel", type=str, default=None, metavar="PATH",
        help="内核 Image (raw binary), 预载到 RAM",
    )
    p.add_argument(
        "--kernel-addr", type=lambda x: int(x, 0), default=None,
        metavar="ADDR",
        help="内核 Image 加载地址 (默认 ram_base + 0x200000)",
    )
    p.add_argument(
        "--bootargs", type=str, default=None, metavar="ARGS",
        help="内核命令行参数, 写入 DTB /chosen/bootargs",
    )
    return p


# ============================================================
#  CLI 工具函数
# ============================================================


def _setup_logger(level: str) -> None:
    """配置 loguru: 彩色输出到 stdout."""
    logger.remove()
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:HH:mm:ss}</green> "
            "[<yellow>{file}:{line}</yellow>|<level>{level}</level>] "
            "<level>{message}</level>"
        ),
        colorize=True,
        level=level,
    )


def _parse_ram_size(ram_str: str) -> int:
    """解析 RAM 大小字符串 (支持 K/M/G 后缀) -> 字节数."""
    s = ram_str.upper()
    if s[-1] in _RAM_MUL:
        return int(s[:-1]) * _RAM_MUL[s[-1]]
    return int(s)


def _load_firmware_image(
    fw_path: str, base_addr: int, ram_base: int,
) -> tuple[FirmwareImage, int]:
    """解析固件并计算 PIE 搬迁偏移.

    Returns:
        (image, load_offset) — 偏移为 0 固件时 *load_offset* = ram_base.
    """
    if not Path(fw_path).exists():
        logger.error(f"文件不存在: {fw_path}")
        sys.exit(1)

    logger.info(f"正在解析固件: {fw_path}")
    image = parse_firmware(fw_path, base_addr=base_addr)
    if image is None:
        logger.error(f"无法解析固件: {fw_path}")
        sys.exit(1)

    logger.info(
        f"格式={image.format}, 入口=0x{image.entry_point:x},"
        f" 段数={len(image.segments)}"
    )
    max_name = max((len(seg.name) for seg in image.segments), default=6)
    max_name = max(max_name, 6)
    for seg in image.segments:
        logger.info(
            f"  {seg.name:<{max_name}s}  vaddr=0x{seg.vaddr:016x}"
            f"  size={fmt_size(len(seg.data)):>10s}"
            f"  memsz={fmt_size(seg.memsz):>10s}"
        )

    load_offset = 0
    if image.format == "elf":
        min_vaddr = min(seg.vaddr for seg in image.segments)
        if min_vaddr >= ram_base:
            return image, load_offset
        load_offset = ram_base
        logger.info(
            f"PIE 固件段偏移 +0x{load_offset:x}"
            f" (min vaddr=0x{min_vaddr:x} < ram_base=0x{ram_base:x})"
        )
    return image, load_offset


def _configure_fdt(
    emu: Emulator,
    ns: argparse.Namespace,
    ram_base: int,
    ram_size: int,
) -> int | None:
    """根据 CLI 参数配置设备树, 返回 DTB 加载地址 (或 None)."""
    if ns.no_fdt:
        return None
    if ns.fdt_file is not None:
        parts = ns.fdt_file.rsplit(":", 1)
        file_path = parts[0]
        custom_addr = int(parts[1], 0) if len(parts) > 1 else None
        addr: int = (
            custom_addr
            if custom_addr is not None
            else (ram_base + ram_size - 0x10000)
        )
        emu.load_dtb_file(file_path, addr=addr)
        return addr
    if ns.fdt is not None:
        addr = (
            ns.fdt
            if ns.fdt != -1
            else (ram_base + ram_size - 0x10000)
        )
        emu.load_dtb(addr)
        return addr
    return None


def _setup_preload(
    emu: Emulator,
    ns: argparse.Namespace,
    fdt_addr: int | None,
    ram_base: int,
) -> int | None:
    """预加载 shellcode, 返回 preload 入口地址 (或 None)."""
    if ns.preload is None:
        return None
    preloader = Preloader(emu)
    preload_entry = preloader.inject_file(ns.preload, addr=ns.preload_addr)
    logger.info(f"预加载载荷: {ns.preload} -> 0x{preload_entry:x}")
    for h in emu.harts:
        h.csrs["misa"].val = (2 << 62) | (1 << 18) | (1 << 20)
    if fdt_addr is not None:
        emu.load_dtb_blob(ram_base + 0x2200000, emu.build_dtb())
    return preload_entry


def _setup_entry_symbol(
    emu: Emulator,
    image: FirmwareImage,
    entry_symbol: str,
) -> int | None:
    """将 *entry_symbol* 的地址写入 a0 (x10), 返回符号 VA."""
    target_entry = image.symbols.get(entry_symbol)
    if target_entry is None:
        logger.error(f"符号不存在: {entry_symbol}")
        sys.exit(1)
    for h in emu.harts:
        h.write_gpr(10, target_entry)
    logger.info(f"a0 ← {entry_symbol} = 0x{target_entry:x}")
    return target_entry


def _load_kernel_image(
    emu: Emulator,
    kernel_path: str,
    kernel_addr: int | None,
    default_addr: int,
) -> tuple[str, int] | None:
    """预载内核 Image 到 RAM, 返回 (path, addr)."""
    if kernel_path is None:
        return None
    addr = kernel_addr if kernel_addr is not None else default_addr
    data = Path(kernel_path).read_bytes()
    emu.bus.write_ram_direct(addr, data)
    logger.info(
        f"内核已预载: {kernel_path} -> PA 0x{addr:x}"
        f" ({fmt_size(len(data))})"
    )
    return kernel_path, addr


# ============================================================
#  main
# ============================================================


def debugger(args: list[str] | None = None) -> None:
    """命令行入口: 加载固件并启动交互调试器."""
    parser = _build_arg_parser()
    ns = parser.parse_args(args)

    _setup_logger(ns.log_level)

    ram_size = _parse_ram_size(ns.ram)
    image, load_offset = _load_firmware_image(
        ns.firmware, ns.base_addr, ns.ram_base,
    )

    effective_entry = image.entry_point + load_offset
    prog_cnt = ns.prog_cnt if ns.prog_cnt is not None else effective_entry

    plat_cfg = PlatformConfig(
        num_harts=ns.harts,
        ram_size=ram_size,
        ram_base=ns.ram_base,
        prog_cnt=prog_cnt,
        periph=PeripheralConfig(),
    )
    emu = Emulator(plat_cfg, bootargs=ns.bootargs)
    emu.load_firmware(image, load_offset=load_offset)
    for h in emu.harts:
        h.pc = prog_cnt

    fdt_addr = _configure_fdt(emu, ns, ns.ram_base, ram_size)
    fdt_note = (
        f", FDT=0x{fdt_addr:x}" if fdt_addr is not None
        else ", DTB=off" if ns.no_fdt
        else ""
    )
    logger.info(
        f"固件已写入 RAM, {ns.harts} hart(s) 就绪,"
        f" PC=0x{prog_cnt:x}{fdt_note}"
    )

    preload_entry = _setup_preload(emu, ns, fdt_addr, ns.ram_base)
    target_entry = (
        _setup_entry_symbol(emu, image, ns.entry_symbol)
        if ns.entry_symbol is not None else None
    )

    if preload_entry is not None:
        for h in emu.harts:
            h.pc = preload_entry
    elif target_entry is not None:
        for h in emu.harts:
            h.pc = target_entry
        logger.warning(
            "无 --preload, 直接跳转到入口 (未设置 sp/gp, 可能出错)"
        )

    dbg = Debugger(emulator=emu, hart_id=0, image=image)
    dbg._fdt_addr = fdt_addr
    dbg._load_offset = load_offset
    dbg._preload_path = ns.preload

    kernel_info = _load_kernel_image(
        emu, ns.kernel, ns.kernel_addr,
        default_addr=ns.ram_base + 0x200000,
    )
    if kernel_info is not None:
        dbg._kernel_path, dbg._kernel_addr = kernel_info

    if ns.sym:
        dbg.load_kernel_symbols(ns.sym, ns.sym_base)

    dbg.repl()


if __name__ == "__main__":
    debugger()
