#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""CLI 入口 — 解析参数、加载固件并启动交互调试器."""

import argparse
from pathlib import Path
import sys

from loguru import logger

from pyremu import configs_gen
from pyremu.debug import Debugger
from pyremu.debug.types import ImageKind, LoadedImage
from pyremu.debug.utils import fmt_size, RAM_MUL
from pyremu.emulator import Emulator
from pyremu.env_inject import Preloader
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils.parse_bin import FirmwareImage, parse_firmware

# virtio-blk MMIO 基址 — 置于默认外设 (UART/SPI/I2C/GPIO/watchdog) 之后的空闲槽。
_VIRTIO_BLK_BASE = 0x1000_5000

# 两个 virtio-net 实例的 MMIO 基址 — 紧接 virtio-blk 之后的空闲槽。
# 中断源编号取 2 与 3.
_VIRTIO_NET_S_BASE = 0x1000_7000
_VIRTIO_NET_M_BASE = 0x1000_8000

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
        "--base-addr",
        type=lambda x: int(x, 0),
        default=0x80000000,
        help="raw binary 的加载基址 (ELF/PE 时忽略, 默认 0x80000000)",
    )
    p.add_argument(
        "--prog-cnt",
        type=lambda x: int(x, 0),
        default=None,
        help="程序计数器 — 每个 hart 的开始执行地址",
    )
    p.add_argument(
        "--ram-base",
        type=lambda x: int(x, 0),
        default=0x8000_0000,
        help="物理内存基址 (默认 0x80000000)",
    )
    p.add_argument(
        "--rom-base",
        type=lambda x: int(x, 0),
        default=configs_gen.ROM_BASE,
        help=f"引导只读存储器基址 (默认 0x{configs_gen.ROM_BASE:x}; 取 0 表示不注册)",
    )
    p.add_argument(
        "--rom-size",
        type=lambda x: int(x, 0),
        default=configs_gen.ROM_SIZE,
        help=f"引导只读存储器大小 (默认 0x{configs_gen.ROM_SIZE:x} 字节)",
    )
    p.add_argument(
        "--fdt",
        type=lambda x: int(x, 0) if x else None,
        default=-1,
        nargs="?",
        const=-1,
        help=(
            "设备树加载地址 (默认 RAM 顶端 -64 KiB; 复位地址落在引导只读存储器内时"
            "取该存储器基址加 ROM_DTB_OFFSET)"
        ),
    )
    p.add_argument(
        "--fdt-file",
        type=str,
        default=None,
        metavar="PATH[:ADDR]",
        help="加载预编译 DTB 文件 (可选 :地址)",
    )
    p.add_argument(
        "--no-fdt",
        action="store_true",
        default=False,
        help="禁用设备树",
    )
    p.add_argument("--harts", type=int, default=1, help="hart 数量 (默认 1)")
    p.add_argument(
        "--ram",
        type=str,
        default="128M",
        help="RAM 大小 (支持 K/M/G 后缀, 默认 128M)",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=[
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ],
        help="日志级别 (默认 INFO)",
    )
    p.add_argument(
        "--preload",
        type=str,
        default=None,
        help="预加载 shellcode 文件路径",
    )
    p.add_argument(
        "--preload-addr",
        type=lambda x: int(x, 0),
        default=None,
        help=(
            "预加载 shellcode 的加载地址，默认 RAM 顶端 -128 KiB; 落在引导只读存储器"
            "的范围内时加载该设备"
        ),
    )
    p.add_argument(
        "--entry-symbol",
        type=str,
        default=None,
        help="目标程序入口符号名，如 'main'",
    )
    p.add_argument(
        "--sym-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="符号文件所在目录, 作为 sym 命令的初始候选目录",
    )
    p.add_argument(
        "--kernel",
        type=str,
        default=None,
        metavar="PATH",
        help="内核 Image 二进制文件, 预载到 RAM",
    )
    p.add_argument(
        "--kernel-addr",
        type=lambda x: int(x, 0),
        default=None,
        metavar="ADDR",
        help="内核 Image 加载地址",
    )
    p.add_argument(
        "--bootargs",
        type=str,
        default=None,
        metavar="ARGS",
        help="内核命令行参数, 写入 DTB /chosen/bootargs",
    )
    p.add_argument(
        "--initrd",
        type=str,
        default=None,
        metavar="PATH",
        help="initramfs 文件, 加载到 RAM 并写入 DTB /chosen",
    )
    p.add_argument(
        "--initrd-addr",
        type=lambda x: int(x, 0),
        default=None,
        metavar="ADDR",
        help="initramfs 加载地址",
    )
    p.add_argument(
        "--disk",
        type=str,
        default=None,
        metavar="PATH",
        help="virtio-blk 磁盘镜像, 挂载为 /dev/vda; 无写权限时只读打开",
    )
    p.add_argument(
        "--hart-logs",
        type=str,
        default=None,
        metavar="DIR",
        help="将各 hart 的串口输出另存到 DIR/hart<N>.log",
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
    """解析 RAM 大小字符串, 支持 K/KB/M/MB/G/GB 后缀, 返回字节数."""
    s = ram_str.upper().rstrip("B")
    if s and s[-1] in RAM_MUL:
        return int(s[:-1]) * RAM_MUL[s[-1]]
    return int(s)


def _is_rom_addr(addr: int, rom_base: int, rom_size: int) -> bool:
    """判断物理地址是否属于引导只读存储器的窗口; rom_base 取 0 时恒为 False."""
    return bool(rom_base) and rom_base <= addr < rom_base + rom_size


def _image_in_rom(
    image: FirmwareImage,
    rom_base: int,
    rom_size: int,
) -> bool:
    """判断固件的全部段是否落在引导只读存储器的窗口内."""
    if not rom_base:
        return False
    if not image.segments:
        return _is_rom_addr(image.entry_point, rom_base, rom_size)
    start = min(seg.vaddr for seg in image.segments)
    end = max(seg.vaddr + seg.memsz for seg in image.segments)
    return rom_base <= start and end <= rom_base + rom_size


def _load_firmware_image(
    fw_path: str,
    base_addr: int,
    ram_base: int,
    rom_base: int = configs_gen.ROM_BASE,
    rom_size: int = configs_gen.ROM_SIZE,
) -> tuple[FirmwareImage, int]:
    """解析固件并计算装载偏移.

    返回 (image, load_offset). 段的链接地址最小值不低于 ram_base, 或全部段落在
    引导只读存储器的窗口内时偏移为 0; 其余 ELF 固件按链接地址加 ram_base 装入.
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
        f"格式={image.format}, 入口=0x{image.entry_point:x}, 段数={len(image.segments)}"
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
        # 段落在引导只读存储器的窗口内时按链接地址装入, 不施加装载偏移.
        if _image_in_rom(image, rom_base, rom_size):
            logger.info(
                f"固件段位于引导只读存储器 [0x{rom_base:x}, "
                f"0x{rom_base + rom_size:x}) 内, 不施加装载偏移"
            )
            return image, load_offset
        min_vaddr = min(seg.vaddr for seg in image.segments)
        if min_vaddr >= ram_base:
            return image, load_offset
        load_offset = ram_base
        logger.info(
            f"PIE 固件段偏移 +0x{load_offset:x}"
            f" (min vaddr=0x{min_vaddr:x} < ram_base=0x{ram_base:x})"
        )
    return image, load_offset


def _default_fdt_addr(
    ram_base: int,
    ram_size: int,
    rom_base: int,
    is_rom_boot: bool,
) -> int:
    """设备树的默认加载地址.

    复位地址位于引导只读存储器的窗口内时取基址加 ROM_DTB_OFFSET, 与零阶段
    引导程序按 _start + 0x10000 算出的交付地址一致; 否则取 RAM 顶端 − 64 KiB.
    """
    if is_rom_boot:
        return rom_base + configs_gen.ROM_DTB_OFFSET
    return ram_base + ram_size - 0x10000


def _configure_fdt(
    emu: Emulator,
    ns: argparse.Namespace,
    ram_base: int,
    ram_size: int,
    default_addr: int | None = None,
) -> int | None:
    """根据 CLI 参数配置设备树, 返回 DTB 加载地址 (或 None).

    default_addr 为 None 时取 RAM 顶端 − 64 KiB.
    """
    if ns.no_fdt:
        return None
    if default_addr is None:
        default_addr = ram_base + ram_size - 0x10000
    if ns.fdt_file:
        parts = ns.fdt_file.rsplit(":", 1)
        file_path = parts[0]
        custom_addr = int(parts[1], 0) if len(parts) > 1 else None
        # 判定用 None 比较: 地址 0 是合法入参 (ram_base 为 0 的固件), 用真值判定会
        # 把这一取值当作未给出.
        addr: int = custom_addr if custom_addr is not None else default_addr
        emu.load_dtb_file(file_path, addr=addr)
        return addr
    if ns.fdt:
        addr = ns.fdt if ns.fdt != -1 else default_addr
        emu.load_dtb(addr)
        return addr
    return None


def _setup_preload(
    emu: Emulator,
    ns: argparse.Namespace,
    fdt_addr: int | None,
    ram_base: int,
) -> tuple[int, int] | None:
    """预加载 shellcode, 返回 (入口地址, 字节数); 未给出 --preload 时为 None."""
    if ns.preload is None:
        return None
    preloader = Preloader(emu)
    preload_entry = preloader.inject_file(ns.preload, addr=ns.preload_addr)
    size = Path(ns.preload).stat().st_size
    logger.info(f"预加载载荷: {ns.preload} -> 0x{preload_entry:x} ({fmt_size(size)})")
    for h in emu.harts:
        h.csrs["misa"].val = (2 << 62) | (1 << 18) | (1 << 20)
    if fdt_addr is not None:
        emu.load_dtb_blob(ram_base + 0x2200000, emu.build_dtb())
    return preload_entry, size


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
    logger.info(f"a0 <- {entry_symbol} = 0x{target_entry:x}")
    return target_entry


def _load_initrd(
    emu: Emulator,
    path: str | None,
    addr: int | None,
    ram_base: int,
    ram_size: int,
) -> None:
    """加载 initramfs 到 RAM 并记录范围 (必须在生成 DTB 之前调用).

    默认地址置于 DTB (RAM 顶端 −64 KiB) 下方, 按 2 MiB 向下对齐,
    以避开内核 Image (ram_base + 固件映像保留区大小) 区域。
    """
    if path is None:
        return
    size = Path(path).stat().st_size
    if addr is None:
        dtb_top = ram_base + ram_size - 0x10000
        addr = (dtb_top - size) & ~0x1FFFFF  # 2 MiB 向下对齐
    emu.load_initrd(path, addr)


def _firmware_span(image: FirmwareImage, load_offset: int) -> tuple[int, int]:
    """固件装入 RAM 后的物理区间; 映像没有段时取入口地址处的空区间."""
    if not image.segments:
        start = image.entry_point + load_offset
        return start, start
    start = min(seg.vaddr for seg in image.segments) + load_offset
    end = max(seg.vaddr + seg.memsz for seg in image.segments) + load_offset
    return start, end


def _record_loaded_images(
    dbg: Debugger,
    ns: argparse.Namespace,
    image: FirmwareImage,
    preload_info: tuple[int, int] | None,
) -> None:
    """把各入参装入的映像登记到调试器, 供 bins-info 与 restart 使用."""
    emu = dbg._emu
    start, end = _firmware_span(image, dbg._load_offset)
    dbg._bins_images[ImageKind.FIRMWARE] = LoadedImage(ns.firmware, start, end - start)
    if preload_info is not None and ns.preload:
        entry, size = preload_info
        dbg._bins_images[ImageKind.PRELOAD] = LoadedImage(ns.preload, entry, size)
    initrd = emu.initrd
    if initrd is not None and ns.initrd:
        dbg._bins_images[ImageKind.INITRD] = LoadedImage(
            ns.initrd, initrd.start, initrd.end - initrd.start
        )
    if ns.fdt_file and dbg._fdt_addr is not None and emu.dtb_addr == dbg._fdt_addr:
        dbg._bins_images[ImageKind.FDT] = LoadedImage(
            ns.fdt_file.rsplit(":", 1)[0], dbg._fdt_addr, emu.dtb_size
        )


def _load_kernel_image(
    emu: Emulator,
    kernel_path: str,
    kernel_addr: int | None,
    default_addr: int,
) -> tuple[str, int, int] | None:
    """预载内核 Image 到 RAM, 返回 (path, addr, 字节数)."""
    if kernel_path is None:
        return None
    addr = kernel_addr if kernel_addr is not None else default_addr
    data = Path(kernel_path).read_bytes()
    emu.bus.write_ram_direct(addr, data)
    logger.info(f"内核已预载: {kernel_path} -> PA 0x{addr:x} ({fmt_size(len(data))})")
    return kernel_path, addr, len(data)


def _build_peripheral_config(
    ns: argparse.Namespace,
) -> tuple[PeripheralConfig, str | None]:
    """按 CLI 参数创建外设配置与内核命令行参数.

    两个 virtio-net 实例恒启用: S 模式一侧实例供受调试程序的协议栈驱动,
    M 模式一侧实例供飞地直通, 二者取不同的 MMIO 基址与中断源编号. --disk 给出时
    启用 virtio-blk, 且未显式给出 --bootargs 时提供默认根挂载参数 (只读, 根文件
    系统位于 ext4 分区).
    """
    periph = PeripheralConfig()
    periph.virtio_net_s_base = _VIRTIO_NET_S_BASE
    periph.virtio_net_m_base = _VIRTIO_NET_M_BASE
    bootargs = ns.bootargs
    if not ns.disk:
        return periph, bootargs
    periph.virtio_blk_base = _VIRTIO_BLK_BASE
    if bootargs is not None:
        return periph, bootargs
    # debootstrap 的 minbase 变体不创建 /sbin/init, 而它是指向 systemd 的符号
    # 链接; 内核按序查找 init 会落到 /bin/sh 并阻塞在无输入的 stdin 读取上, 表现
    # 为指令计数器攀升但无输出. 显式指定 systemd 路径可避开. 打开 systemd 的调试
    # 输出并转发到串口, 便于观察它静默失败时的输出.
    bootargs = (
        "earlycon=sbi console=ttySIF0 "
        "root=/dev/vda ro "
        "init=/usr/lib/systemd/systemd "
        "random.trust_bootloader=on "
        "deferred_probe_timeout=10 "
        "systemd.log_level=debug systemd.log_target=console "
        "systemd.journald.forward_to_console=1"
    )
    return periph, bootargs


# ============================================================
#  main
# ============================================================


def debugger(args: list[str] | None = None) -> None:
    """命令行入口: 加载固件并启动交互调试器."""
    parser = _build_arg_parser()
    ns = parser.parse_args(args)

    _setup_logger(ns.log_level)

    ram_size = _parse_ram_size(ns.ram)
    rom_base = ns.rom_base
    rom_size = ns.rom_size
    image, load_offset = _load_firmware_image(
        ns.firmware,
        ns.base_addr,
        ns.ram_base,
        rom_base,
        rom_size,
    )

    effective_entry = image.entry_point + load_offset
    # 判定用 None 比较: 入口地址 0 是合法入参 (sifive_u54 预设即取 0), 用真值判定
    # 会把这一取值当作未给出而改用映像入口.
    prog_cnt = ns.prog_cnt if ns.prog_cnt is not None else effective_entry
    # 复位地址取预加载载荷的地址, 未给出载荷或其地址时取固件入口. 复位地址落在
    # 引导只读存储器的窗口内时从该存储器启动.
    if ns.preload is not None and ns.preload_addr is not None:
        reset_addr = ns.preload_addr
    else:
        reset_addr = prog_cnt
    is_rom_boot = _is_rom_addr(reset_addr, rom_base, rom_size)

    periph, bootargs = _build_peripheral_config(ns)

    plat_cfg = PlatformConfig(
        num_harts=ns.harts,
        ram_size=ram_size,
        ram_base=ns.ram_base,
        rom_base=rom_base,
        rom_size=rom_size,
        prog_cnt=prog_cnt,
        periph=periph,
        disk_image=ns.disk,
    )
    emu = Emulator(plat_cfg, bootargs=bootargs)
    emu.load_firmware(image, load_offset=load_offset)
    if ns.hart_logs and emu.uart:
        emu.uart.set_hart_log_dir(ns.hart_logs)
        logger.info(f"各 hart 串口输出另存至: {ns.hart_logs}/hart<N>.log")
    for h in emu.harts:
        h.pc = prog_cnt

    # initramfs 必须在生成 DTB 之前加载 (范围需写入 /chosen)。
    _load_initrd(emu, ns.initrd, ns.initrd_addr, ns.ram_base, ram_size)

    default_fdt_addr = _default_fdt_addr(ns.ram_base, ram_size, rom_base, is_rom_boot)
    fdt_addr = _configure_fdt(emu, ns, ns.ram_base, ram_size, default_addr=default_fdt_addr)
    fdt_note = (
        f", FDT=0x{fdt_addr:x}" if fdt_addr is not None else ", DTB=off" if ns.no_fdt else ""
    )
    logger.info(f"固件已写入物理内存, {ns.harts} hart(s) 就绪, PC=0x{prog_cnt:x}{fdt_note}")

    preload_info = _setup_preload(emu, ns, fdt_addr, ns.ram_base)
    target_entry = (
        _setup_entry_symbol(emu, image, ns.entry_symbol) if ns.entry_symbol else None
    )

    # 判定用 None 比较: 预加载载荷与入口符号都可以落在地址 0.
    if preload_info is not None:
        for h in emu.harts:
            h.pc = preload_info[0]
    elif target_entry is not None:
        for h in emu.harts:
            h.pc = target_entry
        logger.warning("无 --preload, 直接跳转到入口 (未设置 sp/gp, 可能出错)")

    dbg = Debugger(emulator=emu, hart_id=0, image=image)
    dbg._fdt_addr = fdt_addr
    dbg._load_offset = load_offset
    _record_loaded_images(dbg, ns, image, preload_info)

    kernel_info = _load_kernel_image(
        emu,
        ns.kernel,
        ns.kernel_addr,
        default_addr=ns.ram_base + configs_gen.FW_RESERVED_MEM_SIZE,
    )
    if kernel_info:
        kernel_path, kernel_addr, kernel_size = kernel_info
        dbg._bins_images[ImageKind.KERNEL] = LoadedImage(kernel_path, kernel_addr, kernel_size)

    if ns.sym_dir:
        dbg._sym_mod_dir = str(Path(ns.sym_dir).expanduser().resolve())
        logger.info(f"符号候选目录: {dbg._sym_mod_dir}")

    dbg.repl()


if __name__ == "__main__":
    debugger()
