#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
Flat Device Tree (DTB) 生成器 — 基于 libfdt 构建设备树 blob.

将平台配置和外设映射序列化为符合 Devicetree 规范的二进制格式,
供固件通过 a1 寄存器接收。 所有 FDT 构建逻辑集中于此模块,
避免散落于 emulator.py 造成臃肿。
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import libfdt

from pyremu.platform import PlatformConfig

if TYPE_CHECKING:
    from pyremu.interrupt.plic import PLIC
    from pyremu.peripheral import GPIO, I2C, SPI, UART, VirtIOBlock


# ============================================================
#  ISA 字符串 -> 扩展名列表 (用于 riscv,isa-extensions)
# ============================================================


def _isa_to_extensions(
    isa: str,
) -> list[str]:
    """将 RISC-V ISA 字符串解析为独立扩展名列表.

    例:
        "rv64imac" -> ["i", "m", "a", "c"]
        "rv64g" -> ["i", "m", "a", "f", "d"]
        "rv64imafdc_zicsr_zifencei" -> ["i", "m", "a", "f", "d", "c", "zicsr", "zifencei"]

    用于生成 DTB ``riscv,isa-extensions`` 属性
    (Linux 6.x+ 优先使用, 旧式 ``riscv,isa`` 字符串已废弃).
    """
    s = isa
    # 去掉 rv32/rv64 前缀
    if s.startswith("rv64"):
        s = s[4:]
    elif s.startswith("rv32"):
        s = s[4:]

    # 分离单字母扩展 (在第一个 '_' 之前) 与多字母扩展
    single_part: str
    multi_parts: list[str]
    if "_" in s:
        parts = s.split("_")
        single_part, *multi_parts = parts
    else:
        single_part = s
        multi_parts = []

    exts: list[str] = []
    for ch in single_part:
        if ch == "g":
            # "g" = "imafd" 缩写
            exts.extend(["i", "m", "a", "f", "d"])
        else:
            exts.append(ch)
    exts.extend(multi_parts)
    return exts


# ============================================================
#  内存区域编码辅助
# ============================================================


def _encode_reg_4mib(
    addr: int,
    size: int,
) -> bytes:
    """将 64-bit 地址/大小编码为 4 个大端 u32 (addr_hi, addr_lo, size_hi, size_lo)."""
    return struct.pack(">IIII", 0, addr, 0, size)


# ============================================================
#  DTB 构建入口
# ============================================================


def build_dtb(
    cfg: PlatformConfig,
    *,
    uart: UART | None = None,
    spi: SPI | None = None,
    i2c_gen: I2C | None = None,
    gpio: GPIO | None = None,
    virtio_blk: VirtIOBlock | None = None,
    plic: PLIC | None = None,
    bootargs: str | None = None,
) -> bytes:
    """基于平台配置和外设映射构建完整的 DTB blob.

    Args:
        cfg: 平台配置, 描述 CPU 拓扑、内存布局和 ISA 特性.
        uart: UART 外设实例 (None 则跳过 UART 节点).
        spi: SPI 外设实例.
        i2c_gen: I2C 外设实例.
        gpio: GPIO 外设实例.
        virtio_blk: virtio-blk 外设实例.
        plic: PLIC 中断控制器实例.
        bootargs: 内核命令行参数, 写入 /chosen/bootargs.

    Returns:
        完整的 DTB blob 字节串, 可直接写入 RAM 供固件使用.
    """
    p = cfg.periph
    sw = libfdt.FdtSw(8192)
    sw.finish_reservemap()

    # ============================================================
    #  根节点
    # ============================================================
    sw.begin_node("")
    sw.property_u32("#address-cells", 2)
    sw.property_u32("#size-cells", 2)
    sw.property_string("compatible", "pyremu,riscv64")
    sw.property_string("model", f"pyremu,{cfg.isa}")

    # ============================================================
    #  chosen — stdout + 内核命令行
    # ============================================================
    if uart is not None or bootargs:
        sw.begin_node("chosen")
        if uart is not None:
            sw.property_string("stdout-path", f"/soc/serial@{p.uart_base:x}")
        if bootargs:
            sw.property_string("bootargs", bootargs)
        sw.end_node()  # chosen

    # ============================================================
    #  aliases — 串口别名 (SiFive UART 驱动 probe 需要 serialN)
    # ============================================================
    if uart is not None:
        sw.begin_node("aliases")
        sw.property_string("serial0", f"/soc/serial@{p.uart_base:x}")
        sw.end_node()  # aliases

    # ============================================================
    #  cpus — 每个 hart 一个 cpu@N 节点
    # ============================================================
    sw.begin_node("cpus")
    sw.property_u32("#address-cells", 1)
    sw.property_u32("#size-cells", 0)
    sw.property_u32("timebase-frequency", cfg.timebase_freq)

    # 构建 riscv,isa-extensions 列表 (Linux 6.x+ 优先使用)
    isa_extensions = _isa_to_extensions(cfg.isa)
    isa_ext_bytes = b"".join(e.encode("ascii") + b"\x00" for e in isa_extensions)

    cpu_phandles: list[int] = []
    for i in range(cfg.num_harts):
        sw.begin_node(f"cpu@{i}")
        sw.property_string("device_type", "cpu")
        sw.property_u32("reg", i)
        sw.property_string("compatible", "riscv")
        # 保留旧式 riscv,isa 向后兼容旧内核 (Linux <6.x)
        sw.property_string("riscv,isa", cfg.isa)
        # 新式 riscv,isa-extensions — 消除 "Falling back to deprecated" 警告
        sw.property("riscv,isa-extensions", isa_ext_bytes)
        sw.property_string("mmu-type", "riscv,sv39")
        sw.property_string("status", "okay")

        # 中断控制器子节点 (供 CLINT interrupts-extended 引用)
        sw.begin_node("interrupt-controller")
        sw.property_string("compatible", "riscv,cpu-intc")
        sw.property_u32("#interrupt-cells", 1)
        sw.property_string("interrupt-controller", "")
        phandle = i + 1
        sw.property_u32("phandle", phandle)
        cpu_phandles.append(phandle)
        sw.end_node()  # interrupt-controller
        sw.end_node()  # cpu@i
    sw.end_node()  # cpus

    # ============================================================
    #  memory
    # ============================================================
    sw.begin_node("memory")
    sw.property_string("device_type", "memory")
    sw.property("reg", _encode_reg_4mib(cfg.ram_base, cfg.ram_size))
    sw.end_node()  # memory

    # ============================================================
    #  soc simple-bus — 挂载所有 MMIO 外设
    # ============================================================
    sw.begin_node("soc")
    sw.property_u32("#address-cells", 2)
    sw.property_u32("#size-cells", 2)
    sw.property_string("compatible", "simple-bus")
    sw.property("ranges", b"")  # 透传

    # -- CLINT --
    clint_base = p.clint_base
    sw.begin_node(f"clint@{clint_base:x}")
    sw.property_string("compatible", "riscv,clint0")
    sw.property("reg", _encode_reg_4mib(clint_base, 0x10000))
    # interrupts-extended: 每个 hart 两条中断 <&cpu_intc 3 &cpu_intc 7>
    ie_cells: list[int] = []
    for ph in cpu_phandles:
        ie_cells.extend([ph, 3, ph, 7])  # M-SW-IRQ=3, M-TIMER-IRQ=7
    sw.property(
        "interrupts-extended",
        struct.pack(">" + "I" * len(ie_cells), *ie_cells),
    )
    sw.end_node()  # clint

    # -- PLIC --
    if plic is not None:
        plic_base = p.plic_base
        plic_phandle = len(cpu_phandles) + 1
        sw.begin_node(f"plic@{plic_base:x}")
        sw.property_u32("phandle", plic_phandle)
        sw.property_string("compatible", "riscv,plic0")
        sw.property_u32("#interrupt-cells", 1)
        sw.property_string("interrupt-controller", "")
        sw.property("reg", _encode_reg_4mib(plic_base, plic.size))
        sw.property_u32("riscv,ndev", 128)
        # interrupts-extended: 每个 hart context 一条 <&cpu_intc 11> (MEIP)
        plic_ie: list[int] = []
        for ph in cpu_phandles:
            plic_ie.extend([ph, 11])  # M-EXT-IRQ=11
        sw.property(
            "interrupts-extended",
            struct.pack(">" + "I" * len(plic_ie), *plic_ie),
        )
        sw.end_node()  # plic

    # -- UART (SiFive NS16550A 兼容) --
    if uart is not None:
        # 固定时钟 — 使 SiFive 串口驱动在无 CCF 时钟控制器的 DTB 中也能 probe.
        # 内核 CONFIG_COMMON_CLK=y 时 devm_clk_get() 若找不到 clocks 属性
        # 返回 EPROBE_DEFER, 驱动永不匹配; 提供 fixed-clock 解决此问题.
        clock_phandle = len(cpu_phandles) + (2 if plic is not None else 1)
        sw.begin_node("clock")
        sw.property_u32("phandle", clock_phandle)
        sw.property_string("compatible", "fixed-clock")
        sw.property_u32("#clock-cells", 0)
        sw.property_u32("clock-frequency", cfg.timebase_freq)
        sw.end_node()  # clock

        sw.begin_node(f"serial@{p.uart_base:x}")
        sw.property_string("compatible", "sifive,uart0")
        sw.property_string("status", "okay")
        sw.property("reg", _encode_reg_4mib(p.uart_base, 0x1000))
        sw.property("clocks", struct.pack(">I", clock_phandle))
        sw.property_u32("clock-frequency", cfg.timebase_freq)
        sw.property(
            "interrupts-extended",
            struct.pack(">II", plic_phandle if plic is not None else 1, 10),
        )
        sw.end_node()  # serial

    # -- SPI --
    if spi is not None:
        sw.begin_node(f"spi@{p.spi_base:x}")
        sw.property_string("compatible", "pyremu,spi0")
        sw.property("reg", _encode_reg_4mib(p.spi_base, 0x1000))
        sw.end_node()  # spi

    # -- I2C --
    if i2c_gen is not None:
        sw.begin_node(f"i2c@{p.i2c_base:x}")
        sw.property_string("compatible", "pyremu,i2c0")
        sw.property("reg", _encode_reg_4mib(p.i2c_base, 0x1000))
        sw.end_node()  # i2c

    # -- GPIO --
    if gpio is not None:
        sw.begin_node(f"gpio@{p.gpio_base:x}")
        sw.property_string("compatible", "pyremu,gpio0")
        sw.property("reg", _encode_reg_4mib(p.gpio_base, 0x1000))
        sw.end_node()  # gpio

    # -- virtio-blk (compatible = "virtio,mmio") --
    if virtio_blk is not None:
        sw.begin_node(f"virtio@{p.virtio_blk_base:x}")
        sw.property_string("compatible", "virtio,mmio")
        sw.property("reg", _encode_reg_4mib(p.virtio_blk_base, 0x200))
        sw.end_node()  # virtio

    sw.end_node()  # soc
    sw.end_node()  # root

    return bytes(sw.as_fdt().as_bytearray())
