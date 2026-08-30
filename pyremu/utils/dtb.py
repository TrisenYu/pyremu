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

from dataclasses import dataclass
import os
import struct
from typing import TYPE_CHECKING

import libfdt

from pyremu.configs_gen import PYREMU_AIA
from pyremu.peripheral.virtio_blk import VIRTIO_BLK_IRQ
from pyremu.platform import InterruptMode, PlatformConfig

if TYPE_CHECKING:
    from pyremu.interrupt.aplic import APLIC
    from pyremu.interrupt.imsic import IMSIC
    from pyremu.interrupt.plic import PLIC
    from pyremu.peripheral import CRNG, GPIO, I2C, SPI, UART, VirtIOBlock
    from pyremu.peripheral.watchdog import HartWatchdog

# APLIC 有线中断触发类型 (interrupts 第二 cell). 与 QEMU virt.c 对齐:
# UART0 / virtio 均为 IRQ_TYPE_LEVEL_HIGH (0x4), 非 0 (NONE).
# 写 0 会让内核 aplic_irq_set_type 配成 SM_INACTIVE, 完成中断永不投递.
_IRQ_TYPE_LEVEL_HIGH = 0x4



@dataclass
class Initrd:
    """已加载的 initramfs 物理内存范围 (供 DTB /chosen 使用).

    start/end 为物理地址; Linux 读取 /chosen/linux,initrd-start 与
    linux,initrd-end 后将该区间的 cpio[.gz] 解包为 rootfs。
    """

    start: int
    end: int


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
#  各节点构建 (build_dtb 的分解: 每个函数写入一个/一组节点)
# ============================================================


def _dtb_chosen(
    sw: libfdt.FdtSw,
    cfg: PlatformConfig,
    uart: UART | None,
    bootargs: str | None,
    initrd: Initrd | None,
) -> None:
    """/chosen — stdout-path + 内核命令行 + initrd 范围."""
    if not (bootargs or initrd is not None):
        return
    p = cfg.periph
    sw.begin_node("chosen")
    if uart is not None:
        sw.property_string("stdout-path", f"/soc/serial@{p.uart_base:x}")
    if bootargs:
        sw.property_string("bootargs", bootargs)

    # 引导期熵种子 —— 使内核在启动即完成 CRNG 初始化, 避免用户态早期阻塞在
    # getrandom()/wait_for_random_bytes() (详见文件顶部注释)。
    # /chosen/rng-seed — 引导期熵种子 (供内核初始化 CRNG).
    #
    # 为什么需要: 无此属性时, 内核启动到用户态早期会阻塞在 wait_for_random_bytes()
    # 直到 CRNG 完成初始化。真实硬件靠中断/设备抖动积累熵，但目前没有实现这一随机数生成器设备
    #
    # 修复原理: drivers/of/fdt.c 的 early_init_dt_scan_chosen 读取 /chosen/rng-seed,
    # 调用 add_bootloader_randomness(); 配合 random.trust_bootloader=on 内核参数,
    # credit_init_bits(len*8) 直接完成 CRNG 初始化。
    # 种子 >= 32 字节 (256 bit) 即可;
    # 使用宿主 os.urandom(64) 提供 512 bit 真随机熵
    sw.property("rng-seed", os.urandom(64))
    if initrd is not None:
        # initramfs (cpio[.gz]) 已加载到 [initrd.start, initrd.end)。
        # Linux 从 /chosen 读取这两个属性; 以 2-cell (u64) 大端编码,
        # of_read_number 按属性长度自动识别 4/8 字节。
        sw.property("linux,initrd-start", struct.pack(">Q", initrd.start))
        sw.property("linux,initrd-end", struct.pack(">Q", initrd.end))
    sw.end_node()


def _dtb_aliases(
    sw: libfdt.FdtSw,
    cfg: PlatformConfig,
    uart: UART | None,
) -> None:
    """/aliases — 串口别名 (SiFive UART 驱动 probe 需要 serialN)."""
    if uart is None:
        return
    sw.begin_node("aliases")
    sw.property_string("serial0", f"/soc/serial@{cfg.periph.uart_base:x}")
    sw.end_node()  # aliases


def _dtb_cpus(
    sw: libfdt.FdtSw,
    cfg: PlatformConfig,
    *,
    is_aia: bool = False,
) -> list[int]:
    """/cpus — 每 hart 一个 cpu@N 节点; 返回各 hart intc 的 phandle 列表."""
    sw.begin_node("cpus")
    sw.property_u32("#address-cells", 1)
    sw.property_u32("#size-cells", 0)
    sw.property_u32("timebase-frequency", cfg.timebase_freq)

    # 构建 riscv,isa 与 riscv,isa-extensions 列表 (Linux 6.x+ 优先使用)
    # AIA 模式下追加 smaia/ssaia 扩展, 使内核 IMSIC 驱动通过
    # riscv_isa_extension_available(NULL, SxAIA) 检查后正常初始化.
    isa_str = cfg.isa
    if is_aia and "_smaia" not in isa_str:
        isa_str = f"{isa_str}_smaia_ssaia"
    isa_extensions = _isa_to_extensions(isa_str)
    isa_ext_bytes = b"".join(e.encode("ascii") + b"\x00" for e in isa_extensions)

    cpu_phandles: list[int] = []
    for i in range(cfg.num_harts):
        sw.begin_node(f"cpu@{i}")
        sw.property_string("device_type", "cpu")
        sw.property_u32("reg", i)
        sw.property_string("compatible", "riscv")
        # 保留旧式 riscv,isa 向后兼容旧内核 (Linux <6.x)
        sw.property_string("riscv,isa", isa_str)
        # 新式 riscv,isa-extensions — 消除 "Falling back to deprecated" 警告
        sw.property("riscv,isa-extensions", isa_ext_bytes)
        sw.property_string("mmu-type", "riscv,sv39")
        sw.property_string("status", "okay")

        # 中断控制器子节点 (供 CLINT interrupts-extended 引用)
        sw.begin_node("interrupt-controller")
        sw.property_string("compatible", "riscv,cpu-intc")
        sw.property_u32("#interrupt-cells", 1)
        sw.property("interrupt-controller", b"")
        phandle = i + 1
        sw.property_u32("phandle", phandle)
        cpu_phandles.append(phandle)
        sw.end_node()  # interrupt-controller
        sw.end_node()  # cpu@i
    sw.end_node()  # cpus
    return cpu_phandles


def _dtb_reserved_memory(
    sw: libfdt.FdtSw,
    ranges: list[tuple[int, int]],
) -> None:
    """/reserved-memory — 从 Linux 中移除物理区域 (no-map).

    用于 enclave 内存池等需要从内核线性映射中完全移除的区域,
    与 PMP 隔离策略保持一致: 内核无法访问这些物理地址 -> 不会分配其中页面.
    """
    if not ranges:
        return
    sw.begin_node("reserved-memory")
    sw.property_u32("#address-cells", 2)
    sw.property_u32("#size-cells", 2)
    sw.property("ranges", b"")
    for idx, (base, size) in enumerate(ranges):
        sw.begin_node(f"reserved-{idx}@{base:x}")
        sw.property("no-map", b"")
        sw.property("reg", _encode_reg_4mib(base, size))
        sw.end_node()
    sw.end_node()  # reserved-memory


def _dtb_memory(
    sw: libfdt.FdtSw,
    cfg: PlatformConfig,
) -> None:
    """/memory — 主 RAM 区间."""
    sw.begin_node("memory")
    sw.property_string("device_type", "memory")
    sw.property("reg", _encode_reg_4mib(cfg.ram_base, cfg.ram_size))
    sw.end_node()  # memory


def _dtb_clint(
    sw: libfdt.FdtSw,
    cfg: PlatformConfig,
    cpu_phandles: list[int],
) -> None:
    """clint@ — 每 hart 两条中断 <&cpu_intc 3 (MSI) &cpu_intc 7 (MTI)>."""
    clint_base = cfg.periph.clint_base
    sw.begin_node(f"clint@{clint_base:x}")
    sw.property_string("compatible", "riscv,clint0")
    sw.property("reg", _encode_reg_4mib(clint_base, 0x10000))
    ie_cells: list[int] = []
    for ph in cpu_phandles:
        ie_cells.extend([ph, 3, ph, 7])  # M-SW-IRQ=3, M-TIMER-IRQ=7
    sw.property(
        "interrupts-extended",
        struct.pack(">" + "I" * len(ie_cells), *ie_cells),
    )
    sw.end_node()  # clint


def _dtb_plic(
    sw: libfdt.FdtSw,
    cfg: PlatformConfig,
    plic: PLIC | None,
    plic_phandle: int | None,
    cpu_phandles: list[int],
) -> None:
    """plic@ — 每 hart 双 context (M+S): <&cpu_intc 11>, <&cpu_intc 9>.

    context 序号: 0=hart0 M, 1=hart0 S, 2=hart1 M, 3=hart1 S, ...
    OpenSBI 初始化 M-context, Linux 认领 S-context.
    """
    if plic is None:
        return
    plic_base = cfg.periph.plic_base
    sw.begin_node(f"plic@{plic_base:x}")
    sw.property_u32("phandle", plic_phandle)
    sw.property_string("compatible", "riscv,plic0")
    sw.property_u32("#interrupt-cells", 1)
    sw.property("interrupt-controller", b"")  # 布尔属性必须零长度
    sw.property("reg", _encode_reg_4mib(plic_base, plic.size))
    sw.property_u32("riscv,ndev", 128)
    plic_ie: list[int] = []
    for ph in cpu_phandles:
        # 标准双 context/hart: 先 M-ext(11) 后 S-ext(9)。枚举顺序即 context 序号
        # (0=hart0 M, 1=hart0 S, ...), Linux 认领奇数 (S) context。
        plic_ie.extend([ph, 11, ph, 9])  # M-EXT-IRQ=11, S-EXT-IRQ=9
    sw.property(
        "interrupts-extended",
        struct.pack(">" + "I" * len(plic_ie), *plic_ie),
    )
    sw.end_node()  # plic


# ============================================================
#  IMSIC 节点 — 拆分为 M/S 两个独立节点以匹配 QEMU virt DT 布局.
#
#  QEMU 针对每个 privilege level 各创建一个 IMSIC 节点, 且
#  interrupts-extended 仅含该 level 对应的中断号码 (M=11, S=9)。
#
#  为什么必须拆分:
#  Linux 内核的 imsic_get_parent_hartid() 在 S 模式 (RV_IRQ_EXT=9) 下
#  遍历 interrupts-extended 计数 — 首条非 SEIP(9) 条目即返回 -EINVAL,
#  导致 nr_parent_irqs=0 并终止驱动初始化。 合并在单一节点的
#  <MEIP, SEIP, MEIP, SEIP> 使循环在 index=0 遇到 MEIP(11) 后立即停止。
#  拆分为两个节点后:
#    - M 节点 (<MEIP,...>) → S 核计数 0 → 失败 → imsic 释放
#    - S 节点 (<SEIP,...>) → S 核计数 N → 成功 → eidelivery 正常置 1
# ============================================================

# M/S 模式对应的 CPU intc 中断号.
_IMSIC_IRQ_M = 11  # MEIP
_IMSIC_IRQ_S = 9   # SEIP

_PAGE_STRIDE_DTB = 0x1000  # IMSIC_MMIO_PAGE_SZ


def _dtb_imsic_file(
    sw: libfdt.FdtSw,
    cfg: PlatformConfig,
    imsic: IMSIC | None,
    phandle: int | None,
    cpu_phandles: list[int],
    *,
    cpu_intc_irq: int,  # _IMSIC_IRQ_M or _IMSIC_IRQ_S
    base_addr: int,
    size: int,
) -> None:
    """生成单个 IMSIC 节点 — 参数化 M/S 以消除重复."""
    if imsic is None:
        return
    sw.begin_node(f"imsic@{base_addr:x}")
    sw.property_u32("phandle", phandle)
    sw.property_string("compatible", "riscv,imsics")
    sw.property_u32("#interrupt-cells", 0)
    sw.property_u32("#msi-cells", 0)
    sw.property("interrupt-controller", b"")
    sw.property("msi-controller", b"")
    sw.property_u32("riscv,num-ids", 255)  # must satisfy (num_ids & 63) == 63
    sw.property_u32("riscv,guest-index-bits", 0)
    # 不设 hart-index-bits — QEMU virt DT 同样省略此属性.
    # 内核默认 stride = PAGE_SIZE << guest_index_bits = 0x1000,
    # 必须与 decode_imsic_addr 的实际布局一致 (连续排列, 每个 hart 一页).
    # hart-index-bits != 0 会使内核算出更大的 stride (0x2000/0x4000...),
    # 导致写 seteipnum 偏移到错误的 hart → IPI 丢失 → SMP 死锁.
    sw.property("reg", _encode_reg_4mib(base_addr, size))
    # 该 privilege level 对应的中断 — 每种各一个 per hart.
    ie: list[int] = []
    for ph in cpu_phandles:
        ie.extend([ph, cpu_intc_irq])
    sw.property(
        "interrupts-extended",
        struct.pack(">" + "I" * len(ie), *ie),
    )
    sw.end_node()  # imsic


def _dtb_aplic(
    sw: libfdt.FdtSw,
    cfg: PlatformConfig,
    aplic: APLIC | None,
    aplic_phandle: int | None,
    imsic_phandle: int | None,
) -> None:
    """aplic@ — 有线→MSI 桥, 经 msi-parent=<&imsic> 投递.

    APLIC does NOT have interrupts-extended because all interrupts are
    delivered as MSIs through the IMSIC identified by msi-parent.
    This matches QEMU's virt machine DT and the RISC-V AIA specification.
    """
    if aplic is None:
        return
    p = cfg.periph
    aplic_base = p.aplic_base
    sw.begin_node(f"aplic@{aplic_base:x}")
    sw.property_u32("phandle", aplic_phandle)
    sw.property_string("compatible", "riscv,aplic")
    sw.property_u32("#interrupt-cells", 2)
    sw.property("interrupt-controller", b"")
    sw.property_u32("riscv,num-sources", 128)
    sw.property("reg", _encode_reg_4mib(aplic_base, aplic.size))
    # msi-parent: route through IMSIC
    sw.property_u32("msi-parent", imsic_phandle)
    sw.end_node()  # aplic


def _dtb_uart(
    sw: libfdt.FdtSw,
    cfg: PlatformConfig,
    uart: UART | None,
    ext_irq_handle_prop: int | None,
    clock_phandle: int,
) -> None:
    """serial@ (+ fixed-clock) — SiFive NS16550A 兼容串口."""
    if uart is None:
        return
    p = cfg.periph
    # 固定时钟 — 使 SiFive 串口驱动在无 CCF 时钟控制器的 DTB 中也能 probe.
    # 内核 CONFIG_COMMON_CLK=y 时 devm_clk_get() 若找不到 clocks 属性
    # 返回 EPROBE_DEFER, 驱动永不匹配; 提供 fixed-clock 解决此问题.
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
    # APLIC expects #interrupt-cells=2: <source flags>
    # PLIC  expects #interrupt-cells=1: <irq>
    irq_parent = ext_irq_handle_prop if ext_irq_handle_prop is not None else 1
    if PYREMU_AIA:
        sw.property(
            "interrupts-extended",
            struct.pack(">III", irq_parent, 10, _IRQ_TYPE_LEVEL_HIGH),
        )
    else:
        sw.property(
            "interrupts-extended",
            struct.pack(">II", irq_parent, 10),
        )
    sw.end_node()  # serial


def _dtb_simple_devices(
    sw: libfdt.FdtSw,
    cfg: PlatformConfig,
    spi: SPI | None,
    i2c_gen: I2C | None,
    gpio: GPIO | None,
    virtio_blk: VirtIOBlock | None,
) -> None:
    """spi/i2c/gpio/virtio — 无中断的简单 MMIO 节点."""
    p = cfg.periph
    if spi is not None:
        sw.begin_node(f"spi@{p.spi_base:x}")
        sw.property_string("compatible", "pyremu,spi0")
        sw.property("reg", _encode_reg_4mib(p.spi_base, 0x1000))
        sw.end_node()  # spi
    if i2c_gen is not None:
        sw.begin_node(f"i2c@{p.i2c_base:x}")
        sw.property_string("compatible", "pyremu,i2c0")
        sw.property("reg", _encode_reg_4mib(p.i2c_base, 0x1000))
        sw.end_node()  # i2c
    if gpio is not None:
        sw.begin_node(f"gpio@{p.gpio_base:x}")
        sw.property_string("compatible", "pyremu,gpio0")
        sw.property("reg", _encode_reg_4mib(p.gpio_base, 0x1000))
        sw.end_node()  # gpio
    if virtio_blk is not None:
        sw.begin_node(f"virtio@{p.virtio_blk_base:x}")
        sw.property_string("compatible", "virtio,mmio")
        sw.property("reg", _encode_reg_4mib(p.virtio_blk_base, 0x200))
        if PYREMU_AIA:
            # APLIC #interrupt-cells=2: <source flags>
            sw.property("interrupts", struct.pack(">II", VIRTIO_BLK_IRQ, _IRQ_TYPE_LEVEL_HIGH))
        else:
            # PLIC #interrupt-cells=1: <irq>
            sw.property_u32("interrupts", VIRTIO_BLK_IRQ)
        sw.end_node()  # virtio


def _dtb_watchdog(
    sw: libfdt.FdtSw,
    cfg: PlatformConfig,
) -> None:
    """watchdog@ — 多 hart 停滞检测设备 (DTB 可见)."""
    wdog_base = getattr(cfg.periph, "watchdog_base", None) or 0x1000_4000
    sw.begin_node(f"watchdog@{wdog_base:x}")
    sw.property_string("compatible", "pyremu,hart-watchdog-1.0")
    sw.property("reg", _encode_reg_4mib(wdog_base, 0x100))
    sw.property_u32("pyremu,num-harts", cfg.num_harts)
    sw.end_node()  # watchdog


def _dtb_crng(
    sw: libfdt.FdtSw,
    cfg: PlatformConfig,
    crng: CRNG | None,
) -> None:
    """crng@ — 模拟随机数生成器 (运行期熵源, DTB 可见)."""
    if crng is None:
        return
    base = cfg.periph.crng_base
    sw.begin_node(f"crng@{base:x}")
    sw.property_string("compatible", "pyremu,crng")
    sw.property("reg", _encode_reg_4mib(base, 0x1000))
    sw.end_node()  # crng


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
    watchdog: HartWatchdog | None = None,
    crng: CRNG | None = None,
    plic: PLIC | None = None,
    imsic: IMSIC | None = None,
    aplic: APLIC | None = None,
    bootargs: str | None = None,
    initrd: Initrd | None = None,
    reserved_ranges: list[tuple[int, int]] | None = None,
) -> bytes:
    """基于平台配置和外设映射构建完整的 DTB blob.

    Args:
        cfg: 平台配置, 描述 CPU 拓扑、内存布局和 ISA 特性.
        uart: UART 外设实例 (None 则跳过 UART 节点).
        spi: SPI 外设实例.
        i2c_gen: I2C 外设实例.
        gpio: GPIO 外设实例.
        virtio_blk: virtio-blk 外设实例.
        crng: 模拟随机数生成器实例 (None 则跳过 crng 节点).
        plic: PLIC 中断控制器实例 (legacy 模式).
        imsic: IMSIC 中断控制器实例 (AIA 模式).
        aplic: APLIC 有线→MSI 桥实例 (AIA 模式).
        bootargs: 内核命令行参数, 写入 /chosen/bootargs.
        reserved_ranges: (base, size) 列表, 生成 /reserved-memory no-map 子节点.

    Returns:
        完整的 DTB blob 字节串, 可直接写入 RAM 供固件使用.
    """
    is_aia = cfg.interrupt_mode == InterruptMode.AIA

    sw = libfdt.FdtSw(8192)
    sw.finish_reservemap()

    # ---- 根节点 ----
    sw.begin_node("")
    sw.property_u32("#address-cells", 2)
    sw.property_u32("#size-cells", 2)
    sw.property_string("compatible", "pyremu,riscv64")
    sw.property_string("model", f"pyremu,{cfg.isa}")

    _dtb_chosen(sw, cfg, uart, bootargs, initrd)
    _dtb_aliases(sw, cfg, uart)
    cpu_phandles = _dtb_cpus(sw, cfg, is_aia=is_aia)
    _dtb_memory(sw, cfg)
    _dtb_reserved_memory(sw, reserved_ranges or [])

    # ---- soc simple-bus — 挂载所有 MMIO 外设 ----
    # phandle 分配: CPU intc 占 [1..num_harts].
    # 随后: AIA → IMSIC_M, IMSIC_S, APLIC 各一; legacy → PLIC.
    page_stride = _PAGE_STRIDE_DTB
    next_phandle = len(cpu_phandles) + 1
    imsic_m_phandle: int | None = None
    imsic_s_phandle: int | None = None
    aplic_phandle: int | None = None
    ext_irq_handle_prop: int | None = None
    if is_aia:
        if imsic is not None:
            imsic_m_phandle = next_phandle
            next_phandle += 1
            imsic_s_phandle = next_phandle
            next_phandle += 1
        if aplic is not None:
            aplic_phandle = next_phandle
            next_phandle += 1
            ext_irq_handle_prop = aplic_phandle
    elif plic is not None:
        ext_irq_handle_prop = next_phandle
        next_phandle += 1

    sw.begin_node("soc")
    sw.property_u32("#address-cells", 2)
    sw.property_u32("#size-cells", 2)
    sw.property_string("compatible", "simple-bus")
    sw.property("ranges", b"")  # 透传
    if ext_irq_handle_prop is not None:
        sw.property_u32("interrupt-parent", ext_irq_handle_prop)

    _dtb_clint(sw, cfg, cpu_phandles)
    if is_aia:
        m_base = cfg.periph.imsic_m_base
        # OpenSBI imsic_data_check 要求 reg size 对齐到
        # 2^hart_index_bits * PAGE_SIZE.  hart_index_bits =
        # ceil(log2(num_harts)), 即 (num_harts-1).bit_length().
        # 非 2 的幂 hart 数 (如 3, 5, 6, 7) 若不补齐,
        # OpenSBI imsic_cold_irqchip_init 失败 → 无 irqchip →
        # sbi_irqchip_process 返回 SBI_ENODEV (-1000).
        if cfg.num_harts > 1:
            hart_index_bits = (cfg.num_harts - 1).bit_length()
        else:
            hart_index_bits = 0
        padded_count = 1 << hart_index_bits  # ≥ num_harts 的 2 的幂
        m_size = padded_count * page_stride
        # M 节点: M-files 范围, MEIP(11) 专用.
        _dtb_imsic_file(sw, cfg, imsic, imsic_m_phandle, cpu_phandles,
                        cpu_intc_irq=_IMSIC_IRQ_M, base_addr=m_base, size=m_size)
        # S 节点: S-files 范围 (M-base + padded_count*0x1000), SEIP(9) 专用.
        _dtb_imsic_file(sw, cfg, imsic, imsic_s_phandle, cpu_phandles,
                        cpu_intc_irq=_IMSIC_IRQ_S,
                        base_addr=m_base + padded_count * page_stride,
                        size=m_size)
        _dtb_aplic(sw, cfg, aplic, aplic_phandle, imsic_s_phandle)
    else:
        _dtb_plic(sw, cfg, plic, ext_irq_handle_prop, cpu_phandles)
    _dtb_uart(sw, cfg, uart, ext_irq_handle_prop, next_phandle)
    _dtb_simple_devices(sw, cfg, spi, i2c_gen, gpio, virtio_blk)
    _dtb_watchdog(sw, cfg)
    _dtb_crng(sw, cfg, crng)

    sw.end_node()  # soc
    sw.end_node()  # root

    return bytes(sw.as_fdt().as_bytearray())
