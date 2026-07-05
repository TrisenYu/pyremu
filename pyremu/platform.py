#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
平台配置 — 描述 CPU 拓扑、内存布局、外设映射和 ISA 特性.

配置与预设分离: 每个预设是独立的工厂函数, 返回填充了合理默认值的
PlatformConfig 实例. 也支持从 JSON / TOML / YAML 文件反序列化.

Usage:
    # 使用预设
    cfg = PlatformConfig.sifive_u54()
    emu = Emulator(cfg)

    # 从配置文件加载
    cfg = PlatformConfig.from_json("platforms/virt.json")
    cfg = PlatformConfig.from_toml("platforms/virt.toml")
    cfg = PlatformConfig.from_yaml("platforms/virt.yaml")

    # 程序化定制
    cfg = PlatformConfig.sifive_u54()
    cfg.periph.uart_base = 0x2000_0000
    emu = Emulator(cfg)
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PeripheralConfig:
    """外设 MMIO 基址 (0 = 禁用)."""

    uart_base: int = 0x1000_0000
    spi_base: int = 0x1000_1000
    i2c_base: int = 0x1000_2000
    gpio_base: int = 0x1000_3000
    clint_base: int = 0x0200_0000
    plic_base: int = 0x0C00_0000  # PLIC 基址 (SiFive standard)
    virtio_blk_base: int = 0  # 0 = 禁用


@dataclass
class PlatformConfig:
    """模拟平台完整配置.

    字段:
        num_harts: hart 数量 (1–8).
        ram_size: 物理 RAM 大小 (字节).
        ram_base: 物理 RAM 起始地址.
        prog_cnt: 程序计数器 — 上电复位入口地址.
        l2_size: L2 缓存大小 (字节).
        isa: RISC-V ISA 字符串 (例: ``'rv64ima'``).
        timebase_freq: mtime 计数器频率 (Hz).
        periph: 外设基址配置.
    """

    num_harts: int = 1
    ram_size: int = 128 * 1024 * 1024  # 128 MiB
    ram_base: int = 0x8000_0000
    prog_cnt: int = 0x8000_0000
    l2_size: int = 256 * 1024  # 256 KiB

    isa: str = "rv64imac_sstc_zicsr_zifencei"
    timebase_freq: int = 10_000_000  # 10 MHz
    pmp_entries: int = 64  # PMP 条目数 (0=禁用, 8/16/64 常见)
    disk_image: str | None = None  # virtio-blk 磁盘镜像路径, None=不挂载

    periph: PeripheralConfig = field(default_factory=PeripheralConfig)

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
    ) -> PlatformConfig:
        """从嵌套字典构建模拟器配置.

        ``periph`` 子字典映射到 PeripheralConfig 字段,
        其余键映射到 PlatformConfig 字段.
        """
        periph_dict: dict[str, Any] = d.get("periph", {})
        periph_fields = set(PeripheralConfig.__dataclass_fields__)
        periph = PeripheralConfig(
            **{k: v for k, v in periph_dict.items() if k in periph_fields}
        )

        plat_fields = set(cls.__dataclass_fields__) - {"periph"}
        plat_kw: dict[str, Any] = {k: v for k, v in d.items() if k in plat_fields}
        return cls(periph=periph, **plat_kw)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> PlatformConfig:
        """从 JSON 文件加载配置."""
        with open(path, "r") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_toml(
        cls,
        path: str | Path,
    ) -> PlatformConfig:
        """从 TOML 文件加载配置."""
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> PlatformConfig:
        """从 YAML 文件加载配置."""
        with open(path, "r") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    # ---- 预设平台 ----
    @classmethod
    def sifive_u54(cls) -> PlatformConfig:
        """SiFive Freedom U54 风格布局.

        DRAM @ 0x8000_0000, CLINT @ 0x0200_0000,
        外设挂在 0x1000_0000 区域.
        """
        return cls(
            num_harts=4,
            isa="rv64imac_sstc_zicsr_zifencei",
            periph=PeripheralConfig(
                uart_base=0x1000_0000,
                spi_base=0x1001_0000,
                i2c_base=0x1002_0000,
                gpio_base=0x1006_0000,
                clint_base=0x0200_0000,
            ),
        )

    @classmethod
    def qemu_virt(cls) -> PlatformConfig:
        """QEMU RISC-V virt 风格布局."""
        return cls(
            num_harts=1,
            isa="rv64imac_sstc_zicsr_zifencei",
            periph=PeripheralConfig(
                uart_base=0x1000_0000,
                spi_base=0x1000_1000,
                i2c_base=0x1000_2000,
                gpio_base=0x1000_3000,
                clint_base=0x0200_0000,
            ),
        )

    @classmethod
    def minimal(cls) -> PlatformConfig:
        """最小化单核平台 — 仅 CLINT + UART.

        内存从 0x0000_0000 起始. 适用于裸金属测试和 CI.
        """
        return cls(
            num_harts=1,
            ram_base=0x0000_0000,
            prog_cnt=0x0000_0000,
            periph=PeripheralConfig(
                uart_base=0x1000_0000,
                spi_base=0,
                i2c_base=0,
                gpio_base=0,
                clint_base=0x0200_0000,
            ),
        )
