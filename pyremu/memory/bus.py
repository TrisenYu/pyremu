#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/09 星期二

"""
共享总线 (Bus) — 统一的物理内存访问层。

Bus 管理:
- 物理 RAM (bytearray)
- L2 共享缓存 (可选, 继承 CacheBase)
- 内存映射设备 (CLINT, 未来的 IMSIC/APLIC/UART 等)

关键设计: 设备 MMIO 地址具有读写副作用 (读可能改变硬件状态),
必须排除于所有缓存机制之外 — 不走 TLB 缓存, 不走 L2 缓存, 直通设备。
"""

from abc import ABC, abstractmethod


class Device(ABC):
    """内存映射设备基类.

    子类: CLINT, (未来) IMSIC, UART, PLIC 等.

    设备地址范围 [base_addr, base_addr + size) 内的所有读写都直通设备,
    不经过任何缓存.
    """

    base_addr: int = 0
    size: int = 0

    @abstractmethod
    def read(self, offset: int, size: int) -> bytes:
        """从设备读取.

        Args:
            offset: 相对设备基址的偏移.
            size: 读取字节数.
        """

    @abstractmethod
    def write(self, offset: int, data: bytes) -> None:
        """向设备写入.

        Args:
            offset: 相对设备基址的偏移.
            data: 写入数据.
        """


class Bus:
    """共享物理总线.

    所有 hart 通过 Bus.read() / Bus.write() 访问物理资源。
    地址路由顺序: 设备 MMIO → L2 缓存 → RAM 直读。

    设备地址的读写保证:
    - 不经过 L2 缓存 (直通设备)
    - 由调用方 (Hart._translate_full) 负责不在 TLB 中缓存这些地址
    - 通过 is_device_addr() 供外部判断
    """

    def __init__(
        self,
        ram_size: int = 128 * 1024 * 1024,
        l2_cache=None,  # L2Cache 实例或 None
    ) -> None:
        self._ram = bytearray(ram_size)
        self._ram_size = ram_size
        self._l2 = l2_cache
        self._devices: dict[int, Device] = {}  # base_addr → device

        # 若 L2 缓存存在, 注入 RAM 后端回调 (L2 只缓存 RAM, 不缓存设备)
        if self._l2 is not None:
            self._l2.set_ram_backend(
                read_fn=self._ram_read_direct,
                write_fn=self._ram_write_direct,
            )

    # ----------------------------------------------------------
    #  设备管理
    # ----------------------------------------------------------

    def add_device(self, base_addr: int, device: Device) -> None:
        """注册一个内存映射设备."""
        device.base_addr = base_addr
        self._devices[base_addr] = device

    def _find_device(self, addr: int) -> tuple[Device | None, int]:
        """查找地址对应的设备及设备内偏移."""
        for base, dev in self._devices.items():
            if base <= addr < base + dev.size:
                return dev, addr - base
        return None, 0

    def is_device_addr(self, addr: int) -> bool:
        """判断物理地址是否属于 MMIO 设备区域 (不可缓存).

        Hart 在 TLB 插入前调用此方法, 对设备地址跳过缓存。
        """
        for base, dev in self._devices.items():
            if base <= addr < base + dev.size:
                return True
        return False

    @property
    def devices(self) -> dict[int, Device]:
        """返回所有已注册设备 (按基址索引)."""
        return self._devices

    # ----------------------------------------------------------
    #  RAM 直接访问 (绕过 L2, 供 L2 回退和调试使用)
    # ----------------------------------------------------------

    def _ram_read_direct(self, addr: int, size: int) -> bytes:
        """直接从 RAM 读取 (绕过 L2 缓存)."""
        if addr + size > self._ram_size:
            # 越界访问返回 0 (模拟未映射物理地址)
            return b"\x00" * size
        return bytes(self._ram[addr : addr + size])

    def _ram_write_direct(self, addr: int, data: bytes) -> None:
        """直接写入 RAM (绕过 L2 缓存)."""
        if addr + len(data) > self._ram_size:
            return
        for i, b in enumerate(data):
            self._ram[addr + i] = b

    # ----------------------------------------------------------
    #  总线读写 (外部接口 — Hart 的 _mem_read_phy / _mem_write_phy 使用)
    # ----------------------------------------------------------

    def read(self, addr: int, size: int) -> bytes:
        """总线读: 设备 (直通) → L2 缓存 → RAM.

        设备地址绕过 L2, 直接读设备寄存器 (有副作用).
        """
        # 设备地址: 直通, 不走缓存
        dev, offset = self._find_device(addr)
        if dev is not None:
            return dev.read(offset, size)

        # L2 缓存 (仅缓存 RAM 地址)
        if self._l2 is not None:
            return self._l2.bus_read(addr, size)

        return self._ram_read_direct(addr, size)

    def write(self, addr: int, data: bytes) -> None:
        """总线写: 设备 (直通) → L2 缓存 → RAM.

        设备地址绕过 L2, 直接写设备寄存器 (有副作用).
        """
        # 设备地址: 直通, 不走缓存
        dev, offset = self._find_device(addr)
        if dev is not None:
            dev.write(offset, data)
            return

        # L2 缓存 (仅缓存 RAM 地址)
        if self._l2 is not None:
            self._l2.bus_write(addr, data)
            return

        self._ram_write_direct(addr, data)

    # ----------------------------------------------------------
    #  属性
    # ----------------------------------------------------------

    @property
    def ram_size(self) -> int:
        return self._ram_size

    @property
    def l2(self):
        return self._l2
