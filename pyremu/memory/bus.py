#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/09 星期二
# Last modified at 2026/06/10 星期三

"""
共享总线 (Bus) — 统一的物理内存访问层。

Bus 管理:
- 物理 RAM (bytearray), 默认基址 0x8000_0000 (RISC-V 标准内存映射)
- L2 共享缓存 (可选, 继承 CacheBase)
- 内存映射设备 (CLINT, 未来的 IMSIC/APLIC/UART 等)

PMA (Physical Memory Attributes):
  地址空间按硬件属性分为三类区域:
  - Main Memory  : [ram_base, ram_base + ram_size) — 可缓存、支持原子操作、全位宽访问
  - I/O (Device) : 通过 add_device() 注册 — 不可缓存、读写可能有副作用
  - Empty / Hole : 其余地址 — 触发 AccessFault (该行为在 Hart._mem_read/_mem_write 中实现)

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


class AccessFaultError(Exception):
    """PMA 访问违例 — 地址不在任何有效区域 (非内存、非设备)."""

    def __init__(self, addr: int, is_write: bool = False) -> None:
        self.addr = addr
        self.is_write = is_write
        super().__init__(
            f"PMA AccessFault: addr=0x{addr:016x} ({'write' if is_write else 'read'})"
        )


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
        ram_base: int = 0x8000_0000,
        l2_cache=None,  # L2Cache 实例或 None
    ) -> None:
        self._ram = bytearray(ram_size)
        self._ram_base = ram_base
        self._ram_end = ram_base + ram_size
        self._ram_size = ram_size
        self._l2 = l2_cache
        self._devices: dict[int, Device] = {}  # base_addr → device

        # VMA 影子映射: 固件链接在低地址但 RAM 在高地址时,
        # 自动将 [shadow_base, shadow_base+shadow_size) 别名到 RAM 起始处.
        self._shadow_base: int | None = None
        self._shadow_size: int = 0

        # 设备查找缓存 — 按基址排序的 (base, end, device) 列表, 用于二分查找.
        # add_device 后置 None, 首次 _find_device 时重建.
        self._device_cache: list[tuple[int, int, Device]] | None = None

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
        self._device_cache = None  # 下次 _find_device 时重建

    def _find_device(self, addr: int) -> tuple[Device | None, int]:
        """二分查找地址对应的设备及设备内偏移.

        将 O(n) 线性扫描替换为 O(log n) 二分查找.
        缓存按基址排序的 (base, end, device) 列表, 仅在 add_device 后重建.
        """
        if self._device_cache is None:
            self._device_cache = sorted(
                (base, base + dev.size, dev)
                for base, dev in self._devices.items()
            )
        cache = self._device_cache

        # bisect_right — 找到第一个 base > addr 的位置
        lo, hi = 0, len(cache)
        while lo < hi:
            mid = (lo + hi) // 2
            if cache[mid][0] <= addr:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx >= 0:
            base, end, dev = cache[idx]
            if addr < end:
                return dev, addr - base
        return None, 0

    # ----------------------------------------------------------
    #  PMA 检查
    # ----------------------------------------------------------

    def set_ram_shadow(self, base: int, size: int) -> None:
        """将 [base, base+size) 别名到 RAM 起始处.

        用于固件 VMA ≠ PA 的场景: 固件链接在低地址但 RAM 在高地址,
        load_offset 搬迁后, VMA 范围仍需要能访问同一物理 RAM.
        """
        self._shadow_base = base
        self._shadow_size = size

    def is_ram_addr(self, addr: int) -> bool:
        """判断物理地址是否属于主存 (Main Memory)."""
        if self._ram_base <= addr < self._ram_end:
            return True
        if self._shadow_base is not None:
            return self._shadow_base <= addr < self._shadow_base + self._shadow_size
        return False

    def is_device_addr(self, addr: int) -> bool:
        """判断物理地址是否属于 MMIO 设备区域 (不可缓存).

        Hart 在 TLB 插入前调用此方法, 对设备地址跳过缓存。
        """
        for base, dev in self._devices.items():
            if base <= addr < base + dev.size:
                return True
        return False

    def is_valid_addr(self, addr: int) -> bool:
        """PMA 检查: 地址是否在有效物理区域 (主存或 I/O 设备).

        Returns:
            True 若地址属于 Main Memory 或已注册的 I/O Device.
            空洞地址 (Empty/Hole) 返回 False, 上层应触发 AccessFault.
        """
        if self.is_ram_addr(addr):
            return True
        if self.is_device_addr(addr):
            return True
        return False

    @property
    def ram_base(self) -> int:
        return self._ram_base

    @property
    def devices(self) -> dict[int, Device]:
        """返回所有已注册设备 (按基址索引)."""
        return self._devices

    # ----------------------------------------------------------
    #  RAM 直接访问 (绕过 L2, 供 L2 回退和调试使用)
    # ----------------------------------------------------------

    def _ram_offset(self, addr: int, size: int) -> int | None:
        """将地址映射到 RAM 偏移量; 不在任何 RAM 范围则返回 None."""
        if self._ram_base <= addr and addr + size <= self._ram_end:
            return addr - self._ram_base
        if self._shadow_base is not None:
            if self._shadow_base <= addr and addr + size <= self._shadow_base + self._shadow_size:
                return addr - self._shadow_base
        return None

    def _ram_read_direct(self, addr: int, size: int) -> bytes:
        """直接从 RAM 读取 (绕过 L2 缓存).

        *addr* 超出 RAM 范围时返回全 0 (模拟未映射物理地址).
        """
        off = self._ram_offset(addr, size)
        if off is None:
            return b"\x00" * size
        return bytes(self._ram[off : off + size])

    def _ram_write_direct(self, addr: int, data: bytes) -> None:
        """直接写入 RAM (绕过 L2 缓存).

        *addr* 超出 RAM 范围时静默丢弃 (由上层 PMA 检查保证不会发生).
        """
        off = self._ram_offset(addr, len(data))
        if off is None:
            return
        self._ram[off : off + len(data)] = data

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
    #  安全读写 (不抛异常, 供调试器等外部调用方使用)
    # ----------------------------------------------------------

    def try_read(self, addr: int, size: int) -> bytes | None:
        """安全读取物理内存 — 失败返回 None 而非抛异常.

        与 read() 的区别: 将设备/L2 的异常转换为 None 返回值,
        调用方无需 try/except. 适用于调试器、内存 dump 等 best-effort 场景.

        Args:
            addr: 物理地址.
            size: 读取字节数 (1/2/4/8).

        Returns:
            成功时返回 bytes, 失败 (设备异常/访问越界) 返回 None.
        """
        try:
            return self.read(addr, size)
        except Exception:
            return None

    def try_write(self, addr: int, data: bytes) -> bool:
        """安全写入物理内存 — 返回 bool 表示成功与否.

        与 write() 的区别: 将设备/L2 的异常转换为 False 返回值,
        调用方无需 try/except. 适用于回滚、内存补丁等 best-effort 场景.

        Args:
            addr: 物理地址.
            data: 写入数据.

        Returns:
            True 表示写入成功, False 表示失败 (设备异常/访问越界).
        """
        try:
            self.write(addr, data)
            return True
        except Exception:
            return False

    # ----------------------------------------------------------
    #  属性
    # ----------------------------------------------------------

    @property
    def ram_size(self) -> int:
        return self._ram_size

    @property
    def l2(self):
        return self._l2
