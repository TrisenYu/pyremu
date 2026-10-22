#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/09 星期二
# Last modified at 2026/06/10 星期三

"""
共享总线 (Bus) — 统一的物理内存访问层。

Bus 管理:
- 物理 RAM, 以 bytearray 存储, 默认基址 0x8000_0000, 采用 RISC-V 标准内存映射
- 共享 L2 缓存, 可选, 继承 CacheBase
- 内存映射设备, 例如 CLINT, 未来的 IMSIC/APLIC/UART 等

RAM 与其访问路径由两个独立的类承担, Bus 只保留设备表与地址路由:

- DirectRam  — bytearray 与地址窗口, 提供绕过 L2 缓存的读写。L2 缓存把它作为 RAM
               后端, Bus 把它作为与设备无关的直接访问入口。
- MainMem    — RAM 与 L2 缓存组成的访问路径, 不含设备。设备持有本对象的读写方法
               而不持有 Bus 的, 以免 Bus 与设备之间形成环形引用。

PMA (Physical Memory Attributes):
  地址空间按硬件属性分为三类区域:
  - Main Memory  : [ram_base, ram_base + ram_size) — 可缓存、支持原子操作、全位宽访问
  - I/O (Device) : 通过 add_device() 注册 — 不可缓存、读写可能有副作用
  - Empty / Hole : 其余地址 — 触发 AccessFault (该行为在 Hart._mem_read/_mem_write 中实现)

关键设计: 设备 MMIO 地址具有读写副作用 (读可能改变硬件状态),
必须排除于所有缓存机制之外 — 不走 TLB 缓存, 不走 L2 缓存, 直通设备。
"""

from abc import ABC, abstractmethod
import ctypes
from functools import partialmethod

from pyremu._native import (
    bus_read_ram,
    bus_write_ram,
    native_available,
)
from pyremu.configs_gen import PYREMU_NO_L2

# Module-level cache for native pointers to bytearray buffers.
# Stored here (not on Bus instances) so that deepcopy() never
# touches ctypes from_buffer() objects, which segfault the GC.
_ram_native_ptrs: dict[int, int] = {}  # id(bytearray) -> raw pointer

NO_L2 = PYREMU_NO_L2 == 1


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
    """PMA 访问违规 地址不在任何有效区域 (非内存、非设备)."""

    def __init__(self, addr: int, is_write: bool = False) -> None:
        self.addr = addr
        self.is_write = is_write
        super().__init__(
            f"PMA AccessFault: addr=0x{addr:016x} ({'write' if is_write else 'read'})"
        )


class DirectRam:
    """RAM 缓冲区及其地址窗口, 提供绕过 L2 缓存的读写.

    L2 缓存以本对象的方法作为 RAM 后端, Bus 以本对象的方法作为与设备无关的直接
    访问入口. 本对象只持有 bytearray 与地址窗口, 不持有 Bus 与任何设备, 故上述
    两处持有本对象的方法时不会与 Bus 之间形成环形引用.
    """

    def __init__(
        self,
        ram_size: int = 128 * 1024 * 1024,
        ram_base: int = 0x8000_0000,
    ) -> None:
        self._ram = bytearray(ram_size)
        self._ram_base = ram_base
        self._ram_end = ram_base + ram_size
        self._ram_size = ram_size

        # PIE 装载偏移 — 固件按链接地址给出的取值落在 [pie_ofs_base,
        # pie_ofs_base + pie_ofs_size) 内的部分, 经 _pie_ofs_to_ram 换算为 RAM 偏移.
        # 由 apply_pie_ofs 在装载固件时写入一次; 此后每一次 RAM 访问与加速执行引擎的
        # 每一轮都要读取它, 故不能留作装载函数内的局部量. 字节数为 0 表示未施加装载
        # 偏移, 此时基址取值无意义 (固件的链接基址可以取 0).
        self._pie_ofs_base: int = 0
        self._pie_ofs_size: int = 0

        self._use_native: bool = native_available()
        self._ram_native_ptr: int = 0
        self._read_out_ptr: int = 0
        # RAM 直读窗口, 供 bus_read_ram / bus_write_ram 一次性传入:
        # (缓冲区首地址, 字节数, 基址, 影子窗口基址, 影子窗口字节数).
        # 影子窗口由 apply_pie_ofs 登记, 故该元组在彼处重建.
        self._ram_win: tuple[int, int, int, int, int] = (
            0,
            self._ram_size,
            self._ram_base,
            0,
            0,
        )
        if self._use_native:
            # 两处地址用 ctypes.addressof 取得;
            # 取地址不用 ctypes.cast, 后者会与所转换的对象构成引用环.
            self._ram_native_ptr = ctypes.addressof(ctypes.c_uint8.from_buffer(self._ram))
            self._read_out = ctypes.c_uint64(0)
            self._read_out_ptr = ctypes.addressof(self._read_out)
            self._ram_win = (
                self._ram_native_ptr,
                self._ram_size,
                self._ram_base,
                0,
                0,
            )

    def apply_pie_ofs(self, link_base: int, link_size: int) -> None:
        """登记 PIE 固件的链接地址区间 [link_base, link_base + link_size).

        固件链接在低地址而 RAM 在高地址时, 各段被写在链接地址加上装载偏移的物理
        地址上; 固件按链接地址给出的取值落在本区间内, 与那些段指同一段字节.
        装载固件时调用一次; 此后每一次 RAM 访问经 _pie_ofs_to_ram 使用本区间,
        加速执行引擎的每一轮经 pie_ofs_pair 取本区间的两个端点, 故本区间在该调用返回
        之后不能失效.
        """
        self._pie_ofs_base = link_base
        self._pie_ofs_size = link_size
        self._ram_win = (
            self._ram_native_ptr,
            self._ram_size,
            self._ram_base,
            link_base,
            link_size,
        )

    def _pie_ofs_to_ram(self, addr: int, size: int) -> int | None:
        """把落在 PIE 链接地址区间内的 [addr, addr + size) 换算为 RAM 偏移.

        区间之外的地址返回 None, 由调用方按未命中处理.
        """
        if self._pie_ofs_size == 0:
            return None
        if self._addr_in_range(addr, size, self._pie_ofs_base, self._pie_ofs_size):
            return addr - self._pie_ofs_base
        return None

    def pie_ofs_pair(self) -> tuple[int, int]:
        """返回 PIE 链接地址区间的基址与字节数; 未施加装载偏移时两者均为 0."""
        return self._pie_ofs_base, self._pie_ofs_size

    def is_ram_addr(self, addr: int) -> bool:
        """判断物理地址是否属于主存 (Main Memory)."""
        if self._ram_base <= addr < self._ram_end:
            return True
        return self._pie_ofs_to_ram(addr, 1) is not None

    @staticmethod
    def _addr_in_range(addr: int, size: int, base: int, extent: int) -> bool:
        """判定区间 [*addr*, *addr* + *size*) 是否落在 [*base*, *base* + *extent*) 之内."""
        return base <= addr and addr + size <= base + extent

    def _ram_offset(self, addr: int, size: int) -> int | None:
        """将地址映射到 RAM 偏移量; 不在任何 RAM 范围则返回 None."""
        if self._addr_in_range(addr, size, self._ram_base, self._ram_size):
            return addr - self._ram_base
        return self._pie_ofs_to_ram(addr, size)

    def read(self, addr: int, size: int) -> bytes:
        """直接从 RAM 读取 (绕过 L2 缓存)."""
        # 越界地址在此处即行返回, 不进入 Rust 快路径. 调用方可能传入任意
        # 64 位取值, 其中包含由栈帧指针一类故障现场算出的负数地址
        # (如 bt 读 fp-16 得到 -1); 这类取值经 ctypes 转为无符号数后不再
        # 落在任何窗口内, 由本判断挡在 FFI 之外.
        off = self._ram_offset(addr, size)
        if off is None:
            return b"\x00" * size

        # Rust native fast path: only for regular access sizes (1/2/4/8).
        # L2 cache line fills (64 B) would overflow the 8-byte read-out buffer.
        if (
            self._use_native
            and self._ram_native_ptr
            and size in (1, 2, 4, 8)
            and bus_read_ram(
                self._ram_win,
                addr,
                size,
                self._read_out_ptr,
            )
        ):
            return ctypes.string_at(self._read_out_ptr, size)

        return bytes(self._ram[off : off + size])

    def write(self, addr: int, data: bytes) -> None:
        """直接写入 RAM (绕过 L2 缓存).

        *addr* 超出 RAM 范围时静默丢弃 (由上层 PMA 检查保证不会发生).
        """
        data_len = len(data)
        # 越界地址不进入 Rust 快路径, 理由同 read
        ofs = self._ram_offset(addr, data_len)
        if ofs is None:
            return

        # Rust native fast path: only for regular access sizes (1/2/4/8).
        # L2 cache line fills produce larger writes via the Python path.
        if (
            self._use_native
            and self._ram_native_ptr
            and data_len in (1, 2, 4, 8)
            and bus_write_ram(
                self._ram_win,
                addr,
                data,
                data_len,
            )
        ):
            return

        self._ram[ofs : ofs + data_len] = data


class MainMem:
    """主存访问路径 — RAM 与共享 L2 缓存, 不含内存映射设备.

    设备的描述符链与数据环位于客户 RAM 内, 设备经本对象的方法读写它们. 路由与
    Bus.read / Bus.write 的非设备分支一致; 设备 MMIO 由 Bus 按其设备表分派,
    不经本对象.
    """

    # 最大 CPU store 大小 (RISC-V: sd = 8 bytes).
    # 超过此阈值的写入必定是 DMA/批量传输, 应绕过 L2 缓存直写 bytearray,
    # 避免缓存行逐出覆盖加速用动态链接库的直接写入.
    _MAX_CPU_STORE = 8

    def __init__(self, ram: DirectRam, l2_cache=None) -> None:
        self._ram = ram
        self._l2 = l2_cache

    def _bypass_l2(self, addr: int, data_len: int) -> bool:
        """RAM 地址是否应绕过 L2 缓存, 直写 bytearray.

        调用动态链接库加速时，直接修改 bytearray, 不经过 L2 缓存。若 Python 侧 L2 缓存行
        覆盖同一 PA 的字节, 写命中时合并旧数据并回写, 会污染 Rust
        的修改, 表现为页表 PTE 或栈数据被覆写为旧值。

        绕过条件:
          - PYREMU_NO_L2=1: 排查专用, RAM 访问全部绕过 L2 缓存以隔离问题。
          - data_len > 8: DMA/批量传输不经过 CPU 缓存。
        """
        return (NO_L2 or data_len > self._MAX_CPU_STORE) and self._ram.is_ram_addr(addr)

    def read(self, addr: int, size: int) -> bytes:
        """主存读: 先取 L2 缓存, 未命中时经 RAM 后端取得."""
        if NO_L2 and self._ram.is_ram_addr(addr):
            return self._ram.read(addr, size)

        if self._l2 is not None:
            return self._l2.bus_read(addr, size)

        return self._ram.read(addr, size)

    def write(self, addr: int, data: bytes) -> None:
        """主存写: 批量与 DMA 传输直写 bytearray, 其余经 L2 缓存写入."""
        if self._bypass_l2(addr, len(data)):
            self._ram.write(addr, data)
            return

        if self._l2 is not None:
            self._l2.bus_write(addr, data)
            return

        self._ram.write(addr, data)


class Bus:
    """共享物理总线.

    所有 hart 通过 Bus.read() / Bus.write() 访问物理资源。
    地址路由顺序: 设备 MMIO 优先, 其次是 L2 缓存, 最后通过访问 RAM 得到。

    设备地址的读写保证:
    - 不经过 L2 缓存, 直接访问设备
    - 由调用方 mem_check_aux.translate_addr 保证这些地址不插入 TLB
    - 通过 is_device_addr() 供外部判断
    """

    def read_ulen(self, addr: int, len: int) -> int | None:
        ret = self.try_read(addr, len)
        return int.from_bytes(ret, "little", signed=False) if ret else None

    read_u64 = partialmethod(read_ulen, len=8)
    read_u32 = partialmethod(read_ulen, len=4)
    read_u16 = partialmethod(read_ulen, len=2)
    read_u08 = partialmethod(read_ulen, len=1)

    def __init__(
        self,
        ram_size: int = 128 * 1024 * 1024,
        ram_base: int = 0x8000_0000,
        l2_cache=None,  # L2Cache 实例或 None
    ) -> None:
        self._ram_direct = DirectRam(ram_size=ram_size, ram_base=ram_base)
        # RAM 缓冲区别名 — 调试器、批量加载与加速执行引擎按同一 bytearray 直接读写
        self._ram = self._ram_direct._ram
        self._l2 = l2_cache
        self._mem = MainMem(self._ram_direct, l2_cache)
        self._devices: dict[int, Device] = {}  # base_addr -> device

        # 已发生的 TLB 刷新次数. 使既有翻译条目失效的动作各算一次, 由执行该动作的
        # 一方经 note_tlb_flush 加一. 加速执行引擎在两次调用之间保留该取值, 故引擎
        # 之外发生的刷新对引擎内既有的条目同样生效; 反过来, 引擎内发生的刷新经每轮
        # 结束时的写回使 Python 侧的两个 TLB 作废.
        self.tlb_flush_cnt = ctypes.c_uint64(0)

        # 设备查找缓存 — 按基址排序的 (base, end, device) 列表, 用于二分查找.
        # add_device 后置 None, 首次 _find_device 时重建.
        self._device_cache: list[tuple[int, int, Device]] | None = None

        # 若 L2 缓存存在, 注入 RAM 后端回调. L2 缓存只缓存 RAM, 不缓存设备.
        # 后端取 DirectRam 的方法而不取 Bus 的: Bus 持有 L2 缓存并按主存访问路径
        # 调用它, 后端再持有 Bus 的方法会使二者之间形成环形引用.
        # 判定用 None 比较: CacheBase 定义了 __len__ (返回有效条目数), 空缓存的
        # 真值为假, 用真值判定会使后端回调在本处被跳过.
        if self._l2 is not None:
            self._l2.set_ram_backend(
                read_fn=self._ram_direct.read,
                write_fn=self._ram_direct.write,
            )

    def note_tlb_flush(self) -> None:
        """登记一次 TLB 刷新, 使两处既有的翻译条目全部作废.

        取自增不减的计数而不是一个"已失效"标志: 标志上的清除与置位并发时, 清除
        可能覆盖紧随其后的置位, 使某个 hart 漏掉一次失效.
        """
        self.tlb_flush_cnt.value += 1

    # ----------------------------------------------------------
    #  设备管理
    # ----------------------------------------------------------

    def add_device(self, base_addr: int, device: Device) -> None:
        """注册一个内存映射设备"""
        device.base_addr = base_addr
        self._devices[base_addr] = device
        self._device_cache = None  # 下次 _find_device 时重建

    def _find_device(self, addr: int) -> tuple[Device | None, int]:
        """二分查找地址对应的设备及设备内偏移"""
        if self._device_cache is None:
            self._device_cache = sorted(
                (base, base + dev.size, dev) for base, dev in self._devices.items()
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

    def apply_pie_ofs(self, link_base: int, link_size: int) -> None:
        """登记 PIE 固件的链接地址区间; 见 DirectRam.apply_pie_ofs."""
        self._ram_direct.apply_pie_ofs(link_base, link_size)

    def pie_ofs_pair(self) -> tuple[int, int]:
        """返回 PIE 链接地址区间的基址与字节数; 未施加装载偏移时两者均为 0."""
        return self._ram_direct.pie_ofs_pair()

    def is_ram_addr(self, addr: int) -> bool:
        """判断物理地址是否属于主存 (Main Memory)."""
        return self._ram_direct.is_ram_addr(addr)

    def is_device_addr(self, addr: int) -> bool:
        """判断物理地址是否属于 MMIO 设备区域 (不可缓存).

        Hart 在 TLB 插入前调用此方法, 设备地址不插入 TLB。
        O(log n) 二分查找, 复用 _find_device 的排序缓存。
        """
        dev, _ = self._find_device(addr)
        return dev is not None

    def is_valid_addr(self, addr: int) -> bool:
        """PMA 检查: 地址是否在有效物理区域 (主存或 I/O 设备).

        Returns:
            True 若地址属于 Main Memory 或已注册的 I/O Device.
            空洞地址 (Empty/Hole) 返回 False, 上层应触发 AccessFault.
        """
        return self.is_ram_addr(addr) or self.is_device_addr(addr)

    @property
    def ram_base(self) -> int:
        return self._ram_direct._ram_base

    @property
    def ram_size(self) -> int:
        return self._ram_direct._ram_size

    @property
    def main_mem(self) -> MainMem:
        """主存访问路径 (RAM 与 L2 缓存), 供设备读写客户 RAM 使用.

        设备持有本对象的读写方法而不持有 Bus 的读写方法: Bus 持有设备表,
        设备再持有 Bus 的方法会使二者之间形成环形引用.
        """
        return self._mem

    @property
    def devices(self) -> dict[int, Device]:
        """返回所有已注册设备 (按基址索引)."""
        return self._devices

    # ----------------------------------------------------------
    #  RAM 直接访问, 绕过 L2 缓存, 供 L2 回退和调试使用
    # ----------------------------------------------------------

    def _ram_read_direct(self, addr: int, size: int) -> bytes:
        """直接从 RAM 读取 (绕过 L2 缓存)."""
        return self._ram_direct.read(addr, size)

    def _ram_write_direct(self, addr: int, data: bytes) -> None:
        """直接写入 RAM (绕过 L2 缓存).

        *addr* 超出 RAM 范围时静默丢弃 (由上层 PMA 检查保证不会发生).
        """
        self._ram_direct.write(addr, data)

    def write_ram_direct(self, addr: int, data: bytes) -> None:
        """绕过 L2 缓存和设备 MMIO, 直接向 RAM 写入数据.

        用于固件/内核镜像等大批量数据加载场景.
        调用方需确保写入完成后 L2 缓存中无目标地址的脏行
        (加载前 hart 尚未运行, 故天然安全).
        """
        self._ram_direct.write(addr, data)

    def flush_l2(self) -> int:
        """将 L2 缓存中全部脏行回写到 RAM (bytearray).

        调用动态链接库加速前必须调用, 确保 Rust 从 bytearray 读取时
        能看到 Python 侧通过 bus.write() 写入的全部数据.

        Returns:
            回写的缓存行数; 无 L2 缓存时返回 0.
        """
        return self._l2.flush_all() if self._l2 is not None else 0

    def invalidate_l2(self) -> int:
        """使 L2 缓存全部行失效 (脏行先回写).

        调用动态链接库加速执行后调用, 确保 Python 侧后续通过 L2 缓存读取时
        不会命中 Rust 直接修改 bytearray 前的过时缓存行.

        Returns:
            失效的缓存行数; 无 L2 缓存时返回 0.
        """
        return self._l2.invalidate_all() if self._l2 is not None else 0

    # ----------------------------------------------------------
    #  总线读写 (外部接口 — Hart 的 _mem_read_phy / _mem_write_phy 使用)
    # ----------------------------------------------------------

    def read(self, addr: int, size: int) -> bytes:
        """总线读: 设备 MMIO 直通优先, 其余地址交给主存访问路径."""
        dev, offset = self._find_device(addr)
        if dev:
            return dev.read(offset, size)

        return self._mem.read(addr, size)

    def write(self, addr: int, data: bytes) -> None:
        """总线写: 设备 MMIO 直通优先, 其余地址交给主存访问路径."""
        dev, offset = self._find_device(addr)
        if dev:
            dev.write(offset, data)
            return

        self._mem.write(addr, data)

    def try_read(self, addr: int, size: int) -> bytes:
        """安全读取物理内存 — 读取不产生异常.

        本函数与 read() 是同一入口, 自身不捕获异常. 未落入任何设备与 RAM 窗口的
        地址由 _ram_read_direct 返回 size 个零字节;
        设备读实现抛出的异常沿 read() 向上传播.
        适用于调试器、内存 dump 等尽力而为的场景.

        Args:
            addr: 物理地址.
            size: 读取字节数 (1/2/4/8).

        Returns:
            读到的字节串.
        """
        return self.read(addr, size)

    def try_write(self, addr: int, data: bytes) -> bool:
        """安全写入物理内存 — 返回 bool 表示成功与否.

        与 write() 的区别: 将设备/L2 缓存的异常转换为 False 返回值,
        调用方无需 try/except. 适用于回滚、内存补丁等尽力而为的场景.

        Args:
            addr: 物理地址.
            data: 写入数据.

        Returns:
            True 表示写入成功, False 表示失败 (设备异常/访问越界).
        """
        try:
            self.write(addr, data)
        except Exception:
            return False
        return True

    # ----------------------------------------------------------
    #  属性
    # ----------------------------------------------------------

    @property
    def l2(self):
        return self._l2
