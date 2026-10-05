#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""virtio-blk MMIO 块设备 (virtio v1.2 规范).

只承载设备语义: feature 集合、0x100 起的配置空间 (容量)、virtio-blk 请求的处理。
MMIO 寄存器堆与 virtqueue 遍历由 [virtio_mmio](pyremu/peripheral/virtio_mmio.py)
提供。

特性:
- virtio v1.0+ MMIO 传输层 (modern, 非 legacy)
- 单 virtqueue (queue 0)
- 支持 VIRTIO_BLK_T_IN / VIRTIO_BLK_T_OUT (读写)
- 磁盘镜像以 raw 格式存储, 通过 pread/pwrite 访问
"""

from __future__ import annotations

from collections.abc import Callable
import os

from pyremu.core.diag import diag
from pyremu.memory.bus import Device
from pyremu.peripheral.virtio_mmio import (
    VIRTIO_F_RING_EVENT_IDX,
    VIRTIO_F_RING_INDIRECT_DESC,
    VIRTIO_F_VERSION_1,
    VIRTIO_MMIO_CONFIG_GENERATION,
    VIRTIO_MMIO_DEVICE_FEATURES,
    VIRTIO_MMIO_DEVICE_FEATURES_SEL,
    VIRTIO_MMIO_DEVICE_ID,
    VIRTIO_MMIO_DRIVER_FEATURES,
    VIRTIO_MMIO_DRIVER_FEATURES_SEL,
    VIRTIO_MMIO_INTERRUPT_ACK,
    VIRTIO_MMIO_INTERRUPT_STATUS,
    VIRTIO_MMIO_MAGIC_VALUE,
    VIRTIO_MMIO_QUEUE_DESC_HIGH,
    VIRTIO_MMIO_QUEUE_DESC_LOW,
    VIRTIO_MMIO_QUEUE_DEVICE_HIGH,
    VIRTIO_MMIO_QUEUE_DEVICE_LOW,
    VIRTIO_MMIO_QUEUE_DRIVER_HIGH,
    VIRTIO_MMIO_QUEUE_DRIVER_LOW,
    VIRTIO_MMIO_QUEUE_NOTIFY,
    VIRTIO_MMIO_QUEUE_NUM,
    VIRTIO_MMIO_QUEUE_NUM_MAX,
    VIRTIO_MMIO_QUEUE_READY,
    VIRTIO_MMIO_QUEUE_SEL,
    VIRTIO_MMIO_SIZE,
    VIRTIO_MMIO_STATUS,
    VIRTIO_MMIO_VENDOR_ID,
    VIRTIO_MMIO_VERSION,
    VIRTIO_STATUS_ACKNOWLEDGE,
    VIRTIO_STATUS_DEVICE_NEEDS_RESET,
    VIRTIO_STATUS_DRIVER,
    VIRTIO_STATUS_DRIVER_OK,
    VIRTIO_STATUS_FAILED,
    VIRTIO_STATUS_FEATURES_OK,
    VirtioMMIO,
    VirtqDesc,
    VirtQueue,
    VRING_DESC_F_INDIRECT,
    VRING_DESC_F_NEXT,
    VRING_DESC_F_WRITE,
)
from pyremu.utils.mask import mask32

# 传输层的常量与类型在本模块再导出一份, 使既有调用方沿用原导入路径。
__all__ = [
    "SECTOR_SIZE",
    "VIRTIO_BLK_CFG_CAPACITY",
    "VIRTIO_BLK_IRQ",
    "VIRTIO_BLK_S_IOERR",
    "VIRTIO_BLK_S_OK",
    "VIRTIO_BLK_S_UNSUPP",
    "VIRTIO_BLK_T_DISCARD",
    "VIRTIO_BLK_T_FLUSH",
    "VIRTIO_BLK_T_IN",
    "VIRTIO_BLK_T_OUT",
    "VIRTIO_BLK_T_WRITE_ZEROES",
    "VIRTIO_F_RING_EVENT_IDX",
    "VIRTIO_F_RING_INDIRECT_DESC",
    "VIRTIO_F_VERSION_1",
    "VIRTIO_MMIO_CONFIG_GENERATION",
    "VIRTIO_MMIO_DEVICE_FEATURES",
    "VIRTIO_MMIO_DEVICE_FEATURES_SEL",
    "VIRTIO_MMIO_DEVICE_ID",
    "VIRTIO_MMIO_DRIVER_FEATURES",
    "VIRTIO_MMIO_DRIVER_FEATURES_SEL",
    "VIRTIO_MMIO_INTERRUPT_ACK",
    "VIRTIO_MMIO_INTERRUPT_STATUS",
    "VIRTIO_MMIO_MAGIC_VALUE",
    "VIRTIO_MMIO_QUEUE_DESC_HIGH",
    "VIRTIO_MMIO_QUEUE_DESC_LOW",
    "VIRTIO_MMIO_QUEUE_DEVICE_HIGH",
    "VIRTIO_MMIO_QUEUE_DEVICE_LOW",
    "VIRTIO_MMIO_QUEUE_DRIVER_HIGH",
    "VIRTIO_MMIO_QUEUE_DRIVER_LOW",
    "VIRTIO_MMIO_QUEUE_NOTIFY",
    "VIRTIO_MMIO_QUEUE_NUM",
    "VIRTIO_MMIO_QUEUE_NUM_MAX",
    "VIRTIO_MMIO_QUEUE_READY",
    "VIRTIO_MMIO_QUEUE_SEL",
    "VIRTIO_MMIO_SIZE",
    "VIRTIO_MMIO_STATUS",
    "VIRTIO_MMIO_VENDOR_ID",
    "VIRTIO_MMIO_VERSION",
    "VIRTIO_STATUS_ACKNOWLEDGE",
    "VIRTIO_STATUS_DEVICE_NEEDS_RESET",
    "VIRTIO_STATUS_DRIVER",
    "VIRTIO_STATUS_DRIVER_OK",
    "VIRTIO_STATUS_FAILED",
    "VIRTIO_STATUS_FEATURES_OK",
    "VirtIOBlock",
    "VRING_DESC_F_INDIRECT",
    "VRING_DESC_F_NEXT",
    "VRING_DESC_F_WRITE",
]

# virtio-blk 配置空间 (offset 0x100+)
VIRTIO_BLK_CFG_CAPACITY = 0x000

# ============================================================
#  virtio-blk 请求类型
# ============================================================

VIRTIO_BLK_T_IN = 0
VIRTIO_BLK_T_OUT = 1
VIRTIO_BLK_T_FLUSH = 4
VIRTIO_BLK_T_DISCARD = 11
VIRTIO_BLK_T_WRITE_ZEROES = 13

# ============================================================
#  virtio-blk 响应状态
# ============================================================

VIRTIO_BLK_S_OK = 0
VIRTIO_BLK_S_IOERR = 1
VIRTIO_BLK_S_UNSUPP = 2

# ============================================================
#  扇区大小 (字节)
# ============================================================

SECTOR_SIZE = 512

# ============================================================
#  PLIC 中断源号 (对齐 QEMU virt: virtio-mmio 设备 IRQ 从 1 起)
# ============================================================

VIRTIO_BLK_IRQ = 1

# 请求头 (virtio_blk_outhdr) 的固定长度
_BLK_OUTHDR_SIZE = 16

# 一条请求的描述符链长度: 请求头、数据缓冲区、状态字节
_BLK_CHAIN_DESC_NUM = 3

# 本设备支持的 feature (64-bit)
#
# VIRTIO_F_RING_EVENT_IDX (bit 29) 刻意不声明: 若声明, 受调试程序的内核驱动会走
# event-index 通知抑制路径 needs_kick = vring_need_event(avail_event, new, old),
# 而设备侧从不更新 used ring 的 avail_event 字段 (始终为 0), 使第二个及之后的
# 缓冲区不再写 QueueNotify, 内核永远等不到 I/O 完成。不声明时驱动退回 flags 模式
# (检查 VRING_USED_F_NO_NOTIFY), 该 flag 设备侧同样从不置位, 故每次添加缓冲区都会 kick。
#
# VIRTIO_F_RING_INDIRECT_DESC (bit 28) 同样不声明: 若声明, 驱动可使用间接描述符,
# 但描述符链遍历不实现间接表, 遇到带 INDIRECT 标志的描述符会因缺失 NEXT 标志而失败,
# 内核取不到 ext4 superblock 后 VFS panic。
_DEVICE_FEATURES = VIRTIO_F_VERSION_1


class VirtIOBlock(Device):
    """virtio-blk MMIO 块设备.

    通过 MMIO 寄存器接口暴露一个 virtio-blk 设备, 受调试程序可以读写磁盘镜像 (raw 格式).

    Usage::

        vblk = VirtIOBlock(
            image_path="disk.img",
            mem_read=bus.read,
            mem_write=bus.write,
        )
        bus.add_device(0x1000_5000, vblk)
    """

    def __init__(
        self,
        image_path: str,
        mem_read: Callable[[int, int], bytes],
        mem_write: Callable[[int, bytes], None],
        queue_size_max: int = 256,
        on_irq: Callable[[int, bool], None] | None = None,
        irq: int = 0,
        read_only: bool = False,
    ) -> None:
        self.base_addr = 0
        self.size = VIRTIO_MMIO_SIZE

        # 打开磁盘镜像 (不存在则创建)。
        # 只读回退: 显式 read_only 或对无写权限的镜像 (如 root 所有的 ext4),
        # 以 O_RDONLY 打开并置 _read_only; 写请求将被静默忽略 (见 _do_write)。
        if not os.path.exists(image_path):
            with open(image_path, "wb") as f:
                f.truncate(0)
        if read_only:
            self._fd = os.open(image_path, os.O_RDONLY)
        else:
            try:
                self._fd = os.open(image_path, os.O_RDWR)
            except PermissionError:
                self._fd = os.open(image_path, os.O_RDONLY)
                read_only = True
        self._read_only = read_only
        self._disk_size = os.lseek(self._fd, 0, os.SEEK_END)

        # 中断投递: 完成中断经 on_irq(irq, level) 上报 (irq=0 或未给出 on_irq 时不投递)。
        # 调用方应传 Emulator.raise_device_irq, 该入口除置位中断控制器外还通知加速执行
        # 引擎并唤醒 WFI 阻塞的 hart; 直接调 plic.set_irq 会略过这两步, 使设备中断在
        # 引擎的单轮加速执行内不可见, 且无法唤醒已进入 WFI 的 hart。
        self._on_irq = on_irq
        self._irq = irq

        self.mmio = VirtioMMIO(
            device_id=0x2,  # block device
            device_features=_DEVICE_FEATURES,
            num_queues=1,
            queue_num_max=queue_size_max,
            mem_read=mem_read,
            mem_write=mem_write,
            config_read=self._config_read,
            handle_request=self._handle_request,
            on_irq=self._set_irq_level,
            chain_max_desc=_BLK_CHAIN_DESC_NUM,
        )

    # ---- Device 接口 ----

    @property
    def read_only(self) -> bool:
        """镜像是否以只读方式打开. 为真时写请求被静默忽略."""
        return self._read_only

    def read(self, offset: int, size: int) -> bytes:
        return self.mmio.read(offset, size)

    def write(self, offset: int, data: bytes) -> None:
        self.mmio.write(offset, data)

    def process_queue(self, max_descriptors: int = 0) -> bool:
        """处理队列 0 中待处理的请求. 语义见 VirtioMMIO.process_queue."""
        return self.mmio.process_queue(0, max_descriptors)

    # ---- 中断 ----

    def _set_irq_level(self, level: bool) -> None:
        """按中断线电平操作中断源. irq=0 或未给出 on_irq 时为空操作."""
        if self._on_irq is not None and self._irq:
            self._on_irq(self._irq, level)

    def raise_irq(self) -> None:
        """拉高本设备中断源 (完成通知)."""
        self.mmio.raise_irq()

    def lower_irq_if_idle(self) -> None:
        """中断已被确认且无残留状态位时, 拉低中断源."""
        self.mmio.lower_irq_if_idle()

    # ---- virtio-blk 配置空间 ----

    @property
    def capacity_sectors(self) -> int:
        """磁盘容量, 以扇区计. 即 virtio-blk 配置空间的首个字段."""
        return self._disk_size // SECTOR_SIZE

    def _config_read(self, local: int) -> int:
        """读取 virtio-blk 设备特定配置. 参数为配置空间内的偏移."""
        sectors = self.capacity_sectors
        if local == VIRTIO_BLK_CFG_CAPACITY:
            return mask32(sectors)
        if local == VIRTIO_BLK_CFG_CAPACITY + 4:
            return mask32(sectors >> 32)
        return 0

    # ---- 请求处理 ----

    def _handle_request(self, _queue: VirtQueue, chain: list[VirtqDesc]) -> tuple[bool, int]:
        """处理一个 virtio-blk 请求的描述符链.

        链的布局: 请求头 (virtio_blk_outhdr, 16 字节) 与数据缓冲区与状态字节。

        返回:
            (是否成功, 记入 used ring 的写入字节数)。受调试程序的 virtio-blk 驱动不读取
            used ring 的写入字节数, 故此处恒返回 0, 与抽取前的行为一致.
        """
        hdr_desc = chain[0]
        if hdr_desc.length < _BLK_OUTHDR_SIZE:
            return False, 0
        if len(chain) < _BLK_CHAIN_DESC_NUM:
            return False, 0

        data_desc = chain[1]
        status_addr = chain[2].addr

        hdr_data = self._mem_read_chunk(hdr_desc.addr, _BLK_OUTHDR_SIZE)
        req_type = int.from_bytes(hdr_data[0:4], "little")
        # ioprio = int.from_bytes(hdr_data[4:8], "little")  # 未使用
        sector = int.from_bytes(hdr_data[8:16], "little")

        if req_type == VIRTIO_BLK_T_IN:
            diag("virtio", f"REQ read  sector={sector} len={data_desc.length}")
            ok = self._do_read(sector, data_desc.addr, data_desc.length)
        elif req_type == VIRTIO_BLK_T_OUT:
            diag("virtio", f"REQ write sector={sector} len={data_desc.length}")
            ok = self._do_write(sector, data_desc.addr, data_desc.length)
        elif req_type == VIRTIO_BLK_T_FLUSH:
            ok = self._do_flush()
        elif req_type in (VIRTIO_BLK_T_DISCARD, VIRTIO_BLK_T_WRITE_ZEROES):
            # 不支持的请求类型
            self._mem_write(status_addr, bytes([VIRTIO_BLK_S_UNSUPP]))
            return True, 0
        else:
            ok = False

        status = VIRTIO_BLK_S_OK if ok else VIRTIO_BLK_S_IOERR
        self._mem_write(status_addr, bytes([status]))

        return ok, 0

    def _do_read(self, sector: int, buf_pa: int, buf_len: int) -> bool:
        """从扇区 *sector* 读取数据到受调试程序的物理地址 *buf_pa*."""
        offset = sector * SECTOR_SIZE
        if offset >= self._disk_size:
            data = b"\x00" * buf_len
        else:
            data = os.pread(self._fd, buf_len, offset)
        # 写入前探查目标页, 残留的 PAGE_POISON 说明页面分配时未被清零。
        try:
            self._mem_read_chunk(buf_pa & ~0xFFF, 64)
        except Exception:
            pass
        self.mmio.write_phys(buf_pa, data)
        return True

    def _do_write(self, sector: int, buf_pa: int, buf_len: int) -> bool:
        """从受调试程序的物理地址 *buf_pa* 写入数据到扇区 *sector*."""
        # 只读镜像: 静默忽略写并返回成功, 避免内核报 I/O 错。
        # 用户已确认暂不支持写磁盘 (root=/dev/vda ro 不会写数据块)。
        if self._read_only:
            return True
        data = self._mem_read_chunk(buf_pa, buf_len)
        offset = sector * SECTOR_SIZE
        os.pwrite(self._fd, data, offset)
        # 更新磁盘大小 (如果写到了文件末尾之后)
        end = offset + len(data)
        self._disk_size = max(self._disk_size, end)
        return True

    def _do_flush(self) -> bool:
        """将文件内容同步到磁盘."""
        if self._read_only:
            return True
        try:
            os.fsync(self._fd)
        except OSError:
            return False
        return True

    # ---- 内存访问 ----

    def _mem_read_chunk(self, pa: int, size: int) -> bytes:
        return self.mmio.read_phys(pa, size)

    def _mem_write(self, pa: int, data: bytes) -> None:
        self.mmio.write_phys(pa, data)
