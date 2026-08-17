#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
virtio-blk MMIO 块设备 (virtio v1.2 规范, MMIO 传输层).

实现 virtio-blk 设备的 MMIO 寄存器接口和 virtqueue 描述符处理,
Guest 可通过该设备读写磁盘镜像 (raw 格式).

特性:
- virtio v1.0+ MMIO 传输层 (modern, 非 legacy)
- 单 virtqueue (queue 0)
- 支持 VIRTIO_BLK_T_IN / VIRTIO_BLK_T_OUT (读写)
- 磁盘镜像以 raw 格式存储, 通过 pread/pwrite 访问

寄存器布局 (OASIS virtio v1.2 MMIO):
  0x000 | 4 | R | MagicValue         | 0x74726976 ("virt")
  0x004 | 4 | R | Version            | 0x2
  0x008 | 4 | R | DeviceID           | 0x2 (block)
  0x00c | 4 | R | VendorID           | 0x0
  0x010 | 4 | R | DeviceFeatures     | bits
  0x014 | 4 | W | DeviceFeaturesSel  | page selector
  0x020 | 4 | W | DriverFeatures     | bits
  0x024 | 4 | W | DriverFeaturesSel  | page selector
  0x030 | 4 | W | QueueSel           | queue index
  0x034 | 4 | R | QueueNumMax        | max queue entries
  0x038 | 4 | W | QueueNum           | set queue size
  0x044 | 4 | W | QueueReady         | activate queue
  0x050 | 4 | W | QueueNotify        | new buffer available
  0x060 | 4 | R | InterruptStatus    | bit 0: used buffer
  0x064 | 4 | W | InterruptACK       | clear interrupt
  0x070 | 4 | R/W | Status           | device status
  0x080 | 4 | W | QueueDescLow       | desc table (low 32)
  0x084 | 4 | W | QueueDescHigh      | desc table (high 32)
  0x090 | 4 | W | QueueDriverLow     | driver area (low 32)
  0x094 | 4 | W | QueueDriverHigh    | driver area (high 32)
  0x0a0 | 4 | W | QueueDeviceLow     | device area (low 32)
  0x0a4 | 4 | W | QueueDeviceHigh    | device area (high 32)
  0x0fc | 4 | R | ConfigGeneration   | config change counter
  0x100 | 8 | R | Capacity           | total 512-byte sectors (u64 LE)
"""

from __future__ import annotations

from collections.abc import Callable
import os
import struct
import sys
from typing import TYPE_CHECKING

from pyremu.core.diag import diag
from pyremu.memory.bus import Device
from pyremu.utils.mask import mask16, mask32

if TYPE_CHECKING:
    from typing import Any as _Any

# ============================================================
#  MMIO 寄存器偏移
# ============================================================

VIRTIO_MMIO_MAGIC_VALUE = 0x000
VIRTIO_MMIO_VERSION = 0x004
VIRTIO_MMIO_DEVICE_ID = 0x008
VIRTIO_MMIO_VENDOR_ID = 0x00C
VIRTIO_MMIO_DEVICE_FEATURES = 0x010
VIRTIO_MMIO_DEVICE_FEATURES_SEL = 0x014
VIRTIO_MMIO_DRIVER_FEATURES = 0x020
VIRTIO_MMIO_DRIVER_FEATURES_SEL = 0x024
VIRTIO_MMIO_QUEUE_SEL = 0x030
VIRTIO_MMIO_QUEUE_NUM_MAX = 0x034
VIRTIO_MMIO_QUEUE_NUM = 0x038
VIRTIO_MMIO_QUEUE_READY = 0x044
VIRTIO_MMIO_QUEUE_NOTIFY = 0x050
VIRTIO_MMIO_INTERRUPT_STATUS = 0x060
VIRTIO_MMIO_INTERRUPT_ACK = 0x064
VIRTIO_MMIO_STATUS = 0x070
VIRTIO_MMIO_QUEUE_DESC_LOW = 0x080
VIRTIO_MMIO_QUEUE_DESC_HIGH = 0x084
VIRTIO_MMIO_QUEUE_DRIVER_LOW = 0x090
VIRTIO_MMIO_QUEUE_DRIVER_HIGH = 0x094
VIRTIO_MMIO_QUEUE_DEVICE_LOW = 0x0A0
VIRTIO_MMIO_QUEUE_DEVICE_HIGH = 0x0A4
VIRTIO_MMIO_CONFIG_GENERATION = 0x0FC

# virtio-blk 配置空间 (offset 0x100+)
VIRTIO_BLK_CFG_CAPACITY = 0x000

# 整个 MMIO 区域大小
VIRTIO_MMIO_SIZE = 0x200

# ============================================================
#  设备状态位
# ============================================================

VIRTIO_STATUS_ACKNOWLEDGE = 1
VIRTIO_STATUS_DRIVER = 2
VIRTIO_STATUS_DRIVER_OK = 4
VIRTIO_STATUS_FEATURES_OK = 8
VIRTIO_STATUS_DEVICE_NEEDS_RESET = 0x40
VIRTIO_STATUS_FAILED = 0x80

# ============================================================
#  Feature bits
# ============================================================

VIRTIO_F_VERSION_1 = 1 << 32
VIRTIO_F_RING_INDIRECT_DESC = 1 << 28
# VIRTIO_F_RING_EVENT_IDX (bit 29) — 刻意不声明.
# 若声明此 feature, 客机驱动走 event-index 通知抑制路径:
#   needs_kick = vring_need_event(avail_event, new, old)
# 但 device 侧从未更新 used ring 的 avail_event 字段 (始终为 0),
# 导致第二个及之后的 buffer 不再写 QueueNotify ->内核永远等不到 I/O 完成.
# 不声明此 feature 时驱动退回到 flags 模式 (检查 VRING_USED_F_NO_NOTIFY),
# 该 flag 我们也从不置位, 因此每次添加 buffer 都会 kick.
VIRTIO_F_RING_EVENT_IDX = 1 << 29
VIRTIO_BLK_F_SIZE_MAX = 1 << 1
VIRTIO_BLK_F_SEG_MAX = 1 << 2
VIRTIO_BLK_F_BLK_SIZE = 1 << 6

# 本设备支持的 feature (64-bit)
# VIRTIO_F_RING_INDIRECT_DESC (bit 28) —
# 若声明, 客机驱动可用 indirect 描述符 (一层跳转), 但 _process_descriptor_chain
# 未实现 indirect 表遍历, 遇到 INDIRECT flag 的 desc 会因缺失 NEXT flag 而 return
# False — 内核拿不到 ext4 superblock ->VFS panic.
# 与 VIRTIO_F_RING_EVENT_IDX 同模式: 声明 feature 但未实现 ->误引导客机 ->移除以退避.
_DEVICE_FEATURES = (
    VIRTIO_F_VERSION_1
)

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
#  virtqueue 描述符标志
# ============================================================

VRING_DESC_F_NEXT = 1
VRING_DESC_F_WRITE = 2
VRING_DESC_F_INDIRECT = 4

# 描述符大小: u64 addr + u32 len + u16 flags + u16 next = 16 bytes
_VRING_DESC_SIZE = 16

# ============================================================
#  扇区大小 (字节)
# ============================================================

SECTOR_SIZE = 512

# ============================================================
#  PLIC 中断源号 (对齐 QEMU virt: virtio-mmio 设备 IRQ 从 1 起)
# ============================================================

VIRTIO_BLK_IRQ = 1


class VirtIOBlock(Device):
    """virtio-blk MMIO 块设备.

    通过 MMIO 寄存器接口暴露一个 virtio-blk 设备,
    Guest 可以读写磁盘镜像 (raw 格式).

    Usage::

        vblk = VirtIOBlock(
            image_path="disk.img",
            mem_read=bus.read,
            mem_write=bus.write,
        )
        bus.add_device(0x1000_8000, vblk)
    """

    def __init__(
        self,
        image_path: str,
        mem_read: Callable[[int, int], bytes],
        mem_write: Callable[[int, bytes], None],
        queue_size_max: int = 256,
        plic: _Any = None,
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

        # PLIC 集成: 完成中断经 plic.set_irq(irq, True) 投递 (irq=0 时不接 PLIC)。
        self._plic = plic
        self._irq = irq

        # 内存访问回调 — 用于读写 Guest 物理内存中的 virtqueue 描述符
        self._mem_read = mem_read
        self._mem_write = mem_write

        # ---- MMIO 寄存器 ----
        self._device_features_sel: int = 0
        self._driver_features_sel: int = 0
        self._driver_features: int = 0
        self._queue_sel: int = 0
        self._queue_num: int = queue_size_max
        self._queue_ready: bool = False
        self._status: int = 0
        self._interrupt_status: int = 0

        # 队列地址 (64-bit)
        self._queue_desc: int = 0
        self._queue_driver: int = 0
        self._queue_device: int = 0

        # 上次看到的可用环索引 (用于检测新请求)
        self._last_avail_idx: int = 0

        self._queue_num_max: int = queue_size_max

    # ---- Device 接口 ----

    def read(self, offset: int, size: int) -> bytes:
        if size not in (1, 2, 4, 8):
            return b"\x00" * size
        val = self._mmio_read(offset)
        return val.to_bytes(size, "little")

    def write(self, offset: int, data: bytes) -> None:
        size = len(data)
        if size not in (1, 2, 4):
            return
        val = int.from_bytes(data, "little")
        self._mmio_write(offset, val, size)

    # ---- MMIO 读 ----

    def _mmio_read(self, offset: int) -> int:
        """读取 MMIO 寄存器 (4 字节, 内部处理对齐)."""
        if offset == VIRTIO_MMIO_MAGIC_VALUE:
            return 0x74726976  # "virt"
        if offset == VIRTIO_MMIO_VERSION:
            return 0x2
        if offset == VIRTIO_MMIO_DEVICE_ID:
            return 0x2  # block device
        if offset == VIRTIO_MMIO_VENDOR_ID:
            return 0x0

        if offset == VIRTIO_MMIO_DEVICE_FEATURES:
            sel = self._device_features_sel
            return mask32(_DEVICE_FEATURES >> (sel * 32))

        if offset == VIRTIO_MMIO_QUEUE_NUM_MAX:
            return self._queue_num_max

        if offset == VIRTIO_MMIO_INTERRUPT_STATUS:
            return self._interrupt_status

        if offset == VIRTIO_MMIO_STATUS:
            return self._status

        if offset == VIRTIO_MMIO_CONFIG_GENERATION:
            return 0  # 配置不会变

        # ---- virtio-blk 配置空间 (0x100+) ----
        if offset >= 0x100:
            return self._config_read(offset)

        # 未实现的寄存器, 返回 0
        return 0

    # ---- MMIO 写 ----

    def _mmio_write(self, offset: int, val: int, size: int) -> None:
        """写入 MMIO 寄存器."""
        if offset == VIRTIO_MMIO_DEVICE_FEATURES_SEL:
            self._device_features_sel = val

        elif offset == VIRTIO_MMIO_DRIVER_FEATURES:
            sel = self._driver_features_sel
            mask_low = mask32(val) << (sel << 5)
            self._driver_features = (self._driver_features & ~(0xFFFF_FFFF << (sel * 32))) | mask_low

        elif offset == VIRTIO_MMIO_DRIVER_FEATURES_SEL:
            self._driver_features_sel = val

        elif offset == VIRTIO_MMIO_QUEUE_SEL:
            self._queue_sel = val

        elif offset == VIRTIO_MMIO_QUEUE_NUM:
            self._queue_num = min(val, self._queue_num_max)

        elif offset == VIRTIO_MMIO_QUEUE_READY:
            self._queue_ready = val != 0
            if self._queue_ready:
                self._last_avail_idx = 0

        elif offset == VIRTIO_MMIO_QUEUE_NOTIFY:
            if val == 0 and self._queue_ready:
                self._process_queue()

        elif offset == VIRTIO_MMIO_INTERRUPT_ACK:
            self._interrupt_status &= ~val
            self._lower_irq_if_idle()

        elif offset == VIRTIO_MMIO_STATUS:
            # 写 0 -> 设备重置
            if val == 0:
                self._reset()
            else:
                self._status = val

        elif offset == VIRTIO_MMIO_QUEUE_DESC_LOW:
            self._queue_desc = (self._queue_desc & 0xFFFF_FFFF_0000_0000) | mask32(val)

        elif offset == VIRTIO_MMIO_QUEUE_DESC_HIGH:
            self._queue_desc = mask32(self._queue_desc) | (mask32(val) << 32)

        elif offset == VIRTIO_MMIO_QUEUE_DRIVER_LOW:
            self._queue_driver = (self._queue_driver & 0xFFFF_FFFF_0000_0000) | mask32(val)

        elif offset == VIRTIO_MMIO_QUEUE_DRIVER_HIGH:
            self._queue_driver = mask32(self._queue_driver) | (mask32(val) << 32)

        elif offset == VIRTIO_MMIO_QUEUE_DEVICE_LOW:
            self._queue_device = (self._queue_device & 0xFFFF_FFFF_0000_0000) | mask32(val)

        elif offset == VIRTIO_MMIO_QUEUE_DEVICE_HIGH:
            self._queue_device = mask32(self._queue_device) | (mask32(val) << 32)

    # ---- virtio-blk 配置空间 ----

    def _config_read(self, offset: int) -> int:
        """读取 virtio-blk 设备特定配置."""
        local = offset - 0x100
        if local == VIRTIO_BLK_CFG_CAPACITY:
            # 低 32-bit 容量 (扇区数)
            return mask32(self._disk_size // SECTOR_SIZE)
        if local == VIRTIO_BLK_CFG_CAPACITY + 4:
            # 高 32-bit 容量
            return mask32((self._disk_size // SECTOR_SIZE) >> 32)
        return 0

    # ---- virtqueue 处理 ----

    def _process_queue(self, max_descriptors: int = 0) -> bool:
        """处理 virtqueue 中的待处理请求.

        Args:
            max_descriptors: 单次最多处理的描述符数 (0=无限制, 用于拆批以响应 Ctrl+C).

        Returns:
            True 如果还有未处理的描述符 (调用方应在检查中断标志后再次调用).
        """
        qnum = self._queue_num
        if qnum == 0 or self._queue_desc == 0 or \
        self._queue_driver == 0 or self._queue_device == 0:
            return False

        # 读取可用环结构
        # [0] flags (u16), [2] idx (u16), [4+] ring (qnum * u16)
        self._read_u16(self._queue_driver)
        avail_idx = self._read_u16(self._queue_driver + 2)

        # 处理可用描述符 (受 max_descriptors 限制, 避免长时间阻塞 Python 线程)
        processed = 0
        while self._last_avail_idx != avail_idx:
            if max_descriptors > 0 and processed >= max_descriptors:
                break

            # 可用环条目: offset 4 + self._last_avail_idx % qnum * 2
            ring_off = 4 + (self._last_avail_idx % qnum) * 2
            desc_head = self._read_u16(self._queue_driver + ring_off)

            # 处理描述符链
            ok = self._process_descriptor_chain(desc_head)
            if not ok:
                # 出错了也不影响后续请求的处理
                pass

            # 写入 used ring 条目
            self._write_used_entry(self._last_avail_idx % qnum, desc_head, 0 if ok else 1)

            self._last_avail_idx += 1
            processed += 1

        if processed > 0:
            # 更新 used ring 的 idx
            self._write_u16(self._queue_device + 2, self._last_avail_idx)
            # 置中断状态位 (bit 0: used buffer notification) 并向 PLIC 拉高中断线
            self._interrupt_status |= 1
            self._raise_irq()
            diag(
                "virtio",
                f"processed {processed} req(s), used_idx={self._last_avail_idx}, "
                f"int_status={self._interrupt_status:#x} -> raise_irq"
            )

        return self._last_avail_idx != avail_idx  # 仍有未处理? 调用方需再次调用

    def _raise_irq(self) -> None:
        """向 PLIC 拉高本设备中断源 (完成通知)。irq=0 或无 PLIC 时为空操作。"""
        if self._plic is not None and self._irq:
            self._plic.set_irq(self._irq, True)

    def _lower_irq_if_idle(self) -> None:
        """中断已被 Guest ACK 且无残留状态位时, 向 PLIC 拉低中断源。"""
        if self._interrupt_status == 0 and self._plic is not None and self._irq:
            self._plic.set_irq(self._irq, False)
            diag("virtio", "lower_irq (acked, int_status=0)")

    def _process_descriptor_chain(self, head: int) -> bool:
        """处理一个描述符链: header -> data -> status."""

        # 读取第一个描述符 -> 请求头 (virtio_blk_outhdr: 16 bytes)
        desc_addr, desc_len, desc_flags, desc_next = self._read_descriptor(head)

        if desc_len < 16:
            return False

        hdr_data = self._mem_read(desc_addr, 16)
        req_type = int.from_bytes(hdr_data[0:4], "little")
        # ioprio = int.from_bytes(hdr_data[4:8], "little")  # 未使用
        sector = int.from_bytes(hdr_data[8:16], "little")

        # 遍历到第二个描述符 -> 数据缓冲区
        if not (desc_flags & VRING_DESC_F_NEXT):
            return False
        desc_addr, desc_len, desc_flags, desc_next = self._read_descriptor(desc_next)

        # 遍历到第三个描述符 -> 状态字节
        if not (desc_flags & VRING_DESC_F_NEXT):
            return False
        status_addr, _status_len, _status_flags, _ = self._read_descriptor(desc_next)

        # 执行 I/O
        if req_type == VIRTIO_BLK_T_IN:
            diag("virtio", f"REQ read  sector={sector} len={desc_len}")
            ok = self._do_read(sector, desc_addr, desc_len)
        elif req_type == VIRTIO_BLK_T_OUT:
            diag("virtio", f"REQ write sector={sector} len={desc_len}")
            ok = self._do_write(sector, desc_addr, desc_len)
        elif req_type == VIRTIO_BLK_T_FLUSH:
            # FLUSH: 把文件内容刷到磁盘
            ok = self._do_flush()
        elif req_type in (VIRTIO_BLK_T_DISCARD, VIRTIO_BLK_T_WRITE_ZEROES):
            # 不支持的请求类型 -> 返回 VIRTIO_BLK_S_UNSUPP
            self._mem_write(status_addr, bytes([VIRTIO_BLK_S_UNSUPP]))
            return True
        else:
            ok = False

        # 写入状态字节
        status = VIRTIO_BLK_S_OK if ok else VIRTIO_BLK_S_IOERR
        self._mem_write(status_addr, bytes([status]))

        return ok

    def _do_read(self, sector: int, buf_pa: int, buf_len: int) -> bool:
        """从扇区 *sector* 读取数据到 Guest 物理地址 *buf_pa*."""
        offset = sector * SECTOR_SIZE
        if offset >= self._disk_size:
            data = b"\x00" * buf_len
        else:
            data = os.pread(self._fd, buf_len, offset)
        # Detect stale PAGE_POISON in target page before write.
        # A clean kernel zeroes pages on alloc; if we see 0xfe here,
        # the zeroing was skipped — emulator TLB/MMU bug.
        try:
            before = self._mem_read(buf_pa & ~0xFFF, 64)
        except Exception:
            before = b""
        if before[:4].count(0xFE) >= 3:
            print(
                f"\n[vblk-poison] pa=0x{buf_pa:x} sector={sector} "
                f"len={buf_len} page_hex={before[:16].hex()}\n",
                file=sys.stderr, flush=True,
            )
        self._mem_write(buf_pa, data)
        return True

    def _do_write(self, sector: int, buf_pa: int, buf_len: int) -> bool:
        """从 Guest 物理地址 *buf_pa* 写入数据到扇区 *sector*."""
        # 只读镜像: 静默忽略写 (返回成功避免内核报 I/O 错)。
        # 用户已确认暂不支持写磁盘 (root=/dev/vda ro 不会写数据块)。
        if self._read_only:
            return True
        data = self._mem_read(buf_pa, buf_len)
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

    # ---- virtqueue 辅助函数 ----

    def _read_descriptor(self, idx: int) -> tuple[int, int, int, int]:
        """读取第 *idx* 个 virtqueue 描述符 -> (addr, len, flags, next)."""
        addr = self._queue_desc + idx * _VRING_DESC_SIZE
        raw = self._mem_read(addr, _VRING_DESC_SIZE)
        desc_addr = int.from_bytes(raw[0:8], "little")
        desc_len = int.from_bytes(raw[8:12], "little")
        desc_flags = int.from_bytes(raw[12:14], "little")
        desc_next = int.from_bytes(raw[14:16], "little")
        return desc_addr, desc_len, desc_flags, desc_next

    def _write_used_entry(self, ring_idx: int, desc_head: int, _len_written: int) -> None:
        """写入 used ring 条目.

        Used ring 布局:
          [0] flags (u16), [2] idx (u16), [4+] ring (ring_idx * 8)
        used ring 条目: [0] id (u32), [4] len (u32) = 8 bytes
        """
        # Used ring 条目偏移: 2 (flags+idx) + ring_idx * 8
        used_ring_start = self._queue_device
        entry_off = used_ring_start + 4 + ring_idx * 8
        raw = struct.pack("<II", desc_head, 0)
        self._mem_write(entry_off, raw)

    def _read_u16(self, pa: int) -> int:
        raw = self._mem_read(pa, 2)
        return int.from_bytes(raw, "little")

    def _write_u16(self, pa: int, val: int) -> None:
        self._mem_write(pa, struct.pack("<H", mask16(val)))

    def _reset(self) -> None:
        """设备重置."""
        self._status = 0
        self._device_features_sel = 0
        self._driver_features_sel = 0
        self._driver_features = 0
        self._queue_sel = 0
        self._queue_ready = False
        self._interrupt_status = 0
        self._queue_desc = 0
        self._queue_driver = 0
        self._queue_device = 0
        self._last_avail_idx = 0
        self._lower_irq_if_idle()
