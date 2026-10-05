#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""virtio-mmio 传输层 (virtio v1.0+ modern, 非 legacy).

只承载传输层: MMIO 寄存器堆 (0x000 至 0x0ff), virtqueue 环的读写与描述符链遍历,
中断状态的置位与清除。设备语义 (feature 集合、0x100 起的配置空间布局、请求处理)
由宿主设备类以回调提供, virtio-blk 与 virtio-net 共用本模块。

寄存器布局 (OASIS virtio v1.2 MMIO):
  0x000 | 4 | R   | MagicValue         | 0x74726976
  0x004 | 4 | R   | Version            | 0x2
  0x008 | 4 | R   | DeviceID           | 设备类型
  0x00c | 4 | R   | VendorID           | 0x0
  0x010 | 4 | R   | DeviceFeatures     | bits
  0x014 | 4 | W   | DeviceFeaturesSel  | page selector
  0x020 | 4 | W   | DriverFeatures     | bits
  0x024 | 4 | W   | DriverFeaturesSel  | page selector
  0x030 | 4 | W   | QueueSel           | queue index
  0x034 | 4 | R   | QueueNumMax        | max queue entries
  0x038 | 4 | W   | QueueNum           | set queue size
  0x044 | 4 | W   | QueueReady         | activate queue
  0x050 | 4 | W   | QueueNotify        | new buffer available
  0x060 | 4 | R   | InterruptStatus    | bit 0: used buffer
  0x064 | 4 | W   | InterruptACK       | clear interrupt
  0x070 | 4 | R/W | Status             | device status
  0x080 | 4 | W   | QueueDescLow       | desc table (low 32)
  0x084 | 4 | W   | QueueDescHigh      | desc table (high 32)
  0x090 | 4 | W   | QueueDriverLow     | driver area (low 32)
  0x094 | 4 | W   | QueueDriverHigh    | driver area (high 32)
  0x0a0 | 4 | W   | QueueDeviceLow     | device area (low 32)
  0x0a4 | 4 | W   | QueueDeviceHigh    | device area (high 32)
  0x0fc | 4 | R   | ConfigGeneration   | config change counter
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import struct

from pyremu.utils.mask import mask16, mask32

# ============================================================
#  MMIO 寄存器偏移与区域大小
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

# 设备特定配置空间的起始偏移。0x0fc 之后到区域末尾留给配置空间。
VIRTIO_MMIO_CONFIG_OFFSET = 0x100

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
VIRTIO_F_RING_EVENT_IDX = 1 << 29

# ============================================================
#  virtqueue 描述符标志
# ============================================================

VRING_DESC_F_NEXT = 1
VRING_DESC_F_WRITE = 2
VRING_DESC_F_INDIRECT = 4

# 描述符大小: u64 addr + u32 len + u16 flags + u16 next = 16 bytes
VRING_DESC_SIZE = 16

# 中断状态位 (InterruptStatus 寄存器)
VIRTIO_MMIO_INT_VRING = 1


@dataclass(slots=True)
class VirtqDesc:
    """一个 virtqueue 描述符.

    Attributes:
        addr: 缓冲区物理地址.
        length: 缓冲区长度 (字节).
        flags: 描述符标志位.
        next: 链中下一个描述符的索引, 仅在 flags 含 VRING_DESC_F_NEXT 时有意义.
    """

    addr: int
    length: int
    flags: int
    next: int


class VirtQueue:
    """一条 virtqueue 的环状态与描述符遍历.

    available ring 由驱动写入, used ring 由本类写入。两部分都在受调试程序的物理内存里,
    经 mem_read 与 mem_write 回调访问。
    """

    def __init__(
        self,
        index: int,
        num_max: int,
        mem_read: Callable[[int, int], bytes],
        mem_write: Callable[[int, bytes], None],
    ) -> None:
        self.index = index
        self.num_max = num_max
        self._mem_read = mem_read
        self._mem_write = mem_write
        # num 初值与重置后的取法都与设备重置前一致: 重置不清 num,
        # 驱动若不重设, 队列长度沿用上一次配置的值。
        self.num: int = num_max
        self.reset()

    def reset(self) -> None:
        """清除队列的激活状态与环地址, 保留 num."""
        self.ready = False
        self.desc: int = 0
        self.driver: int = 0
        self.device: int = 0
        self.last_avail_idx: int = 0

    # ---- 环状态 ----

    def set_desc(self, val: int, high: bool) -> None:
        """写入描述符表地址的一半。high 为真时写高 32 位."""
        if high:
            self.desc = mask32(self.desc) | (mask32(val) << 32)
        else:
            self.desc = (self.desc & 0xFFFF_FFFF_0000_0000) | mask32(val)

    def set_driver(self, val: int, high: bool) -> None:
        """写入 available ring 地址的一半."""
        if high:
            self.driver = mask32(self.driver) | (mask32(val) << 32)
        else:
            self.driver = (self.driver & 0xFFFF_FFFF_0000_0000) | mask32(val)

    def set_device(self, val: int, high: bool) -> None:
        """写入 used ring 地址的一半."""
        if high:
            self.device = mask32(self.device) | (mask32(val) << 32)
        else:
            self.device = (self.device & 0xFFFF_FFFF_0000_0000) | mask32(val)

    def configured(self) -> bool:
        """三处地址均已写入且队列长度非零."""
        return bool(self.num) and bool(self.desc) and bool(self.driver) and bool(self.device)

    # ---- available ring ----

    def avail_idx(self) -> int:
        """读取 available ring 的 idx 字段."""
        return self._read_u16(self.driver + 2)

    def avail_head(self, avail_idx: int) -> int:
        """读取 available ring 中第 *avail_idx* 项的描述符链头索引.

        available ring 项以队列长度作为取模对象循环使用, 故取其索引位置的余数。
        """
        ring_off = 4 + (avail_idx % self.num) * 2
        return self._read_u16(self.driver + ring_off)

    # ---- used ring ----

    def push_used(self, desc_head: int, length: int) -> None:
        """向 used ring 写入一项 (描述符链头索引与写入字节数).

        used ring 布局: [0] flags (u16), [2] idx (u16), [4+] ring (每项 8 字节)。
        """
        ring_idx = self.last_avail_idx % self.num
        entry_off = self.device + 4 + ring_idx * 8
        self._mem_write(entry_off, struct.pack("<II", desc_head, mask32(length)))

    def commit_used(self) -> None:
        """把 used ring 的 idx 推进到当前已处理位置."""
        self._write_u16(self.device + 2, self.last_avail_idx)

    # ---- 描述符 ----

    def read_descriptor(self, idx: int) -> VirtqDesc:
        """读取第 *idx* 个描述符."""
        raw = self._mem_read(self.desc + idx * VRING_DESC_SIZE, VRING_DESC_SIZE)
        return VirtqDesc(
            addr=int.from_bytes(raw[0:8], "little"),
            length=int.from_bytes(raw[8:12], "little"),
            flags=int.from_bytes(raw[12:14], "little"),
            next=int.from_bytes(raw[14:16], "little"),
        )

    def descriptor_chain(self, head: int, max_desc: int) -> list[VirtqDesc] | None:
        """自 *head* 起沿 next 取出描述符链, 至多取 *max_desc* 个.

        成环保护: 链中某一描述符索引第二次出现即判定为环, 返回 None。此外链长以队列
        长度为上界, 越界同样返回 None。两者使遍历步数有界, 不会因驱动给出的畸形链自旋。

        返回:
            描述符列表; 检出成环或越界时为 None.
        """
        descs: list[VirtqDesc] = []
        seen: set[int] = set()
        idx = head
        while True:
            if idx >= self.num_max or idx in seen or len(descs) >= self.num_max:
                return None
            seen.add(idx)
            desc = self.read_descriptor(idx)
            descs.append(desc)
            if not (desc.flags & VRING_DESC_F_NEXT):
                return descs
            if len(descs) >= max_desc:
                return descs
            idx = desc.next

    # ---- 低层访问 ----

    def _read_u16(self, pa: int) -> int:
        return int.from_bytes(self._mem_read(pa, 2), "little")

    def _write_u16(self, pa: int, val: int) -> None:
        self._mem_write(pa, struct.pack("<H", mask16(val)))


class VirtioMMIO:
    """virtio-mmio 寄存器堆、队列调度与中断状态.

    宿主设备类持有本对象, 并把 Device 接口的读写委托给它。设备语义经三个回调给出:

    - config_read: 读 0x100 起的设备特定配置空间, 参数为配置空间内的偏移;
    - handle_request: 处理一个描述符链, 参数为队列与该描述符链,
      返回 (是否成功, 记入 used ring 的写入字节数);
    - on_irq: 中断线电平变化, 参数为真表示拉高、假表示拉低。

    另有一个可选回调 on_notify: 队列通知先交设备, 返回真表示设备已自行处理该队列,
    传输层不再走通用遍历。接收方向的队列由设备在帧到达时主动填充缓冲区, 不能按通用
    遍历消费驱动挂上的可用项, 故留出这个缺口。不给该回调时行为与单队列块设备一致。

    Usage::

        mmio = VirtioMMIO(
            device_id=0x2,
            device_features=VIRTIO_F_VERSION_1,
            num_queues=1,
            queue_num_max=256,
            mem_read=bus.read,
            mem_write=bus.write,
            config_read=dev.config_read,
            handle_request=dev.handle_request,
            on_irq=dev.set_irq_level,
        )
    """

    def __init__(
        self,
        *,
        device_id: int,
        device_features: int,
        num_queues: int,
        queue_num_max: int,
        mem_read: Callable[[int, int], bytes],
        mem_write: Callable[[int, bytes], None],
        config_read: Callable[[int], int],
        handle_request: Callable[[VirtQueue, list[VirtqDesc]], tuple[bool, int]],
        on_irq: Callable[[bool], None],
        chain_max_desc: int = 64,
        on_notify: Callable[[int], bool] | None = None,
    ) -> None:
        # 单条请求允许的最大描述符数, 设备可据此限制链长
        self.chain_max_desc = chain_max_desc
        # 可选: 队列通知先交设备处理. 返回真表示设备已自行处理该队列,
        # 传输层不再走通用遍历. 接收方向的队列由设备在帧到达时主动填充,
        # 不能按通用遍历消费驱动挂上的缓冲区, 故需要这个缺口.
        self._on_notify = on_notify
        self.device_id = device_id
        self.device_features = device_features
        self.queue_num_max = queue_num_max
        self._mem_read = mem_read
        self._mem_write = mem_write
        self._config_read = config_read
        self._handle_request = handle_request
        self._on_irq = on_irq

        self.queues = [
            VirtQueue(i, queue_num_max, mem_read, mem_write) for i in range(num_queues)
        ]

        self.device_features_sel: int = 0
        self.driver_features_sel: int = 0
        self.driver_features: int = 0
        self.queue_sel: int = 0
        self.status: int = 0
        self.interrupt_status: int = 0

    # ---- Device 接口的委托目标 ----

    def read(self, offset: int, size: int) -> bytes:
        if size not in (1, 2, 4, 8):
            return b"\x00" * size
        # 读取宽度小于 4 字节时只取低位: 配置空间按字节映像返回以该偏移为起始的小端字
        # (见 _config_read), 高出本次读取宽度的字节不属于本次读。驱动读 MAC 这类起始
        # 偏移不保证 4 字节对齐的字段时按单字节读, 不截位会在该处溢出。
        val = self.mmio_read(offset) & ((1 << (size * 8)) - 1)
        return val.to_bytes(size, "little")

    def write(self, offset: int, data: bytes) -> None:
        size = len(data)
        if size not in (1, 2, 4):
            return
        self.mmio_write(offset, int.from_bytes(data, "little"), size)

    # ---- 受调试程序物理内存访问 ----

    def read_phys(self, pa: int, size: int) -> bytes:
        """读取受调试程序的物理内存. 设备处理请求数据时使用."""
        return self._mem_read(pa, size)

    def write_phys(self, pa: int, data: bytes) -> None:
        """写入受调试程序的物理内存."""
        self._mem_write(pa, data)

    # ---- MMIO 读 ----

    def mmio_read(self, offset: int) -> int:
        """读取 MMIO 寄存器."""
        if offset == VIRTIO_MMIO_MAGIC_VALUE:
            return 0x74726976  # "virt"
        if offset == VIRTIO_MMIO_VERSION:
            return 0x2
        if offset == VIRTIO_MMIO_DEVICE_ID:
            return self.device_id
        if offset in (VIRTIO_MMIO_VENDOR_ID, VIRTIO_MMIO_CONFIG_GENERATION):
            # 配置不会变
            return 0x0

        if offset == VIRTIO_MMIO_DEVICE_FEATURES:
            return mask32(self.device_features >> (self.device_features_sel * 32))

        if offset == VIRTIO_MMIO_QUEUE_NUM_MAX:
            return self.queue_num_max

        if offset == VIRTIO_MMIO_INTERRUPT_STATUS:
            return self.interrupt_status

        if offset == VIRTIO_MMIO_STATUS:
            return self.status

        if offset >= VIRTIO_MMIO_CONFIG_OFFSET:
            return self._config_read(offset - VIRTIO_MMIO_CONFIG_OFFSET)

        # 未实现的寄存器, 返回 0
        return 0

    # ---- MMIO 写 ----

    def mmio_write(self, offset: int, val: int, size: int) -> None:
        """写入 MMIO 寄存器."""
        if offset == VIRTIO_MMIO_DEVICE_FEATURES_SEL:
            self.device_features_sel = val

        elif offset == VIRTIO_MMIO_DRIVER_FEATURES:
            sel = self.driver_features_sel
            mask_low = mask32(val) << (sel * 32)
            self.driver_features = (
                self.driver_features & ~(0xFFFF_FFFF << (sel * 32))
            ) | mask_low

        elif offset == VIRTIO_MMIO_DRIVER_FEATURES_SEL:
            self.driver_features_sel = val

        elif offset == VIRTIO_MMIO_QUEUE_SEL:
            self.queue_sel = val

        elif offset == VIRTIO_MMIO_QUEUE_NUM:
            self._current_queue().num = min(val, self.queue_num_max)

        elif offset == VIRTIO_MMIO_QUEUE_READY:
            queue = self._current_queue()
            queue.ready = val != 0
            if queue.ready:
                queue.last_avail_idx = 0

        elif offset == VIRTIO_MMIO_QUEUE_NOTIFY:
            queue = self.queue_at(val)
            if queue is not None and queue.ready and \
            (self._on_notify is None or not self._on_notify(queue.index)):
                self.process_queue(queue.index)

        elif offset == VIRTIO_MMIO_INTERRUPT_ACK:
            self.interrupt_status &= ~val
            self.lower_irq_if_idle()

        elif offset == VIRTIO_MMIO_STATUS:
            # 写 0 表示设备重置
            if val == 0:
                self.reset()
            else:
                self.status = val

        elif offset == VIRTIO_MMIO_QUEUE_DESC_LOW:
            self._current_queue().set_desc(val, high=False)

        elif offset == VIRTIO_MMIO_QUEUE_DESC_HIGH:
            self._current_queue().set_desc(val, high=True)

        elif offset == VIRTIO_MMIO_QUEUE_DRIVER_LOW:
            self._current_queue().set_driver(val, high=False)

        elif offset == VIRTIO_MMIO_QUEUE_DRIVER_HIGH:
            self._current_queue().set_driver(val, high=True)

        elif offset == VIRTIO_MMIO_QUEUE_DEVICE_LOW:
            self._current_queue().set_device(val, high=False)

        elif offset == VIRTIO_MMIO_QUEUE_DEVICE_HIGH:
            self._current_queue().set_device(val, high=True)

    def _current_queue(self) -> VirtQueue:
        """按 QueueSel 取当前队列. 选择越界时返回第 0 条, 与单队列设备的既有行为一致."""
        queue = self.queue_at(self.queue_sel)
        return self.queues[0] if queue is None else queue

    def queue_at(self, index: int) -> VirtQueue | None:
        """按索引取队列, 越界返回 None."""
        if 0 <= index < len(self.queues):
            return self.queues[index]
        return None

    # ---- 队列调度 ----

    def process_queue(self, index: int, max_descriptors: int = 0) -> bool:
        """处理第 *index* 条队列中待处理的请求.

        Args:
            index: 队列索引.
            max_descriptors: 单次最多处理的描述符链数, 0 表示不限制。

        Returns:
            True 表示仍有未处理的描述符, 调用方应在检查中断标志后再次调用.
        """
        queue = self.queue_at(index)
        if queue is None or not queue.configured():
            return False

        avail_idx = queue.avail_idx()

        processed = 0
        while queue.last_avail_idx != avail_idx:
            if max_descriptors > 0 and processed >= max_descriptors:
                break

            desc_head = queue.avail_head(queue.last_avail_idx)
            ok, written = self._process_chain(queue, desc_head)
            if not ok:
                # 单个请求失败不影响后续请求的处理
                pass
            queue.push_used(desc_head, written)

            queue.last_avail_idx += 1
            processed += 1

        if processed > 0:
            queue.commit_used()
            self.interrupt_status |= VIRTIO_MMIO_INT_VRING
            self.raise_irq()

        return queue.last_avail_idx != avail_idx  # 仍有未处理? 调用方需再次调用

    def _process_chain(self, queue: VirtQueue, desc_head: int) -> tuple[bool, int]:
        """取出一个描述符链并交给设备处理. 链不可解析时按失败计入 used ring."""
        chain = queue.descriptor_chain(desc_head, self.chain_max_desc)
        if chain is None:
            return False, 0
        return self._handle_request(queue, chain)

    # ---- 中断 ----

    def raise_irq(self) -> None:
        """向中断控制器拉高本设备中断源."""
        self._on_irq(True)

    def lower_irq_if_idle(self) -> None:
        """中断已被确认且无残留状态位时, 拉低中断源."""
        if self.interrupt_status == 0:
            self._on_irq(False)

    # ---- 重置 ----

    def reset(self) -> None:
        """设备重置."""
        self.status = 0
        self.device_features_sel = 0
        self.driver_features_sel = 0
        self.driver_features = 0
        self.queue_sel = 0
        self.interrupt_status = 0
        for queue in self.queues:
            queue.reset()
        self.lower_irq_if_idle()
