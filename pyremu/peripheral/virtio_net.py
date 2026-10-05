#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""virtio-net MMIO 网络设备 (virtio v1.2 规范).

只承载设备语义: feature 集合、0x100 起的配置空间 (MAC、链路状态、队列对数、MTU)、
接收与发送两条 virtqueue 的处理。MMIO 寄存器堆与 virtqueue 遍历由
[virtio_mmio](pyremu/peripheral/virtio_mmio.py) 提供。

特性:
- virtio v1.0+ MMIO 传输层 (modern, 非 legacy)
- 双 virtqueue (queue 0 接收, queue 1 发送)
- 只声明 VIRTIO_F_VERSION_1 与 VIRTIO_NET_F_MAC
- 以太帧的收发经 NetBackend 交给后端, 本轮唯一实现是回环
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable

from pyremu.core.diag import diag
from pyremu.memory.bus import Device
from pyremu.peripheral.virtio_mmio import (
    VIRTIO_F_VERSION_1,
    VIRTIO_MMIO_INT_VRING,
    VIRTIO_MMIO_SIZE,
    VirtioMMIO,
    VirtqDesc,
    VirtQueue,
    VRING_DESC_F_WRITE,
)

# ============================================================
#  设备标识与 feature
# ============================================================

# virtio 设备类型号
VIRTIO_NET_DEVICE_ID = 0x1

# 设备提供 MAC 地址 (bit 5)
VIRTIO_NET_F_MAC = 1 << 5

# 本设备支持的 feature (64-bit)
#
# VIRTIO_F_RING_EVENT_IDX (bit 29) 与 VIRTIO_F_RING_INDIRECT_DESC (bit 28) 的声明
# 问题与 virtio-blk 同源, 理由见 virtio_blk.py 的同名常量注释: 前者要求设备回写
# available ring 的 avail_event 字段, 后者要求描述符链遍历实现间接表, 二者本模块
# 均未实现, 声明了会使驱动走上设备不支持的路径。
#
# VIRTIO_NET_F_MRG_RXBUF 同样不声明: 声明后驱动允许把一个帧拆进多个接收缓冲区,
# 首部还要多出 num_buffers 字段, 接收路径须据此拼帧。不声明时一个帧占一个缓冲区。
#
# VIRTIO_NET_F_CSUM 与 VIRTIO_NET_F_GUEST_CSUM 与各类 GSO feature 一概不声明:
# 声明即表示设备参与校验和与分段卸载, 未实现时会静默产出错误的帧。不声明时
# 驱动在软件里完成这些工作, 设备只需收发完整的以太帧。
_DEVICE_FEATURES = VIRTIO_F_VERSION_1 | VIRTIO_NET_F_MAC

# ============================================================
#  队列
# ============================================================

# 队列 0 接收驱动提供的空缓冲区, 队列 1 接收驱动填入的待发帧
VIRTIO_NET_RX_QUEUE = 0
VIRTIO_NET_TX_QUEUE = 1
VIRTIO_NET_NUM_QUEUES = 2

# 单条发送链允许的描述符数上界。一个帧通常只占一个描述符, 取 16 已足够容纳
# 首部与帧分处两个描述符的情形, 同时使畸形长链尽早被截断。
VIRTIO_NET_CHAIN_MAX_DESC = 16

# 等待接收缓冲区的帧数上界。超出即丢弃并计数, 使驱动长时间不挂缓冲区时
# 内存占用有界, 而不是让待收队列无限增长。
VIRTIO_NET_RX_QUEUE_LIMIT = 64

# ============================================================
#  virtio_net_config 配置空间 (offset 0x100+)
# ============================================================

VIRTIO_NET_CFG_MAC = 0x00  # u8 mac[6]
VIRTIO_NET_CFG_STATUS = 0x06  # __virtio16 status
VIRTIO_NET_CFG_MAX_VQ_PAIRS = 0x08  # __virtio16 max_virtqueue_pairs
VIRTIO_NET_CFG_MTU = 0x0A  # __virtio16 mtu

# 配置空间里实际用到的字节数, 其后的偏移一律读作 0
VIRTIO_NET_CFG_SIZE = 0x0C

# status 字段的位 (VIRTIO_NET_S_*)
VIRTIO_NET_S_LINK_UP = 1

# ============================================================
#  以太网与 virtio 首部
# ============================================================

# 以太帧首部长度: 目的 MAC (6) + 源 MAC (6) + 类型 (2)
ETHERNET_HEADER_SIZE = 14

# 以太网最大传输单元
ETHERNET_MTU = 1500

# 以太帧的最大长度 (不含前导码与帧校验序列)
ETHERNET_FRAME_MAX = ETHERNET_HEADER_SIZE + ETHERNET_MTU

# 每个缓冲区开头的 virtio_net_hdr:
# flags (u8) + gso_type (u8) + hdr_len (u16) + gso_size (u16)
# + csum_start (u16) + csum_offset (u16)
#
# 未协商 VIRTIO_NET_F_MRG_RXBUF 时首部为 10 字节; 协商后末尾多一个
# num_buffers (u16)。本设备不协商该 feature, 故取 10。未协商任何 GSO 与校验和
# 卸载 feature 时, 首部各字段恒为 0。
VIRTIO_NET_HDR_SIZE = 10

# ============================================================
#  中断源编号
# ============================================================

# S 模式一侧实例的中断源编号。既有取值是 VIRTIO_BLK_IRQ = 1 与 UART_IRQ = 10,
# 故 2 空闲。
VIRTIO_NET_S_IRQ = 2

# M 模式一侧实例的中断源编号。两个实例必须取不同的编号: PLIC 的挂起位按中断源
# 编号索引且各 context 共用, 同编号时无法区分, 一方拉高中断线时另一方已使能的
# 同一个中断源会一并置位。
VIRTIO_NET_M_IRQ = 3

# ============================================================
#  默认 MAC 地址
# ============================================================

# 首字节的 bit0 为 0 表示单播, bit1 为 1 表示本地管理地址, 故 0x02 开头的地址
# 不会与任何厂商分配的地址冲突。两个实例取不同值, 回环时可用源地址区分。
DEFAULT_VIRTIO_NET_S_MAC = b"\x02\x00\x00\x00\x00\x01"
DEFAULT_VIRTIO_NET_M_MAC = b"\x02\x00\x00\x00\x00\x02"


class NetBackend(ABC):
    """以太帧的收发后端 — 设备与链路之间的接缝.

    设备只把待发帧交给后端, 并从后端接收到达帧; 帧的走向 (回环、tap 设备、
    双端口互联) 全由后端决定。本轮唯一实现是 LoopbackBackend。
    """

    @abstractmethod
    def bind(self, device: VirtIONet) -> None:
        """绑定收发帧的设备. 设备构造时调用一次."""

    @abstractmethod
    def transmit(self, frame: bytes) -> None:
        """发送一个以太帧.

        Args:
            frame: 完整的以太帧, 不含 virtio_net_hdr.
        """


class LoopbackBackend(NetBackend):
    """回环后端 — 帧原样送回所绑定设备的接收路径, 不做任何首部改写.

    由此 "飞地内 UDP 自收自发" 无需特例即可成立: 发往本机 IP 的报文先触发 ARP
    请求, 该请求回环回来后由本机协议栈应答, 应答再回环回来填入 ARP 表, 之后 UDP
    报文以本机 MAC 为目的发出并回环到已绑定的接收套接字。

    终止性: 接收路径不发送帧, 故回环不会自我增殖; 协议栈对 ARP 应答不再回以应答。
    """

    def __init__(self) -> None:
        self._device: VirtIONet | None = None

    def bind(self, device: VirtIONet) -> None:
        self._device = device

    def transmit(self, frame: bytes) -> None:
        if self._device is None:
            return
        self._device.deliver_frame(frame)


class VirtIONet(Device):
    """virtio-net MMIO 网络设备.

    通过 MMIO 寄存器接口暴露一个 virtio-net 设备, 受调试程序内核与飞地运行时
    各自以标准 virtio-net 驱动收发以太帧。帧的投递目标由 NetBackend 决定。

    Usage::

        net = VirtIONet(
            mac=DEFAULT_VIRTIO_NET_S_MAC,
            mem_read=bus.read,
            mem_write=bus.write,
            on_irq=raise_device_irq,
            irq=VIRTIO_NET_S_IRQ,
        )
        bus.add_device(0x1000_7000, net)
    """

    def __init__(
        self,
        mem_read: Callable[[int, int], bytes],
        mem_write: Callable[[int, bytes], None],
        mac: bytes = DEFAULT_VIRTIO_NET_S_MAC,
        queue_size_max: int = 256,
        on_irq: Callable[[int, bool], None] | None = None,
        irq: int = 0,
        backend: NetBackend | None = None,
    ) -> None:
        if len(mac) != 6:
            raise ValueError(f"MAC 地址须为 6 字节, 实际 {len(mac)} 字节")

        self.base_addr = 0
        self.size = VIRTIO_MMIO_SIZE

        # 中断投递: 完成中断经 on_irq(irq, level) 上报 (irq=0 或未给出 on_irq 时
        # 不投递)。调用方应传 Emulator.raise_device_irq, 理由见 virtio_blk.py。
        self._on_irq = on_irq
        self._irq = irq

        # 待收帧队列: 帧到达时若无空闲缓冲区则在此暂存, 待驱动挂上缓冲区后补投。
        self._rx_frames: deque[bytes] = deque()
        # 因缓冲区不足或队列已满而丢弃的帧数
        self._rx_dropped = 0

        self._backend = backend if backend is not None else LoopbackBackend()
        self._backend.bind(self)

        self.mmio = VirtioMMIO(
            device_id=VIRTIO_NET_DEVICE_ID,
            device_features=_DEVICE_FEATURES,
            num_queues=VIRTIO_NET_NUM_QUEUES,
            queue_num_max=queue_size_max,
            mem_read=mem_read,
            mem_write=mem_write,
            config_read=self._config_read,
            handle_request=self._handle_request,
            on_irq=self._set_irq_level,
            on_notify=self._handle_notify,
            chain_max_desc=VIRTIO_NET_CHAIN_MAX_DESC,
        )
        # MAC 在 mmio 构造前使用, 故两个字段都先于它赋值; 配置空间按字节寻址,
        # 由 _build_config_space 在每次读取时按当前链路状态重建。
        self._mac = bytes(mac)
        self._link_up = True

    # ---- Device 接口 ----

    def read(self, offset: int, size: int) -> bytes:
        return self.mmio.read(offset, size)

    def write(self, offset: int, data: bytes) -> None:
        self.mmio.write(offset, data)

    def process_queue(self, index: int, max_descriptors: int = 0) -> bool:
        """处理第 *index* 条队列中待处理的请求. 语义见 VirtioMMIO.process_queue."""
        return self.mmio.process_queue(index, max_descriptors)

    # ---- 后端接入 ----

    @property
    def mac(self) -> bytes:
        """本设备的 MAC 地址."""
        return self._mac

    @property
    def rx_dropped(self) -> int:
        """因缺少接收缓冲区或待收队列已满而丢弃的帧数."""
        return self._rx_dropped

    @property
    def rx_pending(self) -> int:
        """已到达但尚未交给驱动的帧数."""
        return len(self._rx_frames)

    def deliver_frame(self, frame: bytes) -> None:
        """把一个到达帧交给接收路径. 由后端调用, 回环后端也经由此处投递."""
        if len(frame) > ETHERNET_FRAME_MAX:
            # 超长帧按丢弃处理: 接收缓冲区按以太帧上界配置, 收下也放不进缓冲区。
            self._rx_dropped += 1
            return
        if len(self._rx_frames) >= VIRTIO_NET_RX_QUEUE_LIMIT:
            self._rx_dropped += 1
            return
        self._rx_frames.append(bytes(frame))
        self._fill_rx_buffers()

    # ---- 中断 ----

    def _set_irq_level(self, level: bool) -> None:
        """按中断线电平操作中断源. irq=0 或未给出 on_irq 时为空操作."""
        if self._on_irq is not None and self._irq:
            self._on_irq(self._irq, level)

    def raise_irq(self) -> None:
        """拉高本设备中断源."""
        self.mmio.raise_irq()

    def lower_irq_if_idle(self) -> None:
        """中断已被确认且无残留状态位时, 拉低中断源."""
        self.mmio.lower_irq_if_idle()

    # ---- 配置空间 ----

    def _build_config_space(self) -> bytearray:
        """按当前的 MAC 与链路状态构造配置空间的字节映像."""
        cfg = bytearray(VIRTIO_NET_CFG_SIZE)
        cfg[VIRTIO_NET_CFG_MAC : VIRTIO_NET_CFG_MAC + 6] = self._mac
        status = VIRTIO_NET_S_LINK_UP if self._link_up else 0
        cfg[VIRTIO_NET_CFG_STATUS : VIRTIO_NET_CFG_STATUS + 2] = status.to_bytes(2, "little")
        # 未协商 VIRTIO_NET_F_MQ, 队列对数恒为 1
        pairs = (1).to_bytes(2, "little")
        cfg[VIRTIO_NET_CFG_MAX_VQ_PAIRS : VIRTIO_NET_CFG_MAX_VQ_PAIRS + 2] = pairs
        cfg[VIRTIO_NET_CFG_MTU : VIRTIO_NET_CFG_MTU + 2] = ETHERNET_MTU.to_bytes(2, "little")
        return cfg

    def _config_read(self, local: int) -> int:
        """读取 virtio-net 设备特定配置. 参数为配置空间内的字节偏移.

        返回以 *local* 为起始的 4 字节小端字。驱动读 MAC 走的是逐字节拷贝
        (memcpy_fromio), 读 status 与 mtu 走 2 字节读, 起始偏移不保证 4 字节对齐,
        故按字节映像取字, 而不是按对齐的 32 位字取。
        """
        cfg = self._build_config_space()
        if local < 0 or local >= len(cfg):
            return 0
        word = bytes(cfg[local : local + 4])
        return int.from_bytes(word.ljust(4, b"\x00"), "little")

    # ---- 队列处理 ----

    def _handle_notify(self, index: int) -> bool:
        """队列通知的入口. 返回真表示本设备已自行处理该队列.

        接收队列不能走传输层的通用遍历: 通用遍历会把驱动挂上的每个缓冲区都当作
        一条待处理的请求消费掉, 而没有到达帧时该缓冲区必须留在可用环里。故接收
        队列的通知只用来触发补投, 由本设备把待收帧写进缓冲区。
        """
        if index == VIRTIO_NET_RX_QUEUE:
            self._fill_rx_buffers()
            return True
        return False

    def _handle_request(self, queue: VirtQueue, chain: list[VirtqDesc]) -> tuple[bool, int]:
        """处理一个描述符链. 只有发送队列经此路径.

        Returns:
            (是否成功, 记入 used ring 的写入字节数)。发送方向设备只读不写,
            故写入字节数恒为 0.
        """
        if queue.index != VIRTIO_NET_TX_QUEUE:
            return False, 0
        return self._handle_tx(chain)

    def _handle_tx(self, chain: list[VirtqDesc]) -> tuple[bool, int]:
        """取出一个待发帧并交给后端. 缓冲区布局为 virtio_net_hdr 后接以太帧.

        首部与帧可同处一个描述符, 也可分处相邻的若干描述符, 故先按链顺序拼出完整
        字节序列再切掉首部。
        """
        buf = b"".join(self.mmio.read_phys(desc.addr, desc.length) for desc in chain)
        if (
            len(buf) < VIRTIO_NET_HDR_SIZE
            or len(buf[VIRTIO_NET_HDR_SIZE:]) < ETHERNET_HEADER_SIZE
        ):
            return False, 0
        frame = buf[VIRTIO_NET_HDR_SIZE:]
        if self._backend is not None:
            self._backend.transmit(frame)
        return True, 0

    def _fill_rx_buffers(self) -> None:
        """把待收帧写入驱动的空闲接收缓冲区, 并回写 used ring.

        每个缓冲区容纳一帧, 布局为一个全零的 virtio_net_hdr 后接以太帧。缓冲区
        不足以容纳当前帧时丢弃该帧并保留缓冲区, 使下一个较短的帧仍有机会用它。
        """
        queue = self.mmio.queue_at(VIRTIO_NET_RX_QUEUE)
        if queue is None or not queue.configured():
            return

        avail_idx = queue.avail_idx()
        filled = 0
        while self._rx_frames and queue.last_avail_idx != avail_idx:
            desc_head = queue.avail_head(queue.last_avail_idx)
            chain = queue.descriptor_chain(desc_head, 1)
            if chain is None:
                break

            desc = chain[0]
            if not (desc.flags & VRING_DESC_F_WRITE):
                break

            frame = self._rx_frames[0]
            need = VIRTIO_NET_HDR_SIZE + len(frame)
            if desc.length < need:
                # 缓冲区太小: 丢弃该帧, 保留缓冲区供后续较短的帧使用
                self._rx_frames.popleft()
                self._rx_dropped += 1
                continue

            self.mmio.write_phys(desc.addr, bytes(VIRTIO_NET_HDR_SIZE) + frame)
            queue.push_used(desc_head, need)
            queue.last_avail_idx += 1
            self._rx_frames.popleft()
            filled += 1

        if filled:
            queue.commit_used()
            self.mmio.interrupt_status |= VIRTIO_MMIO_INT_VRING
            self.mmio.raise_irq()
