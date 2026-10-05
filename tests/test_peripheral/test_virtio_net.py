#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""virtio-net MMIO 设备测试.

测试覆盖:
- MMIO 寄存器读写与设备标识
- Feature 协商
- 设备重置与状态机
- 配置空间 (MAC、链路状态、队列对数、MTU)
- 接收与发送两条 virtqueue
- 回环后端的收发语义
- 中断投递与确认
- 接收路径在缓冲区不足与畸形链下的行为
"""

from __future__ import annotations

import struct

import pytest

from pyremu.interrupt.plic import (
    PLIC,
    PLIC_ENABLE_BASE,
    PLIC_ENABLE_STRIDE,
    PLIC_PENDING_BASE,
    PLIC_PRIORITY_BASE,
    PLIC_PRIORITY_STRIDE,
)
from pyremu.peripheral.virtio_mmio import (
    VIRTIO_F_RING_EVENT_IDX,
    VIRTIO_F_RING_INDIRECT_DESC,
    VIRTIO_F_VERSION_1,
    VIRTIO_MMIO_CONFIG_OFFSET,
    VIRTIO_MMIO_DEVICE_FEATURES,
    VIRTIO_MMIO_DEVICE_FEATURES_SEL,
    VIRTIO_MMIO_DEVICE_ID,
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
    VIRTIO_MMIO_STATUS,
    VIRTIO_MMIO_VENDOR_ID,
    VIRTIO_MMIO_VERSION,
    VIRTIO_STATUS_ACKNOWLEDGE,
    VIRTIO_STATUS_DRIVER,
    VIRTIO_STATUS_DRIVER_OK,
    VIRTIO_STATUS_FEATURES_OK,
    VRING_DESC_F_NEXT,
    VRING_DESC_F_WRITE,
    VRING_DESC_SIZE,
)
from pyremu.peripheral.virtio_net import (
    VIRTIO_NET_M_IRQ,
    ETHERNET_FRAME_MAX,
    ETHERNET_MTU,
    NetBackend,
    VIRTIO_NET_CFG_MAC,
    VIRTIO_NET_CFG_MAX_VQ_PAIRS,
    VIRTIO_NET_CFG_MTU,
    VIRTIO_NET_CFG_STATUS,
    VIRTIO_NET_DEVICE_ID,
    VIRTIO_NET_F_MAC,
    VIRTIO_NET_HDR_SIZE,
    VIRTIO_NET_S_IRQ,
    VIRTIO_NET_RX_QUEUE,
    VIRTIO_NET_RX_QUEUE_LIMIT,
    VIRTIO_NET_S_LINK_UP,
    VIRTIO_NET_TX_QUEUE,
    VirtIONet,
)

# ============================================================
#  辅助: 受调试程序 RAM 模拟 — 用 bytearray 承载 virtqueue 与帧
# ============================================================

_GUEST_RAM_BASE = 0x8000_0000
_GUEST_RAM_SIZE = 16 * 1024 * 1024  # 16 MiB

# 网卡中断源接在 hart 0 的 M context (context 序号 2 * hart_id + 0) 上
_PLIC_M_CONTEXT = 0

# MEIP 与 SEIP 在 mip 中的位
_MEIP = 1 << 11
_SEIP = 1 << 9

# 接收缓冲区大小: 一个 virtio_net_hdr 加一个最大的以太帧
_RX_BUF_SIZE = VIRTIO_NET_HDR_SIZE + ETHERNET_FRAME_MAX

# 两套 virtqueue 结构在 RAM 中的偏移, 收发各占一段互不重叠的区间
_Q0_DESC_OFF = 0x1000
_Q0_DRIVER_OFF = 0x2000
_Q0_DEVICE_OFF = 0x3000
_Q1_DESC_OFF = 0x4000
_Q1_DRIVER_OFF = 0x5000
_Q1_DEVICE_OFF = 0x6000

# 接收缓冲区与待发帧所在区域在 RAM 中的偏移
_RX_BUF_OFF = 0x10000
_TX_BUF_OFF = 0x20000

# 描述符表项: 缓冲区物理地址, 缓冲区长度, 标志位, 链中下一个描述符的索引
_Desc = tuple[int, int, int, int]


def _make_gpa(offset: int) -> int:
    """把 bytearray 偏移量转为受调试程序物理地址."""
    return _GUEST_RAM_BASE + offset


def _to_offset(gpa: int) -> int:
    """把受调试程序物理地址转回 bytearray 偏移量."""
    return gpa - _GUEST_RAM_BASE


@pytest.fixture
def guest_ram() -> bytearray:
    """模拟受调试程序物理 RAM."""
    return bytearray(_GUEST_RAM_SIZE)


@pytest.fixture
def mem_read(guest_ram: bytearray):
    """mem_read 回调: (pa, size) -> bytes."""

    def _read(pa: int, size: int) -> bytes:
        off = _to_offset(pa)
        return bytes(guest_ram[off : off + size])

    return _read


@pytest.fixture
def mem_write(guest_ram: bytearray):
    """mem_write 回调: (pa, data) -> None."""

    def _write(pa: int, data: bytes) -> None:
        off = _to_offset(pa)
        guest_ram[off : off + len(data)] = data

    return _write


@pytest.fixture
def net(mem_read, mem_write) -> VirtIONet:
    """创建 virtio-net 设备 (不接中断上报入口)."""
    return VirtIONet(mem_read=mem_read, mem_write=mem_write)


# ============================================================
#  辅助: MMIO 读写
# ============================================================


def _mmio_read(net: VirtIONet, offset: int) -> int:
    return int.from_bytes(net.read(offset, 4), "little")


def _mmio_write(net: VirtIONet, offset: int, val: int) -> None:
    net.write(offset, val.to_bytes(4, "little"))


def _mmio_write64(net: VirtIONet, lo_off: int, hi_off: int, val: int) -> None:
    """把一个 64 位值写入一对 32 位 MMIO 寄存器."""
    _mmio_write(net, lo_off, val & 0xFFFF_FFFF)
    _mmio_write(net, hi_off, (val >> 32) & 0xFFFF_FFFF)


def _read_config_bytes(net: VirtIONet, local: int, size: int) -> bytes:
    """按驱动的读法从配置空间取字节.

    受调试程序内核读配置空间最终落到 __memcpy_fromio, 该实现先把起始位置按单字节
    读对齐, 中段按 4 字节读, 末尾余数再按单字节读。故读 MAC 这类起始偏移不保证
    4 字节对齐的字段时, 落到设备上的既有 4 字节读, 也有单字节读。
    """
    out = bytearray()
    offset = local
    while size and offset % 4:
        out += net.read(VIRTIO_MMIO_CONFIG_OFFSET + offset, 1)
        offset += 1
        size -= 1
    while size >= 4:
        out += net.read(VIRTIO_MMIO_CONFIG_OFFSET + offset, 4)
        offset += 4
        size -= 4
    while size:
        out += net.read(VIRTIO_MMIO_CONFIG_OFFSET + offset, 1)
        offset += 1
        size -= 1
    return bytes(out)


def _read_config_u16(net: VirtIONet, local: int) -> int:
    return int.from_bytes(_read_config_bytes(net, local, 2), "little")


# ============================================================
#  辅助: PLIC 的公开访问路径
# ============================================================


def _enable_plic_m_context(plic: PLIC, irq: int) -> None:
    """在 hart 0 的 M context 上使能一个中断源并给出非零优先级.

    源使能位与优先级默认为 0, 两者都不满足投递条件, 故断言 MEIP 之前须先经公开的
    MMIO 写入路径配置这两项。
    """
    plic.write(PLIC_PRIORITY_BASE + irq * PLIC_PRIORITY_STRIDE, (1).to_bytes(4, "little"))
    word_off = PLIC_ENABLE_BASE + _PLIC_M_CONTEXT * PLIC_ENABLE_STRIDE + (irq // 32) * 4
    plic.write(word_off, (1 << (irq % 32)).to_bytes(4, "little"))


def _plic_pending(plic: PLIC, irq: int) -> bool:
    """读取中断源的挂起位."""
    word = int.from_bytes(plic.read(PLIC_PENDING_BASE + (irq // 32) * 4, 4), "little")
    return bool(word & (1 << (irq % 32)))


# ============================================================
#  辅助: virtqueue 构建
# ============================================================


def _queue_offsets(index: int) -> tuple[int, int, int]:
    """按队列索引取三处结构的 RAM 偏移 (描述符表, 可用环, 已用环)."""
    if index == VIRTIO_NET_RX_QUEUE:
        return _Q0_DESC_OFF, _Q0_DRIVER_OFF, _Q0_DEVICE_OFF
    return _Q1_DESC_OFF, _Q1_DRIVER_OFF, _Q1_DEVICE_OFF


def _configure_queue(net: VirtIONet, index: int, qnum: int = 8) -> None:
    """按索引配置一条 virtqueue, 三处结构取该队列固定的 RAM 偏移."""
    desc_off, driver_off, device_off = _queue_offsets(index)
    _mmio_write(net, VIRTIO_MMIO_QUEUE_SEL, index)
    _mmio_write(net, VIRTIO_MMIO_QUEUE_NUM, qnum)
    _mmio_write64(
        net, VIRTIO_MMIO_QUEUE_DESC_LOW, VIRTIO_MMIO_QUEUE_DESC_HIGH, _make_gpa(desc_off),
    )
    _mmio_write64(
        net, VIRTIO_MMIO_QUEUE_DRIVER_LOW, VIRTIO_MMIO_QUEUE_DRIVER_HIGH,
        _make_gpa(driver_off),
    )
    _mmio_write64(
        net, VIRTIO_MMIO_QUEUE_DEVICE_LOW, VIRTIO_MMIO_QUEUE_DEVICE_HIGH,
        _make_gpa(device_off),
    )
    _mmio_write(net, VIRTIO_MMIO_QUEUE_READY, 1)


def _write_chain(guest_ram: bytearray, index: int, chain: list[_Desc], first: int = 0) -> None:
    """自描述符表的第 *first* 项起依次写入一条描述符链."""
    desc_off, _, _ = _queue_offsets(index)
    for i, (buf_pa, buf_len, flags, next_idx) in enumerate(chain):
        off = desc_off + (first + i) * VRING_DESC_SIZE
        guest_ram[off : off + VRING_DESC_SIZE] = struct.pack(
            "<QIHH", buf_pa, buf_len, flags, next_idx,
        )


def _set_avail_idx(guest_ram: bytearray, index: int, idx: int) -> None:
    """设置第 *index* 条队列可用环的 idx."""
    _, driver_off, _ = _queue_offsets(index)
    guest_ram[driver_off + 2 : driver_off + 4] = struct.pack("<H", idx)


def _write_avail_entry(guest_ram: bytearray, index: int, ring_idx: int, head: int) -> None:
    """向第 *index* 条队列的可用环写入一项."""
    _, driver_off, _ = _queue_offsets(index)
    off = driver_off + 4 + ring_idx * 2
    guest_ram[off : off + 2] = struct.pack("<H", head)


def _used_entry(guest_ram: bytearray, index: int, ring_idx: int) -> tuple[int, int]:
    """读已用环的第 *ring_idx* 项 (描述符链头索引, 设备写入字节数)."""
    _, _, device_off = _queue_offsets(index)
    off = device_off + 4 + ring_idx * 8
    return struct.unpack("<II", bytes(guest_ram[off : off + 8]))


def _used_idx(guest_ram: bytearray, index: int) -> int:
    """读第 *index* 条队列已用环的 idx."""
    _, _, device_off = _queue_offsets(index)
    return struct.unpack("<H", bytes(guest_ram[device_off + 2 : device_off + 4]))[0]


def _kick(net: VirtIONet, index: int) -> None:
    """敲门铃通知设备某条队列有新项."""
    _mmio_write(net, VIRTIO_MMIO_QUEUE_NOTIFY, index)


# ============================================================
#  辅助: 帧与缓冲区
# ============================================================


def _make_frame(payload: bytes = b"payload", ethertype: int = 0x0800) -> bytes:
    """构造一个以太帧: 目的 MAC 加源 MAC 加类型加载荷."""
    dst = b"\x02\x00\x00\x00\x00\x02"
    src = b"\x02\x00\x00\x00\x00\x01"
    return dst + src + ethertype.to_bytes(2, "big") + payload


def _post_rx_buffer(guest_ram: bytearray, slot: int, buf_len: int = _RX_BUF_SIZE) -> int:
    """在接收队列上挂一个空缓冲区, 返回缓冲区物理地址."""
    buf_pa = _make_gpa(_RX_BUF_OFF + slot * 0x1000)
    _write_chain(
        guest_ram, VIRTIO_NET_RX_QUEUE, [(buf_pa, buf_len, VRING_DESC_F_WRITE, 0)],
        first=slot,
    )
    _write_avail_entry(guest_ram, VIRTIO_NET_RX_QUEUE, slot, slot)
    return buf_pa


def _post_rx_buffers(guest_ram: bytearray, count: int) -> list[int]:
    """挂 *count* 个空缓冲区并推进可用环 idx, 返回各缓冲区物理地址."""
    addrs = [_post_rx_buffer(guest_ram, slot) for slot in range(count)]
    _set_avail_idx(guest_ram, VIRTIO_NET_RX_QUEUE, count)
    return addrs


def _submit_tx(guest_ram: bytearray, frame: bytes, slot: int = 0) -> int:
    """把一个待发帧放入发送队列, 返回承载该帧的缓冲区物理地址.

    缓冲区布局与驱动一致: 开头是全零的 virtio_net_hdr, 其后是以太帧。
    """
    buf_pa = _make_gpa(_TX_BUF_OFF + slot * 0x1000)
    buf = bytes(VIRTIO_NET_HDR_SIZE) + frame
    off = _to_offset(buf_pa)
    guest_ram[off : off + len(buf)] = buf
    _write_chain(guest_ram, VIRTIO_NET_TX_QUEUE, [(buf_pa, len(buf), 0, 0)], first=slot)
    _write_avail_entry(guest_ram, VIRTIO_NET_TX_QUEUE, slot, slot)
    _set_avail_idx(guest_ram, VIRTIO_NET_TX_QUEUE, slot + 1)
    return buf_pa


def _read_guest_frame(guest_ram: bytearray, buf_pa: int, length: int) -> bytes:
    """从缓冲区读出一个到达帧, 去掉开头的 virtio_net_hdr."""
    off = _to_offset(buf_pa)
    return bytes(guest_ram[off + VIRTIO_NET_HDR_SIZE : off + length])


class RecordingBackend(NetBackend):
    """记录待发帧的后端, 不把帧送回设备. 用于观察设备交给后端的帧内容."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []

    def bind(self, device: VirtIONet) -> None:
        """本后端不持有设备, 无绑定动作."""

    def transmit(self, frame: bytes) -> None:
        self.frames.append(bytes(frame))


# ============================================================
#  Tests: MMIO 寄存器与设备标识
# ============================================================


class TestMmioRegisters:
    """MMIO 寄存器读写测试."""

    @pytest.mark.parametrize(
        "reg, expect", [
            (VIRTIO_MMIO_MAGIC_VALUE, 0x74726976),  # "virt" 的小端写法
            (VIRTIO_MMIO_VERSION, 0x2),  # modern, 非 legacy
            (VIRTIO_MMIO_DEVICE_ID, VIRTIO_NET_DEVICE_ID),
            (VIRTIO_MMIO_VENDOR_ID, 0x0),
            (VIRTIO_MMIO_QUEUE_NUM_MAX, 256),
        ],
    )
    def test_readonly_register(self, net: VirtIONet, reg: int, expect: int) -> None:
        assert _mmio_read(net, reg) == expect

    def test_device_id_is_network_card(self, net: VirtIONet) -> None:
        """设备类型号为 1, 受调试程序据此选择 virtio-net 驱动."""
        assert VIRTIO_NET_DEVICE_ID == 0x1
        assert _mmio_read(net, VIRTIO_MMIO_DEVICE_ID) == 0x1

    def test_queue_num_clamped_to_num_max(self, net: VirtIONet) -> None:
        _mmio_write(net, VIRTIO_MMIO_QUEUE_SEL, VIRTIO_NET_RX_QUEUE)
        _mmio_write(net, VIRTIO_MMIO_QUEUE_NUM, 8)
        assert net.mmio.queue_at(VIRTIO_NET_RX_QUEUE).num == 8
        _mmio_write(net, VIRTIO_MMIO_QUEUE_NUM, 1024)
        assert net.mmio.queue_at(VIRTIO_NET_RX_QUEUE).num == 256

    def test_two_queues_exist(self, net: VirtIONet) -> None:
        """收发各一条队列, 越界索引不产出队列."""
        assert net.mmio.queue_at(VIRTIO_NET_RX_QUEUE) is not None
        assert net.mmio.queue_at(VIRTIO_NET_TX_QUEUE) is not None
        assert net.mmio.queue_at(2) is None


# ============================================================
#  Tests: feature 协商
# ============================================================


class TestFeatures:
    """Feature 协商测试."""

    def test_version_1_and_mac_declared(self, net: VirtIONet) -> None:
        """第 0 页的第 5 位 (MAC) 与第 1 页的第 0 位 (VERSION_1) 置位."""
        assert VIRTIO_NET_F_MAC == 1 << 5
        _mmio_write(net, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 0)
        assert _mmio_read(net, VIRTIO_MMIO_DEVICE_FEATURES) == VIRTIO_NET_F_MAC
        _mmio_write(net, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 1)
        assert _mmio_read(net, VIRTIO_MMIO_DEVICE_FEATURES) == VIRTIO_F_VERSION_1 >> 32

    def test_ring_features_not_declared(self, net: VirtIONet) -> None:
        """事件索引与间接描述符不得声明.

        声明事件索引要求设备回写可用环的 avail_event 字段, 声明间接描述符要求遍历器
        实现间接表, 二者本设备均未实现, 声明后驱动会走上设备不支持的路径而静默丢失
        收发, 故两位置零。
        """
        assert VIRTIO_F_RING_INDIRECT_DESC == 1 << 28
        assert VIRTIO_F_RING_EVENT_IDX == 1 << 29
        _mmio_write(net, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 0)
        features = _mmio_read(net, VIRTIO_MMIO_DEVICE_FEATURES)
        assert not features & (1 << 28), "不得声明 VIRTIO_F_RING_INDIRECT_DESC"
        assert not features & (1 << 29), "不得声明 VIRTIO_F_RING_EVENT_IDX"

    def test_offload_features_not_declared(self, net: VirtIONet) -> None:
        """拆分接收缓冲区与各类卸载 feature 不得声明."""
        _mmio_write(net, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 0)
        features = _mmio_read(net, VIRTIO_MMIO_DEVICE_FEATURES)
        assert not features & (1 << 0), "不得声明 VIRTIO_NET_F_CSUM"
        assert not features & (1 << 1), "不得声明 VIRTIO_NET_F_GUEST_CSUM"
        assert not features & (1 << 7), "不得声明 VIRTIO_NET_F_GUEST_TSO4"
        assert not features & (1 << 15), "不得声明 VIRTIO_NET_F_MRG_RXBUF"

    def test_status_negotiation_sequence(self, net: VirtIONet) -> None:
        """按 ACKNOWLEDGE / DRIVER / FEATURES_OK / DRIVER_OK 依次推进."""
        for bit in (
            VIRTIO_STATUS_ACKNOWLEDGE,
            VIRTIO_STATUS_DRIVER,
            VIRTIO_STATUS_FEATURES_OK,
            VIRTIO_STATUS_DRIVER_OK,
        ):
            _mmio_write(net, VIRTIO_MMIO_STATUS, _mmio_read(net, VIRTIO_MMIO_STATUS) | bit)
        assert _mmio_read(net, VIRTIO_MMIO_STATUS) == (
            VIRTIO_STATUS_ACKNOWLEDGE | VIRTIO_STATUS_DRIVER
            | VIRTIO_STATUS_FEATURES_OK | VIRTIO_STATUS_DRIVER_OK
        )

    def test_reset_clears_status_and_queues(self, net: VirtIONet) -> None:
        _mmio_write(net, VIRTIO_MMIO_STATUS, VIRTIO_STATUS_ACKNOWLEDGE)
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        assert net.mmio.queue_at(VIRTIO_NET_RX_QUEUE).ready

        _mmio_write(net, VIRTIO_MMIO_STATUS, 0)

        assert _mmio_read(net, VIRTIO_MMIO_STATUS) == 0
        assert not net.mmio.queue_at(VIRTIO_NET_RX_QUEUE).ready
        assert not net.mmio.queue_at(VIRTIO_NET_TX_QUEUE).ready


# ============================================================
#  Tests: 配置空间
# ============================================================


class TestConfigSpace:
    """配置空间读写测试."""

    def test_mac_is_readable_bytewise(self, net: VirtIONet) -> None:
        """MAC 按驱动的读法逐段读出, 与设备给出的地址一致.

        驱动读 MAC 时起始偏移为 0 的字段按 4 字节读, 其后两个字节按单字节读。单字节
        读若不做读取宽度的截位, 以该偏移为起始的小端字 (如 0x00000100) 无法写入一个
        字节, 表现为 OverflowError, 故本用例锁定该截位。
        """
        assert _read_config_bytes(net, VIRTIO_NET_CFG_MAC, 6) == net.mac
        for i in range(6):
            one = net.read(VIRTIO_MMIO_CONFIG_OFFSET + VIRTIO_NET_CFG_MAC + i, 1)
            assert one == bytes([net.mac[i]])

    def test_custom_mac(self, mem_read, mem_write) -> None:
        net = VirtIONet(
            mem_read=mem_read, mem_write=mem_write, mac=b"\x02\xaa\xbb\xcc\xdd\xee",
        )
        want = b"\x02\xaa\xbb\xcc\xdd\xee"
        assert _read_config_bytes(net, VIRTIO_NET_CFG_MAC, 6) == want

    def test_mac_length_validated(self, mem_read, mem_write) -> None:
        with pytest.raises(ValueError, match="6 字节"):
            VirtIONet(mem_read=mem_read, mem_write=mem_write, mac=b"\x02\x00\x00")

    def test_link_status_reports_up(self, net: VirtIONet) -> None:
        assert _read_config_u16(net, VIRTIO_NET_CFG_STATUS) == VIRTIO_NET_S_LINK_UP

    def test_queue_pairs_and_mtu(self, net: VirtIONet) -> None:
        """未协商多队列, 队列对数为 1; MTU 为以太网上界."""
        assert _read_config_u16(net, VIRTIO_NET_CFG_MAX_VQ_PAIRS) == 1
        assert _read_config_u16(net, VIRTIO_NET_CFG_MTU) == ETHERNET_MTU

    def test_read_past_config_returns_zero(self, net: VirtIONet) -> None:
        assert _read_config_bytes(net, 0x100, 4) == b"\x00" * 4


# ============================================================
#  Tests: 发送方向
# ============================================================


class TestTransmit:
    """发送方向测试."""

    def test_tx_frame_reaches_backend(self, mem_read, mem_write, guest_ram) -> None:
        """设备切掉 virtio_net_hdr 后把帧原样交给后端."""
        backend = RecordingBackend()
        net = VirtIONet(mem_read=mem_read, mem_write=mem_write, backend=backend)
        _configure_queue(net, VIRTIO_NET_TX_QUEUE)

        frame = _make_frame(b"hello")
        _submit_tx(guest_ram, frame)
        _kick(net, VIRTIO_NET_TX_QUEUE)

        assert backend.frames == [frame]

    def test_tx_writes_used_ring(self, net: VirtIONet, guest_ram: bytearray) -> None:
        """发送完成后回写已用环: 链头索引为 0, 设备写入字节数为 0."""
        _configure_queue(net, VIRTIO_NET_TX_QUEUE)
        _submit_tx(guest_ram, _make_frame())
        _kick(net, VIRTIO_NET_TX_QUEUE)

        assert _used_idx(guest_ram, VIRTIO_NET_TX_QUEUE) == 1
        assert _used_entry(guest_ram, VIRTIO_NET_TX_QUEUE, 0) == (0, 0)

    def test_tx_short_buffer_rejected(self, mem_read, mem_write, guest_ram) -> None:
        """缓冲区短于 virtio_net_hdr 时按失败处理, 不交给后端."""
        backend = RecordingBackend()
        net = VirtIONet(mem_read=mem_read, mem_write=mem_write, backend=backend)
        _configure_queue(net, VIRTIO_NET_TX_QUEUE)
        buf_pa = _make_gpa(_TX_BUF_OFF)
        _write_chain(guest_ram, VIRTIO_NET_TX_QUEUE, [(buf_pa, 4, 0, 0)])
        _write_avail_entry(guest_ram, VIRTIO_NET_TX_QUEUE, 0, 0)
        _set_avail_idx(guest_ram, VIRTIO_NET_TX_QUEUE, 1)

        _kick(net, VIRTIO_NET_TX_QUEUE)

        assert backend.frames == []
        assert _used_idx(guest_ram, VIRTIO_NET_TX_QUEUE) == 1, "失败的请求同样计入已用环"

    def test_tx_short_frame_rejected(self, mem_read, mem_write, guest_ram) -> None:
        """去掉首部后不足一个以太帧首部长度时按失败处理."""
        backend = RecordingBackend()
        net = VirtIONet(mem_read=mem_read, mem_write=mem_write, backend=backend)
        _configure_queue(net, VIRTIO_NET_TX_QUEUE)
        buf_pa = _make_gpa(_TX_BUF_OFF)
        _write_chain(
            guest_ram, VIRTIO_NET_TX_QUEUE, [(buf_pa, VIRTIO_NET_HDR_SIZE + 4, 0, 0)],
        )
        _write_avail_entry(guest_ram, VIRTIO_NET_TX_QUEUE, 0, 0)
        _set_avail_idx(guest_ram, VIRTIO_NET_TX_QUEUE, 1)

        _kick(net, VIRTIO_NET_TX_QUEUE)

        assert backend.frames == []

    def test_tx_chain_split_across_descriptors(self, mem_read, mem_write, guest_ram) -> None:
        """首部与帧分处两个描述符时按链顺序拼接."""
        backend = RecordingBackend()
        net = VirtIONet(mem_read=mem_read, mem_write=mem_write, backend=backend)
        _configure_queue(net, VIRTIO_NET_TX_QUEUE)

        frame = _make_frame(b"split")
        hdr_pa = _make_gpa(_TX_BUF_OFF)
        body_pa = _make_gpa(_TX_BUF_OFF + 0x800)
        guest_ram[_TX_BUF_OFF : _TX_BUF_OFF + VIRTIO_NET_HDR_SIZE] = (
            bytes(VIRTIO_NET_HDR_SIZE)
        )
        body_off = _to_offset(body_pa)
        guest_ram[body_off : body_off + len(frame)] = frame

        _write_chain(
            guest_ram, VIRTIO_NET_TX_QUEUE, [
                (hdr_pa, VIRTIO_NET_HDR_SIZE, VRING_DESC_F_NEXT, 1),
                (body_pa, len(frame), 0, 0),
            ],
        )
        _write_avail_entry(guest_ram, VIRTIO_NET_TX_QUEUE, 0, 0)
        _set_avail_idx(guest_ram, VIRTIO_NET_TX_QUEUE, 1)

        _kick(net, VIRTIO_NET_TX_QUEUE)

        assert backend.frames == [frame]


# ============================================================
#  Tests: 接收方向
# ============================================================


class TestReceive:
    """接收方向测试."""

    def test_frame_lands_in_posted_buffer(self, net: VirtIONet, guest_ram: bytearray) -> None:
        """已挂缓冲区时到达帧就地写入, 帧前带全零的 virtio_net_hdr."""
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        addrs = _post_rx_buffers(guest_ram, 1)
        _kick(net, VIRTIO_NET_RX_QUEUE)

        frame = _make_frame(b"to-guest")
        net.deliver_frame(frame)

        assert _used_idx(guest_ram, VIRTIO_NET_RX_QUEUE) == 1
        head, written = _used_entry(guest_ram, VIRTIO_NET_RX_QUEUE, 0)
        assert head == 0
        assert written == VIRTIO_NET_HDR_SIZE + len(frame)
        buf_off = _to_offset(addrs[0])
        header = bytes(guest_ram[buf_off : buf_off + VIRTIO_NET_HDR_SIZE])
        assert header == bytes(VIRTIO_NET_HDR_SIZE)
        assert _read_guest_frame(guest_ram, addrs[0], written) == frame

    def test_frame_pending_until_buffer_posted(self, net, guest_ram) -> None:
        """缓冲区尚未挂上时帧暂存, 挂上并通知后补投, 不丢失."""
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        frame = _make_frame(b"queued")
        net.deliver_frame(frame)
        assert net.rx_pending == 1
        assert _used_idx(guest_ram, VIRTIO_NET_RX_QUEUE) == 0

        addrs = _post_rx_buffers(guest_ram, 1)
        _kick(net, VIRTIO_NET_RX_QUEUE)

        assert net.rx_pending == 0
        assert _used_idx(guest_ram, VIRTIO_NET_RX_QUEUE) == 1
        _, written = _used_entry(guest_ram, VIRTIO_NET_RX_QUEUE, 0)
        assert _read_guest_frame(guest_ram, addrs[0], written) == frame

    def test_frames_fill_buffers_in_order(self, net: VirtIONet, guest_ram: bytearray) -> None:
        """多个到达帧按顺序填入各缓冲区."""
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        addrs = _post_rx_buffers(guest_ram, 3)
        _kick(net, VIRTIO_NET_RX_QUEUE)

        frames = [_make_frame(bytes([0x40 + i])) for i in range(3)]
        for frame in frames:
            net.deliver_frame(frame)

        assert _used_idx(guest_ram, VIRTIO_NET_RX_QUEUE) == 3
        for i, frame in enumerate(frames):
            _, written = _used_entry(guest_ram, VIRTIO_NET_RX_QUEUE, i)
            assert _read_guest_frame(guest_ram, addrs[i], written) == frame

    def test_rx_queue_not_consumed_without_frame(self, net, guest_ram) -> None:
        """没有到达帧时通知设备不得消费可用项.

        传输层的通用遍历把每个可用项都当作一条待处理的请求消费掉, 接收方向的缓冲区
        必须留在可用环里等帧到达, 故收发两条队列的处理路径不同。
        """
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        _post_rx_buffers(guest_ram, 2)
        _kick(net, VIRTIO_NET_RX_QUEUE)

        assert _used_idx(guest_ram, VIRTIO_NET_RX_QUEUE) == 0
        assert net.mmio.queue_at(VIRTIO_NET_RX_QUEUE).last_avail_idx == 0

    def test_head_descriptor_carries_frame(self, net: VirtIONet, guest_ram: bytearray) -> None:
        """一个缓冲区承载一个帧, 故只用链头描述符.

        未协商拆分接收缓冲区时驱动每帧只挂一个缓冲区; 链头带下一个描述符时设备仍只写
        链头指向的缓冲区, 后续描述符不被写入。
        """
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        head_pa = _make_gpa(_RX_BUF_OFF)
        tail_pa = _make_gpa(_RX_BUF_OFF + 0x800)
        _write_chain(
            guest_ram, VIRTIO_NET_RX_QUEUE, [
                (head_pa, _RX_BUF_SIZE, VRING_DESC_F_WRITE | VRING_DESC_F_NEXT, 1),
                (tail_pa, _RX_BUF_SIZE, VRING_DESC_F_WRITE, 0),
            ],
        )
        _write_avail_entry(guest_ram, VIRTIO_NET_RX_QUEUE, 0, 0)
        _set_avail_idx(guest_ram, VIRTIO_NET_RX_QUEUE, 1)

        frame = _make_frame(b"head-only")
        net.deliver_frame(frame)

        assert _used_idx(guest_ram, VIRTIO_NET_RX_QUEUE) == 1
        _, written = _used_entry(guest_ram, VIRTIO_NET_RX_QUEUE, 0)
        assert _read_guest_frame(guest_ram, head_pa, written) == frame
        assert guest_ram[_to_offset(tail_pa) : _to_offset(tail_pa) + 4] == b"\x00" * 4

    def test_small_buffer_drops_frame_and_keeps_buffer(self, net, guest_ram) -> None:
        """缓冲区放不下时丢弃该帧并保留缓冲区, 供后续较短的帧使用."""
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        addrs = [_post_rx_buffer(guest_ram, 0, buf_len=VIRTIO_NET_HDR_SIZE + 20)]
        _set_avail_idx(guest_ram, VIRTIO_NET_RX_QUEUE, 1)
        _kick(net, VIRTIO_NET_RX_QUEUE)

        net.deliver_frame(_make_frame(b"too-long-for-buffer"))
        assert net.rx_dropped == 1
        assert _used_idx(guest_ram, VIRTIO_NET_RX_QUEUE) == 0

        # 同一个缓冲区仍可用, 较短的帧应被收下
        short = _make_frame(b"ok")
        net.deliver_frame(short)
        assert net.rx_dropped == 1
        assert _used_idx(guest_ram, VIRTIO_NET_RX_QUEUE) == 1
        _, written = _used_entry(guest_ram, VIRTIO_NET_RX_QUEUE, 0)
        assert _read_guest_frame(guest_ram, addrs[0], written) == short

    def test_oversize_frame_dropped(self, net: VirtIONet) -> None:
        """超过以太帧上界的帧按丢弃处理并计数."""
        net.deliver_frame(_make_frame(b"x" * (ETHERNET_FRAME_MAX + 1)))
        assert net.rx_dropped == 1
        assert net.rx_pending == 0

    def test_pending_queue_bounded(self, net: VirtIONet) -> None:
        """无缓冲区时待收帧数有上界, 超出即丢弃并计数."""
        frame = _make_frame(b"fill")
        for _ in range(VIRTIO_NET_RX_QUEUE_LIMIT + 5):
            net.deliver_frame(frame)

        assert net.rx_pending == VIRTIO_NET_RX_QUEUE_LIMIT
        assert net.rx_dropped == 5

    def test_out_of_range_rx_head_keeps_avail_entry(self, net, guest_ram) -> None:
        """可用环中的链头索引越界时中止填充并保留该可用项, 遍历有界不自旋."""
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        buf_pa = _make_gpa(_RX_BUF_OFF)
        _write_chain(
            guest_ram, VIRTIO_NET_RX_QUEUE, [(buf_pa, _RX_BUF_SIZE, VRING_DESC_F_WRITE, 0)],
        )
        # 可用环项指向 4000, 大于队列长度 8 与描述符表上限 256
        _write_avail_entry(guest_ram, VIRTIO_NET_RX_QUEUE, 0, 4000)
        _set_avail_idx(guest_ram, VIRTIO_NET_RX_QUEUE, 1)

        net.deliver_frame(_make_frame(b"bad-head"))

        assert _used_idx(guest_ram, VIRTIO_NET_RX_QUEUE) == 0
        assert net.rx_pending == 1, "帧应留在待收队列中"

    def test_non_writable_rx_buffer_skipped(self, net, guest_ram) -> None:
        """接收缓冲区缺少可写标志时中止填充, 不写入该缓冲区."""
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        buf_pa = _make_gpa(_RX_BUF_OFF)
        _write_chain(guest_ram, VIRTIO_NET_RX_QUEUE, [(buf_pa, _RX_BUF_SIZE, 0, 0)])
        _write_avail_entry(guest_ram, VIRTIO_NET_RX_QUEUE, 0, 0)
        _set_avail_idx(guest_ram, VIRTIO_NET_RX_QUEUE, 1)

        net.deliver_frame(_make_frame(b"readonly"))

        assert _used_idx(guest_ram, VIRTIO_NET_RX_QUEUE) == 0
        assert guest_ram[_to_offset(buf_pa) : _to_offset(buf_pa) + 4] == b"\x00" * 4


# ============================================================
#  Tests: 回环后端
# ============================================================


class TestLoopback:
    """回环后端语义测试."""

    def test_transmitted_frame_returns_on_receive_queue(self, net, guest_ram) -> None:
        """默认后端是回环: 设备发出的帧回到自身的接收队列, 首部不改写."""
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        _configure_queue(net, VIRTIO_NET_TX_QUEUE)
        addrs = _post_rx_buffers(guest_ram, 1)
        _kick(net, VIRTIO_NET_RX_QUEUE)

        frame = _make_frame(b"round-trip")
        _submit_tx(guest_ram, frame)
        _kick(net, VIRTIO_NET_TX_QUEUE)

        assert _used_idx(guest_ram, VIRTIO_NET_RX_QUEUE) == 1
        _, written = _used_entry(guest_ram, VIRTIO_NET_RX_QUEUE, 0)
        assert _read_guest_frame(guest_ram, addrs[0], written) == frame

    def test_loopback_does_not_self_propagate(self, net, guest_ram) -> None:
        """接收路径不发送帧, 故一次发送只产出一次接收, 不增殖."""
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        _configure_queue(net, VIRTIO_NET_TX_QUEUE)
        _post_rx_buffers(guest_ram, 4)
        _kick(net, VIRTIO_NET_RX_QUEUE)

        _submit_tx(guest_ram, _make_frame(b"once"))
        _kick(net, VIRTIO_NET_TX_QUEUE)

        assert _used_idx(guest_ram, VIRTIO_NET_RX_QUEUE) == 1
        assert net.rx_pending == 0
        assert _used_idx(guest_ram, VIRTIO_NET_TX_QUEUE) == 1


# ============================================================
#  Tests: 中断投递
# ============================================================


class TestInterrupt:
    """中断投递与确认测试.

    中断源接在 hart 0 的 M context 上: 该 context 归 M 模式, 宿主 S 模式既不写它
    也不读它, 故本类同时锁定设备中断在 M context 上置位而不在 S context 上置位。
    """

    @pytest.fixture
    def net_with_plic(self, mem_read, mem_write) -> tuple[VirtIONet, PLIC]:
        plic = PLIC(num_sources=128, num_contexts=2)
        _enable_plic_m_context(plic, VIRTIO_NET_S_IRQ)
        net = VirtIONet(
            mem_read=mem_read, mem_write=mem_write,
            on_irq=plic.set_irq, irq=VIRTIO_NET_S_IRQ,
        )
        return net, plic

    def test_receive_raises_irq(self, net_with_plic, guest_ram: bytearray) -> None:
        """帧到达后中断源置位, 且只在 M context 上产出 MEIP."""
        net, plic = net_with_plic
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        _post_rx_buffers(guest_ram, 1)
        _kick(net, VIRTIO_NET_RX_QUEUE)
        assert not _plic_pending(plic, VIRTIO_NET_S_IRQ)

        net.deliver_frame(_make_frame())

        assert _mmio_read(net, VIRTIO_MMIO_INTERRUPT_STATUS) & 1 == 1
        assert _plic_pending(plic, VIRTIO_NET_S_IRQ)
        mip = plic.get_pending_mip(0)
        assert mip & _MEIP, "M context 上应产出 MEIP"
        assert not mip & _SEIP, "S context 上不得产出 SEIP"

    def test_transmit_raises_irq(self, net_with_plic, guest_ram: bytearray) -> None:
        """发送完成同样置位中断源."""
        net, plic = net_with_plic
        _configure_queue(net, VIRTIO_NET_TX_QUEUE)
        _submit_tx(guest_ram, _make_frame())

        _kick(net, VIRTIO_NET_TX_QUEUE)

        assert _plic_pending(plic, VIRTIO_NET_S_IRQ)

    def test_ack_lowers_irq(self, net_with_plic, guest_ram: bytearray) -> None:
        """受调试程序确认中断后中断源被拉低, MEIP 随之清除."""
        net, plic = net_with_plic
        _configure_queue(net, VIRTIO_NET_RX_QUEUE)
        _post_rx_buffers(guest_ram, 1)
        _kick(net, VIRTIO_NET_RX_QUEUE)
        net.deliver_frame(_make_frame())
        assert _plic_pending(plic, VIRTIO_NET_S_IRQ)

        _mmio_write(net, VIRTIO_MMIO_INTERRUPT_ACK, 1)

        assert _mmio_read(net, VIRTIO_MMIO_INTERRUPT_STATUS) == 0
        assert not _plic_pending(plic, VIRTIO_NET_S_IRQ)
        assert not plic.get_pending_mip(0) & _MEIP

    def test_no_irq_sink_no_crash(self, net: VirtIONet, guest_ram: bytearray) -> None:
        """未给出中断上报入口时完成不应报错, 中断状态仍照常置位."""
        _configure_queue(net, VIRTIO_NET_TX_QUEUE)
        _submit_tx(guest_ram, _make_frame())
        _kick(net, VIRTIO_NET_TX_QUEUE)
        assert _mmio_read(net, VIRTIO_MMIO_INTERRUPT_STATUS) & 1 == 1

    def test_two_instances_do_not_share_irq(self, mem_read, mem_write, guest_ram) -> None:
        """两张网卡取不同源号时中断线互不串扰.

        PLIC 的挂起位按源号索引且全部 context 共用, 两张卡同号则一方拉高线时另一方
        已使能的同一个源一并置位。本用例断言宿主侧的源号不同于飞地侧的源号,
        且一张卡拉高线时另一张卡的源保持不挂起。
        """
        assert VIRTIO_NET_M_IRQ != VIRTIO_NET_S_IRQ
        plic = PLIC(num_sources=128, num_contexts=2)
        _enable_plic_m_context(plic, VIRTIO_NET_S_IRQ)
        _enable_plic_m_context(plic, VIRTIO_NET_M_IRQ)
        debuggee = VirtIONet(
            mem_read=mem_read, mem_write=mem_write,
            on_irq=plic.set_irq, irq=VIRTIO_NET_S_IRQ,
        )
        _configure_queue(debuggee, VIRTIO_NET_RX_QUEUE)
        _post_rx_buffers(guest_ram, 1)
        _kick(debuggee, VIRTIO_NET_RX_QUEUE)

        debuggee.deliver_frame(_make_frame())

        assert _plic_pending(plic, VIRTIO_NET_S_IRQ)
        assert not _plic_pending(plic, VIRTIO_NET_M_IRQ)
