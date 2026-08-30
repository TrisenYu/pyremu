#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""多 hart PMP 隔离回归测试 (native 并发引擎).

锁定修复: native 并发引擎必须为每个 hart 使用独立的 PMP 缓冲切片
(cfg/addr 按 hart_id 偏移 hid*64), 而非所有 hart 共享 hart0 的 PMP。

修复前的错误行为:
- ``_step_native`` 只 marshal ``active[0]._pmp``, 并在批次后把 hart0 的 PMP
  镜像到所有其它 hart。
- ``run_parallel`` 将同一 ``SharedPmpCtx`` 交给每个 hart 线程。
- 多 hart SMP 启动时各 hart 的 OpenSBI warm-boot 并发重写共享 PMP ->
  数据竞争 + 瞬时执行权限丢失 -> 内核取指访问故障 (cause=1)。

这些用例在修复前必失败 (两 hart 的 PMP 被压成同一值), 修复后通过。

run() 无指令配额, 对纯 WFI 无停机固件按设计持续阻塞 (等待 stdin/定时器/看门狗
唤醒), 永不返回。固件需以 ARM semihosting SYS_EXIT 序列 (a0=0x18; slli/ebreak/
srai) 使 run() 经 EXIT_EBREAK 返回; 仅"批次内并发写"用例保留 WFI 作全 hart
屏障, 经设备暂停事件 (notify_processor) 终止 — 见 ``_run_wfi_firmware``。
"""

import ctypes
import struct
import threading

import pytest

from pyremu._native import FfiPmpCtx, native_available, PmpInfo
from pyremu.core.hart import RiscvMode
from pyremu.emulator import Emulator
from pyremu.platform import PeripheralConfig, PlatformConfig

pytestmark = pytest.mark.skipif(
    not native_available(), reason="native 加速库不可用, 跳过 native 并发 PMP 测试"
)

RAM_BASE = 0x8000_0000
# ARM semihosting SYS_EXIT 停机序列: slli x0,x0,0x1f; ebreak; srai x0,x0,7,
# 且 a0 = SH_SYS_EXIT (0x18)。native 引擎仅对带 marker 的 SYS_EXIT 序列经
# exit_reason::EBREAK 停机; 裸 ebreak 保持 NOP。
SEMIHOSTING_EXIT = struct.pack("<III", 0x01F0_1013, 0x0010_0073, 0x4070_5013)
SH_SYS_EXIT = 0x18
# WFI: 批次内并发写测试的"全部 hart 执行完毕"屏障 — WFI_WAIT 仅在所有 hart
# 均进入等待时才退出批次, 保证每个 hart 的 csrw 均已写入各自 PMP 切片.
WFI = 0x1050_0073
# csrw pmpaddr1, x5  ->  csrrw x0, 0x3B1, x5
CSRW_PMPADDR1_X5 = (0x3B1 << 20) | (5 << 15) | (1 << 12) | 0x73


def _run_wfi_firmware(emu: Emulator) -> None:
    """运行 WFI 固件批次, 经设备暂停事件使 run() 返回.

    run() 对全部 hart WFI 等待的情形按设计持续阻塞. 测试只需观测单批次执行
    后的 PMP 状态: 批次完成 (全部 hart 的 csrw 落地 + 进入 WFI) 后, 由外部
    设备暂停请求 (notify_processor — 打断连续执行的唯一通道) 终止 run().
    """
    timer = threading.Timer(0.05, emu.notify_processor)
    timer.start()
    try:
        emu.run()
    finally:
        timer.cancel()


def _make_emu(num_harts: int) -> Emulator:
    cfg = PlatformConfig(
        num_harts=num_harts,
        ram_size=8 * 1024 * 1024,
        ram_base=RAM_BASE,
        prog_cnt=RAM_BASE,
        pmp_entries=16,
        periph=PeripheralConfig(),
    )
    emu = Emulator(cfg)
    return emu


def _set_pmpaddr(hart, idx: int, val: int) -> None:
    hart.csrs[f"pmpaddr{idx}"].val = val
    hart._pmp.invalidate_cache()


def test_per_hart_pmp_roundtrip_not_mirrored():
    """各 hart 的 PMP 经 native 批次后保持独立, 不被镜像成同一值."""
    emu = _make_emu(2)
    # 两个 hart 执行同一段停机序列 (不触碰 PMP), 仅验证 PMP 往返隔离。
    # 以 semihosting SYS_EXIT 序列收尾使 run() 经 EXIT_EBREAK 返回。
    emu.load_code(RAM_BASE, SEMIHOSTING_EXIT)
    for h in emu.harts:
        h.pc = RAM_BASE
        h.mode = RiscvMode.M
        h.write_gpr(10, SH_SYS_EXIT)

    _set_pmpaddr(emu.harts[0], 0, 0xAAAA)
    _set_pmpaddr(emu.harts[1], 0, 0xBBBB)

    emu.run()

    # 修复前: hart1 的 0xBBBB 从未送入 native, 批次后被 hart0 的值镜像覆盖。
    assert emu.harts[0].csrs["pmpaddr0"].val == 0xAAAA
    assert emu.harts[1].csrs["pmpaddr0"].val == 0xBBBB


def test_per_hart_pmp_concurrent_write_isolated():
    """一个 hart 在批次内写 pmpaddr, 只落到自己的切片, 不污染其它 hart."""
    emu = _make_emu(2)
    # 程序: csrw pmpaddr1, x5; 然后 WFI 作全 hart 屏障。
    # 不用 SYS_EXIT 停机: 首个 hart 的 SYS_EXIT 会立即停止整个批次,
    # 抢跑其它 hart 尚未执行的 csrw (已知竞态). WFI 则要求所有 hart 都
    # 进入等待才退出批次, 保证每个 hart 的 csrw 均已落地 — 经设备暂停
    # 事件 (_run_wfi_firmware) 终止 run()。
    prog = struct.pack("<II", CSRW_PMPADDR1_X5, WFI)
    emu.load_code(RAM_BASE, prog)
    for h in emu.harts:
        h.pc = RAM_BASE
        h.mode = RiscvMode.M
    emu.harts[0].write_gpr(5, 0xA000)
    emu.harts[1].write_gpr(5, 0xB000)

    _run_wfi_firmware(emu)

    # 每个 hart 的 csrw 只应写入自己的 PMP 切片。
    assert emu.harts[0].csrs["pmpaddr1"].val == 0xA000
    assert emu.harts[1].csrs["pmpaddr1"].val == 0xB000


def test_four_hart_pmp_all_distinct():
    """4 hart 各持不同 PMP, 经 native 批次后互不干扰 (复现 SMP 竞争场景)."""
    emu = _make_emu(4)
    emu.load_code(RAM_BASE, SEMIHOSTING_EXIT)
    vals = [0x1000, 0x2000, 0x3000, 0x4000]
    for i, h in enumerate(emu.harts):
        h.pc = RAM_BASE
        h.mode = RiscvMode.M
        h.write_gpr(10, SH_SYS_EXIT)
        _set_pmpaddr(h, 1, vals[i])

    emu.run()

    # 修复前: 仅 hart0 的 PMP 被 marshal 并镜像到全部 hart -> 全部压成 0x1000。
    for i, h in enumerate(emu.harts):
        assert h.csrs["pmpaddr1"].val == vals[i], (
            f"hart{i} pmpaddr1=0x{h.csrs['pmpaddr1'].val:x} 期望 0x{vals[i]:x}"
        )


# ------------------------------------------------------------------
#  PmpInfo.num u8 溢出回归 (per-hart 计数 vs 扁平缓冲总长)
# ------------------------------------------------------------------
# 修复前 PmpInfo.num = min(len(cfg), len(addr)) = 每 hart 64 项 * hart 数。
# 4 hart 时 = 256, 写入 FfiPmpCtx.num (c_uint8) 溢出为 0 -> PMP 被静默禁用:
# OpenSBI PMP 探测全为 0 ("Boot HART PMP Count : 0"), pmp_ok(num==0) 放行一切,
# 内核在 4 hart 下 PMP 形同虚设。1~3 hart 因 <=192 未溢出而未暴露。


def _flat_buf(hart_num: int, per_hart: int = 64):
    total = hart_num * per_hart
    return (ctypes.c_uint8 * total)(), (ctypes.c_uint64 * total)()


@pytest.mark.parametrize("hart_num", [1, 2, 3, 4, 8])
def test_pmpinfo_num_is_per_hart_not_flattened(hart_num):
    """PmpInfo.num 必须是每 hart 条目数 (64), 而非扁平缓冲总长 (64*hart_num)."""
    cfg, addr = _flat_buf(hart_num)
    info = PmpInfo(cfg=cfg, addr=addr, hart_num=hart_num)
    assert info.num == 64, f"hart_num={hart_num}: num={info.num} 期望 64"


def test_pmpinfo_num_survives_u8_ffi_at_four_harts():
    """4 hart 的 per-hart num 经 c_uint8 FFI 字段往返仍为 64 (不溢出为 0)."""
    cfg, addr = _flat_buf(4)
    info = PmpInfo(cfg=cfg, addr=addr, hart_num=4)
    ffi = FfiPmpCtx()
    ffi.num = info.num
    # 修复前 info.num=256, 写入 u8 后读回 0 -> PMP 被禁用。
    assert ffi.num == 64, f"FfiPmpCtx.num={ffi.num} 期望 64 (u8 未溢出)"
