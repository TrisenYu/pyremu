#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""UART 多 hart 行缓冲 + 每 hart 日志文件 + 控制台回显所有权测试.

锁定:
- native 并发引擎多核并发写 UART 时, 输出需按 hart 归入各自行缓冲
  (消除交错乱码), 并可定向到 <log_dir>/hart<N>.log。
- 单一输出 owner (QEMU chardev 模型): Rust termio 线程运行期间
  (console_echo=False) Python 行缓冲不重复回显控制台, 仅归档日志。
- TX 环形缓冲的日志消费者 (tx_log_rd) 与 termio 线程排空索引 (tx_drain)
  相互独立, Python 不写 tx_drain。
"""

from __future__ import annotations

import pytest

from pyremu._native import native_available
from pyremu.emulator import Emulator
from pyremu.interrupt.plic import PLIC
from pyremu.peripheral.uart import IP_RXWM, IP_TXWM, UART
from pyremu.platform import PeripheralConfig, PlatformConfig

TXDATA = 0x00


def _make_uart(sink: list[str]) -> UART:
    return UART(base=0x1000_0000, tx_callback=sink.append)


def _write_bytes(uart: UART, text: str) -> None:
    """直接写 TXDATA, 不设 set_writer — UART 字节级自动即时输出."""
    for b in text.encode():
        uart.write(TXDATA, bytes([b]))


def _write_hart_log(uart: UART, hart_id: int, text: str) -> None:
    """直接写 hart 日志文件, 不经 UART 行缓冲."""
    log_f = uart._hart_log_file(hart_id)
    if log_f is not None:
        log_f.write(text)


class TestPerHartLineBuffer:
    """UART 字节级即时输出: 每字节写 TXDATA 立即经 _tx_callback 输出.

    控制台输出为原始字节 (不加 ``[hart N]`` 前缀, 与 QEMU -nographic 一致);
    每 hart 输出分流通过 ``set_hart_log_dir()`` 提供的日志文件查看。
    """

    def test_output_byte_by_byte(self):
        sink: list[str] = []
        uart = _make_uart(sink)
        _write_bytes(uart, "AAA\n")
        _write_bytes(uart, "BBB\n")
        out = "".join(sink)
        assert "AAA\n" in out
        assert "BBB\n" in out

    def test_raw_output_no_prefix(self):
        sink: list[str] = []
        uart = _make_uart(sink)
        _write_bytes(uart, "hello\n")
        _write_bytes(uart, "world\n")
        out = "".join(sink)
        assert "[hart" not in out
        assert "hello\n" in out
        assert "world\n" in out

    def test_per_hart_log_files_isolate_output(self, tmp_path):
        sink: list[str] = []
        uart = _make_uart(sink)
        uart.set_hart_log_dir(str(tmp_path))
        _write_bytes(uart, "a\n")
        _write_bytes(uart, "b\n")
        _write_hart_log(uart, 0, "a\n")
        _write_hart_log(uart, 0, "b\n")
        uart.close_logs()
        assert (tmp_path / "hart0.log").read_text() == "a\nb\n"
        out = "".join(sink)
        assert "[hart" not in out


class TestPerHartLogFiles:
    """set_hart_log_dir: 各 hart 输出另存 hart<N>.log (原始文本, 无 ANSI 标签)."""

    def test_writes_separate_files(self, tmp_path):
        sink: list[str] = []
        uart = _make_uart(sink)
        uart.set_hart_log_dir(str(tmp_path))
        _write_hart_log(uart, 0, "hello from 0\n")
        _write_hart_log(uart, 1, "hello from 1\n")
        uart.close_logs()

        h0 = (tmp_path / "hart0.log").read_text()
        h1 = (tmp_path / "hart1.log").read_text()
        assert h0 == "hello from 0\n"
        assert h1 == "hello from 1\n"

    def test_log_file_has_no_ansi_tag(self, tmp_path):
        sink: list[str] = []
        uart = _make_uart(sink)
        uart.set_hart_log_dir(str(tmp_path))
        _write_hart_log(uart, 0, "clean\n")
        uart.close_logs()
        h0 = (tmp_path / "hart0.log").read_text()
        assert "\033[" not in h0
        assert "[hart 0]" not in h0

    def test_dir_created_if_missing(self, tmp_path):
        sink: list[str] = []
        uart = _make_uart(sink)
        target = tmp_path / "logs" / "sub"
        uart.set_hart_log_dir(str(target))
        _write_hart_log(uart, 3, "deep\n")
        uart.close_logs()
        assert (target / "hart3.log").read_text() == "deep\n"

    def test_disable_stops_file_output(self, tmp_path):
        sink: list[str] = []
        uart = _make_uart(sink)
        uart.set_hart_log_dir(str(tmp_path))
        _write_hart_log(uart, 0, "one\n")
        uart.set_hart_log_dir(None)
        _write_hart_log(uart, 0, "two\n")   # 日志关闭, 不写文件
        _write_bytes(uart, "two\n")          # 但仍走控制台回调
        uart.close_logs()
        h0 = (tmp_path / "hart0.log").read_text()
        assert h0 == "one\n"
        assert "two" in "".join(sink)


class TestConsoleEchoOwnership:
    """单一输出 owner: console_echo=False 时 _tx_callback 不触发,
    os.write 不受约束 (UART 始终自动输出)."""

    def test_echo_disabled_suppresses_callback_only(self):
        sink: list[str] = []
        uart = _make_uart(sink)
        uart.set_console_echo(False)
        uart.write(TXDATA, b"h")  # _tx_callback 不触发, os.write 仍输出
        assert sink == []

    def test_echo_reenabled_restores_callback(self):
        sink: list[str] = []
        uart = _make_uart(sink)
        uart.set_console_echo(False)
        uart.write(TXDATA, b"X")
        uart.set_console_echo(True)
        uart.write(TXDATA, b"Y")
        out = "".join(sink)
        assert "X" not in out
        assert "Y" in out

    def test_echo_disabled_still_writes_hart_logs(self, tmp_path):
        sink: list[str] = []
        uart = _make_uart(sink)
        uart.set_hart_log_dir(str(tmp_path))
        uart.set_console_echo(False)
        _write_hart_log(uart, 2, "logged\n")
        uart.close_logs()
        assert (tmp_path / "hart2.log").read_text() == "logged\n"
        assert sink == []

    def test_callback_gated_by_console_echo(self):
        """_tx_callback 受 console_echo 约束 — 关闭时不触发."""
        sink: list[str] = []
        uart = _make_uart(sink)
        uart.set_console_echo(False)
        uart.write(TXDATA, b"X")
        assert sink == []


@pytest.mark.skipif(
    not native_available(), reason="native 加速库不可用, 跳过 native UART 分流测试",
)
class TestNativeUartPairRouting:
    """native ring buffer 的 (hart_id, byte) 二元组按 hart 分流至日志文件.

    锁定:
    - 每条目 [hid, byte], 日志归档直接从 ring buffer 写 _hart_log_file(hid).
    - 日志消费者持独立读索引 tx_log_rd; tx_drain 归 Rust termio 线程独占,
      Python 侧绝不写入 (修复前双写者竞争导致条目重放/丢失)。
    - 索引为单调 u32, 跨 u32 环绕与跨容量环绕时槽位定位均正确
      (修复前 range(drain, min(tx_wr, tx_cap)) 在 tx_wr 超过容量后失效)。
    """

    def _make_emu(self, tmp_path) -> Emulator:
        cfg = PlatformConfig(
            num_harts=2,
            ram_size=8 * 1024 * 1024,
            ram_base=0x8000_0000,
            prog_cnt=0x8000_0000,
            periph=PeripheralConfig(),
        )
        emu = Emulator(cfg)
        if emu._termio is None:
            pytest.skip("TerminalIO not available (非交互式 TTY 环境)")
        uart = emu.uart
        assert uart is not None
        uart.set_hart_log_dir(str(tmp_path))
        return emu

    def _push_pair(self, emu: Emulator, hid: int, byte: int) -> None:
        termio = emu._termio
        if termio is None:
            pytest.skip("TerminalIO not available (非交互式环境)")
        ecap = termio.TX_CAP // 2
        e = termio.tx_wr.value
        termio.tx_buf[2 * (e % ecap)] = hid
        termio.tx_buf[2 * (e % ecap) + 1] = byte
        termio.tx_wr.value = (e + 1) & 0xFFFF_FFFF

    def test_interleaved_pairs_routed_per_hart(self, tmp_path):
        emu = self._make_emu(tmp_path)
        uart = emu.uart
        assert uart is not None
        # 交错推入: hart0 'H''I''\n', hart1 'Y''O''\n' 逐字节穿插
        seq = [(0, ord("H")), (1, ord("Y")), (0, ord("I")), (1, ord("O")),
               (0, ord("\n")), (1, ord("\n"))]
        for hid, b in seq:
            self._push_pair(emu, hid, b)

        emu._native_flush_uart()
        uart.close_logs()

        assert (tmp_path / "hart0.log").read_text() == "HI\n"
        assert (tmp_path / "hart1.log").read_text() == "YO\n"

    def test_flush_advances_log_index_not_drain(self, tmp_path):
        """日志消费者只推进 tx_log_rd; tx_drain 归 termio 线程独占, Python 不碰."""
        emu = self._make_emu(tmp_path)
        self._push_pair(emu, 0, ord("A"))
        self._push_pair(emu, 0, ord("\n"))
        termio = emu._termio
        assert termio is not None
        assert termio.tx_log_rd == 0
        emu._native_flush_uart()
        # 日志读索引追上写索引 — 所有条目已归档
        assert termio.tx_log_rd == termio.tx_wr.value
        # tx_drain 不被 Python 写入 (termio 线程未运行 ->保持 0)
        assert termio.tx_drain.value == 0

    def test_log_drain_handles_u32_wraparound(self, tmp_path):
        """索引跨 u32 环绕: rd=0xFFFF_FFFE, wr=2 ->4 个条目正确归档.

        修复前 range(drain, min(tx_wr, tx_cap)) 在此场景下为空区间, 条目全丢。
        """
        emu = self._make_emu(tmp_path)
        termio = emu._termio
        assert termio is not None
        ecap = termio.TX_CAP // 2
        start = 0xFFFF_FFFE
        termio._tx_log_rd = start
        termio.tx_wr.value = start
        for b in b"ok\n\n":
            self._push_pair(emu, 0, b)
        assert termio.tx_wr.value == 2  # 已跨 u32 环绕
        # 槽位按 % ecap 定位 (ecap 为 2 的幂, 整除 2^32, 环绕后仍连续)
        assert start % ecap == ecap - 2

        emu._native_flush_uart()
        assert emu.uart is not None
        emu.uart.close_logs()
        assert (tmp_path / "hart0.log").read_text() == "ok\n\n"
        assert termio.tx_log_rd == 2

    def test_log_drain_skips_lapped_entries(self, tmp_path):
        """生产者超圈 (pending > ecap) 时跳到仍有效的最旧条目, 不重复整圈."""
        emu = self._make_emu(tmp_path)
        termio = emu._termio
        assert termio is not None
        ecap = termio.TX_CAP // 2
        # 伪造超圈: 写索引领先读索引 ecap + 4 个条目
        termio._tx_log_rd = 0
        termio.tx_wr.value = ecap + 4
        for i in range(4):
            termio.tx_buf[2 * i] = 0
            termio.tx_buf[2 * i + 1] = ord("a") + i
        emu._native_flush_uart()
        # 读索引应追平写索引 (只消费最后 ecap 个条目, 未卡死/未重复)
        assert termio.tx_log_rd == ecap + 4


# ============================================================
#  TX watermark 电平中断 — 回归
# ============================================================


class TestTxWatermarkInterrupt:
    """TX watermark 电平中断: txcnt>0 时 IP.txwm 恒置位 (FIFO 即时排空模型).

    回归背景: 旧实现 IP.txwm 依赖 Python TXDATA 写路径锁存, 而 Linux 启动后
    TXDATA 写由 Rust 批量引擎 inline 处理 (不经 Python `_write_reg`) →
    IP.txwm 恒为 0; sifive 驱动 start_tx 使能 IE.txwm 后永远等不到中断,
    用户态 tty 输出 (shell 提示符 / 输入回显) 全部滞留内核 TX 环形缓冲,
    仅内核 printk (轮询 console 路径) 可见。且旧 `_update_plic_rx` 只按
    RX 状态拉 PLIC 线, IE.txwm 使能从不触发中断。
    """

    REG_TXCTRL = 0x08
    REG_IE = 0x10
    REG_IP = 0x14

    def _read_ip(self, uart: UART) -> int:
        return int.from_bytes(uart.read(self.REG_IP, 4), "little")

    def test_ip_txwm_set_by_txcnt_without_any_txdata_write(self):
        """txcnt=1 时 IP.txwm 立即可读为 1 — 无需任何 TXDATA 写经过 Python.

        旧行为 (锁存式) 下本测试失败: 未写过 TXDATA ->IP 读 0。
        """
        uart = _make_uart([])
        # Linux sifive 驱动 probe: txctrl = TXEN | (1 << TXCNT_SHIFT)
        uart.write(self.REG_TXCTRL, (0x1 | (1 << 16)).to_bytes(4, "little"))
        assert self._read_ip(uart) & IP_TXWM

    def test_ip_txwm_clear_when_txcnt_zero(self):
        """txcnt=0 ->水位条件永不成立, IP.txwm 读 0."""
        uart = _make_uart([])
        uart.write(self.REG_TXCTRL, (0x1).to_bytes(4, "little"))  # 仅 TXEN
        assert not (self._read_ip(uart) & IP_TXWM)

    def test_ie_txwm_raises_plic_line(self):
        """IE.txwm 使能且水位满足 ->PLIC 中断线拉高 (旧行为: 从不拉高)."""

        plic = PLIC(base_addr=0x0C00_0000, num_sources=4, num_contexts=2)
        uart = UART(base=0x1000_0000, plic=plic, irq=1)
        uart.write(self.REG_TXCTRL, (0x1 | (1 << 16)).to_bytes(4, "little"))
        assert not plic._pending[1], "IE 未使能时不应挂起"
        uart.write(self.REG_IE, (0x1).to_bytes(4, "little"))  # IE.txwm
        assert plic._pending[1], "IE.txwm 使能后 PLIC 线必须拉高"
        # 驱动排空后关闭 IE.txwm ->线拉低
        uart.write(self.REG_IE, (0x0).to_bytes(4, "little"))
        assert not plic._pending[1]

    def test_ip_register_is_read_only(self):
        """SiFive spec: IP 为电平状态只读寄存器, 写入被忽略."""
        uart = _make_uart([])
        uart.write(self.REG_TXCTRL, (0x1 | (1 << 16)).to_bytes(4, "little"))
        uart.write(self.REG_IP, (0xFFFF_FFFF).to_bytes(4, "little"))
        assert self._read_ip(uart) & IP_TXWM, "IP 写入不得清除水位状态"

    def test_ip_combines_tx_and_rx(self):
        """TX 水位与 RX 非空同时成立 ->IP 两位均置位."""
        uart = _make_uart([])
        uart.write(self.REG_TXCTRL, (0x1 | (1 << 16)).to_bytes(4, "little"))
        uart.preload(b"x")
        ip = self._read_ip(uart)
        assert ip & IP_TXWM and ip & IP_RXWM


class TestRxWatermarkInterrupt:
    """RX 触发阈值 rxcnt 位于 rxctrl bits[18:16] (SiFive spec, 与 txcnt 同偏移).

    回归背景: 旧实现误从 bits[2:0] 取 rxcnt — Linux sifive 驱动 probe 写
    rxctrl = RXEN|(0<<16) = 0x1, 旧代码读到 rxcnt=1 (rxen 位), 单字节输入
    (FIFO 占用 1, 不满足 1>1) 永不触发 RX 中断。交互终端中逐字符输入产生
    单字节滞留: 尾字节 (如命令后的 \\n) 永远不被客机读取, 表现为输入冻结
    (zsh 下 UP ARROW 召回命令后回车无响应)。对照 QEMU sifive_uart:
    SIFIVE_UART_GET_RXCNT(rxctrl) = ((rxctrl) >> 16) & 0x7。
    """

    REG_RXCTRL = 0x0C
    REG_IE = 0x10
    REG_IP = 0x14
    RXEN = 0x1

    def _read_ip(self, uart: UART) -> int:
        return int.from_bytes(uart.read(self.REG_IP, 4), "little")

    def test_rxwm_single_byte_after_driver_probe_write(self):
        """驱动 probe 写 rxctrl=RXEN (rxcnt=0) 后, 单字节即置位 IP.rxwm.

        旧行为: rxcnt 误读 bits[2:0]=1 (rxen 位), 单字节不置位 ->本测试失败。
        """
        uart = _make_uart([])
        uart.write(self.REG_RXCTRL, self.RXEN.to_bytes(4, "little"))
        uart.preload(b"x")
        assert self._read_ip(uart) & IP_RXWM

    def test_rxwm_single_byte_raises_plic_line(self):
        """rxctrl=RXEN + IE.rxwm 使能时, 单字节 preload 必须拉高 PLIC 线.

        锁定冻结场景: 尾字节滞留 FIFO 且 PLIC 不挂起 ->客机永不读取。
        """
        plic = PLIC(base_addr=0x0C00_0000, num_sources=4, num_contexts=2)
        uart = UART(base=0x1000_0000, plic=plic, irq=1)
        uart.write(self.REG_RXCTRL, self.RXEN.to_bytes(4, "little"))
        uart.write(self.REG_IE, (0x2).to_bytes(4, "little"))  # IE.rxwm
        uart.preload(b"\n")
        assert plic._pending[1], "单字节到达且阈值满足时 PLIC 线必须拉高"

    def test_rxwm_respects_rxcnt_threshold(self):
        """rxcnt=1 (bits[18:16]) 时: 1 字节不触发, 第 2 字节触发."""
        uart = _make_uart([])
        uart.write(self.REG_RXCTRL, (self.RXEN | (1 << 16)).to_bytes(4, "little"))
        uart.preload(b"a")
        assert not (self._read_ip(uart) & IP_RXWM)
        uart.preload(b"b")
        assert self._read_ip(uart) & IP_RXWM

    def test_rxctrl_write_reevaluates_plic_line(self):
        """降低 rxcnt 使既有 FIFO 内容满足触发条件 ->写 RXCTRL 立即拉高 PLIC.

        旧实现 RXCTRL 写路径不调用 _update_plic_irq ->中断线状态滞后。
        """
        plic = PLIC(base_addr=0x0C00_0000, num_sources=4, num_contexts=2)
        uart = UART(base=0x1000_0000, plic=plic, irq=1)
        uart.write(self.REG_RXCTRL, (self.RXEN | (1 << 16)).to_bytes(4, "little"))
        uart.write(self.REG_IE, (0x2).to_bytes(4, "little"))
        uart.preload(b"a")  # 占用 1, rxcnt=1 ->不满足
        assert not plic._pending[1]
        # 驱动重写 rxcnt=0 ->既有字节立即满足触发条件
        uart.write(self.REG_RXCTRL, self.RXEN.to_bytes(4, "little"))
        assert plic._pending[1], "rxcnt 降低后既有 FIFO 数据必须立即触发中断"
