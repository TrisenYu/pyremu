#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""测试 pyremu.debug.cli — 参数解析、固件加载与启动流程."""

import argparse
from pathlib import Path
import types

from loguru import logger
import pytest

from pyremu.debug import cli, Debugger
from pyremu.debug.utils import RAM_MUL
from pyremu.emulator import Emulator
from pyremu.utils.parse_bin import FirmwareImage, FirmwareSegment

RAM_BASE = 0x8000_0000
RAM_SIZE = 0x0040_0000  # 4 MiB, 容纳默认 DTB 与 initramfs 地址算式


# ============================================================
#  测试装置
# ============================================================


class _FakeBus:
    """记录 write_ram_direct 调用的总线替身."""

    def __init__(self) -> None:
        self.direct_writes: list[tuple[int, bytes]] = []

    def write_ram_direct(self, addr: int, data: bytes) -> None:
        self.direct_writes.append((addr, data))


class _FakeEmu:
    """记录设备树、initramfs 与内核加载调用的 Emulator 替身."""

    def __init__(self) -> None:
        self.dtb_calls: list[int] = []
        self.dtb_file_calls: list[tuple[str, int | None]] = []
        self.dtb_blob_calls: list[tuple[int, bytes]] = []
        self.initrd_calls: list[tuple[str, int]] = []
        self.bus = _FakeBus()

    def load_dtb(self, addr: int) -> None:
        self.dtb_calls.append(addr)

    def load_dtb_file(self, path: str, addr: int | None = None) -> None:
        self.dtb_file_calls.append((path, addr))

    def load_dtb_blob(self, addr: int, blob: bytes) -> None:
        self.dtb_blob_calls.append((addr, blob))

    def load_initrd(self, path: str, addr: int) -> types.SimpleNamespace:
        self.initrd_calls.append((path, addr))
        return types.SimpleNamespace(start=addr, end=addr + 0x100)


def _ns(**kwargs) -> argparse.Namespace:
    """按解析器的默认值建立命名空间, 再用 *kwargs* 覆盖."""
    ns = cli._build_arg_parser().parse_args(["fw.bin"])
    for key, val in kwargs.items():
        setattr(ns, key, val)
    return ns


def _make_emu(ram_size: int = 0x40000) -> Emulator:
    """建立 RAM 足以容纳默认预加载地址的 Emulator."""
    emu = Emulator(num_harts=1, ram_size=ram_size, ram_base=RAM_BASE)
    emu._termio = None
    return emu


def _raw_firmware(tmp_path: Path, data: bytes = b"\x13\x00\x00\x00") -> str:
    path = tmp_path / "fw.bin"
    path.write_bytes(data)
    return str(path)


@pytest.fixture(autouse=True)
def _restore_logger():
    """用例结束后清除本文件配置的 loguru 处理器, 避免影响其它用例."""
    yield
    logger.remove()


# ============================================================
#  参数解析
# ============================================================


class TestBuildArgParser:
    def test_defaults(self):
        ns = cli._build_arg_parser().parse_args(["fw.bin"])
        assert ns.firmware == "fw.bin"
        assert ns.base_addr == RAM_BASE
        assert ns.prog_cnt is None
        assert ns.ram_base == RAM_BASE
        assert ns.fdt == -1
        assert ns.fdt_file is None
        assert ns.no_fdt is False
        assert ns.harts == 1
        assert ns.ram == "128M"
        assert ns.log_level == "INFO"
        assert ns.preload is None
        assert ns.sym is None
        assert ns.kernel is None
        assert ns.disk is None
        assert ns.hart_logs is None

    def test_numeric_options_accept_hex(self):
        ns = cli._build_arg_parser().parse_args(
            [
                "fw.bin",
                "--base-addr",
                "0x80000000",
                "--prog-cnt",
                "0x1000",
                "--ram-base",
                "0x90000000",
                "--preload-addr",
                "0x40",
                "--kernel-addr",
                "0x1000",
                "--initrd-addr",
                "0x2000",
                "--sym-base",
                "0x3000",
            ]
        )
        assert ns.base_addr == 0x8000_0000
        assert ns.prog_cnt == 0x1000
        assert ns.ram_base == 0x9000_0000
        assert ns.preload_addr == 0x40
        assert ns.kernel_addr == 0x1000
        assert ns.initrd_addr == 0x2000
        assert ns.sym_base == 0x3000

    def test_fdt_flag_without_value_is_minus_one(self):
        ns = cli._build_arg_parser().parse_args(["fw.bin", "--fdt"])
        assert ns.fdt == -1

    def test_fdt_flag_with_value_is_parsed(self):
        ns = cli._build_arg_parser().parse_args(["fw.bin", "--fdt", "0x1000"])
        assert ns.fdt == 0x1000

    def test_fdt_flag_with_empty_value_is_none(self):
        ns = cli._build_arg_parser().parse_args(["fw.bin", "--fdt", ""])
        assert ns.fdt is None

    def test_fdt_file_keeps_path_and_address(self):
        ns = cli._build_arg_parser().parse_args(["fw.bin", "--fdt-file", "/tmp/a.dtb:0x40"])
        assert ns.fdt_file == "/tmp/a.dtb:0x40"

    def test_boolean_and_text_options(self):
        ns = cli._build_arg_parser().parse_args(
            [
                "fw.bin",
                "--no-fdt",
                "--harts",
                "4",
                "--ram",
                "256M",
                "--preload",
                "p.bin",
                "--entry-symbol",
                "main",
                "--sym",
                "vmlinux",
                "--kernel",
                "Image",
                "--bootargs",
                "console=ttySIF0",
                "--initrd",
                "initrd.cpio",
                "--disk",
                "root.img",
                "--hart-logs",
                "logs",
            ]
        )
        assert ns.no_fdt is True
        assert ns.harts == 4
        assert ns.ram == "256M"
        assert ns.preload == "p.bin"
        assert ns.entry_symbol == "main"
        assert ns.sym == "vmlinux"
        assert ns.kernel == "Image"
        assert ns.bootargs == "console=ttySIF0"
        assert ns.initrd == "initrd.cpio"
        assert ns.disk == "root.img"
        assert ns.hart_logs == "logs"

    def test_unknown_log_level_is_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli._build_arg_parser().parse_args(["fw.bin", "--log-level", "VERBOSE"])
        assert exc_info.value.code == 2
        assert "VERBOSE" in capsys.readouterr().err


# ============================================================
#  工具函数
# ============================================================


class TestSetupLogger:
    def test_level_gates_messages(self, capsys):
        cli._setup_logger("DEBUG")
        logger.debug("调试可见")
        logger.info("信息可见")
        out = capsys.readouterr().out
        assert "调试可见" in out
        assert "信息可见" in out

    def test_higher_level_suppresses_debug(self, capsys):
        cli._setup_logger("WARNING")
        logger.info("信息不可见")
        logger.warning("告警可见")
        out = capsys.readouterr().out
        assert "信息不可见" not in out
        assert "告警可见" in out


class TestParseRamSize:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1024", 1024),
            ("128M", 128 * RAM_MUL["M"]),
            ("128m", 128 * RAM_MUL["M"]),
            ("128MB", 128 * RAM_MUL["M"]),
            ("64K", 64 * RAM_MUL["K"]),
            ("64KB", 64 * RAM_MUL["K"]),
            ("2G", 2 * RAM_MUL["G"]),
            ("2GB", 2 * RAM_MUL["G"]),
        ],
    )
    def test_suffixes(self, text, expected):
        assert cli._parse_ram_size(text) == expected


# ============================================================
#  固件加载
# ============================================================


class TestLoadFirmwareImage:
    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            cli._load_firmware_image(str(tmp_path / "none.bin"), 0, RAM_BASE)
        assert exc_info.value.code == 1

    def test_unparsable_file_exits(self, tmp_path, monkeypatch):
        path = _raw_firmware(tmp_path)
        monkeypatch.setattr(cli, "parse_firmware", lambda *a, **k: None)
        with pytest.raises(SystemExit) as exc_info:
            cli._load_firmware_image(path, 0, RAM_BASE)
        assert exc_info.value.code == 1

    def test_raw_binary_keeps_zero_offset(self, tmp_path):
        path = _raw_firmware(tmp_path, b"\x13\x00\x00\x00")
        image, load_offset = cli._load_firmware_image(path, RAM_BASE, RAM_BASE)
        assert image.format == "raw"
        assert image.entry_point == RAM_BASE
        assert load_offset == 0

    def test_elf_above_ram_base_keeps_zero_offset(self, tmp_path, monkeypatch):
        path = _raw_firmware(tmp_path)
        image = FirmwareImage(
            entry_point=RAM_BASE + 0x1000,
            segments=[FirmwareSegment(vaddr=RAM_BASE + 0x1000, data=b"\x13", memsz=4)],
            format="elf",
        )
        monkeypatch.setattr(cli, "parse_firmware", lambda *a, **k: image)
        _, load_offset = cli._load_firmware_image(path, 0, RAM_BASE)
        assert load_offset == 0

    def test_elf_below_ram_base_shifts_by_ram_base(self, tmp_path, monkeypatch):
        """映像最小虚拟地址低于内存基址时整段搬迁到内存基址."""
        path = _raw_firmware(tmp_path)
        image = FirmwareImage(
            entry_point=0x1000,
            segments=[
                FirmwareSegment(vaddr=0x1000, data=b"\x13", memsz=4),
                FirmwareSegment(vaddr=0x2000, data=b"\x13", memsz=4),
            ],
            format="elf",
        )
        monkeypatch.setattr(cli, "parse_firmware", lambda *a, **k: image)
        _, load_offset = cli._load_firmware_image(path, 0, RAM_BASE)
        assert load_offset == RAM_BASE

    def test_pe_format_keeps_zero_offset(self, tmp_path, monkeypatch):
        path = _raw_firmware(tmp_path)
        image = FirmwareImage(
            entry_point=0x1000,
            segments=[FirmwareSegment(vaddr=0x1000, data=b"\x13", memsz=4)],
            format="pe",
        )
        monkeypatch.setattr(cli, "parse_firmware", lambda *a, **k: image)
        _, load_offset = cli._load_firmware_image(path, 0, RAM_BASE)
        assert load_offset == 0


# ============================================================
#  设备树
# ============================================================


class TestConfigureFdt:
    def test_no_fdt_returns_none(self):
        emu = _FakeEmu()
        assert cli._configure_fdt(emu, _ns(no_fdt=True), RAM_BASE, RAM_SIZE) is None
        assert emu.dtb_calls == []

    def test_explicit_address_is_used(self):
        emu = _FakeEmu()
        addr = cli._configure_fdt(emu, _ns(fdt=0x9000_0000), RAM_BASE, RAM_SIZE)
        assert addr == 0x9000_0000
        assert emu.dtb_calls == [0x9000_0000]

    def test_default_address_sits_below_ram_top(self):
        emu = _FakeEmu()
        addr = cli._configure_fdt(emu, _ns(fdt=-1), RAM_BASE, RAM_SIZE)
        assert addr == RAM_BASE + RAM_SIZE - 0x10000
        assert emu.dtb_calls == [addr]

    def test_address_zero_disables_device_tree(self):
        """``--fdt 0`` 按未给出处理 — 与 ``-1`` 的取值判定不同."""
        emu = _FakeEmu()
        assert cli._configure_fdt(emu, _ns(fdt=0), RAM_BASE, RAM_SIZE) is None
        assert emu.dtb_calls == []

    def test_prebuilt_file_with_address(self):
        emu = _FakeEmu()
        addr = cli._configure_fdt(emu, _ns(fdt_file="/tmp/a.dtb:0x1000"), RAM_BASE, RAM_SIZE)
        assert addr == 0x1000
        assert emu.dtb_file_calls == [("/tmp/a.dtb", 0x1000)]

    def test_prebuilt_file_with_address_zero(self):
        """地址 0 是合法入参, 不能被当作未给出."""
        emu = _FakeEmu()
        addr = cli._configure_fdt(emu, _ns(fdt_file="/tmp/a.dtb:0"), RAM_BASE, RAM_SIZE)
        assert addr == 0
        assert emu.dtb_file_calls == [("/tmp/a.dtb", 0)]

    def test_prebuilt_file_without_address_uses_default(self):
        emu = _FakeEmu()
        addr = cli._configure_fdt(emu, _ns(fdt_file="/tmp/a.dtb"), RAM_BASE, RAM_SIZE)
        assert addr == RAM_BASE + RAM_SIZE - 0x10000
        assert emu.dtb_file_calls == [("/tmp/a.dtb", addr)]

    def test_prebuilt_file_path_containing_colon(self):
        """路径中的冒号按最后一个分隔, 使含冒号的路径仍可带地址."""
        emu = _FakeEmu()
        addr = cli._configure_fdt(emu, _ns(fdt_file="/tmp/a:b.dtb:0x2000"), RAM_BASE, RAM_SIZE)
        assert addr == 0x2000
        assert emu.dtb_file_calls == [("/tmp/a:b.dtb", 0x2000)]

    def test_no_option_at_all_returns_none(self):
        emu = _FakeEmu()
        assert cli._configure_fdt(emu, _ns(fdt=None), RAM_BASE, RAM_SIZE) is None


# ============================================================
#  预加载、入口符号、initramfs 与内核
# ============================================================


class TestSetupPreload:
    def test_absent_preload_returns_none(self):
        emu = _make_emu()
        assert cli._setup_preload(emu, _ns(preload=None), None, RAM_BASE) is None

    def test_shellcode_is_injected_and_misa_set(self, tmp_path):
        emu = _make_emu()
        code = b"\x13\x00\x00\x00" * 4
        src = tmp_path / "crt0.bin"
        src.write_bytes(code)
        entry = cli._setup_preload(emu, _ns(preload=str(src)), None, RAM_BASE)
        assert entry == (RAM_BASE + emu.bus.ram_size - 0x20000) & ~0xF
        assert emu.bus.try_read(entry, len(code)) == code
        assert emu.harts[0].csrs["misa"].val == (2 << 62) | (1 << 18) | (1 << 20)

    def test_explicit_address_is_used(self, tmp_path):
        emu = _make_emu()
        src = tmp_path / "crt0.bin"
        src.write_bytes(b"\x13\x00\x00\x00")
        addr = RAM_BASE + 0x1000
        entry = cli._setup_preload(
            emu, _ns(preload=str(src), preload_addr=addr), None, RAM_BASE
        )
        assert entry == addr
        assert emu.bus.try_read(addr, 4) == b"\x13\x00\x00\x00"

    def test_missing_shellcode_propagates(self, tmp_path):
        emu = _make_emu()
        with pytest.raises(FileNotFoundError):
            cli._setup_preload(emu, _ns(preload=str(tmp_path / "none.bin")), None, RAM_BASE)

    def test_fdt_present_rebuilds_blob_below_default_top(self, tmp_path, monkeypatch):
        """给出设备树地址时, 预加载之后按原地址重建一份 blob 供固件取用."""
        emu = _make_emu()
        src = tmp_path / "crt0.bin"
        src.write_bytes(b"\x13\x00\x00\x00")
        blobs: list[tuple[int, bytes]] = []
        monkeypatch.setattr(emu, "build_dtb", lambda: b"DTB")
        monkeypatch.setattr(
            emu, "load_dtb_blob", lambda addr, blob: blobs.append((addr, blob))
        )
        cli._setup_preload(emu, _ns(preload=str(src)), 0x87FF0000, RAM_BASE)
        assert blobs == [(RAM_BASE + 0x2200000, b"DTB")]


class TestSetupEntrySymbol:
    def _image(self) -> FirmwareImage:
        return FirmwareImage(
            entry_point=0x1000,
            segments=[FirmwareSegment(vaddr=0x1000, data=b"\x13", memsz=4)],
            format="elf",
            symbols={"main": 0x2000},
        )

    def test_symbol_address_is_written_to_a0(self):
        emu = _make_emu()
        target = cli._setup_entry_symbol(emu, self._image(), "main")
        assert target == 0x2000
        assert emu.harts[0].gprs[10] == 0x2000

    def test_missing_symbol_exits(self):
        emu = _make_emu()
        with pytest.raises(SystemExit) as exc_info:
            cli._setup_entry_symbol(emu, self._image(), "no_such_symbol")
        assert exc_info.value.code == 1


class TestLoadInitrd:
    def test_absent_initrd_returns_before_reading(self):
        emu = _FakeEmu()
        cli._load_initrd(emu, None, None, RAM_BASE, RAM_SIZE)
        assert emu.initrd_calls == []

    def test_default_address_is_aligned_below_dtb(self, tmp_path):
        emu = _FakeEmu()
        size = 0x300000
        src = tmp_path / "initrd.cpio"
        src.write_bytes(b"\x00" * size)
        cli._load_initrd(emu, str(src), None, RAM_BASE, RAM_SIZE)
        expected = (RAM_BASE + RAM_SIZE - 0x10000 - size) & ~0x1FFFFF
        assert emu.initrd_calls == [(str(src), expected)]
        assert expected % 0x200000 == 0

    def test_explicit_address_is_used(self, tmp_path):
        emu = _FakeEmu()
        src = tmp_path / "initrd.cpio"
        src.write_bytes(b"\x00" * 16)
        cli._load_initrd(emu, str(src), 0x8100_0000, RAM_BASE, RAM_SIZE)
        assert emu.initrd_calls == [(str(src), 0x8100_0000)]


class TestLoadKernelImage:
    def test_absent_kernel_returns_none(self):
        emu = _FakeEmu()
        assert cli._load_kernel_image(emu, None, None, 0x8100_0000) is None
        assert emu.bus.direct_writes == []

    def test_default_address_is_used(self, tmp_path):
        emu = _FakeEmu()
        src = tmp_path / "Image"
        src.write_bytes(b"KERNEL")
        result = cli._load_kernel_image(emu, str(src), None, 0x8100_0000)
        assert result == (str(src), 0x8100_0000)
        assert emu.bus.direct_writes == [(0x8100_0000, b"KERNEL")]

    def test_explicit_address_overrides_default(self, tmp_path):
        emu = _FakeEmu()
        src = tmp_path / "Image"
        src.write_bytes(b"KERNEL")
        result = cli._load_kernel_image(emu, str(src), 0x8200_0000, 0x8100_0000)
        assert result == (str(src), 0x8200_0000)


# ============================================================
#  入口
# ============================================================


@pytest.fixture
def repl_capture(monkeypatch):
    """把 Debugger.repl 换成记录实例的替身, 使入口用例不进入交互循环."""
    captured: dict = {}

    def fake_repl(self) -> None:
        captured["dbg"] = self

    monkeypatch.setattr(Debugger, "repl", fake_repl)
    return captured


class TestDebuggerEntry:
    def test_raw_firmware_starts_at_entry(self, tmp_path, repl_capture):
        fw = _raw_firmware(tmp_path)
        cli.debugger([fw, "--ram", "4M", "--no-fdt", "--base-addr", hex(RAM_BASE)])
        dbg = repl_capture["dbg"]
        assert dbg._emu.num_harts == 1
        assert dbg.hart.pc == RAM_BASE
        assert dbg._fdt_addr is None
        assert dbg._load_offset == 0
        assert dbg._preload_path is None

    def test_prog_cnt_overrides_entry(self, tmp_path, repl_capture):
        fw = _raw_firmware(tmp_path)
        cli.debugger(
            [
                fw,
                "--ram",
                "4M",
                "--no-fdt",
                "--base-addr",
                hex(RAM_BASE),
                "--prog-cnt",
                hex(RAM_BASE + 0x400),
            ]
        )
        assert repl_capture["dbg"].hart.pc == RAM_BASE + 0x400

    def test_multiple_harts_share_prog_cnt(self, tmp_path, repl_capture):
        fw = _raw_firmware(tmp_path)
        cli.debugger(
            [
                fw,
                "--ram",
                "4M",
                "--no-fdt",
                "--base-addr",
                hex(RAM_BASE),
                "--harts",
                "3",
            ]
        )
        emu = repl_capture["dbg"]._emu
        assert emu.num_harts == 3
        assert [h.pc for h in emu.harts] == [RAM_BASE] * 3

    def test_default_fdt_address_is_recorded(self, tmp_path, repl_capture):
        fw = _raw_firmware(tmp_path)
        cli.debugger([fw, "--ram", "4M", "--base-addr", hex(RAM_BASE), "--fdt"])
        dbg = repl_capture["dbg"]
        assert dbg._fdt_addr == RAM_BASE + RAM_SIZE - 0x10000
        assert dbg._emu.harts[0].gprs[11] == dbg._fdt_addr

    def test_preload_takes_precedence_over_entry_symbol(self, tmp_path, repl_capture):
        fw = _raw_firmware(tmp_path)
        src = tmp_path / "crt0.bin"
        src.write_bytes(b"\x13\x00\x00\x00")
        addr = RAM_BASE + 0x1000
        cli.debugger(
            [
                fw,
                "--ram",
                "4M",
                "--no-fdt",
                "--base-addr",
                hex(RAM_BASE),
                "--preload",
                str(src),
                "--preload-addr",
                hex(addr),
            ]
        )
        dbg = repl_capture["dbg"]
        assert dbg.hart.pc == addr
        assert dbg._preload_path == str(src)

    def test_virtio_net_instances_are_enabled(self, tmp_path, repl_capture):
        fw = _raw_firmware(tmp_path)
        cli.debugger([fw, "--ram", "4M", "--no-fdt", "--base-addr", hex(RAM_BASE)])
        emu = repl_capture["dbg"]._emu
        assert emu.virtio_net_s is not None
        assert emu.virtio_net_m is not None
        assert emu.virtio_blk is None

    def test_disk_enables_block_device_and_default_bootargs(self, tmp_path, repl_capture):
        fw = _raw_firmware(tmp_path)
        disk = tmp_path / "root.img"
        disk.write_bytes(b"\x00" * 512)
        cli.debugger(
            [
                fw,
                "--ram",
                "4M",
                "--no-fdt",
                "--base-addr",
                hex(RAM_BASE),
                "--disk",
                str(disk),
            ]
        )
        emu = repl_capture["dbg"]._emu
        assert emu.virtio_blk is not None
        assert "root=/dev/vda ro" in emu._bootargs
        assert "init=/usr/lib/systemd/systemd" in emu._bootargs

    def test_explicit_bootargs_win_over_default(self, tmp_path, repl_capture):
        fw = _raw_firmware(tmp_path)
        disk = tmp_path / "root.img"
        disk.write_bytes(b"\x00" * 512)
        cli.debugger(
            [
                fw,
                "--ram",
                "4M",
                "--no-fdt",
                "--base-addr",
                hex(RAM_BASE),
                "--disk",
                str(disk),
                "--bootargs",
                "console=ttySIF0",
            ]
        )
        assert repl_capture["dbg"]._emu._bootargs == "console=ttySIF0"

    def test_hart_logs_directory_is_installed(self, tmp_path, repl_capture):
        fw = _raw_firmware(tmp_path)
        log_dir = tmp_path / "logs"
        cli.debugger(
            [
                fw,
                "--ram",
                "4M",
                "--no-fdt",
                "--base-addr",
                hex(RAM_BASE),
                "--hart-logs",
                str(log_dir),
            ]
        )
        uart = repl_capture["dbg"]._emu.uart
        assert uart._hart_log_dir == str(log_dir)
        assert log_dir.is_dir()

    def test_kernel_default_address_uses_reserved_size(self, tmp_path, repl_capture):
        fw = _raw_firmware(tmp_path)
        kernel = tmp_path / "Image"
        kernel.write_bytes(b"KERNEL")
        cli.debugger(
            [
                fw,
                "--ram",
                "4M",
                "--no-fdt",
                "--base-addr",
                hex(RAM_BASE),
                "--kernel",
                str(kernel),
            ]
        )
        dbg = repl_capture["dbg"]
        assert dbg._kernel_path == str(kernel)
        assert dbg._kernel_addr == RAM_BASE + cli.configs_gen.FW_RESERVED_MEM_SIZE

    def test_kernel_explicit_address(self, tmp_path, repl_capture):
        fw = _raw_firmware(tmp_path)
        kernel = tmp_path / "Image"
        kernel.write_bytes(b"KERNEL")
        addr = RAM_BASE + 0x200000
        cli.debugger(
            [
                fw,
                "--ram",
                "4M",
                "--no-fdt",
                "--base-addr",
                hex(RAM_BASE),
                "--kernel",
                str(kernel),
                "--kernel-addr",
                hex(addr),
            ]
        )
        dbg = repl_capture["dbg"]
        assert dbg._kernel_addr == addr

    def test_missing_entry_symbol_exits(self, tmp_path, repl_capture):
        fw = _raw_firmware(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            cli.debugger(
                [
                    fw,
                    "--ram",
                    "4M",
                    "--no-fdt",
                    "--base-addr",
                    hex(RAM_BASE),
                    "--entry-symbol",
                    "main",
                ]
            )
        assert exc_info.value.code == 1

    def test_entry_symbol_sets_pc_and_a0(self, tmp_path, monkeypatch, repl_capture):
        """给出入口符号时 hart 从该符号起跑, 并告警未设置 sp/gp."""
        fw = _raw_firmware(tmp_path)
        target = RAM_BASE + 0x40
        image = FirmwareImage(
            entry_point=RAM_BASE,
            segments=[FirmwareSegment(vaddr=RAM_BASE, data=b"\x13\x00\x00\x00", memsz=4)],
            format="raw",
            symbols={"main": target},
        )
        monkeypatch.setattr(cli, "parse_firmware", lambda *a, **k: image)
        cli.debugger(
            [
                fw,
                "--ram",
                "4M",
                "--no-fdt",
                "--base-addr",
                hex(RAM_BASE),
                "--entry-symbol",
                "main",
            ]
        )
        dbg = repl_capture["dbg"]
        assert dbg.hart.pc == target
        assert dbg.hart.gprs[10] == target

    def test_missing_firmware_exits(self, tmp_path, repl_capture):
        with pytest.raises(SystemExit) as exc_info:
            cli.debugger([str(tmp_path / "none.bin"), "--ram", "4M", "--no-fdt"])
        assert exc_info.value.code == 1

    def test_missing_symbol_file_is_reported(self, tmp_path, monkeypatch, repl_capture):
        """符号文件缺失时只报告错误, 不终止调试器."""
        fw = _raw_firmware(tmp_path)
        errs: list[str] = []
        monkeypatch.setattr(Debugger, "_err", lambda self, msg: errs.append(msg), raising=True)
        cli.debugger(
            [
                fw,
                "--ram",
                "4M",
                "--no-fdt",
                "--base-addr",
                hex(RAM_BASE),
                "--sym",
                str(tmp_path / "vmlinux"),
            ]
        )
        assert errs and "符号文件不存在" in errs[0]

    def test_initrd_is_loaded_before_device_tree(self, tmp_path, repl_capture):
        """initramfs 的范围要写入设备树, 故必须先生成 DTB 之前加载."""
        fw = _raw_firmware(tmp_path)
        initrd = tmp_path / "initrd.cpio"
        initrd.write_bytes(b"\x00" * 0x1000)
        cli.debugger(
            [
                fw,
                "--ram",
                "4M",
                "--base-addr",
                hex(RAM_BASE),
                "--fdt",
                "--initrd",
                str(initrd),
            ]
        )
        emu = repl_capture["dbg"]._emu
        expected = (RAM_BASE + RAM_SIZE - 0x10000 - 0x1000) & ~0x1FFFFF
        assert emu._initrd.start == expected
        assert emu._initrd.end == emu._initrd.start + 0x1000


# ============================================================
#  CLI 输出
# ============================================================


class TestCliConsoleOutput:
    def test_help_lists_examples(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli._build_arg_parser().parse_args(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--entry-symbol" in out
        assert "firmware.bin --harts 4 --ram 256M" in out
