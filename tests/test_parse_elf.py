#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""固件解析测试: 格式检测, ELF/PE/raw 解析, FirmwareImage."""

from pathlib import Path
import tempfile

import pytest

from pyremu.configs_aux import examed_elf_dir
from pyremu.utils.file_ops import (
    get_abs_filename_arr_from_dir,
    is_file,
)
from pyremu.utils.parse_bin import (
    detect_format,
    FirmwareImage,
    FirmwareSegment,
    parse_firmware,
)

# helper
__reloc_path = lambda x: str(Path(__file__).parent / x)


# ============================================================
#  detect_format — 魔数检测
# ============================================================


class TestDetectFormat:
    """验证文件头魔数识别逻辑."""

    def test_detect_elf_magic(self):
        """\\x7fELF 开头应识别为 elf."""
        with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as f:
            f.write(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64)
            f.flush()
            result = detect_format(f.name)
        Path(f.name).unlink()
        assert result == "elf"

    def test_detect_pe_magic(self):
        """MZ 开头应识别为 pe."""
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"MZ\x90\x00" + b"\x00" * 60)
            f.flush()
            result = detect_format(f.name)
        Path(f.name).unlink()
        assert result == "pe"

    def test_detect_raw_fallback(self):
        """非 ELF/PE 魔数应识别为 raw."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02\x03\x04\x05\x06\x07")
            f.flush()
            result = detect_format(f.name)
        Path(f.name).unlink()
        assert result == "raw"

    def test_detect_empty_file_is_raw(self):
        """空文件也应返回 raw."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"")
            f.flush()
            result = detect_format(f.name)
        Path(f.name).unlink()
        assert result == "raw"


# ============================================================
#  parse_firmware — raw binary
# ============================================================


class TestParseRaw:
    """验证 raw binary 解析."""

    def test_parse_raw_default_base(self):
        """默认基址为 0, 入口为 0."""
        data = b"\x13\x01\x00\x00\x93\x02\x00\x00"
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            f.flush()
            img = parse_firmware(f.name)
        Path(f.name).unlink()

        assert img is not None
        assert img.format == "raw"
        assert img.entry_point == 0
        assert len(img.segments) == 1
        seg = img.segments[0]
        assert seg.vaddr == 0
        assert seg.data == data
        assert seg.memsz == len(data)

    def test_parse_raw_custom_base(self):
        """指定基址后入口和段地址应随之变化."""
        data = b"\x6f\x00\x00\x00"
        base = 0x80000000
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            f.flush()
            img = parse_firmware(f.name, base_addr=base)
        Path(f.name).unlink()

        assert img is not None
        assert img.entry_point == base
        assert img.segments[0].vaddr == base

    def test_parse_raw_non_existent_file_returns_none(self):
        """解析不存在的文件应返回 None (seize_err_if_any)."""
        img = parse_firmware("/nonexistent/file.bin")
        assert img is None


# ============================================================
#  parse_firmware — ELF (需要 bins/elf 下的真实 ELF 文件)
# ============================================================


class TestParseElf:
    """验证 ELF 文件解析."""

    @pytest.fixture
    def elf_paths(self) -> list[str]:
        """获取测试用 ELF 文件列表 (仅真正的 ELF 文件)."""
        elf_dir = examed_elf_dir()
        paths = get_abs_filename_arr_from_dir(elf_dir)
        assert paths is not None and len(paths) > 0, "bins/elf/ 下未找到文件"
        # 仅保留魔数为 ELF 的文件 (排除 .S 源码和 .txt 等)
        return [str(p) for p in paths if is_file(p) and detect_format(p) == "elf"]

    def test_detect_elf_on_real_file(self, elf_paths):
        """真实 ELF 文件应被识别为 elf."""
        assert len(elf_paths) > 0, "未找到真实 ELF 文件"
        for p in elf_paths:
            assert detect_format(p) == "elf", f"{Path(p).name}: expected elf"

    def test_parse_elf_returns_firmware_image(self, elf_paths):
        """parse_firmware 应成功解析 ELF 并返回 FirmwareImage."""
        assert len(elf_paths) > 0, "未找到真实 ELF 文件"
        for p in elf_paths:
            img = parse_firmware(p)
            assert img is not None, f"parse_firmware({Path(p).name}) returned None"
            assert isinstance(img, FirmwareImage)
            assert img.format == "elf"

    def test_parse_elf_has_entry_point(self, elf_paths: str):
        """ELF 应有入口地址 (类型为 int)."""
        assert len(elf_paths) > 0, "未找到真实 ELF 文件"
        for p in elf_paths:
            img = parse_firmware(p)
            assert img is not None
            assert isinstance(img.entry_point, int)

    def test_parse_elf_segment_structure(self, elf_paths: str):
        """ELF 解析结果中各段结构合法: vaddr, data, memsz."""
        assert len(elf_paths) > 0, "未找到真实 ELF 文件"
        for p in elf_paths:
            img = parse_firmware(p)
            assert img is not None
            # 可重定位对象 (.o) 没有 program headers -> segments 可为空
            for seg in img.segments:
                assert isinstance(seg, FirmwareSegment)
                assert seg.vaddr >= 0
                # .bss 段 filesz=0, memsz>0 — data 为空是合法的
                assert seg.memsz >= len(seg.data)
                if len(seg.data) == 0:
                    assert seg.memsz > 0, (
                        f"memsz=0 且 data 为空的段无意义, vaddr=0x{seg.vaddr:x}"
                    )

    def test_parse_elf_linked_executable_has_segments(self, elf_paths):
        """链接过的可执行文件应包含 LOAD 段."""
        has_linked = False
        for p in elf_paths:
            img = parse_firmware(p)
            assert img is not None
            if len(img.segments) <= 0:
                continue
            has_linked = True
            # 入口地址应落在某个段内
            code_addrs = {(seg.vaddr, seg.vaddr + seg.memsz) for seg in img.segments}
            assert any(start <= img.entry_point < end for start, end in code_addrs), (
                f"{Path(p).name}: entry 0x{img.entry_point:x} "
                f"不在任何段内, 段范围: {code_addrs}"
            )
        if not has_linked:
            pytest.skip("当前测试数据仅包含可重定位对象 (.o), 无 LOAD 段")


# ============================================================
#  FirmwareImage / FirmwareSegment 数据类
# ============================================================


class TestFirmwareDataClasses:
    """验证数据类的字段和构造."""

    def test_segment_fields(self):
        seg = FirmwareSegment(vaddr=0x1000, data=b"\x13\x00\x00\x00", memsz=0x1000)
        assert seg.vaddr == 0x1000
        assert seg.data == b"\x13\x00\x00\x00"
        assert seg.memsz == 0x1000

    def test_image_fields(self):
        seg = FirmwareSegment(vaddr=0x80000000, data=b"\x6f", memsz=2)
        img = FirmwareImage(
            entry_point=0x80000000,
            segments=[seg],
            format="raw",
        )
        assert img.format == "raw" and img.entry_point == 0x80000000
        assert len(img.segments) == 1 and img.segments[0].vaddr == 0x80000000
