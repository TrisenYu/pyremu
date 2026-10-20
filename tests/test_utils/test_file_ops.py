#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""文件路径辅助测试.

覆盖: 逐级拼接路径 / 判定文件与路径是否存在 / 列出一个目录内的文件.
"""

from pathlib import Path

from pyremu.utils.file_ops import (
    get_abs_filename_arr_from_dir,
    is_file,
    is_path_existed,
    path_join,
    reloc_path,
)


class TestRelocPath:
    """逐级拼接路径."""

    def test_single_level(self):
        """一段中间目录拼在源路径之后."""
        assert reloc_path("/base", "sub") == "/base/sub"

    def test_multiple_levels_are_applied_in_order(self):
        """多段中间目录按给出的次序逐级拼接."""
        assert reloc_path("/base", "sub", "deep") == "/base/sub/deep"

    def test_without_intermediate_returns_source(self):
        """不给中间目录时返回源路径本身."""
        assert reloc_path("/base") == "/base"

    def test_accepts_path_like_source(self):
        """源路径可取 Path 对象, 返回值一律为字符串."""
        result = reloc_path(Path("/base"), Path("sub"))
        assert isinstance(result, str)
        assert result == "/base/sub"


class TestPathPredicates:
    """文件与路径的存在性判定."""

    def test_existed_file(self, tmp_path):
        """已创建的文件的路径存在, 且判定为文件."""
        target = tmp_path / "some_file.bin"
        target.write_bytes(b"\x01\x02")
        assert is_path_existed(str(target))
        assert is_file(target)

    def test_missing_path(self, tmp_path):
        """未创建的路径不存在, 也不判定为文件."""
        missing = tmp_path / "no_such_file.bin"
        assert not is_path_existed(str(missing))
        assert not is_file(missing)

    def test_directory_is_not_a_file(self, tmp_path):
        """目录本身存在, 但不判定为文件."""
        assert is_path_existed(str(tmp_path))
        assert not is_file(tmp_path)


class TestListFilesInDir:
    """列出一个目录内的文件."""

    def test_only_files_are_returned(self, tmp_path):
        """目录中的子目录不出现在结果里, 文件全部出现."""
        (tmp_path / "a.bin").write_bytes(b"a")
        (tmp_path / "b.bin").write_bytes(b"b")
        (tmp_path / "sub").mkdir()

        found = get_abs_filename_arr_from_dir(tmp_path)

        assert sorted(p.name for p in found) == ["a.bin", "b.bin"]

    def test_empty_dir_yields_empty_list(self, tmp_path):
        """空目录返回空列表."""
        assert get_abs_filename_arr_from_dir(tmp_path) == []
