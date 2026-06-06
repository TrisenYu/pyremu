#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/05 星期五 14:30:49
# Last modified at 2026/06/06 星期六 23:18:05
from pathlib import Path
from typing import Optional

import pytest

from pyremu.file_ops import (
    get_abs_filename_arr_from_dir,
    is_file,
    is_path_existed,
    reloc_path,
)
from pyremu.parse_bin import parse_bin

# test path relocation auxilliary function
__reloc_path = lambda x: str(Path(__file__).parent / x)


@pytest.mark.parametrize(
    "x, y",
    [
        (reloc_path(Path(__file__).parent, "bins"), __reloc_path("bins")),
        (reloc_path(str(Path(__file__).parent), "bins"), __reloc_path("bins")),
    ]
)
def test_path_resolvation(x: Optional[str], y: str) -> None:
    assert x is not None and x == y, (
        "expected equal, but found: \n\t" +
        f"reloc_path: {x}\n\t__reloc_path: {y}"
    )


def test_file_reader():
    bin_paths = get_abs_filename_arr_from_dir(reloc_path(Path(__file__).parent, "bins/elf"))
    assert bin_paths is not None and len(bin_paths) > 0, (
        "unable to extract from bins or return empty binary executable files from bins/elf"
    )
    for fpath in bin_paths:
        assert is_file(fpath) is True
        assert is_path_existed(fpath) is True


if __name__ == "__main__":
    prog_paths = get_abs_filename_arr_from_dir(reloc_path(Path(__file__).parent, "bins/elf"))
    assert prog_paths is not None and len(prog_paths) > 0, (
        "unable to extract from given directory or " +
        "return empty executable files list from the path"
    )
    ret = parse_bin(prog_paths[0])
    print(type(ret), "\n(va) sec-name size len")
    for section in ret.sections:
        print(section.name, section.size, len(section.content))

    for i, bp in enumerate(prog_paths):
        print(i)
        cur = parse_bin(bp)
        if cur is None:
            print('-=-=-= skip none -=-=-=')
            continue

        for sec in cur.sections:
            # 有个概念叫做文件偏移量
            # 提供加载基地址后，就可以和这个偏移量相加得到其在内存中的偏移量
            if hasattr(sec, 'virtual_address'):
                print(sec.virtual_address, end=' ')
            print(sec.name, section.size)
