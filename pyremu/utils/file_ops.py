#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created at 2026/06/05 星期五 13:16:26
# Last modified at 2026/06/06 星期六 16:48:21
from os import PathLike
from pathlib import Path

from pyremu.utils.wrapper import seize_err_if_any


def reloc_path(src: PathLike, *intermidate: str | PathLike) -> str:
    ret = Path(src)
    for mid in intermidate:
        ret = ret.joinpath(mid)
    return str(ret)


def is_file(path: PathLike | str) -> bool:
    return Path(path).is_file()


def is_path_existed(path: str) -> bool:
    return Path(path).exists()


@seize_err_if_any()
def get_abs_filename_arr_from_dir(dir_path: str) -> list[Path]:
    return [f for f in Path(dir_path).iterdir() if f.is_file()]
