#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created at 2026/06/05 星期五 13:17:20
# Last modified at 2026/06/05 星期五 15:19:09
import sys
from traceback import format_exc
from typing import Any, NoReturn, Union


def seize_err_if_any(logger_enable: bool = True):
    """
    if fn encountered any error, then return None as its result.
    Otherwise, return the expected result(s).
    """

    def dec(fn_with_ret_val):
        def wrapper(*args, **kwargs):
            try:
                return fn_with_ret_val(*args, **kwargs)
            except Exception as e:
                if not logger_enable:
                    return None
                dump_stk = format_exc()
                print(
                    f"an exception was detected: {e}\n" +
                    "current trace stack\n" +
                    dump_stk
                )
            return None

        return wrapper

    return dec


def die_if_err(fn):
    """
    if fn encountered any error,then the whole process
    will terminate as quickly as possible.
    """

    def error_dumper(*args, **kwargs) -> Union[NoReturn, Any]:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            dump_stk = format_exc()
            print(
                f"an exception was detected: {e}\n" +
                "dumping current trace stack\n" +
                dump_stk
            )
            sys.exit(1)

    return error_dumper
