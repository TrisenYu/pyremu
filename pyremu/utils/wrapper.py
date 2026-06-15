#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created at 2026/06/05 星期五 13:17:20
# Last modified at 2026/06/05 星期五 15:19:09
import sys
from traceback import format_exc
from typing import Any, Callable, NoReturn, Union


def seize_err_if_any(logger_enable: bool = True):
    """
    如果执行过程中有任何错误，该装饰器函数将返回None以顶替预期要返回的结果
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
    如果执行过程中有任何错误，该装饰器函数将立刻终止进程执行
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


def seize_val_err(
    err_msg: str = "",
) -> Callable:
    """装饰器: 捕获被装饰函数中的 ValueError 并以统一方式报告.

    适用于因用户输入无效整数而触发 ValueError 的命令方法 (如调试器 REPL).
    若被装饰函数的第一个参数 (self) 具有 _err 方法 (如 Debugger),
    则通过 _err → Rich Console 输出; 否则降级为 print.

    Args:
        err_msg: 固定的错误描述字符串, ValueError 发生时输出.
    """

    def dec(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except ValueError:
                self_arg = args[0] if args else None
                if err_msg and self_arg is not None and hasattr(self_arg, "_err"):
                    self_arg._err(err_msg)
                elif err_msg:
                    print(f"错误: {err_msg}")
                return None

        return wrapper

    return dec


def silent_on_err(fn):
    """装饰器: 静默忽略被装饰函数中的所有异常, 失败时返回 None.

    适用于 best-effort 操作 (如回滚时的内存写入、调试信息读取),
    失败不应中断主流程.
    """

    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    return wrapper


def print_exc_on_err(fn):
    """装饰器: 捕获异常并通过 Rich Console 打印 traceback, 返回 None.

    适用于调试器命令方法中依赖外部 IO 的操作 (如 bus.read),
    失败时向用户展示完整异常信息.
    要求被装饰函数的第一个参数 (self) 具有 _console (Rich Console).
    """

    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            self_arg = args[0] if args else None
            if self_arg is not None and hasattr(self_arg, "_console"):
                self_arg._console.print_exception()
            else:
                print(format_exc())
            return None

    return wrapper
