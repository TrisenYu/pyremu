#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""测试 pyremu.utils.wrapper 的异常处理装饰器."""

import pytest

from pyremu.utils.wrapper import (
    die_if_err,
    print_exc_on_err,
    seize_err_if_any,
    seize_val_err,
    silent_on_err,
)


def boom() -> int:
    raise RuntimeError("boom")


class _ErrHost:
    """模拟具有 _err 输出入口的宿主对象, 如 Debugger."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def _err(self, msg: str) -> None:
        self.messages.append(msg)


class _ConsoleRecorder:
    """记录 print_exception 调用次数的替身 Console."""

    def __init__(self) -> None:
        self.print_exception_calls = 0

    def print_exception(self) -> None:
        self.print_exception_calls += 1


class _ConsoleHost:
    """模拟具有 Rich Console 的宿主对象."""

    def __init__(self) -> None:
        self._console = _ConsoleRecorder()


class TestSeizeErrIfAny:
    def test_passes_through_return_value(self):
        @seize_err_if_any()
        def fn() -> int:
            return 7

        assert fn() == 7

    def test_returns_none_and_reports_exception(self, capsys):
        wrapped = seize_err_if_any()(boom)
        assert wrapped() is None
        out = capsys.readouterr().out
        assert "an exception was detected: boom" in out
        assert "RuntimeError" in out

    def test_returns_none_without_report(self, capsys):
        wrapped = seize_err_if_any(logger_enable=False)(boom)
        assert wrapped() is None
        assert capsys.readouterr().out == ""


class TestDieIfErr:
    def test_passes_through_return_value(self):
        @die_if_err
        def fn() -> int:
            return 7

        assert fn() == 7

    def test_exception_terminates_process(self, capsys):
        wrapped = die_if_err(boom)
        with pytest.raises(SystemExit) as exc_info:
            wrapped()
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "an exception was detected: boom" in out
        assert "dumping current trace stack" in out


class TestSeizeValErr:
    def test_passes_through_return_value(self):
        @seize_val_err("bad int")
        def fn(self) -> int:
            return 7

        assert fn(_ErrHost()) == 7

    def test_host_with_err_receives_message(self):
        host = _ErrHost()

        @seize_val_err("bad int")
        def fn(self) -> int:
            raise ValueError("invalid literal for int()")

        assert fn(host) is None
        assert host.messages == ["bad int"]

    def test_host_without_err_reports_by_print(self, capsys):
        class Plain:
            pass

        @seize_val_err("bad int")
        def fn(self) -> int:
            raise ValueError("invalid literal for int()")

        assert fn(Plain()) is None
        assert "错误: bad int" in capsys.readouterr().out

    def test_missing_host_argument_still_reports(self, capsys):
        @seize_val_err("bad int")
        def fn() -> int:
            raise ValueError("invalid literal for int()")

        assert fn() is None
        assert "错误: bad int" in capsys.readouterr().out

    def test_empty_message_is_silent(self, capsys):
        @seize_val_err()
        def fn(self) -> int:
            raise ValueError("invalid literal for int()")

        assert fn(_ErrHost()) is None
        assert capsys.readouterr().out == ""

    def test_other_exception_propagates(self):
        @seize_val_err("bad int")
        def fn(self) -> int:
            raise TypeError("not a value error")

        with pytest.raises(TypeError):
            fn(_ErrHost())


class TestSilentOnErr:
    def test_passes_through_return_value(self):
        @silent_on_err
        def fn() -> int:
            return 7

        assert fn() == 7

    def test_exception_yields_none(self):
        assert silent_on_err(boom)() is None

    def test_keyboard_interrupt_propagates(self):
        """捕获范围限于 Exception, 进程终止类异常仍向上传播."""

        @silent_on_err
        def fn() -> int:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            fn()


class TestPrintExcOnErr:
    def test_passes_through_return_value(self):
        @print_exc_on_err
        def fn() -> int:
            return 7

        assert fn() == 7

    def test_host_with_console_prints_exception(self):
        host = _ConsoleHost()
        assert print_exc_on_err(boom)(host) is None
        assert host._console.print_exception_calls == 1

    def test_host_without_console_prints_traceback(self, capsys):
        class Plain:
            pass

        assert print_exc_on_err(boom)(Plain()) is None
        assert "Traceback (most recent call last)" in capsys.readouterr().out
