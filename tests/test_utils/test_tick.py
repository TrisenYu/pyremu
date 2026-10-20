#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""测试 pyremu.utils.tick 的 CPU 让出与本地时间格式化."""

import io
import re
import select
import time

import pytest

from pyremu.utils import tick


class TestFmtNowMs:
    def test_format_matches_layout(self):
        """取值为 YYYY/MM/DD/HH:MM:SS.mmm."""
        assert re.fullmatch(r"\d{4}/\d{2}/\d{2}/\d{2}:\d{2}:\d{2}\.\d{3}", tick.fmt_now_ms())

    def test_millisecond_field_is_three_digits(self, monkeypatch):
        """微秒不足一毫秒的整数倍时, 毫秒字段补前导零."""

        class FakeDatetime:
            microsecond = 7000

            def __format__(self, spec: str) -> str:
                return "2026/10/15-00:00:00"

        class FakeDatetimeModule:
            @staticmethod
            def now() -> FakeDatetime:
                return FakeDatetime()

        monkeypatch.setattr(tick, "datetime", FakeDatetimeModule)
        assert tick.fmt_now_ms() == "2026/10/15-00:00:00.007"


class TestYieldCpu:
    def test_select_receives_interval(self, monkeypatch):
        intervals: list[float] = []
        monkeypatch.setattr(select, "select", lambda r, w, x, t: intervals.append(t))
        tick.yield_cpu(0.25)
        assert intervals == [0.25]

    def test_sleep_receives_interval_when_select_fails(self, monkeypatch):
        slept: list[float] = []

        def raise_unsupported(r, w, x, t):
            raise io.UnsupportedOperation("fileno")

        monkeypatch.setattr(select, "select", raise_unsupported)
        monkeypatch.setattr(time, "sleep", slept.append)
        tick.yield_cpu(0.5)
        assert slept == [0.5]

    @pytest.mark.parametrize("exc", [TypeError("bad fd"), OSError("bad fd")])
    def test_sleep_receives_interval_when_fd_invalid(self, monkeypatch, exc):
        """标准输入不可作为 select 的监视目标时改用 sleep 等待."""
        slept: list[float] = []

        def raise_exc(r, w, x, t):
            raise exc

        monkeypatch.setattr(select, "select", raise_exc)
        monkeypatch.setattr(time, "sleep", slept.append)
        tick.yield_cpu(0.125)
        assert slept == [0.125]
