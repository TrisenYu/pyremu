#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""设备中断投递目标测试.

覆盖: 中断源到控制器的路由登记 / 未登记中断源的回退控制器 / 无控制器时不做投递.
"""

import threading

from pyremu.interrupt.irq_sink import DeviceIrqSink


class StubController:
    """记录收到的中断线电平的控制器桩件."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def set_irq(self, source: int, pending: bool) -> None:
        self.calls.append((source, pending))


class StubExtIrqCtx:
    """加速执行引擎的通知位桩件."""

    def __init__(self) -> None:
        self.pending = 0


class SinkFixture:
    """投递目标与它触及的三样对象."""

    def __init__(self) -> None:
        self.ext_irq = StubExtIrqCtx()
        self.wake_event = threading.Event()
        self.sink = DeviceIrqSink(self.ext_irq, self.wake_event)


class TestIrqRouting:
    """已登记中断源的投递."""

    def test_bound_source_reaches_its_controller(self):
        """已登记的中断源投递到登记时给出的控制器."""
        fixture = SinkFixture()
        ctrl = StubController()
        fixture.sink.bind(3, ctrl)

        fixture.sink(3)

        assert ctrl.calls == [(3, True)]

    def test_pending_flag_and_wake_event_are_set(self):
        """投递后通知位置位, 唤醒事件被置位."""
        fixture = SinkFixture()
        fixture.sink.bind(3, StubController())

        fixture.sink(3)

        assert fixture.ext_irq.pending == 1
        assert fixture.wake_event.is_set()

    def test_pending_false_is_forwarded(self):
        """电平被清除时按清除投递, 通知位与唤醒事件同样置位."""
        fixture = SinkFixture()
        ctrl = StubController()
        fixture.sink.bind(3, ctrl)

        fixture.sink(3, pending=False)

        assert ctrl.calls == [(3, False)]
        assert fixture.ext_irq.pending == 1
        assert fixture.wake_event.is_set()

    def test_source_is_bound_to_only_one_controller(self):
        """同一中断源重复登记时以最后一次为准."""
        fixture = SinkFixture()
        first = StubController()
        second = StubController()
        fixture.sink.bind(3, first)
        fixture.sink.bind(3, second)

        fixture.sink(3)

        assert first.calls == []
        assert second.calls == [(3, True)]


class TestIrqFallback:
    """未登记中断源的回退控制器."""

    def test_unbound_source_uses_fallback(self):
        """未登记的中断源投递到回退控制器."""
        fixture = SinkFixture()
        fallback = StubController()
        fixture.sink.set_fallback(fallback)

        fixture.sink(9)

        assert fallback.calls == [(9, True)]

    def test_bound_source_takes_precedence_over_fallback(self):
        """已登记的中断源不走回退控制器."""
        fixture = SinkFixture()
        bound = StubController()
        fallback = StubController()
        fixture.sink.bind(9, bound)
        fixture.sink.set_fallback(fallback)

        fixture.sink(9)

        assert bound.calls == [(9, True)]
        assert fallback.calls == []

    def test_without_any_controller_nothing_is_delivered(self):
        """既未登记中断源也没有回退控制器时不做投递, 通知位与唤醒事件不变."""
        fixture = SinkFixture()

        fixture.sink(9)

        assert fixture.ext_irq.pending == 0
        assert not fixture.wake_event.is_set()
