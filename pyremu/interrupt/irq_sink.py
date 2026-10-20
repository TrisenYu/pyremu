#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""设备中断投递目标 — 把设备的中断线电平变化写入中断控制器。

设备经构造参数 ``on_irq`` 持有本对象。该参数不得取作 ``Emulator`` 的绑定方法:
绑定方法持有 Emulator, 而 Emulator 持有设备, 两者互相持有即构成环形引用, 该环
占用的 128 MiB 内存缓冲区因此不由引用计数释放, 只能等循环回收器扫描到它时才
释放。本对象只持有中断控制器、加速执行引擎的通知位与 WFI 唤醒事件, 三者都不
引用 Emulator, 环形引用在构造上不成立。
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyremu._native import FfiExtIrqCtx
    from pyremu.interrupt.aplic import APLIC
    from pyremu.interrupt.plic import PLIC


class DeviceIrqSink:
    """设备中断投递目标"""

    def __init__(self, ext_irq: FfiExtIrqCtx, wake_event: threading.Event) -> None:
        self.ext_irq = ext_irq
        self.wake_event = wake_event
        # 中断源编号与中断控制器的对应关系, 未登记的中断源交给 fallback.
        self.irq_route: dict[int, APLIC | PLIC] = {}
        self.fallback: APLIC | PLIC | None = None

    def bind(self, source: int, ctrl: APLIC | PLIC) -> None:
        """登记一个中断源的控制器."""
        self.irq_route[source] = ctrl

    def set_fallback(self, ctrl: APLIC | PLIC) -> None:
        """登记未在 irq_route 中登记的中断源所用的控制器."""
        self.fallback = ctrl

    def __call__(self, source: int, pending: bool = True) -> None:
        ctrl = self.irq_route.get(source)
        if ctrl is None:
            ctrl = self.fallback
        if ctrl is None:
            return
        ctrl.set_irq(source, pending)
        self.ext_irq.pending = 1
        self.wake_event.set()
