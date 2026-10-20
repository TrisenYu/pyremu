#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""中断控制器框架测试.

覆盖: 中断源的优先级次序与 mip 位掩码 / 中断状态版本号的取模递增 /
``InterruptController`` 的抽象方法集合与子类实现要求.
"""

import pytest

from pyremu.interrupt.controller import (
    INT_SOURCE_MIP_MASK,
    InterruptController,
    IntSource,
    IntStateVersion,
)

# 优先级自高到低: MEI > MSI > MTI > SEI > SSI > STI
PRIORITY_ORDER = [
    IntSource.MEI,
    IntSource.MSI,
    IntSource.MTI,
    IntSource.SEI,
    IntSource.SSI,
    IntSource.STI,
]

EXPECTED_MIP_MASK = {
    IntSource.MEI: 1 << 11,
    IntSource.MSI: 1 << 3,
    IntSource.MTI: 1 << 7,
    IntSource.SEI: 1 << 9,
    IntSource.SSI: 1 << 1,
    IntSource.STI: 1 << 5,
}

ABSTRACT_METHODS = {
    "check_interrupt",
    "send_ipi",
    "clear_ipi",
    "get_mtime",
    "tick",
    "set_mtimecmp",
    "get_mtimecmp",
    "get_next_timer_wakeup",
}


class StubController(InterruptController):
    """实现全部抽象方法的最小控制器, 供接口测试使用."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def check_interrupt(self, hart_id: int) -> tuple[bool, int, IntSource | None]:
        self.calls.append(("check_interrupt", (hart_id,)))
        return (False, 0, None)

    def send_ipi(self, target_hart_id: int) -> None:
        self.calls.append(("send_ipi", (target_hart_id,)))

    def clear_ipi(self, hart_id: int) -> None:
        self.calls.append(("clear_ipi", (hart_id,)))

    def get_mtime(self) -> int:
        self.calls.append(("get_mtime", ()))
        return 0

    def tick(self, cycles: int = 1) -> None:
        self.calls.append(("tick", (cycles,)))

    def set_mtimecmp(self, hart_id: int, val: int) -> None:
        self.calls.append(("set_mtimecmp", (hart_id, val)))

    def get_mtimecmp(self, hart_id: int) -> int:
        self.calls.append(("get_mtimecmp", (hart_id,)))
        return 0

    def get_next_timer_wakeup(self, hart_id: int) -> int:
        self.calls.append(("get_next_timer_wakeup", (hart_id,)))
        return 0


class TestIntSource:
    """中断源的枚举取值与次序."""

    def test_declaration_order_is_priority_order(self):
        """枚举的声明次序即优先级次序, 自高到低."""
        assert list(IntSource) == PRIORITY_ORDER

    def test_enum_value_follows_declaration_order(self):
        """枚举取值随声明次序递增, 故比较取值即比较优先级."""
        values = [source.value for source in PRIORITY_ORDER]
        assert values == sorted(values)
        assert values == list(range(len(PRIORITY_ORDER)))

    def test_highest_and_lowest_source(self):
        """最高优先级为 MEI, 最低为 STI."""
        assert IntSource.MEI.value == 0
        assert IntSource.STI.value == len(PRIORITY_ORDER) - 1

    def test_mip_mask_of_every_source(self):
        """每个中断源对应的 mip 位与该位在规范中的编号一致."""
        assert INT_SOURCE_MIP_MASK == EXPECTED_MIP_MASK

    def test_mip_mask_bits_are_distinct(self):
        """各中断源的位互不重叠, 故一个 mip 取值可唯一分解为中断源集合."""
        bits = list(INT_SOURCE_MIP_MASK.values())
        assert len(set(bits)) == len(bits)


class TestIntStateVersion:
    """中断状态版本号."""

    def test_initial_value_is_zero(self):
        """构造后版本号为 0."""
        assert IntStateVersion().value == 0

    def test_bump_increments_by_one(self):
        """每次递增把版本号加一."""
        version = IntStateVersion()
        version.bump()
        assert version.value == 1
        version.bump()
        assert version.value == 2

    def test_value_stays_within_modulus(self):
        """递增后版本号落在 0 到 MODULUS - 1 的区间内, 到达 MODULUS 时回到 0."""
        modulus = IntStateVersion.MODULUS
        version = IntStateVersion()
        for _ in range(modulus):
            version.bump()
        assert version.value == 0

    def test_two_instances_do_not_share_state(self):
        """两个实例各自计数, 互不影响."""
        first = IntStateVersion()
        second = IntStateVersion()
        first.bump()
        assert (first.value, second.value) == (1, 0)


class TestInterruptControllerInterface:
    """抽象基类的接口约束."""

    def test_abstract_methods_are_the_expected_set(self):
        """抽象方法的集合与中断控制器对外提供的接口一致."""
        assert InterruptController.__abstractmethods__ == frozenset(ABSTRACT_METHODS)

    def test_cannot_be_instantiated_directly(self):
        """抽象基类不能直接实例化."""
        with pytest.raises(TypeError):
            InterruptController()

    def test_missing_one_implementation_still_blocks_instantiation(self):
        """子类少实现一个抽象方法时仍然不能实例化."""

        class Partial(InterruptController):
            def check_interrupt(self, hart_id):
                return (False, 0, None)

        with pytest.raises(TypeError):
            Partial()

    def test_subclass_implementing_all_methods_can_be_instantiated(self):
        """实现全部抽象方法的子类可以实例化, 各接口按子类实现被调用."""
        ctrl = StubController()
        assert ctrl.check_interrupt(0) == (False, 0, None)
        ctrl.send_ipi(1)
        ctrl.clear_ipi(0)
        assert ctrl.get_mtime() == 0
        ctrl.tick()
        ctrl.set_mtimecmp(0, 100)
        assert ctrl.get_mtimecmp(0) == 0
        assert ctrl.get_next_timer_wakeup(0) == 0
        assert [name for name, _ in ctrl.calls] == [
            "check_interrupt",
            "send_ipi",
            "clear_ipi",
            "get_mtime",
            "tick",
            "set_mtimecmp",
            "get_mtimecmp",
            "get_next_timer_wakeup",
        ]

    def test_base_implementations_do_nothing(self):
        """基类的抽象方法体为空: 直接调用基类实现不产生任何动作.

        接口的默认行为不存在, 全部行为由子类给出 —— 该断言锁住这一点.
        """
        ctrl = StubController()
        assert InterruptController.check_interrupt(ctrl, 0) is None
        assert InterruptController.send_ipi(ctrl, 0) is None
        assert InterruptController.clear_ipi(ctrl, 0) is None
        assert InterruptController.get_mtime(ctrl) is None
        assert InterruptController.tick(ctrl) is None
        assert InterruptController.set_mtimecmp(ctrl, 0, 0) is None
        assert InterruptController.get_mtimecmp(ctrl, 0) is None
        assert InterruptController.get_next_timer_wakeup(ctrl, 0) is None

    def test_default_tick_argument_is_one_cycle(self):
        """tick 的默认参数为一个周期."""
        ctrl = StubController()
        ctrl.tick()
        assert ctrl.calls == [("tick", (1,))]
