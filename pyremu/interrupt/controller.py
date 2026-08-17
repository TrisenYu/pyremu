#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/09 星期二

"""
可扩展中断控制器框架。

定义 InterruptController 抽象基类:
- CLINT (SiFive 兼容) 和未来的 IMSIC (AIA) 均实现此接口
- 提供统一的中断检查、IPI 发送/清除、定时器接口

中断优先级 (RISC-V 规范):
    MEI > MSI > MTI > SEI > SSI > STI
    (Machine External > Machine Software > Machine Timer >
     Supervisor External > Supervisor Software > Supervisor Timer)
"""

from abc import ABC, abstractmethod
from enum import Enum


class IntSource(Enum):
    """中断源 — 按优先级从高到低排列.

    用于中断控制器返回待处理中断时确定优先级.
    """

    MEI = 0  # Machine External Interrupt (优先级最高)
    MSI = 1  # Machine Software Interrupt (IPI)
    MTI = 2  # Machine Timer Interrupt
    SEI = 3  # Supervisor External Interrupt
    SSI = 4  # Supervisor Software Interrupt
    STI = 5  # Supervisor Timer Interrupt


# 中断源 -> mip/sip 位掩码映射
INT_SOURCE_MIP_MASK: dict[IntSource, int] = {
    IntSource.MEI: 1 << 11, # MEIP
    IntSource.MSI: 1 << 3,  # MSIP
    IntSource.MTI: 1 << 7,  # MTIP
    IntSource.SEI: 1 << 9,  # SEIP
    IntSource.SSI: 1 << 1,  # SSIP
    IntSource.STI: 1 << 5,  # STIP
}


class InterruptController(ABC):
    """中断控制器抽象基类.

    子类: CLINT (SiFive 兼容), IMSIC (AIA/MSI).

    每个 hart 独立调用 check_interrupt() 来轮询自己的中断状态。
    """

    @abstractmethod
    def check_interrupt(self, hart_id: int) -> tuple[bool, int, IntSource | None]:
        """检查指定 hart 是否有待处理中断.

        Args:
            hart_id: hart 编号.

        Returns:
            (has_pending: bool, mip_value: int, highest_source: IntSource | None)
            - has_pending: 是否有未处理的中断 (mip_val & mie_val != 0)
            - mip_value: 当前 mip CSR 应读取的值 (各中断挂起位)
            - highest_source: 优先级最高的待处理中断源 (用于向量模式)
        """
        pass

    @abstractmethod
    def send_ipi(self, target_hart_id: int) -> None:
        """向目标 hart 发送核间中断 (IPI).

        Args:
            target_hart_id: 目标 hart 编号.
        """
        pass

    @abstractmethod
    def clear_ipi(self, hart_id: int) -> None:
        """清除指定 hart 的软件中断挂起位.

        Args:
            hart_id: hart 编号.
        """
        pass

    @abstractmethod
    def get_mtime(self) -> int:
        """获取全局单调时钟计数值 (mtime)."""
        pass

    @abstractmethod
    def tick(self, cycles: int = 1) -> None:
        """推进全局时钟 *cycles* 个周期 (用于定时器中断)."""
        pass

    @abstractmethod
    def set_mtimecmp(self, hart_id: int, val: int) -> None:
        """设置指定 hart 的定时器比较值 (SSTC stimecmp CSR 同步)."""
        pass

    @abstractmethod
    def get_mtimecmp(self, hart_id: int) -> int:
        """返回指定 hart 的 mtimecmp 裸值."""
        pass

    @abstractmethod
    def get_next_timer_wakeup(self, hart_id: int) -> int:
        """返回 hart 的下一次定时器唤醒时间 (mtime 值); 0 = 无定时器.

        供中断缓存快速路径: 若 mtime < wakeup, 可跳过全量中断检查.
        """
        pass
