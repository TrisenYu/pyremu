#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 23:19:24
# Last modified at 2026/06/06 星期六 23:27:16
from enum import Enum

TrapType = Enum(
    "TrapType", (
        "InstrAddrMisaligned",
        "InstrAccessFault",
        "IllInstr",
        "Breakpoint",
        "LdAddrMisaligned",
        "LdAccessFault",
        "StAddrMisaligned",
        "StAccessFault",
        "EcallFromUmode",
        "EcallFromSmode",
        "EcallFromMmode",
        "InstrPageFault",
        "LdPageFault",
        "StPageFault",
        "SoftInterrupt",
        "UmodeSoftInterrupt",
        "SmodeSoftInterrupt",
        "MmodeSoftInterrupt",
        "UmodeTimerInterrupt",
        "SmodeTimerInterrupt",
        "MmodeTimerInterrupt",
        "UmodeExternInterrupt",
        "SmodeExternInterrupt",
        "MmodeExternInterrupt"
    )
)
