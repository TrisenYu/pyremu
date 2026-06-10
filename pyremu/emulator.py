#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 17:00:47
# Last modified at 2026/06/09 星期二

"""
多核 RISC-V 模拟器顶层 — 管理 Bus、CLINT、外设、多个 Hart 的执行循环。

上电后 hart 从可配置的复位向量 (reset vector) 开始执行。
裸金属固件应被直接加载到复位向量对应的物理地址。

平台拓扑通过 PlatformConfig 描述, 外设按配置自动创建并注册。
可选生成 FDT blob (通过 pyfdt) 供固件以 a1 寄存器接收。

Usage:
    from pyremu.platform import PlatformConfig

    cfg = PlatformConfig.sifive_u54()
    emu = Emulator(cfg)
    emu.load_code(addr=0x80000000, code=my_firmware)
    emu.step()        # 所有 hart 各执行一条指令
    emu.run(1000)     # 执行 1000 个周期
"""

from pyfdt.pyfdt import Fdt, FdtNode, FdtPropertyStrings, FdtPropertyWords

from pyremu.core.decoder import Hart
from pyremu.core.trap import TrapType
from pyremu.interrupt.clint import CLINT
from pyremu.memory.bus import Bus
from pyremu.memory.l2cache import L2Cache
from pyremu.peripheral import GPIO, I2C, SPI, UART
from pyremu.platform import PeripheralConfig, PlatformConfig
from pyremu.utils.parse_bin import FirmwareImage


class Emulator:
    """多核 RISC-V 模拟器.

    管理 N 个 hart, 共享总线、中断控制器和外设.
    提供 step / run 执行循环及状态检查辅助方法.
    """

    def __init__(
        self,
        config: PlatformConfig | None = None,
        **kwargs: int,
    ) -> None:
        """根据 *config* 初始化模拟器.

        兼容旧式 API: ``Emulator(num_harts=4, ram_base=0, ...)`` 等价于
        用对应字段构建 Minimal PlatformConfig.
        *config* 传入时忽略 **kwargs.
        """
        if config is None:
            # 兼容旧式 API: 从 kwargs 构建配置
            if kwargs:
                periph_kw = {"clint_base": 0x0200_0000}
                plat_kw: dict[str, object] = {"periph": PeripheralConfig(**periph_kw)}
                for field_name in PlatformConfig.__dataclass_fields__:
                    if field_name in kwargs:
                        plat_kw[field_name] = kwargs[field_name]
                config = PlatformConfig(**plat_kw)
            else:
                config = PlatformConfig.qemu_virt()
        self._cfg = config
        self._cycle = 0
        self._total_instrs = 0

        ram_size = config.ram_size
        ram_base = config.ram_base

        # L2 缓存 — 始终启用
        l2 = L2Cache(size=config.l2_size)

        # 共享总线
        self.bus = Bus(ram_size=ram_size, ram_base=ram_base, l2_cache=l2)

        # CLINT 设备
        self.clint = CLINT(num_harts=config.num_harts)
        self.clint.base_addr = config.periph.clint_base
        self.bus.add_device(self.clint.base_addr, self.clint)

        # 外设 — 按配置创建, base=0 时跳过
        self.uart: UART | None = None
        self.spi: SPI | None = None
        self.i2c: I2C | None = None
        self.gpio: GPIO | None = None
        self._peripherals: dict[str, object] = {}

        p = config.periph
        if p.uart_base:
            self.uart = UART(base=p.uart_base)
            self.bus.add_device(p.uart_base, self.uart)
            self._peripherals["uart"] = self.uart
        if p.spi_base:
            self.spi = SPI(base=p.spi_base)
            self.bus.add_device(p.spi_base, self.spi)
            self._peripherals["spi"] = self.spi
        if p.i2c_base:
            self.i2c = I2C(base=p.i2c_base)
            self.bus.add_device(p.i2c_base, self.i2c)
            self._peripherals["i2c"] = self.i2c
        if p.gpio_base:
            self.gpio = GPIO(base=p.gpio_base)
            self.bus.add_device(p.gpio_base, self.gpio)
            self._peripherals["gpio"] = self.gpio

        # 创建 harts, 注入后端
        self.harts: list[Hart] = []
        for i in range(config.num_harts):
            h = Hart(id=i, pmp_entries=config.pmp_entries)
            h.pc = config.reset_vector
            h.set_memory_backend(self.bus.read, self.bus.write)
            h.bus = self.bus
            h.interrupt_ctrl = self.clint
            self.harts.append(h)

    # ----------------------------------------------------------
    #  平台
    # ----------------------------------------------------------

    @property
    def config(self) -> PlatformConfig:
        return self._cfg

    @property
    def peripherals(self) -> dict[str, object]:
        return self._peripherals

    # ----------------------------------------------------------
    #  设备树 (FDT/DTB)
    # ----------------------------------------------------------

    def build_dtb(
        self,
    ) -> bytes:
        """基于当前平台配置, 通过 pyfdt 构建 DTB blob.

        固件可将此 blob 加载到 RAM 并通过 a1 寄存器接收其地址.
        """
        p = self._cfg.periph

        fdt = Fdt()
        root = FdtNode("")
        fdt.add_rootnode(root)

        # 根节点属性
        root.append(FdtPropertyWords("#address-cells", [2]))
        root.append(FdtPropertyWords("#size-cells", [2]))
        root.append(FdtPropertyStrings("compatible", ["pyremu,riscv64"]))
        root.append(FdtPropertyStrings("model", ["pyremu,rv64ima"]))

        # cpus
        cpus = FdtNode("cpus")
        cpus.append(FdtPropertyWords("#address-cells", [1]))
        cpus.append(FdtPropertyWords("#size-cells", [0]))
        cpus.append(FdtPropertyWords("timebase-frequency", [self._cfg.timebase_freq]))
        for i in range(self._cfg.num_harts):
            cpu = FdtNode(f"cpu@{i}")
            cpu.append(FdtPropertyStrings("device_type", ["cpu"]))
            cpu.append(FdtPropertyWords("reg", [i]))
            cpu.append(FdtPropertyStrings("compatible", ["riscv"]))
            cpu.append(FdtPropertyStrings("riscv,isa", [self._cfg.isa]))
            cpu.append(FdtPropertyStrings("mmu-type", ["riscv,sv39"]))
            cpu.append(FdtPropertyStrings("status", ["okay"]))
            cpus.append(cpu)
        root.append(cpus)

        # memory
        mem = FdtNode("memory")
        mem.append(FdtPropertyStrings("device_type", ["memory"]))
        mem.append(FdtPropertyWords("reg", [
            0, self._cfg.ram_base, 0, self._cfg.ram_size,
        ]))
        root.append(mem)

        # soc simple-bus
        soc = FdtNode("soc")
        soc.append(FdtPropertyWords("#address-cells", [2]))
        soc.append(FdtPropertyWords("#size-cells", [2]))
        soc.append(FdtPropertyStrings("compatible", ["simple-bus"]))
        soc.append(FdtPropertyWords("ranges", []))  # 透传

        # CLINT
        clint_base = p.clint_base
        clint = FdtNode(f"clint@{clint_base:x}")
        clint.append(FdtPropertyStrings("compatible", ["riscv,clint0"]))
        clint.append(FdtPropertyWords("reg", [0, clint_base, 0, 0x10000]))
        clint.append(FdtPropertyWords(
            "interrupts-extended", list(range(self._cfg.num_harts)),
        ))
        soc.append(clint)

        # UART
        if self.uart is not None:
            uart = FdtNode(f"serial@{p.uart_base:x}")
            uart.append(FdtPropertyStrings("compatible", ["sifive,uart0"]))
            uart.append(FdtPropertyWords("reg", [0, p.uart_base, 0, 0x1000]))
            soc.append(uart)

        # SPI
        if self.spi is not None:
            spi = FdtNode(f"spi@{p.spi_base:x}")
            spi.append(FdtPropertyStrings("compatible", ["pyremu,spi0"]))
            spi.append(FdtPropertyWords("reg", [0, p.spi_base, 0, 0x1000]))
            soc.append(spi)

        # I2C
        if self.i2c is not None:
            i2c = FdtNode(f"i2c@{p.i2c_base:x}")
            i2c.append(FdtPropertyStrings("compatible", ["pyremu,i2c0"]))
            i2c.append(FdtPropertyWords("reg", [0, p.i2c_base, 0, 0x1000]))
            soc.append(i2c)

        # GPIO
        if self.gpio is not None:
            gpio = FdtNode(f"gpio@{p.gpio_base:x}")
            gpio.append(FdtPropertyStrings("compatible", ["pyremu,gpio0"]))
            gpio.append(FdtPropertyWords("reg", [0, p.gpio_base, 0, 0x1000]))
            soc.append(gpio)

        root.append(soc)
        return fdt.to_dtb()

    def load_dtb(
        self,
        addr: int,
    ) -> None:
        """生成 DTB, 写入 RAM, 并将地址写入所有 hart 的 a1 (x11).

        遵循 RISC-V 引导约定: firmware 入口时 a1 指向设备树 blob.
        """
        dtb = self.build_dtb()
        self.load_code(addr, dtb)
        for hart in self.harts:
            hart.write_gpr(11, addr)

    # ----------------------------------------------------------
    #  代码加载
    # ----------------------------------------------------------

    def load_code(
        self,
        addr: int,
        code: bytes,
    ) -> None:
        """将机器码写入物理 RAM 的指定地址.

        裸金属程序应加载到复位向量对应的地址.
        """
        self.bus.write(addr, code)

    def load_firmware(
        self,
        image: FirmwareImage,
    ) -> None:
        """加载由 parse_firmware() 解析得到的固件镜像.

        将镜像的所有内存段写入物理 RAM, 并将所有 hart
        的 PC 设置为镜像的入口地址.

        对于 raw binary, 若入口地址与复位向量不同,
        调用者应在解析时指定 base_addr=reset_vector.
        """
        for seg in image.segments:
            self.bus.write(seg.vaddr, seg.data)
            # 若 memsz > 文件数据长度, 剩余部分零填充
            if seg.memsz > len(seg.data):
                zero_pad = seg.memsz - len(seg.data)
                self.bus.write(seg.vaddr + len(seg.data), b"\x00" * zero_pad)

        # 将所有 hart 的 PC 设置为入口地址
        for hart in self.harts:
            hart.pc = image.entry_point

    # ----------------------------------------------------------
    #  执行
    # ----------------------------------------------------------

    _TRAP_LOOP_THRESHOLD = 3  # 连续 trap 超过此次数视为不可恢复

    @staticmethod
    def _hex(v: int) -> str:
        return f"0x{v:016x}"

    def _dump_hart_state(self, hart: Hart) -> str:
        """构建 hart 状态转储字符串 (GPR + CSR) — 不直接输出."""
        sep = "=" * 70
        lines = [
            sep,
            f"  Hart {hart.id} 进入不可恢复陷态 — 状态转储",
            sep,
            f"  PC       = {self._hex(hart.pc)}",
            f"  Mode     = {hart.mode.name}",
            f"  mstatus  = {self._hex(hart.mstatus_val)}",
            f"  mtvec    = {self._hex(hart.mtvec_val)}",
            f"  mepc     = {self._hex(hart.mepc_val)}",
            f"  mcause   = {self._hex(hart.mcause_val)}",
            f"  mtval    = {self._hex(hart.mtval_val)}",
            f"  satp     = {self._hex(hart.satp_val)}",
            sep + " GPRs",
        ]
        for i in range(16):
            lo = hart.gprs[i]
            hi = hart.gprs[i + 16]
            lines.append(
                f"  {lo.name:<3} {lo.alias:<5} = {self._hex(lo.val)}  "
                f"{hi.name:<3} {hi.alias:<5} = {self._hex(hi.val)}"
            )
        lines.append(sep)
        return "\n".join(lines)

    # ----------------------------------------------------------
    #  执行
    # ----------------------------------------------------------

    def step(self) -> int:
        """所有 hart 各执行一条指令 (round-robin).

        每条指令执行后检查中断, 每个周期推进 CLINT 时钟.
        若 hart 进入不可恢复的陷态 (连续 trap 超过阈值),
        则转储全部寄存器状态并暂停该 hart.

        Returns:
            本轮执行的指令数.
        """
        executed = 0
        for hart in self.harts:
            if hart._halted:
                continue

            # WFI 等待状态: 不取指/执行, 但仍检查中断唤醒
            if hart._waiting:
                hart.check_pending_interrupts()
                continue

            pc_before = hart.pc

            instr_bytes = self.bus.read(hart.pc, 4)
            instr = int.from_bytes(instr_bytes, "little", signed=False)

            try:
                advance = hart.exec_instr(instr)
            except NotImplementedError:
                hart._take_trap(TrapType.IllInstr, tval=instr, is_interrupt=False)
                advance = 0

            # 正常顺序执行 → 清零连续 trap 计数
            if advance != 0 and hart.pc == pc_before:
                hart.pc = (hart.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF
                hart._consecutive_traps = 0

            # 连续 trap 超过阈值 → 标记为不可恢复
            if hart._consecutive_traps >= self._TRAP_LOOP_THRESHOLD:
                hart._halted = True
                continue

            # 指令边界 — 检查中断
            hart.check_pending_interrupts()
            executed += 1

        self._cycle += 1
        self._total_instrs += executed
        self.clint.tick(1)
        return executed

    def run(self, max_cycles: int) -> int:
        """执行 *max_cycles* 个周期.

        Returns:
            实际执行的周期数.
        """
        for _ in range(max_cycles):
            self.step()
        return self._cycle

    # ----------------------------------------------------------
    #  状态检查辅助 (调试用)
    # ----------------------------------------------------------

    def dump_hart_regs(self, hart_id: int = 0) -> dict:
        """导出指定 hart 的关键寄存器状态."""
        h = self.harts[hart_id]
        return {
            "hart_id": h.id,
            "pc": h.pc,
            "mode": h.mode.name,
            "mstatus": hex(h.mstatus_val),
            "mepc": hex(h.mepc_val),
            "mcause": hex(h.mcause_val),
            "mtval": hex(h.mtval_val),
            "mie": h.mie,
            "gprs": {f"x{i}": hex(h.gprs[i].val) for i in range(32)},
        }

    def dump_memory(self, addr: int, size: int) -> bytes:
        """读取物理内存的 *size* 字节."""
        return self.bus.read(addr, size)

    def mem_hexdump(self, addr: int, size: int) -> str:
        """返回物理内存的十六进制 dump 字符串."""
        data = self.bus.read(addr, size)
        lines = []
        for offset in range(0, len(data), 16):
            chunk = data[offset : offset + 16]
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"{addr + offset:016x}  {hex_part:<48s}  |{ascii_part}|")
        return "\n".join(lines)

    # ----------------------------------------------------------
    #  属性
    # ----------------------------------------------------------

    @property
    def num_harts(self) -> int:
        return self._cfg.num_harts

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def total_instructions(self) -> int:
        return self._total_instrs

    @property
    def reset_vector(self) -> int:
        return self._cfg.reset_vector
