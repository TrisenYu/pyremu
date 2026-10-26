#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""LoadOps — 固件、设备树与初始化内存盘的装载.

装载目标为物理地址; 落在引导只读存储器窗口内的内容经 BootRom.load 写入, 其余
写入主存 bytearray 并预热 L2 缓存.
"""

from pathlib import Path

from pyremu.configs_aux import cfg_int
from pyremu.utils import dtb
from pyremu.utils.parse_bin import FirmwareImage

# 不超过该字节数的内容在直写主存之外再经总线写一次, 使 Python 执行路径命中
# L2 缓存; 更大的内容只直写主存. 大段逐缓存行经总线写入会膨胀到数十秒.
_FAST_LOAD_THRESHOLD = cfg_int("FAST_LOAD_THRESHOLD")


class LoadOps:
    """固件、设备树与初始化内存盘的装载步骤."""

    def build_dtb(self) -> bytes:
        """基于当前平台配置, 通过 libfdt 构建 DTB blob.

        固件可将此 blob 加载到物理内存并通过 a1 寄存器接收其地址.
        """
        devices = dtb.DtbDevices(
            uart=self.uart,
            spi=self.spi,
            i2c_gen=self.i2c,
            gpio=self.gpio,
            virtio_blk=self.virtio_blk,
            virtio_net_s=self.virtio_net_s,
            watchdog=self.watchdog,
            crng=self.crng,
            plic=self.plic,
            imsic=self.imsic,
            aplic_s=self.aplic,
        )
        return dtb.build_dtb(
            self._cfg,
            devices=devices,
            bootargs=self._bootargs,
            initrd=self._initrd,
            reserved_ranges=self._cfg.reserved_memory_ranges,
        )

    def load_dtb(
        self,
        addr: int,
    ) -> None:
        """生成 DTB, 写入物理内存, 并将地址写入所有 hart 的 a1 (x11).

        遵循 RISC-V 引导约定: 固件入口时 a1 指向设备树 blob.
        """
        dtb = self.build_dtb()
        self.load_dtb_blob(addr, dtb)

    def load_dtb_file(
        self,
        path: str,
        addr: int | None = None,
    ) -> None:
        """加载预编译的 DTB 文件到物理内存, 并将地址写入所有 hart 的 a1.

        addr 为 None 时取 RAM 顶端减 64 KiB.
        """
        if addr is None:
            addr = self._cfg.ram_base + self._cfg.ram_size - 0x10000
        dtb = Path(path).read_bytes()
        self.load_dtb_blob(addr, dtb)

    def load_dtb_blob(
        self,
        addr: int,
        dtb: bytes,
    ) -> None:
        """将 dtb 写入物理内存的 addr, 并设所有 hart 的 a1."""
        self._dtb_addr = addr
        self._dtb_size = len(dtb)
        self.load_code(addr, dtb)
        for hart in self.harts:
            hart.write_gpr(11, addr)

    # ----------------------------------------------------------
    #  映像数据装载
    # ----------------------------------------------------------

    def load_initrd(
        self,
        path: str,
        addr: int,
    ) -> dtb.Initrd:
        """将 initramfs (cpio[.gz]) 加载到 RAM 的 addr, 记录物理范围.

        返回 Initrd(start, end); 随后 build_dtb 会把该范围写入
        /chosen/linux,initrd-start 与 /chosen/linux,initrd-end, 供内核解包为
        rootfs. 必须在生成 DTB 之前调用, 使该范围进入设备树.
        """
        data = Path(path).read_bytes()
        self.load_code(addr, data)
        self._initrd = dtb.Initrd(start=addr, end=addr + len(data))
        return self._initrd

    def load_code(
        self,
        addr: int,
        code: bytes,
    ) -> None:
        """将机器码写入物理 RAM 或引导只读存储器的指定地址.

        裸金属程序应加载到复位向量对应的地址. 地址落在引导只读存储器的窗口内时
        经 BootRom.load 写入, 不经总线写路径, 也不预热 L2 缓存.

        其余地址直接写入主存 bytearray, 即加速执行引擎读取指令与数据的数据源;
        不超过 _FAST_LOAD_THRESHOLD 的内容另外经总线写一次, 使 Python 执行路径
        命中 L2 缓存. 大段内容逐缓存行经总线写入会膨胀到数十秒, 故只直写主存.
        """
        rom = self.bus.boot_rom
        if rom is not None and rom.contains(addr):
            rom.load(addr - rom.base_addr, code)
            return
        self.bus.write_ram_direct(addr, code)
        if len(code) <= _FAST_LOAD_THRESHOLD:
            self.bus.write(addr, code)

    def _write_segment(self, seg, vaddr_offset: int) -> None:
        """将段的数据写入物理地址 vaddr + vaddr_offset, 并按 memsz 把余下部分填零.

        目标落在引导只读存储器的窗口内时经 BootRom.load 写入; 其余目标写入主存
        bytearray, 即加速执行引擎读取指令与数据的数据源. 不超过
        _FAST_LOAD_THRESHOLD 的写入同时写进 L2 缓存, 供 Python 执行路径读取.
        """
        vaddr = seg.vaddr + vaddr_offset
        rom = self.bus.boot_rom
        if rom is not None and rom.contains(vaddr):
            rom.load(vaddr - rom.base_addr, seg.data)
            if seg.memsz > len(seg.data):
                pad = vaddr + len(seg.data)
                rom.load(pad - rom.base_addr, b"\x00" * (seg.memsz - len(seg.data)))
            return

        # 加速执行引擎直接读取这份 bytearray
        self.bus.write_ram_direct(vaddr, seg.data)
        # 不超过阈值的段另外写一次总线, 使 Python 执行路径命中 L2 缓存
        if len(seg.data) <= _FAST_LOAD_THRESHOLD:
            self.bus.write(vaddr, seg.data)

        if seg.memsz > len(seg.data):
            zero_pad = seg.memsz - len(seg.data)
            zp_addr = vaddr + len(seg.data)
            self.bus.write_ram_direct(zp_addr, b"\x00" * zero_pad)
            if zero_pad <= _FAST_LOAD_THRESHOLD:
                self.bus.write(zp_addr, b"\x00" * zero_pad)

    def load_firmware(
        self,
        image: FirmwareImage | None,
        load_offset: int = 0,
    ) -> None:
        """加载由 parse_firmware() 解析得到的固件镜像.

        将镜像的所有内存段写入物理内存, 并把所有 hart 的 PC 设置为镜像的入口
        地址加 load_offset.

        load_offset 用于 PIE 固件搬迁: ELF 段链接在地址 0 而 RAM 从
        ram_base 开始时, 传入 load_offset=ram_base 即可把各段上移.
        """
        if image is None:
            raise ValueError("载入了无效的镜像")

        for seg in image.segments:
            self._write_segment(seg, load_offset)

        # load_offset 非 0 时施加 PIE 装载偏移: 固件链接在低地址而主存位于高地址,
        # S 模式经 mret 进入时使用链接地址, 该地址也须访问到同一批字节.
        if load_offset != 0 and image.segments:
            for seg in image.segments:
                self._write_segment(seg, 0)
            link_min = min(seg.vaddr for seg in image.segments)
            link_max = max(seg.vaddr + seg.memsz for seg in image.segments)
            self.bus.apply_pie_ofs(link_min, link_max - link_min)

        # 将所有 hart 的 PC 设置为入口地址
        for hart in self.harts:
            hart.pc = image.entry_point + load_offset

        # 装入的段覆盖了既有的内存内容, 两个 TLB 中既有的翻译一律作废: 页表本身
        # 可能就在被覆盖的范围内.
        for hart in self.harts:
            hart.itlb.flush_all()
            hart.dtlb.flush_all()
        self.bus.note_tlb_flush()
