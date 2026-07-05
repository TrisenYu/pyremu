# pyremu RISC-V 启动流程

> 基于 SiFive HiFive Unleashed 启动链 (ZSBL -> FSBL -> OpenSBI)
> 最后更新: 2026-06-21

## 总体架构

```
┌──────────────────────────────────────────────────────────┐
│ pyremu Emulator (模拟 HiFive Unleashed)                  │
│                                                          │
│  RAM: 0x80000000 - 0x88000000 (128 MiB)                  │
│  UART: 0x10000000 (SiFive NS16550)                       │
│  CLINT: 0x02000000 (mtime + mtimecmp + MSIP)             │
│                                                          │
│  ┌─────────────────────────────────────────────────┐     │
│  │ ZSBL stub (0x87FE0000, ~40 B)                    │     │
│  │   -> 读 mhartid, 设 a1=DTB, 跳 FSBL               │     │
│  ├─────────────────────────────────────────────────┤     │
│  │ FSBL stub (0x87FE0028, ~100 B)                    │     │
│  │   -> 置 coldboot flag, 设 mtvec,                  │     │
│  │     patch fw_next_mode->0 (冷启动可选),            │     │
│  │     unimp -> trap -> OpenSBI                       │     │
│  ├─────────────────────────────────────────────────┤     │
│  │ DTB (0x87FF0000, ~1 KB)                           │     │
│  │   自动生成 or --fdt-file                          │     │
│  ├─────────────────────────────────────────────────┤     │
│  │ DTB 副本 (0x82200000)                              │     │
│  │   fdt_get_address() 返回值                        │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
│  ┌─────────────────────────────────────────────────┐     │
│  │ OpenSBI (0x80000000, PIE 搬迁)                    │     │
│  │   fw_jump.elf / fw_payload.elf                    │     │
│  │   -> fw_platform_init -> sbi_init -> init_coldboot  │     │
│  │   -> sbi_hart_hang (WFI 空闲)                      │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

## 启动流程 (逐步)

### 阶段 1: Emulator 初始化

1. `Emulator(cfg)` — 创建 Bus, L2 缓存, CLINT, UART, harts
2. `emu.load_firmware(image, load_offset=ram_base)` — PIE 搬迁
3. `emu.load_dtb(addr)` — 生成 DTB 写入 RAM, 设所有 hart a1=DTB
4. Debugger patches:
   - `0x94`: BSS 循环 `blt` -> `j 0x98` (保留 coldboot flag)
   - `misa = (2 << 0) | (1 << 20)` (MXL=RV64 + U-bit)
   - DTB 副本 @ `ram_base + 0x2200000` (`fdt_get_address()`)
5. Preloader 注入 ZSBL+FSBL stub @ `ram_base + ram_size - 128KiB`
6. 设 PC = preload_addr -> 启动

### 阶段 2: ZSBL stub

```
csrrs t0, mhartid, x0     # 读 hart ID
beq   t0, zero, +12       # boot hart 跳过 WFI
wfi                        # 非 boot hart: 等待中断
j     -4                   # 循环
# boot hart:
auipc a1, ...              # a1 = DTB (PC 相对寻址)
auipc t0, ...              # t0 = FSBL 入口
jalr  zero, 0(t0)          # 跳 FSBL
```

### 阶段 3: FSBL stub (warm)

```
# 置位 coldboot flag @ 0x800842F8
imm64 t0, 0x800842F8
imm64 t1, 1
sw    t1, 0(t0)            # *(flag) = 1

# 设 mtvec = OpenSBI 入口 (0x80000000)
imm64 t0, 0x80000000
csrrw x0, mtvec, t0

# a0 = mhartid, a1 = DTB
csrrs a0, mhartid, x0
auipc a1, ...              # DTB (PC 相对)

# unimp -> trap -> mtvec -> OpenSBI
unimp
```

### 阶段 3': FSBL stub (cold — 额外)

比 warm 多一步: patch `fw_next_mode` 返回 0

```
# fw_next_mode @ 0x80000780: li a0,1 -> addi a0,x0,0
imm64 t0, 0x80000780
imm64 t1, 0x00000513       # addi a0, x0, 0
sw    t1, 0(t0)

# fw_next_mode+4: c.ret
imm64 t1, 0x8082           # c.jr ra
sw    t1, 4(t0)
```

### 阶段 4: OpenSBI 启动

```
fw_boot_hart() -> -1 (自己是 boot hart)
_try_lottery (AMOSWAP) -> 抢到
PIE relocation (~7500 步)
_reset_regs (清 GPR, 保留 a0-a4)
BSS zero (patch: 单次迭代 -> flag 存活)
mtvec 设置, mstatus 设置, stack 设置
fw_platform_init(FDT):
  fdt_path_offset("/") -> root
  fdt_path_offset("/cpus") -> hart_count=1
  fdt_parse_hart_id -> hartid
  fdt_driver_init_by_offset -> 初始化 override 驱动
  (CLINT/UART 通过 fdt_driver_init_all 由 coldboot 初始化)
_start_warm -> sbi_init
init_coldboot:
  sbi_domain_init
  sbi_timer_init -> fdt_timer_init -> aclint-mtimer
  sbi_platform_early_init -> generic_early_init -> fdt_serial_init -> sifive_uart
  sbi_boot_print_banner -> UART 输出 Logo
  sbi_boot_print_general / sbi_boot_print_hart -> UART 输出
  sbi_domain_startup
  sbi_hsm_hart_start_finish -> sbi_hart_hang (WFI 空闲)
```

### 阶段 5: WFI 空闲

```
_wfi_loop:
  wfi
  j _wfi_loop
```

OpenSBI 完成初始化, 所有 hart 进入 WFI 等待 SBI ecall 或中断。

## DTB 必须的节点

```
/ {
    chosen { stdout-path = "/soc/serial@10000000"; }
    cpus {
        cpu@0 {
            reg = <0>;
            compatible = "riscv";
            mmu-type = "riscv,sv39";
            riscv,isa = "rv64imac";
            interrupt-controller {
                compatible = "riscv,cpu-intc";
                #interrupt-cells = <1>;
                interrupt-controller;
                phandle = <1>;
            };
        };
    };
    memory@80000000 { reg = <0x0 0x80000000 0x0 0x8000000>; };
    soc {
        clint@2000000 {
            compatible = "riscv,clint0";
            interrupts-extended = <&cpu_intc 3 &cpu_intc 7>;
        };
        serial@10000000 {
            compatible = "sifive,uart0";
            status = "okay";
        };
    };
};
```

## Emulator 修复记录

| Bug | 症状 | 修复 |
|-----|------|------|
| `_sext` 负 Python int | BEQ/BNE 误判 | 规范化到 [0, 2^64) |
| C.ADD -> C.MV | PIE 循环 GOT 未重定位 | bit12 判断 ADD vs MV |
| C.SUB sf=3 未实现 | IllInstr in _scratch_init | 新增 sf=3 分支 |
| DTB CLINT ie 格式 | timer init SBI_ENODEV | phandle 引用 + cpu-intc 子节点 |
| `misa` MXL+U-bit 缺失 | coldboot 被跳过 | debugger 注入 |
| `fw_next_mode`=1 | warm boot 路径 | cold stub patch -> 0 |
| `fdt_get_address`=0x2200000 | serial init 读错 FDT | DTB 副本 @ 正确地址 |
| BSS 零填充清除 flag | init_warmboot 死循环等待 | debugger patch blt->j |

## 编译桩文件

桩文件由汇编源码通过 makefile 构建:

```bash
make -C tests/src-env zsbl_fsbl_stub
# -> tests/bins/firm-bin/zsbl_fsbl_stub_asm.bin      (热启动)
# -> tests/bins/firm-bin/zsbl_fsbl_stub_cold_asm.bin  (冷启动)
```

## 命令行

```bash
# 冷启动 (OpenSBI Logo + 串口输出)
python -m pyremu.debugger --ram-base=0x80000000 \
    --preload tests/bins/firm-bin/zsbl_fsbl_stub_cold_asm.bin \
    tests/bins/elf/custom_opensbi_fw_jump.elf

# 热启动 (无输出, 更快)
python -m pyremu.debugger --ram-base=0x80000000 \
    --preload tests/bins/firm-bin/zsbl_fsbl_stub_asm.bin \
    tests/bins/elf/custom_opensbi_fw_jump.elf
# 使用 --hart=2 启用两个hart
```
