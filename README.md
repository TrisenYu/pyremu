# Pyremu — RISC-V 64 位模拟器

|缩略 |含义       |
|:--:|:--------:|
|Py  | python   |
|r   | riscv    |
|emu | emulator |

Pyremu 是一个用 Python (CPython 3.14) 编写的本地 **RISC-V 指令集模拟器与交互式调试器**，
面向固件调试、TEE开发、操作系统调试等场景

模拟多 hart，支持 RV64 **IMAC** 指令扩展、特权级 (U/S/H/M/D)、
Sv39 虚拟内存、L2 缓存 MESI 一致性协议、CLINT 时钟/核间中断。

## 快速上手

```bash
# 运行全部测试
make test


# 加载零阶段加载器作为上电初始化代码，
# 设置内存基址为0x8000_0000
# 同时以交互式调试器加载固件
python -m pyremu.debugger \
--ram-base=0x80000000     \
--preload tests/bins/firm-bin/zsbl_fsbl_stub_cold_asm.bin \
tests/bins/elf/custom_opensbi_fw_payload.elf

# 纯 Python API 示例
python examples/demo_emulator.py
```

## 已实现的 RISC-V 特性

| 模块 | 内容 |
|------|------|
| **指令集** | RV64 I (基础整数), M (乘除), A (原子操作 LR/SC/AMO), C (压缩指令), Zicsr |
| **特权级** | U/S/H/M/D 五级, ECALL/EBREAK/MRET/SRET, WFI (TW 检查, 中断唤醒) |
| **虚拟内存** | Sv39 页表遍历 (4 KiB 页 + 2 MiB 超级页), TLB (全相联 FIFO/LRU) |
| **内存保护** | PMP (NAPOT/NA4/TOR, L 位锁定, 最多 64 条), PMA 可配置 |
| **缓存** | 共享 L2 缓存, MESI 一致性协议, 按 hart 的 mdid 域标记 |
| **中断** | CLINT (mtime 定时器 + MSIP 软件中断), 委派 (medeleg/mideleg) |
| **外设** | UART (16550 风格), SPI, I2C, GPIO |
| **平台** | FDT 设备树生成, PlatformConfig 预设 (qemu_virt / sifive_u54) |

## 交互式调试器 (rvdb)

```
rvdbg[0] bt              # GDB 风格栈回溯
rvdbg[0] disasm 0x80000000 +32  # 反汇编
rvdbg[0] step 10         # 单步 10 条指令
rvdbg[0] regs            # 全部寄存器
rvdbg[0] mem 0x80000000 64  # 内存 dump
rvdbg[0] csr satp        # CSR 读写
rvdbg[0] undo            # 快照回滚
```

依赖 `prompt_toolkit` (Tab 补全, 历史) + `rich` (彩色格式化)。

## 项目结构

```
pyremu/
  emulator.py          # 多 hart 执行循环 (round-robin, trap 检测)
  debugger.py           # rvdb 交互式调试器
  platform.py           # PlatformConfig 平台描述
  core/                 # Hart, 解码器, trap 处理, 寄存器模型
  memory/               # MMU, TLB, L2 缓存, PMP, 总线
  peripheral/           # UART, SPI, I2C, GPIO
  interrupt/            # CLINT, 中断控制器抽象
  utils/                # 反汇编器, FDT 生成, ELF 解析
  env_inject/           # 预加载 shellcode 注入
tests/                  # 1130 条测试 (15 套件)
examples/               # 编程式使用示例
third-party/            # 第三方固件 & S-mode 运行时
```

## 测试

14 个测试套件，834 条用例，覆盖指令执行、内存翻译、异常/中断、外设、调试器命令等模块：

| 套件 | 内容 |
|------|------|
| `test_emulator.py` (55) | 多 hart 执行循环, 固件加载, PMP, WFI, UART 输出 |
| `test_trap.py` (99) | trap 编解码, M/S 投递, 委派, 各种异常/中断 |
| `test_debugger.py` (220) | 调试器 REPL, 反汇编, 断点, 栈回溯 |
| `test_disasm.py` (59) | 全部指令格式反汇编 |
| `test_mmu.py` (37) | Sv39 页表遍历, PTE 标志, 超级页 |
| `test_compressed.py` (19) | C 扩展全部已实现指令 |
| `test_amo.py` (17) | LR/SC/AMO 原子操作 |
| `test_pmp.py` (25) | PMP 编解码与匹配 |
| `test_tlb.py` (15) | TLB 插入/命中/驱逐 |
| `test_l2cache.py` (10) | L2 缓存 MESI 状态转换 |
| `test_bus.py` (14) | 总线读写, 设备路由, PMA |
| `test_clint.py` (12) | 定时器与软件中断 |
| `test_parse_elf.py` (14) | ELF/PE/raw 格式解析 |

运行: `make test` 或 `uv run pytest tests/`

## 第三方代码

- `third-party/rust_smode_entry/` — Rust 编写的 S-mode TEE 管理器，以 PIE 位置无关方式编译链接，由 M-mode 加载到动态分配的物理内存中运行
## 开发命令

```bash
make cov-test                     # 带覆盖率的测试
uv run pytest tests/              # 全部测试
uv run pytest tests/test_trap.py  # 单个套件
uv run ruff check .               # 代码风格
uv run ruff format .              # 自动格式化
python tools/profile_emu.py       # cProfile 性能分析
```
