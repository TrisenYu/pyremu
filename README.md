# Pyremu — RISC-V 64 位模拟器

|缩略 |含义        |
|:--:|:---------:|
|Py  | python    |
|r   | riscv/rust|
|emu | emulator  |

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
| **指令集** | RV64 GC (IMAFD + Zicsr + C), 含 FP 算术/转换/比较/乘加 + 压缩 FP (c.fld/c.fsd/c.fldsp/c.fsdsp) |
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

## Rust Native 批量执行引擎

`pyremu/_native/` 包含一个 Rust 编写的 **批量指令执行引擎** (`libdecode.so`)，将
fetch-decode-execute 循环从 Python 移入 Rust，单次 FFI 调用批量执行最多 100000 条
指令。相比纯 Python 路径，单条指令执行开销从 ~4000ns 降低到 ~10ns（~400× 提升）。

### 工作原理

```
Python                              Rust (libdecode.so)
  │                                     │
  │  flush_l2()                         │
  │  marshal HartState ──────────────→  │
  │  run_batch() ────────────────────→  │  for hart in harts:
  │                                     │    for instr in 0..max_instrs:
  │                                     │      fetch → decode → execute
  │                                     │      if ecall/csr/mmio → exit
  │                                     │
  │  ←──────────────────────────── return BatchResult
  │  invalidate_l2()                    │
  │  handle EXIT_SYS / EXIT_TRAP       │
  │                                     │
```

每条 batch 边界执行 `flush_l2()`（将 Python 侧的 L2 脏行回写到 `bytearray`）和
`invalidate_l2()`（丢弃 Rust 直接写入 `bytearray` 后 L2 中的过时行），确保两条
数据路径的一致性。

### 构建

```bash
# Debug 构建（带符号，便于 gdb/lldb 调试）
cargo build --manifest-path pyremu/_native/Cargo.toml
cp pyremu/_native/target/debug/libdecode.so pyremu/_native/libdecode.so

# Release 构建（优化，生产使用）
cargo build --release --manifest-path pyremu/_native/Cargo.toml
cp pyremu/_native/target/release/libdecode.so pyremu/_native/libdecode.so

# 运行 Rust 侧单元测试（~148 条）
cargo test --manifest-path pyremu/_native/Cargo.toml
```

构建依赖: Rust 工具链 (edition 2021), 无需 `maturin`/`setuptools-rust`——输出为标准
cdylib，通过 CPython 内置 `ctypes` 加载。

### 纯 Python 回退

`.so` 缺失或不兼容时，模拟器自动回退到纯 Python 执行路径。可通过环境变量显式禁用:

```bash
PYREMU_NATIVE_BATCH=0 python -m pyremu.debugger tests/bins/elf/...
```

也可以通过 `Emulator` 构造参数控制:

```python
emu = Emulator(cfg, native_batch=False)
```

### 已支持的指令（Phase A–E）

Rust 引擎内联处理全部 RV64IMAC 指令:
- **Phase A**: ALU (R/I/U/J/B/fence), M 扩展, RV64 32-bit 操作
- **Phase B**: Load/Store (含 Sv39 MMU 翻译, TLB, PMP 检查)
- **Phase C**: System (CSR 读写, ECALL/EBREAK/MRET/SRET/WFI/SFENCE.VMA, medeleg 委派)
- **Phase D**: AMO (LR/SC/AMO*), Compressed (C0/C1/C2)
- **Phase E**: 中断检查 (MTI/MSI/STI 优先级, mideleg 委派)

不支持的 CSR 或 MMIO 设备访问会退出到 Python 逐条处理（`EXIT_SYS`），处理完继续
回到 Rust 批量执行。

### 状态结构体布局

`HartState` 和 `BatchResult` 通过 `#[repr(C)]` 锁定内存布局，Rust 和 Python ctypes
两侧必须逐字节一致。修改字段时需同步更新:

| 文件 | 作用 |
|------|------|
| [state.rs](pyremu/_native/src/state.rs) | Rust `#[repr(C)]` 结构体定义 |
| [hart.py](pyremu/core/hart.py) | Python ctypes `_fields_` + `marshal_hart`/`unmarshal_hart` |

验证命令:
```bash
cargo test --manifest-path pyremu/_native/Cargo.toml   # sizeof/align
uv run pytest tests/test_emulator.py::TestNativeBatchLayout   # Python 侧
```

## 注意事项

### Linux 内核 FPU 配置

模拟器支持 **F/D 浮点扩展**（RV64GC）。载入的 Linux 内核必须开启 `CONFIG_FPU=y`，
否则内核的 trap handler 无法识别 FP 指令的懒切换陷态（mstatus.FS=Off → IllegalInstr），
会直接向用户态进程投递 SIGILL，导致 ld-linux 等硬浮点 ABI（lp64d）程序无法启动。

验证内核 config:
```bash
grep CONFIG_FPU .config    # 必须是 CONFIG_FPU=y
```

### 压缩浮点指令

**rv64gc**（含 C + F + D 扩展）会生成 16-bit 压缩浮点指令:

| 指令 | 编码 | quadrant | 说明 |
|------|------|----------|------|
| `c.fld`  | C0 funct3=001 | 00 | 从 rs1'+uimm 加载 8 字节到 FPR rd' |
| `c.fsd`  | C0 funct3=101 | 00 | 将 FPR rd' 的 8 字节存入 rs1'+uimm |
| `c.fldsp` | C2 funct3=001 | 10 | 从 sp+uimm 加载 8 字节到 FPR rd |
| `c.fsdsp` | C2 funct3=101 | 10 | 将 FPR rs2 的 8 字节存入 sp+uimm |

硬浮点（lp64d）Debian/Ubuntu 用户态大量使用这些指令（如 ld-linux 保存/恢复 FP 寄存器）。
确认内核开启 `CONFIG_FPU=y` + `CONFIG_RISCV_ISA_F=y` + `CONFIG_RISCV_ISA_D=y`。

### 跨页取指与数据访问

RISC-V 指令取指固定 4 字节，当 PC 处于页末（offset ≥ 0xFFE）时，取指跨越两个 VA 页。
Sv39 页表下这两个 VA 页可能映射到**非连续**物理页，模拟器对每个 VA 页独立做 MMU 翻译后
拼接结果。非对齐数据 load/store 跨越页边界时同理。

此机制对正确运行 ld-linux 等动态链接器至关重要——动态链接器常在页边界附近执行代码，
链接后的 .plt/.got 等段跨越 VA 页时，物理页不连续会导致取指拼接错误。

### 开发命令

```bash
make cov-test                     # 带覆盖率的测试
uv run pytest tests/              # 全部测试
uv run pytest tests/test_trap.py  # 单个套件
uv run ruff check .               # 代码风格
uv run ruff format .              # 自动格式化
python tools/profile_emu.py       # cProfile 性能分析
```
