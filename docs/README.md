# Pyremu — RISC-V 64 位模拟器

|缩略 |含义        |
|:--:|:---------:|
|Py  | python    |
|r   | riscv/rust|
|emu | emulator  |

Pyremu 是一个用 Python (CPython 3.14) 编写的 **RISC-V 指令集模拟器与交互式调试器**，
面向固件调试、TEE 开发、操作系统调试等场景。

模拟多 hart，支持 RV64 **IMAC** 指令扩展、三级特权级 (U/S/M)、
Sv39 虚拟内存、L2 缓存 MESI 一致性协议、CLINT、PLIC、APLIC、IMSIC等基本中断设备。

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
| **指令集** | RV64 IMAC + Zicsr + C, M 扩展 (mul/div/rem) 和 A 扩展 (LR/SC/AMO) |
| **特权级** | U/S/M, ECALL/EBREAK/MRET/SRET (含 medeleg/mideleg 委派), WFI (TW 检查, 中断唤醒) |
| **虚拟内存** | Sv39 页表遍历 (4 KiB 页 + 2 MiB 超级页), TLB (全相联 FIFO/LRU) |
| **内存保护** | PMP (NAPOT/NA4/TOR, 最多 64 条, MPRV 感知), PMA 可配置 |
| **缓存** | 共享 L2 缓存, MESI 一致性协议 |
| **中断** | CLINT (mtime 定时器 + MSIP IPI), PLIC (平台级中断控制器, 电平语义, M/S 双 context) |
| **外设** | UART (SiFive 16550 子集), virtio-blk, SPI, I2C, GPIO, Hart Watchdog |
| **平台** | FDT 设备树生成, PlatformConfig 预设 (qemu_virt / sifive_u54) |


## 项目结构

```
pyremu/
├── emulator.py            多 hart 执行循环 (run: 并发加速执行; step: 逐指令交错, 停止条件)
├── debugger.py            rvdb 交互式调试器入口
├── platform.py            PlatformConfig 平台配置 (预设dataclass，一定程度支持从JSON/TOML/YAML解析得到)
├── core/                  处理器核心: hart / decoder / trap_handler / mem_check_aux / 寄存器
├── memory/                内存子系统: mmu (仅Sv39)/tlb/l2cache(MESI)/bus/pmp
├── peripheral/            外设: uart / virtio_blk / spi / i2c / gpio / watchdog / termio
├── interrupt/             中断: controller / clint / plic / aplic / imsic / aia
├── utils/                 工具: disassem (反汇编) / parse_bin (固件解析) / dtb (设备树)
├── debug/                 调试器实现 (REPL 命令分发, 栈回溯, 断点)
├── env_inject/            运行时注入 (shellcode 预加载)
└── _native/               Rust 并发执行引擎 (libdecode.so + libtermio.so)

bsp/                       板级支持包
├── rust_smode_entry/      S-mode 飞地运行时 (U-mode 系统调用 ->SBI ecall) 
├── tee_aux_tools/         TEE 辅助工具源 (驱动 + /eval 评测套件)
│   ├── linux-driver/        Linux 飞地内核驱动 (/dev/tee_enclave)
│   ├── cache-probe-exploit/ TEE 评测套件 (缓存侧信道等, 部署为 /eval/cache-probe-exploit)
│   └── stress-test/         TEE 压力测试套件 (部署为 /eval/stress-test)
├── linux/                 内核源码树 (供驱动模块构建)
├── musl-gc-sysroot/       musl sysroot 产物 (rv64gc/lp64d, 由 build-musl-sysroot.sh 生成)
└── setup-rootfs/          根文件系统构建 (debootstrap + 测试程序/驱动打包)

fn_apps/                   飞地可信应用载荷 (Rust + C, 静态 musl 链接)
├── orbit/ chem/ graphene/ ising/ fem/ dsp/   Rust 计算载荷
├── knots/                 C 计算载荷
└── bin/                   构建产物 (供飞地加载)

tests/                     pytest 套件 + 测试源码
├── test_*.py              单元/集成测试 (trap / mmu / pmp / tlb / clint / emulator / debugger ...)
└── src-*/                 src-env (汇编) / src-alg (算法基准) / src-rv8 (toy, 部署为 /eval/toy-progs)

examples/                  编程式使用示例 (纯 Python API / M->S / S->U 演示)
docs/                      设计文档与 CHANGELOG (CHANGELOG.md 为索引, 条目按月归档于 docs/changelogs/)
tools/                     开发辅助脚本 (性能分析等)
vendor/                    第三方子模块 (musl 等)
```


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


## 测试

覆盖指令执行、内存翻译、异常/中断、外设、调试器命令、压缩指令差分验证等模块。
运行: `make test` 或 `uv run pytest tests/`

## Rust Native 并发执行引擎

`pyremu/_native/` 包含一个 Rust 编写的 **thread-per-hart 并发执行引擎** (`libdecode.so` +
`libtermio.so`)，将 fetch-decode-execute 循环从 Python 移入 Rust，以 OS 线程并发
执行直至 MMIO 设备访问或外部中断退出。

CLINT 定时器/MSIP、UART TX、virtio-blk MMIO、压缩指令、AMO 原子操作等热路径全部在
Rust 内联处理，无需退出到 Python。仅 PLIC claim/complete 等需要 Python 侧设备状态的
操作才触发 FFI 边界。

### 工作原理

Python 侧通过 `marshal_hart()` 将全部 hart 的寄存器文件序列化为 `#[repr(C)]` FFI 结构体,
与 RAM buffer / PMP / CLINT / UART / virtio 上下文一起传入 Rust 引擎 (`run_parallel`)。
Rust 以 thread-per-hart 模型并发执行 fetch-decode-execute 循环, 通过共享原子变量同步
CLINT 中断与 `tlb_gen` 计数器 (SFENCE.VMA 广播)。

**外部中断内联**: 当 stdin/UART 输入到达时, daemon 线程注入 UART RX FIFO 并通过
`FfiExtIrqCtx` (跨 FFI 共享结构体) 通知 Rust 引擎。Rust 内联设置 `mip.SEIP/MEIP`
触发 guest 中断处理, 全程不退出 batch——仅在 guest PLIC 驱动读 claim/complete
寄存器时自然退出。

每条 batch 边界执行 `flush_l2()`（将 Python 侧的 L2 脏行回写到 `bytearray`）和
`invalidate_l2()`（丢弃 Rust 直接写入 `bytearray` 后 L2 中的过时行），确保两条
数据路径的一致性。

### 构建

```bash
make build-native # Release 构建 + 复制到 pyremu/_native/libdecode.so
```
也可以手动构建:

```bash
cargo build --release --manifest-path pyremu/_native/Cargo.toml
cp pyremu/_native/target/release/libdecode.so pyremu/_native/libdecode.so

cargo test --manifest-path pyremu/_native/Cargo.toml   # Rust 侧单元测试
```

构建依赖: Rust 工具链 (edition 2021), 无需 `maturin`/`setuptools-rust`——输出为标准
cdylib，通过 CPython 内置 `ctypes` 加载。

当 `.so` 缺失或不兼容时，模拟器将回退到纯 Python 执行路径。但执行用时较长，且目前缺乏维护。

### 已支持的指令

Rust 引擎内联处理全部 RV64IMAC 指令:
- ALU (R/I/U/J/B/fence), M 扩展, RV64 32-bit 操作
- Load/Store (含 Sv39 MMU 翻译, TLB, PMP 检查)
- System (CSR 读写, ECALL/EBREAK/MRET/SRET/WFI/SFENCE.VMA, medeleg 委派)
- AMO (LR/SC/AMO*), Compressed (C0/C1/C2)
- 中断检查 (MTI/MSI/SEI/STI 优先级, mideleg 委派)

不支持的 CSR 或 MMIO 设备访问会退出到 Python 逐条处理，处理完继续回到 Rust 并发执行。

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

## 快捷键

| 按键 | 功能 |
|------|------|
| **Ctrl+Q** | 运行中暂停，回到调试器 |

### 开发命令

```bash
make cov-test                     # 带覆盖率的测试
uv run pytest tests/              # 全部测试
uv run pytest tests/test_trap.py  # 单个套件
uv run ruff check .               # 代码风格
uv run ruff format .              # 自动格式化
python tools/profile_emu.py       # cProfile 性能分析
```
