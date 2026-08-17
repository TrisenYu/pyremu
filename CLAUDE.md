# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pyremu is a RISC-V emulator written in Python (CPython 3.14). It models a multi-hart in-order CPU core targeting the RV64 IMA_C ISA with Zicsr, privilege levels (U/S/H/M/D), Sv39 MMU with TLB, L2 cache with MESI protocol, CLINT timer/IPI controller, basic peripherals (UART/SPI/I2C/GPIO), an FDT generator, and an interactive debugger (rvdb). The project is in active mid-stage development.

**Implemented**: RV64 I (base integer), M (mul/div), A (atomics — LR/SC/AMO), C (compressed), Zicsr (CSR read/write), FENCE/FENCE.I, ECALL/EBREAK/MRET/SRET, WFI (true pipeline stop with interrupt wake-up, TW trap), SFENCE.VMA, Sv39 address translation (4 KiB pages + 2 MiB superpages), trap delegation (medeleg/mideleg) to S-mode, PMP (NAPOT/NA4/TOR, M-mode bypass, MPRV), configurable PMA (ram_base + device MMIO routing), a full RV64 disassembler (I/M/A/C/Zicsr/privileged), an interactive debugger with rich TUI + prompt_toolkit REPL (disasm, stack backtrace, multi-step, command repeat, snapshot/rollback), a platform configuration system (dataclass-based presets + JSON/TOML/YAML deserialization).  

**Not yet implemented**: RVV 1.0 Vector extension (opcode 0x57, ~200 条指令: vsetivli/vsetvl/vector load/store/arithmetic/permute), peripheral interrupt generation.

## Commands

```bash
# Run all tests (823 tests, 14 suites)
# in project directory.
make test

# Run a single test file
uv run pytest tests/test_trap.py

# Run a specific test
uv run pytest tests/test_mmu.py::TestPTEPPN::test_ppn0_field

# Lint & format
uv run ruff check .
uv run ruff format .

# Run with coverage report
uv run pytest --cov=. --cov-report=term

# Launch interactive debugger (rvdb)
python -m pyremu.debugger tests/bins/elf/nonsense.o
ulimit -v 4194304 && PYTHONPATH=/path/to/pyremu timeout 120 uv run /tmp/diag.py
```

## Architecture

### Package structure

```
pyremu/
  emulator.py               # 多核执行循环 (round-robin, 停止条件: 断点/停机/全 halted/超时)
  debugger.py                # 交互式调试器 rvdb (prompt_toolkit + rich)
  platform.py                # PlatformConfig — 平台配置 (dataclass 预设 + JSON/TOML/YAML)

  core/                     # 处理器核心
    hart.py                 # HartWithRegs, RiscvMode, mstatus 位常量
    decoder.py              # Hart(HartWithRegs), exec_instr(), 全部指令 handler 分发
    trap_handler.py         # 独立陷态函数: deliver_trap, trap_ecall, trap_mret, trap_sret, handle_wfi, check_pending_interrupts
    mem_check_aux.py        # 内存访问辅助函数 (mem_read/mem_write/validate_csr/translate_addr)
    registers.py            # Reg, FPR, CSR 模型 + 工厂函数 + CSR 表
    trap.py                 # TrapType 枚举, cause code ↔ name 双向映射

  memory/                   # 内存子系统
    mmu.py                  # PTE, Sv39 页表遍历, translate_va()
    tlb.py                  # TLB (继承 CacheBase, 全相联 FIFO/LRU)
    cache_base.py           # CacheBase + CacheLineBase 抽象基类
    l2cache.py              # L2 共享缓存 (MESI 协议)
    cache.py                # CacheEntry, TLB_SIZE
    bus.py                  # Bus + Device ABC + AccessFaultError + PMA 检查
    pmp.py                  # PMP (NAPOT/NA4/TOR, L 位锁定, 最多 64 条目)

  peripheral/               # 内存映射外设 (Device 子类)
    uart.py                 # SiFive 风格 UART (TX/RX/ctrl/div)
    spi.py                  # SPI 主控 (ctrl/status/tx/rx/div)
    i2c.py                  # I2C 主控 (ctrl/status/data/addr/prescaler)
    gpio.py                 # GPIO 控制器 (in/out/dir)

  interrupt/                # 中断子系统
    controller.py           # InterruptController ABC, IntSource
    clint.py                # CLINT (mtime/mtimecmp 定时器 + MSIP IPI)
    aia.py                  # AIA/IMSIC

  env_inject/               # 运行时注入
    preload.py              # Preloader — 将 shellcode 注入 RAM
    snippets.py             # 预置 shellcode 片段

  utils/                    # 工具
    file_ops.py             # 文件路径操作
    disassem.py             # RV64 反汇编器 (I/M/A/C/Zicsr/privileged, ~686 行)
    wrapper.py              # 异常处理装饰器: seize_* / silent_on_err / print_exc_on_err / die_if_err
    str_aux.py              # 格式化辅助: fmt_addr, fmt_hexdump
    parse_bin.py            # 固件解析 (ELF/PE/raw binary, 基于 LIEF)
    fdt.py                  # Flat Device Tree (FDT) 生成器 — 构建 DTB blob
```

### Data flow

1. `utils/parse_bin.py` -> `parse_firmware()` 用 LIEF 解析 ELF/PE/raw binary -> `FirmwareImage` (entry_point + segments + symbols).
2. `emulator.Emulator` 创建 harts + Bus + CLINT + peripherals, 调用 `load_firmware(image)` 将各段写入 RAM, 所有 hart 的 PC 设到入口地址.
3. 可选: `emulator.load_dtb(addr)` 将 FDT blob 加载到 RAM 并将地址写入 `a1` (x11), 供固件通过设备树发现外设.
4. 执行循环 (`step` / `run`): 每个未 halted 的 hart:
   - 若 `_waiting` (WFI 等待): 仅检查中断唤醒, 不取指/执行
   - 否则: 取指 4 字节 -> `Hart.exec_instr(instr)` -> 推进 PC 或跳转
   - 指令边界检查中断 -> CLINT 时钟 tick.
5. `debugger.Debugger` 包装 Emulator, 提供单步 (`step_one`)、快照/回滚、REPL 命令分发.

### WFI 低功耗等待

Hart 执行 WFI 指令时:
- 若已有待处理且使能的中断 (`mip & mie ≠ 0`) -> 立即返回 (NOP), 中断在指令边界投递
- 否则 hart 进入 `_waiting` 状态, PC 指向 WFI 下一条指令
- 等待中的 hart 在 `step()` 中被跳过 (不取指/执行), 仅轮询中断
- 当中断变为挂起且使能时 `deliver_trap` 唤醒 hart (清除 `_waiting`)
- `mstatus.TW=1` 且非 M 模式执行 WFI -> IllInstr 陷态

### Class hierarchy

```
Reg (pydantic BaseModel, core/registers.py)
├── CSR  — 加 access(CsrAccess), xlen, strip_w/strip_mmode/add_dmode
└── FPR — 浮点寄存器 (val: float)

CacheLineBase (memory/cache_base.py) — tag, valid, dirty
├── TLBLine     — ppn, perm, level
└── L2CacheLine — data, mesi state

CacheBase (memory/cache_base.py) — lookup/insert/flush 框架
├── TLB     — 全相联, FIFO/LRU 替换
└── L2Cache — 共享, MESI 一致性

Device (memory/bus.py) — ABC, read(offset, size) / write(offset, data)
├── CLINT   — 定时器 + IPI
├── UART    — NS16550 风格串口
├── SPI     — 主模式 SPI 控制器
├── I2C     — 主模式 I2C 控制器
└── GPIO    — 通用 I/O

PMP (memory/pmp.py) — PMP 条目管理, NAPOT 编解码, 地址匹配检查

PlatformConfig (platform.py) — dataclass, 描述 CPU 拓扑、内存布局、外设映射
├── CpuConfig / MemoryConfig / PeripheralConfig / IsaConfig
└── 工厂方法: qemu_virt(), sifive_u54(), dump_json/toml/yaml()

HartWithRegs (core/hart.py)
    gprs[32], fprs[32], csrs (name->CSR), itlb/dtlb,
    pc, mode, _bus: Bus, _interrupt_ctrl: InterruptController,
    _mem_read_phy, _mem_write_phy,
    reservation (LR/SC), _halted, _waiting
    mstatus/mtvec/mepc/mcause/… 快捷 property
└── Hart (core/decoder.py)  ← 仅继承 HartWithRegs
        exec_instr() -> 返回 2/4 (PC 需推进) 或 0 (PC 已被修改)
        调用 mem_check_aux 中的独立函数: mem_read(hart, va, size) /
        mem_write(hart, va, data) / validate_csr(hart, addr, write)
        调用 trap_handler 中的独立函数: deliver_trap(hart, ...) /
        trap_ecall(hart) / handle_wfi(hart) 等
        handle_alu/handle_op_imm/handle_op32/handle_ld/handle_st/
        handle_br/handle_jalr/handle_sys/handle_fence/handle_amo/
        handle_compressed -> _handle_compressed_c0/_c1/_c2
```

### Instruction execution flow

`Hart.exec_instr(instr: int) -> int`:

1. 若 `parse_compressed(instr)` -> `handle_compressed(instr & 0xFFFF)` -> 按象限 (C0/C1/C2) 分发.
2. 否则按 `Opc(parse_opcode(instr))` 分发到对应 handler.
3. 任何 `ValueError` / `NotImplementedError` (非法编码、未实现 opcode) 统一转为 `deliver_trap(self, TrapType.IllInstr, ...)`.
4. 返回 0 (PC 已修改, 如跳转/trap) 或 2/4 (调用方负责 `pc += advance`).

**字段提取器** (module-level, decoder.py):
`parse_opcode`, `parse_rd`, `parse_func3`, `parse_rs1`, `parse_rs2`, `parse_func7`,
`parse_imm12_se`, `parse_imm_s`, `parse_imm_b`, `parse_imm_j`, `parse_imm20_raw`,
`parse_compressed` (低 2 位 ≠ 3 -> 16-bit), `_sext(val, bits)`.

**枚举**: `Opc`, `aluOp`, `sysOp`, `brFn3`, `ldFn3`, `stFn3`, `sysFn12`, `AmoFunct5`, `AmoWidth`.

### Memory translation & PMA

内存访问通过 `mem_check_aux.py` 中的独立函数完成 (hart 为显式第一参数):

```
mem_read(hart, va, size) / mem_write(hart, va, data)
  -> _translate_addr(hart, va)
      -> satp.MODE == Bare -> VA 即 PA (直接通过)
      -> hart.dtlb.lookup(vpn)  -> hit: 返回 PA
      -> miss: translate_va() -> sv39_walk(root_ppn, va) -> 3 级页表遍历
              L1 (VPN[2]) -> L2 (VPN[1]) -> L3 (VPN[0])
              支持 4 KiB 叶子页和 2 MiB 超级页
              -> 结果插入 TLB
      -> 翻译失败 -> deliver_trap(hart, InstrPageFault / LdPageFault / StPageFault, ...)
  -> PMP 检查 -> PMA 检查
  -> hart._mem_read_phy(pa, size) / hart._mem_write_phy(pa, data)
  -> Bus.read/write -> 设备 MMIO (直通) -> L2 缓存 -> RAM
```

**PMA (Physical Memory Attributes)** — Bus 提供:
- `is_ram_addr(addr)` / `is_device_addr(addr)` / `is_valid_addr(addr)`
- `ram_base` 可配置 (默认 `0x8000_0000`), RAM 覆盖 `[ram_base, ram_base + ram_size)`
- 设备 MMIO 绕过 L2 缓存, 直通设备 (读写可能有副作用)
- `AccessFaultError` 供上层 PMA 违例使用

**安全读写** — Bus 同时提供不抛异常的版本, 供调试器等外部调用方使用:
- `try_read(addr, size) -> bytes | None` — 失败返回 None
- `try_write(addr, data) -> bool` — 失败返回 False
- 将设备/L2 缓存异常统一转换为返回值, 调用方无需 try/except

### Trap handling & delegation

全部陷态处理函数位于 [core/trap_handler.py](pyremu/core/trap_handler.py), 以 hart 为显式第一参数的独立函数 (遵循"避免 mixin 隐式依赖"原则):

`deliver_trap(hart, cause, tval, is_interrupt)` — 委派感知的陷态入口:

1. **委派检查**: 若 `mode != M` 且 `medeleg[exc_code]` (异常) 或 `mideleg[exc_code]` (中断) 置位 -> 委派到 S 模式
2. **S 模式投递** (`_trap_deliver_smode`): 保存到 `sepc`/`scause`/`stval`, 更新 `SPIE`/`SIE`/`SPP`, 切换到 S 模式, 跳转 `stvec`
3. **M 模式投递** (`_trap_deliver_mmode`): 保存到 `mepc`/`mcause`/`mtval`, 更新 `MPIE`/`MIE`/`MPP`, 切换到 M 模式, 跳转 `mtvec`
4. M 模式下发生的 trap **永不委派**

`check_pending_interrupts(hart)`:
- 完整优先级: MEI > MSI > MTI > SEI > SSI > STI
- 已委派中断需 S 模式全局使能 (`SIE`); 非委派 M 级中断可抢占 S 模式

`trap_ecall(hart)`: 根据 `hart.mode` 分发 `EcallFromUmode` / `Smode` / `Mmode`.

`trap_mret(hart)`: 恢复 `mode ← MPP`, `MIE ← MPIE`, `pc ← mepc`.

`trap_sret(hart)`: 恢复 `mode ← SPP`, `SIE ← SPIE`, `pc ← sepc`.

`handle_wfi(hart)`: 若 `mstatus.TW=1` 且非 M 模式 -> IllInstr; 若已有待处理中断 -> 立即返回 (NOP); 否则 hart 进入 `_waiting` 状态.

> 连续执行 (run/continue) 无指令配额与连续 trap 兜底 (与 QEMU 一致): 死循环 /
> 非法指令 trap loop 不会自动暂停 hart, 由停止条件 (断点命中 / semihosting 停机 /
> 全部 hart halted / 时钟源超时) 或外部设备暂停事件 (Ctrl+Q) 终止执行.

**trap.py** 提供: `TrapType` 枚举 (14 异常 + 10 中断), `trap_cause_code(trap) -> int`, `trap_is_interrupt(trap) -> bool`, `trap_cause_name(mcause_val) -> str` (mcause 值 -> 可读名称).

## Debugger (rvdb)

交互式 RISC-V 调试器, 位于 [debugger.py](pyremu/debugger.py).

**依赖**: `prompt_toolkit` (REPL: 方向键历史, Tab 补全, FileHistory 持久化), `rich` (Console, Table, Panel — 彩色格式化输出).

**设计原则**: Emulator 是纯执行引擎 (无 I/O), Debugger 负责所有用户界面和状态呈现.

**符号与段注释**: 反汇编输出和栈回溯自动附加符号名 (来自 ELF `.symtab`/`.dynsym`) 和
段名 (来自 ELF section headers, 如 `.text`, `.rodata`, `.data`). 符号以黄色高亮, 段名以
灰色显示于对应行末 `; <sym>  .text` 注释中. `_find_segment(addr)` 按地址查找所属段,
`_resolve_symbol(syms, addr)` 查找符号名.

**Tab 补全**: 动态从 `register_gpr()` / `register_fpr()` / `register_csr()` 提取名称, 加命令名综合构建 `WordCompleter`.

**主要命令**:
- 执行: `s`/`step [n]`, `c`/`continue`, `r`/`run [n]`, `undo`/`rollback`
- 寄存器读: `regs`, `reg <name>`, `csr <name>`, `csr list`, `pc [addr]`, `mode`, `mstatus`, `tlb [vpn]`, `cache [set] [way]`, `satp`
- 寄存器写: `set <name> <val>`, `csrw <name> <val>`, `pc <addr>`
- 内存: `mem <addr> [size]`, `disasm <addr> [len]`, `sym [filt]`
- 栈回溯: `stack`/`bt`/`frame`/`f` — GDB 风格, #01 起始编号, `frame N` 切换当前帧并显示其栈内存
- 配置: `hart <id>`, `help`/`?`, `q`/`quit`

**GDB 风格特性**: 空输入 (直接按回车) 重复上一条命令.

## Examples

[examples/](examples/) 目录包含三个编程式使用示例 (非交互式):

| 文件 | 说明 |
|------|------|
| [demo_emulator.py](examples/demo_emulator.py) | 纯 Python API: 加载固件, 运行 N 周期, 检查 hart 状态, dump 内存和寄存器 |
| [demo_debugger.py](examples/demo_debugger.py) | Debugger 编程接口: 反汇编入口, 单步观测 M->S 模式切换, 检查寄存器 |
| [demo_m_to_s.py](examples/demo_m_to_s.py) | M->S 移交完整演示: 模拟 OpenSBI -> OS boot, 观测 UART 输出和 WFI 状态 |
| [demo_s_to_u.py](examples/demo_s_to_u.py) | M->S->U 完整演示: UART 输入, Sv39 栈保护页, fib 栈帧验证, 病态进程 S 模式终止 |

运行: `uv run python examples/demo_emulator.py` (各文件均可独立运行).

### 测试汇编与算法

- [tests/src-env/](tests/src-env/) — 汇编测试源码 (M->S 移交, Sv39 页表设置, ZSBL 启动, UART 输入)
  - 所有目标通过 [makefile](tests/src-env/makefile) 构建, 使用 `/opt/custom-llvm/bin/` 下的自定义 LLVM 工具链
  - `make build-m2s` 编译 M->S 测试固件 (`s_mode_hello.elf`)
  - `make build-s2u` 编译 M->S->U 测试固件 (`u_mode_run_fib.elf`, 含 Sv39 + UART 输入 + fib)
  - `make build-multi` 编译多程序内核 (`kernel.elf`: kernel.s + prog_fib.s + prog_nqueen.s)
  - `make all` 构建全部目标
- [tests/src-alg/](tests/src-alg/) — C++ 算法基准 (n-queen, subset), 供未来性能测试

## Peripheral devices

外设模型位于 [pyremu/peripheral/](pyremu/peripheral/), 全部实现 `Device` 接口.

### UART (SiFive NS16550 风格)

- 寄存器: TX (0x00), RX (0x04), TXCTRL (0x08), RXCTRL (0x0C), IE (0x10), IP (0x14), DIV (0x18)
- TX: 写入数据存入内部 buffer (调试用); RX: 可预加载数据供固件读取
- DIV: 波特率除数, 控制 TX/RX 使能

### SPI 主控

- 寄存器: CTRL (0x00), STATUS (0x04), TXDATA (0x08), RXDATA (0x0C), DIV (0x10)
- 全双工: 写 TXDATA 同时捕获到 RXDATA (shift 模型)
- STATUS: 空闲/忙标志

### I2C 主控

- 寄存器: CTRL (0x00), STATUS (0x04), DATA (0x08), ADDR (0x0C), PRESCALE (0x10)
- 支持 START/STOP/ACK 控制; 简化为即时完成模型
- STATUS 含 RXDATA 有效、TX 完成、ACK 等标志

### GPIO

- 寄存器: INPUT_VAL (0x00), INPUT_EN (0x04), OUTPUT_EN (0x08), OUTPUT_VAL (0x0C)
- 每个 GPIO 位可独立配置方向
- 外部引脚值可通过 `set_pin()` 方法注入

### 默认 MMIO 地址

| 设备 | 默认基址 | 大小 |
|------|---------|------|
| CLINT | 0x0200_0000 | 64 KiB |
| UART0 | 0x1000_0000 | 4 KiB |
| SPI0  | 0x1000_1000 | 4 KiB |
| I2C0  | 0x1000_2000 | 4 KiB |
| GPIO0 | 0x1000_3000 | 4 KiB |

## Flat Device Tree (FDT)

[pyremu/utils/fdt.py](pyremu/utils/fdt.py) — 基于 libfdt 的 DTB 生成器.

- 使用 `libfdt.FdtSw` 顺序 API (`begin_node/end_node/property_string/property_u32/property`) 构建设备树
- Emulator 在初始化外设后可选通过 `build_dtb()` 生成 DTB 并加载到 RAM
- DTB 地址写入 `a1` (x11) 寄存器供固件使用

## Testing

673 tests across 14 files, all passing:

| File | Cases | 覆盖内容 |
|------|-------|---------|
| test_parse_elf.py | 14 | ELF/PE/raw 格式检测与解析, FirmwareSegment 模型 |
| test_trap.py | 99 | trap cause code 编解码, `deliver_trap` M/S 投递, mtvec/stvec direct/vectored, ECALL/EBREAK/MRET/SRET, medeleg/midegl 委派, CSR 特权级检查, 访存对齐/PMA 故障, 缺页异常 (Sv39 Ld/StPageFault), 定时器中断, WFI, mem_check_aux 集成 |
| test_pmp.py | 25 | PMP NAPOT 编解码, TOR/NA4/NAPOT 匹配, R/W/X 权限, M 模式旁路, CSR 条目范围, hart 集成 |
| test_tlb.py | 15 | TLB insert/lookup/miss, 原地更新, FIFO 驱逐与环绕, 单 VPN 刷新与全刷新 |
| test_mmu.py | 37 | PTE flag/PPN 读写, leaf/pointer 检测, 权限检查, VPN 分解, Sv39 3 级 4 KiB 页表遍历, 2 MiB 超级页, Bare 模式 |
| test_amo.py | 17 | LR/SC/AMOSWAP/AMOADD/AMOXOR/AMOAND/AMOOR/AMOMIN/AMOMAX/AMOMINU/AMOMAXU (.W/.D) |
| test_compressed.py | 19 | C0/C1/C2 全部已实现压缩指令, 含非法编码陷态 |
| test_disasm.py | 59 | 所有指令格式反汇编 (R/I/S/B/U/J + CSR + priv + AMO + compressed) |
| test_emulator.py | 55 | 多 hart 执行循环, 固件加载, 内存 dump, PC 推进, store 指令写入 RAM, AUIPC sign-extend 回归, M->S 模式切换, PMP 配置, WFI 低功耗等待, UART 输出, 汇编反汇编集成, GPR 值规范化回归 |
| test_bus.py | 14 | 总线读写, 设备注册与路由, PMA 检查 (RAM 范围, 设备检测, 空洞地址), try_read/try_write |
| test_clint.py | 12 | mtime 递增, mtimecmp 定时器中断, MSIP 软件中断 |
| test_cache_base.py | 10 | CacheBase/CacheLineBase 抽象接口 |
| test_l2cache.py | 10 | L2Cache MESI 状态转换, 读写分配, 回写 |
| test_debugger.py | 220 | 调试器 REPL 命令分发, 反汇编, 断点 (addr/instr/opcode), PC 校验, 栈回溯, 符号表, info/status |

## Important design notes

### FFI struct 布局锁定: Python ctypes ↔ Rust `#[repr(C)]` 同步

`HartState`, `TlbEntry`, `BatchResult` 通过 ctypes 跨越 FFI 边界传递.
**两侧布局必须逐字节一致**, 否则 Rust 在错误偏移处读取字段 → SIGBUS.

核心约束:
- **array-of-structs, 非 struct-of-arrays**. `[TlbEntry; 32]` 在两侧必须是
  连续 24 字节条目数组, 不能拆成 `itlb_vpn[32] + itlb_ppn[32] + ...`.
- 修改 `HartState` / `TlbEntry` / `BatchResult` 字段时, 必须同步更新三方:
  1. Rust `#[repr(C)]` struct (`pyremu/_native/src/state.rs`)
  2. Python ctypes `_fields_` (`pyremu/core/hart.py`)
  3. `marshal_hart()` / `unmarshal_hart()` 字段读写
- 验证命令:
  ```bash
  cargo test --manifest-path pyremu/_native/Cargo.toml  # Rust 侧 sizeof/align
  uv run pytest tests/test_emulator.py::TestNativeBatchLayout  # Python 侧
  ```
- `TestNativeBatchLayout` 锁死: TlbEntry ≡ 24B, HartState 对齐 + < 4KB,
  itlb/dtlb 确保 `.vpn` / `.ppn` / `.valid` 属性存在 (即 struct 非分离数组).

**教训**: 初次实现 Phase B 时, Rust 侧新增 `itlb: [TlbEntry; 32]` 但 Python 侧误用
分离数组布局. 旧测试因 `PYREMU_NATIVE_BATCH=0` 从未触发 native batch → 静默通过.
首次固件启动才暴露 SIGBUS. 参见 [CHANGELOG.md](CHANGELOG.md) 2026-07-05 条目.

### GPR 值的 64-bit 规范化与 Python 位运算陷阱

Python 的任意精度整数在位运算 (`|`, `&`, `^`) 中表现不同于有限位宽硬件:
负 Python int (如 `-805306368`) 在位运算中携带无限个前导 `1`, 而 64-bit 掩码
后的正 int (如 `0xFFFFFFFFD0000000`) 仅保留低 64 位, 两者 `==` 不相等—
即使它们代表同一硬件 bit pattern.

**当前策略**: `_sext(val, bits)` 对 `bits≤64` 归一化返回值为 `[0, 2^64)` 范围内的
无符号 Python int. 所有 RISC-V 立即数和 32-bit 操作的结果写入 GPR 前均经过此规范化.
使用 `_sint64()` / `_uint64()` ctypes 包装器进行有符号/无符号比较时传入规范化值同样正确.

**相关修复**: [CHANGELOG.md](CHANGELOG.md) — 2026-06-19 `_sext()` 规范化 + BEQ/BNE 误判.

### CSR 写入与 property setter 副作用

部分 CSR 寄存器 (如 `satp`) 在写入时需要触发副作用 (更新缓存的 MMU 模式).
`HartWithRegs` 通过 property setter 封装了这些副作用. 但 CSR 指令 (`csrrw` / `csrw` 等)
通过 `write_csr()` 方法直接设置 `csrs[name].val`, 会绕过 property setter.

**当前修复**: [hart.py:149-156](pyremu/core/hart.py#L149) — `write_csr()` 对 `satp` 显式路由到
`self.satp_val = val` (经 property setter). 若后续新增有副作用的 CSR, 需同步更新此白名单.

```python
# ❌ 直接写 CSR 对象 — 绕过 satp_val setter, _mmu_mode 不更新
self.csrs["satp"].val = val

# ✅ 经 property setter — 同步更新 _mmu_mode
self.satp_val = val
```

### SFENCE.VMA 编码

SFENCE.VMA 指令的 funct12 编码为 **0x120** (funct7=0b0001001 在 bits[31:25],
rs2=0 时; 若 rs2≠0, funct12 = 0x120 | rs2). 非 0x104.

SFENCE.VMA 刷新全部 hart 的 itlb 和 dtlb. 在 `csrw satp` 之后必须执行
`sfence.vma zero, zero` 以生效地址翻译模式切换.

### sscratch 交换: S 模式 trap handler 栈安全

当 U 模式触发 trap (如 StorePageFault) 时, sp 指向 U 模式栈 (可能已在保护页内).
若 trap handler 直接使用 sp, 会二次触发页错误. 标准解决是 sscratch 交换:

```asm
# S 模式入口:
    la   t0, stack_top_s
    csrw sscratch, t0       # sscratch ← S 栈顶
    sret

# trap handler 第一条指令:
s_trap_handler:
    csrrw sp, sscratch, sp   # sp ↔ sscratch (sp=S栈, sscratch=故障时U sp)
    addi sp, sp, -96
    ...
    # 退出前恢复:
    csrr t0, sscratch
    sd   t0, 72(sp)          # 保存故障时 sp
    ...
    la   t0, stack_top_s
    csrw sscratch, t0        # 恢复 sscratch 指向 S 栈 (供下次 trap)
    ld   t0, 72(sp)
    csrw sscratch, t0
    addi sp, sp, 96
    csrrw sp, sscratch, sp   # sp = 故障时 sp, sscratch = S 栈顶
    sret
```

### Sv39 页表 identity 映射 (汇编)

在固件中手工构建 Sv39 页表时, PPN 值必须与物理地址匹配. 页表自身在 BSS 中的
物理地址决定其 PPN. 计算 PTE 值时, 使用 `make_pte` 公式:

```python
def make_pte(flags, ppn):
    val = flags
    val |= (ppn & 0x3FF) << 10           # PPN[9:0]  -> bits[19:10]
    val |= ((ppn >> 10) & 0x1FF) << 20   # PPN[18:10] -> bits[28:20]
    val |= ((ppn >> 19) & 0x1FFFFFFF) << 29  # PPN[43:19] -> bits[53:29]
    return val
```

关键: `li` 伪指令加载大于 32 位的常量时可能展开为复杂的指令序列,
在调试时建议用 objdump 确认实际编码, 或用移位+OR 显式构造.

### UART 输入: 三层架构

UART RX 通过三层解耦实现鲁棒的字符输入:

```
UART RXDATA ──poll──-> ring buffer ──getc──-> line buffer ──parse──-> 结果
  (IP.rxwm)         (64B FIFO)   (阻塞)    (退格编辑)     (宽松)
```

- `uart_poll`: 检查 UART IP.rxwm, 搬运到 ring buffer (单次上限防饿死)
- `uart_getc`: 从 ring buffer 取字节, 空时先调 poll 再重试
- `readline`: 逐字节读取, 遇 `\n`/`\r` 终止, 支持 BS (0x08) / DEL (0x7F) 退格编辑并回显 `\b \b`
- `parse_int_simple`: 从 null 结尾字符串收集连续数字字符, 转换为整数; 无数字时报错

### medeleg 委派: 页错误需显式委派

仅委派 `ECALL from U-mode` (bit 8) 不足以保证 S 模式能捕获 U 模式页错误.
必须同时委派:

```asm
li   t0, (1 << 8)    # ECALL from U-mode
ori  t0, t0, (1 << 12)  # InstrPageFault
ori  t0, t0, (1 << 13)  # LdPageFault
ori  t0, t0, (1 << 15)  # StPageFault
csrw medeleg, t0
```

未委派的页错误会路由到 M 模式, M 模式 handler 若仅做 `mepc+4; mret`, 则错误
被静默跳过 (指令不执行但无任何可见效果, 保护页形同虚设).



### 每个 bug 修复必须附带回归测试

项目中发现的每一个 bug, 修复时必须同步补充至少一个针对性测试用例,
验证修复后的正确行为并锁定回归底线.

测试用例要求:
- 能复现修复前的错误行为
- 覆盖 bug 的精确触发条件
- 如涉及位掩码/偏移量, 选择能使修前/修后产生不同结果的具体值

反例 — 已有的 `test_megapage_ppn_mask_regression` 使用 `PPN=0xABCD0` (bit 9=0),
而 bug 恰好在 bit 9=1 时触发, 因此该测试未能拦住 Sv39 2 MiB 掩码回归.

正例 — [test_mmu.py](tests/test_mmu.py) `test_megapage_ppn_bit9_preserved`:
使用真实触发值 `PPN=0x80200` (bit 9=1), 且追加了显式断言 `pa != old_buggy_pa`
确保旧掩码产生的错误值不再出现.

### Sv39 2 MiB 超级页: PPN 掩码必须清 9 位而非 10 位

2 MiB 超级页的 page offset 为 VA[20:0] (21 bits), vpn[0] = VA[20:12] (9 bits)
替换 PPN[8:0]. PPN bit 9 及以上属于物理地址有效位, 不能清零.

```python
# ❌ 掩码清 10 位 — PPN bit 9 被错误清零
ppn = (pte.ppn & ~0x3FF) | vpn[2]

# ✅ 掩码清 9 位
ppn = (pte.ppn & ~0x1FF) | vpn[2]
```

## Changelog

关键 bug 修复记录在 [CHANGELOG.md](CHANGELOG.md) 中, 包含:
- `_sext()` 返回负 Python int 导致 BEQ/BNE 误判 — 规范化到 `[0, 2^64)`
- SFENCE.VMA funct12 编码错误 (0x104 -> 0x120)
- `csrw satp` 绕过 `_mmu_mode` 更新
- TLB 不可迭代 (缺少 `__iter__`)
- C.JALR / CSRRW 等 rd==rs1 读写竞争

## Stub modules (no implementation yet)

- [interrupt/aia.py](pyremu/interrupt/aia.py) — AIA/IMSIC 高级中断控制器 (仅空壳).

## Planned: RVV 1.0 Vector Extension (opcode 0x57)

RISC-V "V" 向量扩展为 RV64 基础 ISA 增加 ~200 条向量指令, 操作数宽度从 8-bit 到 64-bit, 支持 LMUL (1/2/4/8) 分组、mask/tail 策略。关键 opcode: `0x57` (OP-V)。

**当前状态**: 未实现。任何 V 扩展指令命中 `Opc` 枚举未覆盖的 opcode 0x57, 经 `exec_instr()` -> `ValueError` -> `IllInstr` 陷态。

**已知影响**: 本仓库中的 `custom_opensbi_fw_payload.elf` 已以 `-march=rv64imac` (不带 `v`) 重编译, 不再触发此问题. 直接从上游编译的 PLATFORM=generic 固件若启用 V 扩展仍需 `-march=rv64imac`.

**实现计划 (低优先级)**:
- Phase 1: `vsetivli` / `vsetvl` / `vsetvli` — 配置向量长度, 基本 CSR (vl/vtype/vstart/vxsat/vxrm/vcsr)
- Phase 2: 向量 load/store (`vle8.v`/`vse8.v` 等 unit-stride, 宽度 × LMUL)
- Phase 3: 向量整数算术 (vadd/vsub/vmul/vand/vor/vxor 等, .vv/.vx/.vi)
- Phase 4: 向量 permute (vrgather/vslide/vmerge/vid.v 等)
- Phase 5: 向量浮点 (vfadd/vfmul 等, 需 F/D 扩展先就绪)

**关键 opcode**:
- `0x57`: OP-V (向量算术/配置)
- `0x27`: OP-FV (向量浮点, 需 F/D)
- `0x07`: vector load
- `0x27`: vector store

**参考**: RISC-V V 规范 1.0 (https://github.com/riscv/riscv-v-spec), RVV intrinsics in LLVM 22+.

## OpenSBI firmware compatibility

`tests/bins/elf/custom_opensbi_fw_payload.elf` 是一个自定义 OpenSBI build (PLATFORM=generic):
- 入口点 0x0, 静态 PIE, 段从 0x0 展开
- 内置 `sbi_domain` 框架 (domain 注册/启动/内存区域/PMP 隔离)
- `.coffer_enclave_man` (0x180000, 512 KiB): enclave 管理器占位段 (全零, 未链接实际代码)
- `.payload` (0x200000, 8 KiB): 微型测试 payload (SBI ecall 打印)
- 需要 `ram_base=0` 加载, FDT 通过 a1 传入 (需含 `/chosen/stdout-path`)
- **已修复**: V 扩展指令已通过 `-march=rv64imac` 重编译移除; `_sext` 规范化 bug 修复后固件可成功通过 `fw_platform_init` 到达 `_start_hang`
- 启动流程: `_start` -> `fw_boot_hart`(-1) -> `_try_lottery`(AMOSWAP) -> PIE 重定位 -> BSS 零填充 (约 800 KB, ~300K 指令) -> `_scratch_init` -> `fw_platform_init` (FDT 解析 `/cpus`, `/chosen`, 遍历子节点) -> `_fdt_reloc_done`(设 `_boot_status=1`) -> `_start_warm` -> `sbi_init()` -> `_start_hang`(WFI 空闲)

## Code style

- **函数签名**: 左括号 `(` 和返回类型 `) -> Type:` 各自独占一行, 参数逐行缩进.
- **注释**: 架构/设计原理/"为什么"使用中文; 行内注释可使用任一语言.
- **行宽**: 95 字符 (ruff line-length = 95).
- **Imports**: isort 排序, `combine-as-imports = true`. 所有导入必须在文件开头, 禁止在函数内部 import.
- **ruff 规则**: F, E, N, I, W, PL, PERF (忽略 E731 单行 lambda, PLR2004 魔数比较).
- **禁止导入私有符号**: 不 `from module import _PrivateName`; 对面内私有映射直接在调用点内联.
- **测试命名**: `test_<what>_<outcome>` 或 `test_<outcome>`.
- **文件行数上限**: 功能代码 (非测试) 单文件不超过 1500 行; 超过则拆分为 mixin 或模块.
- **函数行数上限**: 承载业务逻辑的函数 (非测试/命令分发/配置) 不超过 100 行; 超出则提取辅助函数.
- **重构文件的安全流程**: 1) 复制旧文件为 `.txt` 后缀 (避免 Python 模块发现冲突)
  2) 在新文件上修改 3) `pytest` 验证全部通过 4) 删除 `.txt` 旧文件.
- **删除文件原则**: 功能已被替代的空壳文件应删除, 并清理所有引用和文档.

对于汇编文件的编写，不能将多条语句压在一行作为某种功能的定义。

错误用法：
```asm
    li t0, 10; divu t1, s0, t0; remu t2, s0, t0
    addi t2, t2, '0'; sb t2, 0(s1); addi s1, s1, -1; mv s0, t1
    bnez s0, putdec_loop
```

正确用法：
```asm
    li t0, 10
    # 计算除法结果与余数
    divu t1, s0, t0
    remu t2, s0, t0
    addi t2, t2, '0'
    # 偏移字符串索引位置
    sb t2, 0(s1)
    addi s1, s1, -1
    mv s0, t1
    # 检查是否仍需要继续
    bnez s0, putdec_loop
```

### 避免 mixin 的隐式依赖

当 mixin 需要宿主类提供大量属性 (导致 `Requires the host class to provide:` 注释和静态分析告警),
优先转为独立函数, 将对象作为显式第一参数:

```python
# ❌ mixin — self.pc, self._pmp, self.dtlb 等来自父类, 静态分析不可见
class MemoryAccessor:
    def _mem_read(self, addr, size): ...

# ✅ 独立函数 — hart 参数类型明确, 所有属性可追溯
def mem_read(hart: HartWithRegs, addr: int, size: int) -> bytes: ...
```

### 异常处理: 在源头封装, 而非在各调用方分散

当多个调用方围绕同一 API 重复 try/except 时, 应在 API 层提供安全包装:

```python
# ❌ 每个调用方各自 try/except
try: data = bus.read(addr, 4)
except Exception: data = None

# ✅ 在 API 层一次性封装
def try_read(self, addr, size) -> bytes | None: ...
data = bus.try_read(addr, 4)
if data is None: ...
```

## Error handling

[utils/wrapper.py](pyremu/utils/wrapper.py) 提供以下装饰器:

| 装饰器 | 作用 | 适用场景 |
|--------|------|---------|
| `@seize_err_if_any(logger_enable=True)` | 捕获所有异常, 记录 traceback, 返回 `None` | 通用兜底 |
| `@die_if_err` | 捕获异常, 打印 traceback, `sys.exit(1)` | 致命错误 |
| `@seize_val_err(msg)` | 捕获 `ValueError`, 经 `self._err` (Rich) 或 `print` 报告 | 用户输入非法整数 |
| `@silent_on_err` | 静默忽略异常, 返回 `None` | best-effort 操作 |
| `@print_exc_on_err` | 捕获异常并通过 Rich Console 打印 traceback | IO 操作失败时展示详情 |

**装饰器设计约定**:
- 命名遵循 `seize_*` 前缀 (与已有 `seize_err_if_any` 一致)
- 用于实例方法时, 优先通过 `self._err` / `self._console` (Rich) 输出; 仅在没有这些属性时降级为 `print`
- 优先考虑在 API 源头提供安全方法 (如 `Bus.try_read`), 而非在各调用方使用 `@silent_on_err`

## Dependencies

- **pydantic** — 寄存器模型
- **lief** — ELF/PE 解析
- **libfdt** (pylibfdt) — 设备树 (FDT/DTB) 生成
- **rich** — 调试器终端 UI (Table, Panel, Console)
- **prompt-toolkit** — 调试器 REPL (历史, 补全)
- **loguru** — 日志 (仅 CLI 入口使用)
- **pytest** / **pytest-cov** — 测试框架
- **pyyaml** — 平台配置文件反序列化 (YAML)
- **ruff** — lint & format

## Custom LLVM toolchain

项目使用位于 `/opt/custom-llvm/bin/` 的自定义 LLVM 工具链 (基于 LLVM 22.0.0git,
作者自行修改扩展). 构建脚本 [tests/src-env/makefile](tests/src-env/makefile) 顶部通过
`custom_bin_dir=/opt/custom-llvm/bin` 引用全部工具. 各工具及常用选项:

| 工具 | 用途 | 常用参数 |
|------|------|---------|
| `llvm-mc` | 汇编器 | `--arch=riscv64 --triple=riscv64 --mattr=+m,+a` |
| `clang` | C 编译器 | `-march=rv64imac -mabi=lp64 -ffreestanding -nostdlib -O2` |
| `ld.lld` | 链接器 | `-T <script.ld> <input.o...> -o <out.elf>` |
| `llvm-objdump` | 反汇编 | `-d` (反汇编), 查看 M 扩展指令需加 `--mattr=+m` |
| `llvm-objcopy` | 格式转换 | `-O binary <in.elf> <out.bin>` |
| `llvm-readelf` | ELF 分析 | `-s` (符号表), `-S` (段头), `-h` (文件头), `-x <section>` (hex dump) |

### 关键注意事项

1. **`llvm-objdump` 解码 M 扩展指令**: 默认不启用 M 扩展反汇编,
   所有 mul/div/rem 类指令显示为 `<unknown>`.
   必须加 `--mattr=+m` 才能正确解码:
   ```bash
   /opt/custom-llvm/bin/llvm-objdump -d --mattr=+m <file.o>
   ```
   **指令本身编码正确, emulator 可正常执行** — 这只是反汇编显示问题.

2. **`--mattr=+m,+a`**: makefile 中汇编器使用此标志启用 M (乘除) 和 A (原子) 扩展.
   若 C 代码使用除法 (产生 `div`/`rem` 指令), 编译器需 `-march=rv64imac` (含 `m`).

3. **`li` 伪指令**: 加载大于 32 位的常量时 `li` 展开为多指令序列,
   调试时建议用 `llvm-objdump -d --mattr=+m` 确认实际编码.

TEE 飞地扩展细节见 [[tee-enclave-extension]] (本地介绍文件).
