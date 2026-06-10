# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pyremu is a RISC-V emulator written in Python (CPython 3.14). It models a multi-hart in-order CPU core targeting the RV64 IMA_C ISA with Zicsr, privilege levels (U/S/H/M/D), Sv39 MMU with TLB, L2 cache with MESI protocol, CLINT timer/IPI controller, basic peripherals (UART/SPI/I2C/GPIO), an FDT generator, and an interactive debugger (rvdb). The project is in active mid-stage development.

**Implemented**: RV64 I (base integer), M (mul/div), A (atomics — LR/SC/AMO), C (compressed), Zicsr (CSR read/write), FENCE/FENCE.I, ECALL/EBREAK/MRET/SRET, WFI (wait-for-interrupt with TW trap), SFENCE.VMA, Sv39 address translation (4 KiB pages + 2 MiB superpages), trap delegation (medeleg/mideleg) to S-mode, PMP (NAPOT/NA4/TOR, M-mode bypass, MPRV), configurable PMA (ram_base + device MMIO routing), and an interactive debugger with rich TUI + prompt_toolkit REPL (disasm, stack backtrace, multi-step, command repeat, snapshot/rollback).

**Not yet implemented**: floating-point (F/D/Zfh), AIA/IMSIC (stub), peripheral interrupt generation.

## Commands

```bash
# Run all tests (358 tests, 12 suites)
uv run pytest

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
```

## Architecture

### Package structure

```
pyremu/
  emulator.py               # 多核执行循环 (round-robin, trap 循环检测)
  debugger.py                # 交互式调试器 rvdb (prompt_toolkit + rich)

  core/                     # 处理器核心
    hart.py                 # HartWithRegs, RiscvMode, mstatus 位常量
    decoder.py              # Hart.exec_instr(), 全部指令 handler, trap 处理/委派
    registers.py            # Reg, FPR, CSR 模型 + 工厂函数 + CSR 表
    trap.py                 # TrapType 枚举, cause code ↔ name 双向映射

  memory/                   # 内存子系统
    mmu.py                  # PTE, Sv39 页表遍历, translate_va()
    tlb.py                  # TLB (继承 CacheBase, 全相联 FIFO/LRU)
    cache_base.py           # CacheBase + CacheLineBase 抽象基类
    l2cache.py              # L2 共享缓存 (MESI 协议)
    cache.py                # CacheEntry, TLB_SIZE
    bus.py                  # Bus + Device ABC + AccessFaultError + PMA 检查

  peripheral/               # 内存映射外设 (Device 子类)
    uart.py                 # SiFive 风格 UART (TX/RX/ctrl/div)
    spi.py                  # SPI 主控 (ctrl/status/tx/rx/div)
    i2c.py                  # I2C 主控 (ctrl/status/data/addr/prescaler)
    gpio.py                 # GPIO 控制器 (in/out/dir)

  interrupt/                # 中断子系统
    controller.py           # InterruptController ABC, IntSource
    clint.py                # CLINT (mtime/mtimecmp 定时器 + MSIP IPI)
    aia.py                  # AIA/IMSIC (桩)

  env_inject/               # 运行时注入
    preload.py              # Preloader — 将 shellcode 注入 RAM
    snippets.py             # 预置 shellcode 片段

  utils/                    # 工具
    file_ops.py             # 文件路径操作
    disassem.py             # RV64 反汇编器 (I/M/A/C/Zicsr/privileged)
    wrapper.py              # @seize_err_if_any / @die_if_err 装饰器
    parse_bin.py            # 固件解析 (ELF/PE/raw binary, 基于 LIEF)
    fdt.py                  # Flat Device Tree (FDT) 生成器 — 构建 DTB blob
```

### Data flow

1. `utils/parse_bin.py` → `parse_firmware()` 用 LIEF 解析 ELF/PE/raw binary → `FirmwareImage` (entry_point + segments + symbols).
2. `emulator.Emulator` 创建 harts + Bus + CLINT + peripherals, 调用 `load_firmware(image)` 将各段写入 RAM, 所有 hart 的 PC 设到入口地址.
3. 可选: `emulator.load_dtb(addr)` 将 FDT blob 加载到 RAM 并将地址写入 `a1` (x11), 供固件通过设备树发现外设.
4. 执行循环 (`step` / `run`): 每个未 halted 的 hart:
   - 若 `_waiting` (WFI 等待): 仅检查中断唤醒, 不取指/执行
   - 否则: 取指 4 字节 → `Hart.exec_instr(instr)` → 推进 PC 或跳转
   - 指令边界检查中断 → CLINT 时钟 tick.
5. `debugger.Debugger` 包装 Emulator, 提供单步 (`step_one`)、快照/回滚、REPL 命令分发.

### WFI 低功耗等待

Hart 执行 WFI 指令时:
- 若已有待处理且使能的中断 (`mip & mie ≠ 0`) → 立即返回 (NOP), 中断在指令边界投递
- 否则 hart 进入 `_waiting` 状态, PC 指向 WFI 下一条指令
- 等待中的 hart 在 `step()` 中被跳过 (不取指/执行), 仅轮询中断
- 当中断变为挂起且使能时 `_take_trap` 唤醒 hart (清除 `_waiting`)
- `mstatus.TW=1` 且非 M 模式执行 WFI → IllInstr 陷态

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

HartWithRegs (core/hart.py)
    gprs[32], fprs[32], csrs (name→CSR), itlb/dtlb,
    pc, mode, _mem_read_phy, _mem_write_phy,
    reservation (LR/SC), _halted, _consecutive_traps
    mstatus/mtvec/mepc/mcause/… 快捷 property
└── Hart (core/decoder.py)
        exec_instr() → 返回 2/4 (PC 需推进) 或 0 (PC 已被修改)
        handle_alu/handle_op_imm/handle_op32/handle_ld/handle_st/
        handle_br/handle_jalr/handle_sys/handle_fence/handle_amo/
        handle_compressed → _handle_compressed_c0/_c1/_c2
        _take_trap → _trap_deliver_smode / _trap_deliver_mmode
        _trap_ecall/_trap_ebreak/_trap_mret/_trap_sret
        _mem_read/_mem_write → _translate_full → TLB → sv39_walk → PA
```

### Instruction execution flow

`Hart.exec_instr(instr: int) -> int`:

1. 若 `parse_compressed(instr)` → `handle_compressed(instr & 0xFFFF)` → 按象限 (C0/C1/C2) 分发.
2. 否则按 `Opc(parse_opcode(instr))` 分发到对应 handler.
3. 任何 `ValueError` / `NotImplementedError` (非法编码、未实现 opcode) 统一转为 `_take_trap(TrapType.IllInstr)`.
4. 返回 0 (PC 已修改, 如跳转/trap) 或 2/4 (调用方负责 `pc += advance`).

**字段提取器** (module-level, decoder.py):
`parse_opcode`, `parse_rd`, `parse_func3`, `parse_rs1`, `parse_rs2`, `parse_func7`,
`parse_imm12_se`, `parse_imm_s`, `parse_imm_b`, `parse_imm_j`, `parse_imm20_raw`,
`parse_compressed` (低 2 位 ≠ 3 → 16-bit), `_sext(val, bits)`.

**枚举**: `Opc`, `aluOp`, `sysOp`, `brFn3`, `ldFn3`, `stFn3`, `sysFn12`, `AmoFunct5`, `AmoWidth`.

### Memory translation & PMA

```
_mem_read(va, size) / _mem_write(va, data)
  → _translate_full(va)
      → satp.MODE == Bare → VA 即 PA (直接通过)
      → dtlb.lookup(vpn)  → hit: 返回 PA
      → miss: translate_va() → sv39_walk(root_ppn, va) → 3 级页表遍历
              L1 (VPN[2]) → L2 (VPN[1]) → L3 (VPN[0])
              支持 4 KiB 叶子页和 2 MiB 超级页
              → 结果插入 TLB
      → 翻译失败 → _take_trap(InstrPageFault / LdPageFault / StPageFault)
  → _mem_read_phy(pa, size) / _mem_write_phy(pa, data)
  → Bus.read/write → 设备 MMIO (直通) → L2 缓存 → RAM
```

**PMA (Physical Memory Attributes)** — Bus 提供:
- `is_ram_addr(addr)` / `is_device_addr(addr)` / `is_valid_addr(addr)`
- `ram_base` 可配置 (默认 `0x8000_0000`), RAM 覆盖 `[ram_base, ram_base + ram_size)`
- 设备 MMIO 绕过 L2 缓存, 直通设备 (读写可能有副作用)
- `AccessFaultError` 供上层 PMA 违例使用

### Trap handling & delegation

`Hart._take_trap(cause, tval, is_interrupt)` — 委派感知的陷态入口:

1. **委派检查**: 若 `mode != M` 且 `medeleg[exc_code]` (异常) 或 `mideleg[exc_code]` (中断) 置位 → 委派到 S 模式
2. **S 模式投递** (`_trap_deliver_smode`): 保存到 `sepc`/`scause`/`stval`, 更新 `SPIE`/`SIE`/`SPP`, 切换到 S 模式, 跳转 `stvec`
3. **M 模式投递** (`_trap_deliver_mmode`): 保存到 `mepc`/`mcause`/`mtval`, 更新 `MPIE`/`MIE`/`MPP`, 切换到 M 模式, 跳转 `mtvec`
4. M 模式下发生的 trap **永不委派**

`check_pending_interrupts`:
- 完整优先级: MEI > MSI > MTI > SEI > SSI > STI
- 已委派中断需 S 模式全局使能 (`SIE`); 非委派 M 级中断可抢占 S 模式

`_trap_ecall()`: 根据 `self.mode` 分发 `EcallFromUmode` / `Smode` / `Mmode`.

`_trap_mret()`: 恢复 `mode ← MPP`, `MIE ← MPIE`, `pc ← mepc`.

`_trap_sret()`: 恢复 `mode ← SPP`, `SIE ← SPIE`, `pc ← sepc`.

连续 trap 检测: `_take_trap` 递增 `_consecutive_traps`; 指令正常执行时清零. 超过阈值 (3) 则 hart 进入 `_halted` 状态并转储全部寄存器.

**trap.py** 提供: `TrapType` 枚举 (14 异常 + 10 中断), `trap_cause_code(trap) → int`, `trap_is_interrupt(trap) → bool`, `trap_cause_name(mcause_val) → str` (mcause 值 → 可读名称).

## Debugger (rvdb)

交互式 RISC-V 调试器, 位于 [debugger.py](pyremu/debugger.py).

**依赖**: `prompt_toolkit` (REPL: 方向键历史, Tab 补全, FileHistory 持久化), `rich` (Console, Table, Panel — 彩色格式化输出).

**设计原则**: Emulator 是纯执行引擎 (无 I/O), Debugger 负责所有用户界面和状态呈现.

**Tab 补全**: 动态从 `register_gpr()` / `register_fpr()` / `register_csr()` 提取名称, 加命令名综合构建 `WordCompleter`.

**主要命令**:
- 执行: `s`/`step [n]`, `c`/`continue`, `r`/`run [n]`, `undo`/`rollback`
- 寄存器读: `regs`, `reg <name>`, `csr <name>`, `csr list`, `pc [addr]`, `mode`, `mstatus`, `tlb [vpn]`, `cache [set] [way]`, `satp`
- 寄存器写: `set <name> <val>`, `csrw <name> <val>`, `pc <addr>`
- 内存: `mem <addr> [size]`, `disasm <addr> [len]`, `sym [filt]`, `stack`/`bt`
- 配置: `hart <id>`, `help`/`?`, `q`/`quit`

**GDB 风格特性**: 空输入 (直接按回车) 重复上一条命令.

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

[pyremu/utils/fdt.py](pyremu/utils/fdt.py) — 纯 Python FDT/DTB 生成器 (无外部依赖).

- `FdtBuilder` 类: 通过 `add_node()` / `add_prop()` 构建设备树
- `build()` 返回符合规范的 DTB blob (magic=0xD00DFEED)
- Emulator 在初始化外设后可选生成 DTB, 加载到 RAM, 并写入 `a1` (x11) 寄存器

## Testing

265 tests across 12 files, all passing:

| File | Cases | 覆盖内容 |
|------|-------|---------|
| test_parse_elf.py | 14 | ELF/PE/raw 格式检测与解析, FirmwareSegment 模型 |
| test_trap.py | 100 | trap cause code 编解码, `_take_trap` M/S 投递, mtvec/stvec direct/vectored, ECALL/EBREAK/MRET/SRET, medeleg/midegl 委派, CSR 特权级检查, 访存对齐/PMA 故障, 缺页异常 (Sv39 Ld/StPageFault), 定时器中断 (MTI, 委派, 优先级), WFI (NOP/等待/定时器唤醒/IPI唤醒/TW 陷态) |
| test_pmp.py | 25 | PMP NAPOT 编解码, TOR/NA4/NAPOT 匹配, R/W/X 权限, M 模式旁路, CSR 条目范围, hart 集成 |
| test_tlb.py | 15 | TLB insert/lookup/miss, 原地更新, FIFO 驱逐与环绕, 单 VPN 刷新与全刷新 |
| test_mmu.py | 37 | PTE flag/PPN 读写, leaf/pointer 检测, 权限检查, VPN 分解, Sv39 3 级 4 KiB 页表遍历, 2 MiB 超级页, Bare 模式 |
| test_amo.py | 17 | LR/SC/AMOSWAP/AMOADD/AMOXOR/AMOAND/AMOOR/AMOMIN/AMOMAX/AMOMINU/AMOMAXU (.W/.D) |
| test_compressed.py | 19 | C0/C1/C2 全部已实现压缩指令, 含非法编码陷态 |
| test_disasm.py | 59 | 所有指令格式反汇编 (R/I/S/B/U/J + CSR + priv + AMO + compressed) |
| test_emulator.py | 20 | 多 hart 执行循环, 固件加载, 内存 dump, PC 推进, store 指令写入 RAM |
| test_bus.py | 14 | 总线读写, 设备注册与路由, PMA 检查 (RAM 范围, 设备检测, 空洞地址) |
| test_clint.py | 12 | mtime 递增, mtimecmp 定时器中断, MSIP 软件中断 |
| test_cache_base.py | 10 | CacheBase/CacheLineBase 抽象接口 |
| test_l2cache.py | 10 | L2Cache MESI 状态转换, 读写分配, 回写 |

## Stub modules (no implementation yet)

- [interrupt/aia.py](pyremu/interrupt/aia.py) — AIA/IMSIC 高级中断控制器 (仅空壳).

## Code style

- **函数签名**: 左括号 `(` 和返回类型 `) -> Type:` 各自独占一行, 参数逐行缩进.
- **注释**: 架构/设计原理/"为什么"使用中文; 行内注释可使用任一语言.
- **行宽**: 95 字符 (ruff line-length = 95).
- **Imports**: isort 排序, `combine-as-imports = true`. 所有导入必须在文件开头, 禁止在函数内部 import.
- **ruff 规则**: F, E, N, I, W, PL, PERF (忽略 E731 单行 lambda, PLR2004 魔数比较).
- **禁止导入私有符号**: 不 `from module import _PrivateName`; 对面内私有映射直接在调用点内联.
- **测试命名**: `test_<what>_<outcome>` 或 `test_<outcome>`.
- **文件行数上限**: 功能代码 (非测试) 单文件不超过 1500 行; 超过则拆分为 mixin 或模块.
- **重构文件的安全流程**: 1) 复制旧文件为 `.txt` 后缀 (避免 Python 模块发现冲突)
  2) 在新文件上修改 3) `pytest` 验证全部通过 4) 删除 `.txt` 旧文件.
- **删除文件原则**: 功能已被替代的空壳文件应删除, 并清理所有引用和文档.

## Error handling

[utils/wrapper.py](pyremu/utils/wrapper.py):
- `@seize_err_if_any(logger_enable=True)` — 捕获所有异常, 记录 traceback, 返回 `None`.
- `@die_if_err` — 捕获异常, 打印 traceback, `sys.exit(1)`.

## Dependencies

- **pydantic** — 寄存器模型
- **lief** — ELF/PE 解析
- **rich** — 调试器终端 UI (Table, Panel, Console)
- **prompt-toolkit** — 调试器 REPL (历史, 补全)
- **loguru** — 日志 (仅 CLI 入口使用)
- **pytest** / **pytest-cov** — 测试框架
- **ruff** — lint & format
