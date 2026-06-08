# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pyremu is a RISC-V emulator written in Python (CPython 3.14). It models a multi-hart in-order CPU core targeting the RV64 IMA ISA with CSRs, privilege levels (U/S/H/M/D), Sv39 MMU with TLB, and stubs for an AIA (Advanced Interrupt Architecture) controller. The project is in mid development — the integer base, M-extension, Zicsr, trap handling, and Sv39 address translation are implemented; floating-point, atomics, compressed instructions, and the top-level emulator loop are not yet wired up.

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_trap.py

# Run a specific test
uv run pytest tests/test_mmu.py::TestPTEPPN::test_ppn0_field

# Lint & format
uv run ruff check .
uv run ruff format .
```

## Architecture

### Package structure

```
pyremu/
  emulator.py               # 多核执行循环

  core/                     # 处理器核心
    hart.py                 # HartWithRegs, RiscvMode, mstatus 位常量
    decoder.py              # Hart 指令执行, AMO, 压缩指令, trap 处理
    registers.py            # Reg, FPR, GPR/FPR 工厂, CSR 定义
    trap.py                 # TrapType 枚举, cause code 映射
    disassem.py             # 反汇编 (桩)
    fetch_instr.py          # 取指单元 (桩)

  memory/                   # 内存子系统
    mmu.py                  # PTE, Sv39 页表遍历, 地址翻译
    tlb.py                  # TLB (继承 CacheBase)
    cache_base.py           # CacheBase + CacheLineBase 抽象
    l2cache.py              # L2 共享缓存, MESI 协议
    cache.py                # CacheEntry, TLB_SIZE
    bus.py                  # Bus + Device ABC
    rom.py                  # ROM 加载 (桩)

  interrupt/                # 中断子系统
    controller.py           # InterruptController ABC, IntSource
    clint.py                # CLINT 设备 (IPI + 定时器)
    aia.py                  # AIA/IMSIC (桩)

  utils/                    # 工具函数
    file_ops.py             # 文件路径操作
    wrapper.py              # 错误处理装饰器
    parse_bin.py            # ELF 解析 (LIEF)
```

### Data flow (intended)

1. `pyremu/utils/parse_bin.py` loads an ELF via [LIEF](https://github.com/lief-project/LIEF) → extracts `.text` / entry point.
2. `pyremu/emulator.py` creates harts, loads code into physical memory, and runs the multi-hart fetch–decode–execute loop.
3. Each hart (`core/hart.py` → `core/decoder.py`) executes instructions one at a time through `Hart.exec_instr(instr: int)`.

### Register model

All registers are defined in [core/registers.py](pyremu/core/registers.py) (合并了 abs_reg + gpr_fpr + csr):
- `Reg(name, alias, restricts, val: int)` — Pydantic BaseModel, general-purpose and CSR base class.
- `FPR(name, alias, restricts, val: float)` — floating-point registers.
- `CSR(Reg)` — adds `access: CsrAccess` and `xlen: int` fields, plus `strip_w()` / `strip_mmode()` / `add_dmode()` helpers.

GPRs and FPRs are pre-built lists returned by `register_gpr()` / `register_fpr()`; each hart `deepcopy`s its own set on init. x0 is hardwired to zero by convention.

CSR lookup uses `check_csr(csr_id: int) -> (bool, name)`; `register_csr()` returns a `{name: CSR}` dict for hart initialization.

### Class hierarchy

```
Reg (pydantic BaseModel, core/registers.py)
├── CSR (core/registers.py) — adds access control and bit-width
└── FPR (core/registers.py) — floating-point variant

CacheLineBase (memory/cache_base.py) — abstract cache line: tag, valid, dirty
├── TLBLine (memory/tlb.py) — TLB entry: ppn, perm, level
└── L2CacheLine (memory/l2cache.py) — L2 cache line: data, mesi state

CacheBase (memory/cache_base.py) — abstract cache: lookup/insert/flush framework
├── TLB (memory/tlb.py) — fully associative TLB with FIFO/LRU
└── L2Cache (memory/l2cache.py) — shared L2 cache with MESI

HartWithRegs (core/hart.py)
    contains: gprs[32], fprs[32], csrs (name→CSR dict), itlb/dtlb (TLB instances),
              pc, mode, _mem_read_phy, _mem_write_phy, reservation state
    provides: read/write_gpr/fpr/csr, mstatus/mtvec/mepc/… shortcut properties
└── Hart (core/decoder.py)
        adds: exec_instr(), handle_alu(), handle_op_imm(), handle_op32(),
              handle_ld(), handle_st(), handle_br(), handle_jalr(), handle_sys(),
              handle_fence(), handle_amo(), handle_compressed(),
              _take_trap(), _trap_ecall(), _trap_ebreak(),
              _trap_mret(), _trap_sret(), _mem_read(), _mem_write(),
              _translate_full(), set_memory_backend()
```

### Instruction execution

[core/decoder.py](pyremu/core/decoder.py) is the largest file. `Hart.exec_instr(instr: int)` dispatches on the 7-bit opcode via the `Opc` enum. Returns the instruction byte count (2/4) for PC advancement, or 0 if PC was already modified by the instruction.

**Instruction-field extractors** (module-level lambdas/functions):
- `parse_opcode`, `parse_rd`, `parse_func3`, `parse_rs1`, `parse_rs2`, `parse_func7`
- `parse_imm12_raw`, `parse_imm12_se`, `parse_imm20_raw`
- `parse_imm_s` (S-type), `parse_imm_b` (B-type), `parse_imm_j` (J-type)
- `parse_compressed` — detects 16-bit compressed instructions (low 2 bits ≠ 11)
- `_sext(val, bits)` — sign extension helper

**Enums** (also in decoder.py): `Opc`, `aluOp`, `sysOp`, `brFn3`, `ldFn3`, `stFn3`, `sysFn12`

**Implemented**: RV64 I (base integer: ALU, branches, loads/stores, JAL/JALR, LUI/AUIPC), M (mul/div/mulh/…), Zicsr (CSR read/write/set/clear), FENCE/FENCE.I, ECALL/EBREAK/MRET/SRET, SFENCE.VMA, RV64 32-bit word ops (ADDW/SUBW/…).

**Not yet implemented**: floating-point (F/D/Zfh), atomics (A — AMO opcode raises `NotImplementedError`), compressed (C — not detected by exec_instr), full trap delegation (medeleg/mideleg), WFI (no-op).

### Memory translation path

Memory accesses go through a full VA→PA translation pipeline:

```
_mem_read(va, size) / _mem_write(va, data)
  → _translate_full(va)
      → if satp.MODE == Bare: VA is PA
      → TLB lookup (dtlb.lookup(vpn))
          hit  → return PA
          miss → translate_va(va, satp_val, _mem_read_phy)
                   → sv39_walk(root_ppn, va, mem_read_phy)
                      3-level page table walk:
                        L1 (VPN[2]) → L2 (VPN[1]) → L3 (VPN[0])
                        supports 4 KiB pages and 2 MiB superpages
                   → insert result into TLB
      → if translation fails: trigger page fault trap
  → _mem_read_phy(pa, size) / _mem_write_phy(pa, data)
```

Key components:
- [memory/mmu.py](pyremu/memory/mmu.py) — `PTE` class (64-bit page table entry with flag getters/setters, PPN decomposition, `is_leaf()`/`is_ptr()`/`check_perm()`), `sv39_walk()`, `translate_va()`, VPN decomposition helpers (`_sv39_vpn`, `_sv48_vpn`), constants (`PAGE_SIZE`, `SATP_MODE_BARE`, `SATP_MODE_SV39`)
- [memory/tlb.py](pyremu/memory/tlb.py) — `TLB` class inheriting `CacheBase` (fully associative, FIFO/LRU): `lookup(vpn) → (hit, ppn, perm)`, `insert(vpn, ppn, perm, level)`, `flush(vpn)`, `flush_all()`
- [memory/cache.py](pyremu/memory/cache.py) — `CacheEntry` struct (vpn, ppn, perm, valid, dirty, level), `TLB_SIZE = 256`
- [memory/l2cache.py](pyremu/memory/l2cache.py) — `L2Cache` shared cache with MESI coherence protocol
- [memory/bus.py](pyremu/memory/bus.py) — `Bus` (shared physical memory + device routing) and `Device` ABC

Physical memory is injected via `Hart.set_memory_backend(read_fn, write_fn)` — the hart does not own physical memory; the caller provides callbacks `(addr, size) -> bytes` and `(addr, data) -> None`.

### Trap handling

All trap logic lives in `Hart` ([core/decoder.py](pyremu/core/decoder.py)):

**`_take_trap(cause, tval, is_interrupt)`** — central trap entry:
1. Saves `pc → mepc`, sets `mcause` and `mtval`
2. Updates mstatus: `MPIE ← MIE`, `MIE ← 0`, `MPP ← current mode`
3. Switches to M-mode
4. Jumps to mtvec: direct mode (`pc ← BASE`) or vectored mode (`pc ← BASE + 4×code`, interrupts only)

**`_trap_ecall()`** — dispatches to `EcallFromUmode`/`Smode`/`Mmode` based on `self.mode`

**`_trap_ebreak()`** — `Breakpoint` trap with `tval = pc`

**`_trap_mret()`** — restores `mode ← MPP`, `MIE ← MPIE`, `MPIE ← 1`, `MPP ← U`, `pc ← mepc`

**`_trap_sret()`** — restores `mode ← SPP`, `SIE ← SPIE`, `SPIE ← 1`, `SPP ← U`, `pc ← sepc`

Trap delegation (medeleg/mideleg) is not yet implemented — all traps go to M-mode.

Supporting definitions:
- [core/trap.py](pyremu/core/trap.py) — `TrapType` enum (14 exceptions + 10 interrupts), `trap_cause_code(trap) → int`, `trap_is_interrupt(trap) → bool`
- [core/hart.py](pyremu/core/hart.py) — mstatus bit constants (`MSTATUS_MIE`, `MSTATUS_MPIE`, `MSTATUS_MPP`, `MSTATUS_SIE`, `MSTATUS_SPIE`, `MSTATUS_SPP`, …), mode-encoding dicts (`_MPP_TO_MODE`, `_MODE_TO_MPP`, etc.)

### Privilege modes

`RiscvMode` enum in [core/hart.py](pyremu/core/hart.py): `U=0, S=1, H=2, M=4, D=8` (bit-encoded for permission masking). The `CsrAccess` enum in [core/registers.py](pyremu/core/registers.py) encodes per-mode read/write permissions but is not yet enforced at runtime.

## Testing

Tests live in [tests/](tests/) and use pytest:

| File | Cases | Coverage |
|------|-------|----------|
| [tests/test_parse_elf.py](tests/test_parse_elf.py) | 3 | ELF parsing with LIEF |
| [tests/test_trap.py](tests/test_trap.py) | 20 | `trap_cause_code`/`trap_is_interrupt` encoding, `_take_trap` context save & mtvec direct/vectored jump, ECALL (U/S/M), EBREAK, MRET/SRET state restore, SFENCE.VMA flush |
| [tests/test_tlb.py](tests/test_tlb.py) | 15 | TLB insert/lookup/miss, in-place update, FIFO eviction & wrap-around, single-VPN flush & flush-all, edge cases (zero VPN, large VPN, default size) |
| [tests/test_mmu.py](tests/test_mmu.py) | 37 | PTE flag read/write & independence, PPN decomposition/reassembly, leaf/pointer detection, permission checks (U/S, read/write/execute), VPN decomposition, Sv39 3-level 4 KiB page walk, 2 MiB superpage, Bare mode, invalid PTE handling |

## Stub modules (no implementation yet)

- [core/fetch_instr.py](pyremu/core/fetch_instr.py) — instruction fetch unit.
- [core/disassem.py](pyremu/core/disassem.py) — disassembly/formatting.
- [memory/rom.py](pyremu/memory/rom.py) — load binaries into emulated ROM.
- [interrupt/aia.py](pyremu/interrupt/aia.py) — interrupt controller (IMSIC, IPI, IOMMU).

## Code style

- **Function signatures**: opening paren `(` and return type `) -> Type:` each go on their own line, with parameters on separate indented lines:

  ```python
  def func(
      self,
      param1: int,
      param2: str,
  ) -> bool:
      ...
  ```

- **Comments**: use Chinese for higher-level explanatory comments describing architecture, design rationale, or "why"; inline code comments may be in either language.
- **Line length**: 95 chars (configured in ruff).
- **Imports**: sorted with `ruff` (isort rules), `combine-as-imports = true`.

## Error handling conventions

[utils/wrapper.py](pyremu/utils/wrapper.py) provides two decorators:
- `@seize_err_if_any(logger_enable=True)` — catches all exceptions, logs traceback, returns `None`.
- `@die_if_err` — catches all exceptions, prints traceback, calls `sys.exit(1)`.

## Dependencies

- **pydantic** — register models (`Reg`, `FPR`, `CSR`)
- **lief** — ELF parsing
- **loguru** — logging (not yet wired up in most modules)
- **pytest** — testing
- **ruff** — linting & formatting
- **pygments** — syntax highlighting (not yet used in-code)
