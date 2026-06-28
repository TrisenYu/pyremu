# Changelog

本文档记录开发过程中发现的根因级 bug 及其修复, 供后续开发参考。

---

## 2026-06-28

### ZSBL `coldboot_done` 硬编码地址失效 — 固件重编译后自旋死等 (zsbl_fsbl_stub.S)

**根因**: ZSBL 向 `coldboot_done` 写入 1 以跳过 OpenSBI 的 hart lottery (AMOSWAP).
该地址以 `li t0, 0x800842F8` 硬编码在汇编中. 固件重编译后 (FPU 代码加入 BSS),
`coldboot_done` 符号从 `0x442F8` 移动到 `0x842F8`, 硬编码的旧地址仍为 `0x800442F8`,
写入无效位置. `init_warmboot` 读取正确的 `0x800842F8` 时永远为 0,
在 `div a0, a0, zero` + `lbu` + `beqz` 自旋中死循环.

**症状**: `make emu` 启动后无 OpenSBI logo 输出, hart 在 `0x8000E670`
(`init_warmboot+0x30`) 处自旋数百万周期不前进. `fw_platform_init` 可正常通过
(设备树参数正确时), 但 `sbi_init` → `init_coldboot` → `init_warmboot` 卡死.

**修复**:
- [zsbl_fsbl_stub.S](tests/src-env/zsbl_fsbl_stub.S): 硬编码地址改为预处理器宏
  `COLD_BOOT_DONE_ADDR` 与 `RAM_BASE`, 由 makefile 通过 `-D` 传入
- [makefile](tests/src-env/makefile): 新增 `llvm-readelf -s` 自动提取固件 ELF
  中 `coldboot_done` / `coldboot_lottery` 符号偏移, 经 Python 计算物理地址
  (`RAM_BASE + vaddr`) 后作为 `-D` 标志传入汇编器. 类似
  `rust_smode_entry/config.mk` 的配置化方式, 固件重编译后无需手动更新地址
- [makefile](makefile): `emu` 目标添加完整依赖链:
  `build-fw` → 拷贝固件到 `tests/` → `$(zsbl_fsbl)` (自动提取符号重建) → 启动调试器

**设计原则**: 硬编码跨二进制地址不可靠. 构建系统应从固件符号表自动提取,
通过 `-D` 预处理器宏注入汇编器, 消除手工维护.

### 固件移除未使用的 FPU 扩展 (`f`/`d`)

**背景**: pyremu 模拟器未实现 F/D 浮点扩展. 此前 `-march=rv64imafdc_ztee` 使
`__riscv_flen` 被定义, `riscv_hardfp.S` 中 `get_f64_reg` / `put_f64_reg` 等
32×2 条 FPU 指令被编译进固件. 虽因 `MSTATUS_FS` 守卫未被调用,
但增大了 `.text` 体积, 且 `march` 包含 `f`/`d` 易导致编译器在
memset/memcpy 等函数中生成 FPU 访存指令 (`fsd`/`fld`).

**修改**: [Makefile:415](third-party/custom-opensbi/Makefile#L415):
`-march=rv64imafdc_ztee` → `-march=rv64imac_ztee`.
ABI 保持 `lp64` (soft-float), 编译器不再生成任何 FPU 指令.

---

## 2026-06-19

### `_sext()` 返回负 Python int 导致 BEQ/BNE 误判 (decoder + disassembler)

**根因**: `_sext(val, bits)` 对符号位为 1 的值返回负 Python int (如 `-805306368`
代表 `0xFFFFFFFFD0000000`). Python 的任意精度整数在位运算 (`|`, `&`, `^`) 中表现
为无限个前导 `1`, 与通过 `LUI`+`ADDI` 等路径产生的正 64-bit 值 (如
`18446744072905162477` 代表同一比特模式) 在 `==` 比较下不相等.

硬件中两值均为 `0xFFFFFFFFD00DFEED`, 但 emulator 中 Python `a5 == a6` 为 `False`,
导致 `BNE a5, a6` 误分支.

**症状**: `custom_opensbi_fw_payload.elf` 在 `fw_platform_init` → `fdt_ro_probe_` 中,
magic 值 `0xD00DFEED` 通过 SLLIW 拼装 (产生负 int) 与 LUI+ADDI 构建的预期值
(产生正 64-bit int) 比较, `BNE` 误判为不等, 固件进入 `0x1F9D4` WFI 死循环.

**修复**:
- [decoder.py:54-62](pyremu/core/decoder.py#L54-L62): `_sext()` 对 `bits≤64` 归一化
  到 `[0, 2^64)` 无符号范围, 确保任意路径构建的同值 bit pattern 在 Python `==` 下相等.
- [disassem.py:41-46](pyremu/utils/disassem.py#L41-L46): `_fmt_imm()` 检测 bit 63
  置位时还原为有符号显示 (如 `0xFF…F0` → `-16`).

**新增测试** (`tests/test_emulator.py`):
- `TestRegValueCanonicalization.test_slliw_produces_64bit_canonical`
- `TestRegValueCanonicalization.test_slliw_and_lui_produce_equal_values`
- `TestRegValueCanonicalization.test_beq_with_values_from_different_paths`
- `TestRegValueCanonicalization.test_bne_with_values_from_different_paths`

**影响范围**: 所有 32-bit 操作 (SLLIW / ADDIW / SRLIW / SRAIW / OP32 等) 写入
GPR 的值现在均为规范化 64-bit 表示, 消除了此前位运算与 `==`/`!=` 比较的不一致.

---

### Debugger: 设备树默认自动生成 (`--fdt` 默认启用)

**背景**: `custom_opensbi_fw_payload.elf` 为 PLATFORM=generic 编译, 无嵌入式 DTB
(二进制内搜索不到 `d00dfeed` magic), 完全依赖 `a1` 寄存器接收外部设备树.
此前 `--fdt` 默认关闭, 用户不传参时 a1=0, `fw_platform_init` 读地址 0 的 magic
不匹配 → 返回 `FDT_ERR_BADMAGIC` → `0x1F9D4` WFI 死循环 (正确行为但体验差).

**修改**:
- [debugger.py:2156-2163](pyremu/debugger.py#L2156-L2163): `--fdt` 默认值由 `None` 改为 `-1` (auto),
  新增 `--no-fdt` 标志用于显式禁用.
- [debugger.py:2242-2250](pyremu/debugger.py#L2242-L2250): 在 `--no-fdt` 未设置时自动生成 DTB
  并写入 `ram_base + ram_size - 64 KiB`, 通过 `load_dtb()` 将地址传入所有 hart 的 a1.

**影响**: OpenSBI / Linux kernel 等依赖设备树的固件开箱即用, 无需手动传 `--fdt`.
裸金属程序 (不关心设备树) 可通过 `--no-fdt` 保持旧行为.

---

### Debugger: 反汇编/栈回溯增加段名与符号注释

[debugger.py:1667-1676](pyremu/debugger.py#L1667-L1676) — 新增 `_find_segment(addr)` 方法,
利用 `FirmwareSegment.name` (已由 `parse_bin.py` 从 ELF section headers 提取).

反汇编输出每行末尾追加 `; <symbol>  .text` 类注释:
- 符号名 (黄色高亮, 来自 `FirmwareImage.symbols`)
- 段名 (灰色, 连续同段时仅首次显示)

栈回溯 (`stack` / `bt`) 每帧 PC 后同样追加段名+符号注释.

---

### Debugger: `_exec` 助手中的 PC 推进条件修正

测试辅助 `_exec()` 此前仅在 `hart.pc == 0x8000_0000` (初始值) 时推进 PC,
导致在非初始 PC 上执行非分支指令时 PC 不推进. 改为检查 `hart.pc == saved_pc`
(指令执行前后未变化才推进), 修正了 BNE 测试的假失败.

---

## 2026-06-14

### C.JALR rd_rs1==ra 读写竞争 (decoder)

**根因**: C.JALR 指令在 `rd_rs1 == x1(ra)` 时, 先写 `ra = pc+2` 再读 `pc = gprs[rd_rs1]`.
因 rd_rs1==ra, 读到的是刚写入的 pc+2 而非原始 ra 值, 导致跳转目标错误.

**修复**: [decoder.py:1152-1155](pyremu/core/decoder.py#L1152) — 先读 `target = gprs[rd_rs1]`, 再写 ra.
新增测试 `test_c_jalr_rd_eq_rs1_uses_old_value`.

### CSRRW / CSRRS / CSRRC rd==rs1 读写竞争 (decoder)

**根因**: `csrrw rd, csr, rs1` 在 `rd==rs1` 时 (如 `csrrw sp, sscratch, sp`),
emulator 先执行 `gprs[rd] = old_csr` 修改了寄存器, 再读 `gprs[rs1]` 作为 CSR 新值.
因 rd==rs1, 读到的是已修改的旧 CSR 值, CSR 被错误写回旧值,
破坏了 sscratch 的交换语义 (U sp 丢失).

**症状**: 第一次 trap 正常, 后续 trap 返回后 sp 指向 S 栈顶而非 U 栈.

**修复**: [decoder.py:854-857](pyremu/core/decoder.py#L854) — 修改 `gprs[rd]` 前先读 `gprs[rs1]`.
新增回归测试 `tests/test_trap.py::TestCsrrwRdRs1`.

### SFENCE.VMA 编码错误 (decoder + disassembler + test)

**根因**: RISC-V SFENCE.VMA 指令的 funct12 编码为 `0x120` (funct7=0b0001001, rs2=0 时), 但 decoder 和
disassembler 均硬编码为 `0x104`, 导致合法编码的 SFENCE.VMA 被误判为非法指令 (Illegal Instruction,
mcause=2), 进而路由到 M 模式 trap handler (因为 medeleg 未委派该异常), M 模式 handler 仅做
`mepc+4; mret` 跳过该指令, SFENCE.VMA 实际从未执行, TLB 在 satp 写入后未被刷新.

**症状**: `csrw satp` → `sfence.vma` → Illegal Instruction trap to M-mode →
指令被跳过 → TLB 可能残留旧条目.

**修复**: [decoder.py:845](pyremu/core/decoder.py#L845), [disassem.py:307](pyremu/utils/disassem.py#L307),
[tests/test_trap.py:532-547](tests/test_trap.py#L532) — 将 `0x104` 改为 `0x120`, 并新增回归测试
`test_sfence_vma_rejects_wrong_funct12`.

**影响范围**: 所有依赖 SFENCE.VMA 刷新 TLB 的代码路径 (Sv39 页表切换, 进程地址空间切换).

---

### `csrw satp` 绕过 `_mmu_mode` 更新

**根因**: `HartWithRegs.satp_val` 是一个 property, 其 setter 负责同步更新 `self._mmu_mode` (从 satp
MODE 字段提取翻译模式). 然而 CSR 写入指令 (`csrrw` / `csrw` 等) 通过 `write_csr()` 方法直接设置
`self.csrs["satp"].val = val`, 完全绕过了 `satp_val` 的 property setter. 因此 `csrw satp, t0` 之后
`hart.mmu_mode` 仍为 0 (Bare), MMU 从不启用, 所有虚拟地址被当作物理地址直通.

**症状**: satp 值为 `0x8000000000080003` (MODE=Sv39), 但 `hart.mmu_mode` 返回 0 (Bare),
`_translate_addr()` 执行 identity 映射而非页表遍历, 保护页 / guard page 无效.

**修复**: [hart.py:149-156](pyremu/core/hart.py#L149) — `write_csr()` 中检测 csr_name == "satp"
时路由到 `self.satp_val = val` (经 property setter), 其余 CSR 保持原有直接赋值路径.

**影响范围**: 所有通过 CSR 指令写入 satp 的场景. 若代码使用 `hart.satp_val = X` (Python API) 则不受影响.

---

### TLB 不可迭代 (CacheBase 缺少 `__iter__`)

**根因**: `CacheBase` 定义了 `__len__` (返回有效条目数) 但未定义 `__iter__`, 导致调试/诊断脚本中
`for entry in hart.dtlb` 报 `TypeError: 'TLB' object is not iterable`.

**症状**: 无法在 REPL 或诊断脚本中遍历 TLB 条目, 排查 MMU 问题时缺失关键可见性.

**修复**: [cache_base.py:213-215](pyremu/memory/cache_base.py#L213) — 为 `CacheBase` 新增
`__iter__` 方法, 仅迭代 `valid=True` 的条目.

**影响范围**: 仅调试/诊断接口, 不影响运行时行为.

---

## 设计说明

### 避免 property setter 被绕过

CSR 模块中部分 CSR 寄存器具有"写入即产生副作用"的语义 (如 satp 触发 MMU 模式切换), 而 `HartWithRegs`
通过 property 封装了这些副作用. 当前的 CSR 访问路径有二:

1. **Python API**: `hart.satp_val = X` → 触发 `satp_val.setter` → 副作用执行 ✓
2. **CSR 指令执行**: `handle_sys()` → `write_csr()` → `csrs["satp"].val = X` → 副作用被绕过 ✗

修复方案 (本次采用): 在 `write_csr()` 中添加白名单检测, 对已知有副作用的 CSR (目前仅 satp)
显式路由到 property setter. 长期方案: 在 CSR 模型 (`Reg` / `CSR`) 中引入 `on_write` 回调机制.
