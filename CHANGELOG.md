# Changelog

本文档记录开发过程中发现的根因级 bug 及其修复, 供后续开发参考。

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
