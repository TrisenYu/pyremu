# 调试记录: custom_opensbi_fw_payload.elf 启动流程

> 固件: `tests/bins/elf/custom_opensbi_fw_payload.elf`
> 构建: OpenSBI PLATFORM=generic, FW_PAYLOAD, PIE, 无嵌入式 DTB
> 日期: 2026-06-20

## 背景

该固件是 OpenSBI PLATFORM=generic + FW_PAYLOAD 模式编译的自定义构建。
FW_PAYLOAD 模式下 OpenSBI 和测试 payload 打包在同一个 ELF 中, 冷启动
(cold boot) 由前一级 FSBL (First Stage Boot Loader) 完成。

pyremu 充当 FSBL 角色: 加载固件、生成/注入 DTB、通过 a1 寄存器传递 FDT 地址。

## 调试过程中发现的三个 Emulator Bug

### Bug #1: `_sext()` 负 Python int 导致 BEQ/BNE 误判

**症状**: `fw_platform_init` → `fdt_path_offset(fdt, "/")` → `fdt_ro_probe_` 中
magic 检查失败, 返回 -9 (FDT_ERR_BADVERSION), 固件进入 `0x1F9D4` WFI 死循环。

**根因**: `_sext(val, 32)` 对符号位为 1 的值返回负 Python int (如 `-805306368`
代表 `0xFFFFFFFFD0000000`)。Python 任意精度整数在位运算 (`|`, `&`) 中表现为
无限个前导 1, 与通过 `LUI`+`ADDI` 路径的正 64-bit 值在 `==` 比较下不相等。
硬件中两值为同一 64-bit 比特模式, 但 `BNE` 误判为不等。

**修复**: `decoder.py:_sext()` 对 `bits≤64` 归一化到 `[0, 2^64)` 无符号范围。
`disassem.py:_fmt_imm()` 同步处理规范化值的有符号显示。

### Bug #2: C.ADD 被当作 C.MV 执行

**症状**: PIE 重定位循环中的 `c.add t5, t2` 和 `c.add t3, t2` 被当作
`c.mv t5, t2` / `c.mv t3, t2` 执行。GOT 条目全被写成 `PIE_offset` 而非
`addend + PIE_offset`, 存储目标地址也错为 `PIE_offset`。GOT 实际未被重定位,
`fw_platform_init` 解引用空指针, LdAccessFault。

**根因**: C.ADD (funct3=4, bit12=1, rs2≠0) 在 `_handle_compressed_c2` 中
落入 `rs2≠0` 分支, 未区分 bit12, 统一按 C.MV 处理 (`rd = rs2` 而非 `rd += rs2`)。

**修复**: `decoder.py:_handle_compressed_c2()` 增加 bit12 判断: bit12=1 时执行
`rd += rs2` (C.ADD), bit12=0 时执行 `rd = rs2` (C.MV)。

### Bug #3: C.SUB (RV64C sf=3) 未实现

**症状**: `_scratch_init` 中 `c.sub x15, x14` (0x8F99) 落入 `sf=0b11` 分支,
触发 NotImplementedError → IllInstr trap。

**根因**: RV64 C 扩展的 C1 ALU 操作中, `bits[11:10]=11` (sf=3) 编码了
C.SUB / C.XOR / C.OR / C.AND 指令 (3-bit 压缩寄存器), 但 handler 仅覆盖
sf=00,01,10。

**修复**: `decoder.py:_handle_compressed_c1_alu()` 新增 sf=0b11 分支,
按 bits[6:5] 分发 SUB/XOR/OR/AND。

## OpenSBI FW_PAYLOAD 启动模型分析

### 启动流程

```
_start (0x0, PIE → 0x80000000)
  ├─ fw_boot_hart() → -1 (自己就是 boot hart)
  ├─ _try_lottery → AMOSWAP 抢 lottery
  ├─ PIE 重定位 (~7,500 步)
  ├─ BSS 零填充 (~300,000 步, 约 800 KB)
  ├─ _scratch_init
  ├─ fw_platform_init: FDT 解析 /cpus, /chosen, CLINT, UART
  ├─ _start_warm → blt hart_id, hart_count → boot
  └─ sbi_init → cold boot 或 warm boot
```

### 单 hart 无法通过同步栅栏

`sbi_init` 内 `init_warmboot` 有一个 hart 同步栅栏:

```
0x8000e670: div  x10, x10, x0    # x10 = -1
0x8000e674: lbu  x10, -868(x11)  # x10 = *(x11 - 868)
0x8000e678: fence
0x8000e67c: beq  x10, x0, 0x8000e670  # 等待 flag 非零
```

DWARF 标志地址 `*(x11-868)` 由 `sbi_hsm` 模块管理。FW_PAYLOAD 模式下
`sbi_init` 发现 `next_addr` 已指向 `.payload` 段, 认为 FSBL 已完成 cold boot,
直接走 warm boot 路径。但实际没有 FSBL, cold boot 从未发生, 标志永不为 1。

**hart 数必须与 DTB /cpus 节点数一致**: `fw_platform_init` 从 DTB 获取
`hart_count`, `sbi_init` 用此值建立同步栅栏。DTB 写 4 个 CPU 但只起 2 个
hart → 死等另外 2 个。

### 直接跳转 payload 失败

`--prog-cnt=0x80200000` 绕过 OpenSBI 初始化直接执行 payload, 但
payload 通过 `jalr` 调用 SBI 函数表 → 表为空指针 → 跳转到 0x0 →
IllInstr (0x00000000 非法指令)。

### 结论

该固件不适合在 pyremu 中完整运行——`FW_PAYLOAD` 模式依赖 FSBL 完成冷启动。
如需调试 OpenSBI, 建议使用 `FW_JUMP` 或 `FW_DYNAMIC` 构建。

## 相关测试

| 文件 | 新增测例 |
|------|---------|
| `tests/test_emulator.py` | `TestRegValueCanonicalization` (4 条) |
| `tests/test_compressed.py` | C.ADD (2 条), C.SUB (1 条), C.OR (1 条) |
