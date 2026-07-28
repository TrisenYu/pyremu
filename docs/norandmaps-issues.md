# zsh 崩溃排查记录

## 调试环境

- `nokaslr norandmaps`: 地址稳定
- 固件: custom OpenSBI + fw_jump, FW_JUMP_ADDR=0x80200000
- 内核: Linux Image 加载于 0x80200000
- RAM: 2G @ 0x80000000

## 崩溃现场 (稳定复现)

```
PC   = 0x2aaaaf1a60  PA = 0x81003a60
      sb x14, 0(x15)     # x15 = x8 + x18 = 0x2AAAB661F0 + (-560) = 0x2AAAB65FC0 (代码段!)
sp   = 0x3ffffffb40  PA = 0x81051b40
x8   = 0x2AAAB661F0   (s0: 结构体指针)
x18  = 0xFFFFFFFFFFFFFDD0 = -560  (s2: 偏移, 错误值)
```

## 已确认的指令链

```
1. c.mv x18, x10; c.addi x18, 8    # x18 = 合法指针 0x2AAAB72368
2. c.lwsp x18, 16(sp)  raw=0x4912  # 0x2aaaaf18a2: x18 ← 栈值(32-bit sext)
3. bne x18, x0, 0x...a52            # 非零 → 跳过热初始化
4. c.mv x11, x18; c.li x10, 6       # func(6, -560)
5. add x15, x8, x18                 # x15 = s0 - 560
6. sb x14, 0(x15)                    # 写 1 到 *(s0-560) → 覆写代码段
```

## 污染源

sp+0x10 (VA 0x3ffffffb50, PA 0x81051b50):
```
d0 fd ff ff 3f 00 00 00  → u64 = 0x0000003FFFFFFDD0
```
低 32-bit = 0xFFFFFDD0 ≠ 0 → 程序走"已初始化"旧路径 → 崩溃。

**期望值**: 第一次调用该函数时 sp+0x10 应为 0（触发初始化路径）。

## 排除项

| 项目 | 结论 |
|------|------|
| 指令解码 | C.LWSP raw=0x4912 正确编码 ✓ |
| ALU | 65 测试全通过 ✓ |
| 固件-内核物理重叠 | 固件终点 0x8018E225 < 内核起点 0x80200000 ✓ |
| 当前场景 | 无 TEE 飞地/无 PMP 域隔离需求, 纯 Linux 启动 |

## 待排查方向

1. **栈页清零**: 内核分配新栈页时需 memset(0)。若 TLB 返回错误 PPN → 清零写到错误物理页 → 真正映射的页保留脏数据。
2. **OpenSBI init_sm() 副作用**: `activate_lpmp(0)` 和 `init_owners_bitmap()` 在纯 Linux 启动中不应影响内存, 但需验证 PMP 条目是否意外阻挡了内核内存访问。
3. **DTB memory 节点**: emulator 生成的 DTB 报告 `[0x80000000, 2G]` 全部可用, 未标记固件占用区为 reserved。内核可能分配固件区域内的物理页。
4. **QEMU 对照**: 同位置 sp+0x10 值, 若为 0 则确认是 pyremu 侧问题。
