# Test Coverage Gaps

最后更新: 2026-06-30

本文档记录 pyremu 测试覆盖率较低模块中尚未覆盖的方法和边界用例, 按优先级分级供逐次补齐。

## 覆盖率总览

| 模块 | 行数 | 已覆盖 | 覆盖率 | 状态 |
|------|------|--------|--------|------|
| `pyremu/debugger.py` | 2265 | 944 | 58.32% | 已新增 16 用例 |
| `pyremu/emulator.py` | 331 | 115 | 65.26% | 已新增 17 用例 |
| `pyremu/memory/cache.py` | 10 | 7 | 30.00% | 疑似死代码 |
| `pyremu/env_inject/preload.py` | 14 | 7 | 50.00% | 待补 |
| `pyremu/env_inject/snippets.py` | 124 | 73 | 41.13% | 待补 |

---

## 1. Cache 子系统

### 高优先级

| 方法 | 位置 | 说明 |
|------|------|------|
| `CacheBase._alloc_entry()` | `cache_base.py:155` | **死代码** — TLB.insert() 和 L2 读/写都内联了自己的 victim 选择逻辑, 从不调用基类 `_alloc_entry()`. 应确认后清理 |
| `L2Cache.invalidate()` 缺陷 | `l2cache.py:387-400` | (1) 逐出时未维护 `CacheBase._tag_to_idx` 字典, 造成 stale entry (2) 第 394 行注释位于 `continue` 之后, 不可达 |

### 中优先级

| 方法 | 位置 | 说明 |
|------|------|------|
| `MESIState.SHARED` | `l2cache.py` | S 状态已定义但代码中从未进入 — 无 snoop-read 路径使 E→S 过渡可达 |
| 零长度访问 | `read()`/`write()` | `size=0` 或 `data=b""` 的行为未定义 |

### 低优先级

| 方法 | 位置 | 说明 |
|------|------|------|
| `_pick_victim_in_set()` tiebreaker | `l2cache.py:358` | 全部条目同 `last_access` 时选第一个, 等价类划分未测试 |
| `_find_index()` stale cleanup | `cache_base.py:117` | `_tag_to_idx` 脏条目防御性删除路径未命中 |

---

## 2. Emulator

### 中优先级

| 方法 | 位置 | 说明 |
|------|------|------|
| `step()` 取指故障路径 | `emulator.py:509-512` | PMP 违规或未映射 VA 时 hart 跳过, 未测试 |
| `load_firmware()` BSS 零填充 | `emulator.py:358-360` | `memsz > len(data)` 的分支需构造 BSS 段 |
| `load_firmware()` PIE 影子映射 | `emulator.py:367-376` | `load_offset != 0` 的 VMA 影子写需要 PIE ELF |
| `load_dtb_file()` | `emulator.py:296` | 从文件加载预编译 DTB |

### 低优先级

| 方法 | 位置 | 说明 |
|------|------|------|
| `dump_hart_regs()` hart_id>0 | `emulator.py:616` | 仅测了 hart 0 |
| `TimeoutError` PC=None 分支 | `emulator.py:549` | 无 hart 时的超时消息格式 |

---

## 3. Debugger

### 高优先级 (功能正确性)

| 方法 | 位置 | 说明 |
|------|------|------|
| `step_one()` WFI 等待路径 | `debugger.py:619-621` | WFI 等待中 hart 被跳过的分支 |
| `_run_loop()` 多 hart + 断点 | `debugger.py:1046-1048` | `emu.step()` + `_check_multi_hart_bp()` 多核执行路径 |
| `_run_loop()` async 模式 | `debugger.py:1036-1038` | 异步断点暂停分支 |

### 中优先级 (接口覆盖)

| 方法 | 位置 | 说明 |
|------|------|------|
| `cmd_vmem` | `debugger.py:2748` | 虚拟地址内存视图 (强制 MMU 翻译) |
| `cmd_vdisasm` | `debugger.py:2502` | 虚拟地址反汇编 (强制 MMU 翻译) |
| `cmd_watch` | `debugger.py:1168` | 写监视 (watchpoint) |
| `_try_set_symbol_bp` | `debugger.py:713` | 符号名解析为断点地址 (需 _image 含符号表) |
| `cmd_restart` DTB/preload 路径 | `debugger.py:954-968` | restart 时的 DTB 重载和 preload 重注入 |

### 低优先级 (调试/展示)

| 方法 | 位置 | 说明 |
|------|------|------|
| `_walk_prev_mode_frames` | `debugger.py:3200` | 跨特权级栈帧遍历 (需构造复杂 trap 场景) |
| `_add_prev_mode_frame` | `debugger.py:3024` | 边界帧添加 (同上) |
| `_show_trap_context` | `debugger.py:1107` | trap 上下文 Panel dump 输出验证 |
| `repl()` | `debugger.py:3664` | 主 REPL 循环 (KeyboardInterrupt, EOF, 空输入重复) |
| `main()` | `debugger.py:3803` | CLI 入口 (需进程级测试) |
| `_sigint_run` | `debugger.py:406` | 二次 Ctrl+C 强制终止 |
| `_trim_history` | `debugger.py:377` | 历史文件裁剪 + OSError 容错 |

---

## 4. `cache.py` (30% 覆盖) — 疑似死代码

`CacheEntry` 类 (7 属性: perm, level, mdid, vpn, ppn, valid, dirty) 似未被任何活跃代码使用.
功能已被 `cache_base.py` 中的 `TLBLine` / `L2CacheLine` 数据类替代.
**建议**: 确认无引用后删除.

---

## 5. `preload.py` (50% 覆盖)

- `Preloader.inject()` — `addr` 自动选择路径仅有间接覆盖
- `Preloader.inject_file()` — **未测试** (需读文件系统)

---

## 6. `snippets.py` (41% 覆盖)

未测试函数 (均需集成测试环境):
- `imm64()`, `set_sp()`, `set_gp()` — 立即数加载片段
- `csr_write()`, `set_mtvec()` — CSR 操作片段
- `switch_to_umode()` — U 模式切换 (mstatus.MPP 操作)
- `hosted_bootstrap()` — 托管程序注入脚本
- `zsbl_stub()`, `fsbl_stub()` — ZSBL/FSBL 模拟
- `opensbi_coldboot_stub()` — OpenSBI 冷启动桩

指令编码辅助函数 (`_r_type`, `_i_type`, `_u_type`, `_b_type`, `_s_type`, `_j_type`) 已通过 `test_emulator.py` 间接覆盖。

---

## 补充记录

### 2026-06-30 — cache / debugger / emulator 共 +44 用例

- **Cache** (+11): `bus_read`/`bus_write` 别名, invalidate 边界 (E-state/no-op/no-RAM), hit_rate 统计, `__iter__`/`entries`, 直接映射
- **Emulator** (+17): step() 边界 (halted/连续 trap/IllInstr), load_firmware None, run() 超时, 外设属性, SPI/I2C/GPIO 注册, build_dtb/load_dtb
- **Debugger** (+16): _fmt_instr_count, _colorize_asm, _ctrl_flow_kind*, _ip_bits, 10 个 CSR detail 命令 smoke test, cmd_pmp, cmd_pt, privilege boundary
