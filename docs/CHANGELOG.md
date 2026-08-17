## 2026-08-14 — 移除连续 trap 兜底 + 时钟源超时改为绝对 deadline

### 移除连续 trap 检测 (对齐 QEMU)

**背景**: 旧实现以 `_consecutive_traps` 计数器 + `TRAP_LOOP_THRESHOLD=3` 启发式在
连续 trap 时 `halt` hart (转储寄存器). QEMU 无此逻辑 — 死循环 / 非法指令 trap loop
在真实硬件上就是无限执行, 由调试器/超时接管, 不应由 emulator 自行暂停.

**修复**: 完全删除该逻辑 (Rust + Python + 配置 + 测试):
- Rust: `HartState.consecutive_traps` 字段 (以 `_pad` 5→6 保持 repr(C) 布局字节不变),
  `TRAP_LOOP_THRESHOLD`, `track_consecutive_traps` → 重命名为纯 PC 推进 `advance_pc`
- Python: `hart._consecutive_traps`, `emulator._TRAP_LOOP_THRESHOLD`,
  `trap_handler.deliver_trap` 递增, `debug/exec.py`/`debug/status.py` 显示行
- 配置: `emu-configs.mk` `TRAP_LOOP_THRESHOLD`, `configs_gen.py` 重生成

**影响**: 死循环固件不再被自动 halt, 连续执行 (run/continue) 依赖停止条件终止:
断点命中 / semihosting 停机 / 全部 hart halted / 时钟源超时 / Ctrl+Q 设备暂停.

### 时钟源超时: 相对 tick → 绝对 deadline

**现象**: `run(timeout=...)` 对 trap loop 永不触发 — `FfiWatchdogCtx.timeout_ticks`
为相对 tick 数, 看门狗以 `time_base_val + timeout_ticks` 判定; 而 trap loop 令单轮
加速执行极速退出, `time_base_val` 每轮重置, 相对 tick 永远无法越过 deadline.

**修复**: 改为绝对 deadline (`run()` 起点 mtime + `timeout_sec * timebase_hz`),
Rust 看门狗以 `cur >= timeout_deadline` 判定:
- `FfiWatchdogCtx.timeout_ticks` → `timeout_deadline` (Rust/Python/FFI 三侧同步)
- `emulator.run()` 纯 Python 路径补上等价超时判定 (原先无看门狗线程、无超时检查)

### 纯 Python `run()` 补齐断点与超时判定

native 引擎在 Rust 内联比对断点 / 看门狗线程判超时; 纯 Python 路径 (`_speedup_hart_states
is None`) 原先既不查断点也不查超时, `continue` 在无 termio 场景下会永久挂死.
新增 `_check_bp_hit_py()` 与绝对 deadline 检查, 使两条路径停止语义一致.

### 回归测试

- `tests/test_legacy_debugger.py::test_continue_dispatches`: 改为 `ram_base=0` 使
  0x1000 落在 RAM 内 + 在 0x1004 设断点, `continue` 执行完 NOP 命中断点即刻暂停.
- 删除依赖连续 trap halt 的测试: `test_consecutive_traps_halt_hart`,
  `test_consecutive_stack_overflows_count`, `test_consecutive_guard_page_traps_count`,
  `test_consecutive_traps_halt`, `test_consecutive_trap_halt_and_status`.

## 2026-07-30 — 事件驱动执行引擎 + PMP MPRV 修复 + Ctrl+Q 调试器暂停

### PMP MPRV — `clear` 崩溃根因修复 (最关键)

**现象**: OpenSBI `clear` 命令间歇性 `SIGSEGV` (`badaddr=0x80017802`, `cause=1`,
`epc=0x3ff7ebed74` in libc.so).

**根因**: `pmp_ok` 对 M 模式取指错误应用了 MPRV (RISC-V 规范 §4.1.12 明确规定
"Instruction access-fault and instruction page-fault exceptions are unaffected by
MPRV"). OpenSBI 的 `sbi_get_insn` 设 `MPRV=1 (MPP=U)` 后, 其内部的指令取指
(0x80017802) 被 PMP 以 U 模式权限检查 → `InstrAccessFault` → OpenSBI 将其转发到
S 模式 → 用户进程触发第二个 trap → 双重故障.

**修复** ([pmp.rs](pyremu/_native/cpu/src/pmp.rs), [trap.rs](pyremu/_native/cpu/src/trap.rs),
[trap_handler.py](pyremu/core/trap_handler.py)):
- M 模式取指无条件绕过 PMP (两处: `num==0` 分支 + 主 M 模式 bypass 分支)
- `deliver_trap_mmode` 在进入 M 模式 handler 时清除 MPRV (bit 17), 防止 handler
  自身的栈操作使用错误 MMU 翻译 — 两侧 (Rust + Python) 同步修复
- MPRV 清除确保嵌套中断 (在 OpenSBI `sbi_get_insn` 的 MPRV=1 窗口内) 不会
  将 M 模式栈地址用 S 模式页表翻译

### PMP `num==0` 语义修正

**现象**: PMP 条目数为 0 时, `pmp_ok` 原返回 `true` (无限制), 违反了规范 §3.7.1:
"when no PMP entries are implemented, S and U mode accesses are denied."

**修复** ([pmp.rs](pyremu/_native/cpu/src/pmp.rs)): `num==0` 时仅 M 模式允许访问,
S/U 模式全部拒绝 (M 模式取指无条件允许, MPRV 影响数据访存). 所有测试 PMP context
改用单条 permissive TOR 条目 (`[0, u64::MAX)`).

### 事件驱动 WFI 自旋 — 移除定时器休眠

完全重写 WFI 自旋逻辑 ([wfi.rs](pyremu/_native/cpu/src/interrupt/wfi.rs)):

- 移除 `park_timeout` + `wfi_threads` 线程注册/唤醒机制
- 移除三级退避 (hot spin → yield → park)
- 纯事件驱动: `spin_loop()` (前 64 次) → `yield_now()` (后续)
- 全部 `fetch_add`/`fetch_sub` 改为 Release/Acquire 序, 修复 `Relaxed` 内存序
  导致的 wfi_count 不一致

### 跨线程原子 MSIP 投递通道

**根因**: 发送方 hart 直接写 `(*states).mip |= 1<<3` — 非原子 RMW, 与接收方
`sync_mtip`/`sync_msip` 的并发 mip 修改存在数据竞争.

**修复** ([concurrent.rs](pyremu/_native/cpu/src/concurrent.rs),
[clint.rs](pyremu/_native/cpu/src/interrupt/clint.rs)):
- 新增 `ModuleState.msip_pending: Box<[AtomicU64]>` — 每个 hart 独立的原子通知槽
- `clint_write_msip_concurrent`: 发送方 `fetch_or(1<<3, Release)` 写入目标 hart 的
  原子槽, 替代直接写 `HartState.mip`
- `sync_msip`: 接收方先 `swap(0, Acquire)` 排干原子槽, 再合并入 `state.mip`
- CLINT MSIP level bit 原子 test-and-clear: `fetch_and(0xFE, AcqRel)` 替代
  load-then-store, 消除 TOCTOU 窗口 (两次 MSIP 写之间 level bit 被误清)
- 移除 `clint_write_msip_concurrent` 中的 `unpark()` (不再需要 — 无 park)
- `msip_pending: Cell<*const AtomicU64>` 延迟初始化 (Rust 侧 `ModuleState`
  创建后才能取到地址)

### 连续执行 — 移除 `max_instrs_per_hart`

**修复** ([hart_sched.rs](pyremu/_native/cpu/src/hart_sched.rs)):
- 删除 `instr_count >= max_instrs_per_hart` 检查 — harts 持续运行直到
  `stop_flag` (Ctrl+Q) 或 MMIO 退出 (设备访问回 Python)
- MMIO 退出逻辑改为: `stop_flag` 已置时不重复设置, 直接 return

### Ctrl+Q 调试器暂停 + Ctrl+C 客机透传

**设计** (QEMU Ctrl+A x 模型, 用户选择 Ctrl+Q 替代):

**终端模式** ([base.py](pyremu/debug/base.py)):
- ISIG 关闭: Ctrl+C 不生成 SIGINT, 作为 `0x03` 字节透传给客机 (zsh 收到 SIGINT)
- IXON 关闭: Ctrl+Q/Ctrl+S (XON/XOFF) 透传, 不被终端驱动消费
- ICRNL 显式开启: `\r` → `\n` 转换, Enter 键正常

**Ctrl+Q 拦截** ([base.py](pyremu/debug/base.py) `_stdin_daemon_loop`):
- Daemon 检测 `data.find(b'\x11')` → 单次 Ctrl+Q 立即暂停
- 置 `stop_flag.value = 1` + `_wake_event.set()` → Rust WFI 退出 → 主循环检测
- Ctrl+Q 之前的待处理字节先注入 UART, 再 return

**主循环检测** ([exec.py](pyremu/debug/exec.py)):
- `step()` 返回后检查 `_native_stop_flag` ≠ 0 → `_paused = True`, 进入 REPL

**SIGINT 处理** ([base.py](pyremu/debug/base.py)):
- 运行模式: `signal(SIGINT, SIG_IGN)` — 完全忽略, 避免 Ctrl+C 误入 prompt_toolkit

**`_step_native` stop_flag 保护** ([emulator.py](pyremu/emulator.py)):
- Daemon 已置 `stop_flag=1` 时不清零 (避免 Ctrl+Q 信号丢失)

### 页表遍历 A/D 位原子更新 (CAS)

**修复** ([translate.rs](pyremu/_native/cpu/src/translate.rs)):
- `write_pte_if_changed` → `write_pte_cas`: 用 `AtomicU64::compare_exchange(AcqRel, Acquire)`
  替代盲写 `store(Release)`
- CAS 失败 (另一 hart 并发修改 PTE) → 调用方重新读取 PTE 并验证, 避免用陈旧
  A/D 位覆盖并发 unmap/remap 操作

### TLB 代际刷新 (Python 侧)

**修复** ([emulator.py](pyremu/emulator.py), [tlb.py](pyremu/memory/tlb.py)):
- Python TLB 刷新从无条件 (每 batch 后全刷) 改为 gen 条件: 仅在 `tlb_gen` 实际
  变化时刷新 — 匹配 QEMU 风格
- `TLBLine.gen` 字段: 记录插入时的 `tlb_gen`, lookup 时比较, 不匹配则视为失效
  (SFENCE.VMA 语义)
- `_tlb_gen_before` 快照: batch 前保存 gen 值, batch 后比较决定是否刷新

### WFI mtime 推进: 无定时器时的回退

**修复** ([emulator.py](pyremu/emulator.py)):
- WFI + 全部 idle 时, 若无定时器, 仍推进 mtime 1 tick — 真实硬件时间永远流逝
- 64 位自然截断 (无 cap), 信任硬件 wrap-around 语义

### 访存指令 Acquire fence

**修复** ([hart_sched.rs](pyremu/_native/cpu/src/hart_sched.rs)):
- `fetch_instr` 在读取指令字节前加 `atomic::fence(Acquire)` — 与 `ram_write_raw`
  的 Release store 配对, 确保 hart 能观察到另一个 hart 的 store→page 指令字节

### C.LWSP / C.LDSP rd=0 保留编码陷态

**修复** ([handlers.rs](pyremu/_native/cpu/src/handlers.rs), [decoder.py](pyremu/core/decoder.py)):
- RISC-V 规范: C.LWSP 和 C.LDSP 的 rd=0 (x0) 是保留编码, 必须触发 IllInstr
- Rust 和 Python 两侧同步修复, 新增回归测试 3 条

### C.FLDSP FS=0 陷态

**修复** ([handlers.rs](pyremu/_native/cpu/src/handlers.rs)):
- 浮点扩展未启用 (mstatus.FS=0) 时执行 C.FLDSP → IllInstr 陷态

### 反汇编改进

- C.SLLI rd=0: 不再显示 `c.?`, 改为正常显示 `c.slli x0, shamt` (HINT 语义,
  CPU 执行 NOP, 反汇编展示表观语义)
- C.FLDSP rd=0: 正常显示 `c.fldsp ft0, ...` (ft0 是合法 FP 目的地)

### 删除 `run_batch` 串行引擎

- 删除 [exec.rs](pyremu/_native/cpu/src/exec.rs) (2,403 行) + `mod exec` + `pub use exec::*`
- 删除 Python FFI 绑定 (`_lib.run_batch`) + 完整 `run_batch()` 函数 (156 行)
- `run_parallel` 是唯一执行路径

### Bus 循环导入修复

- `bus.py` 内联 `PYREMU_NO_L2` 环境变量检查, 消除 `bus → core.diag → core.__init__ → ... → bus` 循环

### Rust S-mode 运行时重构

- `call.rs` → 移入 `syscall/` 子目录 (按 ecall 功能拆分为多文件)
- 新增 `ecall_aux.rs` (辅助 ecall 包装器)
- `syscall.rs` → 拆分为 `syscall/` 模块

### TEE 内核驱动 + 缓存侧信道 PoC (新增, 未跟踪)

- `bsp/tee_enclave_drv/`: Linux 字符设备驱动 (`/dev/tee_enclave`),
  ioctl 接口 (ENTER/GET_ID/GET_MEM), SBI ecall 内联
- `tests/src-sidecache/`: cache side-channel 探测 PoC (flush+reload,
  TLB leak payload), 待集成测试

### 修复清单

| 修复 | 文件 | 影响 |
|------|------|------|
| M 模式取指忽略 MPRV | pmp.rs | `clear` 崩溃根因 |
| MPRV 清除于 M 模式 trap 入口 | trap.rs, trap_handler.py | 嵌套中断 M 栈损坏 |
| PMP num==0 拒绝 S/U | pmp.rs | 规范合规 |
| 事件驱动 WFI (无 sleep) | wfi.rs, concurrent.rs | 并发安全 + 延迟 |
| 原子 MSIP 通道 | clint.rs, concurrent.rs | 数据竞争消除 |
| 移除 max_instrs_per_hart | hart_sched.rs, emulator.py | 连续执行 |
| Ctrl+Q 暂停 + Ctrl+C 透传 | base.py, exec.py, emulator.py | 调试器 UX |
| Enter 键 (ICRNL) | base.py | 终端 I/O |
| PTE A/D 原子 CAS | translate.rs | 并发页表安全 |
| TLB gen 条件刷新 | emulator.py, tlb.py | 性能 |
| WFI mtime 回退推进 | emulator.py | 时钟单调性 |
| fetch_instr Acquire fence | hart_sched.rs | 指令可见性 |
| C.LWSP/C.LDSP rd=0 陷态 | handlers.rs, decoder.py | 规范合规 |
| C.FLDSP FS=0 陷态 | handlers.rs | 规范合规 |
| 删除 run_batch 引擎 | exec.rs, __init__.py | 代码清理 |
| Bus 循环导入 | bus.py | 导入顺序 |


## 2026-07-21 — 多核 TLB/缓存一致性 + 中断投递修复

### 多核 TLB 一致性: 跨批次 Python TLB 刷新 (最关键修复)

**现象**: Linux 多核启动后用户空间命令 (如 `ls`) 间歇性 SIGSEGV, 动态链接器
`ld-linux-riscv64-lp64d.so.1` 在 nonsense VA (如 `0x33d = NULL+0x33d`) 触发
Load page fault (cause=13), 寄存器显示指针为小整数 (如 `a5=7`) 而非有效地址。

**根因 — 三重 TLB 陈旧窗口**:

1. **Python TLB 不参与 marshal/unmarshal** ([hart.py](pyremu/core/hart.py)):
   "Python TLB is ground truth" 设计假设 Python TLB 条目在跨批次间一直有效,
   但 Rust 批量执行期间内核可能修改页表 + SFENCE.VMA — Rust TLB 正确刷新,
   Python TLB 的陈旧条目却毫发无损地存活到下一轮 Python 模式执行。
   - 修复 ([emulator.py](pyremu/emulator.py)): 每轮 Rust batch 后无条件
     `flush_all()` 全部 hart 的 itlb/dtlb。过度失效永远安全, 热 TLB 在
     下轮 Python 执行中快速重建。

2. **Rust batch 全部空闲退出时 WFI hart 的 TLB 未刷新**:
   WFI hart 在主循环中不检查 `tlb_gen`。当 batch 因全部 idle 退出时,
   WFI hart 的陈旧 Rust TLB 条目被 unmarshal 到 Python 侧, 下一轮 Python
   模式执行命中错误 VA→PA 映射。
   - 修复 ([hart_sched.rs](pyremu/_native/cpu/src/hart_sched.rs)): WFI 自旋
     返回 false (batch 退出) 时, 与主循环同样的 `tlb_gen` 比较+刷新逻辑。

3. **Python SFENCE.VMA 仅刷本地 hart**:
   匹配 RISC-V 规范单 hart 语义, 但未提供与 Rust 引擎 `tlb_gen` 广播相同的
   安全网。若 hart A 在 Python 模式下改 PTE + SFENCE.VMA, hart B 的陈旧
   Python TLB 保留到下一 batch。
   - 修复 ([decoder.py](pyremu/core/decoder.py)): Python SFENCE.VMA 同步
     刷新全部 hart 的 itlb/dtlb, 匹配 Rust 引擎广播语义。

**多核一致性完整链条** (修复后):

| 边界 | 内存 | TLB | L2 |
|------|------|-----|----|
| Python→Rust | `flush_l2()` 回写脏行 | Rust 冷启动 | `flush_l2()` |
| Rust 内部 | 共享 bytearray + TSO | `tlb_gen` 代际广播 | 绕过 L2 |
| Rust→Python | bytearray 直接可见 | `flush_all()` 无条件刷 | `invalidate_l2()` |

### 跨核 MSIP 投递: batch 边界竞争修复

**现象**: 间歇性 TLB shootdown 死锁 — hart A 向 hart B 发 MSIP 请求 shootdown,
MSIP 在 batch 执行期间到达, batch 退出后 Python 一致性检查将新到达的 MSIP
当作"已投递但 CLINT 未清除"错误清零 → IPI 永久丢失 → hart B 永不执行 shootdown。

**根因**: `_step_native` 的 MSIP 一致性检查仅依据 hart 的 mip.MSIP 和 CLINT._msip,
无法区分"旧 MSIP 已投递待清除"与"新 MSIP 在 batch 期间到达"。

- 修复 ([emulator.py](pyremu/emulator.py)): 保存 `_pre_batch_msip` (batch 前
  CLINT MSIP 电平), 一致性检查增加 `_pre_batch_msip[hid]` 条件 — 仅当 MSIP
  在 batch 前就已挂起时才允许清除 CLINT._msip。

### PLIC 电平中断语义: claim/complete 修复

**现象**: UART TX watermark 中断在一次 claim 后永久丢失 — ISR 每次仅发
FIFO 深度个字符, TXDATA 写由 Rust inline 处理不经 Python, complete 后
无人再调 `set_irq` 重新置位 pending。

- 修复 ([plic.py](pyremu/interrupt/plic.py)): 新增 `_level[]` 数组记录设备侧
  电平状态; `set_irq` 同步设置 `_level` 和 `_pending`; `_do_complete` 在
  电平仍为高时自动重新置位 pending, 匹配 QEMU `sifive_plic` 行为。

### WFI 唤醒标记生命周期修复

**现象**: `trap_mret` / `trap_sret` 无条件清除 `_wfi_woken`, 导致
`while (state != READY) wfi()` 轮询循环在 trap handler 返回后立即重回睡眠
(而非将 WFI 视为 NOP 推进 PC 重新检查条件)。

- 修复 ([trap_handler.py](pyremu/core/trap_handler.py)): `trap_mret`/`trap_sret`
  不再清除 `_wfi_woken`; 改为在 `handle_wfi` 中, 若 `_wfi_woken` 置位,
  将 WFI 视为 NOP 后清除标记。同时新增 `try_wfi_wakeup()` 供 idle poll 使用。

### PLIC DTB 布尔属性编码修正

**现象**: Linux 内核打印 `interrupt-controller: Boolean property without
a value` 警告。

- 修复 ([dtb.py](pyremu/utils/dtb.py)): `property_string("interrupt-controller", "")`
  改为 `property("interrupt-controller", b"")` — DT 布尔属性必须零长度值,
  空字符串含 `\0` 终止符 (长度为 1), 违反 DT 规范。

### 调试器 stdin try 块收紧

- 修复 ([base.py](pyremu/debug/base.py)): `_feed_uart_stdin` 中原先一个宽泛的
  try/except 包裹全部 I/O 操作, 改为在每处具体调用点 (`uart.preload`,
  `select.select`, `os.read`) 精确捕获、提前返回。

### Rust 并发测试: cross_hart_msip_wakes_target 竞态修复

- 修复 ([concurrent.rs](pyremu/_native/cpu/src/concurrent.rs)): hart 0 增加
  延迟循环 (512 迭代) 确保 hart 1 先进入 WFI; PC 处增加安全 `jal` 防止
  执行越界进入未初始化 RAM 导致 MRET 死循环覆盖 mcause。

### HartState mip_val setter 优化

- [hart.py](pyremu/core/hart.py): `mip_val` setter 改用 `_csr_write_raw`
  绕过 pydantic 模型验证 (~10× 加速 CSR 写入热路径), 同时消除递归调用风险。

### virtio-blk QueueNotify 内联处理 → WFI 空闲死锁修复

**现象**: 内核启动到 `printk: legacy bootconsole [sbi0] disabled` 后, 四个 hart
全部陷在 `cpu_do_idle` WFI 中无法推进。UART 输出出现逐行重复 (如
`console [ttySIF0] enabled` 打印两次)。间歇性发生, 与 timer 是否恰好命中有关。

**根因**: Rust batch 对 virtio `QueueNotify` 写操作采用内联处理 — 仅设
`notify_pending=1` 而不退出 batch (减少 FFI 开销)。内核写完 QueueNotify
后进入 WFI 等待 I/O 完成中断。`wfi_check_all_idle` 检测到所有 hart 空闲后,
发现有定时器待触发 → timer fast-forward → 唤醒 hart → 定时器 handler 执行
→ 设新定时器 → 再次 WFI → 循环反复。**virtqueue 始终得不到 Python 侧处理**
(仅在 batch 退出后进行), 磁盘 I/O 永久挂起。

**修复** ([handlers.rs](pyremu/_native/cpu/src/handlers.rs) +
[hart_sched.rs](pyremu/_native/cpu/src/hart_sched.rs)):

- `DevCtx` 新增 `has_pending_python_work()` 方法 — 封装"是否有设备将工作
  延迟到 Python 侧"的检查语义。当前检查 virtio `notify_pending` 标志,
  未来其他内联 MMIO 设备可直接扩展。
- `hart_worker` 在进入 `wfi_spin` **之前**调用此方法: 若为真则立即以
  `WFI_WAIT` 退出 batch, 而非进入 WFI 自旋 → Python 处理 virtqueue →
  拉高完成中断 → 下一 batch 内核立即收到 I/O 响应。
- WFI 模块 (`wfi.rs`) **完全不变** — 保持纯 CPU 概念, 不耦合任何设备类型。

## 2026-07-16

### UART 交互式终端即时回显 — QEMU 风格部分行刷新

**现象**: 修复乱码和 stdin 死锁后, 交互式 shell 仍不理想 — `# ` 提示符和字符
回显延迟显示, 需等待 `\n` 或 WFI 空闲轮询 (~50ms) 才批量出现。

**根因 — 两层缓冲**: 
1. **UART 行缓冲**: `_write_reg` 仅在遇 `\n` 时调 `_flush_hart` (带 `[hart N]`
   前缀) 输出; 不以 `\n` 结尾的字节 (提示符 `# `、字符回显) 滞留在
   `_line_bufs` 中。
2. **flush_all 加前缀**: `flush_all()` 调用 `_flush_hart`, 给每段残余字节加
   `[hart N]` 前缀。此前为修复"逐字符乱码"将 `flush_all` 限制为 WFI 条件调用,
   但副作用是部分行在非 WFI 期间完全不显示。

**修复 — QEMU 风格无前缀部分行刷新**:
- UART ([uart.py](pyremu/peripheral/uart.py)): 新增 `_flush_hart_partial(hid)` —
  输出原始字节不加 `[hart N]` 前缀; `flush_all()` 改为调用此方法。
  `_flush_hart` 保留不变, 仅在 `\n` 时调用 (完整行加前缀)。
- Emulator ([emulator.py](pyremu/emulator.py)):
  - `_step_native`: `_native_flush_uart` 之后立即无条件 `flush_all()`,
    确保每批次 Rust ring buffer 中的字节立即显示。
  - `_native_finalize`: 入口处无条件 `flush_all()`, 刷新 ECALL/MMIO
    处理中产生的额外 UART 输出。
  - 移除旧的 WFI 条件 `flush_all` 守卫 (已不需要, 部分行刷新不加前缀)。

**效果**: 完整行 (kernel log) 仍带 `[hart N]` 前缀; 部分行 (shell 提示符、
字符回显) 不加前缀直接输出, 匹配 QEMU `-nographic` 行为。

### UART TX 逐字符乱码修复 + stdin 死锁修复

**现象 A — 逐字符 `[hart 0] X` 乱码**: Linux 启动输出每字符被单独
`[hart 0]` 包裹, 产生 `[hart 0] [[hart 0]  [hart 0] 7...` 乱码, 完全不可读。

**根因 — 两层叠加**:
1. **`_native_finalize` 无条件 `flush_all()`**: `flush_all()` 在每批次末尾
   无条件调用, 导致只要 `_line_bufs` 中有未完成的字符 (未遇 `\n`) 就被立即
   刷出为独立行。
2. **OpenSBI TXDATA 轮询 → 每字符一个 batch**: OpenSBI `sifive_uart_putchar`
   先读 TXDATA 检查 TX FIFO 满 (bit 31), 再写 TXDATA。Native 引擎对 UART
   读一律 exit 到 Python, 而写则 inline 缓冲。每字符的读→写序列跨 batch,
   逐字符累积到 `_line_bufs` 后立即被 (1) 刷出 → 每字符一个 `[hart 0] X`。

**修复 A**:
- `_native_finalize` ([emulator.py](pyremu/emulator.py)): `flush_all()` 加
  `wfi_waiting > 0` 条件, 仅在全部 hart WFI 空闲时刷新残余行缓冲。
  正常以 `\n` 结尾的行在 `_write_reg` 中已即时经 `_flush_hart` 输出,
  不受此影响; shell 提示符 `# ` 无 `\n` 仍需此兜底。
- `try_handle_uart_concurrent` ([concurrent.rs](pyremu/_native/src/concurrent.rs)):
  TXDATA 读 (offset 0) 返回 `Some(0)` (FIFO 不满), 不再 exit 到 Python。
  RXDATA 读仍走 Python (需实际输入数据)。Load 路径增加 UART inline 读调用
  (与 CLINT/virtio 读 inline 并列)。

**现象 B — 客机无法接收终端输入 + `# ` 提示符不可见**:
1. **stdin 转发死锁**: `_wfi_sleep_if_idle` 阻塞在 `_wake_event.wait()` 时,
   Python 线程完全睡眠; `step()` 不返回 → `_feed_uart_stdin` (在 `step()` 之后)
   永不被调用 → 用户输入滞留 stdin buffer → UART RX 永远为空 → PLIC 中断
   永不触发 → 客机永久卡在 WFI。
2. **`# ` 提示符**: shell 输出 `# ` (无 `\n`) 后立即 `read()` → 内核 WFI。
   只有 WFI 空闲时 `flush_all()` 才将 `# ` 从行缓冲刷出。

**修复 B — WFI 睡眠内联轮询**:
- `Emulator._idle_poll_cb` ([emulator.py](pyremu/emulator.py)): 可选回调,
  debugger 在 `_enter_run_mode` 时挂载, `_enter_repl_mode` 时清除。
- `_wfi_sleep_if_idle` 修改: 当 `_idle_poll_cb` 存在时以 ~50ms 短间隔循环
  (而非单次阻塞 `wait()`), 每轮调用回调检查 stdin。回调返回 `True` 时立即
  退出睡眠, 使 PLIC 中断能在下一 batch 被投递。
- `Debugger._idle_poll` ([base.py](pyremu/debug/base.py)): 转发 stdin +
  `flush_all()`, 确保 WFI 期间 (a) 用户输入立即可用, (b) shell 提示符
  在 ~50ms 内显示而非等到 step() 返回。

### UART TX 行缓冲 + 空闲刷新 + RX PLIC 中断接线

**TX 问题 — `/bin/sh` 提示符不可见**:
- **现象**: `init=/bin/sh` 启动后内核打印 "Run /bin/sh as init process" 但 `# ` 提示符
  不出现, 用户在 QEMU 中能看到但在 pyremu 中看不到。
- **根因**: UART `_write_reg` 按行缓冲累加字节, 仅在 `\n` (0x0A) 时刷新到回调。
  shell 的 `# ` 不以换行结尾, 写入后立即阻塞 `read(stdin)` → 字符永久滞留在行缓冲。
- **修复**: 在系统进入 WFI 全空闲时调用 `flush_all()` 刷新所有 hart 的剩余行缓冲。
  空闲检测点:
  - `_native_finalize`: `_wfi_sleep_if_idle` 前 (native 并发引擎)
  - `Debugger._run_until`: `all_idle` (纯 Python 路径, cnt==0 且全部 WFI)
  - 仅在实际空闲时刷新, 不影响内核启动日志 (所有 `printk` 输出均以 `\n` 结尾,
    正常路径已即时刷新)

**设计取舍 — 逐字符 vs 行缓冲**:
  初次尝试逐字符输出 (模拟真实 UART 无行缓冲) 导致多 hart 输出字符级交错
  (`[hart 0] w[hart 1] w[hart 2] w`), 破坏了 lottery_boot 的按 hart 去交错保证。
  回退到行缓冲 + WFI 空闲刷新方案: 多 hart 输出按行隔离 (buffer per hart), 空闲时
  刷新部分行 (单 hart shell prompt) 不引入交差错乱。

**RX 问题 — 客机无法接收终端输入**:
- **现象**: 用户运行期间无法输入内容 (除 Ctrl+C 暂停), 即便键入字符客机也完全不响应。
  `Debugger._feed_uart_stdin` 确实将字符 preload 到 `_rx_buf` 并置 `IP_RXWM`,
  但 UART 未接入 PLIC → 内核驱动收不到中断 → 永不读取 RXDATA。
- **修复**:
  - `UART(plic=..., irq=...)` — 新增 PLIC 引用与中断源编号参数
  - `UART_IRQ = 10` — 对齐 QEMU virt 平台, 与 DTB `serial@.../interrupts-extended` 一致
  - `_update_plic_rx()` — 根据 RX buffer 状态 + IE.RXIE 同步 PLIC 中断线
    (buffer 非空 + RXIE=1 → `plic.set_irq(10, True)`; 反之拉低)
  - `preload()` / `_read_reg(RXDATA)` / `_write_reg(IE)` → 调 `_update_plic_rx()`
  - PLIC claim 清除 `_pending[10]` 后若 buffer 仍有剩余字节, RXDATA 读取时重新拉高
  - `emulator.py` 构造 UART 时传入 `self.plic` + `UART_IRQ`

### 默认 bootargs 清理
- 移除 `loglevel=8 debug ignore_loglevel dyndbg=...` (每次 SBI/timer 调用产生海量日志,
  严重拖慢模拟器); 调试版本注释保留供日后使用
- 默认仅有 `earlycon=sbi console=ttySIF0 random.trust_bootloader=on`
- 测试更新: `TestDeviceTree` 不再断言 `keep_bootcon`

## 2026-07-14

### 多 hart SMP 启动崩溃修复 — per-hart PMP 隔离 (native 并发引擎)

**现象**: `make emu-linux-jump hart_num=4` 启动时, 内核在 SMP bringup 阶段崩溃:
```
Oops - instruction access fault [#1]
CPU: 0 ... epc : handle_exception+0x0 ... cause: 0x1 (instruction access fault)
[<...>] cpuhp_bringup_ap → wait_for_completion → ...
```
1/2 hart 正常启动到 VFS panic; 3 hart 崩溃并可打印 Oops; 4 hart 卡在
`smp: Bringing up secondary CPUs ...` 后无任何输出。故障随 hart 数递增而加剧
—— 典型的数据竞争特征。

**根因 — native 并发引擎跨 hart 共享单一 PMP**:
- 每个 Python hart 有独立 `Pmp` ([hart.py](pyremu/core/hart.py) `self._pmp`),
  但 `_step_native` 只把 `active[0]._pmp` 送入 native, 批次后再镜像到其它 hart。
- `run_parallel` 将同一 `SharedPmpCtx` (`Arc<Send+Sync>`) 交给每个 hart 线程。
- SMP bringup 时各 hart 的 OpenSBI warm-boot 并发执行 `sbi_hart_pmp_configure`
  (清空再重写 8 条 PMP 表项)。多线程无同步地写同一 `cfg`/`addr` 裸指针数组 →
  **数据竞争 + 瞬时执行权限丢失**: 某 hart 清表项的窗口内, boot hart 取指命中
  内核 text 却被共享 PMP 拒绝 → `cause=1` 取指访问故障 (而非 `0xc` 缺页 —
  证明是物理 PMP 拒绝, 非 MMU)。

**修复 — 每 hart 独立 PMP 切片**:
- FFI 契约不变 (`FfiPmpCtx.cfg/addr` 指针), 但缓冲改为 `num_harts * 64` 项连续
  数组, hart `h` 使用 `[h*64, h*64+64)` 切片:
  - `emulator.py` `_native_marshal_pmp` / `_native_unmarshal_pmp` — 逐 hart 拷入/
    拷回自己的切片; 不再镜像; 仅在切片确被 Rust 改写时才 `sync_from_flat` (省 CSR 回写)。
  - `concurrent.rs` `run_parallel` — spawn 时 `pmp_send.cfg/addr.add(hid*64)`, 每个
    线程只见自己的切片。
  - `exec.rs` `run_batch` (已弃用) 同步按 `hid*64` 偏移。
- 回归测试 [tests/test_pmp_smp.py](tests/test_pmp_smp.py): 2/4 hart 各写不同 PMP,
  批次后互不干扰 (修前必失败 — 全被压成 hart0 的值)。

### 后续修复 — `PmpInfo.num` u8 溢出致 4 hart PMP 被静默禁用

**现象**: 上述 per-hart 切片修复后, 4 hart 启动仍无 Linux 串口输出; hart0 到达
S-mode 内核 (运行 5000 万+ 指令) 却在 `vprintk_store` 反复 `LdAccessFault`
(cause=5), 每次 printk 均故障 → 无输出; 从核卡在 `0x8000a6f8` M-mode warm-boot。
OpenSBI 报 "Boot HART PMP Count : 0", `show pmps` 全部 OFF —— 但内核仍能跑数千万
S-mode 指令, 自相矛盾 (num>0 且全 OFF 时 S-mode 应全部拒绝)。

**根因 — per-hart 切片引入的整型溢出**:
- per-hart PMP 缓冲为 `64 * num_harts` 项扁平数组, 但 `PmpInfo.num` 误算为
  `min(len(cfg), len(addr))` = *扁平总长* (应为 *每 hart* 条目数)。
- `FfiPmpCtx.num` 是 `c_uint8`。4 hart 时总长 `64*4 = 256`, 写入 u8 溢出为
  `256 & 0xFF = 0`。1~3 hart (64/128/192) 未溢出故未暴露。
- `num=0` 触发两条独立路径, 恰好互相掩盖:
  - `csr.rs` 的 pmpcfg/pmpaddr 读写以 `n < pmp.num` 为门控 → 全部 no-op/返回 0 →
    OpenSBI 探测不到任何 PMP 表项 → "PMP Count : 0" → 跳过 lpmp / 内存域配置。
  - `handlers.rs::pmp_ok` 首行 `if pmp.num == 0 { return true; }` → 放行一切访问 →
    内核照跑 (解释了"全 OFF 却能跑"的矛盾)。
- OpenSBI 未配置内存域, 4 hart SMP 交接留下不一致的 M-mode 陷态/域状态, 最终在
  内核 printk 路径引发 `LdAccessFault` 风暴。

**修复**: `PmpInfo` 按 `hart_num` 反算每 hart 条目数 `num = total // hart_num`
(单 hart 默认 `hart_num=1` 向后兼容), 并在 `num > 0xFF` 时显式 `ValueError`
防止未来再次静默溢出。`_native_marshal_pmp` 传入 `hart_num=len(self.harts)`。
- 修复后 4 hart `num=64` (与 1~3 hart 一致), 启动产生 Linux 输出, 4 个 hart 均达
  S-mode, 到达与单 hart 相同的预期 VFS panic (待挂载 rootfs)。
- 回归测试 [tests/test_pmp_smp.py](tests/test_pmp_smp.py):
  `test_pmpinfo_num_is_per_hart_not_flattened` (参数化 1/2/3/4/8 hart) +
  `test_pmpinfo_num_survives_u8_ffi_at_four_harts` (经 `FfiPmpCtx.num` u8 往返仍为
  64) —— 修前 4 hart 断言 `num==64` 必失败 (得 0)。

### 停用并移除 `run_batch` (Python 侧)

`step()` / `run()` 早已统一走 `run_parallel` (`_step_native` 的 `concurrent=True`
分支), 串行 `run_batch` 分支为死代码。移除 Python 侧 `run_batch` 调用与 import,
`_step_native` 只用 `run_parallel`。Rust `run_batch` 保留但标注 `#[deprecated]`。

### `_step_native` / `build_dtb` 拆分

- `_step_native` (268 行) 拆为编排器 + 9 个职责单一的 helper
  (`_native_marshal_{dev,pmp,clint,uart}` / `_native_unmarshal_{pmp,clint}` /
  `_native_flush_uart` / `_native_handle_exit` / `_native_finalize`)。
- `build_dtb` (125 语句) 拆为 `_dtb_{chosen,aliases,cpus,memory,clint,plic,uart,
  simple_devices,watchdog}` —— 每个函数写入一个/一组节点, 保持 begin/end 顺序。
- `_NATIVE_MAX_INSTRS` 常量改为可变成员 `self._native_max_instrs` (调试器多步覆盖
  的是实例值, 不再改写"常量"; 修正类型检查器 `Literal[100000]` 告警)。

### initramfs (debootstrap rootfs) 接入

- `utils/dtb.py`: 新增 `Initrd(start, end)` dataclass; `build_dtb(initrd=...)` 在
  `/chosen` 写入 `linux,initrd-start` / `linux,initrd-end` (u64 大端), 即使无
  bootargs 也生成 `/chosen`。
- `emulator.py`: `load_initrd(path, addr)` 写入 RAM + 记录范围, 进入生成的 DTB。
- `debug/cli.py`: `--initrd PATH` / `--initrd-addr ADDR` (默认置于 DTB 下方 2 MiB
  对齐, 避开内核 Image)。
- `makefile`: `build-initramfs` (debootstrap 目录 → `fakeroot cpio -H newc | gzip`);
  `emu-linux-jump INITRD=$(INITRAMFS)` 折叠可选 rootfs (自动追加 `root=/dev/ram0
  rdinit=...`)。
- 回归 [tests/test_initrd.py](tests/test_initrd.py): DTB u64 编码往返 + `load_initrd`。

**说明**: native 并发引擎目前不向 guest 投递 PLIC 外部中断, 故 virtio-blk
`root=/dev/vda` 会因等不到完成中断而挂起; rootfs 采用 initramfs (无需中断,
内核解包 cpio 到 tmpfs)。


### Thread-per-hart 并发执行引擎 + CSR_MTOPI 死锁修复

**背景**: Linux SMP 双核启动时 Hart 0 在 S-mode WFI 空闲, Hart 1 在 M-mode
`tlb_sync` 自旋 (等待 Hart 0 消费 TLB shootdown 事件), 系统陷入沉默。
AMO trace 显示 Hart 1 正确设置了 IPI 并递增 `tlb_sync`, 但 Hart 0 从未消费。

**根因 — OpenSBI AIA 中断路径误激活**:
`CSR_MTOPI` (0xFB0) 返回 `(0, CSR_OK)`, OpenSBI 的 `__check_ext_csr(CSR_MTOPI)`
probe 成功 → `SBI_HART_EXT_SMAIA` 被检测为存在 → `sbi_trap_handler` 选择 AIA 路径:

```c
if (sbi_hart_has_extension(..., SBI_HART_EXT_SMAIA))
    rc = sbi_trap_aia_irq();                             // ← AIA 路径
else
    rc = sbi_trap_nonaia_irq(mcause & ~MCAUSE_IRQ_MASK); // ← 本应走此路径
```

`sbi_trap_aia_irq()` 循环读取 `CSR_MTOPI` 获取最高优先级中断 ID, 但 `mtopi`
始终返回 0 (无中断) → 循环体从不执行 → `sbi_ipi_process()` 从不被调用 →
TLB shootdown handler 永远不触发 → 死锁.

**修复**:
- `csr.rs`: `0xFB0` read → `(0, CSR_ILL)`, write → `CSR_ILL`
  (OpenSBI probe 捕获 IllInstr → SMAIA 不检测 → non-AIA 路径 → 基于 mcause 正确分发)
- `registers.py`: `mtopi` 标记 `.not_implemented()` → `CsrAccessError` → IllInstr
- 其他 AIA CSR (`mvien`, `mvip`, `mvienh`, `mviph`) 保持 `(0, CSR_OK)` — 读零无副作用

**架构 — `run_parallel` (thread-per-hart)**:
`pyremu/_native/src/concurrent.rs` — 每个活跃 hart 一个 OS 线程, 取代顺序
round-robin batch. 真正并发执行, 不存在 batch 边界截断临界区的问题.
- 共享 RAM: x86 TSO 下常规 load/store 直接可见, AMO 用 `AtomicU32`/`AtomicU64`
- CLINT: `mtime`/`mtimecmp`/`msip` 均为 per-hart `Atomic*` — lock-free
- 停止机制: `AtomicBool` stop flag, 任一 hart 触发 trap/ECALL/MMIO 时协调退出
- WFI: spin-loop 检查 MSIP + mip&mie, 全 idle 检测返回 Python 做 `time.sleep()`
- 环境变量 `PYREMU_NATIVE_SERIAL=1` 回退到 `run_batch`

### 诊断字段重构: HartDiag + 调试器默认隐藏

- `state.rs`: 25 个诊断计数器打包为 `HartDiag` `#[repr(C)]` 子结构,
  `HartState` 以 `diag: HartDiag` 单字段暴露, FFI 布局不变
- `base.py`: `_show_diag` 标志位, `PYREMU_DIAG_VERBOSE=1` 控制
- `status.py`: `info` 概览 MSIP 列 + 详情 "MSIP edges"/"no-trap snap" 行默认隐藏
- `makefile`: `emu-linux-jump-diag` target 自动启用

### 控制台输出修复
- `emulator.py`: 默认 bootargs 添加 `console=ttySIF0`
- `dtb.py`: `/chosen` 恢复 `stdout-path`

## 2026-07-05

### FFI struct 布局不匹配: Python struct-of-arrays ↔ Rust array-of-structs → SIGBUS

**触发场景**: 更新 `libdecode.so` (Rust cdylib) 后, 若 Python ctypes `HartState` 字段
布局未同步更新, 运行时 Rust 在错误偏移处读取 TLB 条目 → 读到垃圾 PPN →
page walk 访问无效物理地址 → SIGBUS (signal 7).

**根因**: Rust 将 TLB 存储为 `[TlbEntry; 32]` (array-of-structs), 每个 `TlbEntry` 24 字节
连续排布. 初版 Python ctypes 定义 **误用 struct-of-arrays**: 将各字段拆为独立数组
(`itlb_vpn: uint64[32] + itlb_ppn: uint64[32] + ...`). 编译器/ABI 层面两种布局的
字节偏移完全不同:

```
 Rust AoS: TlbEntry[0]   = bytes 0-23    (vpn, ppn, perm, level, valid, mdid, _pad)
           TlbEntry[1]   = bytes 24-47   ...
 Py  SoA:  itlb_vpn[0..31]  = bytes 0-255
           itlb_ppn[0..31]  = bytes 256-511
           itlb_valid[0]    = bytes 1024-1055  ...
```

Rust 在 `itlb[0].ppn` 期望读取 offset +8 的 u64 (PPN0), 但 Python SoA 布局中
offset +8 落在 `itlb_vpn[1]` (VPN1) 内部. 被污染的 PPN 指向无效物理地址,
`sv39_walk` 用该地址读 PTE 时触发 SIGBUS.

**为什么低覆盖率测试未能发现**: 旧测试均使用 `PYREMU_NATIVE_BATCH=0` 降级为纯 Python
路径, 从未调用 `run_batch` → Rust 从不访问 TLB → 布局不匹配被静默掩盖.
第一次启用 native batch 的固件启动流程才暴露此问题.

**修复**:
1. 定义 `TlbEntry(ctypes.Structure)` — vpn(u64) + ppn(u64) + 4×u8 + pad(u32) = 24B
2. HartState 改用 `TlbEntry * 32` (array-of-structs, 与 Rust 逐字节对齐)
3. 新增 `TestNativeBatchLayout` (6 个用例) 锁死 `sizeof`/对齐/结构形态
4. CLAUDE.md 追加 "FFI struct 布局锁定" 条目, 要求修改 HartState 时两侧同步验证

**通用原则**: 每当你修改 Rust `#[repr(C)]` struct (尤其是增减字段) 时:
1. 同步更新 Python ctypes `_fields_`
2. 运行 `TestNativeBatchLayout` 验证 `sizeof()` 和结构层级
3. 确保 `cargo test` (Rust 侧) 和 `uv run pytest tests/test_emulator.py::TestNativeBatchLayout` 齐过

---


## 2026-06-29

### C.LW/C.SW 压缩指令 uimm 位域解码 swap — instr[5]↔instr[6] 互换

**症状**: 压缩指令差分测试中 `C.LW`/`C.SW` offset≠0 时读/写到错误地址
(如 offset=4 -> 解码为 offset=64 -> 越界崩溃).

**根因**: [decoder.py:1074-1080](pyremu/core/decoder.py#L1074) 中 C.LW/C.SW 的 uimm 解码
将 spec 定义的位域互换: `uimm[2]` 从 `instr[5]` 读取 (应为 `instr[6]`),
`uimm[6]` 从 `instr[6]` 读取 (应为 `instr[5]`). RISC-V C 扩展规范 Table 24.2 明确:

```
C.LW uimm = {instr[5], instr[12:10], instr[6]}  (uimm[6], uimm[5:3], uimm[2])
```

offset=0 时两个位均为 0, swap 不影响结果, 故旧测试全部通过.
offset≠0 时 (如 C.LW x8, 4(x9)) 两个位值不同, swap 导致解码错误.

**修复**: 交换 `(instr >> 5) & 0x1` 与 `(instr >> 6) & 0x1` 在 uimm 构造中的位置.
同时修复注释 `{instr[6], ..., instr[5]}` -> `{instr[5], ..., instr[6]}`.

**验证**: 100 项压缩指令差分测试 (`test_compressed_diff.py`) 全部通过,
含 offset≠0 的 C.LW/C.SW/C.LD/C.SD/C.LWSP/C.SWSP/C.LDSP/C.SDSP 用例.

### 压缩指令差分测试全覆盖 — 100 项 C vs 32-bit 行为对比

新增 [tests/test_compressed_diff.py](tests/test_compressed_diff.py),
对每条 RV64C 指令与其等价的 32-bit 指令做行为一致性验证:

- **覆盖范围**: C.ADDI, C.ADDIW, C.LI, C.LUI, C.ADDI16SP, C.ADDI4SPN,
  C.SRLI, C.SRAI, C.ANDI, C.SUB/C.XOR/C.OR/C.AND, C.SUBW/C.ADDW,
  C.LW/C.SW, C.LD/C.SD, C.SLLI, C.MV, C.JR, C.JALR,
  C.LWSP/C.LDSP, C.SWSP/C.SDSP, C.J, C.BEQZ, C.BNEZ
- **方法**: 用 LLVM 定制工具链 (`/opt/custom-llvm/bin/llvm-mc`) 生成参考机器码,
  经 `.option norvc` / `.option rvc` 分别获得 32-bit 与 16-bit 编码,
  在相同初始 GPR/内存状态下执行, 比对全部 32 个 GPR 及内存副作用
- **编码验证**: 测试编码经 `llvm-objdump -d --mattr=+c` 逐条验证与 LLVM 输出一致,
  遵循 LLVM 的内部编码约定 (sf=11->C.SUB, sf=01->C.SRAI 等)

### 调试方法: 指令级计数定位死循环

用于排查 `sbi_memcmp` 无限循环的方法论:

```python
# 在 emulator.step() 或 exec_instr() 中插入计数器
_instr_counts = {}  # PC -> 执行次数

# 每执行一条指令:
_instr_counts[hart.pc] = _instr_counts.get(hart.pc, 0) + 1
if _instr_counts[hart.pc] > 100000:
    # 反汇编当前 PC, 打印附近指令和寄存器状态
    print(f"Likely infinite loop at {pc:#x}: {disasm(pc)}")
    print(f"  executed {_instr_counts[hart.pc]} times")
    dump_regs(hart)
```

此方法可快速将死循环定位到具体指令地址, 结合 `llvm-addr2line` 映射回 C 源码行.

---

## 2026-06-28

### rv64imafdc 固件 FDT 解析死循环 — 编译器 march 缺扩展导致代码生成差异

**症状**: `rv64imafdc_ztee` (缺显式 zicsr/zifencei) 编译的 OpenSBI 固件
在 `fw_platform_init` -> `fdt_driver_init_by_offset` -> `sbi_memcmp` 中
无限循环 (110k+ `sbi_memcmp` 调用且持续增长). 而 `rv64g_ztee` (= imafd + zicsr + zifencei)
编译产物正常运行.

**2026-06-29 实测**:

| march | 启动 | 说明 |
|-------|------|------|
| `rv64g_ztee` | ✅ 正常 | 基准 (imafd + zicsr + zifencei + ztee) |
| `rv64imafdc_ztee` | ❌ 卡在 sbi_memcmp | 缺 zicsr, zifencei |
| `rv64imafdc_ztee_zicsr_zifencei` | ✅ 正常 | f/d + zicsr/zifencei 都有 |

**结论**: `zicsr`/`zifencei` 和 `f`/`d` **两者都必须保留**.
当前 [Makefile:415](bsp/custom-opensbi/Makefile#L415) 使用
`-march=rv64imafdc_ztee_zicsr_zifencei`.

**根因**: 去掉 zicsr/zifencei 或 f/d 后, 编译器 (LLVM 22 定制版) 生成不同的指令序列,
在 FDT 属性解析的 byteswap 中读取到错误数据 (`lw` 读到 `0x0F000000` 而非 `0x04000000`),
导致 prop_len 错误 -> 扫描越过 FDT 边界 -> sbi_memcmp 收到垃圾参数 -> 死循环.

**已排除的假设**:
- ❌ L2 缓存数据损坏 — 18 项压力测试通过
- ❌ 压缩指令解码 bug — C0/C1/C2 handler 全部验证
- ❌ GPR sign-extend 规范化 — 已修复且回归测试通过
- ❌ `sbi_memset`/`sbi_memcpy` 8 字节优化 — 当前已回退为逐字节版本

**下一步**: 对比 `rv64imafdc_ztee` 与 `rv64imafdc_ztee_zicsr_zifencei` 两个固件的
`fdt_get_property_namelen_` 反汇编, 定位编译器生成的差异指令.

### ZSBL `coldboot_done` 硬编码地址失效 — 固件重编译后自旋死等 (zsbl_fsbl_stub.S)

**根因**: ZSBL 向 `coldboot_done` 写入 1 以跳过 OpenSBI 的 hart lottery (AMOSWAP).
该地址以 `li t0, 0x800842F8` 硬编码在汇编中. 固件重编译后 (FPU 代码加入 BSS),
`coldboot_done` 符号从 `0x442F8` 移动到 `0x842F8`, 硬编码的旧地址仍为 `0x800442F8`,
写入无效位置. `init_warmboot` 读取正确的 `0x800842F8` 时永远为 0,
在 `div a0, a0, zero` + `lbu` + `beqz` 自旋中死循环.

**症状**: `make emu` 启动后无 OpenSBI logo 输出, hart 在 `0x8000E670`
(`init_warmboot+0x30`) 处自旋数百万周期不前进. `fw_platform_init` 可正常通过
(设备树参数正确时), 但 `sbi_init` -> `init_coldboot` -> `init_warmboot` 卡死.

**修复**:
- [zsbl_fsbl_stub.S](tests/src-env/zsbl_fsbl_stub.S): 硬编码地址改为预处理器宏
  `COLD_BOOT_DONE_ADDR` 与 `RAM_BASE`, 由 makefile 通过 `-D` 传入
- [makefile](tests/src-env/makefile): 新增 `llvm-readelf -s` 自动提取固件 ELF
  中 `coldboot_done` / `coldboot_lottery` 符号偏移, 经 Python 计算物理地址
  (`RAM_BASE + vaddr`) 后作为 `-D` 标志传入汇编器. 类似
  `rust_smode_entry/config.mk` 的配置化方式, 固件重编译后无需手动更新地址
- [makefile](makefile): `emu` 目标添加完整依赖链:
  `build-fw` -> 拷贝固件到 `tests/` -> `$(zsbl_fsbl)` (自动提取符号重建) -> 启动调试器

**设计原则**: 硬编码跨二进制地址不可靠. 构建系统应从固件符号表自动提取,
通过 `-D` 预处理器宏注入汇编器, 消除手工维护.

### 固件移除未使用的 FPU 扩展 (`f`/`d`)

**背景**: pyremu 模拟器未实现 F/D 浮点扩展. 此前 `-march=rv64imafdc_ztee` 使
`__riscv_flen` 被定义, `riscv_hardfp.S` 中 `get_f64_reg` / `put_f64_reg` 等
32×2 条 FPU 指令被编译进固件. 虽因 `MSTATUS_FS` 守卫未被调用,
但增大了 `.text` 体积, 且 `march` 包含 `f`/`d` 易导致编译器在
memset/memcpy 等函数中生成 FPU 访存指令 (`fsd`/`fld`).

**修改**: [Makefile:415](bsp/custom-opensbi/Makefile#L415):
`-march=rv64imafdc_ztee` -> `-march=rv64imac_ztee`.
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

**症状**: `custom_opensbi_fw_payload.elf` 在 `fw_platform_init` -> `fdt_ro_probe_` 中,
magic 值 `0xD00DFEED` 通过 SLLIW 拼装 (产生负 int) 与 LUI+ADDI 构建的预期值
(产生正 64-bit int) 比较, `BNE` 误判为不等, 固件进入 `0x1F9D4` WFI 死循环.

**修复**:
- [decoder.py:54-62](pyremu/core/decoder.py#L54-L62): `_sext()` 对 `bits≤64` 归一化
  到 `[0, 2^64)` 无符号范围, 确保任意路径构建的同值 bit pattern 在 Python `==` 下相等.
- [disassem.py:41-46](pyremu/utils/disassem.py#L41-L46): `_fmt_imm()` 检测 bit 63
  置位时还原为有符号显示 (如 `0xFF…F0` -> `-16`).

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
不匹配 -> 返回 `FDT_ERR_BADMAGIC` -> `0x1F9D4` WFI 死循环 (正确行为但体验差).

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

**症状**: `csrw satp` -> `sfence.vma` -> Illegal Instruction trap to M-mode ->
指令被跳过 -> TLB 可能残留旧条目.

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

## 2026-06-30

### 测试覆盖率提升 — cache / debugger / emulator 共 +44 用例

针对低覆盖模块系统性补充测试用例, 覆盖原有空白方法和关键边界路径.

**Cache 子系统** ([test_l2cache.py](tests/test_l2cache.py), +11 用例):
- `TestL2BusInterface`: `bus_read` / `bus_write` 总线别名接口
- `TestL2Invalidate`: E 状态 clean invalidate, 不存在地址 no-op, 已失效行 no-op, 无 RAM backend 不崩溃
- `TestL2HitRate`: 全命中/全 miss/混合命中率统计验证
- `TestL2Iter`: `__iter__` 有效条目迭代, 空缓存返回空列表
- `TestL2SingleWay`: 直接映射 (ways=1) 逐出语义
- `TestL2Properties::test_entries_property`: `entries` 属性返回内部列表

**Emulator** ([test_emulator.py](tests/test_emulator.py), +17 用例):
- `TestStepEdgeCases`: halted hart 跳过, 连续 trap 停止, 未实现 opcode->IllInstr
- `TestLoadFirmware`: `image=None` -> ValueError
- `TestRunTimeout`: 超时 TimeoutError 抛出, `run()` 返回 int, `yield_every=0` 不限速
- `TestEmulatorProperties`: `peripherals` 字典, `cycle`/`total_instructions` 属性, `mem_hexdump`
- `TestPeripheralInit`: SPI/I2C/GPIO 默认总线注册地址验证
- `TestDeviceTree`: `build_dtb` 返回合法 FDT magic, 含 memory 节点, `load_dtb_blob` 设 a1, `load_dtb` 完整流程

**Debugger** ([test_debugger.py](tests/test_debugger.py), +16 用例):
- `TestFmtInstrCount`: K/M/B human-readable 格式化
- `TestColorizeAsm`: unknown sentinel, branch/jump green, fence/AMO yellow, normal plain, 立即数 magenta, 寄存器保护
- `TestCtrlFlowKind`: 32-bit/16-bit 指令控制流分类 (JAL/JALR/ECALL/EBREAK/MRET/SRET=term, BEQ/BNE=branch, C.J/C.JAL/C.JR/C.JALR=term, C.BEQZ/C.BNEZ=branch)
- `TestIpBits`: 中断位定义表完整性 (9 条, bit 唯一, MSIP bit=3)
- `TestCsrDetailCommands`: 10 个 CSR 详情命令 smoke test (mcause/scause/mtvec/stvec/mip/mie/sip/sie/medeleg/mideleg)
- `TestCmdPmp`: PMP 无条目/有 TOR 条目显示
- `TestCmdPt`: Bare/Sv39 三级页表遍历显示
- `TestPrivilegeBoundary`: `_resolve_boundary_frame`, `_is_valid_mmode_code`

**总览**: cache +11, emulator +17, debugger +16 = **+44 用例**, 全部 434 通过.

### 去耦合重构完成 — 指令解码 / MMU 逻辑迁出 debugger

完成此前遗留的去耦合重构, debugger.py 中所有手动拆指令位段和 MMU 地址翻译逻辑均迁出.

**迁至 [decoder.py](pyremu/core/decoder.py)**:
- `parse_func12(instr)` — funct12 字段提取 (bits[31:20])
- `decode_c_sdsp(half)` — 16-bit c.sdsp 解码 -> `(rs2, uimm) | None`
- `decode_sd_sp(instr)` — 32-bit sd to sp 解码 -> `(rs2, imm) | None`

**迁至 [mmu.py](pyremu/memory/mmu.py)**:
- `satp_root_ppn(satp_val)` — 从 satp CSR 提取 44-bit 根页表 PPN
- `sv39_canonical_va(va)` — Sv39 规范 VA 验证 (bits[63:39] == bit[38])

**debugger.py 站点清理**: 5 处手动 opcode/funct3/funct12 提取和 2 处手写 satp/VA 操作
全部替换为 parse/mmu 函数调用.

### S-mode 飞地 mepc 循环修复 (C 固件)

[suspend_enclave_handler](bsp/custom-opensbi/lib/enclave_ext/ext_ecall.c) 新增
`trap_regs->mepc += 4` 推进 host 返回地址越过 ecall 指令, 避免 host 被无限重新执行
CREATE/ENTER ecall 的 bug.

### 新测试参考文档

新增 [docs/test-coverage-gaps.md](docs/test-coverage-gaps.md) —
cache / debugger / emulator / preload / snippets 模块的覆盖率缺口全量清单,
按优先级 (高/中/低) 分级, 供逐次补齐.

---

## 设计说明

### 避免 property setter 被绕过

CSR 模块中部分 CSR 寄存器具有"写入即产生副作用"的语义 (如 satp 触发 MMU 模式切换), 而 `HartWithRegs`
通过 property 封装了这些副作用. 当前的 CSR 访问路径有二:

1. **Python API**: `hart.satp_val = X` -> 触发 `satp_val.setter` -> 副作用执行 ✓
2. **CSR 指令执行**: `handle_sys()` -> `write_csr()` -> `csrs["satp"].val = X` -> 副作用被绕过 ✗

修复方案 (本次采用): 在 `write_csr()` 中添加白名单检测, 对已知有副作用的 CSR (目前仅 satp)
显式路由到 property setter. 长期方案: 在 CSR 模型 (`Reg` / `CSR`) 中引入 `on_write` 回调机制.
