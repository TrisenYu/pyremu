# Known Issues

## 1. 多 hart WFI 空闲时偶发长时间停滞

**现象**: 两 hart 均在 `cpu_do_idle` (PC=`0xffffffff80013678`)，指令计数数分钟不变。

**影响**: 轻度。内核空闲无事时自然进 WFI，watchdog 注 MSIP 后唤醒但不做有用功即回 WFI。不影响交互（Ctrl+C 正常响应）。

**排查方向**: watchdog 在 debugger 路径的推进速率（`_wfi_poll_stdin_loop` 结尾 1 tick/ms），以及 MSIP 唤醒后内核是否真正执行了调度检查。

---

## 2. TLB 陈旧条目导致用户态 SIGSEGV（偶发，badaddr 指向 OpenSBI 区域）

**现象**: `clear`/`zsh` 等用户态程序偶发 SIGSEGV，`badaddr=0x80017802`（OpenSBI 代码区），`vdisasm` 单独走页表遍历显示正确 PA 在 RAM 中，说明 Rust 引擎 TLB 返回了错误 PPN。

**影响**: 重度。偶发但致命——zsh 作为 init 崩溃会触发 kernel panic。

**修复 (2026-07-28)**:
- Rust TLB 增加 `asid: u16` 字段，`tlb_lookup` 检查 ASID 匹配，`tlb_insert` 记录当前 ASID
- Python TLB 同步增加 ASID 支持（`lookup`/`insert` 参数，`TLBLine.asid` 字段）
- 内核使用 ASID=0（无标记），ASID 修复对此 case**不生效**——根因在 batch 内 Rust 引擎 SFENCE.VMA 跨 hart 广播的时序窗

**待查**: `mark_tlb_dirty_if_stale` 在每条指令前调用，逻辑上看正确。需要在线复现时抓 TLB 条目对比 `vdisasm` 结果来定位 Rust 引擎内的具体时序。

---

## 3. `restart` 命令后 DTB 丢失导致 virtio-blk 无法探测

**现象**: `restart` 后内核 panic `VFS: Unable to mount root fs on unknown-block(254,0)`。

**根因**: `cmd_restart` 调用 `load_firmware` 后固件 ELF 覆盖了 DTB 区域，且原始 DTB 地址未保存。

**修复 (2026-07-28)**: Emulator 记录 `_dtb_addr`，`restart` 时用原始地址 + `build_dtb()` 重建完整 DTB（含 virtio-blk 节点 + 原始 bootargs）。

---

## 4. Ctrl+C 在启动阶段触发终端 EIO 崩溃

**现象**: 启动时按 Ctrl+C，prompt_toolkit 的 `tcsetattr` 返回 EIO。

**修复 (2026-07-28)**: `dispatch.py` REPL 循环捕获 `OSError` 优雅退出。

---

## 5. 多 hart 控制台输出字节级交错

**现象**: lottery_boot 测试中 4 个 hart 同时写 UART，控制台输出字符交错。

**根因**: 真实硬件 UART 无 hart 感知。旧代码 `_line_bufs` + `set_writer` 在模拟器层面做了行缓冲分行——这是过度模拟，掩盖了固件未做并发保护的问题。

**修复 (2026-07-28)**: UART 字节级即时输出（符合真实硬件行为），测试改为通过 `set_hart_log_dir` 日志文件验证每 hart 独立输出。

---

## 6. PLIC `_do_claim` 错误清除 `_level` 导致 TX 中断丢失

**现象**: 串口输出 ~8 字节后卡住，需输入触发后续输出。

**根因**: `_do_claim` 清除了 `_level[src]`，TXDATA 走 Rust inline 不调 `set_irq`，`_level` 永远不恢复，`_do_complete` 时 pending 不重挂。

**修复 (2026-07-28)**: `_do_claim` 只清 pending，level 由设备 `set_irq` 独占控制（符合 RISC-V PLIC 规范）。

---

## 7. 输入延迟——回车需额外按键触发

**现象**: 键入命令后回车不执行，需再按其他键（如 Backspace）才触发。

**根因**: stdin 经 Rust termio 环形缓冲到 Python `drain_rx()` 转发——`drain_rx` 只在 step 边界和 idle poll 周期调用，两次调用之间输入滞留在缓冲。

**修复 (2026-07-28)**: `_step_native`/`step()` 每轮前调用 `drain_rx()`，`_wfi_poll_stdin_loop` 用固定 2ms 间隔检查 ring buffer（不依赖定时器周期）。

---

## 8. M 模式 MPRV=1 时取指被 PMP 拒绝导致 Hart halted

**现象**: Hart 1 M 模式 `InstrAccessFault`，`mepc=mtval=0x80000e30`（OpenSBI 代码），`mstatus.MPRV=1` + `MPP=3`(M)。

**根因**: PMP 检查未区分取指和访存——取指应始终用当前特权级（RISC-V spec §3.1.6.3 取指无视 MPRV），但代码向 PMP 传递了含 MPRV=1 的完整 `mstatus_val`。Rust `pmp_ok` 同理。

**修复 (2026-07-28)**:
- Python: `check_instruction_fetch` 传 PMP 前清零 MPRV 位
- Rust: `pmp_ok` 和 `pmp_check` 增加 MPP=M 判断直接返回 true
- 回归测试: `test_mmode_fetch_ignores_mprv` (Python) + `test_mmode_mprv1_mpp_m_bypasses_even_if_matched` (Rust)
