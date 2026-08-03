# Known Issues

## 1. 多 hart WFI / TLB-shootdown IPI 死锁

**现象**: 多 hart 长时间卡在 `cpu_do_idle` (WFI) / `tlb_process_once` (OpenSBI)，
一个 hart 在 M-mode 自旋等待另一 hart 的 TLB 同步确认 (525M vs 126M 指令计数差异)。

**2026-07-31 修复**:
- `sync_msip` 改为 level-triggered (plain `load(Acquire)` 替代 `fetch_and(0xFE, AcqRel)`)，
  匹配真实 SiFive CLINT 硬件行为。`msip_pending` atomic channel 独立提供
  edge-triggered 交付，防止连续 MSIP 边沿塌陷。
- `wfi_spin` 移除 `std::thread::yield_now()`，改为连续 `spin_loop()`，
  消除接收 hart 主动让出 CPU 后被发送 hart 饿死、MSIP atomic channel 无法被检测的问题。

**待验证**: Linux 多核启动 IPI 压力测试、`PnP ACPI: disabled` 后 PLIC 中断稳定性。

---

## 2. 终端输入延迟 + Ctrl+C 不生效

**现象**:
- 键入后有可感知的字符回显延迟
- Ctrl+C 在客机 zsh 中不产生 SIGINT

**当前状态**:
- 终端 ISIG/IXON 已关闭，Ctrl+C 作为 raw 0x03 注入 UART RX FIFO
- `FfiExtIrqCtx` 跨 FFI 外部中断通知机制已实现（daemon 注入后置 pending → Rust 内联设置 SEIP/MEIP → guest 处理）
- 端到端延迟和 TTY VINTR 处理待验证
