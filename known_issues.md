# Known Issues

## 1. 多 hart WFI / PLIC 偶发停滞

**现象**: 多 hart 长时间卡在 `cpu_do_idle` (WFI)，指令计数不动；
启动日志中偶见 `PnP ACPI: disabled` 后 PLIC 中断丢失或 virtio-blk 探测超时。

**影响**: 轻度。内核空闲时自然进 WFI，但偶有唤醒失败导致交互无响应。

**排查方向**:
- `wfi_check_all_idle` 与 `wfi_sync_and_check` 的竞态条件
- `ext_irq` 从 daemon 注入到 Rust 退出之间的延迟是否导致 PLIC 状态滞后

---

## 2. 终端输入延迟 + Ctrl+C 不生效

**现象**:
- 键入后有可感知的字符回显延迟
- Ctrl+C 在客机 zsh 中不产生 SIGINT

**当前状态**:
- 终端 ISIG/IXON 已关闭，Ctrl+C 作为 raw 0x03 注入 UART RX FIFO
- `FfiExtIrqCtx` 跨 FFI 外部中断通知机制已实现（daemon 注入后置 pending → Rust 退出 → Python 同步 PLIC → guest 处理）
- 端到端延迟和 TTY VINTR 处理待验证
