# TEE 主机侧终止飞地 — 设计与安全模型

## 概述

允许宿主机 (host) 申请终止一个正在运行的飞地 (enclave)。
设计目标: **M-mode 提供机制, S-mode 飞地管理器提供策略**——host 只能申请,
飞地拥有最终决定权。

## 威胁模型

| 层级 | 假设 |
|------|------|
| M-mode (OpenSBI) | 可信。是 TCB 的一部分 |
| S-mode Rust 管理器 | 可信。运行在飞地内部, 隔离于 host |
| Host (Linux 内核) | **不可信**。可能被攻陷并尝试恶意终止飞地 |
| Host 用户进程 | 不可信。可能伪造终止请求 |

## 两阶段路线

### Phase A: Management Token (已实现)

```
CREATE(token) ──────────────→ M-mode 存 mgmt_token
                               创建飞地 → 切上下文

Host 进程 (持有 token):
REQUEST_SHUTDOWN(eid, token) → M-mode 验证 token
                                  └ 匹配: 设 shutdown_requested, 发 IPI
                                  └ 不匹配: 返回 SBI_ERR_DENIED

目标 hart:  IPI → set mip.SSIP → S-mode software interrupt
            → QUERY_REQUESTS ecall → flags=SHUTDOWN_REQUESTED
            → SHUTDOWN ecall → M-mode 清理 + 切回 host
```

**安全保证**: 只有持有 CREATE 时传入 token 的 host 进程才能申请终止飞地。
防止同一 host 上其他进程随意杀飞地。

**已知限制**: 若 host 内核被攻陷, 攻击者可从内核内存中提取 token。
需要 Phase B 解决。

### Phase B: M-mode 自签名 Attestation (设计)

替换 token 为 M-mode 对 host 完整性的签名断言:

```
CREATE(attest_policy) ──────→ M-mode 存储策略 (如 "PCR[1] 必须匹配已知
                               良好值")

Host 进程:
REQUEST_SHUTDOWN(eid) ──────→ M-mode:
                              1. 读取当前 PCR[0..N] (启动度量链)
                              2. 用 M-mode 私钥签名:
                                 quote = SIG_Mmode({PCR[0..N], eid, challenge})
                              3. 将 quote 发给 S-mode

目标 hart: IPI → SSIP → QUERY_REQUESTS → flags=SHUTDOWN_REQUESTED
            → GET_ATTESTATION_QUOTE ecall → 获取 quote + PCR 值
            → S-mode 用预置公钥验签:
                └ 签名有效且 PCR 匹配策略 → SHUTDOWN
                └ 签名无效或 PCR 不匹配 → 拒绝 (host 可能已被攻陷)
```

**安全保证**: 即使 host 内核被完全攻陷, 攻击者也**无法伪造 M-mode 的签名**。
若 host 被篡改 (如 rootkit 注入), PCR 值与配置的策略不匹配, 飞地拒绝终止。

### PCR 度量链

```
启动阶段                   PCR        度量内容
────────────────────────   ───        ────────
ZSBL (ROM)                PCR[0]     自身 (启动时固定)
OpenSBI (M-mode)          PCR[1]     .text + .rodata + 配置
Host Kernel (Linux)       PCR[2]     Image 或 vmlinux 哈希
Host 用户态 (init/systemd) PCR[3]     initramfs 或关键用户态组件
```

M-mode 在 `fw_boot_hart` 启动过程中用 SHA-256 计算并存入 `measured_boot_log[]`。
PCR 值在飞地生命周期内**不可篡改** (M-mode 内存对 host 不可访问)。

## 通信协议

### 新增 ecall (M-mode)

| Func ID | 名称 | 调用方 | 参数 | 返回 |
|---------|------|--------|------|------|
| 410 | `REQUEST_SHUTDOWN` | Host | a0=enclave_id, a1=token (Phase A) / reserved (Phase B) | a0=0 成功, 错误码 |
| 411 | `QUERY_REQUESTS` | S-mode enclave | — | a0=flags (bit0=SHUTDOWN_REQUESTED) |
| (TBD) | `GET_ATTESTATION_QUOTE` | S-mode enclave | a0=buffer_pa, a1=buf_size | a0=quote_size | (Phase B) |

### 中断路径

```
Host hart ──ecall──→ M-mode (设标志 + 存 quote)
                     │
                     ├─ 目标 hart == 当前 hart → 不发送 IPI
                     │  (host 与飞地在同一 hart 时, host 处于 ecall 阻塞,
                     │   飞地仅在 SUSPEND 后才会收到通知)
                     │
                     └─ 目标 hart != 当前 hart → send_shutdown_notification_ipi(hartid)
                                                  │
                                                  └→ 目标 hart M-mode IPI handler:
                                                     csr_set(CSR_MIP, MIP_SSIP)
                                                     → S-mode 收到 software interrupt
```

### IPI hart mask (支持 >64 核)

使用 `sbi_ipi_send_many` 的 `hmask + hbase` 两参数接口:

```c
void send_shutdown_notification_ipi(u32 hartid) {
    u32 hbase = hartid & ~63U;
    unsigned long hmask = 1UL << (hartid - hbase);
    sbi_ipi_send_many(hmask, hbase, event, NULL);
}
```

支持任意 hartid (0..32767), 不限于 64 核。

## 关键数据结构

```c
// enclave_types.h
struct enclave_meta_info {
    // ... 原有字段 ...
    u8  shutdown_requested; // host 已申请终止此飞地
    u64 mgmt_token;         // Phase A: 管理令牌 (Phase B: 替换为 attest_policy)
};

// 请求标志位
#define ENCLAVE_REQ_SHUTDOWN  (1 << 0)

// Phase B 新增:
// #define ENCLAVE_REQ_ATTESTATION_AVAILABLE (1 << 1)
// struct attestation_quote {
//     u32 pcr[4];        // PCR0..PCR3
//     u64 enclave_id;
//     u64 challenge;      // 防重放
//     u8  sig[64];        // secp256r1 ECDSA
// };
```

## SHUTDOWN 上下文切换修复

原有 `shutdown_enclave_handler` 在清理飞地资源后直接返回调用方的 S-mode,
导致 hart 卡在已释放的飞地上下文中。修复后与 SUSPEND 一致地切回 host:

```c
// 修复后:
alter_hart_ctx_for_enclave(enclave_id, HOST_MDID, trap_regs);
trap_regs->mepc += 4;
trap_regs->a0 = 0;
```

## 文件索引

| 层 | 文件 | 职责 |
|----|------|------|
| M-mode 类型 | `include/enclave_ext/enclave_types.h` | 结构体定义, func ID |
| M-mode ecall | `lib/enclave_ext/ext_ecall.c` | CREATE / REQUEST_SHUTDOWN / QUERY_REQUESTS / SHUTDOWN |
| M-mode IPI | `lib/enclave_ext/ext_ipi.c` | IPI 事件注册, shutdown 通知发送 |
| M-mode 上下文 | `lib/enclave_ext/enclave_mem_state.c` | alter_hart_ctx_for_enclave, running_hart 追踪 |
| M-mode PMP | `lib/enclave_ext/pmp_aux.c` | activate_lpmp (host/enclave 统一路径) |
| M-mode 内存 | `lib/enclave_ext/mem_man.c` | 池管理, 位图初始化 |
| S-mode 调度 | `rust_smode_entry/src/sched.rs` | tick_and_check_quota, check_pending_requests |
| S-mode trap | `rust_smode_entry/src/trap.rs` | SSIP + timer interrupt 分发 |
| S-mode ecall | `rust_smode_entry/src/ecall_aux.rs` | query_requests / suspend / exit |
| S-mode 配置 | `rust_smode_entry/config.mk` | TIME_QUOTA, TIMER_INTERVAL |
