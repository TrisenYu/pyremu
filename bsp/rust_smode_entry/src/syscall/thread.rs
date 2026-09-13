//! 线程/进程类系统调用粘合层。
//!
//! 仅做寄存器参数到内核原语的搬运; 实际逻辑在 crate::concurrency::thread (线程表与调度)
//! 与 crate::concurrency::futex (futex 语义)。当前线程的生命周期 syscall (clone / exit /
//! gettid / sched_yield / set_tid_address) 与同步原语 futex 都落在本文件。

#![allow(dead_code)]

use crate::ecall_aux;
use crate::concurrency::futex;
use crate::concurrency::thread;
use crate::trap::TrapGprs;

use super::EINVAL;

// ---------------------------------------------------------------
//  clone (220)
// ---------------------------------------------------------------

/// clone: 创建共享地址空间线程。返回子线程对外标识或负 errno。
/// 参数 (Linux rv64): flags = a0, stack = a1, ptid = a2, tls = a3, ctid = a4。
pub fn clone_handler(
    gprs: &TrapGprs,
    flags: u64,
    stack: u64,
    ptid: u64,
    tls: u64,
    ctid: u64,
) -> u64 {
    thread::clone_thread(gprs, flags, stack, ptid, tls, ctid)
}

// ---------------------------------------------------------------
//  exit (93) — 仅终止当前线程
// ---------------------------------------------------------------

/// exit: 终止当前线程。主线程或组内最后可运行线程退出时退化为整个飞地退出。
pub fn exit_handler(code: u64) -> u64 {
    thread::exit_current_thread(code)
}

// ---------------------------------------------------------------
//  gettid (178)
// ---------------------------------------------------------------

/// gettid: 返回当前线程的对外标识 (槽下标 + 1)。
pub fn gettid_handler() -> u64 {
    thread::current_external_thread_id()
}

// ---------------------------------------------------------------
//  sched_yield (124)
// ---------------------------------------------------------------

/// sched_yield: 主动让出 CPU。若存在其它可运行线程, 由 trap.rs 完成切换。
pub fn sched_yield_handler() -> u64 {
    thread::sched_yield_current();
    0
}

// ---------------------------------------------------------------
//  set_tid_address (96)
// ---------------------------------------------------------------

/// set_tid_address: 登记线程退出时的清零并唤醒地址, 返回当前线程对外标识。
pub fn set_tid_address_handler(addr: u64) -> u64 {
    thread::set_tid_address(addr)
}

// ---------------------------------------------------------------
//  futex (98)
// ---------------------------------------------------------------

/// futex: 参数 (Linux rv64): uaddr = a0, op = a1, val = a2,
/// timeout = a3, uaddr2 = a4。REQUEUE 以 a3 作 nr_requeue。
pub fn futex_handler(uaddr: u64, op: u64, val: u64, arg3: u64, uaddr2: u64) -> u64 {
    futex::futex(uaddr, op, val, arg3, uaddr2)
}

// ---------------------------------------------------------------
//  kill (129) / tkill (130) / tgkill (131)
// ---------------------------------------------------------------

/// 信号编号到退出码的偏移: 进程被信号终止时, shell 观察到的退出码为 128+信号编号。
const SIGNAL_EXIT_OFFSET: u64 = 128;

/// kill/tkill/tgkill: 飞地内没有信号投递机制, 非零信号按默认处置终止整个飞地,
/// 退出码取 128+信号编号 (与 U 模式同步异常终止载荷的约定一致)。
///
/// 这是 musl abort() 的必经之路: 它先 raise(SIGABRT), 失败才落到 a_crash()。
/// 若此处返回 ENOSYS, abort 会走到 a_crash, 载荷以 139 (SIGSEGV) 退出并掩盖
/// 真实终止原因 (例如 Rust std 的 panic); 走到这里则得到 134 (SIGABRT)。
/// 已安装处理函数的信号同样直接终止 —— 运行时尚未提供信号处理机制。
pub fn signal_handler(sig: u64) -> u64 {
    if sig == 0 {
        // 信号 0 仅做存在性检查, 不投递。
        return 0;
    }
    ecall_aux::enclave_call_exit(SIGNAL_EXIT_OFFSET + sig)
}

// ---------------------------------------------------------------
//  rt_sigprocmask (135)
// ---------------------------------------------------------------

// Linux 的 how 取值 (include/uapi/asm-generic/signal.h)。
const SIG_BLOCK: u64 = 0;
const SIG_UNBLOCK: u64 = 1;
const SIG_SETMASK: u64 = 2;

/// 内核侧 sigset 的字节数 (rv64)。
const KERNEL_SIGSET_SIZE: u64 = 8;

/// rt_sigprocmask: 参数 (Linux rv64) 为 how = a0, set = a1, oldset = a2,
/// sigsetsize = a3。
///
/// 运行时没有信号掩码状态: 合法的 how 一律按成功处理, 有 oldset 时写入空掩码;
/// 非法的 how 返回 EINVAL。这与 Linux 一致, 也是 libunwind 判断地址可读性的
/// 依赖 —— 它以 ~0 充当 how 调用本接口, 断言该调用必然失败并置 errno
/// (见 vendor/riscv-llvm-toolchain/libunwind/src/UnwindCursor.hpp 的 isReadableAddr)。
pub fn sigprocmask_handler(how: u64, oldset: u64, sigsetsize: u64) -> u64 {
    if how > SIG_SETMASK {
        return EINVAL;
    }
    if oldset != 0 {
        let n = core::cmp::min(sigsetsize, KERNEL_SIGSET_SIZE) as usize;
        if n > 0 {
            unsafe { core::ptr::write_bytes(oldset as *mut u8, 0, n) };
        }
    }
    0
}
