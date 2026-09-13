//! 控制台底层 I/O 原语。
//!
//! 文件描述符 0/1/2 为控制台 (标准输入/输出/错误), 由 fs.rs 的文件系统层在
//! 处理 read/write/writev/readv/ppoll 时特殊路由到这里; 本模块只提供两块最底层的
//! 搬运原语与输入源, 不持有任何 fd 状态。输入源是恒为空的内建桩, 见 uart_getc。

use crate::ecall_aux;

/// 控制台写入的分块字节数。
const CONSOLE_CHUNK: usize = 256;

/// 非阻塞接收一个字节。
///
/// 飞地没有字符输入源: SBI legacy 不提供非阻塞 RX, 也未定义可用的输入 ecall。
/// 恒返回 None (无数据), 由调用方按文件结束处理。若改为 panic 或空转,
/// 任何读取标准输入的载荷都会挂死而不是正常结束。
#[allow(unused)]
fn uart_getc() -> Option<u8> {
    None
}

/// 把 *len* 个字节送交控制台。逐块拷贝到运行时栈上再经 DBCN 输出:
/// DBCN 把缓冲区地址按物理地址解释, 而载荷缓冲区是只在其自身页表中有效的
/// U 模式虚拟地址, 直接传入会落到无关物理内存。拷进栈缓冲后地址由
/// va_to_pa 折算, 与运行时自身的诊断输出走同一条已验证的路径。
/// 每块一次 ecall, 由 M-mode 的控制台锁保证整块原子写出, 不与其他 hart 交织。
pub(crate) unsafe fn console_write_bytes(buf: *const u8, len: u64) -> u64 {
    let mut chunk = [0u8; CONSOLE_CHUNK];
    let mut off: u64 = 0;
    while off < len {
        let n = core::cmp::min(CONSOLE_CHUNK as u64, len - off) as usize;
        for i in 0..n {
            chunk[i] = unsafe { buf.add((off + i as u64) as usize).read_volatile() };
        }
        ecall_aux::sbi_console_write(&chunk[..n]);
        off += n as u64;
    }
    len
}

/// 从控制台读取字节到 buf, 遇换行或满 len 返回。
///
/// 输入源耗尽 (飞地没有字符输入通道) 时立即返回已读字节数的 0 值, 即文件结束,
/// 使读取方按 EOF 收尾而不是停留在等待中。
pub(crate) unsafe fn read_stdin(buf: *mut u8, len: u64) -> u64 {
    let mut count = 0;
    while count < len {
        match uart_getc() {
            Some(b) => {
                unsafe { buf.add(count as usize).write_volatile(b) };
                count += 1;
                if b == b'\n' || b == b'\r' {
                    break;
                }
            }
            None => break,
        }
    }
    count
}
