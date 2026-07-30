//! I/O 类系统调用: read(63), write(64), writev(66), close(57), lseek(62)。

use crate::uart;

use super::ENOSYS;

// ---------------------------------------------------------------
//  辅助
// ---------------------------------------------------------------

#[inline]
fn is_stdio(fd: u64) -> bool {
    fd <= 2
}

// ---------------------------------------------------------------
//  write handler (64)
// ---------------------------------------------------------------

pub fn write_handler(fd: u64, buf: *const u8, len: u64) -> u64 {
    if fd != 1 && fd != 2 {
        return ENOSYS;
    }

    let mut written = 0;
    while written < len {
        let b = unsafe { buf.add(written as usize).read_volatile() };
        uart::uart_putc(b);
        written += 1;
    }
    len
}

// ---------------------------------------------------------------
//  read handler (63)
// ---------------------------------------------------------------

/// 从 UART 阻塞读取字节到 buf，遇换行或满 len 返回。
unsafe fn read_stdin(buf: *mut u8, len: u64) -> u64 {
    let mut count = 0;
    while count < len {
        match uart::uart_getc() {
            Some(b) => {
                unsafe { buf.add(count as usize).write_volatile(b) };
                count += 1;
                if b == b'\n' || b == b'\r' {
                    break;
                }
            }
            None => continue,
        }
    }
    count
}

pub fn read_handler(fd: u64, buf: *mut u8, len: u64) -> u64 {
    if !is_stdio(fd) {
        return ENOSYS;
    }
    if fd != 0 {
        return ENOSYS;
    }
    unsafe { read_stdin(buf, len) }
}

// ---------------------------------------------------------------
//  writev handler (66) — musl __stdio_write 使用 writev 而非 write
// ---------------------------------------------------------------

pub fn writev_handler(fd: u64, iov_ptr: u64, iovcnt: u64) -> u64 {
    if fd != 1 && fd != 2 {
        return ENOSYS;
    }

    let mut total: u64 = 0;
    for i in 0..iovcnt {
        let slot = iov_ptr.wrapping_add(i.wrapping_mul(16));
        let iov_base = unsafe { (slot as *const u64).read_volatile() };
        let iov_len = unsafe { (slot.wrapping_add(8) as *const u64).read_volatile() };
        for j in 0..iov_len {
            let b = unsafe { (iov_base as *const u8).add(j as usize).read_volatile() };
            uart::uart_putc(b);
        }
        total = total.wrapping_add(iov_len);
    }
    total
}

// ---------------------------------------------------------------
//  close handler (57)
// ---------------------------------------------------------------

pub fn close_handler(_fd: u64) -> u64 {
    0 // 飞地无 fd 管理，直接返回成功
}

// ---------------------------------------------------------------
//  lseek handler (62)
// ---------------------------------------------------------------

pub fn lseek_handler(_fd: u64, _offset: u64, _whence: u64) -> u64 {
    0 // musl __stdio_exit flush 后调, 返回 0 即可
}
