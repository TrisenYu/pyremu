use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use crate::state::FfiUartCtx;

/// Inline read of UART shadow registers (参照 QEMU sifive_uart_read).
fn handle_uart_read(offset: u64, uart: &FfiUartCtx) -> Option<u64> {
    match offset {
        0 => Some(0), // TXDATA: bit31 = FIFO full; always 0
        0x08 => Some(uart.txctrl as u64),
        0x10 => Some(uart.ie as u64),
        0x14 => {
            // TX FIFO 恒空 (inline 即时处理), TXWM 恒为 1 驱动内核 TX 中断
            // -> TTY 输出缓冲 flush -> prompt 即时出现.
            let txwm = 1u32;
            let rxcnt = uart.rxctrl & 0x7;
            let rxwm = if uart.rx_fifo_len > rxcnt { 1u32 << 1 } else { 0u32 };
            Some((txwm | rxwm) as u64)
        }
        _ => None,
    }
}

pub(crate) fn try_handle_uart_concurrent(
    pa: u64, is_write: bool, write_data: u64, hid: u8, uart: &FfiUartCtx,
) -> Option<u64> {
    if uart.base == 0 { return None; }
    let offset = pa.wrapping_sub(uart.base);
    if offset >= 0x100 { return None; }
    if !is_write { return handle_uart_read(offset, uart); }

    // TXDATA: ring buffer (log archive) + direct stdout (参照 QEMU fd_chr_write).
    if offset == 0 && uart.tx_buf as usize != 0 {
        let ecap = uart.tx_cap / 2;
        if ecap <= 0 { return Some(0); }
        let wr_atomic = unsafe { &*(uart.tx_wr as *const AtomicU32) };
        while UART_TX_LOCK
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        { std::hint::spin_loop(); }
        let w = wr_atomic.load(Ordering::Relaxed);
        let e = (w % ecap) as usize;
        let byte = write_data as u8;
        unsafe {
            *uart.tx_buf.add(2 * e) = hid;
            *uart.tx_buf.add(2 * e + 1) = byte;
        }
        wr_atomic.store(w.wrapping_add(1), Ordering::Release);
        UART_TX_LOCK.store(false, Ordering::Release);
        // Python TX (_tx_callback) 负责 stdout 时 (no_stdout=1):
        // 仅写 ring buffer, 由 drain_tx_logs->_flush_hart->_tx_callback 输出.
        if uart.no_stdout == 0 {
            unsafe { libc::write(1, &byte as *const u8 as *const libc::c_void, 1); };
        }
        return Some(0);
    }
    // IE, TXCTRL, RXCTRL, IP, DIV -> Python for PLIC updates.
    return None;
}

pub(crate) static UART_TX_LOCK: AtomicBool = AtomicBool::new(false);
