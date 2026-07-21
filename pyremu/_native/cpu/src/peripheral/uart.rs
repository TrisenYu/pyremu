use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use crate::state::FfiUartCtx;
use crate::diag;

pub(crate) fn try_handle_uart_concurrent(
    pa: u64,
    is_write: bool,
    write_data: u64,
    hid: u8,
    uart: &FfiUartCtx,
) -> Option<u64> {
    if uart.base == 0 {
        return None;
    }
    let offset = pa.wrapping_sub(uart.base);
    if offset >= 0x100 {
        return None; // not UART range
    }
    if !is_write {
        match offset {
            0 => {
                // TXDATA read: bit 31 = TX FIFO full flag; always 0 (infinite TX FIFO).
                return Some(0);
            }
            0x08 => {
                // TXCTRL read: use shadow copy.
                return Some(uart.txctrl as u64);
            }
            0x10 => {
                // IE read: use shadow copy.
                return Some(uart.ie as u64);
            }
            0x14 => {
                // IP read: compute from shadow registers.
                // txwm: TX FIFO occupancy < txcnt; FIFO always empty → txcnt>0 ⇒ txwm set
                let txcnt = (uart.txctrl >> 16) & 0x7;
                let txwm = if txcnt > 0 { 1u32 } else { 0u32 };
                // rxwm: rx_fifo_len > rxcnt ⇒ rxwm set
                let rxcnt = uart.rxctrl & 0x7;
                let rxwm = if uart.rx_fifo_len > rxcnt { 1u32 << 1 } else { 0u32 };
                return Some((txwm | rxwm) as u64);
            }
            _ => {}
        }
        diag::log_line(&format!(
            "[diag-uart] rd pa={:#x} off={:#x} hid={} -> fallback to Python",
            pa, offset, hid,
        ));
        return None;
    }

    // TXDATA write — buffer to shared ring buffer.
    if offset == 0 && uart.tx_buf as usize != 0 {
        let ecap = uart.tx_cap / 2;
        if ecap > 0 {
            let wr_atomic = unsafe { &*(uart.tx_wr as *const AtomicU32) };
            while UART_TX_LOCK
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                std::hint::spin_loop();
            }
            let w = wr_atomic.load(Ordering::Relaxed);
            let e = (w % ecap) as usize;
            unsafe {
                *uart.tx_buf.add(2 * e) = hid;
                *uart.tx_buf.add(2 * e + 1) = write_data as u8;
            }
            wr_atomic.store(w.wrapping_add(1), Ordering::Release);
            UART_TX_LOCK.store(false, Ordering::Release);
        }
        return Some(0);
    }
    // All other writes (IE, TXCTRL, RXCTRL, IP, DIV) → Python for PLIC updates.
    diag::log_line(&format!(
        "[diag-uart] wr pa={:#x} off={:#x} val={:#x} hid={} -> fallback to Python",
        pa, offset, write_data, hid,
    ));
    return None;
}

/// UART TX push 串行锁 — 见 try_handle_uart_concurrent 中的发布顺序说明。
pub(crate) static UART_TX_LOCK: AtomicBool = AtomicBool::new(false);
