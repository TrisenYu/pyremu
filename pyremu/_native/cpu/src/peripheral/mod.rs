use crate::handlers::DevCtx;

pub mod uart;

pub(crate) fn is_device_addr(pa: u64, dev: &DevCtx) -> bool {
    for i in 0..dev.num as usize {
        let base = unsafe { *dev.bases.add(i) };
        let end = unsafe { *dev.ends.add(i) };
        if pa >= base && pa < end {
            return true;
        }
    }
    false
}
