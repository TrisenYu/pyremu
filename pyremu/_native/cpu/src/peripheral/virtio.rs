//! virtio-blk inline MMIO handler — avoids acceleration exits for register access.
//!
//! Device features, queue descriptors, and config space reads are handled
//! directly in Rust.  ``QueueNotify`` (0x050) deliberately returns ``None``
//! so the caller forces an MMIO exit; Python then re-executes the store
//! instruction and ``_mmio_write`` calls ``_process_queue()`` immediately.
//! This matches the PLIC claim/complete pattern where device MMIO naturally
//! provides exit points for I/O processing.

use super::DevCtx;

/// Device features advertised by the virtio-blk device (64-bit).
///
/// VIRTIO_F_RING_EVENT_IDX (bit 29) and VIRTIO_F_RING_INDIRECT_DESC (bit 28)
/// are deliberately NOT advertised: the Python-side virtqueue processor
/// (_process_descriptor_chain) does not implement avail_event writes or
/// indirect-descriptor-table traversal.  Advertising either feature causes
/// the guest driver to take code paths that break on our device.
/// Without them the driver falls back to flags-based notification
/// (VRING_USED_F_NO_NOTIFY) and direct descriptor chains — both of which
/// work correctly.
const VIRTIO_DEVICE_FEATURES: u64 = 1u64 << 32; // VIRTIO_F_VERSION_1

/// Try to handle a virtio-blk MMIO access inline inside the speedup execution engine.
///
/// The virtio base address and mutable state pointer are embedded in ``DevCtx``
/// so that no additional parameters need to be threaded through the handler
/// call chain.  Returns:
/// - ``Some(data)`` for a successful read (caller writes *data* to the
///   destination register).
/// - ``Some(0)`` for a successful write.
/// - ``None`` if *pa* is not within the virtio MMIO range, or if the access
///   must force an MMIO exit (``QueueNotify`` — Python re-executes the
///   store instruction and ``_mmio_write`` calls ``_process_queue()``).
pub fn try_handle_virtio(
	pa: u64,
	is_write: bool,
	write_data: u64,
	_size: u8,
	dev: &DevCtx,
) -> Option<u64> {
	if dev.virtio_base == 0 {
		return None;
	}
	let offset = pa.wrapping_sub(dev.virtio_base);
	if offset >= 0x200 {
		return None;
	}
	if is_write {
		virtio_write(offset, write_data, dev.virtio_raw)
	} else {
		virtio_read(offset, dev.virtio_raw)
	}
}

/// Handle virtio-blk MMIO writes inline.  See ``try_handle_virtio`` for the
/// return-value contract.
fn virtio_write(offset: u64, write_data: u64, raw: *mut crate::state::FfiVirtIoCtx) -> Option<u64> {
	match offset {
		// DeviceFeaturesSel (0x014) — page selector
		0x014 => unsafe {
			(*raw).device_features_sel = write_data as u32;
			Some(0)
		},
		// DriverFeatures (0x020) — guest features for selected page
		0x020 => unsafe {
			let sel = (*raw).driver_features_sel;
			let mask = (write_data as u64 & 0xFFFF_FFFF) << (sel * 32);
			(*raw).driver_features =
				((*raw).driver_features & !(0xFFFF_FFFFu64 << (sel * 32))) | mask;
			Some(0)
		},
		// DriverFeaturesSel (0x024) — page selector
		0x024 => unsafe {
			(*raw).driver_features_sel = write_data as u32;
			Some(0)
		},
		// QueueSel (0x030)
		0x030 => unsafe {
			(*raw).queue_sel = write_data as u32;
			Some(0)
		},
		// QueueNum (0x038) — capped at queue_num_max
		0x038 => unsafe {
			(*raw).queue_num = core::cmp::min(write_data as u32, (*raw).queue_num_max);
			Some(0)
		},
		// QueueReady (0x044)
		0x044 => unsafe {
			(*raw).queue_ready = if write_data != 0 { 1 } else { 0 };
			Some(0)
		},
		// QueueNotify (0x050) — set notify_pending AND force MMIO exit
		// (return None).  Without this, AIA-mode harts (where interrupt
		// handling is CSR-based and inline) would defer queue processing
		// to the next round, adding multi-second latency.
		// notify_pending triggers _native_unmarshal_virtio → _process_queue(),
		// and the MMIO exit triggers Python re-execution of the store
		// → _mmio_write → _process_queue() as a redundant but safe fallback.
		0x050 => unsafe {
			(*raw).notify_pending = 1;
			None
		},
		// InterruptStatus (0x060) — writing sets bits (unusual, but handle inline)
		0x060 => unsafe {
			(*raw).interrupt_status |= write_data as u32;
			Some(0)
		},
		// InterruptACK (0x064) — clear bits; flag PLIC lowering if all zero
		0x064 => unsafe {
			let old = (*raw).interrupt_status;
			(*raw).interrupt_status = old & !(write_data as u32);
			if (*raw).interrupt_status == 0 && old != 0 {
				(*raw).irq_maybe_lower = 1;
			}
			Some(0)
		},
		// Status (0x070) — writing 0 resets the device
		0x070 => {
			if write_data != 0 {
				unsafe {
					(*raw).status = write_data as u32;
				}
				return Some(0);
			}
			unsafe {
				(*raw).status = 0;
				(*raw).device_features_sel = 0;
				(*raw).driver_features_sel = 0;
				(*raw).driver_features = 0;
				(*raw).queue_sel = 0;
				(*raw).queue_ready = 0;
				(*raw).interrupt_status = 0;
				(*raw).queue_desc = 0;
				(*raw).queue_driver = 0;
				(*raw).queue_device = 0;
			}
			Some(0)
		}
		// QueueDescLow / High (0x080 / 0x084)
		0x080 => unsafe {
			(*raw).queue_desc =
				((*raw).queue_desc & 0xFFFF_FFFF_0000_0000) | (write_data as u64 & 0xFFFF_FFFF);
			Some(0)
		},
		0x084 => unsafe {
			(*raw).queue_desc =
				((*raw).queue_desc & 0xFFFF_FFFF) | ((write_data as u64 & 0xFFFF_FFFF) << 32);
			Some(0)
		},
		// QueueDriverLow / High (0x090 / 0x094)
		0x090 => unsafe {
			(*raw).queue_driver =
				((*raw).queue_driver & 0xFFFF_FFFF_0000_0000) | (write_data as u64 & 0xFFFF_FFFF);
			Some(0)
		},
		0x094 => unsafe {
			(*raw).queue_driver =
				((*raw).queue_driver & 0xFFFF_FFFF) | ((write_data as u64 & 0xFFFF_FFFF) << 32);
			Some(0)
		},
		// QueueDeviceLow / High (0x0A0 / 0x0A4)
		0x0A0 => unsafe {
			(*raw).queue_device =
				((*raw).queue_device & 0xFFFF_FFFF_0000_0000) | (write_data as u64 & 0xFFFF_FFFF);
			Some(0)
		},
		0x0A4 => unsafe {
			(*raw).queue_device =
				((*raw).queue_device & 0xFFFF_FFFF) | ((write_data as u64 & 0xFFFF_FFFF) << 32);
			Some(0)
		},
		// Config space (0x100+) — read-only in hardware; writes are ignored.
		// Other undefined offsets — ignored (writes have no effect).
		_ => Some(0),
	}
}

/// Handle virtio-blk MMIO reads inline.  See ``try_handle_virtio`` for the
/// return-value contract.
fn virtio_read(offset: u64, raw: *mut crate::state::FfiVirtIoCtx) -> Option<u64> {
	match offset {
		// MagicValue (0x000)
		0x000 => Some(0x74726976),
		// Version (0x004)
		0x004 => Some(0x2),
		// DeviceID (0x008) — 2 = block device
		0x008 => Some(0x2),
		// VendorID (0x00C)
		0x00C => Some(0x0),
		// DeviceFeatures (0x010) — page-selected
		0x010 => {
			let sel = unsafe { (*raw).device_features_sel };
			Some((VIRTIO_DEVICE_FEATURES >> (sel * 32)) & 0xFFFF_FFFF)
		}
		// QueueNumMax (0x034)
		0x034 => unsafe { Some((*raw).queue_num_max as u64) },
		// InterruptStatus (0x060)
		0x060 => unsafe { Some((*raw).interrupt_status as u64) },
		// Status (0x070)
		0x070 => unsafe { Some((*raw).status as u64) },
		// ConfigGeneration (0x0FC)
		0x0FC => Some(0),
		// Config space (0x100+)
		off if off >= 0x100 => {
			let local = off - 0x100;
			match local {
				// Capacity (u64, low 32 at 0x000, high 32 at 0x004)
				0x000 => unsafe { Some((*raw).capacity & 0xFFFF_FFFF) },
				0x004 => unsafe { Some(((*raw).capacity >> 32) & 0xFFFF_FFFF) },
				_ => Some(0),
			}
		}
		_ => Some(0),
	}
}
