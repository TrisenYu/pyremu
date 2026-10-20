//! 模块映像的描述块。
//!
//! 映像自偏移 0 起声明自己的布局: 入口在映像内的偏移、映像本体的字节数、可写静态数据的
//! 起始偏移与字节数。取入方在映像取得可执行权限之前逐项核对, 不符则拒绝载入以及后续的执行
//!
//! 描述块由模块的链接脚本以段内数据命令生成, 见 `modules/module.ld`; 各字段是同一份链接
//! 内各段地址的差值, 故不产生重定位项, 映像在模块窗口内的任何地址都给出相同取值。

use crate::constants::{ PAGE_SIZE, SIG_LEN };
use crate::mem_prim::align::align_up;

/// 描述块的字节数, 也是模块入口偏移的下界。
pub const IMAGE_DESC_LEN: usize = 64;

/// 描述块的魔数, 取值为 ASCII 的 "SMD1" 按小端读出的结果。
pub const IMAGE_DESC_MAGIC: u32 = u32::from_le_bytes(*b"SMD1");

/// 描述块当前的布局版本。模块与本侧的取值不一致即拒绝。
pub const IMAGE_DESC_ABI_VERSION: u32 = 1;

/// 映像描述块。位于映像偏移 0。
///
/// 字段顺序与字节数由 `modules/module.ld` 的数据命令逐项给出, 两侧必须一致。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuleImageDesc {
	/// 魔数, 取 [`IMAGE_DESC_MAGIC`]。
	pub magic: u32,
	/// 布局版本, 取 [`IMAGE_DESC_ABI_VERSION`]。
	pub abi_version: u32,
	/// 入口在映像内的偏移, 按 4 字节对齐。
	///
	/// 取段地址的差值而非绝对地址: 映像的加载基址在链接时未知。
	pub entry: u32,
	/// 映像本体的字节数, 不含映像末尾的签名。
	pub body_len: u32,
	/// 可写静态数据的起始偏移, 等于映像本体按页取整的结果。
	///
	/// 取入方自映像偏移 0 至该偏移按指令类映射为可读可执行, 自该偏移起按数据类映射为
	/// 可读写, 故该取值即取入方建映射时所用的可执行部分长度。
	pub data_class_off: u32,
	/// 可写静态数据的字节数, 即映像中初值全为 0 的部分。
	pub bss_size: u32,
	/// 保留, 恒为 0。
	pub flags: u32,
	/// 保留, 恒为 0。
	pub reserved: [u32; 9],
}

/// 描述块不被接受的原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescError {
	/// 描述块短于 [`IMAGE_DESC_LEN`], 或魔数不符。
	BadMagic,
	/// 描述块声明的布局版本与本侧的不一致。
	BadAbiVersion,
	/// 描述块声明的映像本体字节数与宿主交付的字节数不一致。
	///
	/// 含义是交付的映像陈旧、被截断, 或根本不是这个模块的映像。
	BodyLenMismatch,
	/// 可写静态数据的起始偏移与映像本体按页取整的结果不一致。
	DataClassOffMismatch,
	/// 入口偏移未按 4 字节对齐。
	EntryMisaligned,
	/// 入口偏移未落在映像的可执行部分之内。
	EntryOutOfRange,
	/// 可写静态数据的末尾越过模块槽位的上界。
	StaticDataTooLarge,
	/// 保留字段非 0: 该映像按另一版布局构建。
	ReservedNotZero,
}

/// 解析映像描述块并逐项核对它与取入方的约定。
///
/// *bytes* 是宿主交付的映像, 自映像偏移 0 起, 字节数为宿主报告的交付字节数, 即映像本体
/// 加上末尾的签名字节。*slot_size* 是模块槽位的字节数, 描述块所声明的静态数据不得越出
/// 该范围。
///
/// 各字段按小端读出。核对通过时返回的 `data_class_off` 等于取入方按交付字节数算出的可执
/// 行部分长度: 映像本体字节数与交付字节数的等式、`data_class_off` 与映像本体按页取整的
/// 等式, 两项合起来给出这一结论。
pub fn parse_image_desc(
	bytes: &[u8],
	slot_size: u64
) -> Result<ModuleImageDesc, DescError> {
	if bytes.len() < IMAGE_DESC_LEN {
		return Err(DescError::BadMagic);
	}
	let field = |index: usize| -> u32 {
		let at = index * 4;
		u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
	};
	let desc = ModuleImageDesc {
		magic: field(0),
		abi_version: field(1),
		entry: field(2),
		body_len: field(3),
		data_class_off: field(4),
		bss_size: field(5),
		flags: field(6),
		reserved: [
			field(7),
			field(8),
			field(9),
			field(10),
			field(11),
			field(12),
			field(13),
			field(14),
			field(15),
		],
	};

	if desc.magic != IMAGE_DESC_MAGIC {
		return Err(DescError::BadMagic);
	}
	if desc.abi_version != IMAGE_DESC_ABI_VERSION {
		return Err(DescError::BadAbiVersion);
	}
	if (desc.body_len as usize) + SIG_LEN != bytes.len() {
		return Err(DescError::BodyLenMismatch);
	}
	let data_class_off = align_up(desc.body_len as u64, PAGE_SIZE);
	if (desc.data_class_off as u64) != data_class_off {
		return Err(DescError::DataClassOffMismatch);
	}
	if desc.entry % 4 != 0 {
		return Err(DescError::EntryMisaligned);
	}
	if
		(desc.entry as u64) < (IMAGE_DESC_LEN as u64) ||
		(desc.entry as u64) >= data_class_off
	{
		return Err(DescError::EntryOutOfRange);
	}
	if data_class_off + (desc.bss_size as u64) > slot_size {
		return Err(DescError::StaticDataTooLarge);
	}
	if desc.flags != 0 || desc.reserved.iter().any(|value| *value != 0) {
		return Err(DescError::ReservedNotZero);
	}
	Ok(desc)
}



#[cfg(test)]
mod tests {
	use super::*;
	use crate::constants::CHUNK_2M_SIZE;

	/// 一份可接受的映像: 描述块之后是映像本体, 末尾一个签名。
	const BODY_LEN: u32 = 0x80;
	const IMAGE_LEN: usize = (BODY_LEN as usize) + SIG_LEN;
	const ENTRY: u32 = IMAGE_DESC_LEN as u32;

	/// 按给定的字段取值组装一份映像。描述块以外的字节为 0, 不影响核对。
	fn image_with(
		entry: u32,
		body_len: u32,
		data_class_off: u32,
		bss_size: u32
	) -> [u8; IMAGE_LEN] {
		let mut bytes = [0u8; IMAGE_LEN];
		let mut put = |index: usize, value: u32| {
			let at = index * 4;
			bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
		};
		put(0, IMAGE_DESC_MAGIC);
		put(1, IMAGE_DESC_ABI_VERSION);
		put(2, entry);
		put(3, body_len);
		put(4, data_class_off);
		put(5, bss_size);
		bytes
	}

	/// 合于约定的映像被接受, 且各字段按小端读出。
	#[test]
	fn test_parse_accepts_a_well_formed_image() {
		let bytes = image_with(ENTRY, BODY_LEN, PAGE_SIZE as u32, 0x40);
		let desc = parse_image_desc(&bytes, CHUNK_2M_SIZE).expect("well formed image");
		assert_eq!(desc.magic, IMAGE_DESC_MAGIC);
		assert_eq!(desc.abi_version, IMAGE_DESC_ABI_VERSION);
		assert_eq!(desc.entry, ENTRY);
		assert_eq!(desc.body_len, BODY_LEN);
		assert_eq!(desc.data_class_off, PAGE_SIZE as u32);
		assert_eq!(desc.bss_size, 0x40);
	}

	/// 描述块的尺寸与字段偏移由链接脚本的字段顺序固定。
	#[test]
	fn test_image_desc_layout_is_fixed() {
		assert_eq!(core::mem::size_of::<ModuleImageDesc>(), IMAGE_DESC_LEN);
		assert_eq!(core::mem::align_of::<ModuleImageDesc>(), 4);
		assert_eq!(core::mem::offset_of!(ModuleImageDesc, magic), 0);
		assert_eq!(core::mem::offset_of!(ModuleImageDesc, abi_version), 4);
		assert_eq!(core::mem::offset_of!(ModuleImageDesc, entry), 8);
		assert_eq!(core::mem::offset_of!(ModuleImageDesc, body_len), 12);
		assert_eq!(core::mem::offset_of!(ModuleImageDesc, data_class_off), 16);
		assert_eq!(core::mem::offset_of!(ModuleImageDesc, bss_size), 20);
		assert_eq!(core::mem::offset_of!(ModuleImageDesc, flags), 24);
		assert_eq!(core::mem::offset_of!(ModuleImageDesc, reserved), 28);
	}

	/// 魔数不符与描述块过短都按不是模块映像拒绝。
	#[test]
	fn test_parse_rejects_a_foreign_magic() {
		let mut bytes = image_with(ENTRY, BODY_LEN, PAGE_SIZE as u32, 0);
		bytes[0] ^= 0xff;
		assert_eq!(parse_image_desc(&bytes, CHUNK_2M_SIZE), Err(DescError::BadMagic));

		let short = [0u8; IMAGE_DESC_LEN - 4];
		assert_eq!(parse_image_desc(&short, CHUNK_2M_SIZE), Err(DescError::BadMagic));
	}

	/// 布局版本不同即拒绝: 该映像按另一版约定构建。
	#[test]
	fn test_parse_rejects_another_abi_version() {
		let mut bytes = image_with(ENTRY, BODY_LEN, PAGE_SIZE as u32, 0);
		bytes[4..8].copy_from_slice(&(IMAGE_DESC_ABI_VERSION + 1).to_le_bytes());
		assert_eq!(
			parse_image_desc(&bytes, CHUNK_2M_SIZE),
			Err(DescError::BadAbiVersion)
		);
	}

	/// 映像本体字节数与交付字节数不符即拒绝, 分歧覆盖陈旧与截断两个方向。
	#[test]
	fn test_parse_rejects_a_body_len_at_odds_with_the_delivered_bytes() {
		// 声明多于交付: 映像被截断。
		let bytes = image_with(ENTRY, BODY_LEN + 4, PAGE_SIZE as u32, 0);
		assert_eq!(
			parse_image_desc(&bytes, CHUNK_2M_SIZE),
			Err(DescError::BodyLenMismatch)
		);

		// 声明少于交付: 多出的字节既不在映像本体内, 也不在签名内。
		let bytes = image_with(ENTRY, BODY_LEN - 4, PAGE_SIZE as u32, 0);
		assert_eq!(
			parse_image_desc(&bytes, CHUNK_2M_SIZE),
			Err(DescError::BodyLenMismatch)
		);
	}

	/// 可写静态数据的起始偏移须等于映像本体按页取整的结果, 差一个页即拒绝。
	#[test]
	fn test_parse_rejects_a_data_class_offset_off_the_page_boundary() {
		let bytes = image_with(ENTRY, BODY_LEN, (PAGE_SIZE as u32) * 2, 0);
		assert_eq!(
			parse_image_desc(&bytes, CHUNK_2M_SIZE),
			Err(DescError::DataClassOffMismatch)
		);
	}

	/// 入口须按 4 字节对齐, 且落在可执行部分之内: 描述块自身之内、可执行部分末尾及其上都
	/// 不接受。
	#[test]
	fn test_parse_rejects_an_entry_outside_the_executable_part() {
		// 未按 4 字节对齐。
		let bytes = image_with(ENTRY + 2, BODY_LEN, PAGE_SIZE as u32, 0);
		assert_eq!(
			parse_image_desc(&bytes, CHUNK_2M_SIZE),
			Err(DescError::EntryMisaligned)
		);

		// 落在描述块之内。
		let bytes = image_with(0, BODY_LEN, PAGE_SIZE as u32, 0);
		assert_eq!(
			parse_image_desc(&bytes, CHUNK_2M_SIZE),
			Err(DescError::EntryOutOfRange)
		);

		// 落在可执行部分的末尾之上。
		let bytes = image_with(PAGE_SIZE as u32, BODY_LEN, PAGE_SIZE as u32, 0);
		assert_eq!(
			parse_image_desc(&bytes, CHUNK_2M_SIZE),
			Err(DescError::EntryOutOfRange)
		);
	}

	/// 静态数据越出模块槽位即拒绝, 边界取值取越出与正好放下两种。
	#[test]
	fn test_parse_rejects_static_data_beyond_the_slot() {
		let bytes = image_with(ENTRY, BODY_LEN, PAGE_SIZE as u32, CHUNK_2M_SIZE as u32);
		assert_eq!(
			parse_image_desc(&bytes, CHUNK_2M_SIZE),
			Err(DescError::StaticDataTooLarge)
		);

		let bytes = image_with(
			ENTRY,
			BODY_LEN,
			PAGE_SIZE as u32,
			(CHUNK_2M_SIZE - PAGE_SIZE) as u32
		);
		assert!(parse_image_desc(&bytes, CHUNK_2M_SIZE).is_ok());
	}

	/// 保留位非 0 即拒绝: 该映像按另一版布局构建, 保留位已被占用。
	#[test]
	fn test_parse_rejects_a_non_zero_reserved_field() {
		let mut bytes = image_with(ENTRY, BODY_LEN, PAGE_SIZE as u32, 0);
		bytes[24..28].copy_from_slice(&(1u32).to_le_bytes());
		assert_eq!(
			parse_image_desc(&bytes, CHUNK_2M_SIZE),
			Err(DescError::ReservedNotZero)
		);

		let mut bytes = image_with(ENTRY, BODY_LEN, PAGE_SIZE as u32, 0);
		bytes[60..64].copy_from_slice(&(1u32).to_le_bytes());
		assert_eq!(
			parse_image_desc(&bytes, CHUNK_2M_SIZE),
			Err(DescError::ReservedNotZero)
		);
	}
}
