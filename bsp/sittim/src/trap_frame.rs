//! 陷态栈帧内存布局。合计 264 字节。

// ---------------------------------------------------------------
//  通用寄存器在陷态栈帧内的编号
// ---------------------------------------------------------------

#[allow(unused)]
pub const RA: usize = 1;
pub const SP: usize = 2;
pub const GP: usize = 3;
pub const TP: usize = 4;
pub const T0: usize = 5;
pub const T1: usize = 6;
pub const T2: usize = 7;
pub const S0: usize = 8;
pub const S1: usize = 9;
pub const A0: usize = 10;
pub const A1: usize = 11;
pub const A2: usize = 12;
pub const A3: usize = 13;
pub const A4: usize = 14;
pub const A5: usize = 15;
pub const A6: usize = 16;
pub const A7: usize = 17;
pub const S2: usize = 18;
pub const S3: usize = 19;
pub const S4: usize = 20;
pub const S5: usize = 21;
pub const S6: usize = 22;
pub const S7: usize = 23;
pub const S8: usize = 24;
pub const S9: usize = 25;
pub const S10: usize = 26;
pub const S11: usize = 27;
pub const T3: usize = 28;
pub const T4: usize = 29;
pub const T5: usize = 30;
pub const T6: usize = 31;

// ---------------------------------------------------------------
//  TrapGprs
// ---------------------------------------------------------------

/// 汇编 SAVE_CONTEXT 后的寄存器快照
#[repr(C)]
pub struct TrapGprs {
	/// x0..x31
	/// x0 由 SAVE_CONTEXT 写入 0
	pub regs: [u64; 32],
	/// 陷入前的 sp：来自 U 模式时为用户 sp，来自 S 模式时为该模式的 sp。
	pub sp_before_trap: u64,
}

#[allow(unused)]
impl TrapGprs {
	#[inline]
	pub fn zero(&self) -> u64 {
		self.regs[0]
	}
	#[inline]
	pub fn ra(&self) -> u64 {
		self.regs[RA]
	}
	#[inline]
	pub fn gp(&self) -> u64 {
		self.regs[GP]
	}
	#[inline]
	pub fn tp(&self) -> u64 {
		self.regs[TP]
	}
	#[inline]
	pub fn t0(&self) -> u64 {
		self.regs[T0]
	}
	#[inline]
	pub fn t1(&self) -> u64 {
		self.regs[T1]
	}
	#[inline]
	pub fn t2(&self) -> u64 {
		self.regs[T2]
	}
	#[inline]
	pub fn s0(&self) -> u64 {
		self.regs[S0]
	}
	#[inline]
	pub fn s1(&self) -> u64 {
		self.regs[S1]
	}
	#[inline]
	pub fn a0(&self) -> u64 {
		self.regs[A0]
	}
	#[inline]
	pub fn a1(&self) -> u64 {
		self.regs[A1]
	}
	#[inline]
	pub fn a2(&self) -> u64 {
		self.regs[A2]
	}
	#[inline]
	pub fn a3(&self) -> u64 {
		self.regs[A3]
	}
	#[inline]
	pub fn a4(&self) -> u64 {
		self.regs[A4]
	}
	#[inline]
	pub fn a5(&self) -> u64 {
		self.regs[A5]
	}
	#[inline]
	pub fn a6(&self) -> u64 {
		self.regs[A6]
	}
	#[inline]
	pub fn a7(&self) -> u64 {
		self.regs[A7]
	}
	#[inline]
	pub fn s2(&self) -> u64 {
		self.regs[S2]
	}
	#[inline]
	pub fn s3(&self) -> u64 {
		self.regs[S3]
	}
	#[inline]
	pub fn s4(&self) -> u64 {
		self.regs[S4]
	}
	#[inline]
	pub fn s5(&self) -> u64 {
		self.regs[S5]
	}
	#[inline]
	pub fn s6(&self) -> u64 {
		self.regs[S6]
	}
	#[inline]
	pub fn s7(&self) -> u64 {
		self.regs[S7]
	}
	#[inline]
	pub fn s8(&self) -> u64 {
		self.regs[S8]
	}
	#[inline]
	pub fn s9(&self) -> u64 {
		self.regs[S9]
	}
	#[inline]
	pub fn s10(&self) -> u64 {
		self.regs[S10]
	}
	#[inline]
	pub fn s11(&self) -> u64 {
		self.regs[S11]
	}
	#[inline]
	pub fn t3(&self) -> u64 {
		self.regs[T3]
	}
	#[inline]
	pub fn t4(&self) -> u64 {
		self.regs[T4]
	}
	#[inline]
	pub fn t5(&self) -> u64 {
		self.regs[T5]
	}
	#[inline]
	pub fn t6(&self) -> u64 {
		self.regs[T6]
	}

	// syscall 返回时写回的 setter
	#[inline]
	pub fn set_a0(&mut self, v: u64) {
		self.regs[A0] = v;
	}
	#[inline]
	pub fn set_a7(&mut self, v: u64) {
		self.regs[A7] = v;
	}
}
