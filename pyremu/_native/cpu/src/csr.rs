//! 对 CSR 寄存器组的读取与写入.
//!
//! 在 ``HartState`` 中存在定义的 CSR 可直接由本模块处理. 而未知的 CSR 与写入
//! 需退出到 Python, 交由 ``registers.py`` 处理.

use std::sync::atomic::Ordering;

use crate::concurrent::{ConcurrentClintCtx, NS_PER_INSTR};
use crate::ffi::InstrToBeExec;
use crate::handlers::PmpCtx;
use crate::interrupt::clint::write_stimecmp;
use crate::interrupt::{
	compute_mtopi, compute_stopi, imsic_reg_read, imsic_reg_write, imsic_topei_claim_iid,
	imsic_topei_peek, sync_imsic_one, IID_M_IPI, IID_S_IPI,
};
use crate::state::{exit_reason, riscv_mode, HartState, CFG_AIA, CPU_FREQ_HZ};
use crate::trap::deliver_illegal_instruction;

// ============================================================
//  CsrContext: csr_read 与 csr_write 的合并参数
// ============================================================

/// 执行期间读取或写入一个 CSR 所需的全部状态.
///
/// ``clint`` 提供实时的 ``mtime`` 而非固定快照, 并给出 ``mtimecmp`` 供
/// MTIP 得到响应后推进.
/// ``extern_hart_id`` 是调用方给出的索引, 从 0 开始, 与 ``state`` 中 CSR
/// ``mhartid`` 的值可以不同.
pub struct CsrContext<'a> {
	pub state: &'a mut HartState,
	pub extern_hart_id: u32,
	pub pmp: &'a PmpCtx,
	pub clint: &'a ConcurrentClintCtx,
}

impl<'a> CsrContext<'a> {
	#[inline]
	pub fn new(
		state: &'a mut HartState,
		extern_hart_id: u32,
		pmp: &'a PmpCtx,
		clint: &'a ConcurrentClintCtx,
	) -> Self {
		CsrContext {
			state,
			extern_hart_id,
			pmp,
			clint,
		}
	}

	/// 从共享 CLINT 计数器读取实时 ``mtime``, 按原子加载.
	#[inline]
	pub fn mtime(&self) -> u64 {
		unsafe { &*self.clint.mtime }.load(Ordering::Relaxed)
	}

	/// 把 ``mtimecmp[extern_hart_id]`` 推进到 ``mtime + delta``,
	/// 使定时器中断在得到响应后不会立即再次触发.
	#[inline]
	pub fn bump_mtimecmp(&self, delta: u64) {
		let t = self.mtime();
		let target = t.wrapping_add(delta);
		unsafe {
			(&*self.clint.mtimecmp.add(self.extern_hart_id as usize))
				.store(target, Ordering::Release);
		}
	}
}

// ============================================================
//  CSR 地址常量
// ============================================================

// 机器级
pub const MSTATUS: u16 = 0x300;
pub const MISA: u16 = 0x301;
pub const MEDELEG: u16 = 0x302;
pub const MIDELEG: u16 = 0x303;
pub const MIE: u16 = 0x304;
pub const MTVEC: u16 = 0x305;
pub const MCOUNTEREN: u16 = 0x306;
pub const MSCRATCH: u16 = 0x340;
pub const MEPC: u16 = 0x341;
pub const MCAUSE: u16 = 0x342;
pub const MTVAL: u16 = 0x343;
pub const MIP: u16 = 0x344;

// 机器信息
pub const MVENDORID: u16 = 0xF11;
pub const MARCHID: u16 = 0xF12;
pub const MIMPID: u16 = 0xF13;
pub const MHARTID: u16 = 0xF14;

// S 模式
pub const SSTATUS: u16 = 0x100;
pub const SIE: u16 = 0x104;
pub const STVEC: u16 = 0x105;
pub const SCOUNTEREN: u16 = 0x106;
pub const SSCRATCH: u16 = 0x140;
pub const SEPC: u16 = 0x141;
pub const SCAUSE: u16 = 0x142;
pub const STVAL: u16 = 0x143;
pub const SIP: u16 = 0x144;
pub const SATP: u16 = 0x180; // 地址翻译与保护
pub const STIMECMP: u16 = 0x14D;

// AIA IMSIC CSR
pub const MISELECT: u16 = 0x350;
pub const MIREG: u16 = 0x351;
pub const SISELECT: u16 = 0x150;
pub const SIREG: u16 = 0x151;
pub const MTOPEI: u16 = 0x35C;
pub const STOPEI: u16 = 0x15C;

// 用户级浮点 CSR, 由 F/D 扩展提供
pub const FFLAGS: u16 = 0x001;
pub const FRM: u16 = 0x002;
pub const FCSR: u16 = 0x003;

// 机器级计数器
pub const MCYCLE: u16 = 0xB00;
pub const MINSTRET: u16 = 0xB02;
pub const MCYCLEH: u16 = 0xB80;
pub const MINSTRETH: u16 = 0xB82;

// 用户级计数器
pub const CYCLE: u16 = 0xC00;
pub const TIME: u16 = 0xC01;
pub const INSTRET: u16 = 0xC02;
pub const CYCLEH: u16 = 0xC80;
pub const TIMEH: u16 = 0xC81;
pub const INSTRETH: u16 = 0xC82;

/// 每条已执行完成的指令对应的时钟周期数.
///
/// 时钟周期数由指令计数与 NS_PER_INSTR, CPU_FREQ_HZ 两个生成常量换算:
///
///     cycles = instrs * NS_PER_INSTR * CPU_FREQ_HZ / 1e9
pub const CYCLES_PER_INSTR: u64 = NS_PER_INSTR * CPU_FREQ_HZ / 1_000_000_000;

/// 本 hart 的 `mcycle` 值, 由该 hart 已执行完成的指令数换算得出.
///
/// 计数按 hart 独立推进, 故同一个 hart 上两次采样之差只包含该 hart 自身
/// 执行的指令, 与其它 hart 是否运行无关.
#[inline]
pub fn cycles_of(state: &HartState) -> u64 {
	state.total_instrs.wrapping_mul(CYCLES_PER_INSTR)
}

// ============================================================
//  mstatus 与 sstatus 的位域掩码
// ============================================================

const MSTATUS_SIE: u64 = 1 << 1; // S 模式中断使能
const MSTATUS_SPIE: u64 = 1 << 5; // S 模式中断的先前使能值
const MSTATUS_UBE: u64 = 1 << 6; // 用户级大端模式
const MSTATUS_SPP: u64 = 1 << 8; // 陷入 S 模式前的特权级
const MSTATUS_VS: u64 = 0b11 << 9; // 向量状态, 由 H 扩展使用
const MSTATUS_FS: u64 = 0b11 << 13; // 浮点单元状态
const MSTATUS_XS: u64 = 0b11 << 15; // 用户扩展状态
const MSTATUS_MPRV: u64 = 1 << 17; // 修改访存特权级
const MSTATUS_SUM: u64 = 1 << 18; // 允许 S 模式访问用户级内存
const MSTATUS_MXR: u64 = 1 << 19; // 把可执行页读作可读
const MSTATUS_SD: u64 = 1 << 63; // 状态为脏, 只读

// sstatus 是 mstatus 的受限视图, 见 RISC-V 特权规范 4.1.1 节.
// 读取 sstatus 只返回当前 S 模式程序可访问的那部分 mstatus 位.
const SSTATUS_READ_MASK: u64 = MSTATUS_SIE
	| MSTATUS_SPIE
	| MSTATUS_UBE
	| MSTATUS_SPP
	| MSTATUS_VS
	| MSTATUS_FS
	| MSTATUS_XS
	| MSTATUS_MPRV
	| MSTATUS_SUM
	| MSTATUS_MXR
	| MSTATUS_SD;

// 写入 sstatus 只更新可写的那部分位, VS 与 SD 为只读
const SSTATUS_WRITE_MASK: u64 = SSTATUS_READ_MASK & !(MSTATUS_VS | MSTATUS_SD);

// ============================================================
//  公开接口
// ============================================================

/// 返回状态: 0 为成功, 处理函数返回前进 4 字节; 1 为 IllInstr 陷态;
/// 2 为退出到 Python, 对未知 CSR 的读取或写入需要 Python 侧处理.
pub const CSR_OK: u8 = 0;
pub const CSR_ILL: u8 = 1;
pub const CSR_EXIT: u8 = 2;

/// 用户级计数器 CSR 在 counteren 寄存器中对应的位; 非此类 CSR 返回 ``None``.
///
/// 0xC00-0xC1F 为 cycle, time, instret 与 hpmcounter3-31,
/// 0xC80-0xC9F 为这些计数器的高位部分.
fn counter_enable_bit(addr: u16) -> Option<u64> {
	match addr {
		0xC00..=0xC1F => Some(1 << (addr - 0xC00)),
		0xC80..=0xC9F => Some(1 << (addr - 0xC80)),
		_ => None,
	}
}

/// 读取一个 CSR 的值, 返回 ``(value, status)``.
/// status 为 CSR_OK 时 value 有效, 调用方写入 rd.
/// status 为 CSR_ILL 时产生 IllInstr 陷态.
/// status 为 CSR_EXIT 时退出到 Python.
pub fn csr_read(ctx: &mut CsrContext, addr: u16) -> (u64, u8) {
	// 检查特权级
	let priv_req = (addr >> 8) as u8 & 0x3; // 位 [9:8]
	if !csr_priv_ok(ctx.state.mode, priv_req) {
		return (0, CSR_ILL);
	}

	// 计数器使能检查. 用户级计数器 CSR 0xC00-0xC1F 与 0xC80-0xC9F 的特权
	// 编码为 0, 上一步的特权检查不会拒绝这些 CSR, 可读性由 counteren 寄存器决定:
	//   S 模式: mcounteren 的对应位
	//   U 模式: mcounteren 与 scounteren 的对应位同时置位
	// 高半 CSR 与低半共用同一位.
	if let Some(bit) = counter_enable_bit(addr) {
		match ctx.state.mode {
			riscv_mode::S => {
				if ctx.state.mcounteren & bit == 0 {
					return (0, CSR_ILL);
				}
			}
			riscv_mode::U => {
				if ctx.state.mcounteren & bit == 0 || ctx.state.scounteren & bit == 0 {
					return (0, CSR_ILL);
				}
			}
			_ => {}
		}
	}

	match addr {
        // 存放在 HartState 中的机器级 CSR
        MSTATUS => (ctx.state.mstatus, CSR_OK),
        MEDELEG => (ctx.state.medeleg, CSR_OK),
        MIDELEG => (ctx.state.mideleg, CSR_OK),
        MIE     => (ctx.state.mie, CSR_OK),
        MTVEC   => (ctx.state.mtvec, CSR_OK),
        MSCRATCH => (ctx.state.mscratch, CSR_OK),
        MEPC    => (ctx.state.mepc, CSR_OK),
        MCAUSE  => (ctx.state.mcause, CSR_OK),
        MTVAL   => (ctx.state.mtval, CSR_OK),
        MIP     => (ctx.state.mip.load(Ordering::Acquire), CSR_OK),
        MCOUNTEREN => (ctx.state.mcounteren, CSR_OK),

        // 存放在 HartState 中的 S 模式 CSR
        SSTATUS => (ctx.state.mstatus & SSTATUS_READ_MASK, CSR_OK),
        SIE     => (ctx.state.mie & ctx.state.mideleg, CSR_OK),
        SIP     => (ctx.state.mip.load(Ordering::Acquire) & ctx.state.mideleg, CSR_OK),
        STVEC   => (ctx.state.stvec, CSR_OK),
        SSCRATCH => (ctx.state.sscratch, CSR_OK),
        SEPC    => (ctx.state.sepc, CSR_OK),
        SCAUSE  => (ctx.state.scause, CSR_OK),
        STVAL   => (ctx.state.stval, CSR_OK),
        SATP    => (ctx.state.satp, CSR_OK),
        SCOUNTEREN => (ctx.state.scounteren, CSR_OK),
        STIMECMP => (ctx.state.stimecmp, CSR_OK),

        // 浮点 CSR, 要求 mstatus.FS 不为 Off
        FFLAGS => if fp_off(ctx.state) { (0, CSR_ILL) } else { ((ctx.state.fcsr & 0x1F) as u64, CSR_OK) },
        FRM    => if fp_off(ctx.state) { (0, CSR_ILL) } else { (((ctx.state.fcsr >> 5) & 0x7) as u64, CSR_OK) },
        FCSR   => if fp_off(ctx.state) { (0, CSR_ILL) } else { ((ctx.state.fcsr & 0xFF) as u64, CSR_OK) },

        // 机器信息寄存器
        MVENDORID => (0, CSR_OK),
        MARCHID   => (0, CSR_OK),
        MIMPID    => (0, CSR_OK),
        MHARTID   => (ctx.state.mhartid as u64, CSR_OK),

        // MISA: 报告 RV64IMAC
        MISA => (0x800000000014112Du64, CSR_OK),

        // time, 取自 CLINT 的 mtime
        TIME => (ctx.mtime(), CSR_OK),
        // timeh: mtime 的 bits[63:32]
        TIMEH => (ctx.mtime() >> 32, CSR_OK),

        // 机器级与用户级计数器, 按本 hart 已执行完成的指令数换算.
        // 高半 CSR 返回同一计数的 bits[63:32].
        MCYCLE | CYCLE => (cycles_of(ctx.state), CSR_OK),
        MINSTRET | INSTRET => (ctx.state.total_instrs, CSR_OK),
        MCYCLEH | CYCLEH => ((cycles_of(ctx.state) >> 32) & 0xFFFF_FFFF, CSR_OK),
        MINSTRETH | INSTRETH => ((ctx.state.total_instrs >> 32) & 0xFFFF_FFFF, CSR_OK),

        0x5C0 => (ctx.state.mdid, CSR_OK),
        0x5C1 => (ctx.state.pmpsplit as u64, CSR_OK),

        // ---- AIA IMSIC CSR, 是否可用由 CFG_AIA 判定 ----
        MISELECT => {
            if CFG_AIA { (ctx.state.imsic_m.select as u64, CSR_OK) }
            else { (0, CSR_ILL) }
        }
        MIREG => {
            if CFG_AIA { imsic_reg_read(&ctx.state.imsic_m, ctx.state.imsic_m.select) }
            else { (0, CSR_ILL) }
        }
        SISELECT => {
            if CFG_AIA { (ctx.state.imsic_s.select as u64, CSR_OK) }
            else { (0, CSR_ILL) }
        }
        SIREG => {
            if CFG_AIA { imsic_reg_read(&ctx.state.imsic_s, ctx.state.imsic_s.select) }
            else { (0, CSR_ILL) }
        }
        MTOPEI => {
            if !CFG_AIA {
                return (0, CSR_ILL);
            }
            // mtopei 读取 IMSIC M 文件中优先级最高的外部中断并得到响应其 eip 位.
            // 按 AIA 规范, 该 CSR 只看到 IMSIC M 文件中的中断: 外部中断的
            // IID 不小于 6, 软件中断经 seteipnum 路由时 IID 为 1 或 3.
            // CLINT 的 MSIP 与 MTIP 不在此可见, 分发器读取 0xFB0 的 mtopi
            // 取得合并视图.
            let (raw, _) = imsic_topei_peek(&ctx.state.imsic_m);
            if raw != 0 {
                let iid = (raw >> 16) as u32;
                imsic_topei_claim_iid(&mut ctx.state.imsic_m, iid);
                sync_imsic_one(ctx.state, true);
            }
            return (raw, CSR_OK);
        }
        STOPEI => {
            if !CFG_AIA {
                return (0, CSR_ILL);
            }
            // stopei 是 mtopei 的 S 模式对应: 只涉及 IMSIC S 文件,
            // 包含外部中断与 IPI, 不含经 0xDB0 的 stopi 分发的
            // 传统 SSIP 与 STIP.
            let (raw, _) = imsic_topei_peek(&ctx.state.imsic_s);

            if raw != 0 {
                let iid = (raw >> 16) as u32;
                imsic_topei_claim_iid(&mut ctx.state.imsic_s, iid);
                sync_imsic_one(ctx.state, false);
            }
            return (raw, CSR_OK);
        }
        0xFB0 /* mtopi */ => {
            if CFG_AIA { compute_mtopi(ctx.state, ctx.mtime()) }
            else { (0, CSR_ILL) }
        }
        0xDB0 /* stopi */ => {
            if CFG_AIA { compute_stopi(ctx.state, ctx.mtime()) }
            else { (0, CSR_ILL) }
        }
        // 其余 AIA CSR, 如 mvien 与 mvip
        0x308 /* mvien */ | 0x309 /* mvip */
        | 0x318 /* mvienh */ | 0x319 /* mviph */ => (0, CSR_OK),

        // 调试与跟踪触发 CSR 0x7A0-0x7AF 未实现. 返回 0 而非 IllInstr:
        // OpenSBI 不把 IllInstr 陷态处理逻辑委派给 S 模式, S 模式的探测会陷入机器级,
        // 其处理函数调用 sbi_trap_error 后进入 WFI, 使内核停止推进.
        // 返回 0 让内核可以探测这些 CSR 而不陷态. 具备 Sdext 的硬件在
        // 机器级可访问它们; 不具备时视为未实现并读作 0.
        0x7A0..=0x7AF => (0, CSR_OK),

        // 仅 RV32 使用的 CSR, 在 RV64 上读取合法, 返回 0.
        0x310 /* mstatush */ | 0x312 /* medelegh */ | 0x313 /* midelegh */ => (0, CSR_OK),

        // ---- PMP CSR, 用于 RV64 ----
        // RV64 上编号为奇数的 pmpcfg 寄存器非法.
        a if (0x3A1..=0x3AF).contains(&a) && (a & 1) != 0 => (0, CSR_ILL),
        // 编号为偶数的 pmpcfg: 读取 8 个配置字节, 按小端序装入 u64.
        a if (0x3A0..=0x3AE).contains(&a) && (a & 1) == 0 => {
            let entry_base = ((a - 0x3A0) / 2) * 8;
            if (entry_base as u8) >= ctx.pmp.num {
                return (0, CSR_OK); // 对未实现的 PMP 寄存器操作，取值恒为 0
            }
            let mut val: u64 = 0;
            let base = entry_base as usize;
            let n = ((ctx.pmp.num as usize) - base).min(8);
            for i in 0..n {
                let b = unsafe { *ctx.pmp.cfg.add(base + i) } as u64;
                val |= b << (i * 8);
            }
            return (val, CSR_OK);
        }
        // PMP 地址寄存器 0x3B0-0x3EF: 每个条目一个, 各 8 字节.
        a if (0x3B0..=0x3EF).contains(&a) => {
            let n = (a - 0x3B0) as usize;
            if n < ctx.pmp.num as usize {
                (unsafe { *ctx.pmp.addr.add(n) }, CSR_OK)
            } else {
                (0, CSR_OK)
            }
        }

        // 其余情形则退出到 Python
        _ => (0, CSR_EXIT),
    }
}

/// 写入一个 CSR 的值, 返回 ``status``.
/// status 为 CSR_OK 时写入成功.
/// status 为 CSR_ILL 时产生 IllInstr 陷态.
/// status 为 CSR_EXIT 时退出到 Python 处理.
pub fn csr_write(ctx: &mut CsrContext, addr: u16, val: u64) -> u8 {
	let priv_req = (addr >> 8) as u8 & 0x3;
	if !csr_priv_ok(ctx.state.mode, priv_req) {
		return CSR_ILL;
	}

	match addr {
        MSTATUS => {
            // SD 只读
            let sd = ctx.state.mstatus & MSTATUS_SD;
            let wr = val & !MSTATUS_SD;
            ctx.state.mstatus = wr | sd;
            CSR_OK
        }
        MEDELEG => { ctx.state.medeleg = val; CSR_OK }
        MIDELEG => { ctx.state.mideleg = val; CSR_OK }
        MIE     => { ctx.state.mie = val; CSR_OK }
        MTVEC   => { ctx.state.mtvec = val; CSR_OK }
        MSCRATCH => { ctx.state.mscratch = val; CSR_OK }
        MEPC    => { ctx.state.mepc = val; CSR_OK }
        MCAUSE  => { ctx.state.mcause = val; CSR_OK }
        MTVAL   => { ctx.state.mtval = val; CSR_OK }
        MIP     => {
            // MSIP, STIP, MTIP, SEIP 与 MEIP 为只读位, 由 CLINT, SSTC
            // 定时器或外部中断控制器驱动; 写入时保留原值, 只允许写
            // 软件可写位 SSIP 与 USIP. STIP 与 SEIP 镜像硬件状态,
            // 受调试程序的 ``csrw mip`` 不得覆写它们.
            let ro_mask: u64 = (1 << 3) | (1 << 5) | (1 << 7) | (1 << 9) | (1 << 11);
            ctx.state.mip.store((ctx.state.mip.load(Ordering::Acquire) & ro_mask) | (val & !ro_mask), Ordering::Release);
            CSR_OK
        }
        MCOUNTEREN => { ctx.state.mcounteren = val; CSR_OK }

        SSTATUS => {
            let mask = SSTATUS_WRITE_MASK;
            ctx.state.mstatus = (ctx.state.mstatus & !mask) | (val & mask);
            CSR_OK
        }
        SIE => {
            // sie 是 mie 的受限视图, 只有 mideleg 委派给 S 模式的位可写
            let mask = ctx.state.mideleg;
            ctx.state.mie = (ctx.state.mie & !mask) | (val & mask);
            CSR_OK
        }
        SIP => {
            // sip 是 mip 的只读视图, 唯一可写位是 SSIP, 即位 1. 与 sie 不同,
            // sip 并非对所有 mideleg 委派位都可写: STIP 由 SSTC 定时器驱动,
            // SEIP 由 PLIC 驱动, 均不得被受调试程序的 ``csrw sip`` 覆写.
            // 飞地上下文切换写回陈旧 sip 时, 若覆写这些位会丢失挂起中断.
            let wr_mask: u64 = 1 << 1; // SSIP 是唯一软件可写位
            ctx.state.mip.store((ctx.state.mip.load(Ordering::Acquire) & !wr_mask) | (val & wr_mask), Ordering::Release);
            CSR_OK
        }
        STVEC   => { ctx.state.stvec = val; CSR_OK }
        SSCRATCH => { ctx.state.sscratch = val; CSR_OK }
        SEPC    => { ctx.state.sepc = val; CSR_OK }
        SCAUSE  => { ctx.state.scause = val; CSR_OK }
        STVAL   => { ctx.state.stval = val; CSR_OK }
        SCOUNTEREN => { ctx.state.scounteren = val; CSR_OK }
        SATP    => {
            // ASID 即位 [59:44] 固定为 0, 属于 WARL: 加速执行引擎的 TLB
            // 与 Python 侧的 TLB 都不用 ASID 标记条目. 写入时清除 ASID 位,
            // MODE 与 PPN 按写入值保存, 随后刷新两个 TLB.
            let val = val & !(0xFFFFu64 << 44);
            let new_mode = (val >> 60) as u8;
            ctx.state.satp = val;
            ctx.state.mmu_mode = new_mode;
            tlb_flush_inline(ctx.state);
            CSR_OK
        }

        // 机器信息寄存器只读, 写入被忽略
        MISA | MVENDORID | MARCHID | MIMPID | MHARTID     => CSR_OK,

        // time 与 timeh 分别为 0xC01 与 0xC81.
        // ``csrrs rd, time, x0`` 会把读到的值回写到该 CSR, 故写操作静默
        // 忽略并返回成功. 若返回 CSR_EXIT, 每条 rdtime 都会退出到 Python,
        // 引起内核性能下降并使得内核因意外的时钟输入而阻塞在对定时中断的响应上.
        TIME | TIMEH => CSR_OK,

        // 模拟器一侧，mcycle, minstret 及其时间 CSR 寄存器的计数由本 hart 的指令数换算
        // 将忽略任何写入.
        MCYCLE | MINSTRET | MCYCLEH | MINSTRETH => CSR_OK,

        // cycle 与 instret 为只读计数器
        CYCLE | INSTRET | CYCLEH | INSTRETH => CSR_OK,

        STIMECMP => {
            ctx.state.stimecmp = val;
            write_stimecmp(ctx.state, ctx.mtime(), val, ctx.clint.timebase_hz);
            CSR_OK
        }

        // 浮点 CSR: 写入把 mstatus.FS 置为 Dirty 并置位 SD.
        FFLAGS => {
            if fp_off(ctx.state) { return CSR_ILL; }
            ctx.state.fcsr = (ctx.state.fcsr & !0x1F) | (val as u32 & 0x1F);
            ctx.state.mstatus |= MSTATUS_FS | MSTATUS_SD;
            CSR_OK
        }
        FRM => {
            if fp_off(ctx.state) { return CSR_ILL; }
            ctx.state.fcsr = (ctx.state.fcsr & !0xE0) | ((val as u32 & 0x7) << 5);
            ctx.state.mstatus |= MSTATUS_FS | MSTATUS_SD;
            CSR_OK
        }
        FCSR => {
            if fp_off(ctx.state) { return CSR_ILL; }
            ctx.state.fcsr = val as u32 & 0xFF;
            ctx.state.mstatus |= MSTATUS_FS | MSTATUS_SD;
            CSR_OK
        }

        0x5C0 => { ctx.state.mdid = val; CSR_OK }
        0x5C1 => { ctx.state.pmpsplit = val as u8; CSR_OK }

        // 调试与跟踪触发 CSR 0x7A0-0x7AF 未实现, 写入被静默忽略.
        // 不具备 Sdext 时这些 CSR 按 WARL 处理, 取值固定为 0.
        0x7A0..=0x7AF => CSR_OK,

        // ---- AIA IMSIC CSR, 是否可用由 CFG_AIA 判定 ----
        MISELECT => {
            if CFG_AIA { ctx.state.imsic_m.select = val as u32; CSR_OK }
            else { CSR_ILL }
        }
        MIREG => {
            if CFG_AIA {
                let select = ctx.state.imsic_m.select;
                let rc = imsic_reg_write(&mut ctx.state.imsic_m, select, val);
                // MIREG 的写入可能改动 eip, eie, eidelivery 与 eithreshold,
                // 需要据此更新 MEIP 与 SEIP; 否则下一次 check_pending_interrupts
                // 会读到过时的 mip.
                sync_imsic_one(ctx.state, true);
                rc
            } else { CSR_ILL }
        }
        SISELECT => {
            if CFG_AIA { ctx.state.imsic_s.select = val as u32; CSR_OK }
            else { CSR_ILL }
        }
        SIREG => {
            if CFG_AIA {
                let select = ctx.state.imsic_s.select;
                let rc = imsic_reg_write(&mut ctx.state.imsic_s, select, val);
                sync_imsic_one(ctx.state, false);
                rc
            } else { CSR_ILL }
        }
        MTOPEI | STOPEI => {
            if !CFG_AIA {
                return CSR_ILL;
            }
            // 按 RISC-V AIA 规范, 写入 mtopei 或 stopei 会得到响应写入值中编码的 IID.
            let iid = ((val >> 16) & 0x7FF) as u32;
            if iid == 0 {
                return CSR_OK;
            }
            let file = if addr == MTOPEI { &mut ctx.state.imsic_m } else { &mut ctx.state.imsic_s };
            imsic_topei_claim_iid(file, iid);
            sync_imsic_one(ctx.state, addr == MTOPEI);
            // IPI 的 IID 还需清除 MSIP 或 SSIP.
            if addr == MTOPEI && iid == IID_M_IPI {
                ctx.state.mip.fetch_and(!(1 << 3), Ordering::AcqRel);
            } else if addr == STOPEI && iid == IID_S_IPI {
                ctx.state.mip.fetch_and(!(1 << 1), Ordering::AcqRel);
            }
            return CSR_OK;
        }
        0xFB0 /* mtopi */ => {
            if !CFG_AIA {
                return CSR_ILL;
            }
            let iid = ((val >> 16) & 0x7FF) as u32;
            if iid == 0 {
                return CSR_OK;
            } else if iid == IID_M_IPI {
                // 机器级 IPI, 即 MSIP: 得到响应 IMSIC M 文件的 eip 并清除 MSIP.
                imsic_topei_claim_iid(&mut ctx.state.imsic_m, iid);
                sync_imsic_one(ctx.state, true);
                ctx.state.mip.fetch_and(!(1 << 3), Ordering::AcqRel);
            } else if iid == 7 {
                // IRQ_M_TIMER, 即 MTIP: 清除 mip 中对应位.
                // 仅当 mtimecmp 仍已到期时才推进, 固件侧 sbi_timer_process()
                // 可能已经设定了更远的期限; 无条件 bump_mtimecmp(4)
                // 会覆盖该期限, 使 MTIP 每 4 条指令再次触发,
                // 反复进入机器级陷态处理函数中.
                ctx.state.mip.fetch_and(!(1 << 7), Ordering::AcqRel);
                let cur = ctx.mtime();
                let cmp = unsafe {
                    &*ctx.clint.mtimecmp.add(ctx.extern_hart_id as usize)
                }.load(Ordering::Acquire);
                if cmp <= cur {
                    ctx.bump_mtimecmp(4);
                }
            } else if iid == 11 {
                // IID 为 11 表示 MEI, 即机器级外部中断的主标识.
                // 此处不执行任何操作: 外部中断只能经 MTOPEI 得到响应,
                // 否则 ``csrr mtopi`` 的读改写会提前清除最高的 eip,
                // 使 OpenSBI 的 csr_swap(MTOPEI) 读到空值.
            } else {
                // IMSIC 次要 IID: 得到响应对应的 eip 位并同步 MEIP.
                imsic_topei_claim_iid(&mut ctx.state.imsic_m, iid);
                sync_imsic_one(ctx.state, true);
            }
            return CSR_OK;
        }
        0xDB0 /* stopi */ => {
            if !CFG_AIA {
                return CSR_ILL;
            }
            let iid = ((val >> 16) & 0x7FF) as u32;
            if iid == 0 {
                return CSR_OK;
            } else if iid == IID_S_IPI {
                // S 模式 IPI, 即 SSIP: 得到响应 IMSIC S 文件的 eip 并清除 SSIP.
                imsic_topei_claim_iid(&mut ctx.state.imsic_s, iid);
                sync_imsic_one(ctx.state, false);
                ctx.state.mip.fetch_and(!(1 << 1), Ordering::AcqRel);
            } else if iid == 5 {
                // IRQ_S_TIMER, 即 STIP: 清除 mip 中对应位并推进 stimecmp,
                // 使该中断不会立即再次触发.
                ctx.state.mip.fetch_and(!(1 << 5), Ordering::AcqRel);
                let now = ctx.mtime();
                if ctx.state.stimecmp <= now {
                    ctx.state.stimecmp = now + 4;
                    // 推进 stimecmp; 若不设定, sync_mtip 的活跃分支
                    // 退化为共享 mtime 的比较, 会被其它活跃 hart 的进度提前置位.
                    write_stimecmp(ctx.state, now, ctx.state.stimecmp, ctx.clint.timebase_hz);
                }
            } else if iid == 9 {
                // IID 为 9 表示 SEI, 即 S 模式外部中断的主标识.
                // 回写原值不执行任何操作: 外部中断只能通过 STOPEI 得到响应.
                // 若在此处提前得到响应最高的 eip, 内核 imsic_handle_irq 的
                // csr_swap(STOPEI) 将读到空值, IPI 握手无法完成.
            } else {
                // IMSIC 次要 IID: 得到响应对应的 eip 位并同步 SEIP.
                imsic_topei_claim_iid(&mut ctx.state.imsic_s, iid);
                sync_imsic_one(ctx.state, false);
            }
            return CSR_OK;
        }
        // 其余 AIA CSR 不涉及 IMSIC, 写入被忽略.
        0x308 /* mvien */ | 0x309 /* mvip */
        | 0x318 /* mvienh */ | 0x319 /* mviph */ => CSR_OK,

        // 仅 RV32 使用的 CSR, 在 RV64 上写入被忽略.
        0x310 /* mstatush */ | 0x312 /* medelegh */ | 0x313 /* midelegh */ => CSR_OK,

        // ---- PMP CSR, 用于 RV64 ----
        // RV64 上编号为奇数的 pmpcfg 寄存器非法.
        a if (0x3A1..=0x3AF).contains(&a) && (a & 1) != 0 => CSR_ILL,
        // 编号为偶数的 pmpcfg 把 u64 拆成 8 个配置字节.
        a if (0x3A0..=0x3AE).contains(&a) && (a & 1) == 0 => {
            let entry_base = ((a - 0x3A0) as usize / 2) * 8;
            if (entry_base as u8) < ctx.pmp.num {
                // 仅写入 [entry_base, num) 内的条目. num 小于 entry_base + 8 时
                // 截断, 避免越界写入破坏 cfg 的假设.
                let n = ((ctx.pmp.num as usize) - entry_base).min(8);
                for i in 0..n {
                    let b = ((val >> (i * 8)) & 0xFF) as u8;
                    unsafe { *ctx.pmp.cfg.add(entry_base + i) = b; }
                }
            }
            CSR_OK
        }
        // PMP 地址寄存器 0x3B0-0x3EF: 每个条目一个, 各 8 字节.
        a if (0x3B0..=0x3EF).contains(&a) => {
            let n = (a - 0x3B0) as usize;
            if n < ctx.pmp.num as usize {
                unsafe { *ctx.pmp.addr.add(n) = val; }
            }
            CSR_OK
        }

        // 其余情形退出到 Python
        _ => CSR_EXIT,
    }
}

// ============================================================
//  辅助函数
// ============================================================

/// 刷新两个 TLB, 不引入 translate 模块.
#[inline]
fn tlb_flush_inline(state: &mut HartState) {
	for e in state.itlb.iter_mut() {
		e.valid = 0;
	}
	for e in state.dtlb.iter_mut() {
		e.valid = 0;
	}
}

/// 浮点单元是否被禁用, 即 mstatus.FS 是否为 Off. 访问浮点 CSR 要求 FS 不为 0.
#[inline]
fn fp_off(state: &HartState) -> bool {
	(state.mstatus & MSTATUS_FS) == 0
}

/// 判定当前模式是否有权访问特权级为 priv_req 的 CSR.
#[inline]
fn csr_priv_ok(mode: u8, priv_req: u8) -> bool {
	match priv_req {
		0 => true,                  // 用户级可读
		1 => mode >= riscv_mode::S, // S 模式
		2 => mode >= riscv_mode::H, // 虚拟化 S 模式, 未使用
		3 => mode >= riscv_mode::M, // 机器级
		_ => false,
	}
}

/// 处理一条 CSR 指令, 包括 CSRRW, CSRRS, CSRRC, CSRRWI, CSRRSI 与 CSRRCI.
pub fn handle_csr(
	ctx: &mut CsrContext,
	rd: u8,
	rs1: u8,
	csr_addr: u16,
	funct3: u8,
	instr: u32,
	instr_group: &mut InstrToBeExec,
) -> u64 {
	// 读取 CSR 的旧值
	let (old_val, status) = csr_read(ctx, csr_addr);
	if status == CSR_ILL {
		deliver_illegal_instruction(ctx.state, instr as u64);
		return 0;
	}
	if status == CSR_EXIT {
		instr_group.exit_reason = exit_reason::ECALL;
		instr_group.exit_instr = instr;
		return EXIT_SENTINEL;
	}

	// 按 funct3 计算新值
	let rs1_val = if funct3 >= 4 {
		// CSRRWI, CSRRSI 与 CSRRCI 使用零扩展的 5 位无符号 rs1 字段.
		rs1 as u64
	} else {
		read_gpr(ctx.state, rs1)
	};

	let new_val: u64 = match funct3 & 0x3 {
		1 => rs1_val,            // CSRRW 与 CSRRWI
		2 => old_val | rs1_val,  // CSRRS 与 CSRRSI
		3 => old_val & !rs1_val, // CSRRC 与 CSRRCI
		_ => {
			deliver_illegal_instruction(ctx.state, instr as u64);
			return 0;
		}
	};

	// mtopei 与 stopei 得到响应读取时查到的 IID, 而不是当前最高优先级的 IID.
	// 接收方 A 查看与得到响应之间, 发送方 B 可能置位新的 eip; 若按最高优先级
	// 重新扫描, 会得到响应 B 的中断而丢失 A 的确认.
	if csr_addr == MTOPEI || csr_addr == STOPEI {
		let is_mfile = csr_addr == MTOPEI;
		let file = if is_mfile {
			&mut ctx.state.imsic_m
		} else {
			&mut ctx.state.imsic_s
		};
		let iid = (old_val >> 16) as u32;
		imsic_topei_claim_iid(file, iid);
		// 软件中断还需清除对应的 mip 位: M 文件的 IID 为 3 时清除 MSIP,
		// S 文件的 IID 为 1 时清除 SSIP. 纯 AIA 模式下这些位为 0,
		// IPI 经 MEIP 与 SEIP 路由; CLINT 与 IMSIC 混用时需要显式清除.
		if is_mfile && iid == 3 {
			ctx.state.mip.fetch_and(!(1 << 3), Ordering::AcqRel); // MSIP
		} else if !is_mfile && iid == 1 {
			ctx.state.mip.fetch_and(!(1 << 1), Ordering::AcqRel); // SSIP
		}
		// 定时器中断的 IID 为 5 与 7; 它们经 mtopi 与 stopi 路径得到响应,
		// 不经过 IMSIC. 若内核以定时器 IID 写入 STOPEI 或 MTOPEI,
		// 则清除对应的 mip 位并推进比较寄存器, 使中断不会立即再次触发.
		if !is_mfile && iid == 5 {
			ctx.state.mip.fetch_and(!(1 << 5), Ordering::AcqRel);
			if ctx.state.stimecmp > 0 && ctx.state.stimecmp <= ctx.mtime() {
				ctx.state.stimecmp = ctx.mtime() + 4;
			}
		}
		if is_mfile && iid == 7 {
			ctx.state.mip.fetch_and(!(1 << 7), Ordering::AcqRel);
			// 仅当 mtimecmp 仍已到期时才推进, 避免覆盖 sbi_timer_process()
			// 已经设定的未来期限.
			let cur = ctx.mtime();
			let cmp = unsafe { &*ctx.clint.mtimecmp.add(ctx.extern_hart_id as usize) }
				.load(Ordering::Acquire);
			if cmp <= cur {
				ctx.bump_mtimecmp(4);
			}
		}
		// 得到响应后重新同步该 IMSIC 文件, 更新 mip 中的 MEIP 与 SEIP;
		// 否则下一次 check_pending_interrupts 会读到过时的位并产生多余的陷态.
		sync_imsic_one(ctx.state, is_mfile);
		if rd != 0 {
			write_gpr(ctx.state, rd, old_val);
		}
		return 4;
	}

	// 即使 rd 为 0 也必须写入 CSR, rd 为 0 时只是跳过把旧值写入 rd.
	let wstatus = csr_write(ctx, csr_addr, new_val);
	if wstatus == CSR_ILL {
		deliver_illegal_instruction(ctx.state, instr as u64);
		return 0;
	}
	if wstatus == CSR_EXIT {
		instr_group.exit_reason = exit_reason::ECALL;
		instr_group.exit_instr = instr;
		return EXIT_SENTINEL;
	}

	// STIMECMP 的写路径立即更新 mip.STIP, 此处不再基于共享 mtime 重新比较.
	// sync_mtip 仍在下一指令边界运行, 在此之前 stopi 读到的 mip.STIP
	// 已是最新值.

	if rd != 0 {
		write_gpr(ctx.state, rd, old_val);
	}
	4
}

// ============================================================
//  GPR 读取与写入
// ============================================================

#[inline]
fn read_gpr(state: &HartState, rs: u8) -> u64 {
	if rs == 0 {
		0
	} else {
		state.gprs[rs as usize]
	}
}

#[inline]
fn write_gpr(state: &mut HartState, rd: u8, val: u64) {
	if rd != 0 {
		state.gprs[rd as usize] = val;
	}
}

/// 处理函数返回该值表示请求退出到 Python.
pub const EXIT_SENTINEL: u64 = u64::MAX;

// ============================================================
//  测试
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;

	fn test_state() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.mode = riscv_mode::M;
		s
	}

	fn test_pmp() -> PmpCtx {
		PmpCtx {
			cfg: std::ptr::null_mut(),
			addr: std::ptr::null_mut(),
			num: 0,
		}
	}

	/// 创建一个 ``ConcurrentClintCtx``, 其中 ``mtime`` 指向真实计数器,
	/// 使测试中的 ``ctx.mtime()`` 可用. ``mtimecmp`` 与 ``msip`` 为空指针,
	/// 需要推进定时器的测试必须自行设置.
	fn test_clint(mtime_val: u64) -> ConcurrentClintCtx {
		let mtime_box = Box::new(std::sync::atomic::AtomicU64::new(mtime_val));
		let mtime_ptr = Box::into_raw(mtime_box) as *const std::sync::atomic::AtomicU64;
		ConcurrentClintCtx {
			base: 0,
			mtime: mtime_ptr,
			mtimecmp: std::ptr::null(),
			msip: std::ptr::null(),
			num_harts: 1,
			msip_pending: std::cell::Cell::new(std::ptr::null()),
			hart_threads: std::cell::Cell::new(std::ptr::null()),
			hart_states: std::cell::Cell::new(std::ptr::null()),
			timebase_hz: 0, // 单元测试禁用时钟源推进, 保持确定性
		}
	}

	fn test_csr_ctx<'a>(state: &'a mut HartState, clint: &'a ConcurrentClintCtx) -> CsrContext<'a> {
		// 使测试过程中，引用具有 'static 生命周期.
		CsrContext::new(state, 0, Box::leak(Box::new(test_pmp())), clint)
	}

	fn test_csr_ctx_pmp<'a>(
		state: &'a mut HartState,
		pmp: &'a PmpCtx,
		clint: &'a ConcurrentClintCtx,
	) -> CsrContext<'a> {
		CsrContext::new(state, 0, pmp, clint)
	}

	#[test]
	fn read_mstatus() {
		let mut s = test_state();
		s.mstatus = 0x1880; // MPP 为 S, MPIE 为 1
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let (v, st) = csr_read(&mut ctx, MSTATUS);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, 0x1880);
	}

	#[test]
	fn read_sstatus_view() {
		let mut s = test_state();
		s.mstatus = MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_SPP;
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let (v, st) = csr_read(&mut ctx, SSTATUS);
		assert_eq!(st, CSR_OK);
		assert_eq!(
			v & SSTATUS_READ_MASK,
			MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_SPP
		);
	}

	#[test]
	fn write_satp_flushes_tlb() {
		let mut s = test_state();
		s.mmu_mode = 8;
		// 预先设置好 TLB
		s.itlb[0].valid = 1;
		s.dtlb[0].valid = 1;
		// 写入 satp
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SATP, 0x8000000000000000u64); // Sv39 模式, PPN 为 0
			assert_eq!(st, CSR_OK);
		}
		assert_eq!(s.mmu_mode, 8);
		// 两个 TLB 的条目均应失效
		assert_eq!(s.itlb[0].valid, 0);
		assert_eq!(s.dtlb[0].valid, 0);
	}

	/// satp 的 ASID 位 [59:44] 固定为 0, TLB 条目不使用 ASID 标记.
	/// 写入全 1 的 ASID 后读回必须为 0, MODE 与 PPN 保持不变.
	/// 测试取值 satp 为 0x8000100000082f1b, 即 Sv39 模式, ASID 为 1, PPN 为 0x82f1b.
	#[test]
	fn write_satp_asid_hardwired_zero() {
		let mut s = test_state();
		// 写入全 1 的 ASID, 再读回检查可写位
		let probe = 0x8000000000000000u64 | (0xFFFFu64 << 44) | 0x82f1b;
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SATP, probe);
			assert_eq!(st, CSR_OK);
			let (v, st) = csr_read(&mut ctx, SATP);
			assert_eq!(st, CSR_OK);
			assert_eq!(
				v, 0x8000000000082f1b,
				"ASID must read back as 0; MODE/PPN preserved"
			);
			assert_ne!(v, probe, "old behavior (raw ASID stored) must not reappear");
		}
		// 掩码不得影响 MODE 与 PPN
		assert_eq!(s.mmu_mode, 8);
	}

	#[test]
	fn write_sstatus_updates_mstatus_fields() {
		let mut s = test_state();
		// 通过 sstatus 写入 SIE 位
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SSTATUS, MSTATUS_SIE);
			assert_eq!(st, CSR_OK);
		}
		assert_eq!(s.mstatus & MSTATUS_SIE, MSTATUS_SIE);
	}

	#[test]
	fn write_unknown_csr_triggers_exit() {
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let st = csr_write(&mut ctx, 0x7C0, 0); // 自定义 CSR, 加速引擎未实现
		assert_eq!(st, CSR_EXIT);
	}

	/// sie 位于 0x104, 是 mie 位于 0x304 的受限视图.
	/// 写入 sie 只更新 mideleg 委派的位在 mie 中的值.
	#[test]
	fn write_sie_updates_mie() {
		let mut s = test_state();
		s.mideleg = 1 << 5; // STIE 已委派
		s.mie = 0;
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SIE, 1 << 5); // 通过 sie 写入 STIE
			assert_eq!(st, CSR_OK);
		}
		assert_eq!(
			s.mie & (1 << 5),
			1 << 5,
			"mie.STIE must be set when sie.STIE is written"
		);
		// 未委派的位不得写入 mie
		s.mideleg = 1 << 5;
		s.mie = 0;
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SIE, (1 << 5) | (1 << 3)); // STIE 与 MSIP
			assert_eq!(st, CSR_OK);
		}
		assert_eq!(
			s.mie & (1 << 3),
			0,
			"non-delegated bits in sie write must be masked"
		);
		assert_eq!(s.mie & (1 << 5), 1 << 5, "delegated STIE must still be set");
	}

	/// 读取 sie 返回 mie 与 mideleg 的交集, 只显示已委派的位.
	#[test]
	fn read_sie_shows_delegated_bits() {
		// 构造具有指定 mie 与 mideleg 的状态
		let mut s2 = test_state();
		s2.mideleg = (1 << 5) | (1 << 1); // STIE 与 SSIE 已委派
		s2.mie = (1 << 5) | (1 << 3); // mie 中 STIE 与 MSIE 置位
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s2, &clint);
		let (v, st) = csr_read(&mut ctx, SIE);
		assert_eq!(st, CSR_OK);
		assert_eq!(
			v,
			1 << 5,
			"sie must only show delegated bits (STIE), not MSIE"
		);
	}

	/// sip 位于 0x144
	#[test]
	fn write_sip_updates_mip() {
		let mut s = test_state();
		s.mideleg = 1 << 1; // 已委派 SSIP
		s.mip.store(0, Ordering::Release);
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SIP, 1 << 1); // 通过 sip 写入 SSIP
			assert_eq!(st, CSR_OK);
		}
		assert_eq!(
			s.mip.load(Ordering::Acquire) & (1 << 1),
			1 << 1,
			"mip.SSIP must be set when sip.SSIP is written"
		);
	}

	/// sip 位于 0x144, 与 sie 不同: 只有 SSIP 即位 1 可写, STIP 与 SEIP
	/// 由 SSTC 定时器与 PLIC 驱动, 不得被 ``csrw sip`` 覆写. 写回陈旧 sip 时,
	/// STIP 与 SEIP 的挂起状态必须保留.
	#[test]
	fn write_sip_preserves_stip_and_seip() {
		let mut s = test_state();
		// STIP, SEIP 与 SSIP 全部委派到 S 模式.
		s.mideleg = (1 << 5) | (1 << 9) | (1 << 1);
		s.mip
			.store((1 << 5) | (1 << 9) | (1 << 1), Ordering::Release);
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SIP, 0); // 尝试清除全部挂起位
			assert_eq!(st, CSR_OK);
		}
		let mip = s.mip.load(Ordering::Acquire);
		assert_eq!(
			mip & (1 << 5),
			1 << 5,
			"STIP is hardware-driven, must be preserved through sip write"
		);
		assert_eq!(
			mip & (1 << 9),
			1 << 9,
			"SEIP is hardware-driven, must be preserved through sip write"
		);
		assert_eq!(
			mip & (1 << 1),
			0,
			"SSIP is the only writable bit through sip, must be cleared"
		);
	}

	/// mip 位于 0x344, 只读位 STIP 与 SEIP 不得被受调试程序的 ``csrw mip`` 覆写.
	/// 一条 ``csrw mip, 0`` 必须保留 SSTC 定时器与外部中断的挂起位.
	#[test]
	fn write_mip_preserves_stip_and_seip() {
		let mut s = test_state();
		s.mip
			.store((1 << 5) | (1 << 9) | (1 << 1), Ordering::Release);
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, MIP, 0); // 尝试清除全部挂起位
			assert_eq!(st, CSR_OK);
		}
		let mip = s.mip.load(Ordering::Acquire);
		assert_eq!(
			mip & (1 << 5),
			1 << 5,
			"STIP is read-only, must be preserved"
		);
		assert_eq!(
			mip & (1 << 9),
			1 << 9,
			"SEIP is read-only, must be preserved"
		);
		assert_eq!(
			mip & (1 << 1),
			0,
			"SSIP is software-writable, must be cleared"
		);
	}

	#[test]
	fn read_time() {
		let mut s = test_state();
		let clint = test_clint(123456789);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let (v, st) = csr_read(&mut ctx, 0xC01);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, 123456789);
	}

	#[test]
	fn read_timeh() {
		let mut s = test_state();
		let clint = test_clint(0x123456789AB);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let (v, st) = csr_read(&mut ctx, 0xC81);
		assert_eq!(st, CSR_OK);
		// 0x123456789AB 右移 32 位
		assert_eq!(v, 0x123);
	}

	/// mcycle 与 cycle 由本 hart 已执行完成的指令数换算.
	#[test]
	fn read_mcycle_from_per_hart_instr_count() {
		let mut s = test_state();
		s.total_instrs = 5;
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let (v, st) = csr_read(&mut ctx, MCYCLE);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, 5 * CYCLES_PER_INSTR);
		let (v, st) = csr_read(&mut ctx, CYCLE);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, 5 * CYCLES_PER_INSTR);
	}

	/// minstret 与 instret 直接给出本 hart 已执行完成的指令数.
	#[test]
	fn read_minstret_from_per_hart_instr_count() {
		let mut s = test_state();
		s.total_instrs = 0x1_2345_6789;
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let (v, st) = csr_read(&mut ctx, MINSTRET);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, 0x1_2345_6789);
		let (v, st) = csr_read(&mut ctx, INSTRET);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, 0x1_2345_6789);
	}

	/// 高半 CSR 返回同一计数的 bits[63:32].
	#[test]
	fn read_counter_high_halves() {
		let mut s = test_state();
		s.total_instrs = 0x1_2345_6789;
		let cycles = 0x1_2345_6789u64.wrapping_mul(CYCLES_PER_INSTR);
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let (v, st) = csr_read(&mut ctx, MCYCLEH);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, (cycles >> 32) & 0xFFFF_FFFF);
		let (v, st) = csr_read(&mut ctx, CYCLEH);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, (cycles >> 32) & 0xFFFF_FFFF);
		let (v, st) = csr_read(&mut ctx, MINSTRETH);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, 0x1);
		let (v, st) = csr_read(&mut ctx, INSTRETH);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, 0x1);
	}

	/// 两个 hart 的计数器互不影响: 各自的读数只由自身的指令数给出.
	#[test]
	fn counter_reads_are_independent_between_harts() {
		let mut s0 = test_state();
		let mut s1 = test_state();
		s0.total_instrs = 3;
		s1.total_instrs = 7;
		let clint = test_clint(0);
		let (v0, st0) = {
			let mut ctx = test_csr_ctx(&mut s0, &clint);
			csr_read(&mut ctx, MCYCLE)
		};
		let (v1, st1) = {
			let mut ctx = test_csr_ctx(&mut s1, &clint);
			csr_read(&mut ctx, MCYCLE)
		};
		assert_eq!((v0, st0), (3 * CYCLES_PER_INSTR, CSR_OK));
		assert_eq!((v1, st1), (7 * CYCLES_PER_INSTR, CSR_OK));
	}

	/// U 模式读计数器要求 mcounteren 与 scounteren 的对应位同时置位.
	///
	#[test]
	fn counter_read_in_u_mode_requires_both_counteren() {
		let clint = test_clint(0);

		// mcounteren 置位而 scounteren 未置位: 拒绝.
		let mut s = test_state();
		s.mode = riscv_mode::U;
		s.mcounteren = 0b111;
		s.scounteren = 0;
		{
			let mut ctx = test_csr_ctx(&mut s, &clint);
			assert_eq!(csr_read(&mut ctx, CYCLE).1, CSR_ILL);
		}

		// mcounteren 未置位而 scounteren 置位: 拒绝.
		let mut s = test_state();
		s.mode = riscv_mode::U;
		s.mcounteren = 0;
		s.scounteren = 0b111;
		{
			let mut ctx = test_csr_ctx(&mut s, &clint);
			assert_eq!(csr_read(&mut ctx, CYCLE).1, CSR_ILL);
		}

		// 二者同时置位: 允许.
		let mut s = test_state();
		s.mode = riscv_mode::U;
		s.mcounteren = 0b111;
		s.scounteren = 0b111;
		{
			let mut ctx = test_csr_ctx(&mut s, &clint);
			assert_eq!(csr_read(&mut ctx, CYCLE).1, CSR_OK);
		}
	}

	/// 高半计数器与低半共用 counteren 的同一位.
	#[test]
	fn counter_high_half_shares_enable_bit() {
		let clint = test_clint(0);
		let mut s = test_state();
		s.mode = riscv_mode::U;
		s.mcounteren = 0b001;
		s.scounteren = 0b001;
		let mut ctx = test_csr_ctx(&mut s, &clint);
		assert_eq!(csr_read(&mut ctx, CYCLE).1, CSR_OK);
		assert_eq!(csr_read(&mut ctx, CYCLEH).1, CSR_OK);
		assert_eq!(csr_read(&mut ctx, INSTRET).1, CSR_ILL);
		assert_eq!(csr_read(&mut ctx, INSTRETH).1, CSR_ILL);
	}

	/// S 模式是否可读计数器只由 mcounteren 判定, 不读取 scounteren.
	#[test]
	fn counter_read_in_s_mode_uses_mcounteren_only() {
		let clint = test_clint(0);
		let mut s = test_state();
		s.mode = riscv_mode::S;
		s.mcounteren = 0b010;
		s.scounteren = 0;
		let mut ctx = test_csr_ctx(&mut s, &clint);
		assert_eq!(csr_read(&mut ctx, TIME).1, CSR_OK);
		assert_eq!(csr_read(&mut ctx, TIMEH).1, CSR_OK);
		assert_eq!(csr_read(&mut ctx, CYCLE).1, CSR_ILL);
	}

	/// M 模式读计数器不受 counteren 限制.
	#[test]
	fn counter_read_in_m_mode_ignores_counteren() {
		let clint = test_clint(0);
		let mut s = test_state();
		s.mcounteren = 0;
		s.scounteren = 0;
		let mut ctx = test_csr_ctx(&mut s, &clint);
		assert_eq!(csr_read(&mut ctx, CYCLE).1, CSR_OK);
		assert_eq!(csr_read(&mut ctx, INSTRET).1, CSR_OK);
	}

	/// ``csrrs a0, mcycle, x0`` 必须在一轮加速执行内完成:
	/// 读路径给出计数, 回写路径返回成功, 不退出到 Python.
	#[test]
	fn csrrs_mcycle_stays_in_acceleration() {
		let mut s = test_state();
		s.total_instrs = 123;
		let mut r = make_instr_group();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let adv = handle_csr(&mut ctx, 10, 0, MCYCLE, 0b010, 0xB000_2573, &mut r);
		assert_eq!(adv, 4, "rdmcycle 必须在一轮加速执行中完成");
		assert_eq!(r.exit_reason, 0, "rdmcycle 不得设置退出原因");
		assert_eq!(ctx.state.gprs[10], 123 * CYCLES_PER_INSTR);
	}

	#[test]
	fn csr_priv_check() {
		assert!(csr_priv_ok(riscv_mode::M, 3));
		assert!(csr_priv_ok(riscv_mode::M, 1));
		assert!(!csr_priv_ok(riscv_mode::S, 3));
		assert!(csr_priv_ok(riscv_mode::S, 1));
		assert!(csr_priv_ok(riscv_mode::U, 0));
		assert!(!csr_priv_ok(riscv_mode::U, 1));
	}

	// ---- handle_csr 测试: rd 为 x0 的情形 ----

	fn make_instr_group() -> InstrToBeExec {
		unsafe { std::mem::zeroed() }
	}

	/// csrrw x0, mscratch, t0.
	/// 即使 rd 为 x0 也必须写入 CSR, 只跳过把旧值写入 rd.
	#[test]
	fn handle_csr_write_with_rd_x0() {
		let mut s = test_state();
		s.gprs[5] = 0xDEAD_BEEF; // t0 为待写入的值
		let mut r = make_instr_group();
		let (advance, mscratch_val) = {
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			// CSRRW x0, mscratch, t0: funct3 为 001, rs1 为 5 即 t0, rd 为 0, csr 为 0x340
			let adv = handle_csr(&mut ctx, 0, 5, MSCRATCH, 0b001, 0x34051073, &mut r);
			(adv, ctx.state.mscratch)
		};
		assert_eq!(advance, 4);
		assert_eq!(
			mscratch_val, 0xDEAD_BEEF,
			"mscratch must be written even though rd==x0"
		);
	}

	/// csrrwi x0, mscratch, 5.
	#[test]
	fn handle_csr_write_imm_with_rd_x0() {
		let mut s = test_state();
		let mut r = make_instr_group();
		let (advance, mscratch_val) = {
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			// CSRRWI x0, mscratch, 5: funct3 为 101, rs1 字段为 5, rd 为 0, csr 为 0x340
			let adv = handle_csr(&mut ctx, 0, 5, MSCRATCH, 0b101, 0x340552f3, &mut r);
			(adv, ctx.state.mscratch)
		};
		assert_eq!(advance, 4);
		assert_eq!(
			mscratch_val, 5,
			"mscratch must be written via csrwi even though rd==x0"
		);
	}

	/// csrrs t0, mscratch, x0: rd 为 t0, rs1 为 x0, 无写入位.
	/// 旧值写入 rd, CSR 保持不变.
	#[test]
	fn handle_csr_read_to_rd() {
		let mut s = test_state();
		s.mscratch = 0xCAFE;
		let mut r = make_instr_group();
		let (advance, mscratch_val, gpr5_val) = {
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			// CSRRS t0, mscratch, x0: funct3 为 010, rs1 为 0, rd 为 5
			let adv = handle_csr(&mut ctx, 5, 0, MSCRATCH, 0b010, 0x3402b2f3, &mut r);
			(adv, ctx.state.mscratch, ctx.state.gprs[5])
		};
		assert_eq!(advance, 4);
		assert_eq!(mscratch_val, 0xCAFE, "mscratch unchanged on read");
		assert_eq!(gpr5_val, 0xCAFE, "old value written to rd");
	}

	// ---- PMP CSR 写入测试，不退出到 Python ----

	/// 构造 PmpCtx, 使 PMP CSR 的写入可以与底层存储比对.
	fn test_pmp_real(num: u8) -> (PmpCtx, Vec<u8>, Vec<u64>) {
		let mut cfg = vec![0u8; num as usize];
		let mut addr = vec![0u64; num as usize];
		let ctx = PmpCtx {
			cfg: cfg.as_mut_ptr(),
			addr: addr.as_mut_ptr(),
			num,
		};
		(ctx, cfg, addr)
	}

	#[test]
	fn pmpcfg_write_packs_bytes() {
		let (pmp, cfg, _addr) = test_pmp_real(16);
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx_pmp(&mut s, &pmp, &clint);
		// pmpcfg0 位于 0x3A0, 覆盖条目 0 至 7.
		// 写入条目 0 为 0xAB, 条目 2 为 0xCD, 分别位于字节位置 0 与 2.
		let val: u64 = 0xAB | (0xCDu64 << 16);
		let st = csr_write(&mut ctx, 0x3A0, val);
		assert_eq!(st, CSR_OK);
		assert_eq!(cfg[0], 0xAB);
		assert_eq!(cfg[1], 0x00);
		assert_eq!(cfg[2], 0xCD);
		// 条目 7 后的字节不得改动.
		assert_eq!(cfg[8], 0x00);
	}

	#[test]
	fn pmpcfg_write_out_of_range_is_noop() {
		// num 为 4, 但分配 8 字节 cfg 缓冲区, 以便断言条目 4 至 7 未被写入;
		// pmpcfg0 的 8 字节写若越过 num, 会污染堆内存.
		let mut cfg = vec![0u8; 8];
		let mut addr = vec![0u64; 4];
		let pmp = PmpCtx {
			cfg: cfg.as_mut_ptr(),
			addr: addr.as_mut_ptr(),
			num: 4,
		};
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx_pmp(&mut s, &pmp, &clint);
		// pmpcfg0 写入条目 0 至 7, 但 num 为 4, 只存在条目 0 至 3.
		let st = csr_write(&mut ctx, 0x3A0, 0xFF_FF_FF_FF_FF_FF_FF_FFu64);
		assert_eq!(st, CSR_OK);
		// 有效条目 0 至 3 被写入.
		assert_eq!(cfg[0], 0xFF);
		assert_eq!(cfg[1], 0xFF);
		assert_eq!(cfg[2], 0xFF);
		assert_eq!(cfg[3], 0xFF);
		// 越界条目 4 至 7 不得被写入.
		assert_eq!(cfg[4], 0x00);
		assert_eq!(cfg[5], 0x00);
		assert_eq!(cfg[6], 0x00);
		assert_eq!(cfg[7], 0x00);
	}

	#[test]
	fn pmpcfg_odd_returns_ill() {
		let (pmp, _cfg, _addr) = test_pmp_real(8);
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx_pmp(&mut s, &pmp, &clint);
		// pmpcfg1 位于 0x3A1, 在 RV64 上非法.
		let st = csr_write(&mut ctx, 0x3A1, 0);
		assert_eq!(st, CSR_ILL);
	}

	#[test]
	fn pmpaddr_write_works() {
		let (pmp, _cfg, addr) = test_pmp_real(16);
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx_pmp(&mut s, &pmp, &clint);
		// pmpaddr3 位于 0x3B3, 写入条目 3.
		let st = csr_write(&mut ctx, 0x3B3, 0xDEAD_BEEF_CAFE_BABEu64);
		assert_eq!(st, CSR_OK);
		assert_eq!(addr[3], 0xDEAD_BEEF_CAFE_BABE);
		assert_eq!(addr[2], 0, "adjacent entries untouched");
		assert_eq!(addr[4], 0, "adjacent entries untouched");
	}

	#[test]
	fn pmpaddr_write_out_of_range_is_noop() {
		let (pmp, _cfg, addr) = test_pmp_real(4);
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx_pmp(&mut s, &pmp, &clint);
		// pmpaddr4 位于 0x3B4, 但 num 为 4, 有效索引为 0 至 3.
		let st = csr_write(&mut ctx, 0x3B4, 0xFFFF_FFFF_FFFF_FFFFu64);
		assert_eq!(st, CSR_OK);
		assert_eq!(addr[3], 0, "last valid entry untouched");
	}

	#[test]
	fn pmp_write_does_not_exit() {
		let (pmp, _cfg, _addr) = test_pmp_real(8);
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx_pmp(&mut s, &pmp, &clint);
		// 全部 PMP 写入都应返回 CSR_OK, 不退出到 Python.
		assert_eq!(csr_write(&mut ctx, 0x3A0, 0), CSR_OK); // pmpcfg0
		assert_eq!(csr_write(&mut ctx, 0x3A2, 0), CSR_OK); // pmpcfg2
		assert_eq!(csr_write(&mut ctx, 0x3B0, 0), CSR_OK); // pmpaddr0
		assert_eq!(csr_write(&mut ctx, 0x3B7, 0), CSR_OK); // pmpaddr7
		assert_eq!(csr_write(&mut ctx, 0x3EF, 0), CSR_OK); // pmpaddr63, 最后一个条目
	}

	// ============================================================
	//  AIA CSR 的可用性, 由编译期常量 CFG_AIA 分发期望值
	// ============================================================
	//
	// CFG_AIA 为 false 时全部 AIA CSR 返回 CSR_ILL, 防止内核探测到
	// 不存在的 IMSIC 硬件. CFG_AIA 为 true 时返回 CSR_OK, IMSIC 寄存器
	// 可正常读写.
	//
	// 期望值按 CFG_AIA 计算而非硬编码, 使测试在两种构建配置下都正确.

	const AIA_CSRS: [(u16, &str); 8] = [
		(MISELECT, "miselect"),
		(MIREG, "mireg"),
		(SISELECT, "siselect"),
		(SIREG, "sireg"),
		(MTOPEI, "mtopei"),
		(STOPEI, "stopei"),
		(0xFB0, "mtopi"),
		(0xDB0, "stopi"),
	];

	fn aia_expected_status() -> u8 {
		if CFG_AIA {
			CSR_OK
		} else {
			CSR_ILL
		}
	}

	#[test]
	fn aia_csr_read_status() {
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let expected = aia_expected_status();
		for (addr, name) in &AIA_CSRS {
			let (_val, st) = csr_read(&mut ctx, *addr);
			assert_eq!(st, expected, "{name} (0x{addr:X}) read status mismatch");
		}
	}

	#[test]
	fn aia_csr_write_status() {
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let expected = aia_expected_status();
		for (addr, name) in &AIA_CSRS {
			let st = csr_write(&mut ctx, *addr, 0xDEAD_BEEFu64);
			assert_eq!(st, expected, "{name} (0x{addr:X}) write status mismatch");
		}
	}

	/// stopi 位于 0xDB0, 回写必须不执行任何操作.
	///
	/// ``csrr stopi`` 读出主标识 SEI, IID 为 9; 内核读改写回写该值时,
	/// 若在此处得到响应最高的 eip, 会提前清除外部中断, 使后续 STOPEI 的
	/// csr_swap 读到空值, IPI 握手无法完成. 同时，回写后 eip 位必须保持挂起.
	#[test]
	fn stopi_write_back_does_not_claim_external_eip() {
		if !CFG_AIA {
			return; // 仅在 AIA 构建下有意义
		}
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);

		// 准备 S 文件: eidelivery 为 1, eie 与 eip 置位 IID 10 对应的位.
		ctx.state.imsic_s.present = 1;
		ctx.state.imsic_s.eidelivery = 1;
		ctx.state.imsic_s.eip_ext_any.store(1, Ordering::Relaxed);
		ctx.state.imsic_s.eie[0].store(1 << 10, Ordering::Relaxed);
		ctx.state.imsic_s.eip[0].store(1 << 10, Ordering::Relaxed);

		// csrr stopi 读出主标识 SEI, 高 16 位为 9, 低半为优先级.
		let (val, st) = csr_read(&mut ctx, 0xDB0);
		assert_eq!(st, CSR_OK);
		assert_eq!(
			(val >> 16) & 0x7FF,
			9,
			"stopi must report SEI major identity"
		);

		// 必须不执行任何操作并保持 eip 挂起.
		let wst = csr_write(&mut ctx, 0xDB0, val);
		assert_eq!(wst, CSR_OK);
		assert_ne!(
			ctx.state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << 10),
			0,
			"stopi write-back must NOT claim the external eip"
		);
	}

	/// mtopi 位于 0xFB0.
	#[test]
	fn mtopi_write_back_does_not_claim_external_eip() {
		if !CFG_AIA {
			return; // 仅在 AIA 构建下有意义
		}
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);

		ctx.state.imsic_m.present = 1;
		ctx.state.imsic_m.eidelivery = 1;
		ctx.state.imsic_m.eip_ext_any.store(1, Ordering::Relaxed);
		ctx.state.imsic_m.eie[0].store(1 << 10, Ordering::Relaxed);
		ctx.state.imsic_m.eip[0].store(1 << 10, Ordering::Relaxed);

		let (val, st) = csr_read(&mut ctx, 0xFB0);
		assert_eq!(st, CSR_OK);
		assert_eq!(
			(val >> 16) & 0x7FF,
			11,
			"mtopi must report MEI major identity"
		);

		let wst = csr_write(&mut ctx, 0xFB0, val);
		assert_eq!(wst, CSR_OK);
		assert_ne!(
			ctx.state.imsic_m.eip[0].load(Ordering::Acquire) & (1 << 10),
			0,
			"mtopi write-back must NOT claim the external eip"
		);
	}

	/// handle_csr: CFG_AIA 为 false 时读取 AIA CSR 应产生 IllInstr 陷态.
	#[test]
	fn handle_csr_aia_miselect_read() {
		let mut s = test_state();
		let mut r = make_instr_group();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		// csrrs t0, miselect, x0: funct3 为 CSRRS, rd 为 5, rs1 为 0, csr 为 0x350
		let advance = handle_csr(&mut ctx, 5, 0, MISELECT, 0b010, 0x3502F073u32, &mut r);
		if CFG_AIA {
			assert_eq!(advance, 4, "AIA enabled -> normal advance");
			assert_eq!(ctx.state.mcause, 0, "mcause must not be set");
		} else {
			assert_eq!(advance, 0, "AIA disabled -> IllInstr redirects PC");
			assert_eq!(
				ctx.state.mcause & 0x7FFF_FFFF_FFFF_FFFF,
				2,
				"mcause must be IllInstr (2)"
			);
		}
	}

	/// handle_csr: CFG_AIA 为 false 时写入 AIA CSR 应产生 IllInstr 陷态.
	#[test]
	fn handle_csr_aia_miselect_write() {
		let mut s = test_state();
		s.gprs[10] = 0x70; // a0 为 IMSIC_EITHRESHOLD 的选择值
		let mut r = make_instr_group();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		// csrrw x0, miselect, a0: funct3 为 CSRRW, rd 为 0, rs1 为 10, csr 为 0x350
		let advance = handle_csr(&mut ctx, 0, 10, MISELECT, 0b001, 0x35051073u32, &mut r);
		if CFG_AIA {
			assert_eq!(advance, 4, "AIA enabled -> normal advance");
			assert_eq!(ctx.state.imsic_m.select, 0x70, "miselect stored");
		} else {
			assert_eq!(advance, 0, "AIA disabled -> IllInstr redirects PC");
			assert_eq!(ctx.state.mcause & 0x7FFF_FFFF_FFFF_FFFF, 2);
		}
	}

	/// handle_csr: OpenSBI 用 csr_read_allowed(CSR_MTOPI) 探测 AIA 是否存在.
	/// CFG_AIA 为 false 时该读取必须触发 IllInstr 并设置 mcause.
	#[test]
	fn handle_csr_mtopi_read() {
		let mut s = test_state();
		s.pc = 0x8000_0000; // 模拟真实的取指地址
		let mut r = make_instr_group();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		// csrrs t1, mtopi, x0: funct3 为 CSRRS, rd 为 6, rs1 为 0, csr 为 0xFB0
		let advance = handle_csr(&mut ctx, 6, 0, 0xFB0, 0b010, 0xFB032F73u32, &mut r);
		if CFG_AIA {
			assert_eq!(advance, 4, "AIA enabled -> normal advance");
		} else {
			assert_eq!(advance, 0, "AIA disabled -> IllInstr redirects PC");
			assert_eq!(ctx.state.mcause & 0x7FFF_FFFF_FFFF_FFFF, 2);
			assert_ne!(ctx.state.mepc, 0, "mepc must be saved by trap delivery");
		}
	}

	#[test]
	fn interrupt_no_act_for_mdid_and_pmpsplit() {
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);

		let (_v, st) = csr_read(&mut ctx, 0x5C0);
		assert_eq!(st, CSR_OK, "mdid read always OK");
		assert_eq!(csr_write(&mut ctx, 0x5C0, 3), CSR_OK);
		assert_eq!(ctx.state.mdid, 3);

		// pmpsplit 位于 0x5C1
		assert_eq!(csr_write(&mut ctx, 0x5C1, 1), CSR_OK);
		assert_eq!(ctx.state.pmpsplit, 1);
	}

	#[test]
	fn full_width_write_mdid() {
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);

		// 300 大于 u8 的最大值 255, 写入后读回仍为 300.
		assert_eq!(csr_write(&mut ctx, 0x5C0, 300), CSR_OK);
		assert_eq!(ctx.state.mdid, 300);

		assert_eq!(csr_write(&mut ctx, 0x5C0, 256), CSR_OK);
		assert_eq!(ctx.state.mdid, 256);

		// 读回与写一致
		let (v, st) = csr_read(&mut ctx, 0x5C0);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, 256);
	}
}
