//! 向宿主请求模块映像的接口。
//!
//! 映像的字节由宿主给出并写入飞地自己分配、自己映射的映像接收缓冲区, 飞地不指定映像
//! 占用的物理内存由谁分配。`ext_mod::loader` 只经本接口提出申请与登记缓冲区, 不直接发起
//! ecall, 故主机目标上的用例可以用本接口的实现代替它; 目标侧的实现见本文件末尾的
//! `EcallTransport`。

#[cfg(target_arch = "riscv64")]
use crate::constants::{
	ENCLAVE_MODULE_STATUS_OK, ENCLAVE_SUSPEND_MODULE_LOAD, ENCLAVE_SUSPEND_MODULE_SIZE,
};
#[cfg(target_arch = "riscv64")]
use crate::ecall_aux;

/// 向宿主请求模块映像。
pub trait Transport {
	/// 申请模块映像的字节数, 由宿主按模块编号报告。模块不存在时返回 None。
	///
	/// 报告的字节数即宿主侧映像的字节数, 含尾部的签名; 飞地按它决定映像接收缓冲区的
	/// 大小, 不自行推算。
	fn request_size(&self, module_id: u32) -> Option<u64>;

	/// 登记模块映像接收缓冲区: `va` 为它的起始虚拟地址, `len` 为它的字节数。
	///
	/// 该缓冲区由本飞地从自己的 S 模式页池分配物理页并逐页映射, 交付的映像字节写入其中。
	/// 登记被拒时返回假, 调用方按取入失败处理。
	fn register_img_recv_buf(&self, va: u64, len: u64) -> bool;

	/// 请求宿主交付模块映像的字节, `size` 为宿主报告的映像字节数。
	///
	/// 成功时返回宿主实际写入映像接收缓冲区的字节数; 宿主交付失败时返回 None。
	fn fetch(&self, module_id: u32, size: u64) -> Option<u64>;
}

// ---------------------------------------------------------------
//  目标侧实现
// ---------------------------------------------------------------

/// 经 M 模式 ecall 与宿主驱动请求模块映像的通路。
///
/// 一次请求由一对调用闭合: 飞地以 `(模块编号 << 32) | 让出原因` 让出 (见
/// `ENCLAVE_SUSPEND_MODULE_SIZE` 与 `ENCLAVE_SUSPEND_MODULE_LOAD`), M 模式把请求
/// 记入该飞地的元信息并交还宿主; 宿主经 503 号调用读出请求, 经 504 号或 505 号调用应答,
/// 再经 RESUME 交还飞地; 飞地随后经 508 号调用取回交付结果。
///
/// 模块编号是唯一随请求传给宿主的量: 映像的字节数由宿主自己报告并经 504 号调用记入
/// M 模式的元信息, 飞地按它决定映像接收缓冲区的大小, 二者都不由飞地推算。
///
/// 请求发出后本通路不再返回, 直到宿主应答并经 RESUME 交还飞地: 飞地在此不轮询, 也
/// 不需要超时, 等待上界由宿主侧决定。
#[cfg(target_arch = "riscv64")]
pub struct EcallTransport;

#[cfg(target_arch = "riscv64")]
impl Transport for EcallTransport {
	fn request_size(&self, module_id: u32) -> Option<u64> {
		let (status, size, _, _) = exchange(ENCLAVE_SUSPEND_MODULE_SIZE, module_id);
		if status == ENCLAVE_MODULE_STATUS_OK {
			Some(size)
		} else {
			None
		}
	}

	fn register_img_recv_buf(&self, va: u64, len: u64) -> bool {
		ecall_aux::enclave_call_module_img_recv_buf(va, len) == 0
	}

	fn fetch(&self, module_id: u32, _size: u64) -> Option<u64> {
		let (status, _, _, written) = exchange(ENCLAVE_SUSPEND_MODULE_LOAD, module_id);
		if status != ENCLAVE_MODULE_STATUS_OK {
			return None;
		}
		Some(written)
	}
}

/// 以给定的让出原因请求一次模块应答, 返回 508 号调用取回的四项结果。
///
/// 让出之后飞地停在 SUSPEND 那条 ecall 上, 直到宿主应答并经 RESUME 交还; 交还时飞地
/// 取回的是自己让出时的寄存器, 宿主的值不在其中, 故结果一律以 508 号调用取回。
#[cfg(target_arch = "riscv64")]
fn exchange(reason: u64, module_id: u32) -> (u64, u64, u64, u64) {
	ecall_aux::enclave_call_suspend(((module_id as u64) << 32) | reason);
	ecall_aux::enclave_call_module_result()
}
