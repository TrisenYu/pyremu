// easy_kern.c — 无 Linux 的 S 模式极简 host 示例
//
// 用途: 在仿真器上快速验证飞地载荷真正进入运行时与 main(),
// 免去完整 Linux 引导 (约 60 至 130 倍慢于真实时间, 难以迭代).
//
// 运行位置与方式:
//   OpenSBI fw_jump (M 模式) 完成初始化后 mret 到 0x80200000, easy_kern
//   以 S 模式 (satp=Bare, 虚拟地址即物理地址, mdid=0 即 host) 从 entry.S
//   的 _start 开始执行, 随后直接经 SBI ecall 驱动飞地生命周期:
//
//     CREATE(id), 随后 ENTER(cfrac 载荷); 飞地自愿让出时反复 RESUME, 直至载荷自行退出
//
// 载荷 (默认 cfrac 静态 ELF) 由 payload_embed.S 以 .incbin 嵌入本二进制,
// ENTER 时经 a3/a4 把载荷指针与大小传给 M 模式, 由 M 模式拷入飞地内存.
// 裸机 freestanding, 不引入任何头文件; riscv64 下 long 即 64 位.

// 飞地 SBI 扩展号与函数号 (与 bsp/tee_aux_tools/linux-driver / bsp/sittim 一致)
#define ENCLAVE_EXT_ID     0x20221222UL
#define ENCLAVE_CREATE     400UL
#define ENCLAVE_ENTER      401UL
#define ENCLAVE_SHUTDOWN   403UL
#define ENCLAVE_RESUME     405UL

// SBI 旧版 console putchar 扩展 (与运行时 sbi_putchar 约定一致)
#define SBI_LEGACY_PUTCHAR_EXT 1UL

typedef unsigned long u64;
typedef long           s64;

// 由 payload_embed.S 提供: 内嵌载荷的起止地址
extern char payload_blob[];
extern char payload_blob_end[];

// 经 SBI 旧版 putchar 输出单个字符到串口.
static void putc(char c) {
	register u64 a0 asm("a0") = (u64)(unsigned char)c;
	register u64 a6 asm("a6") = 0;
	register u64 a7 asm("a7") = SBI_LEGACY_PUTCHAR_EXT;
	asm volatile("ecall" : "+r"(a0) : "r"(a6), "r"(a7) : "memory");
}

// 输出字符串 (不含换行).
static void puts(const char *s) {
	for (; *s; s++)
		putc(*s);
}

// 以十六进制输出 64 位值 (带 0x 前缀, 便于与符号地址对照).
static void puthex(u64 v) {
	static const char digits[] = "0123456789abcdef";
	int i;
	putc('0');
	putc('x');
	for (i = 60; i >= 0; i -= 4)
		putc(digits[(v >> i) & 0xf]);
}

// 以十进制输出无符号 64 位值.
static void putdec(u64 v) {
	char buf[20];
	int i = 0;
	if (v == 0) {
		putc('0');
		return;
	}
	while (v > 0) {
		buf[i++] = (char)('0' + v % 10);
		v /= 10;
	}
	while (i > 0)
		putc(buf[--i]);
}

// 飞地扩展 ecall: a7=扩展号, a6=函数号, a0 至 a4 为参数.
// 返回 ecall 返回后 M 模式写入 a0 的值 (语义见各 handler).
static u64 enclave_ecall(u64 func, u64 a0, u64 a1, u64 a2, u64 a3, u64 a4) {
	register u64 r_a0 asm("a0") = a0;
	register u64 r_a1 asm("a1") = a1;
	register u64 r_a2 asm("a2") = a2;
	register u64 r_a3 asm("a3") = a3;
	register u64 r_a4 asm("a4") = a4;
	register u64 r_a6 asm("a6") = func;
	register u64 r_a7 asm("a7") = ENCLAVE_EXT_ID;
	asm volatile("ecall"
		     : "+r"(r_a0)
		     : "r"(r_a1), "r"(r_a2), "r"(r_a3), "r"(r_a4),
		       "r"(r_a6), "r"(r_a7)
		     : "memory");
	return r_a0;
}

// 创建飞地并返回其 ID. 阻塞至运行时 boot suspend 让出才返回;
// 失败时 M 模式框架在 a0 填入负错误码.
static u64 enclave_create(u64 mgmt_token) {
	return enclave_ecall(ENCLAVE_CREATE, mgmt_token, 0, 0, 0, 0);
}

// 进入飞地, 提供载荷与 argv. 语义同 enclave_ecall 返回约定:
// 正数(飞地 ID)为飞地自愿让出, 0 为飞地自行退出, 负数为 SBI 错误.
static s64 enclave_enter(u64 id, u64 argc, u64 argv, u64 payload, u64 size) {
	return (s64)enclave_ecall(ENCLAVE_ENTER, id, argc, argv, payload, size);
}

// 恢复挂起的飞地, 返回约定同 enclave_enter.
static s64 enclave_resume(u64 id) {
	return (s64)enclave_ecall(ENCLAVE_RESUME, id, 0, 0, 0, 0);
}

// 反复进入并在飞地自愿让出后恢复, 直到飞地自行退出 (返回 0) 或 SBI 错误 (负值).
static s64 run_enclave(u64 id, u64 argc, u64 argv, u64 payload, u64 size) {
	s64 r;
	for (;;) {
		r = enclave_enter(id, argc, argv, payload, size);
		puts("[easy_kern] enter returned "); putdec((u64)r); putc('\n');
		if (r <= 0)
			return r;
		// 飞地自愿让出: 持续恢复, 直至退出或出错
		do {
			r = enclave_resume(id);
			puts("[easy_kern] resume returned "); putdec((u64)r); putc('\n');
		} while (r > 0);
		return r;
	}
}

int main(void) {
	// 载荷 argv: 仅程序名, 令 cfrac 走内置小合数快速自检路径
	static u64 argv_slots[1];
	static const char argv0[] = "cfrac";
	u64 payload_size;
	u64 id;
	s64 ret;

	argv_slots[0] = (u64)argv0;
	payload_size = (u64)(payload_blob_end - payload_blob);

	puts("[easy_kern] booted at 0x80200000, creating enclave\n");
	puts("[easy_kern] payload size = "); putdec(payload_size); putc('\n');

	// 第一步: 创建飞地 (阻塞至运行时 boot suspend 让出, 返回飞地 ID)
	id = enclave_create(0);
	puts("[easy_kern] create returned id = "); putdec(id); putc('\n');
	if ((s64)id <= 0) {
		puts("[easy_kern] create failed, stop\n");
		return 1;
	}

	// 第二步: 进入飞地并运行载荷, 飞地自愿让出时自动恢复, 直至退出
	ret = run_enclave(id, 1, (u64)argv_slots, (u64)payload_blob, payload_size);
	puts("[easy_kern] enclave exited with ret = "); putdec((u64)ret); putc('\n');
	puts("[easy_kern] demo done, entering wfi\n");
	return 0;
}
