#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 测试定义开始区域 */
#include <riscv_tee.h>
/* 测试定义结束区域 */

int main(int argc, char *argv[]) {
	unsigned long *tval = (unsigned long *)(0x12345678), a = 0x31415926, b = 0xFF, c,
				  *tptr = &a;
	char *ptr_str		= (char *)(0xFFFF000023456789);
	scanf("%lu", &c);
	for (unsigned long i = 0; i < c; i++) {
		a ^= b << (i & 0xFF);
	}
	/* 测试调用开始区域 */
#define decry(x, y) (__riscv_midecry(x, y))
#define encry(x, y) (__riscv_miencry(x, y))
#define hashm(x, y) (__riscv_mihashm(x, y))
	unsigned long tmp = 2 - decry(tval, 0xff) + encry(tptr, 0x10)
							- encry(tval, 0x0a) * encry(tptr, 0x31415926)
							+ hashm(tval, 0x2345678abcdeaf2bull) / encry(tval, a)
							+ decry(tptr, a ^ 0x2718281828459045)
							+ (hashm(tval, a & 0x27182818) >> 3)
							+ (decry(tval, a * 0x2718) << 4) + (~decry(tval, a + 0x27))
						^ (!decry(tval, a >> 0x2));
#undef decry
#undef encry
#undef hashm
	asm volatile("csrr %[res], mworld" : [res] "=r"(c)::"memory");
	printf(
		"%lu %lu\n",
		tmp,
		__builtin_riscv_mihashm(ptr_str, __riscv_midecry(tptr, a >> 4) + c));
	/* 测试调用结束区域 */
	return 0;
}