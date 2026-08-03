/* hello_payload.c — picolibc 飞地 payload 测试
 *
 * 验证 syscalls.c 桥接 + picolibc libc 链接链:
 *   printf -> puts -> fputs -> write -> ecall write(64) -> Rust 运行时 -> UART
 *   malloc -> sbrk -> ecall brk(214) -> Rust 运行时
 *
 * Compiled by Makefile
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
	puts("[payload] Hello from picolibc enclave payload!");
	printf("[payload] argc=%d\n", argc);
	if (argv && argv[0]) {
		printf("[payload] argv[0]=%s\n", argv[0]);
	}

	/* 测试 malloc in TEE */
	char *buf = malloc(128);
	if (buf) {
		strcpy(buf, "malloc-works");
		printf("[payload] malloc(128)='%s'\n", buf);
		free(buf);
	} else {
		puts("[payload] malloc failed");
	}

	puts("[payload] done");
	return 0;
}
