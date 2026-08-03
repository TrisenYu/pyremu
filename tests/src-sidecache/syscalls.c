/*
 * 本文件将这些函数桥接到 Rust 运行时的 Linux rv64 ecall 接口.
 *
 * 提供的符号:
 *   write, read, _exit, sbrk  — 核心, 桥接到 ecall syscall
 *   close, lseek, isatty      — 桩, 返回安全默认值
 *
 * 编译为 .o 后与 picolibc 的 libc.a 链接.
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* ---- RISC-V Linux syscall 号 (与 Rust 运行时对齐) ---- */
#define SYS_write 64
#define SYS_read  63
#define SYS_exit  93
#define SYS_brk	  214

/* ---- 原始 ecall syscall ---- */

static long syscall3(long num, long a0, long a1, long a2) {
	register long r_a7 asm("a7") = num;
	register long r_a0 asm("a0") = a0;
	register long r_a1 asm("a1") = a1;
	register long r_a2 asm("a2") = a2;
	__asm__ volatile("ecall" : "+r"(r_a0) : "r"(r_a1), "r"(r_a2), "r"(r_a7) : "memory");
	return r_a0;
}

/* ---- 文件 I/O ---- */

ssize_t write(int fd, const void *buf, size_t len) {
	if (fd != 1 && fd != 2) {
		return -1;
	}
	/* Rust 运行时 write handler 逐字节写 UART, 最多 65536 字节 */
	if (len > 65536) {
		len = 65536;
	}
	return syscall3(SYS_write, fd, (long)buf, (long)len);
}

ssize_t read(int fd, void *buf, size_t len) {
	if (fd != 0) {
		return -1;
	}
	return syscall3(SYS_read, fd, (long)buf, (long)len);
}

/* ---- 进程控制 ---- */

__attribute__((noreturn)) void _exit(int code) {
	syscall3(SYS_exit, code, 0, 0);
	__builtin_unreachable();
}

/* ---- 堆 (picolibc malloc 用 sbrk) ---- */

void *sbrk(intptr_t increment) {
	long ret = syscall3(SYS_brk, increment, 0, 0);
	if (ret < 0) {
		return (void *)-1;
	}
	return (void *)ret;
}

/* ---- 桩: 不实现但返回安全默认值的函数 ---- */

int close(int fd) {
	(void)fd;
	return 0;
}

off_t lseek(int fd, off_t offset, int whence) {
	(void)fd;
	(void)offset;
	(void)whence;
	return 0;
}

int isatty(int fd) {
	return (fd >= 0 && fd <= 2) ? 1 : 0;
}

/* picolibc 某些配置可能需要 fstat */
int fstat(int fd, struct stat *st) {
	(void)fd;
	if (st) {
		/* 最小填充: 告诉 libc stdout 是字符设备, 行缓冲模式 */
		st->st_mode	   = (fd <= 2) ? 0020000 /* S_IFCHR */ : 0100000 /* S_IFREG */;
		st->st_size	   = 0;
		st->st_blksize = 0;
	}
	return 0;
}
