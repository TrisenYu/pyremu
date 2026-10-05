/* zero_argv_main.c — 以 argc=0 且 argv=NULL 调用 cfrac 的 main
 *
 * 宿主经 TEE_IOC_ENTER 转入载荷时可不传参数 (argc 为 0, argv 指针为 0), 载荷在飞地
 * 内即以此为入参启动; 回归程序的池填充阶段正是这样转入载荷的. 本夹具在 qemu 下复现
 * 这一组入参, 使该路径可在宿主侧回归.
 *
 * 载荷的 main 由构建期以 -Dmain=cfrac_main 改名 (见 ../Makefile), 避免与本文件的
 * main 重名.
 *
 * 断言见 ../Makefile: 打印内置合数的分解式, 退出码为 0, 不出现 usage 行. 修复前这一
 * 组入参落入 usage 分支, 并因 progName 取自 *argv (此时为 NULL) 而打印 "(null)".
 *
 * SPDX-License-Identifier: MIT
 */

extern int cfrac_main(int argc, char **argv);

int main(void) { return cfrac_main(0, (char **)0); }
