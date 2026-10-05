/*
 * M -> S -> U 测试固件链接脚本.
 *
 * 单一汇编文件 (u_mode_run_fib.s) 依次完成 M 模式初始化, S 模式初始化与
 * U 模式进程执行, 三者的代码与数据同处一个文件, 但按特权级分成两组段:
 *
 *   S 组 (.stext/.srodata/.sdata/.sbss): 0x8000_0000 起
 *   U 组 (.utext/.urodata/.udata/.ubss): 0x8001_0000 起
 *
 * 之所以必须分组, 是因为 S 模式取指永不许可落在 U=1 的页上
 * (RISC-V §4.3.2 基础规则, 与 sstatus.SUM 无关): 两组代码要落在权限位不同的
 * 页上, 共用一页会使 S 模式在启用 Sv39 之后无法取指. 固件的页表按下面两个
 * 地址建立映射, 故这两处起始地址与页表条目必须一致:
 *
 *   VA 0x8000_0000–0x8000_FFFF  R+W+X    (16 页, S 组)
 *   VA 0x8001_0000–0x8001_1FFF  R+W+X+U  ( 2 页, U 组)
 *
 * 两个 ASSERT 分别锁定 S 组不越过 0x8001_0000, 与 U 组不超过映射的两页.
 *
 * 输出入口: _start (u_mode_run_fib.s 定义)
 */

OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS {
    . = 0x80000000;

    .tohost : ALIGN(0x1000) {
        *(.tohost)
    }

    . += 0x1000;

    .text : ALIGN(0x1000) {
        __TEXT_START__ = .;
        *(.stext .stext.*)
        __TEXT_END__ = .;
    }

    .rodata : ALIGN(8) {
        *(.srodata .srodata.*)
    }

    .data : ALIGN(8) {
        *(.sdata .sdata.*)
    }

    .bss : ALIGN(0x1000) {
        __SBSS_START__ = .;
        *(.sbss .sbss.*)
        __SBSS_END__ = .;
    }

    ASSERT(. <= 0x80010000,
           "S 组段超出 0x8000_FFFF: 页表只把 0x8000_0000 起的 16 页映射为 S 模式可执行")

    . = 0x80010000;

    .utext : ALIGN(0x1000) {
        __UTEXT_START__ = .;
        *(.utext .utext.*)
        __UTEXT_END__ = .;
    }

    .urodata : ALIGN(8) {
        *(.urodata .urodata.*)
    }

    .udata : ALIGN(8) {
        *(.udata .udata.*)
    }

    .ubss : ALIGN(8) {
        __UBSS_START__ = .;
        *(.ubss .ubss.*)
        __UBSS_END__ = .;
    }

    ASSERT(. <= 0x80012000,
           "U 组段超出 0x8001_1FFF: 页表只把 0x8001_0000 起的 2 页映射为 U 模式可访问")

    _end = .;
}
