# 适配 Rust S-mode 运行时环境:
#   - Rust 运行时已将 ELF 段加载到 U-mode 地址空间
#   - 入口 _start, 设好栈后调 main(argc, argv)
#   - main 返回后调 _exit()
#
# 编译:
#   riscv64-unknown-elf-gcc -c crt0.S -o crt0.o

.section .text.init, "ax", @progbits
.globl _start
.type _start, @function

_start:
    # argc/argv 由 Rust 运行时通过 a0/a1 传入
    # 设置栈指针 (链接脚本定义 _stack 或使用默认地址)
    la      sp, _stack

    # 调 main(argc, argv)
    call    main

    # main 返回 → exit
    mv      a0, a0            # main 返回值传入 a0
    # 如果 main 返回 0, a0 已经是
    tail    _exit

.section .bss
.align 4
.space 4096
_stack:
