# MemInteractInfoPass — LLVM IR 内存分配函数插桩

统计一个载荷自身调用内存分配/释放函数的次数与字节数, 用于量化飞地 (enclave)
运行期堆内存交互强度, 指导 brk/mmap 按需扩容策略的调优。

## 组成

| 文件 | 作用 |
|------|------|
| `mem_interact_info.cpp` | LLVM new pass manager 插件, 注册两个 pass: 计数 + 退出点报告 |
| `mem_interact_runtime.c` | 运行时计数库, 累积次数/字节数, 提供 `__mem_interact_report(reason)` 打印汇总 |
| `demo_prog.c` | 演示载荷, 覆盖 6 类典型分配/释放 |
| `Makefile` | 构建插件 + 端到端演示 |

## 识别的类别 (kind 顺序与 `g_names` 严格一致)

`malloc` `calloc` `realloc` `free` `aligned_alloc` `posix_memalign` `mmap`
`munmap` `brk` `sbrk` `__rust_alloc` `__rust_alloc_zeroed` `__rust_realloc`
`__rust_dealloc` `operator new` `operator new[]` `operator delete`
`operator delete[]`

字节数语义: `calloc` 记 `n*size`; `free`/`delete`/`brk` 记 0 (仅计数);
`__rust_realloc` 记 `new_size`; C++ 对齐重载按 `_Znwm`/`_Znam`/`_ZdlPv`/`_ZdaPv`
前缀匹配。

## 构建与演示

```bash
cd tools/mem-interact-info
make          # 构建 libmem_interact_info.so
make demo     # 端到端: 插桩 demo_prog.c -> 链接运行时 -> 运行打印汇总
```

预期输出首行 `exit=return`, 随后 `malloc/calloc/realloc/free/mmap/munmap` 六行各带 `count` 与 `bytes`。

## 两段式插桩流程 (系统 clang 19 不支持 -fpasses=)

系统 `clang`/`opt` 为 LLVM 19; `-fpasses=` 需 clang 20+, 故必须分两步:

```bash
# 1. 先产出 LLVM IR (以本机为例; 交叉编译见下)
clang -S -emit-llvm -O0 prog.c -o prog.ll

# 2. 经 opt 加载插件, 依次跑两个 pass (计数 + 退出点报告)
opt -load-pass-plugin=./libmem_interact_info.so \
    -passes=mem-interact-info,mem-interact-report \
    prog.ll -S -o prog_instr.ll

# 3. 链接运行时 + 被插桩 IR
clang prog_instr.ll mem_interact_runtime.c -o prog
```

注意 `-load-pass-plugin` 需以 `./` 前缀 (或绝对路径) 指定, 否则 `dlopen` 不在
当前目录查找。

## 交叉编译到 riscv64-musl 载荷

对飞地载荷 (如 `fn_apps/cfrac`) 统计, 前两步加 `--target` 即可; 第三步仍用
交叉 `clang` 链接 musl 静态库:

```bash
# 1. 载荷源码 -> LLVM IR
clang --target=riscv64-linux-musl -march=rv64gc -mabi=lp64d \
    -O0 -S -emit-llvm cfrac.c -o cfrac.ll

# 2. opt 插桩 (IR 与目标无关, 本机 opt 即可; 同样跑两个 pass)
opt -load-pass-plugin=./libmem_interact_info.so \
    -passes=mem-interact-info,mem-interact-report \
    cfrac.ll -S -o cfrac_instr.ll

# 3. 交叉链接 (runtime.c 一并交叉编译; 具体库/头路径见 fn_apps/config.mk)
#    注意 runtime 也需 -nostdinc + musl -isystem, 否则其 <stdio.h> 命中宿主
#    glibc 的 __float128 (riscv64 不支持) 报错。
clang --target=riscv64-linux-musl -march=rv64gc -mabi=lp64d \
    -static -nostdinc -fuse-ld=lld \
    -isystem vendor/musl/include -isystem vendor/musl/arch/riscv64 \
    -isystem vendor/musl/arch/generic -isystem vendor/musl/obj/include \
    -Bbsp/musl-gc-sysroot/lib -Lbsp/musl-gc-sysroot/lib \
    -nodefaultlibs -lc -lgcc cfrac_instr.ll mem_interact_runtime.c -o cfrac_instr

# 4. qemu 运行验证
qemu-riscv64 cfrac_instr
```

## 设计要点

- **两个 pass 各司其职**: `mem-interact-info` 只做分配函数计数;
  `mem-interact-report` 只找退出点 (`main` 的 `ret` + 显式 `exit`/`_exit`/
  `abort` 调用) 并插入报告。拆分后可单独跑任一 pass (如只计数、报告由别处触发)。
- **报告不依赖 `atexit`**: 报告由 pass 在退出点直接插入, 故 `main` 正常返回、
  显式 `exit`、甚至 `abort` (经 `fflush(stdout)` 保证输出) 都能打印汇总, 无需
  依赖 `.init_array` 的 constructor 或 `atexit` 注册。
- **只插桩载荷自身**: pass 经 opt 作用于载荷 IR, musl 与 `mem_interact_runtime.c`
  不注入, 故不会统计 musl 内部 malloc, 也不会因 runtime 的 `printf` 内部分配而递归。
- **计数用结构体**: 每类分配/释放各持一个 `struct MemInteractEntry { uint64_t count; uint64_t bytes; }`,
  组成 `g_entries[MAX_KIND]`, 取代旧的 `g_count/g_bytes` 并行数组。
- **退出原因标注**: `__mem_interact_report(reason)` 的 reason 为 `0=return / 1=exit /
  2=abort`, 由 pass 在退出点插入, 输出首行 `exit=<reason>` 标注载荷的自然退出方式。
- **kind 双向契约**: `mem_interact_info.cpp` 的 `K_*` 枚举与
  `mem_interact_runtime.c` 的 `g_names` 顺序必须逐位一致; 改一处需同步另一处。
- **静态识别限制**: 仅识别直接调用 (`CallBase::getCalledFunction`), 间接函数指针
  调用 (如经 `dlsym` 取 malloc 再调用) 不插桩。
