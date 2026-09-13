// mem_interact_runtime.c — MemInteractInfoPass 的运行时计数库
//
// 提供 __mem_interact_trace(kind, size) 与 __mem_interact_report(reason)。trace 累积
// 各类别调用次数与字节数; report 打印汇总, 由 pass 在程序退出点 (main 返回 /
// exit / abort) 插入对其的调用, 不依赖 atexit。reason 区分退出路径 (见 MemExitReason),
// 以便标注载荷的自然退出方式 (返回/exit/abort)。本文件编译时不注入 pass (否则
// printf 内部 malloc 会递归), 故统计范围仅限被插桩的载荷自身调用, 不含 musl 与
// 本库内部分配。
//
// g_names / g_exit_names 的顺序必须与 mem_interact_info.cpp 的 K_* / ExitReason 枚举
// 严格一致。

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_KIND 32

// 单个分配/释放类别的计数: 调用次数 + 申请字节总量。成员均为 uint64_t。
struct MemInteractEntry {
    uint64_t count; // 调用次数
    uint64_t bytes; // 申请的字节总量
};

// 每类分配/释放函数各持有一个 entry, 按 kind 索引 (与 pass 的 K_* 枚举一致)。
static struct MemInteractEntry g_entries[MAX_KIND];

// 退出原因: 标注载荷自然终止的方式。顺序与 mem_interact_info.cpp 的 ExitReason 一致。
typedef enum {
    MEM_EXIT_RETURN = 0, // main 正常返回
    MEM_EXIT_EXIT = 1,   // 显式 exit/_exit/quick_exit
    MEM_EXIT_ABORT = 2,  // abort
} MemExitReason;

static const char *const g_names[] = {
    "malloc",
    "calloc",
    "realloc",
    "free",
    "aligned_alloc",
    "posix_memalign",
    "mmap",
    "munmap",
    "brk",
    "sbrk",
    "__rust_alloc",
    "__rust_alloc_zeroed",
    "__rust_realloc",
    "__rust_dealloc",
    "operator new",
    "operator new[]",
    "operator delete",
    "operator delete[]",
};

static const char *const g_exit_names[] = {
    "return",
    "exit",
    "abort",
};

void __mem_interact_trace(int64_t kind, int64_t size) {
    if (kind < 0 || kind >= MAX_KIND) {
        return;
    }
    g_entries[kind].count++;
    if (size > 0) {
        g_entries[kind].bytes += (uint64_t)size;
    }
}

void __mem_interact_report(uint64_t reason) {
    size_t n_exit = sizeof(g_exit_names) / sizeof(g_exit_names[0]);
    const char *exit_name = reason < n_exit ? g_exit_names[reason] : "?";
    printf("<mem-interact-info> exit=%s\n", exit_name);

    size_t n_names = sizeof(g_names) / sizeof(g_names[0]);
    int any = 0;
    for (int i = 0; i < MAX_KIND; i++) {
        if (!g_entries[i].count) {
            continue;
        }
        any = 1;
        const char *name = (size_t)i < n_names ? g_names[i] : "?";
        printf("<mem-interact-info> %-16s count=%lu bytes=%lu\n",
               name, (unsigned long)g_entries[i].count, (unsigned long)g_entries[i].bytes);
    }
    if (!any) {
        puts("<mem-interact-info> (no allocations)");
    }
    fflush(stdout);
}
