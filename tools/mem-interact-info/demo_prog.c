// demo_prog.c — MemInteractInfoPass 的演示载荷
//
// 覆盖 pass 可识别的典型分配/释放类别, 用于端到端验证:
//   make demo    # 构建插件 -> 编译本文件 -> opt 插桩 -> 运行打印汇总
//
// 注意: mmap 必须同时指定 MAP_PRIVATE|MAP_ANONYMOUS (0x22 = 34), 否则
// fd=-1 时映射失败 (mmap 返回 MAP_FAILED), munmap 分支不会执行, 汇总中
// 缺少 munmap 行属于演示载荷自身的错误, 而非插桩遗漏。

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

int main(void) {
    void *a = malloc(100);
    void *b = calloc(10, 20);
    a = realloc(a, 300);
    memset(a, 0, 300);
    free(a);
    free(b);

    void *m = mmap(0, 8192, 3, 34, -1, 0); /* PROT_READ|WRITE, MAP_PRIVATE|ANONYMOUS */
    if (m != (void *)-1)
        munmap(m, 8192);
    return 0;
}
