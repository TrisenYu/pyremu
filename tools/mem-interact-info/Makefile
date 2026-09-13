# tools/mem-interact-info/Makefile — 构建 MemInteractInfoPass 插件与演示
#
# 插件: libmem_interact_info.so (LLVM new pass manager 插件, 经 opt 加载)
# 演示: 编译 demo_prog.c -> opt 插桩 -> 链接运行时 -> 运行打印汇总
#
# 注意: 系统 clang 19 不支持 -fpasses= (clang 20+ 才有), 故必须两段式:
#   clang -S -emit-llvm ...        # 先产出 LLVM IR
#   opt -load-pass-plugin=... -passes=mem-interact-info   # 再经 opt 插桩
#   clang instrumented.ll runtime.c  # 最后链接成本机可执行

LLVM_CONFIG := llvm-config
CXX         := clang++
CC          := clang
OPT         := opt

# --cxxflags 已含 -std=c++17 -fno-exceptions 与头文件路径; 追加共享库与 PIC。
CXXFLAGS := -shared -fPIC $(shell $(LLVM_CONFIG) --cxxflags)

PLUGIN := libmem_interact_info.so
DEMO_BIN := demo_prog.exe

all: $(PLUGIN)

$(PLUGIN): mem_interact_info.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# 端到端演示: 插桩 -> 链接 -> 运行, 汇总应含 malloc/calloc/realloc/free/mmap/munmap。
$(DEMO_BIN): $(PLUGIN) demo_prog.c mem_interact_runtime.c
	$(CC) -S -emit-llvm -O0 demo_prog.c -o demo_prog.ll
	$(OPT) -load-pass-plugin=./$(PLUGIN) -passes=mem-interact-info,mem-interact-report demo_prog.ll -S -o demo_prog_instr.ll
	$(CC) demo_prog_instr.ll mem_interact_runtime.c -o $(DEMO_BIN)

demo: $(DEMO_BIN)
	./$(DEMO_BIN)

clean:
	rm -f $(PLUGIN) $(DEMO_BIN) demo_prog.ll demo_prog_instr.ll

.PHONY: all demo clean
