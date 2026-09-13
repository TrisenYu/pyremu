// mem_interact_info.cpp — 内存分配/释放插桩 + 退出点报告 (LLVM new pass manager 插件)
//
// 本插件注册两个 pass, 各司其职, 可单独或组合使用:
//   1. mem-interact-info   — 在每个匹配的 malloc/calloc/realloc/free/mmap/brk/
//                            __rust_alloc/operator new 等调用前插入
//                            __mem_interact_trace(kind, size), 供运行时累积调用
//                            次数与字节数。
//   2. mem-interact-report — 找到载荷的退出点 (main 的 ret / 显式 exit 族调用),
//                            在其前插入 __mem_interact_report(reason), 退出时打印
//                            汇总; reason 标注返回/exit/abort 三种退出路径。
//
// 只对载荷自身源码生效 (经 opt 注入), 不重编译 musl/运行时, 故不会统计 musl
// 内部 malloc (避免递归与噪声), 也不统计计数运行时自身的 printf 内部分配。
// 类别 id 必须与 mem_interact_runtime.c 的 g_names 顺序一致 (见 K_* 枚举);
// 退出原因与 g_exit_names 一致 (见 ExitReason 枚举)。
//
// 构建: 见同目录 Makefile (经系统 llvm-config/clang++ 编译共享插件)。

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {

// 分配类别 id —— 顺序与 mem_interact_runtime.c 的 g_names 严格一致。
enum AllocKind {
	K_MALLOC = 0,
	K_CALLOC,
	K_REALLOC,
	K_FREE,
	K_ALIGNED_ALLOC,
	K_POSIX_MEMALIGN,
	K_MMAP,
	K_MUNMAP,
	K_BRK,
	K_SBRK,
	K_RUST_ALLOC,
	K_RUST_ALLOC_ZEROED,
	K_RUST_REALLOC,
	K_RUST_DEALLOC,
	K_OP_NEW,
	K_OP_NEW_ARRAY,
	K_OP_DELETE,
	K_OP_DELETE_ARRAY,
	K_MAX
};

// size 参数下标 (-1 表示无 size 参数, 计数但字节记 0)。
static const int kSizeArg[K_MAX] = {
	0,	// malloc(size)
	0,	// calloc(n, size) —— 特殊: n*size, 见下方
	1,	// realloc(ptr, size)
	-1, // free(ptr)
	1,	// aligned_alloc(align, size)
	2,	// posix_memalign(memptr, align, size)
	1,	// mmap(addr, length, ...)
	1,	// munmap(addr, length)
	-1, // brk(addr) —— 增量不可静态求得, 只计数
	0,	// sbrk(increment)
	0,	// __rust_alloc(size, align)
	0,	// __rust_alloc_zeroed(size, align)
	3,	// __rust_realloc(ptr, old_size, align, new_size)
	1,	// __rust_dealloc(ptr, size, align)
	0,	// operator new(size)
	0,	// operator new[](size)
	-1, // operator delete(void*)
	-1, // operator delete[](void*)
};

static bool hasPrefix(StringRef s, StringRef prefix) {
	// clang-format off
	return s.size() >= prefix.size() &&
		s.substr(0, prefix.size()) == prefix;
	// clang-format on
}

// 按被调函数名匹配, 返回类别 id; 不匹配返回 -1。
static int matchAlloc(StringRef name) {
	static const char *const kNames[K_MAX] = {
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
	};
	for (int i = 0; i < K_MAX; i++) {
		if (!kNames[i]) {
			continue; // C++ operator 走下方前缀匹配
		}
		if (name == kNames[i]) {
			return i;
		}
	}
	// C++ operator new/delete (含对齐重载 _ZnwmSt11align_val_t 等), 按前缀匹配。
	if (hasPrefix(name, "_Znwm")) {
		return K_OP_NEW;
	}
	if (hasPrefix(name, "_Znam")) {
		return K_OP_NEW_ARRAY;
	}
	if (hasPrefix(name, "_ZdlPv")) {
		return K_OP_DELETE;
	}
	if (hasPrefix(name, "_ZdaPv")) {
		return K_OP_DELETE_ARRAY;
	}
	return -1;
}

// 退出原因 —— 与 mem_interact_runtime.c 的 g_exit_names 顺序一致。
// 由本 pass 在退出点插入: 0=return, 1=exit, 2=abort。
enum ExitReason {
	EXIT_RETURN = 0, // main 正常返回
	EXIT_EXIT = 1,   // 显式 exit/_exit/quick_exit
	EXIT_ABORT = 2,  // abort
};

// 显式退出函数 -> 退出原因; 非退出调用返回 -1。
static int exitReason(StringRef name) {
	if (name == "abort") {
		return EXIT_ABORT;
	}
	if (name == "exit" || name == "_exit" || name == "_Exit" ||
		name == "quick_exit") {
		return EXIT_EXIT;
	}
	return -1;
}

// ---------------------------------------------------------------
//  pass 1: mem-interact-info —— 分配函数计数
// ---------------------------------------------------------------

struct MemInteractInfoPass : PassInfoMixin<MemInteractInfoPass> {
	PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
		LLVMContext &Ctx = M.getContext();
		IntegerType *I64 = Type::getInt64Ty(Ctx);

		// 声明外部 trace 函数:
		// void __mem_interact_trace(i64 kind, i64 size)。
		// 定义由 mem_interact_runtime.c 提供, 链接时解析。
		// clang-format off
		FunctionCallee traceFn = M.getOrInsertFunction(
			"__mem_interact_trace",
			FunctionType::get(
				Type::getVoidTy(Ctx),
				{I64, I64},
				false
			)
		);
		// clang-format on
		bool changed = false;
		for (Function &F : M) {
			if (F.isDeclaration()) {
				continue;
			}
			for (BasicBlock &BB : F) {
				for (Instruction &I : make_early_inc_range(BB)) {
					auto *call = dyn_cast<CallBase>(&I);
					if (!call) {
						continue;
					}
					Function *callee = call->getCalledFunction();
					if (!callee) {
						// 静态分析，跳过不好识别的间接调用
						continue;
					}
					int kind = matchAlloc(callee->getName());
					if (kind < 0) {
						continue;
					}

					Value *size = nullptr;
					if (kind == K_CALLOC) {
						// calloc(n, size) -> n * size
						IRBuilder<> B(call);
						size = B.CreateMul(
							call->getArgOperand(0),
							call->getArgOperand(1),
							"alloc_size"
						);
					} else {
						int sizeArg = kSizeArg[kind];
						if (sizeArg >= 0 && (unsigned)sizeArg < call->arg_size()) {
							size = call->getArgOperand(sizeArg);
						}
					}
					if (!size) {
						size = ConstantInt::get(I64, 0);
					}

					IRBuilder<> B(call);
					B.CreateCall(traceFn, {B.getInt64(kind), size});
					changed = true;
				}
			}
		}
		return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
	}
};

// ---------------------------------------------------------------
//  pass 2: mem-interact-report —— 退出点插入报告
// ---------------------------------------------------------------

struct MemInteractReportPass : PassInfoMixin<MemInteractReportPass> {
	PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
		LLVMContext &Ctx = M.getContext();

		// 声明外部报告函数: void __mem_interact_report(uint64_t reason)。
		// clang-format off
		FunctionCallee reportFn = M.getOrInsertFunction(
			"__mem_interact_report",
			FunctionType::get(
				Type::getVoidTy(Ctx),
				{Type::getInt64Ty(Ctx)},
				false
			)
		);
		// clang-format on

		bool changed = false;
		for (Function &F : M) {
			if (F.isDeclaration()) {
				continue;
			}
			bool isMain = F.getName() == "main";
			for (BasicBlock &BB : F) {
				for (Instruction &I : make_early_inc_range(BB)) {
					// 退出点 1: main 的 ret (正常返回路径)。
					if (auto *ret = dyn_cast<ReturnInst>(&I)) {
						if (isMain) {
							IRBuilder<> B(ret);
							B.CreateCall(reportFn, {B.getInt64(EXIT_RETURN)});
							changed = true;
						}
						continue;
					}
					auto *call = dyn_cast<CallBase>(&I);
					if (!call) {
						continue;
					}
					Function *callee = call->getCalledFunction();
					if (!callee) {
						continue;
					}
					// 退出点 2: 显式 exit/_exit/abort 等 (调用后不再返回)。
					int reason = exitReason(callee->getName());
					if (reason >= 0) {
						IRBuilder<> B(call);
						B.CreateCall(reportFn, {B.getInt64(reason)});
						changed = true;
					}
				}
			}
		}
		return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
	}
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
	// clang-format off
	return {
		LLVM_PLUGIN_API_VERSION,
		"MemInteract",
		"v0.2",
		[](PassBuilder &PB) {
			PB.registerPipelineParsingCallback(
				[](StringRef Name,
					ModulePassManager &MPM,
					ArrayRef<PassBuilder::PipelineElement>) {
					if (Name == "mem-interact-info") {
						MPM.addPass(MemInteractInfoPass());
						return true;
					}
					if (Name == "mem-interact-report") {
						MPM.addPass(MemInteractReportPass());
						return true;
					}
					return false;
				});
		}
	};
	// clang-format on
}
