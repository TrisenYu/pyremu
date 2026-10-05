/* json_cpp — 以 json_calc 同一输入集为输入的 nlohmann/json 解析 churn 载荷
 *
 * 与同目录 main.c (cJSON) 共用构建期生成的内嵌输入 json_config.inc: input/ 下
 * 每份真实第三方配置各保留 pretty 与 minified 两种形态, 一并编码进 json_inputs[],
 * 故载体内无文件系统依赖, 两种形态仅字符串外的空白不同, 解析所得结果一致.
 *
 * 与 cJSON 的差异在于分配粒度: nlohmann::json 的每个节点是一个 basic_json 对象
 * (内含联合体与类型标记), 对象成员另经有序映射承载、数组经动态数组承载, 键名与
 * 字符串值各持一份 std::string。同一份文档的分配次数与总字节数都明显高于 cJSON
 * 的裸节点链表, 因此本载荷与 json_calc 互补 —— 同一批文档, 不同的分配粒度与
 * 释放时序, 共同压测 musl mallocng 与飞地按需增长的堆。
 *
 * 每轮对一份文档执行 parse / dump / 再 parse / 相等比对, 全程计入 churn:
 *   parse  为每个 token 构造 basic_json 并登记到父容器;
 *   dump   经 serializer 反复追加增长内部缓冲区;
 *   比对   递归遍历两棵树。
 *
 * 除合法文档外, input/bad/ 下的语法错误文档也被内嵌 (追加在合法输入之后), 由
 * json_expect_ok 逐项标注: 对它们断言解析抛出 parse_error, 从而覆盖异常路径与
 * 栈回退中的析构 (即成功路径之外的另一种分配/回收次序)。
 *
 * 输出分三段:
 *   1. 逐项进度: 每个输入每次迭代输出一个字符, 不逐项换行 —— '.' 断言正确且通过
 *      (合法文档解析成功), '~' 断言错误且通过 (语法错误文档被拒), '!' 断言正确但
 *      失败, '#' 断言错误但失败; 每 8 个字符后隔一个空格, 每 64 个字符换一行;
 *   2. 失败明细: 有失败时逐项给出首次失败所在的轮次与原因 (异常描述或往返比对不符),
 *      不在运行途中打断进度行;
 *   3. 收尾汇总: 逐行给出各类型节点的节点数与节点自身、键名、字符串值所占的字节数,
 *      再给出分配器实测的堆占用峰值与全部输入释放后仍在用的字节数。
 * 任何一段都不打印整棵树的庞大序列化结果。运行时不接受参数 (argv 被忽略)。
 *
 * 编译: 见同目录 Makefile (riscv64gc musl 静态, libc++)。
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <elf.h>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/auxv.h>
#include <vector>

#include <nlohmann/json.hpp>

#include "json_config.inc" /* 定义 json_inputs[] (每个输入文件的 pretty + minified) */

/* 迭代次数可用编译期宏覆盖。标定依据 (qemu-riscv64, 16 份输入): 本载荷 1 轮
 * 0.20 s, 同目录 cJSON 载荷 4 轮 1.40 s (每轮 0.35 s); 本载荷单份文档的分配次数
 * 与总字节数远高于 cJSON, 故取 1 轮已给出与之相当的分配压力与运行时长。 */
#ifndef JSON_CPP_ITERS
#define JSON_CPP_ITERS 1
#endif

/* 进度字符的排版: 每 8 个字符后隔一个空格, 每 64 个字符换一行。 */
#define MARK_GROUP 8
#define MARK_LINE 64

/* 输入形态数量 (编译期常量): json_inputs 由 json_config.inc 提供, 新增输入文件
 * 只须放进 input/ 目录 (构建期按文件名排序收录), 无需改动本文件。 */
#define JSON_INPUT_COUNT (sizeof(json_inputs) / sizeof(json_inputs[0]))

/* ---- 分配计数器 ----
 * nlohmann::json 的节点、容器与字符串都经模板参数传入的分配器, 故用一个薄分配器
 * 包装 libc 的 malloc/free, 即可记下本载荷真实的堆占用。记账需要知道每块的字节数,
 * 而释放时只给出指针, 故分配时在返回的指针之前加一个头部存放字节数。 */
union alloc_header {
	std::size_t size;
	std::max_align_t align; /* 使返回的指针保持最大对齐 */
};

static std::size_t live_bytes = 0;
static std::size_t peak_bytes = 0;

template <class T>
struct counting_allocator {
	using value_type = T;

	counting_allocator() = default;
	template <class U>
	counting_allocator(const counting_allocator<U> &) {}

	T *allocate(std::size_t n) {
		union alloc_header *block = static_cast<union alloc_header *>(
			std::malloc(sizeof(union alloc_header) + n * sizeof(T)));

		if (block == nullptr) {
			throw std::bad_alloc();
		}
		block->size = n * sizeof(T);
		live_bytes += block->size;
		if (live_bytes > peak_bytes) {
			peak_bytes = live_bytes;
		}
		return reinterpret_cast<T *>(block + 1);
	}

	void deallocate(T *ptr, std::size_t) noexcept {
		union alloc_header *block = reinterpret_cast<union alloc_header *>(ptr) - 1;

		live_bytes -= block->size;
		std::free(block);
	}

	template <class U>
	bool operator==(const counting_allocator<U> &) const {
		return true;
	}
	template <class U>
	bool operator!=(const counting_allocator<U> &) const {
		return false;
	}
};

/* 字符串类型同样换成带计数的分配器, 使 dump 的内部缓冲区与各节点的字符串值一并
 * 计入。 */
using json_string = std::basic_string<char, std::char_traits<char>, counting_allocator<char>>;
using json = nlohmann::basic_json<std::map, std::vector, json_string, bool, std::int64_t,
								  std::uint64_t, double, counting_allocator,
								  nlohmann::adl_serializer, std::vector<std::uint8_t>>;

/* ---- 节点构成 ----
 * 逐类型统计节点数, 并按节点自身的粒度统计字节数: 每个节点一块 sizeof(json), 对象
 * 成员的键名与字符串值各按其长度计。有序映射的树节点与动态数组的容量冗余不属于节点
 * 自身, 由分配计数器实测的峰值体现。 */
enum node_kind { KIND_OBJ, KIND_ARR, KIND_STR, KIND_NUM, KIND_LIT, KIND_OTH, KIND_COUNT };

static const char *const kind_names[KIND_COUNT] = {
	"object", "array", "string", "number", "literal", "other",
};

struct node_stats {
	std::size_t nodes[KIND_COUNT];
	std::size_t bytes[KIND_COUNT];
};

/* 递归累计一棵树的节点构成 (根含自身)。 */
static void accumulate(const json &value, struct node_stats *stats) {
	std::size_t bytes = sizeof(json);
	enum node_kind kind;

	switch (value.type()) {
	case json::value_t::object:
		kind = KIND_OBJ;
		for (auto it = value.cbegin(); it != value.cend(); ++it) {
			bytes += it.key().size();
			accumulate(it.value(), stats);
		}
		break;
	case json::value_t::array:
		kind = KIND_ARR;
		for (const json &child : value) {
			accumulate(child, stats);
		}
		break;
	case json::value_t::string:
		kind = KIND_STR;
		bytes += value.get_ref<const json_string &>().size();
		break;
	case json::value_t::number_integer:
	case json::value_t::number_unsigned:
	case json::value_t::number_float:
		kind = KIND_NUM;
		break;
	case json::value_t::boolean:
	case json::value_t::null:
		kind = KIND_LIT;
		break;
	default:
		/* binary 与 discarded 只由显式构造产生, 解析 JSON 文本不产生两者。 */
		kind = KIND_OTH;
		break;
	}
	stats->nodes[kind]++;
	stats->bytes[kind] += bytes;
}

/* ---- 进度与失败明细 ---- */

/* 已输出的进度字符数, 决定下一个字符落在行内何处。 */
static std::size_t marks_done = 0;

/* 输出一个进度字符: 每 MARK_GROUP 个字符后隔一个空格, 每 MARK_LINE 个字符换一行,
 * 并立即输出到控制台。 */
static void emit_mark(char mark) {
	if (marks_done != 0) {
		if (marks_done % MARK_LINE == 0) {
			std::putchar('\n');
		} else if (marks_done % MARK_GROUP == 0) {
			std::putchar(' ');
		}
	}
	std::putchar(mark);
	std::fflush(stdout);
	marks_done++;
}

/* 逐输入的失败记录: 只记首次失败, 使收尾打印的是首次出现该失败时的现场。 */
struct fail_record {
	int iter;
	char note[192];
};

static struct fail_record failures[JSON_INPUT_COUNT];

static void note_failure(std::size_t idx, int iter, const char *note) {
	if (failures[idx].note[0] != '\0') {
		return;
	}
	failures[idx].iter = iter;
	std::snprintf(failures[idx].note, sizeof(failures[idx].note), "%s", note);
}

/* 对一份文档执行一次 parse/序列化/再解析/往返比对, 成功返回 0, 失败返回非 0。
 * 失败原因记入该输入的失败明细, 使进度行不被打断; 解析成功时把该树的节点构成累计到
 * stats。 */
static int churn_once(std::size_t idx, int iter, const char *text,
					  struct node_stats *stats) {
	char note[192];
	json root;

	try {
		root = json::parse(text);
	} catch (const json::parse_error &err) {
		/* parse_error 携带出错字节偏移 (自 1 计), 单列一分支以取该偏移。 */
		std::snprintf(note, sizeof(note), "parse error at byte %zu (%s)",
					  static_cast<std::size_t>(err.byte), err.what());
		note_failure(idx, iter, note);
		return 1;
	} catch (const json::exception &err) {
		std::snprintf(note, sizeof(note), "parse failed (%s)", err.what());
		note_failure(idx, iter, note);
		return 1;
	}
	accumulate(root, stats);

	/* 往返自检: 序列化后重新解析, 与原始树整体比对, 确认解析忠实无失真。 */
	json_string printed;
	try {
		printed = root.dump();
	} catch (const json::exception &err) {
		std::snprintf(note, sizeof(note), "dump failed (%s)", err.what());
		note_failure(idx, iter, note);
		return 1;
	}
	try {
		json reparsed = json::parse(printed);
		if (!(root == reparsed)) {
			note_failure(idx, iter, "dump/parse round-trip mismatch");
			return 1;
		}
	} catch (const json::exception &err) {
		std::snprintf(note, sizeof(note), "reparse failed (%s)", err.what());
		note_failure(idx, iter, note);
		return 1;
	}
	return 0;
}

/* 对语法错误文档断言解析被拒: 解析器抛出 parse_error 即符合预期; 期间构造的部分
 * 对象随栈回退析构, 故失败路径同样计入分配与回收压力。返回 0 表示符合预期。 */
static int reject_once(std::size_t idx, int iter, const char *text) {
	try {
		json root = json::parse(text);
		note_failure(idx, iter, "malformed document was accepted");
		return 1;
	} catch (const json::parse_error &) {
		return 0;
	} catch (const json::exception &err) {
		char note[192];

		std::snprintf(note, sizeof(note), "unexpected exception (%s)", err.what());
		note_failure(idx, iter, note);
		return 1;
	}
}

/* 打印逐类型的节点构成与实测堆占用。实测值取自分配器: peak 为全程在用的最大值,
 * live_at_end 为全部输入释放后仍在用的字节数, 正常应为 0。 */
static void print_summary(const struct node_stats *stats) {
	std::size_t nodes = 0;
	std::size_t bytes = 0;

	std::printf("\njson_cpp: AST summary (per node kind, last iteration over %zu inputs)\n",
				static_cast<std::size_t>(JSON_INPUT_COUNT));
	std::printf("  %-8s %10s  %12s\n", "kind", "nodes", "bytes");
	for (int k = 0; k < KIND_COUNT; k++) {
		std::printf("  %-8s %10zu  %12zu\n", kind_names[k], stats->nodes[k],
					stats->bytes[k]);
		nodes += stats->nodes[k];
		bytes += stats->bytes[k];
	}
	std::printf("  %-8s %10zu  %12zu\n", "total", nodes, bytes);
	std::printf("json_cpp: heap bytes (counted by the payload allocator): peak=%zu "
				"live_at_end=%zu\n",
				peak_bytes, live_bytes);
}

/* ---- 异常展开自检 ----
 * 飞地运行时合成的 musl 栈 auxv 必须给出程序头表的三项 (AT_PHDR/AT_PHENT/
 * AT_PHNUM)。musl 静态链接版的 dl_iterate_phdr 以此为唯一数据源, libunwind 经它
 * 定位 .eh_frame_hdr 才能取到展开表; 缺这三项时遍历到的程序头为 0 个, 任何一次
 * 抛出都在 _Unwind_RaiseException 处被判为栈已到底, 由 __cxa_throw 转为
 * std::terminate 并 abort (载荷以退出码 134 退出)。缺陷只在抛出时显形, 故这里
 * 先显式读 auxv 三项, 再做一次跨多层的抛出与捕获: 既要捕获到, 又要沿途每一层的
 * 局部对象都被析构, 才能说明栈是被逐层展开而非被整体跳过。 */

static int unwind_dtor_count = 0;

struct UnwindProbe {
	~UnwindProbe() { unwind_dtor_count++; }
};

/* 递归 depth 层后在最内层抛出。每层各持一个带析构的局部对象,
 * 展开时逐层析构, 故成功的展开应析构 depth + 1 次。 */
static void unwind_probe_throw(int depth) {
	UnwindProbe guard;
	if (depth == 0) {
		throw std::runtime_error("unwind probe");
	}
	unwind_probe_throw(depth - 1);
}

static int unwind_selfcheck(void) {
	unsigned long phdr = getauxval(AT_PHDR);
	unsigned long phent = getauxval(AT_PHENT);
	unsigned long phnum = getauxval(AT_PHNUM);
	if (phdr == 0 || phent == 0 || phnum == 0) {
		std::printf("json_cpp: FAIL unwinding: auxv lacks program headers "
					"(AT_PHDR=%lu AT_PHENT=%lu AT_PHNUM=%lu)\n",
					phdr, phent, phnum);
		return 1;
	}

	const int frames = 8;
	int caught = 0;
	unwind_dtor_count = 0;
	try {
		unwind_probe_throw(frames);
	} catch (const std::runtime_error &err) {
		caught = (std::strcmp(err.what(), "unwind probe") == 0);
	}

	const int expected_dtor = frames + 1;
	if (!caught || unwind_dtor_count != expected_dtor) {
		std::printf("json_cpp: FAIL unwinding: caught=%d dtor=%d (expected %d)\n",
					caught, unwind_dtor_count, expected_dtor);
		return 1;
	}

	std::printf("json_cpp: unwinding OK (AT_PHDR=0x%lx AT_PHENT=%lu AT_PHNUM=%lu, "
				"%d frames, %d dtors)\n",
				phdr, phent, phnum, frames, unwind_dtor_count);
	return 0;
}

int main() {
	struct node_stats stats = {{0}, {0}};
	std::size_t n_malformed = 0;
	std::size_t n_failed = 0;

	for (std::size_t i = 0; i < JSON_INPUT_COUNT; i++) {
		if (!json_expect_ok[i]) {
			n_malformed++;
		}
	}

	/* 先验异常展开通路, 再进入 churn: 抛出在飞地内不可用时应立刻显形,
	 * 而不是被解析异常的错误信息掩盖。 */
	if (unwind_selfcheck() != 0) {
		return 1;
	}

	std::printf("json_cpp: %zu inputs (%zu valid + %zu malformed) x %d iters = %zu checks\n",
				static_cast<std::size_t>(JSON_INPUT_COUNT),
				static_cast<std::size_t>(JSON_INPUT_COUNT) - n_malformed, n_malformed,
				JSON_CPP_ITERS, static_cast<std::size_t>(JSON_INPUT_COUNT) * JSON_CPP_ITERS);

	/* 每轮开始时清零节点统计, 故汇总反映的是最后一轮遍历全部输入的结果。 */
	for (int iter = 0; iter < JSON_CPP_ITERS; iter++) {
		std::memset(&stats, 0, sizeof(stats));
		for (std::size_t i = 0; i < JSON_INPUT_COUNT; i++) {
			int failed;
			char mark;

			if (json_expect_ok[i]) {
				failed = churn_once(i, iter, json_inputs[i], &stats);
				mark = failed ? '!' : '.';
			} else {
				failed = reject_once(i, iter, json_inputs[i]);
				mark = failed ? '#' : '~';
			}
			emit_mark(mark);
			if (failed != 0) {
				n_failed++;
			}
		}
	}
	if (marks_done % MARK_LINE != 0) {
		std::putchar('\n');
	}

	if (n_failed != 0) {
		std::printf("\njson_cpp: %zu failed checks\n", n_failed);
		std::printf("  marks: . valid-passed  ~ malformed-rejected  ! valid-failed  "
					"# malformed-accepted\n");
		for (std::size_t i = 0; i < JSON_INPUT_COUNT; i++) {
			if (failures[i].note[0] != '\0') {
				std::printf("  input#%zu (iter %d): %s\n", i, failures[i].iter,
							failures[i].note);
			}
		}
	}

	print_summary(&stats);

	if (live_bytes != 0) {
		std::printf("json_cpp: FAIL %zu bytes still allocated after every input was "
					"released\n",
					live_bytes);
		return 1;
	}
	if (n_failed != 0) {
		std::printf("json_cpp: FAIL %zu of %zu checks\n", n_failed,
					static_cast<std::size_t>(JSON_INPUT_COUNT) * JSON_CPP_ITERS);
		return 1;
	}
	std::printf("json_cpp: %zu inputs (%zu valid + %zu malformed) x %d parse: "
				"ALL OK (nlohmann %d.%d.%d)\n",
				static_cast<std::size_t>(JSON_INPUT_COUNT),
				static_cast<std::size_t>(JSON_INPUT_COUNT) - n_malformed, n_malformed,
				JSON_CPP_ITERS, NLOHMANN_JSON_VERSION_MAJOR,
				NLOHMANN_JSON_VERSION_MINOR, NLOHMANN_JSON_VERSION_PATCH);
	return 0;
}
