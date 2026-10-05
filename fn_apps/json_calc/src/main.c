/* json_calc — 以 input/ 下全部 JSON 文档为输入的 cJSON 解析 churn 载荷
 *
 * 输入集在构建期确定: input/ 目录下每个 .json 文件即一份真实第三方配置
 * (如 libinjection 的 sqlparse_data.json、npm 的 package-lock 等), 顶层结构互异,
 * 故载荷不假设任何固定键名. 每个文件保留两种文本形态, 由 scripts/embed_json.py
 * 一并编码为 json_inputs[] 数组写进 json_config.inc:
 *     - 原版 pretty (多行缩进, 即仓库内原始排版)
 *     - 由 scripts/minify_json.py 压成的单行、无字符串外空白的浏览器/API 形态
 *       (病态用例, 压测 cJSON 在无换行/缩进锚点下对连续 token 的解析)
 *   两种形态仅字符串外的空白不同, 解析所得结果一致.
 *
 * 遍历 json_inputs 数组, 对每份文档反复 cJSON_Parse / cJSON_Print / cJSON_Parse
 * / cJSON_Compare / cJSON_Delete:
 *   Parse   每识别一个 token 即分配一次 cJSON 节点, 字符串值另经 cJSON_strdup
 *           拷贝, 构成大量短生命周期小对象 malloc;
 *   Print   序列化用 printbuffer 反复增长;
 *   Delete  递归后序释放整棵树, 构成对应 free.
 * 其中 Print 后再 Parse 并用 cJSON_Compare 比对是结构无关的往返自检: 确认解析
 * 忠实无失真 (与顶层键名无关), 同时把每份文档的解析/序列化/再解析都计入 churn.
 * 数组遍历 × 迭代次数往复即对 libc malloc/free (musl mallocng) 的分配与回收压力.
 *
 * 除合法文档外, input/bad/ 下的语法错误文档也被内嵌 (追加在合法输入之后), 由
 * json_expect_ok 逐项标注: 对它们断言解析被拒, 从而覆盖错误返回路径 —— cJSON 解析
 * 失败时释放已构造的部分树, 是成功路径之外的另一种分配/回收次序.
 *
 * 输出分三段:
 *   1. 逐项进度: 每个输入每次迭代输出一个字符, 不逐项换行 —— '.' 断言正确且通过
 *      (合法文档解析成功), '~' 断言错误且通过 (语法错误文档被拒), '!' 断言正确但
 *      失败, '#' 断言错误但失败; 每 8 个字符后隔一个空格, 每 64 个字符换一行;
 *   2. 失败明细: 有失败时逐项给出首次失败所在的轮次与原因 (cJSON_GetErrorPtr 的
 *      错误偏移与附近文本片段), 不在运行途中打断进度行;
 *   3. 收尾汇总: 逐行给出各类型节点的节点数与节点自身、键名、字符串值所占的字节数,
 *      再给出分配计数器实测的堆占用峰值与全部输入释放后仍在用的字节数.
 * 任何一段都不打印整棵树的庞大序列化结果.
 *
 * 运行时不接受参数 (argv 被忽略).
 * 编译: 见同目录 Makefile (riscv64gc musl 静态).
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cJSON.h>

#include "json_config.inc" /* 定义 json_inputs[] (合法输入 + 语法错误文档) 与逐项判定数组 */

/* 迭代次数可用编译期宏覆盖, 默认取值令 qemu-riscv64 自检有界. */
#ifndef JSON_ITERS
#define JSON_ITERS 4
#endif

#define SNIPPET_MAX 40

/* 进度字符的排版: 每 8 个字符后隔一个空格, 每 64 个字符换一行. */
#define MARK_GROUP 8
#define MARK_LINE 64

/* 输入形态数量 (编译期常量): json_inputs 由 json_config.inc 提供, 新增输入文件
 * 只须放进 input/ 目录 (构建期按文件名排序收录), 无需改动本文件. */
#define JSON_INPUT_COUNT (sizeof(json_inputs) / sizeof(json_inputs[0]))

/* ---- 分配计数器 ----
 * cJSON 的分配与释放全部经 cJSON_InitHooks 登记的函数. 登记自定义钩子后
 * global_hooks.reallocate 被置空, printbuffer 与 print 改为分配、拷贝、释放三步,
 * 故所有路径仍全部经过钩子. 在钩子里包装 libc 分配器并记下在用字节数与峰值, 即得到
 * 本载荷真实的堆占用. 记账需要知道每块的字节数, 而 free 只给出指针, 故分配时在返回给
 * cJSON 的指针之前加一个头部存放字节数. */
union alloc_header {
	size_t size;
	max_align_t align; /* 使返回给 cJSON 的指针保持最大对齐 */
};

static size_t live_bytes = 0;
static size_t peak_bytes = 0;

static void *counting_malloc(size_t size) {
	union alloc_header *block = malloc(sizeof(union alloc_header) + size);

	if (block == NULL) {
		return NULL;
	}
	block->size = size;
	live_bytes += size;
	if (live_bytes > peak_bytes) {
		peak_bytes = live_bytes;
	}
	return block + 1;
}

static void counting_free(void *ptr) {
	union alloc_header *block;

	if (ptr == NULL) {
		return;
	}
	block = (union alloc_header *)ptr - 1;
	live_bytes -= block->size;
	free(block);
}

/* ---- 节点构成 ----
 * 逐类型统计节点数, 并按 cJSON 的分配粒度统计字节数: 每个节点一块 sizeof(cJSON),
 * 对象成员的键名与字符串值各一块 (均含结尾的 NUL). printbuffer 一类的容量冗余不属于
 * 节点自身, 由分配计数器实测的峰值体现. */
enum node_kind { KIND_OBJ, KIND_ARR, KIND_STR, KIND_NUM, KIND_LIT, KIND_OTH, KIND_COUNT };

static const char *const kind_names[KIND_COUNT] = {
	"object", "array", "string", "number", "literal", "other",
};

struct node_stats {
	size_t nodes[KIND_COUNT];
	size_t bytes[KIND_COUNT];
};

/* 递归累计一棵树的节点构成 (根含自身). type 的低 8 位是类型, 高位是 cJSON_IsReference
 * 与 cJSON_StringIsConst 一类的标志, 故按 0xFF 掩码取类型. */
static void accumulate(const cJSON *item, struct node_stats *stats) {
	const cJSON *cursor = item;

	while (cursor != NULL) {
		size_t bytes = sizeof(cJSON);
		enum node_kind kind;

		if (cursor->string != NULL) {
			bytes += strlen(cursor->string) + 1;
		}
		switch (cursor->type & 0xFF) {
		case cJSON_Object:
			kind = KIND_OBJ;
			break;
		case cJSON_Array:
			kind = KIND_ARR;
			break;
		case cJSON_String:
			kind = KIND_STR;
			if (cursor->valuestring != NULL) {
				bytes += strlen(cursor->valuestring) + 1;
			}
			break;
		case cJSON_Number:
			kind = KIND_NUM;
			break;
		case cJSON_True:
		case cJSON_False:
		case cJSON_NULL:
			kind = KIND_LIT;
			break;
		default:
			/* cJSON_Raw 只由 cJSON_CreateRaw 构造, 解析不产生. */
			kind = KIND_OTH;
			break;
		}
		stats->nodes[kind]++;
		stats->bytes[kind] += bytes;
		accumulate(cursor->child, stats);
		cursor = cursor->next;
	}
}

/* ---- 进度与失败明细 ---- */

/* 已输出的进度字符数, 决定下一个字符落在行内何处. */
static size_t marks_done = 0;

/* 输出一个进度字符: 每 MARK_GROUP 个字符后隔一个空格, 每 MARK_LINE 个字符换一行,
 * 并立即输出到控制台. */
static void emit_mark(char mark) {
	if (marks_done != 0) {
		if (marks_done % MARK_LINE == 0) {
			putchar('\n');
		} else if (marks_done % MARK_GROUP == 0) {
			putchar(' ');
		}
	}
	putchar(mark);
	fflush(stdout);
	marks_done++;
}

/* 逐输入的失败记录: 只记首次失败, 使收尾打印的是首次出现该失败时的现场. */
struct fail_record {
	int iter;
	char note[192];
};

static struct fail_record failures[JSON_INPUT_COUNT];

static void note_failure(size_t idx, int iter, const char *note) {
	if (failures[idx].note[0] != '\0') {
		return;
	}
	failures[idx].iter = iter;
	snprintf(failures[idx].note, sizeof(failures[idx].note), "%s", note);
}

/* 把解析失败的位置写成一行说明: cJSON_GetErrorPtr 指向出错处的文本, 换算为距输入的
 * 字节偏移, 并取其后至多 SNIPPET_MAX 个字符作片段, 不展开全文; 出错位置无法取得时
 * 偏移记 0. 说明写入调用方提供的缓冲区. */
static void format_parse_error(char *out, size_t out_size, const char *text) {
	const char *error_at = cJSON_GetErrorPtr();
	size_t offset = error_at != NULL ? (size_t)(error_at - text) : 0;
	char snippet[SNIPPET_MAX + 1];
	size_t len = 0;

	if (error_at != NULL) {
		while (error_at[len] != '\0' && error_at[len] != '\n' && error_at[len] != '\r' &&
			   len < SNIPPET_MAX) {
			snippet[len] = error_at[len];
			len++;
		}
	}
	snippet[len] = '\0';
	snprintf(out, out_size, "parse error near offset %zu (\"%s\")", offset, snippet);
}

/* 对一份文档执行一次 parse/序列化/再解析/往返比对/回收, 成功返回 0, 失败返回非 0.
 * 失败原因记入该输入的失败明细, 使进度行不被打断; 解析成功时把该树的节点构成累计到
 * stats. */
static int churn_once(size_t idx, int iter, const char *text, struct node_stats *stats) {
	char note[192];
	cJSON *root = cJSON_Parse(text);
	char *printed;
	cJSON *reparsed;
	cJSON_bool equal;

	if (root == NULL) {
		format_parse_error(note, sizeof(note), text);
		note_failure(idx, iter, note);
		return 1;
	}
	accumulate(root, stats);

	/* 往返自检: 序列化后重新解析, 与原始树逐节点比对, 确认解析忠实无失真. */
	printed = cJSON_Print(root);
	if (printed == NULL) {
		cJSON_Delete(root);
		note_failure(idx, iter, "cJSON_Print returned NULL");
		return 1;
	}
	reparsed = cJSON_Parse(printed);
	if (reparsed == NULL) {
		format_parse_error(note, sizeof(note), printed);
		cJSON_free(printed);
		cJSON_Delete(root);
		note_failure(idx, iter, note);
		return 1;
	}
	equal = cJSON_Compare(root, reparsed, 1);
	cJSON_Delete(reparsed);
	cJSON_free(printed);
	cJSON_Delete(root);
	if (!equal) {
		note_failure(idx, iter, "print/parse round-trip mismatch");
		return 1;
	}
	return 0;
}

/* 对语法错误文档断言解析被拒: cJSON 在失败时释放已构造的部分树, 故失败路径同样计入
 * 分配与回收压力. 返回 0 表示符合预期 (解析被拒), 非 0 表示文档竟被接受. */
static int reject_once(size_t idx, int iter, const char *text) {
	cJSON *root = cJSON_Parse(text);

	if (root == NULL) {
		return 0;
	}
	cJSON_Delete(root);
	note_failure(idx, iter, "malformed document was accepted");
	return 1;
}

/* 打印逐类型的节点构成与实测堆占用. 实测值取自分配计数器: peak 为全程在用的最大值,
 * live_at_end 为全部输入释放后仍在用的字节数, 正常应为 0. */
static void print_summary(const struct node_stats *stats) {
	size_t nodes = 0;
	size_t bytes = 0;

	printf("\njson_calc: AST summary (per node kind, last iteration over %zu inputs)\n",
		   (size_t)JSON_INPUT_COUNT);
	printf("  %-8s %10s  %12s\n", "kind", "nodes", "bytes");
	for (int k = 0; k < KIND_COUNT; k++) {
		printf("  %-8s %10zu  %12zu\n", kind_names[k], stats->nodes[k], stats->bytes[k]);
		nodes += stats->nodes[k];
		bytes += stats->bytes[k];
	}
	printf("  %-8s %10zu  %12zu\n", "total", nodes, bytes);
	printf("json_calc: heap bytes (counted by the cJSON hooks): peak=%zu live_at_end=%zu\n",
		   peak_bytes, live_bytes);
}

int main(void) {
	cJSON_Hooks hooks = {counting_malloc, counting_free};
	struct node_stats stats = {{0}, {0}};
	size_t n_malformed = 0;
	size_t n_failed = 0;

	cJSON_InitHooks(&hooks);

	for (size_t i = 0; i < JSON_INPUT_COUNT; i++) {
		if (!json_expect_ok[i]) {
			n_malformed++;
		}
	}
	printf("json_calc: %zu inputs (%zu valid + %zu malformed) x %d iters = %zu checks\n",
		   (size_t)JSON_INPUT_COUNT, (size_t)JSON_INPUT_COUNT - n_malformed, n_malformed,
		   JSON_ITERS, (size_t)JSON_INPUT_COUNT * JSON_ITERS);

	/* 每轮开始时清零节点统计, 故汇总反映的是最后一轮遍历全部输入的结果. */
	for (int iter = 0; iter < JSON_ITERS; iter++) {
		memset(&stats, 0, sizeof(stats));
		for (size_t i = 0; i < JSON_INPUT_COUNT; i++) {
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
			if (failed) {
				n_failed++;
			}
		}
	}
	if (marks_done % MARK_LINE != 0) {
		putchar('\n');
	}

	if (n_failed != 0) {
		printf("\njson_calc: %zu failed checks\n", n_failed);
		printf("  marks: . valid-passed  ~ malformed-rejected  ! valid-failed  "
			   "# malformed-accepted\n");
		for (size_t i = 0; i < JSON_INPUT_COUNT; i++) {
			if (failures[i].note[0] != '\0') {
				printf("  input#%zu (iter %d): %s\n", i, failures[i].iter, failures[i].note);
			}
		}
	}

	print_summary(&stats);

	if (live_bytes != 0) {
		printf("json_calc: FAIL %zu bytes still allocated after every input was "
			   "released\n",
			   live_bytes);
		return 1;
	}
	if (n_failed != 0) {
		printf("json_calc: FAIL %zu of %zu checks\n", n_failed,
			   (size_t)JSON_INPUT_COUNT * JSON_ITERS);
		return 1;
	}
	printf("json_calc: %zu inputs (%zu valid + %zu malformed) x %d parse: ALL OK\n",
		   (size_t)JSON_INPUT_COUNT, (size_t)JSON_INPUT_COUNT - n_malformed, n_malformed,
		   JSON_ITERS);
	return 0;
}
