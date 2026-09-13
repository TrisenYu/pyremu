/* json_calc — 以 input/ 下全部 JSON 文档为输入的 cJSON 解析 churn 载荷
 *
 * 输入集在构建期确定: input/ 目录下每个 .json 文件即一份真实第三方配置
 * (如 libinjection 的 sqlparse_data.json、npm 的 package-lock 等), 顶层结构互
 * 异, 故载荷不假设任何固定键名。每个文件保留两种物理形态, 由
 * scripts/embed_json.py 一并编码为 json_inputs[] 数组写进 json_config.inc:
 *     - 原版 pretty (多行缩进, 即仓库内原始排版)
 *     - 由 scripts/minify_json.py 压成的单行、无字符串外空白的浏览器/API 形态
 *       (病态用例, 压测 cJSON 在无换行/缩进锚点下对连续 token 的解析)
 *   两种形态语义逐字节等价 (round-trip 相等), 仅物理排版不同.
 *
 * 遍历 json_inputs 数组, 对每份文档反复 cJSON_Parse / cJSON_Print / cJSON_Parse
 * / cJSON_Compare / cJSON_Delete:
 *   Parse   每识别一个 token 即分配一次 cJSON 节点, 字符串值另经 cJSON_strdup
 *           拷贝, 构成大量短生命周期小对象 malloc;
 *   Print   序列化用 printbuffer 反复 realloc 增长;
 *   Delete  递归后序释放整棵树, 构成对应 free.
 * 其中「Print 后再 Parse 并用 cJSON_Compare 比对」是结构无关的往返自检: 确认
 * 解析忠实无失真 (与顶层键名无关), 同时把每份文档的解析/序列化/再解析都计入
 * churn。数组遍历 × 迭代次数往复即对 libc malloc/free (musl mallocng) 的分配与
 * 回收压力。
 *
 * 除合法文档外, input/bad/ 下的语法错误文档也被内嵌 (追加在合法输入之后), 由
 * json_expect_ok 逐项标注: 对它们断言解析被拒, 从而覆盖错误返回路径 —— cJSON
 * 解析失败时释放已构造的部分树, 是成功路径之外的另一种分配/回收次序。
 *
 * 输出仅解析结论: 成功时打印字节数与节点数; 失败时给出 cJSON_GetErrorPtr 的
 * 错误偏移与附近文本片段, 而不打印任何整棵树的庞大序列化结果。
 *
 * 运行时不接受参数 (argv 被忽略)。
 * 编译: 见同目录 Makefile (riscv64gc musl 静态).
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <cJSON.h>

#include "json_config.inc" /* 定义 json_inputs[] (合法输入 + 语法错误文档) 与逐项判定数组 */

/* 迭代次数可用编译期宏覆盖, 默认取值令 qemu-riscv64 自检有界. */
#ifndef JSON_ITERS
#define JSON_ITERS 4
#endif

#define SNIPPET_MAX 40

/* 输入形态数量 (编译期常量): json_inputs 由 json_config.inc 提供, 新增输入文件
 * 只须放进 input/ 目录 (构建期按文件名排序收录), 无需改动本文件. */
#define JSON_INPUT_COUNT (sizeof(json_inputs) / sizeof(json_inputs[0]))

/* 整棵树的节点总数 (根含自身), 作为解析结果的规模度量. */
static size_t count_nodes(const cJSON *item) {
	size_t count = 1;
	const cJSON *cursor = item ? item->child : NULL;
	while (cursor != NULL) {
		count += count_nodes(cursor);
		cursor = cursor->next;
	}
	return count;
}

/* 打印解析失败原因: cJSON_GetErrorPtr 指向出错处的文本, 换算为距输入的字节
 * 偏移, 并取其后至多 SNIPPET_MAX 个字符作片段, 不展开全文. idx 标注失败归属的
 * 输入形态下标. */
static void report_parse_error(size_t idx, const char *text) {
	const char *error_at = cJSON_GetErrorPtr();
	size_t offset = error_at != NULL ? (size_t)(error_at - text) : 0;
	char snippet[SNIPPET_MAX + 1];
	size_t len = 0;
	if (error_at != NULL) {
		while (error_at[len] != '\0' && error_at[len] != '\n' &&
			   error_at[len] != '\r' && len < SNIPPET_MAX) {
			snippet[len] = error_at[len];
			len++;
		}
	}
	snippet[len] = '\0';
	printf("json_calc: FAIL input#%zu: parse error near offset %zu (\"%s\")\n",
		   idx, offset, snippet);
}

/* 对一份文档执行一次 parse/序列化/再解析/往返比对/回收, 成功返回 0, 失败返回非 0. */
static int churn_once(size_t idx, const char *text) {
	size_t len = strlen(text);

	cJSON *root = cJSON_Parse(text);
	if (root == NULL) {
		report_parse_error(idx, text);
		return 1;
	}
	size_t nodes = count_nodes(root);

	/* 往返自检: 序列化后重新解析, 与原始树逐节点比对, 确认解析忠实无失真. */
	char *printed = cJSON_Print(root);
	if (printed == NULL) {
		cJSON_Delete(root);
		printf("json_calc: FAIL input#%zu: cJSON_Print returned NULL\n", idx);
		return 1;
	}
	cJSON *reparsed = cJSON_Parse(printed);
	if (reparsed == NULL) {
		report_parse_error(idx, printed);
		cJSON_free(printed);
		cJSON_Delete(root);
		return 1;
	}
	cJSON_bool equal = cJSON_Compare(root, reparsed, 1);
	cJSON_Delete(reparsed);
	cJSON_free(printed);
	cJSON_Delete(root);
	if (!equal) {
		printf("json_calc: FAIL input#%zu: print/parse round-trip mismatch\n", idx);
		return 1;
	}

	printf("json_calc: input#%zu parse OK (bytes=%zu nodes=%zu)\n", idx, len, nodes);
	return 0;
}

/* 对语法错误文档断言解析被拒: cJSON 在失败时释放已构造的部分树, 故失败路径同样
 * 计入分配与回收压力. 返回 0 表示符合预期 (解析被拒), 非 0 表示文档竟被接受. */
static int reject_once(size_t idx, const char *text) {
	size_t len = strlen(text);

	cJSON *root = cJSON_Parse(text);
	if (root != NULL) {
		cJSON_Delete(root);
		printf("json_calc: FAIL input#%zu: malformed document was accepted (bytes=%zu)\n",
			   idx, len);
		return 1;
	}
	printf("json_calc: input#%zu rejected OK (bytes=%zu)\n", idx, len);
	return 0;
}

int main(void) {
	size_t n_malformed = 0;
	for (size_t i = 0; i < JSON_INPUT_COUNT; i++) {
		if (!json_expect_ok[i]) {
			n_malformed++;
		}
	}

	for (int iter = 0; iter < JSON_ITERS; iter++) {
		for (size_t i = 0; i < JSON_INPUT_COUNT; i++) {
			int failed = json_expect_ok[i] ? churn_once(i, json_inputs[i])
										   : reject_once(i, json_inputs[i]);
			if (failed) {
				return 1;
			}
		}
	}
	printf("json_calc: %zu inputs (%zu valid + %zu malformed) x %d parse: ALL OK\n",
		   JSON_INPUT_COUNT, JSON_INPUT_COUNT - n_malformed, n_malformed, JSON_ITERS);
	return 0;
}
