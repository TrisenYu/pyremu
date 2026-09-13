/* json_cpp — 以 json_calc 同一输入集为输入的 nlohmann/json 解析 churn 载荷
 *
 * 与同目录 main.c (cJSON) 共用构建期生成的内嵌输入 json_config.inc: input/ 下
 * 每份真实第三方配置各保留 pretty 与 minified 两种形态, 一并编码进 json_inputs[],
 * 故载体内无文件系统依赖, 两种形态语义逐字节等价 (往返自检相等).
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
 * 输出仅解析结论 (成功时逐份打印字节数与节点数, 收尾打印总览), 失败时打印 FAIL
 * 与异常描述, 不打印整棵树的序列化结果。运行时不接受参数 (argv 被忽略)。
 *
 * 编译: 见同目录 Makefile (riscv64gc musl 静态, libc++)。
 */

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

#include <nlohmann/json.hpp>

#include "json_config.inc" /* 定义 json_inputs[] (每个输入文件的 pretty + minified) */

/* 迭代次数可用编译期宏覆盖。标定依据 (qemu-riscv64, 16 份输入): 本载荷 1 轮
 * 0.20 s, 同目录 cJSON 载荷 4 轮 1.40 s (每轮 0.35 s); 本载荷单份文档的分配次数
 * 与总字节数远高于 cJSON, 故取 1 轮已给出与之相当的分配压力与运行时长。 */
#ifndef JSON_CPP_ITERS
#define JSON_CPP_ITERS 1
#endif

using json = nlohmann::json;

/* 输入形态数量 (编译期常量): json_inputs 由 json_config.inc 提供, 新增输入文件
 * 只须放进 input/ 目录 (构建期按文件名排序收录), 无需改动本文件。 */
#define JSON_INPUT_COUNT (sizeof(json_inputs) / sizeof(json_inputs[0]))

/* 整棵树的节点总数 (根含自身), 作为解析结果的规模度量。
 * 对象按成员值遍历, 成员键名不计入 —— 与 cJSON 侧的节点口径一致。 */
static std::size_t count_nodes(const json &value) {
	std::size_t count = 1;
	if (value.is_object() || value.is_array()) {
		for (const json &child : value) {
			count += count_nodes(child);
		}
	}
	return count;
}

/* 对一份文档执行一次 parse/dump/再解析/往返比对, 成功返回 0, 失败返回非 0。 */
static int churn_once(std::size_t idx, const char *text) {
	std::size_t len = std::strlen(text);
	json root;
	try {
		root = json::parse(text);
	} catch (const json::parse_error &err) {
		/* parse_error 携带出错字节偏移 (自 1 计), 单列一分支以取该偏移。 */
		std::printf("json_cpp: FAIL input#%zu: parse error at byte %zu (%s)\n", idx,
					static_cast<std::size_t>(err.byte), err.what());
		return 1;
	} catch (const json::exception &err) {
		std::printf("json_cpp: FAIL input#%zu: parse failed (%s)\n", idx, err.what());
		return 1;
	}
	std::size_t nodes = count_nodes(root);

	/* 往返自检: 序列化后重新解析, 与原始树整体比对, 确认解析忠实无失真。 */
	std::string printed;
	try {
		printed = root.dump();
	} catch (const json::exception &err) {
		std::printf("json_cpp: FAIL input#%zu: dump failed (%s)\n", idx, err.what());
		return 1;
	}
	try {
		json reparsed = json::parse(printed);
		if (!(root == reparsed)) {
			std::printf("json_cpp: FAIL input#%zu: dump/parse round-trip mismatch\n",
						idx);
			return 1;
		}
	} catch (const json::exception &err) {
		std::printf("json_cpp: FAIL input#%zu: reparse failed (%s)\n", idx, err.what());
		return 1;
	}

	std::printf("json_cpp: input#%zu parse OK (bytes=%zu nodes=%zu)\n", idx, len, nodes);
	return 0;
}

/* 对语法错误文档断言解析被拒: 解析器抛出 parse_error 即符合预期; 期间构造的
 * 部分对象随栈回退析构, 故失败路径同样计入分配与回收压力. 返回 0 表示被拒. */
static int reject_once(std::size_t idx, const char *text) {
	std::size_t len = std::strlen(text);

	try {
		json root = json::parse(text);
		std::printf(
			"json_cpp: FAIL input#%zu: malformed document was accepted (bytes=%zu)\n",
			idx, len);
		return 1;
	} catch (const json::parse_error &) {
		std::printf("json_cpp: input#%zu rejected OK (bytes=%zu)\n", idx, len);
		return 0;
	} catch (const json::exception &err) {
		std::printf("json_cpp: FAIL input#%zu: unexpected exception (%s)\n", idx,
					err.what());
		return 1;
	}
}

int main(void) {
	std::size_t n_malformed = 0;
	for (std::size_t i = 0; i < JSON_INPUT_COUNT; i++) {
		if (!json_expect_ok[i]) {
			n_malformed++;
		}
	}

	for (int iter = 0; iter < JSON_CPP_ITERS; iter++) {
		for (std::size_t i = 0; i < JSON_INPUT_COUNT; i++) {
			int failed = json_expect_ok[i] ? churn_once(i, json_inputs[i])
										   : reject_once(i, json_inputs[i]);
			if (failed != 0) {
				return 1;
			}
		}
	}
	std::printf("json_cpp: %zu inputs (%zu valid + %zu malformed) x %d parse: "
				"ALL OK (nlohmann %d.%d.%d)\n",
				static_cast<std::size_t>(JSON_INPUT_COUNT),
				static_cast<std::size_t>(JSON_INPUT_COUNT) - n_malformed, n_malformed,
				JSON_CPP_ITERS, NLOHMANN_JSON_VERSION_MAJOR,
				NLOHMANN_JSON_VERSION_MINOR, NLOHMANN_JSON_VERSION_PATCH);
	return 0;
}
