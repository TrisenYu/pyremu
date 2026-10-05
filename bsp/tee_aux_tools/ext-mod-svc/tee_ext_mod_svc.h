/* tee_ext_mod_svc.h - 宿主侧的模块请求服务
 *
 * 飞地取入模块时经 M 模式让出并等待宿主应答 (见 bsp/custom-opensbi/lib/enclave_ext/
 * enclave_api/ext_mod/ 下的三个 handler). 宿主的动作是: 取出挂起的请求, 按请求类别
 * 报告映像的字节数或交付映像的字节, 然后恢复飞地. 本文件提供该应答流程与模块表,
 * 供回归工具 (cache-probe-exploit/tee_ecall_regress.c) 与最小装载器
 * (ext-mod-loader/) 共用.
 *
 * 模块表来自模块目录下的 modules.list, 每行为 "<模块编号> <映像文件名>",
 * 映像文件名相对该目录给出. 模块编号须与模块映像自身声明的编号一致, 不一致时取入
 * 在飞地内以编号不符失败 (见 sittim 的 ext_mod::LoadError::IdMismatch).
 *
 * 模块表只登记编号与路径, 映像文件本身按需读取: 飞地请求某个模块时,
 * tee_ext_mod_serve 才打开该映像并报告其字节数或交付其字节.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef TEE_EXT_MOD_SVC_H
#define TEE_EXT_MOD_SVC_H

#include <stdint.h>

#include "tee_enclave.h"

/* 映像文件路径与清单行的长度上限. */
#define TEE_EXT_MOD_PATH_MAX 512
/* 一次登记的最大模块数. */
#define TEE_EXT_MOD_TABLE_MAX 32

struct tee_ext_mod_entry {
	uint32_t module_id;
	char path[TEE_EXT_MOD_PATH_MAX];
};

struct tee_ext_mod_table {
	struct tee_ext_mod_entry entries[TEE_EXT_MOD_TABLE_MAX];
	int count;
};

/* 读取 dir/modules.list 并登记各项. 返回登记的模块数; 清单不可读或某行格式有误时
 * 返回负 errno, 不做部分登记. 本函数不访问清单所列的映像文件. */
int tee_ext_mod_table_load(struct tee_ext_mod_table *table, const char *dir);

/* 服务一次挂起的模块请求. 返回 1 表示已应答一项请求, 0 表示当前无挂起请求,
 * 负 errno 表示调用期失败. */
int tee_ext_mod_serve(
	int fd,
	const struct tee_ext_mod_table *table,
	uint64_t enclave_id,
	uint64_t token
);

#endif /* TEE_EXT_MOD_SVC_H */
