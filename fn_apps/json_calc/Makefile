# json_calc — 以 input/ 下全部 JSON 文档为输入的 JSON 解析 churn 载荷 (两个实现)
#
# 两个载荷共享同一输入集与同一构建期内嵌 (json_config.inc), 实现对拍:
#   json_calc_payload  cJSON (C, MIT, 见 cjson/LICENSE)
#   json_cpp_payload   nlohmann/json (C++, MIT, 见 nlohmann/LICENSE.MIT)
#
# cJSON 的 Parse/Print/Delete 各自大量调用 libc 分配器, 以 input/ 下每份真实第三方
# 配置 (sqlparse_data.json、npm-cli-pkg-lock.json 等) 为输入构成分配与回收的压力.
# 每个文件先经 scripts/minify_json.py 压成单行, 再由 scripts/embed_json.py 把两种形态
# (pretty 原版 + minified 单行) 一并编码为 json_inputs[] 数组写进 json_config.inc,
# 载体内无文件系统依赖.
#
# 两者的分配粒度不同 (cJSON 为裸节点链表, nlohmann 每节点是 basic_json 对象另加
# std::string 与有序映射), 故互为补充而非重复: 同一批文档, 不同的分配次数与释放时机.
#
# musl 交叉工具链配置 (标准 clang + riscv64gc/lp64d 硬浮点) 见 ../config.mk;
# C++ 载荷另需 libc++ (头与静态库均在 bsp/musl-gc-sysroot 内).

include ../config.mk

CLANGXX := clang++

CJSON_DIR    := cjson
NLOHMANN_DIR := nlohmann
BUILD        := build

# cJSON 头目录 + 生成内嵌 (json_config.inc) 所在目录
CFLAGS += -isystem $(CJSON_DIR) -I$(BUILD)

# C++ 编译参数: -nostdinc++ 屏蔽宿主 libstdc++ 头, 改引 sysroot 内 libc++ 头;
# 其余 -isystem 与 C 侧同理 (libc++ 头经 #include_next 引用 musl 的 C 头).
# nlohmann 取 $(NLOHMANN_DIR) 的父目录, 使上游标准引用形式 <nlohmann/json.hpp> 生效
# (vendor 目录名即库名, 与上游安装布局 include/nlohmann/ 一致).
CXXFLAGS := $(ARCH) -std=c++17 -O2 -Wall \
  -nostdinc++ \
  -isystem $(MUSL_SYSROOT)/include/c++/v1 \
  -isystem $(MUSL_SYSROOT)/include \
  -isystem $(MUSL_SRC)/include \
  -isystem $(MUSL_SRC)/arch/riscv64 \
  -isystem $(MUSL_SRC)/arch/generic \
  -isystem $(MUSL_SRC)/obj/include \
  -isystem $(NLOHMANN_DIR)/.. -I$(BUILD)

OUT     := $(BUILD)/json_calc_payload
CPP_OUT := $(BUILD)/json_cpp_payload

all: $(OUT) $(CPP_OUT)

# 输入集为 input/ 下所有 .json 文件, 按文件名排序, 顺序稳定. 每个文件保留两种形态 ——
# 原版 pretty 与 minify 压成的单行, 后一种无缩进与换行, 即浏览器与 API 实际接收的形态;
# 由 scripts/embed_json.py 一并编码为 json_inputs[] 数组写进 json_config.inc, 供
# main.c 遍历压测. 新增输入只须把文件放进 input/, 无需改动本 Makefile 与 main.c.
JSON_SRC := $(sort $(wildcard input/*.json))
JSON_MIN := $(patsubst input/%.json,$(BUILD)/%.min.json,$(JSON_SRC))
# 语法错误文档 (input/bad/ 子目录): 不参与 minify, 由 embed_json.py 原样收录并追加在
# 合法输入之后; 此处仅声明为依赖, 以便改动畸形文档能触发 json_config.inc 重新生成。
JSON_BAD := $(sort $(wildcard input/bad/*.json))

$(BUILD)/%.min.json: input/%.json scripts/minify_json.py
	@mkdir -p $(@D)
	python3 scripts/minify_json.py $< $@

$(BUILD)/json_config.inc: $(JSON_SRC) $(JSON_MIN) $(JSON_BAD) scripts/embed_json.py
	@mkdir -p $(@D)
	python3 scripts/embed_json.py $@ $(JSON_SRC) $(JSON_MIN)

$(BUILD)/cJSON.o: $(CJSON_DIR)/cJSON.c
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) -c $< -o $@

$(BUILD)/main.o: src/main.c $(BUILD)/json_config.inc
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) -c $< -o $@

# -nodefaultlibs 屏蔽 clang 对 musl 目标的隐式 -lgcc_eh, 显式列运行库;
# -lm 供 cJSON.o 中 compare_double 的 fabs (编译即带引用, 与是否调用无关).
$(OUT): $(BUILD)/cJSON.o $(BUILD)/main.o
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) $(BUILD)/cJSON.o $(BUILD)/main.o \
	  $(LDFLAGS) -nodefaultlibs -lc -lm -lgcc -o $@

run: $(OUT)
	$(QEMU) $(OUT)

# C++ 载荷: nlohmann/json 单头 (nlohmann/json.hpp) + libc++。
# -nodefaultlibs 屏蔽 clang 隐式追加 -lgcc_eh; 显式列 C++ 运行库。
# 链接顺序须为 libc++、libunwind、libc、libgcc, 因为 libc++ 引用后三者符号。
$(CPP_OUT): src/main_cpp.cpp $(BUILD)/json_config.inc $(NLOHMANN_DIR)/json.hpp
	@mkdir -p $(@D)
	$(CLANGXX) --target=$(TARGET) $(CXXFLAGS) $< $(LDFLAGS) -nodefaultlibs \
	  -lc++ -lunwind -lc -lgcc -o $@

run-cpp: $(CPP_OUT)
	$(QEMU) $(CPP_OUT)

clean:
	rm -rf $(BUILD)

.PHONY: all run run-cpp clean
