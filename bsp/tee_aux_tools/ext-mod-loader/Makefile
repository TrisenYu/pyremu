# bsp/tee_aux_tools/ext-mod-loader/Makefile
# 最小扩展模块装载器 (源目录, 部署为 /eval/ext-mod-loader)
#
# 目标:
#   make all     编译宿主程序
#   make clean   清理
#
# tee_enclave.h 取自 ../cache-probe-exploit: 该头是 Linux 驱动用户态接口的唯一定义,
# 各工具目录各留一份副本已在部署时出现过未同步的副本, 故本工具经包含路径引用它,
# 不复写第二份。

phony =

# 取消 GNU Make 内置的 %: %.c 隐式模式规则:
# 防止在源码目录直接生成无 bin/ 前缀的二进制.
%: %.c

CROSS_CC ?= $(shell command -v riscv64-linux-gnu-gcc 2>/dev/null || echo riscv64-unknown-linux-gnu-gcc)
CFLAGS	 = -static -O2 -Wall

OUT_DIR	  = bin
INCLUDES  = -I../cache-probe-exploit -I../ext-mod-svc
SVC_SRC	  = ../ext-mod-svc/tee_ext_mod_svc.c
SVC_HDR	  = ../ext-mod-svc/tee_ext_mod_svc.h

all: $(OUT_DIR)/ext_mod_loader
phony += all

$(OUT_DIR)/ext_mod_loader: ext_mod_loader.c ../cache-probe-exploit/tee_enclave.h $(SVC_SRC) $(SVC_HDR)
	@mkdir -p $(@D)
	$(CROSS_CC) $(CFLAGS) $(INCLUDES) ext_mod_loader.c $(SVC_SRC) -o $@

clean:
	rm -rf $(OUT_DIR)
phony += clean

.PHONY: $(phony)
