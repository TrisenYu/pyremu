# /eval/ext-mod-loader/Makefile -- 最小扩展模块装载器
#
# 用法:
#   make load         装载 PAYLOAD 指定的载荷 (默认 orbit), 应答它的模块请求
#   make help         显示本帮助
#   make install      加载 TEE 驱动 (insmod)
#   make uninstall    卸载 TEE 驱动
#
# 模块目录 /modules 由 local-build.sh 安装: 模块映像与清单 modules.list 同处该目录.

DRV		= ../tee_enclave_drv.ko
BIN		= ext_mod_loader
BIN_DIR	= /eval/ext-mod-loader
MODULES	?= /modules
PAYLOAD	?= /eval/cache-probe-exploit/fn_apps/orbit

.PHONY: load help install uninstall

load: install
	@echo "=== 扩展模块装载 (payload=$(PAYLOAD)) ==="
	"$(BIN_DIR)/$(BIN)" "$(PAYLOAD)" "$(MODULES)"

help:
	@echo "=== /eval/ext-mod-loader ==="
	@echo ""
	@echo "  make load         装载载荷并应答其模块请求 (PAYLOAD= 可覆盖, 默认 orbit)"
	@echo "  make install      加载驱动 (insmod)"
	@echo "  make uninstall    卸载驱动 (rmmod)"
	@echo ""

install:
	@if [ -c /dev/tee_enclave ]; then \
		echo "[eval] 驱动已加载 (/dev/tee_enclave 存在)"; \
	else \
		echo "[eval] 加载驱动..."; \
		/sbin/insmod "$(BIN_DIR)/$(DRV)" && echo "[eval] 驱动加载完成. /dev/tee_enclave 就绪."; \
	fi

uninstall:
	@/sbin/rmmod tee_enclave_drv 2>/dev/null && echo "[eval] 驱动已卸载" || true
