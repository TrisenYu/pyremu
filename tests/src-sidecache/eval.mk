# /eval/cache-probe-exploit/Makefile -- TEE 缓存侧信道探测与利用测试套件
#
# 用法:
#   make              默认目标: 跑完全部缓存侧信道测试 (同 make probe)
#   make probe        跑完全部缓存侧信道测试 (跨飞地 Flush+Reload + 生命周期 + 并发 + TLB/integrity)
#   make cross-enclave 跨飞地 Flush+Reload 缓存侧信道测试
#   make benign       良性飞地生命周期演示
#   make concurrent   并发飞地创建测试
#   make malice-avail DoS: 耗尽飞地内存池 (阻塞式, 独占池, 单独运行)
#   make malice-conf  TLB 侧信道探测
#   make malice-integ 完整性边界探测
#   make malice-all   运行全部攻击 (含阻塞式 avail)
#   make help         显示本帮助
#   make install      加载 TEE 驱动 (insmod)
#   make uninstall    卸载 TEE 驱动
#
# 另见: /eval/stress-test/ 批处理飞地生命周期压力测试。

DRV     = ../tee_enclave_drv.ko
TEST    = tee_test
MALICE  = tee_malice
PAYLOAD = hello_payload
CONCUR  = tee_concurrent
XCACHE  = cross_enclave_cache
VICTIM  = victim_cache
ATTACKER = attacker_cache

TEE_BIN_DIR := /eval/cache-probe-exploit

.PHONY: probe help install benign uninstall malice-avail malice-conf malice-integ \
        malice-all cross-enclave concurrent

# 默认目标 (首个目标): 一次跑完全部缓存侧信道测试.
# malice-avail 会 pause() 阻塞并独占内存池, 无法并入自动流程,
# 需要时单独以 make malice-avail 运行.
probe: install
	@echo "=== 全部缓存侧信道测试 ==="
	@echo ""
	@echo "--- [1/4] 跨飞地 Flush+Reload 缓存侧信道 ---"
	"$(TEE_BIN_DIR)/$(XCACHE)" "$(TEE_BIN_DIR)/$(VICTIM)" "$(TEE_BIN_DIR)/$(ATTACKER)"
	@echo ""
	@echo "--- [2/4] 良性飞地生命周期 ---"
	"$(TEE_BIN_DIR)/$(TEST)" "$(TEE_BIN_DIR)/$(PAYLOAD)"
	@echo ""
	@echo "--- [3/4] 并发飞地创建 ---"
	"$(TEE_BIN_DIR)/$(CONCUR)" "$(TEE_BIN_DIR)/$(PAYLOAD)" 4
	@echo ""
	@echo "--- [4/4] TLB 侧信道 + 完整性探测 ---"
	"$(TEE_BIN_DIR)/$(MALICE)" conf
	"$(TEE_BIN_DIR)/$(MALICE)" integ
	@echo ""
	@echo "=== 全部缓存侧信道测试完成 ==="

cross-enclave: install
	@echo "=== 跨飞地 Flush+Reload 缓存侧信道测试 ==="
	"$(TEE_BIN_DIR)/$(XCACHE)" "$(TEE_BIN_DIR)/$(VICTIM)" "$(TEE_BIN_DIR)/$(ATTACKER)"

benign: install
	@echo "=== 良性飞地生命周期演示 ==="
	@echo ""
	@echo "  步骤1 查询内存池与当前 mdid"
	@echo "  步骤2 创建飞地 (固件加载内置管理器)"
	@echo "  步骤3 以自定义载荷进入飞地"
	@echo "  步骤4 飞地运行, 挂起, 返回宿主"
	@echo "  步骤5 关闭飞地 (释放槽位与内存)"
	@echo ""
	"$(TEE_BIN_DIR)/$(TEST)" "$(TEE_BIN_DIR)/$(PAYLOAD)"

concurrent: install
	@echo "=== 并发飞地创建测试 ==="
	"$(TEE_BIN_DIR)/$(CONCUR)" "$(TEE_BIN_DIR)/$(PAYLOAD)" 4

malice-avail: install
	@echo "=== 攻击: 内存池耗尽 (DoS) ==="
	"$(TEE_BIN_DIR)/$(MALICE)" avail

malice-conf: install
	@echo "=== 攻击: TLB 侧信道探测 ==="
	"$(TEE_BIN_DIR)/$(MALICE)" conf

malice-integ: install
	@echo "=== 攻击: 完整性边界探测 ==="
	"$(TEE_BIN_DIR)/$(MALICE)" integ

malice-all: install
	@echo "=== 全部攻击 (含阻塞式 avail) ==="
	"$(TEE_BIN_DIR)/$(MALICE)" all

help:
	@echo "=== /eval TEE 测试套件 ==="
	@echo ""
	@echo "  make              默认: 跑完全部缓存侧信道测试 (同 make probe)"
	@echo "  make probe        跑完全部缓存侧信道测试 (Flush+Reload + 生命周期 + 并发 + TLB/integrity)"
	@echo "  make cross-enclave 跨飞地 Flush+Reload 缓存侧信道测试"
	@echo "  make benign       良性飞地生命周期 (query -> create -> enter -> shutdown)"
	@echo "  make concurrent   并发飞地创建测试"
	@echo "  make malice-avail DoS: 耗尽飞地内存池 (阻塞式, 独占池, 单独运行)"
	@echo "  make malice-conf  TLB 侧信道探测 (跨 mdid 域)"
	@echo "  make malice-integ 完整性探测 (ID 猜测, 内核 VA 注入)"
	@echo "  make malice-all   运行全部攻击 (含阻塞式 avail)"
	@echo ""
	@echo "  make install      加载驱动 (insmod)"
	@echo "  make uninstall    卸载驱动 (rmmod)"
	@echo ""

install:
	@if [ -c /dev/tee_enclave ]; then \
		echo "[eval] 驱动已加载 (/dev/tee_enclave 存在)"; \
	else \
		echo "[eval] 加载驱动..."; \
		/sbin/insmod "$(TEE_BIN_DIR)/$(DRV)" && echo "[eval] 驱动加载完成. /dev/tee_enclave 就绪."; \
	fi

uninstall:
	@/sbin/rmmod tee_enclave_drv 2>/dev/null && echo "[eval] 驱动已卸载" || true
