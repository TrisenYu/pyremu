# /eval/cache-probe-exploit/Makefile -- TEE side-channel probe & exploit suite
#
# Usage:
#   make help            Show this help
#   make install         Load the TEE driver (insmod)
#   make benign          Legitimate enclave lifecycle demo
#   make malice-avail    DoS: exhaust enclave memory pool
#   make malice-conf     TLB side-channel probe
#   make malice-integ    Integrity boundary probing
#   make malice-all      Run all attacks
#   make probe           Run all probe & exploit tests (alias for malice-all)
#   make concurrent      Concurrent enclave creation test
#   make uninstall       Unload the TEE driver
#
# See also: /eval/stress-test/ for batch enclave lifecycle stress.

DRV    = ../tee_enclave_drv.ko
TEST   = tee_test
MALICE = tee_malice
PAYLOAD = hello_payload
CONCUR  = tee_concurrent

TEE_BIN_DIR := /eval/cache-probe-exploit

.PHONY: help probe install benign uninstall malice-avail malice-conf malice-integ malice-all

help:
	@echo "=== /eval TEE Test Suite ==="
	@echo ""
	@echo "  make benign          Legitimate enclave lifecycle (query -> create -> enter -> shutdown)"
	@echo ""
	@echo "  make malice-avail    DoS:  exhaust enclave memory pool"
	@echo "  make malice-conf     TLB side-channel probe across mdid domains"
	@echo "  make malice-integ    Integrity probe (ID guessing, kernel VA injection)"
	@echo "  make malice-all      Run all attacks"
	@echo "  make probe           Run all probe & exploit tests (alias for malice-all)"
	@echo "  make concurrent      Concurrent enclave creation test"
	@echo ""
	@echo "  make install          Load driver (insmod)"
	@echo "  make uninstall        Unload driver (rmmod)"
	@echo ""

install:
	@if [ -c /dev/tee_enclave ]; then \
		echo "[eval] driver already loaded (/dev/tee_enclave exists)"; \
	else \
		echo "[eval] loading driver..."; \
		/sbin/insmod "$(TEE_BIN_DIR)/$(DRV)" && echo "[eval] driver loaded. /dev/tee_enclave ready."; \
	fi

benign: install
	@echo "=== Benign Enclave Lifecycle Demo ==="
	@echo ""
	@echo "  step 1. query memory pool and current mdid"
	@echo "  step 2. create enclave (firmware loads built-in manager)"
	@echo "  step 3. enter enclave with custom payload"
	@echo "  step 4. enclave runs, suspends, back to host"
	@echo "  step 5. shutdown enclave (frees slot and memory)"
	@echo ""
	"$(TEE_BIN_DIR)/$(TEST)" "$(TEE_BIN_DIR)/$(PAYLOAD)"

malice-avail: install
	@echo "=== ATTACK: Memory Pool Exhaustion (DoS) ==="
	"$(TEE_BIN_DIR)/$(MALICE)" avail

malice-conf: install
	@echo "=== ATTACK: TLB Side-Channel Probe ==="
	"$(TEE_BIN_DIR)/$(MALICE)" conf

malice-integ: install
	@echo "=== ATTACK: Integrity Boundary Probe ==="
	"$(TEE_BIN_DIR)/$(MALICE)" integ

probe: malice-all

malice-all: install
	@echo "=== ALL ATTACKS ==="
	"$(TEE_BIN_DIR)/$(MALICE)" all

concurrent: install
	@echo "=== Concurrent Enclave Creation Test ==="
	"$(TEE_BIN_DIR)/$(CONCUR)" "$(TEE_BIN_DIR)/$(PAYLOAD)" 4

uninstall:
	@/sbin/rmmod tee_enclave_drv 2>/dev/null && echo "[eval] driver unloaded" || true
