# /eval/stress-test/Makefile — TEE enclave lifecycle stress test suite
#
# Usage:
#   make help           Show this help
#   make install        Load the TEE driver (insmod)
#   make stress         Run batch enclave stress test (2/20/200/2000/20000)
#   make uninstall      Unload the TEE driver
#
# See also: /eval/cache-probe-exploit/ for side-channel probe & exploit tests.

DRV      = ../tee_enclave_drv.ko
STRESS   = tee_stress
PAYLOAD  = stress_payload

TEE_BIN_DIR := /eval/stress-test

.PHONY: help install stress uninstall

help:
	@echo "=== /eval TEE Stress Test Suite ==="
	@echo ""
	@echo "  make stress          Batch enclave lifecycle stress (2/20/200/2000/20000)"
	@echo ""
	@echo "  make install         Load driver (insmod)"
	@echo "  make uninstall       Unload driver (rmmod)"

install:
	@if [ -c /dev/tee_enclave ]; then \
		echo "[eval] driver already loaded (/dev/tee_enclave exists)"; \
	else \
		echo "[eval] loading driver..."; \
		/sbin/insmod "$(TEE_BIN_DIR)/$(DRV)" && echo "[eval] driver loaded. /dev/tee_enclave ready."; \
	fi

stress: install
	@echo "=== Enclave Lifecycle Stress Test ==="
	@echo "  batches: 2 -> 20 -> 200 -> 2000 -> 20000"
	@echo ""
	"$(TEE_BIN_DIR)/$(STRESS)" "$(TEE_BIN_DIR)/$(PAYLOAD)"

uninstall:
	@/sbin/rmmod tee_enclave_drv 2>/dev/null && echo "[eval] driver unloaded" || true
