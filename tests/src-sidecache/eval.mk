# /eval/Makefile -- TEE enclave driver test suite
#
# Usage:
#   make help          Show this help
#   make install       Load the TEE driver (insmod)
#   make benign        Legitimate enclave lifecycle demo
#   make malice-avail   DoS: exhaust enclave memory pool
#   make malice-conf    TLB side-channel probe
#   make malice-integ   Integrity boundary probing
#   make malice-all     Run all attacks
#   make uninstall     Unload the TEE driver

DRV    = tee_enclave_drv.ko
TEST   = tee_test
MALICE = tee_malice
PAYLOAD = hello_payload

D = /eval/tee-test

.PHONY: help install benign uninstall \
        malice-avail malice-conf malice-integ malice-all

help:
	@echo "=== /eval TEE Test Suite ==="
	@echo ""
	@echo "  make benign          Legitimate enclave lifecycle (query -> create -> enter -> shutdown)"
	@echo ""
	@echo "  make malice-avail    DoS:  exhaust enclave memory pool"
	@echo "  make malice-conf     TLB side-channel probe across mdid domains"
	@echo "  make malice-integ    Integrity probe (ID guessing, kernel VA injection)"
	@echo "  make malice-all      Run all attacks"
	@echo ""
	@echo "  make install          Load driver (insmod)"
	@echo "  make uninstall        Unload driver (rmmod)"
	@echo ""

install:
	@if [ -c /dev/tee_enclave ]; then \
		echo "[eval] driver already loaded (/dev/tee_enclave exists)"; \
	else \
		echo "[eval] loading driver..."; \
		/sbin/insmod $(D)/$(DRV) && echo "[eval] driver loaded. /dev/tee_enclave ready."; \
	fi

benign: install
	@echo "=== Benign Enclave Lifecycle Demo ==="
	@echo ""
	@echo "  step 1. query memory pool & current mdid"
	@echo "  step 2. create enclave (firmware loads built-in manager)"
	@echo "  step 3. enter enclave with custom payload"
	@echo "  step 4. enclave runs -> suspends -> back to host"
	@echo "  step 5. shutdown enclave (frees slot + memory)"
	@echo ""
	$(D)/$(TEST) $(D)/$(PAYLOAD)

malice-avail: install
	@echo "=== ATTACK: Memory Pool Exhaustion (DoS) ==="
	$(D)/$(MALICE) avail

malice-conf: install
	@echo "=== ATTACK: TLB Side-Channel Probe ==="
	$(D)/$(MALICE) conf

malice-integ: install
	@echo "=== ATTACK: Integrity Boundary Probe ==="
	$(D)/$(MALICE) integ

malice-all: install
	@echo "=== ALL ATTACKS ==="
	$(D)/$(MALICE) all

uninstall:
	@/sbin/rmmod tee_enclave_drv 2>/dev/null && echo "[eval] driver unloaded" || true
