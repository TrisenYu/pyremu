# ============================================================
#  pyremu/makefile — 从 configs.mk 生成 pyremu/configs_gen.py
# ============================================================
CONFIG_GEN = configs_gen.py

$(CONFIG_GEN): ../configs.mk ../emu-configs.mk FORCE
	@rm -f $@.tmp
	@echo '# generated from configs.mk by Makefile — do not edit' >> $@.tmp
	@echo '# mem layout' >> $@.tmp
	@echo 'RESERVED_MEM_BASE = $(RESERVED_MEM_BASE)' >> $@.tmp
	@echo 'RESERVED_MEM_SIZE = $(RESERVED_MEM_SIZE)' >> $@.tmp
	@echo '' >> $@.tmp
	@echo '# arguments for emulator' >> $@.tmp
	@echo 'TLB_ENTRIES = $(TLB_ENTRIES)' >> $@.tmp
	@echo 'NATIVE_MAX_INSTRS = $(NATIVE_MAX_INSTRS)' >> $@.tmp
# mcycle 计数频率
	@echo 'CPU_FREQ_HZ = $(CPU_FREQ_HZ)' >> $@.tmp
	@echo '' >> $@.tmp
# 诊断用相关定义
	@echo 'PYREMU_NATIVE_SPEEDUP = $(PYREMU_NATIVE_SPEEDUP)' >> $@.tmp
	@echo 'PYREMU_DIAG_LOG = "$(PYREMU_DIAG_LOG)"' >> $@.tmp
	@echo 'PYREMU_DIAG_VERBOSE = $(PYREMU_DIAG_VERBOSE)' >> $@.tmp
	@echo 'PYREMU_TRACE_SRET = $(PYREMU_TRACE_SRET)' >> $@.tmp
	@echo 'PYREMU_TRACE_TRAPS = $(PYREMU_TRACE_TRAPS)' >> $@.tmp
	@echo 'PYREMU_TRACE_PMP = $(PYREMU_TRACE_PMP)' >> $@.tmp
	@echo 'PYREMU_AIA_DIAG_INTERVAL = $(PYREMU_AIA_DIAG_INTERVAL)' >> $@.tmp
# 中断子系统与虚拟机扩展相关
	@echo '' >> $@.tmp
	@echo 'PYREMU_AIA = $(PYREMU_AIA)' >> $@.tmp
	@echo 'IMSIC_M_BASE = $(IMSIC_M_BASE)' >> $@.tmp
	@echo 'IMSIC_S_BASE = $(IMSIC_S_BASE)' >> $@.tmp
	@echo 'APLIC_BASE = $(APLIC_BASE)' >> $@.tmp
	@echo 'WFI_WATCHDOG_MS = $(WFI_WATCHDOG_MS)' >> $@.tmp
	@echo 'PYREMU_H_EXT = $(PYREMU_H_EXT)' >> $@.tmp
	@echo 'PYREMU_NO_L2 = $(PYREMU_NO_L2)' >> $@.tmp
# 启动所需二进制相关路径
	@echo '' >> $@.tmp
	@echo 'BINS_DIR = "$(bins_dir)"' >> $@.tmp
	@echo 'ELF_DIR = "$(elf_dir)"' >> $@.tmp
	@echo 'FIRM_DIR = "$(firm_dir)"' >> $@.tmp
	@cmp -s $@.tmp $@ 2>/dev/null && rm -f $@.tmp || mv -f $@.tmp $@
	@echo 'refresh configs_gen.py'

FORCE:
.PHONY: FORCE
