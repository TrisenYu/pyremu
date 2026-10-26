# ============================================================
#  pyremu/makefile — 从 mk/ 下的配置分片生成 pyremu/configs_gen.py
# ============================================================
CONFIG_GEN = configs_gen.py

$(CONFIG_GEN): $(wildcard ../mk/*.mk) FORCE
	@rm -f $@.tmp
	@echo '# generated from mk/*.mk by Makefile — do not edit' >> $@.tmp
	@echo '# mem layout' >> $@.tmp
	@echo 'RESERVED_MEM_BASE = $(RESERVED_MEM_BASE)' >> $@.tmp
	@echo 'RESERVED_MEM_SIZE = $(RESERVED_MEM_SIZE)' >> $@.tmp
	@echo 'FW_RESERVED_MEM_BASE = $(FW_RESERVED_MEM_BASE)' >> $@.tmp
	@echo 'FW_RESERVED_MEM_SIZE = $(FW_RESERVED_MEM_SIZE)' >> $@.tmp
	@echo '' >> $@.tmp
	@echo '# platform layout' >> $@.tmp
	@echo 'NUM_HARTS = $(NUM_HARTS)' >> $@.tmp
	@echo 'RAM_BASE = $(RAM_BASE)' >> $@.tmp
	@echo 'RAM_SIZE = $(RAM_SIZE)' >> $@.tmp
	@echo 'ROM_BASE = $(ROM_BASE)' >> $@.tmp
	@echo 'ROM_SIZE = $(ROM_SIZE)' >> $@.tmp
	@echo 'ROM_DTB_OFFSET = $(ROM_DTB_OFFSET)' >> $@.tmp
	@echo 'PROG_CNT = $(PROG_CNT)' >> $@.tmp
	@echo 'L2_SIZE = $(L2_SIZE)' >> $@.tmp
	@echo 'ISA = "$(ISA)"' >> $@.tmp
	@echo 'UART_BASE = $(UART_BASE)' >> $@.tmp
	@echo 'SPI_BASE = $(SPI_BASE)' >> $@.tmp
	@echo 'I2C_BASE = $(I2C_BASE)' >> $@.tmp
	@echo 'GPIO_BASE = $(GPIO_BASE)' >> $@.tmp
	@echo 'CLINT_BASE = $(CLINT_BASE)' >> $@.tmp
	@echo 'PLIC_BASE = $(PLIC_BASE)' >> $@.tmp
	@echo 'WATCHDOG_BASE = $(WATCHDOG_BASE)' >> $@.tmp
	@echo 'CRNG_BASE = $(CRNG_BASE)' >> $@.tmp
	@echo 'VIRTIO_BLK_BASE = $(VIRTIO_BLK_BASE)' >> $@.tmp
	@echo 'VIRTIO_NET_S_BASE = $(VIRTIO_NET_S_BASE)' >> $@.tmp
	@echo 'VIRTIO_NET_M_BASE = $(VIRTIO_NET_M_BASE)' >> $@.tmp
	@echo '' >> $@.tmp
	@echo '# arguments for emulator' >> $@.tmp
	@echo 'TLB_ENTRIES = $(TLB_ENTRIES)' >> $@.tmp
	@echo 'PMP_ENTRIES = $(PMP_ENTRIES)' >> $@.tmp
# mcycle 计数频率
	@echo 'CPU_FREQ_HZ = $(CPU_FREQ_HZ)' >> $@.tmp
# 时钟源频率
	@echo 'TIMER_FREQ = $(TIMER_FREQ)' >> $@.tmp
	@echo 'DEFAULT_TIMEOUT = $(DEFAULT_TIMEOUT)' >> $@.tmp
	@echo 'WFI_MAX_SLEEP = $(WFI_MAX_SLEEP)' >> $@.tmp
	@echo 'VIRTIO_PROCESS_BATCH = $(VIRTIO_PROCESS_BATCH)' >> $@.tmp
	@echo 'FAST_LOAD_THRESHOLD = $(FAST_LOAD_THRESHOLD)' >> $@.tmp
	@echo 'NS_PER_INSTR = $(NS_PER_INSTR)' >> $@.tmp
	@echo '' >> $@.tmp
# 中断子系统与虚拟机扩展相关
	@echo '' >> $@.tmp
	@echo 'PYREMU_AIA = $(PYREMU_AIA)' >> $@.tmp
	@echo 'IMSIC_M_BASE = $(IMSIC_M_BASE)' >> $@.tmp
	@echo 'IMSIC_S_BASE = $(IMSIC_S_BASE)' >> $@.tmp
	@echo 'APLIC_S_BASE = $(APLIC_S_BASE)' >> $@.tmp
	@echo 'APLIC_M_BASE = $(APLIC_M_BASE)' >> $@.tmp
	@echo 'WFI_WATCHDOG_MS = $(WFI_WATCHDOG_MS)' >> $@.tmp
	@echo 'PYREMU_H_EXT = $(PYREMU_H_EXT)' >> $@.tmp
	@echo 'PYREMU_NO_L2 = $(PYREMU_NO_L2)' >> $@.tmp
	@echo 'PYREMU_NO_TLB = $(PYREMU_NO_TLB)' >> $@.tmp
	@echo 'PYREMU_HAS_SEMIHOSTING = $(PYREMU_HAS_SEMIHOSTING)' >> $@.tmp
# 启动所需二进制相关路径
	@echo '' >> $@.tmp
	@echo 'BINS_DIR = "$(bins_dir)"' >> $@.tmp
	@echo 'ELF_DIR = "$(elf_dir)"' >> $@.tmp
	@echo 'FIRM_DIR = "$(firm_dir)"' >> $@.tmp
	@cmp -s $@.tmp $@ 2>/dev/null && rm -f $@.tmp || mv -f $@.tmp $@
	@echo 'refresh configs_gen.py'

FORCE:
.PHONY: FORCE
