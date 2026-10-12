include ../config.mk

BUILD := build
OUT := $(BUILD)/malloc_churn
LDFLAGS += -L$(BUILD)/lib

all: $(OUT)

$(BUILD)/lib/libgcc_eh.a: $(MUSL_LIB)/libgcc.a
	@mkdir -p $(@D)
	cp $< $@

$(OUT): src/t-test1.c $(BUILD)/lib/libgcc_eh.a
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) src/t-test1.c $(LDFLAGS) -o $@

run: $(OUT)
	$(QEMU) $(OUT) 100 2 10000 10000 32

clean:
	rm -rf $(BUILD)

.PHONY: all run clean
