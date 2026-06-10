phony =

# 需要将测试目录下的源码通过专用工具链编译成二进制文件

## make CROSS_COMPILE=riscv64-linux-gnu- CC=/opt/custom-llvm/bin/clan
# CFLAGS=-I. -O2 -ggdb -march=rv64imafdc -mabi=lp64d -Wall -mcmodel=medany -ffreestanding -nostdlib -fno-pic
# -mexplicit-relocs -fno-tree-loop-distribute-patterns
# CCASFLAGS=-I. -mcmodel=medany -ffreestanding -nostdlib -fno-pic # -mexplicit-relocs
# LDFLAGS=-nostdlib -nostartfiles -static -Wl,--no-dynamic-linker

test:
	pytest
phony += test

cov-test:
	pytest --cov=. --cov-report=term
phony += cov-test

opt-cc=/opt/custom-llvm/bin/clang
opt-src-dir=$(pwd)/tests/src/
opt-bin-dir=$(pwd)/tests/bins/
custom-build:
	$(opt-cc) -c -O2 -o 
phony += custom-build

.PHONY: $(phony)