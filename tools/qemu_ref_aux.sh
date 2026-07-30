# -s: 在 TCP :1234 开启 GDB client
# -S: 启动时暂停 CPU，等待附着的GDB
# -monitor: QEMU monitor (telnet :5555) 用于查 TLB
qemu-system-riscv64 \
  -machine virt \
  -m 2G -smp 1 \
  -kernel third-party/linux/arch/riscv/boot/Image \
  -drive file=third-party/setup-rootfs/debootstrap/riscv-sd.ext4,format=raw,if=none,id=drv0 \
  -device virtio-blk-device,drive=drv0 \
  -append "earlycon=sbi console=ttyS0 root=/dev/vda rw init=/bin/zsh norandmaps" \
  -nographic

# -s -S \
#   -monitor tcp:127.0.0.1:5555,server,nowait

# 用 RISC-V GDB 连接
# /opt/custom-llvm/bin/llvm-gdb \
#   -ex "target remote :1234" \
#   third-party/linux/vmlinux
