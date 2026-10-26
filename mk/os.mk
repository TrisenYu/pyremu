# os.mk — 内核启动与根文件系统配置

# fw_payload 模式: Linux Image 嵌入 OpenSBI, 经 M 模式向 S 模式移交后直接启动 (ZSBL 设 coldboot_done=1)。
# fw_jump 模式: OpenSBI 经 mret 跳至 0x80400000, 内核 Image 预载该地址, vmlinux 供调试符号。
# 该地址等于 FW_RESERVED_MEM_BASE + FW_RESERVED_MEM_SIZE, 即固件映像保留区的上界。
LINUX_IMG        = $(linux_dir)/arch/riscv/boot/Image
LINUX_VMLINUX    = $(linux_dir)/vmlinux

# ---- initramfs / rootfs (debootstrap) ----
# ROOTFS_DIR: debootstrap 输出目录 (打包为 cpio.gz)。
# INITRD: 传给 emu-linux 的 initramfs 路径 (为空则不挂载 rootfs)。
#   用法: make emu-linux INITRD=$(INITRAMFS) NUM_HARTS=4 ram=2G
ROOTFS_DIR ?= $(bsp_dir)/rootfs
INITRAMFS   = $(bins_dir)/initramfs.cpio.gz
INITRD     ?=

# DISK: virtio-blk 磁盘镜像 (ext4/raw), 挂载为 /dev/vda。默认指向 debootstrap 生成的
#   rootfs; 文件不存在时自动跳过 (回退到无根文件系统, 即 VFS panic)。root 所有的镜像
#   自动只读打开。覆盖: make emu-linux DISK=/path/to/other.ext4
DISK       ?= $(bsp_dir)/setup-rootfs/debootstrap/riscv-sd.ext4
# ram / rdinit 可覆盖; 挂载 initramfs 时建议加大 RAM (全量驻留内存)。
# 6G: 飞地内存池 (RESERVED_MEM_SIZE, 见 mk/memory.mk) 自 0x83000000 起占用
# 2.25 GiB, 尾部到 ~4.30 GiB, 其余留给内核、根文件系统与 initramfs.
ram        ?= 6G
rdinit     ?= /bin/bash

# ---- Kernel bootargs ----
# dyndbg: dynamic debug 控制 (内核 pr_debug/dev_dbg), 可覆盖.
#   +p 启用全部, func NAME +p 按函数, file PATH +p 按文件, 留空关闭.
dyndbg         ?=
# func run_init_process +p
init           ?= /bin/zsh
bootargs_extra ?=
# norandmaps
# nokaslr norandmaps
_bootargs   = earlycon=sbi console=ttySIF0
# _bootargs  += dyndbg="$(dyndbg)"
_bootargs  += root=/dev/vda rw
_bootargs  += $(if $(init),init=$(init))
_bootargs  += $(if $(INITRD),rdinit=$(rdinit)) $(bootargs_extra)
bootargs_arg    = --bootargs='$(_bootargs)'
# INITRD 非空时追加 --initrd (rdinit 已在 bootargs 中).
initrd_args     = $(if $(INITRD),--initrd=$(INITRD))
# DISK 指向的镜像存在时追加 --disk.
disk_args       = $(if $(wildcard $(DISK)),--disk=$(DISK))
