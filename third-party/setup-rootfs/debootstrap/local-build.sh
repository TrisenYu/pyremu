#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/07/09 星期四 17:57:40
# Last modified at 2026/07/09 星期四 22:25:08
set -euo pipefail

SUITE="trixie"
ROOTFS_DIR="./tmp-rootfs" # relative path to this shell script
MIRROR="https://mirrors.tuna.tsinghua.edu.cn/debian"
EXT4_IMG_SIZE_MB=4096
EXT4_IMG_FILE="./riscv-sd.ext4"
CPIO_OUT="./debian-riscv-initrd.cpio.gz"
HOSTNAME="riscv-trixie-sd"
ROOT_PASSWD="Password..."

cleanup_mount() {
    mountpoint -q "${ROOTFS_DIR}/tmp" && umount -l "${ROOTFS_DIR}/tmp"
    mountpoint -q "${ROOTFS_DIR}/dev/pts" && umount -l "${ROOTFS_DIR}/dev/pts"
    mountpoint -q "${ROOTFS_DIR}/dev" && umount -l "${ROOTFS_DIR}/dev"
    mountpoint -q "${ROOTFS_DIR}/sys" && umount -l "${ROOTFS_DIR}/sys"
    mountpoint -q "${ROOTFS_DIR}/proc" && umount -l "${ROOTFS_DIR}/proc"
	echo "unmount done"
}

# 1. executed by root
if [ "$(id -u)" -ne 0 ]; then
	echo -e "\033[31mrequire root privilege\033[0m"
	exit 1
fi
# 2. /etc/os-release and debian
if [ ! -f "/etc/os-release" ]; then
    echo -e "\033[31m[ERROR] require debian-distribution. /etc/os-release is lacked. \033[0m"
    exit 1
fi
. "/etc/os-release"
if [[ "${ID}" != "debian" && ! "${ID_LIKE:-}" =~ debian ]]; then
    echo -e "\033[31m[ERROR] unable to execute this shell script\033[0m"
    echo "recognize distribution as: ${PRETTY_NAME}"
    echo "only support Debian / Ubuntu / Kali / PopOS"
    exit 1
fi

trap cleanup_mount EXIT

# 3. environment setup
apt update && apt install -y debootstrap qemu-user-static binfmt-support e2fsprogs util-linux
if [ -d "${ROOTFS_DIR}" ]; then
    echo ">> clean up pre-existed ${ROOTFS_DIR}"
    rm -rf "${ROOTFS_DIR}"
fi
mkdir -p "${ROOTFS_DIR}"

# 4. compile by standard interpreter and fetch essential dependecies
QEMU_BIN="/usr/bin/qemu-riscv64-static"
if [ ! -f "${QEMU_BIN}" ]; then
    QEMU_BIN="$(which qemu-riscv64-static 2>/dev/null || true)"
fi
if [ -z "${QEMU_BIN}" ] || [ ! -f "${QEMU_BIN}" ]; then
    echo -e "\033[31m[ERROR] qemu-riscv64-static not found.\033[0m"
    echo "Install it with: apt install qemu-user-static"
    exit 1
fi
echo ">> qemu-riscv64-static: ${QEMU_BIN}"

debootstrap --arch=riscv64 --foreign --variant=minbase "${SUITE}" "${ROOTFS_DIR}" "${MIRROR}"
# 将 qemu 静态二进制放入 debootstrap 创建的目录结构中,
# 确保 chroot 内可执行 riscv64 二进制.
# 顺序必须在 debootstrap --foreign 之后 (它创建 base dirs),
# 在 chroot --second-stage 之前 (它需要 qemu 来运行 riscv 程序).
install -D -m755 "${QEMU_BIN}" "${ROOTFS_DIR}/usr/bin/qemu-riscv64-static"
DEBIAN_FRONTEND=noninteractive LANG=C chroot "${ROOTFS_DIR}" /debootstrap/debootstrap --second-stage

# 5. mount devices in host
mount -t proc none "${ROOTFS_DIR}/proc"
mount -t sysfs none "${ROOTFS_DIR}/sys"
mount --bind /dev "${ROOTFS_DIR}/dev"
mount --bind /dev/pts "${ROOTFS_DIR}/dev/pts"
mount --bind /tmp "${ROOTFS_DIR}/tmp"

# 6. setup basic environment
LANG=C DEBIAN_FRONTEND=noninteractive chroot "${ROOTFS_DIR}" /bin/bash <<EOF
# replace the deb sources.
cat > /etc/apt/sources.list <<'SRC'
deb ${MIRROR} trixie main contrib non-free non-free-firmware
deb ${MIRROR} trixie-updates main contrib non-free non-free-firmware
deb ${MIRROR}-security trixie-security main contrib non-free non-free-firmware
SRC

apt update -y
apt install -y dialog libterm-readline-perl-perl systemd gcc build-essential \
	flex bison vim python3 libc6 zsh
apt upgrade -y && apt clean
rm -rf /var/cache/apt/archives/*

echo "root:${ROOT_PASSWD}" | chpasswd

echo "${HOSTNAME}" > /etc/hostname
cat > /etc/hosts <<'HOST'
127.0.0.1   localhost ${HOSTNAME}
::1         localhost
HOST

cat > /etc/fstab <<'FS'
proc    /proc   proc    defaults    0 0
sysfs   /sys    sysfs   defaults    0 0
devtmpfs /dev  devtmpfs defaults    0 0
tmpfs   /tmp    tmpfs   size=64M    0 0
tmpfs   /var/run tmpfs  defaults    0 0
FS
EOF

# 7. cleanup host binary and devices
rm -f "${ROOTFS_DIR}/usr/bin/qemu-riscv64-static"
cleanup_mount

# 8. cpio for initramfs
pushd "${ROOTFS_DIR}" >/dev/null
find . -print0 | cpio --null -ov --format=newc | gzip -9 > "../${CPIO_OUT}"
popd
echo ">> initramfs: ${CPIO_OUT}"

# 9. ext4 file-system used in block device
dd if=/dev/zero of="${EXT4_IMG_FILE}" bs=1M count="${EXT4_IMG_SIZE_MB}" status=none
mkfs.ext4 -F "${EXT4_IMG_FILE}"

mkdir -p mnt-tmp
mount -o loop "${EXT4_IMG_FILE}" ./mnt-tmp
cp -r "${ROOTFS_DIR}"/* ./mnt-tmp/
umount ./mnt-tmp
rmdir mnt-tmp

echo "well done..."
