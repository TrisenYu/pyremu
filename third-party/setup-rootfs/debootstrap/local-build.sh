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

# ---- 终端渲染颜色 (用法: echo -e "${RED}error${NC}") ----
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
NC='\033[0m' # No Color

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
	echo -e "${RED}require root privilege${NC}"
	exit 1
fi
# 2. /etc/os-release and debian
if [ ! -f "/etc/os-release" ]; then
    echo -e "${RED}[ERROR] require debian-distribution. /etc/os-release is lacked.${NC}"
    exit 1
fi
. "/etc/os-release"
if [[ "${ID}" != "debian" && ! "${ID_LIKE:-}" =~ debian ]]; then
    echo -e "${RED}[ERROR] unable to execute this shell script${NC}"
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
    echo -e "${RED}[ERROR] qemu-riscv64-static not found.${NC}"
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
apt install -y dialog libterm-readline-perl-perl systemd systemd-sysv \
	gcc build-essential flex bison vim python3 libc6 zsh git curl wget \
		clang llvm lld kmod
apt clean
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

# ---- zsh will be set as default shell ----
chsh -s /bin/zsh root

# ---- oh-my-zsh ----
# RUNZSH=no   zsh
# CHSH=no     shell
# Gitee for Chinese Networking Environment:
#   OH_MY_ZSH_URL=https://gitee.com/mirrors/oh-my-zsh/raw/master/tools/install.sh
#   ZSH_PLUGIN_PREFIX=https://gitee.com/mirrors
OH_MY_ZSH_URL="\${OH_MY_ZSH_URL:-https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh}"
curl -fsSL "\${OH_MY_ZSH_URL}" | RUNZSH=no CHSH=no sh

# ---- plugins ----
ZSH_PLUGIN_PREFIX="\${ZSH_PLUGIN_PREFIX:-https://github.com}"
ZSH_CUSTOM="/root/.oh-my-zsh/custom"
git clone --depth=1 \
	"\${ZSH_PLUGIN_PREFIX}/zsh-users/zsh-autosuggestions" \
	"\${ZSH_CUSTOM}/plugins/zsh-autosuggestions"
git clone --depth=1 \
	"\${ZSH_PLUGIN_PREFIX}/zsh-users/zsh-syntax-highlighting" \
	"\${ZSH_CUSTOM}/plugins/zsh-syntax-highlighting"

if [ ! -f /root/.oh-my-zsh/templates/zshrc.zsh-template ]; then
	mkdir -p /root/.oh-my-zsh/templates
	echo "export ZSH=\"/root/.oh-my-zsh\"" > /root/.oh-my-zsh/templates/zshrc.zsh-template
	echo "ZSH_THEME=\"gentoo\"" >> /root/.oh-my-zsh/templates/zshrc.zsh-template
	echo "plugins=(git)" >> /root/.oh-my-zsh/templates/zshrc.zsh-template
	echo "source \$ZSH/oh-my-zsh.sh" >> /root/.oh-my-zsh/templates/zshrc.zsh-template
fi
cp /root/.oh-my-zsh/templates/zshrc.zsh-template /root/.zshrc

sed -i "s/^plugins=(git)$/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/" \
	/root/.zshrc || true

sed -i "s/^ZSH_THEME=\".*\"$/ZSH_THEME=\"gentoo\"/" /root/.zshrc || true

EOF

# 7. toy benchmark programs -> /eval/toy-progs
TOY_SRC="$(cd "$(dirname "$0")/../../../tests/src-rv8" && pwd)"
TOY_DEST="${ROOTFS_DIR}/eval/toy-progs"
mkdir -p "${TOY_DEST}"

CROSS_CC=""
# prefer clang >=16, fall back to riscv64-linux-gnu-gcc
for cc in clang-19 clang-18 clang-17 clang-16 clang riscv64-linux-gnu-gcc; do
    if command -v "$cc" >/dev/null 2>&1; then
        CROSS_CC="$cc"
        break
    fi
done

if [ -n "${CROSS_CC}" ] && [ -f "${TOY_SRC}/Makefile" ]; then
    echo ">> Cross-compiling toy programs with ${CROSS_CC} ..."
    case "${CROSS_CC}" in
        clang*)
            make -C "${TOY_SRC}" -j"$(nproc)" \
                CC="${CROSS_CC}" CXX="${CROSS_CC}++" STRIP="llvm-strip" || true
            ;;
        *)
            make -C "${TOY_SRC}" -j"$(nproc)" \
                CC="${CROSS_CC}" CXX="${CROSS_CC/cc/++}" STRIP="${CROSS_CC/gcc/strip}" || true
            ;;
    esac
    # install stripped binaries, rename (drop _O3_stripped suffix)
    for f in "${TOY_SRC}/bin/riscv64/"*_stripped.o; do
        [ -f "$f" ] || continue
        name="$(basename "$f" .o)"
        install -m755 "$f" "${TOY_DEST}/${name}"
    done
    # install unstripped copies for debugging
    for f in "${TOY_SRC}/bin/riscv64/"*.o; do
        [ -f "$f" ] || continue
        case "$(basename "$f")" in *_stripped.o) continue ;; esac
        name="$(basename "$f" .o)"
        [ -f "${TOY_DEST}/${name}_stripped" ] || install -m755 "$f" "${TOY_DEST}/${name}"
    done
else
    echo ">> No cross-compiler, installing prebuilt binaries ..."
    cp -v "${TOY_SRC}/bin/riscv64/"*_stripped.o "${TOY_DEST}/" 2>/dev/null || true
fi

echo ">> Toy programs installed:"
ls -la "${TOY_DEST}"

# 8. TEE enclave driver + userspace test programs
echo ">> Building TEE enclave driver & test programs ..."
TEE_DRV_SRC="$(cd "$(dirname "$0")/../../../third-party/tee_enclave_drv" && pwd)"
TEE_TEST_SRC="$(cd "$(dirname "$0")/../../../tests/src-sidecache" && pwd)"
TEE_DEST="${ROOTFS_DIR}/eval/tee-test"
mkdir -p "${TEE_DEST}"

# Build kernel module (requires linux tree at ../../linux)
KDIR="$(cd "$(dirname "$0")/../../../third-party/linux" && pwd)"
CLANG_CC="/opt/custom-llvm/bin/clang"
if [ -x "${CLANG_CC}" ] && [ -d "${KDIR}" ]; then
    make -C "${KDIR}" M="${TEE_DRV_SRC}" ARCH=riscv \
        CC="${CLANG_CC}" LD="${CLANG_CC/clang/ld.lld}" \
        STRIP="${CLANG_CC/clang/llvm-strip}" modules 2>&1 | tail -3
    cp -v "${TEE_DRV_SRC}/tee_enclave_drv.ko" "${TEE_DEST}/"
else
    echo ">> SKIP driver build: clang=${CLANG_CC} kdir=${KDIR}"
fi

# Build userspace test programs (static, with riscv64-linux-gnu-gcc)
GCC_CROSS="riscv64-linux-gnu-gcc"
if command -v "${GCC_CROSS}" >/dev/null 2>&1; then
    for prog in tee_test tee_malice; do
        src="${TEE_TEST_SRC}/${prog}.c"
        if [ -f "${src}" ]; then
            ${GCC_CROSS} -static -O2 -Wall -I "${TEE_TEST_SRC}" \
                "${src}" -o "${TEE_DEST}/${prog}" && \
                echo "  ${prog}  OK"
        fi
    done
    # Also copy the header for reference
    cp -v "${TEE_TEST_SRC}/tee_enclave.h" "${TEE_DEST}/"
    # Copy pre-built enclave payloads
    for payload in hello_musl victim_musl probe_musl tlb_leak_musl; do
        [ -f "${TEE_TEST_SRC}/bin/${payload}" ] && \
            cp -v "${TEE_TEST_SRC}/bin/${payload}" "${TEE_DEST}/${payload}"
    done
    # symlink for eval.mk: hello_payload -> hello_musl
    ln -sf hello_musl "${TEE_DEST}/hello_payload"
    # Install the eval Makefile to /eval/
    EVAL_DEST="${ROOTFS_DIR}/eval"
    mkdir -p "${EVAL_DEST}"
    cp -v "${TEE_TEST_SRC}/eval.mk" "${EVAL_DEST}/Makefile"
else
    echo ">> SKIP test programs: ${GCC_CROSS} not found"
fi

echo ">> TEE components:"
ls -la "${TEE_DEST}"

# 9. cleanup host binary and devices
rm -f "${ROOTFS_DIR}/usr/bin/qemu-riscv64-static"
cleanup_mount

pushd "${ROOTFS_DIR}" >/dev/null
find . -print0 | cpio --null -ov --format=newc | gzip -9 > "../${CPIO_OUT}"
popd
# 10. cpio for initramfs
echo ">> initramfs: ${CPIO_OUT}"

# 11. ext4 file-system used in block device
dd if=/dev/zero of="${EXT4_IMG_FILE}" bs=1M count="${EXT4_IMG_SIZE_MB}" status=none
mkfs.ext4 -F "${EXT4_IMG_FILE}"

mkdir -p mnt-tmp
mount -o loop "${EXT4_IMG_FILE}" ./mnt-tmp
cp -r "${ROOTFS_DIR}"/* ./mnt-tmp/
umount ./mnt-tmp
rmdir mnt-tmp

# 12. 将输出文件的所有权归还给调用 sudo 的原始用户,
#     避免 emulator 以普通用户运行时因 PermissionError
#     回退到 O_RDONLY ->ext4 journal 无法 replay ->文件系统不可写.
if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
	chown "${SUDO_USER}:${SUDO_USER}" "${EXT4_IMG_FILE}" "${CPIO_OUT}"
	echo ">> ownership of output files returned to ${SUDO_USER}"
fi

echo "well done..."
