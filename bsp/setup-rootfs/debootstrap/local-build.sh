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

RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
NC='\033[0m' # No Color

TOY_SRC="$(cd "$(dirname "$0")/../../../tests/src-rv8" && pwd)"
TOY_DEST="${ROOTFS_DIR}/eval/toy-progs"

cleanup_mount() {
    mountpoint -q "${ROOTFS_DIR}/tmp" && umount -l "${ROOTFS_DIR}/tmp"
    mountpoint -q "${ROOTFS_DIR}/dev/pts" && umount -l "${ROOTFS_DIR}/dev/pts"
    mountpoint -q "${ROOTFS_DIR}/dev" && umount -l "${ROOTFS_DIR}/dev"
    mountpoint -q "${ROOTFS_DIR}/sys" && umount -l "${ROOTFS_DIR}/sys"
    mountpoint -q "${ROOTFS_DIR}/proc" && umount -l "${ROOTFS_DIR}/proc"
	echo "unmount done"
}

# executed by root
if [ "$(id -u)" -ne 0 ]; then
	echo -e "${RED}require root privilege${NC}"
	exit 1
fi
# /etc/os-release and debian
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

# environment setup
apt update && apt install -y debootstrap qemu-user-static binfmt-support e2fsprogs util-linux
if [ -d "${ROOTFS_DIR}" ]; then
    echo ">> clean up pre-existed ${ROOTFS_DIR}"
    rm -rf "${ROOTFS_DIR}"
fi
mkdir -p "${ROOTFS_DIR}"

# compile by standard interpreter and fetch essential dependecies
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
# qemu for emulating the basic target environment
install -D -m755 "${QEMU_BIN}" "${ROOTFS_DIR}/usr/bin/qemu-riscv64-static"
DEBIAN_FRONTEND=noninteractive LANG=C chroot "${ROOTFS_DIR}" /debootstrap/debootstrap --second-stage

# mount devices in host
mount -t proc none "${ROOTFS_DIR}/proc"
mount -t sysfs none "${ROOTFS_DIR}/sys"
mount --bind /dev "${ROOTFS_DIR}/dev"
mount --bind /dev/pts "${ROOTFS_DIR}/dev/pts"
mount --bind /tmp "${ROOTFS_DIR}/tmp"

# setup basic environment
# ---- replace the deb sources
# ---- update host sources
# ---- zsh will be set as default shell
# ---- install oh-my-zsh
# ---- plugins ----
# RUNZSH=no   zsh
# CHSH=no     shell
# Gitee for Chinese Networking Environment:
#   OH_MY_ZSH_URL=https://gitee.com/mirrors/oh-my-zsh/raw/master/tools/install.sh
#   ZSH_PLUGIN_PREFIX=https://gitee.com/mirrors
LANG=C DEBIAN_FRONTEND=noninteractive chroot "${ROOTFS_DIR}" /bin/bash <<EOF
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

chsh -s /bin/zsh root

OH_MY_ZSH_URL="\${OH_MY_ZSH_URL:-https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh}"
curl -fsSL "\${OH_MY_ZSH_URL}" | RUNZSH=no CHSH=no sh

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

# toy benchmark programs -> /eval/toy-progs
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
TEE_DRV_SRC="$(cd "$(dirname "$0")/../../../bsp/tee_enclave_drv" && pwd)"
TEE_TEST_SRC="$(cd "$(dirname "$0")/../../../tests/src-sidecache" && pwd)"
TEE_DEST="${ROOTFS_DIR}/eval/cache-probe-exploit"
mkdir -p "${TEE_DEST}"

# Build kernel module (requires linux tree at ../../linux)
KDIR="$(cd "$(dirname "$0")/../../../bsp/linux" && pwd)"
CLANG_CC="/opt/custom-llvm/bin/clang"
if [ -x "${CLANG_CC}" ] && [ -d "${KDIR}" ]; then
    make -C "${KDIR}" M="${TEE_DRV_SRC}" ARCH=riscv \
        CC="${CLANG_CC}" LD="${CLANG_CC/clang/ld.lld}" \
        STRIP="${CLANG_CC/clang/llvm-strip}" modules 2>&1 | tail -3
    cp -v "${TEE_DRV_SRC}/tee_enclave_drv.ko" "${ROOTFS_DIR}/eval/"
else
    echo ">> SKIP driver build: clang=${CLANG_CC} kdir=${KDIR}"
fi

# Build via Makefile (single source of truth for program list),
# then copy all resulting binaries + payloads at once.
# 需要同时构建 musl 飞地载荷 (hello_payload / victim_cache / attacker_cache):
# cross_enclave_cache 探测与 benign/concurrent 生命周期均依赖它们, 只跑 all
# 会导致 /eval 下缺失载荷, probe 无法执行.
GCC_CROSS="riscv64-linux-gnu-gcc"
if command -v "${GCC_CROSS}" >/dev/null 2>&1; then
    make -C "${TEE_TEST_SRC}" -j"$(nproc)" CROSS_CC="${GCC_CROSS}" all musl
    install -m755 "${TEE_TEST_SRC}"/bin/* "${TEE_DEST}/" 2>/dev/null || true
    cp -v "${TEE_TEST_SRC}/tee_enclave.h" "${TEE_DEST}/"
    cp -v "${TEE_TEST_SRC}/eval.mk" "${TEE_DEST}/Makefile"
else
    echo ">> SKIP test programs: ${GCC_CROSS} not found"
fi

echo ">> TEE components:"
ls -la "${TEE_DEST}"

# TEE enclave stress test programs
STRESS_SRC="$(cd "$(dirname "$0")/../../../tests/src-stress" && pwd)"
STRESS_DEST="${ROOTFS_DIR}/eval/stress-test"
mkdir -p "${STRESS_DEST}"

if command -v "${GCC_CROSS}" >/dev/null 2>&1; then
    make -C "${STRESS_SRC}" -j"$(nproc)" CROSS_CC="${GCC_CROSS}" all
    install -m755 "${STRESS_SRC}"/bin/* "${STRESS_DEST}/" 2>/dev/null || true
    cp -v "${STRESS_SRC}/tee_enclave.h" "${STRESS_DEST}/"
    cp -v "${STRESS_SRC}/eval.mk" "${STRESS_DEST}/Makefile"
else
    echo ">> SKIP stress tests: ${GCC_CROSS} not found"
fi

echo ">> Stress test components:"
ls -la "${STRESS_DEST}"

# Top-level /eval/Makefile — delegates to subdirectories
cat > "${ROOTFS_DIR}/eval/Makefile" <<'EVALMK'
# /eval/Makefile — TEE test suite entry point
#
#   make help                  Show this help
#   make probe                 Side-channel probe & exploit tests
#   make stress                Batch enclave lifecycle stress tests

.PHONY: help probe stress

help:
	@echo "=== /eval TEE Test Suite ==="
	@echo "  make probe    cache-probe-exploit (全部缓存侧信道: Flush+Reload + 生命周期 + 并发 + TLB/integrity)"
	@echo "  make stress   stress-test (batch enclave lifecycle 2/20/200/2000/20000)"
	@echo ""
	@echo "  cd cache-probe-exploit && make help   for attack details (含阻塞式 DoS: malice-avail)"
	@echo "  cd stress-test && make help           for stress test details"

probe:
	$(MAKE) -C cache-probe-exploit probe

stress:
	$(MAKE) -C stress-test stress
EVALMK

echo ">> /eval/Makefile created"

# cleanup host binary and devices
rm -f "${ROOTFS_DIR}/usr/bin/qemu-riscv64-static"
cleanup_mount

# cpio for initramfs
pushd "${ROOTFS_DIR}" >/dev/null
find . -print0 | cpio --null -ov --format=newc | gzip -9 > "../${CPIO_OUT}"
popd
echo ">> initramfs: ${CPIO_OUT}"

# ext4 file-system used in block device
dd if=/dev/zero of="${EXT4_IMG_FILE}" bs=1M count="${EXT4_IMG_SIZE_MB}" status=none
mkfs.ext4 -F "${EXT4_IMG_FILE}"

mkdir -p mnt-tmp
mount -o loop "${EXT4_IMG_FILE}" ./mnt-tmp
cp -r "${ROOTFS_DIR}"/* ./mnt-tmp/
umount ./mnt-tmp
rmdir mnt-tmp

# set proper privilege via chown
if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
	chown "${SUDO_USER}:${SUDO_USER}" "${EXT4_IMG_FILE}" "${CPIO_OUT}"
	echo ">> ownership of output files returned to ${SUDO_USER}"
fi

echo "well done..."
