#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/07/09 星期四 17:57:40
# Last modified at 2026/07/09 星期四 22:25:08
set -euo pipefail

SUITE="trixie"
ROOTFS_DIR="./tmp-rootfs" # relative path to this shell script
MIRROR="https://mirrors.aliyun.com/debian"
EXT4_IMG_SIZE_MB=4096
EXT4_IMG_FILE="./riscv-sd.ext4"
CPIO_OUT="./debian-riscv-initrd.cpio.gz"
HOSTNAME="riscv-trixie-sd"
ROOT_PASSWD="Password..."

RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
NC='\033[0m' # No Color

GCC_CROSS="riscv64-linux-gnu-gcc"

QEMU_BIN="/usr/bin/qemu-riscv64-static"

TOY_SRC="$(cd "$(dirname "$0")/../../../tests/src-rv8" && pwd)"
TOY_DEST="${ROOTFS_DIR}/eval/toy-progs"
FNAPP_SRC="$(cd "$(dirname "$0")/../../../fn_apps" && pwd)"
TEE_DEST="${ROOTFS_DIR}/eval/cache-probe-exploit"
FNAPP_DEST="${TEE_DEST}/fn_apps"
STRESS_SRC="$(cd "$(dirname "$0")/../../../bsp/tee_aux_tools/stress-test" && pwd)"
STRESS_DEST="${ROOTFS_DIR}/eval/stress-test"
# Enclave modules: images and their manifest modules.list share /modules, read by
# both regress.sh and the module loader.
MODULE_SRC="$(cd "$(dirname "$0")/../../../bsp/sittim/modules" && pwd)"
MODULE_DEST="${ROOTFS_DIR}/modules"
# Module loader: a host program, deployed as /eval/ext-mod-loader together with its
# eval.mk (installed as that directory's Makefile).
EXT_MOD_SRC="$(cd "$(dirname "$0")/../../../bsp/tee_aux_tools/ext-mod-loader" && pwd)"
EXT_MOD_DEST="${ROOTFS_DIR}/eval/ext-mod-loader"

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
TEE_DRV_SRC="$(cd "$(dirname "$0")/../../../bsp/tee_aux_tools/linux-driver" && pwd)"
TEE_TEST_SRC="$(cd "$(dirname "$0")/../../../bsp/tee_aux_tools/cache-probe-exploit" && pwd)"
mkdir -p "${TEE_DEST}"

# Build kernel module (requires linux tree at ../../linux).
KDIR="$(cd "$(dirname "$0")/../../../bsp/linux" && pwd)"
if command -v "${GCC_CROSS}" >/dev/null 2>&1 && [ -d "${KDIR}" ]; then
    make -C "${KDIR}" M="${TEE_DRV_SRC}" ARCH=riscv \
        CROSS_COMPILE="${GCC_CROSS/gcc/}" modules 2>&1 | tail -3
    cp -v "${TEE_DRV_SRC}/tee_enclave_drv.ko" "${ROOTFS_DIR}/eval/"
else
    echo ">> SKIP driver build: gcc=${GCC_CROSS} kdir=${KDIR}"
fi

# Build via Makefile (single source of truth for program list),
# then copy all resulting binaries + payloads at once.
# require to build musl payload (hello_payload / victim_cache / attacker_cache):
# Otherwise, Makefile under /eval will execute as expected.
if command -v "${GCC_CROSS}" >/dev/null 2>&1; then
    make -C "${TEE_TEST_SRC}" -j"$(nproc)" CROSS_CC="${GCC_CROSS}" all musl
    install -m755 "${TEE_TEST_SRC}"/bin/* "${TEE_DEST}/"
    cp -v "${TEE_TEST_SRC}/tee_enclave.h" "${TEE_DEST}/"
    cp -v "${TEE_TEST_SRC}/eval.mk" "${TEE_DEST}/Makefile"
else
    echo ">> SKIP test programs: ${GCC_CROSS} not found"
fi

# 8b. fn_apps enclave payloads (all 7), loaded at runtime by tee_ecall_regress.
# depends on cargo + riscv64gc-unknown-linux-musl + vendor/musl submodule.
# Kept in a dedicated fn_apps/ subdir: tee_ecall_regress scans the whole dir as
# payloads, so it must be isolated from the host programs (cross_enclave_cache /
# tee_test / ...) and musl probe payloads (hello_payload / victim_*).
#
# ensure root can compile rust code when lacking of rust environment
CARGO_BIN="$(command -v cargo 2>/dev/null || true)"
INVOKER_HOME=""
if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
    INVOKER_HOME="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
fi
if [ -z "${CARGO_BIN}" ] && [ -n "${INVOKER_HOME}" ] && [ -x "${INVOKER_HOME}/.cargo/bin/cargo" ]; then
    CARGO_BIN="${INVOKER_HOME}/.cargo/bin/cargo"
fi

# Run cargo with the invoker's HOME: root's HOME holds no rustup toolchain.
# Shared by fn_apps, the enclave modules and the module signer.
CARGO_ENV=()
if [ -n "${INVOKER_HOME}" ]; then
    CARGO_ENV=(env HOME="${INVOKER_HOME}" RUSTUP_HOME="${INVOKER_HOME}/.rustup" \
        CARGO_HOME="${INVOKER_HOME}/.cargo")
fi

if [ -n "${CARGO_BIN}" ]; then
    echo ">> Building fn_apps enclave payloads (cargo=${CARGO_BIN}) ..."
    if "${CARGO_ENV[@]}" make -C "${FNAPP_SRC}" CARGO="${CARGO_BIN}" all 2>&1 | tail -8; then
        mkdir -p "${FNAPP_DEST}"
        install -m755 "${FNAPP_SRC}"/bin/* "${FNAPP_DEST}/"
        # dir for testcases
		mkdir -p "${FNAPP_DEST}/test"
        install -m644 "${FNAPP_SRC}"/chibicc/test/inp_src.c "${FNAPP_DEST}/test/inp_src.c"
        echo ">> fn_apps payloads installed"
    else
        echo ">> WARN: fn_apps build failed; skipping enclave payloads"
    fi
else
    echo ">> SKIP fn_apps: cargo not found"
fi

LANGLANDS_SYSROOT="${FNAPP_SRC}/Langlands/build/sysroot/lib/libflint.a"
if [ -f "${LANGLANDS_SYSROOT}" ]; then
    echo ">> Building Langlands payloads ..."
    if make -C "${FNAPP_SRC}" langlands 2>&1 | tail -4; then
        mkdir -p "${FNAPP_DEST}"
        # make can succeed without producing either payload; regress.sh scans the
        # payload directory, so a missing one silently leaves the summary incomplete.
        for f in Lfunc modular_form; do
            if [ ! -f "${FNAPP_SRC}/bin/${f}" ]; then
                echo ">> ERROR: Langlands build produced no ${FNAPP_SRC}/bin/${f}" >&2
                exit 1
            fi
        done
        install -m755 "${FNAPP_SRC}/bin/Lfunc" "${FNAPP_SRC}/bin/modular_form" \
            "${FNAPP_DEST}/"
        echo ">> Langlands payloads installed"
    else
        echo ">> WARN: Langlands build failed; skipping"
    fi
else
    echo ">> SKIP Langlands: run fn_apps/Langlands/build-deps.sh first"
fi

# 8c. Enclave module images (bsp/sittim/modules). The host hands a module to the
# enclave while it runs, so images are installed apart from the payloads: every
# image sits in /modules next to the manifest modules.list, which names the files
# relative to that directory. The make target `dist` writes that flattened copy.
if [ -n "${CARGO_BIN}" ]; then
    echo ">> Building enclave modules (cargo=${CARGO_BIN}) ..."
    "${CARGO_ENV[@]}" make -C "${MODULE_SRC}" CARGO="${CARGO_BIN}" dist
    mkdir -p "${MODULE_DEST}"
    install -m644 "${MODULE_SRC}"/dist/* "${MODULE_DEST}/"
    echo ">> Enclave modules installed:"
    ls -la "${MODULE_DEST}"
else
    echo ">> SKIP enclave modules: cargo not found"
fi

# 8d. Module loader: the smallest host program that drives the load path on its own.
if command -v "${GCC_CROSS}" >/dev/null 2>&1; then
    make -C "${EXT_MOD_SRC}" CROSS_CC="${GCC_CROSS}" all
    mkdir -p "${EXT_MOD_DEST}"
    install -m755 "${EXT_MOD_SRC}/bin/ext_mod_loader" "${EXT_MOD_DEST}/"
    cp -v "${EXT_MOD_SRC}/eval.mk" "${EXT_MOD_DEST}/Makefile"
else
    echo ">> SKIP module loader: ${GCC_CROSS} not found"
fi

# 8e. Headless regression entry — the headless test boots with init=/eval/regress.sh
# to auto-run the ecall regression (insmod driver + tee_ecall_regress + tee_concurrent);
# its [regress] ... markers are asserted from UART.
cat > "${ROOTFS_DIR}/eval/regress.sh" <<'REGRESS'
#!/bin/sh
echo "Regress init starting"
# This script is the kernel init, so nothing mounts /etc/fstab on its behalf;
# without this /proc and /dev stay empty, and both counting the online harts and
# creating the /dev/tee_enclave node fail.
mount -a
/sbin/insmod /eval/tee_enclave_drv.ko 2>&1
cd /eval/cache-probe-exploit || exit 1

# Stage 1: one hart loads every payload under fn_apps, one enclave at a time.
# The second argument is the module directory: payloads that request a module are
# served from it while the regression keeps resuming the enclave. It is omitted
# when the build installed no modules, because the regression rejects a directory
# it cannot read the manifest from.
if [ -d /modules ]; then
	set -- /modules
else
	set --
fi
./tee_ecall_regress /eval/cache-probe-exploit/fn_apps "$@"
rc_seq=$?
echo "Regress sequential rc=$rc_seq"

# Stage 2: several harts enter their own enclaves concurrently and run the same
# payload.  orbit is chosen because it exits by itself with argc = 0 (ENTER from
# tee_concurrent passes no arguments) and stays within the regression's time
# budget.
CONC_PAYLOAD=/eval/cache-probe-exploit/fn_apps/orbit

# One child per online hart, capped at 4.
harts=$(grep -c ^processor /proc/cpuinfo 2>/dev/null)
case "$harts" in
'' | *[!0-9]*) harts=1 ;;
esac
if [ "$harts" -gt 4 ]; then
	harts=4
fi

rc_conc=0
if [ ! -f "$CONC_PAYLOAD" ]; then
	echo "Regress concurrent skipped: $CONC_PAYLOAD not found"
elif [ "$harts" -lt 2 ]; then
	# Two harts at least are required for the requests to overlap; on one hart
	# they can only run serially, which concurrent-two-harts rejects by design.
	echo "Regress concurrent skipped: $harts online hart(s)"
else
	echo "Regress concurrent starting (procs=$harts)"
	./tee_concurrent "$CONC_PAYLOAD" "$harts"
	rc_conc=$?
	echo "Regress concurrent rc=$rc_conc"
fi

if [ "$rc_seq" -eq 0 ] && [ "$rc_conc" -eq 0 ]; then
	echo "Regress summary: ALL PASS"
	exit 0
fi
echo "Regress summary: FAILED"
exit 1
REGRESS
chmod +x "${ROOTFS_DIR}/eval/regress.sh"

echo ">> TEE components:"
ls -la "${TEE_DEST}"

# TEE enclave stress test programs

mkdir -p "${STRESS_DEST}"

if command -v "${GCC_CROSS}" >/dev/null 2>&1; then
    make -C "${STRESS_SRC}" -j"$(nproc)" CROSS_CC="${GCC_CROSS}" all
    install -m755 "${STRESS_SRC}"/bin/* "${STRESS_DEST}/"
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
#   make modules               Load an enclave module through the standalone loader

.PHONY: help probe stress modules

help:
	@echo "=== /eval TEE Test Suite ==="
	@echo "  make probe    cache-probe-exploit (全部缓存侧信道: Flush+Reload + 生命周期 + 并发 + TLB/integrity)"
	@echo "  make stress   stress-test (batch enclave lifecycle 2/20/200/2000/20000)"
	@echo "  make modules  ext-mod-loader (one payload, one module from /modules)"
	@echo ""
	@echo "  cd cache-probe-exploit && make help   for attack details (含阻塞式 DoS: malice-avail)"
	@echo "  cd stress-test && make help           for stress test details"
	@echo "  cd ext-mod-loader && make help        for loader details"

probe:
	$(MAKE) -C cache-probe-exploit probe

stress:
	$(MAKE) -C stress-test stress

modules:
	$(MAKE) -C ext-mod-loader load
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
