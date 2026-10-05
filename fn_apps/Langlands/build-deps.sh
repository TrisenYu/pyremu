#!/bin/sh
# SPDX-LICENSE-IDENTIFIER: MIT
#
# 交叉编译 Langlands 载荷的数学库依赖, 产出 riscv64 musl 静态库。
#
#   GMP 6.3.0   任意精度整数与有理数, MPFR 与 FLINT 的底层依赖
#   MPFR 4.2.2  任意精度浮点
#   FLINT 3.6.0 数论与任意精度球算术(ball arithmetic)库, 含 arb (带误差界的实数) 与
#               acb (带误差界的复数), 以及 Dirichlet 特征、Dirichlet L 函数、模形式
#               (eta, j, 模判别式) 与 Ramanujan tau 函数等现成例程
#
# 源码来自仓库根的三个 git 子模块 vendor/gmp / vendor/mpfr / vendor/flint (见 .gitmodules)。
# GMP 官方只用 Mercurial 管理源码, 不提供 git 仓库, vendor/gmp 是上游 Mercurial 仓库的 git
# 镜像, 固定在 6.3.0 的发布提交上。
#
# 三棵源码树都不含生成好的 configure, 只有 configure.ac 与各自的引导入口, 故本脚本先引导
# 再构建 (GMP 走 autoreconf); 引导需要本机安装 autoconf、automake 与 libtool, 缺 libtool 时
# Debian 系执行 apt install libtool-bin。
#
# 库的构建在源码树之外进行 (VPATH 构建), 产物落在 build/ 之下; 源码树只被写入引导生成的
# 文件 (configure、Makefile.in 等), 需要时以 git submodule foreach git clean -xdf 清理。
#
# 产物 (位于 .gitignore 的 **/build/ 之下, 不进 git):
#   build/sysroot/  头文件与静态库, 由各载荷 Makefile 以 DEPS_SYSROOT 引用
#   build/gmp/  build/mpfr/  build/flint/  三个库的构建目录
#   build/logs/     各库的 configure 与编译日志
#
# 用法:
#   ./build-deps.sh                  从 vendor/{gmp,mpfr,flint} 构建
#   FORCE=1 ./build-deps.sh          重新引导与编译 (先删除 build/)
#   JOBS=N ./build-deps.sh           指定并行度, 默认取 nproc
#   SRC_ROOT=/path ./build-deps.sh   从其它位置的 gmp/ mpfr/ flint/ 源码树构建
#                                    (该目录下三个源码树的名称与本脚本一致; 用于验证脚本改动)

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

SRC_ROOT=${SRC_ROOT:-$REPO/vendor}
MUSL_SRC="$REPO/vendor/musl"
MUSL_SYSROOT="$REPO/bsp/musl-gc-sysroot"

BUILD="$HERE/build"
PREFIX="$BUILD/sysroot"
LOGS="$BUILD/logs"

TRIPLE=riscv64-linux-musl
JOBS=${JOBS:-$(nproc)}

# 交叉编译命令, 与 fn_apps/config.mk 一致:
#   -static         musl sysroot 只提供静态库, 链接测试程序须静态链接; clang 默认生成
#                  位置无关可执行文件, 会去找不存在的 crtbeginS.o, 亦由该选项避免
#   -fuse-ld=lld   本机的 GNU ld 不带 riscv64 目标, 必须显式使用 LLD
#   -nostdinc      阻断宿主 glibc 头文件, 改用 musl 源码头文件
#   -L$PREFIX/lib  提供 libgcc_eh.a 的替身副本 (见下), clang 驱动为目标链接隐式追加
#                  -lgcc_eh, 而 musl sysroot 只提供 libgcc.a
# clang 内建头目录须一并引入: musl 不提供 stdatomic.h, FLINT 依赖该头文件。
#
# 上面的 -fuse-ld=lld 只作用于经 clang 驱动发起的链接。FLINT 的构建规则另有一处直接调用
# $(LD) 把每个子目录的目标文件合并为单个可重定位目标文件, configure 探测到的是宿主
# GNU ld, 而它读不了 riscv64 目标文件 (报 file in wrong format), 故 FLINT 的 configure
# 另传 LD=ld.lld。ld.lld 按输入的目标文件识别体系结构, 不受宿主体系结构限制。
BUILTIN_INCLUDE=$(clang -print-resource-dir)/include
LD_TARGET=ld.lld

CC_TARGET="clang --target=$TRIPLE --sysroot=$MUSL_SYSROOT"
CC_TARGET="$CC_TARGET -march=rv64gc -mabi=lp64d -static -fuse-ld=lld -nostdinc"
CC_TARGET="$CC_TARGET -isystem $BUILTIN_INCLUDE"
CC_TARGET="$CC_TARGET -isystem $MUSL_SRC/include -isystem $MUSL_SRC/arch/riscv64"
CC_TARGET="$CC_TARGET -isystem $MUSL_SRC/arch/generic -isystem $MUSL_SRC/obj/include"
CC_TARGET="$CC_TARGET -L$PREFIX/lib"

# C++ 命令只用于 configure 的编译器探测, 三个库均为 C 库; 参数与
# fn_apps/json_calc/Makefile 的 C++ 载荷一致
CXX_TARGET="clang++ --target=$TRIPLE --sysroot=$MUSL_SYSROOT"
CXX_TARGET="$CXX_TARGET -march=rv64gc -mabi=lp64d -static -fuse-ld=lld -nostdinc++"
CXX_TARGET="$CXX_TARGET -isystem $BUILTIN_INCLUDE"
CXX_TARGET="$CXX_TARGET -isystem $MUSL_SYSROOT/include/c++/v1 -isystem $MUSL_SYSROOT/include"
CXX_TARGET="$CXX_TARGET -isystem $MUSL_SRC/include -isystem $MUSL_SRC/arch/riscv64"
CXX_TARGET="$CXX_TARGET -isystem $MUSL_SRC/arch/generic -isystem $MUSL_SRC/obj/include"
CXX_TARGET="$CXX_TARGET -L$PREFIX/lib"

# 传给 make 的额外变量, 默认空。取值为不含空白的单个赋值, 故 configure_and_make 内的展开
# 不会误分词; 目前只有 GMP 需要它 (见该库的构建处)。
MAKE_VARS=

# 子模块源码树缺 configure 时先引导。三个上游的引导入口不同:
# GMP 用 autoconf 体系, MPFR 用 autogen.sh, FLINT 用 bootstrap.sh。
bootstrap()
{
	dir=$1
	if [ -x "$SRC_ROOT/$dir/configure" ]; then
		return
	fi
	echo "引导 $dir ($SRC_ROOT/$dir)"
	(
		cd "$SRC_ROOT/$dir"
		if [ -x ./bootstrap.sh ]; then
			./bootstrap.sh
		elif [ -x ./autogen.sh ]; then
			./autogen.sh
		elif [ -f ./configure.ac ]; then
			autoreconf -i
		else
			echo "无法引导 $dir: 既无 configure 也无引导脚本" >&2
			exit 1
		fi
	) > "$LOGS/$dir-bootstrap.log" 2>&1
}

configure_and_make()
{
	dir=$1
	shift
	echo "构建 $dir"
	rm -rf "$BUILD/$dir"
	mkdir -p "$BUILD/$dir"
	(
		cd "$BUILD/$dir"
		"$SRC_ROOT/$dir/configure" --host=$TRIPLE --prefix="$PREFIX" \
			CC="$CC_TARGET" CXX="$CXX_TARGET" CFLAGS="-O2" "$@" \
			> "$LOGS/$dir-configure.log" 2>&1
		make -j"$JOBS" $MAKE_VARS > "$LOGS/$dir-make.log" 2>&1
		make install $MAKE_VARS >> "$LOGS/$dir-make.log" 2>&1
	)
}

# FLINT 的静态库把每个子目录的目标文件合并成单个可重定位目标文件后再打包, 链接时
# 会把整个子目录的代码全部拉进载荷, 载荷因此膨胀到十倍于实际所需。载荷须放入飞地的
# 2 MiB 内存分区, 故按单个目标文件重新打包, 使链接器只取用到的成员; 单个目标文件与
# 合并对象的内容一致, 均保留在构建目录中。
#
# ar 的 r 是替换式追加: 目标归档已存在时, 其中的成员不因未被列出而移除, 合并对象会
# 继续留在归档里并排在新成员之前, 链接器仍会拉取它们, 重新打包因此等于无效。故先删除
# make install 放入的归档再打包, 并在打包后校验归档中不含合并对象。
repack_flint_archive()
{
	echo "重新打包 libflint.a (按单个目标文件)"
	find "$BUILD/flint/build" -name '*.o' ! -name '*_merged.o' | sort \
		> "$LOGS/flint-objects.txt"
	rm -f "$PREFIX/lib/libflint.a"
	ar rcs "$PREFIX/lib/libflint.a" $(cat "$LOGS/flint-objects.txt")
	if ar t "$PREFIX/lib/libflint.a" | grep -q '_merged\.o$'; then
		echo "重新打包后 libflint.a 仍含合并对象成员" >&2
		exit 1
	fi
}

if [ -n "${FORCE:-}" ]; then
	echo "FORCE=1: 删除 $BUILD"
	rm -rf "$BUILD"
fi

# 已完成的构建与源码树绑定: 换用其它源码树时结果不可比, 一并重建。
stamp="$BUILD/.source-root"
if [ -d "$BUILD" ] && [ "$(cat "$stamp" 2>/dev/null || echo)" != "$SRC_ROOT" ]; then
	echo "源码树已改变, 删除 $BUILD 后重建"
	rm -rf "$BUILD"
fi

for d in gmp mpfr flint; do
	test -d "$SRC_ROOT/$d" || {
		echo "缺少依赖源码 $SRC_ROOT/$d" >&2
		echo "若是未检出的子模块, 请执行: git submodule update --init vendor/gmp vendor/mpfr vendor/flint" >&2
		exit 1
	}
done

test -f "$MUSL_SYSROOT/lib/libc.a" || {
	echo "缺少 musl sysroot: $MUSL_SYSROOT" >&2
	echo "请先执行 $REPO/bsp/build-musl-sysroot.sh" >&2
	exit 1
}

mkdir -p "$PREFIX/lib" "$LOGS"
echo "$SRC_ROOT" > "$stamp"

# clang 驱动为目标链接隐式追加 -lgcc_eh, 而 musl 只提供 libgcc.a; 以同名副本补齐,
# 使各库的 configure 链接测试与载荷链接都无需额外参数 (载荷 Makefile 已按
# knots 与 json_calc 的做法各自补齐, 此处供依赖构建阶段使用)。
cp "$MUSL_SYSROOT/lib/libgcc.a" "$PREFIX/lib/libgcc_eh.a"

if [ ! -f "$PREFIX/lib/libgmp.a" ]; then
	bootstrap gmp
	# GMP 的 git 树不含生成好的文档, VPATH 构建下文档规则找不到 version.texi 而失败
	# (发布 tarball 自带 doc/gmp.info, 故不触发该规则)。MAKEINFO 取 true 使文档规则空跑,
	# 载荷只用到 gmp 的头文件与静态库。
	MAKE_VARS=MAKEINFO=true
	configure_and_make gmp --disable-shared --enable-static
	MAKE_VARS=
fi

if [ ! -f "$PREFIX/lib/libmpfr.a" ]; then
	bootstrap mpfr
	configure_and_make mpfr --disable-shared --enable-static --with-gmp="$PREFIX"
fi

if [ ! -f "$PREFIX/lib/libflint.a" ]; then
	bootstrap flint
	configure_and_make flint --disable-shared --enable-static \
		--with-gmp="$PREFIX" --with-mpfr="$PREFIX" LD="$LD_TARGET"
	repack_flint_archive
fi

echo "依赖安装完成: $PREFIX"
ls -l "$PREFIX/lib" | grep -E "libgmp\.a|libmpfr\.a|libflint\.a"
