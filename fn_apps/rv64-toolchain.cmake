# ================================================================
#  riscv64 musl 静态交叉编译工具链 (CMake)
#
#  用途: 交叉编译 fn_apps/dune 的 C++ (DUNE FEM) 载荷及其 C 依赖,
#        与 fn_apps/config.mk 共用同一套工具链。
#
#  用法:
#    cmake -B <build-dir> \
#      -DCMAKE_TOOLCHAIN_FILE=fn_apps/riscv64-toolchain.cmake \
#      <source-dir>
#
#  工具链构成 (与 fn_apps/config.mk 一致):
#    - 编译器: 标准 clang / clang++, 目标三元组 riscv64-linux-musl
#    - ABI:    -march=rv64gc -mabi=lp64d (双精度硬浮点), 与模拟器 ISA
#      (rv64imacfd) 及 Rust 预编译 std (lp64d) 一致
#    - 链接:   lld 全静态, crt*.o / libc.a / libm.a / libgcc.a 取自
#      bsp/musl-gc-sysroot/lib
#    - 头文件: C 头仍直接引用 vendor/musl 源码树 (sysroot 的 C include 目录
#      已被 build-musl-sysroot.sh 剔除); C++ 头 (libc++) 位于 sysroot 的
#      include/c++/v1, 由 build-libcxx.sh 安装
#
#  注意: sysroot 已含 C 运行库 (musl 的 libc.a/libm.a/libgcc.a) 与 C++ 运行库
#        (libc++/libc++abi/libunwind, 见 bsp/build-libcxx.sh)。C++ 载荷走
#        libc++ 静态链接; libc++abi 已合并进 libc++.a, 无需单独 -lc++abi。
# ================================================================

# 本文件位于 fn_apps/, 据此推导仓库根与 sysroot 路径 (可重定位, 不硬编码绝对路径)
set(FN_APPS_DIR  "${CMAKE_CURRENT_LIST_DIR}")
set(REPO         "${FN_APPS_DIR}/..")
set(MUSL_SRC     "${REPO}/vendor/musl")
set(MUSL_SYSROOT "${REPO}/bsp/musl-gc-sysroot")
set(MUSL_LIB     "${MUSL_SYSROOT}/lib")

set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# 编译器与目标三元组 (CMake 以 --target=<三元组> 透传给 clang / clang++)
set(CMAKE_C_COMPILER          clang)
set(CMAKE_C_COMPILER_TARGET   riscv64-linux-musl)
set(CMAKE_CXX_COMPILER        clang++)
set(CMAKE_CXX_COMPILER_TARGET riscv64-linux-musl)

# try_compile 阶段仅编译为静态库做探测: 宿主无法运行 riscv64 可执行文件
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# 架构与 ABI 标志
set(ARCH_FLAGS "-march=rv64gc -mabi=lp64d")

# musl 头文件搜索路径 (sysroot 无 include, 直接引用源码树)。
# 以空格连接的单一字符串, 而非 CMake 列表: 避免被展开成带分号的列表
# 污染 CMAKE_C_FLAGS_INIT, 导致编译命令把 -isystem 与其后路径拆成两条。
set(MUSL_ISYSTEM_FLAGS
    "-isystem ${MUSL_SRC}/include -isystem ${MUSL_SRC}/arch/riscv64 -isystem ${MUSL_SRC}/arch/generic -isystem ${MUSL_SRC}/obj/include")

# C: 与 config.mk 完全一致, 追加 -nostdinc 屏蔽宿主 glibc 头
set(CMAKE_C_FLAGS_INIT   "${ARCH_FLAGS} -nostdinc ${MUSL_ISYSTEM_FLAGS}")
# C++: 用 -nostdinc++ 屏蔽宿主 C++ 标准库头, 保留 clang 内建头 (stddef 等),
#      再以 -isystem 引入 libc++ 头 (bsp/musl-gc-sysroot/include/c++/v1,
#      含 __config_site 与 __cxxabi_config.h, 二者均在 v1/ 下无 triple 子目录),
#      及 musl 的 C 头。
set(LIBCXX_ISYSTEM_FLAGS "-isystem ${MUSL_SYSROOT}/include/c++/v1")
set(CMAKE_CXX_FLAGS_INIT
    "${ARCH_FLAGS} -nostdinc++ ${LIBCXX_ISYSTEM_FLAGS} ${MUSL_ISYSTEM_FLAGS}")

# 静态链接: -B 定位 crt*.o, -L 定位 libc.a/libm.a/libgcc.a。
# -nodefaultlibs 屏蔽 clang 对 musl 目标隐式追加的 -lgcc_eh (sysroot 未提供该库),
# 转而显式列出运行库, 避免复制 libgcc.a 充当 libgcc_eh.a 的 hack。
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -fuse-ld=lld -B${MUSL_LIB} -L${MUSL_LIB} -nodefaultlibs")

# 显式链接的运行库 (CMake 将其置于目标文件之后):
#   C:   musl libc + libgcc 基础算术
#   C++: libc++ 静态库 (已合并 libc++abi 对象) + libunwind + musl libc + libgcc。
#        顺序须为 libc++ -> libunwind -> libc -> libgcc: libc++ 引用
#        _Unwind_* (libunwind) 与 libc 符号, libgcc 提供内建函数与基础算术。
set(CMAKE_C_STANDARD_LIBRARIES   "-lc -lgcc")
set(CMAKE_CXX_STANDARD_LIBRARIES "-lc++ -lunwind -lc -lgcc")

# find_* 仅在交叉 sysroot 与 musl 源码树内检索, 程序类工具仍在宿主查找
set(CMAKE_FIND_ROOT_PATH "${MUSL_SYSROOT};${MUSL_SRC}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# 归档工具 (与 build-musl-sysroot.sh 一致)
set(CMAKE_AR     llvm-ar)
set(CMAKE_RANLIB llvm-ranlib)
