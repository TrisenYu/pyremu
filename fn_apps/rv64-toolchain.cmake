# ================================================================
#  Cmake for riscv64 musl
#
#  用途: 交叉编译 fn_apps/dune 的 C++ 载荷及其 C 依赖
#
#  用法:
#    cmake -B <build-dir> \
#      -DCMAKE_TOOLCHAIN_FILE=fn_apps/riscv64-toolchain.cmake \
#      <source-dir>
#
#  注意: sysroot 已含 C 运行库libc.a/libm.a/libgcc.a 与 C++ 运行库
#       libc++/libc++abi/libunwind
# 		无需单独 -lc++abi
# ================================================================

set(FN_APPS_DIR  "${CMAKE_CURRENT_LIST_DIR}")
set(REPO         "${FN_APPS_DIR}/..")
set(MUSL_SRC     "${REPO}/vendor/musl")
set(MUSL_SYSROOT "${REPO}/bsp/musl-gc-sysroot")
set(MUSL_LIB     "${MUSL_SYSROOT}/lib")

set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER          clang)
set(CMAKE_CXX_COMPILER        clang++)
set(CMAKE_C_COMPILER_TARGET   riscv64-linux-musl)
set(CMAKE_CXX_COMPILER_TARGET riscv64-linux-musl)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
# ABI setting
set(ARCH_FLAGS "-march=rv64gc -mabi=lp64d")

set(MUSL_ISYSTEM_FLAGS
    "-isystem ${MUSL_SRC}/include -isystem ${MUSL_SRC}/arch/riscv64 -isystem ${MUSL_SRC}/arch/generic -isystem ${MUSL_SRC}/obj/include")

set(CMAKE_C_FLAGS_INIT   "${ARCH_FLAGS} -nostdinc ${MUSL_ISYSTEM_FLAGS}")
set(LIBCXX_ISYSTEM_FLAGS "-isystem ${MUSL_SYSROOT}/include/c++/v1")
set(CMAKE_CXX_FLAGS_INIT
    "${ARCH_FLAGS} -nostdinc++ ${LIBCXX_ISYSTEM_FLAGS} ${MUSL_ISYSTEM_FLAGS}")

set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -fuse-ld=lld -B${MUSL_LIB} -L${MUSL_LIB} -nodefaultlibs")

set(CMAKE_C_STANDARD_LIBRARIES   "-lc -lgcc")
set(CMAKE_CXX_STANDARD_LIBRARIES "-lc++ -lunwind -lc -lgcc")

set(CMAKE_FIND_ROOT_PATH "${MUSL_SYSROOT};${MUSL_SRC}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_AR     llvm-ar)
set(CMAKE_RANLIB llvm-ranlib)
