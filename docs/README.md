# Pyremu — RISC-V 64 位模拟器

|缩略 |含义        |
|:--:|:---------:|
|Py  | python    |
|r   | riscv/rust|
|emu | emulator  |

Pyremu 是一个用 CPython3.15.0 编写的 **RISC-V 指令集模拟器与交互式调试器**，面向固件调试、TEE 开发、操作系统调试等场景。

可模拟多 hart 执行，支持 RV64 **IMAFDC** 指令扩展与 Zicsr/Zifencei/Zbb/Sstc，U/S/M 三级特权级，Sv39 虚拟内存，L2 缓存 MESI 一致性协议，以及 CLINT、PLIC、AIA基础的 (IMSIC 与 APLIC) 等中断控制器。

## 快速上手

```bash
uv sync && source .venv/bin/activate

# 加载零阶段加载器作为上电初始化代码，
# 设置内存基址为0x8000_0000
# 同时以交互式调试器加载固件
python -m pyremu.debugger \
--ram-base=0x80000000     \
--preload build/firm-bin/kei.sav \
build/elf/custom_opensbi_fw_payload.elf # 这个需要自己建opensbi或者别的兼容sbi调用规范的elf文件。

# 纯 Python API 示例
python examples/demo_emulator.py

# 项目根目录下
make test                         # 仅运行对pyremu的测试项
make cov-test                     # 带覆盖率的测试

uv run pytest tests/              # 全部测试
uv run pytest tests/test_trap/    # 单个套件
uv run ruff check .               # 代码风格
uv run ruff format .              # 自动格式化
uv run python tools/dump_dtb.py   # 导出当前平台的 DTB
uv run python tools/jmp_instr_counter.py  # 统计基本块终结指令的静态条数

```

## 提交前检查 (pre-commit)

仓库自带的 [tools/git-hooks/pre-commit](tools/git-hooks/pre-commit) 在每次 `git commit` 前
运行检查与格式化: 逐个文件做格式检查, 需要格式化的文件由钩子格式化并加入索引, 格式化
失败或静态检查不通过时中止提交。Git 默认只从 `.git/hooks/` 读取钩子, 需要在仓库根目录
执行一次以下命令, 把钩子目录指向该脚本:

```bash
git config core.hooksPath tools/git-hooks
```

该设置写入当前克隆的 `.git/config`, 不影响其他克隆; 重新克隆或换机器后需要再次执行。
取消绑定用 `git config --unset core.hooksPath`。


## 已实现的 RISC-V 特性

| 模块 | 内容 |
|------|------|
| **指令集** | RV64 IMAFDC：基础整数、乘除（M）、原子操作与 LR/SC（A）、单精度与双精度浮点（F/D）、压缩指令（C） |
| **扩展** | Zicsr：CSR 读写；Zifencei：FENCE.I；Zbb：位操作；Sstc：`stimecmp` 与 `stimecmph` |
| **特权级** | U/S/M 三级；ECALL、EBREAK、MRET、SRET；陷态经 `medeleg` 与 `mideleg` 委派到 S 模式；WFI 使 hart 停止取指与执行，中断到达后唤醒，`mstatus.TW` 置位时非 M 模式执行 WFI 引起非法指令陷态 |
| **虚拟内存** | Sv39 三级页表遍历，4 KiB 页与 2 MiB 超级页；每 hart 独立的指令 TLB 与数据 TLB，全相联，FIFO 或 LRU 替换；SFENCE.VMA 刷新全部 hart 的 TLB |
| **内存保护** | PMP 默认由 `PlatformConfig.pmp_entries` 配置；NAPOT、NA4、TOR 匹配；`mstatus.MPRV` 置位且 `MPP` 为 S 或 U 时按该特权级检查；PMA 按 RAM 窗口与设备 MMIO 路由，未落入任何窗口的访问报告访问故障 |
| **缓存** | 共享 L2 缓存，MESI 一致性协议 |
| **中断** | CLINT：`mtime`/`mtimecmp` 定时器与 MSIP 核间中断；PLIC：电平语义，每 hart 一对 M/S context；AIA：每 hart 一个 IMSIC M 文件与一个 S 文件，APLIC 的 S 域与 M 域把有线中断转为 MSI；投递优先级 MEI > MSI > MTI > SEI > SSI > STI |
| **外设** | UART、SPI、I2C、GPIO、看门狗、随机数发生器；virtio-mmio 块设备与网卡，网卡分宿主侧与飞地侧两个实例；终端输入输出桥接 |
| **平台** | FDT 设备树生成；PlatformConfig 预设 qemu_virt 与 sifive_u54，支持从 JSON、TOML、YAML 反序列化 |

未实现：
- RVV 1.0 向量扩展（opcode 0x57）
- 以及 H 扩展的两阶段地址翻译与 IMSIC guest 中断文件。


## 项目结构

```
pyremu/                       模拟器包
  emulator.py                 Emulator 的构造流程与公开接口
  debugger.py                 rvdb 交互式调试器入口
  platform.py                 PlatformConfig 平台配置, 支持 JSON、TOML 与 YAML 反序列化
  configs_aux.py              配置读取, 环境变量优先, 回退到 configs_gen
  configs_gen.py              由 mk/ 下配置分片生成的常量, 不手工编辑
  emu/                        执行循环、状态同步与装配的拆分实现
    setup.py                  总线、中断控制器、外设与 hart 的装配
    load.py                   固件、设备树与初始内存盘的装载
    run.py                    执行循环、时钟推进与 WFI 休眠
    native.py                 加速执行引擎的状态同步与退出处理
  debug/                      调试器实现, 由多个操作类组合而成
    base.py                   共享状态、hart 访问与输出
    cli.py                    命令行入口与参数解析
    dispatch.py               REPL 命令分发与帮助
    exec.py                   指令执行、快照回滚与写监控
    breakpoint.py             断点的设置、命中检查与条件求值
    reg.py                    通用寄存器与 CSR 的读写命令
    mem.py                    虚拟与物理内存读写、地址解析、反汇编
    stk_frame.py              栈回溯与跨特权级边界帧
    status.py                 hart 状态与 CSR 位域显示
    tlb_cache.py              TLB 与 L2 缓存的视图
    mmu_view.py               PMP、SATP 与页表遍历的视图
    bins.py                   命令行传入的各映像所占的物理地址范围
    disk.py                   磁盘映像的格式摘要与块布局
    sym/                      多来源符号的解析、加载与登记
  core/                       处理器核心
    hart.py                   寄存器文件、特权级与 mstatus 位常量
    decoder.py                指令解码与全部指令处理函数
    trap_handler.py           陷态投递、中断检查与陷态返回
    mem_check_aux.py          访存、CSR 校验与地址翻译
    registers.py              寄存器与 CSR 模型
    trap_def.py               陷态类型与 cause 编码的互查
  memory/                     内存子系统
    mmu.py                    Sv39 页表遍历
    tlb.py                    指令 TLB 与数据 TLB
    cache_base.py             缓存抽象基类
    l2cache.py                共享 L2 缓存与 MESI 协议
    bus.py                    总线、设备注册与 PMA 检查
    pmp.py                    物理内存保护
  interrupt/                  中断子系统
    clint.py                  定时器与核间中断
    plic.py                   外部中断控制器
    imsic.py                  MSI 中断文件
    aplic.py                  有线中断转 MSI
    irq_sink.py               设备中断的投递目标
  peripheral/                 内存映射外设
    uart.py                   串口
    spi.py                    SPI 主控
    i2c.py                    I2C 主控
    gpio.py                   通用输入输出
    watchdog.py               看门狗
    crng.py                   随机数发生器
    virtio_mmio.py            virtio-mmio 传输层
    virtio_blk.py             virtio 块设备
    virtio_net.py             virtio 网卡
    termio.py                 终端输入输出桥接
  disk/                       磁盘映像格式的只读解析, 每种文件系统格式一个模块
  utils/                      工具
    disassem.py               RV64 反汇编器
    parse_bin.py              ELF、PE 与纯二进制的固件解析
    dtb.py                    设备树生成
    str_aux.py                地址与十六进制转储的格式化
    mask.py                   位宽掩码
    wrapper.py                异常处理装饰器
    regname.py                寄存器名称查找
    tick.py                   时钟辅助
    file_ops.py               文件路径操作
  env_inject/                 运行时注入
    preload.py                把预置代码片段写入 RAM
  _native/                    Rust 加速执行引擎与终端守护线程, 产出 libdecode.so 与 libtermio.so

bsp/                          板级支持包
  sittim/                     可嵌入 OpenSBI 的 S 模式运行时
  kei-boot/                   模拟器的零阶段加载器
  ref-impl/                   飞地二进制接口、扩展模块与构建补丁的参考实现
  tee_aux_tools/              辅助驱动、签名工具与评测套件
    linux-driver/             Linux 飞地交互内核驱动, 提供 /dev/tee_enclave
    ext-mod-loader/           扩展模块加载器
    ext-mod-svc/              扩展模块服务
    sign-mod/                 为扩展模块文件追加 ECDSA 签名
    cache-probe-exploit/      TEE 评测套件, 部署到根文件系统的 /eval/cache-probe-exploit
    stress-test/              TEE 压力测试套件, 部署到根文件系统的 /eval/stress-test
  linux/                      内核源码树, 需要自行复制到本仓库
  musl-gc-sysroot/            musl 与 libc++ 的交叉编译 sysroot
  setup-rootfs/               用 debootstrap 构建根文件系统, 并打包测试程序与驱动

fn_apps/                      可信应用载荷
  orbit/                      轨道动力学的多体积分载荷
  chem/                       量子化学的 Hartree-Fock 载荷
  graphene/                   石墨烯紧束缚能带的计算载荷
  ising/                      二维 Ising 模型的蒙特卡洛载荷
  fem/                        二维 Poisson 方程的有限元载荷
  dsp/                        FFT 谱分析、FIR 滤波与 OFDM 基带载荷
  linalg/                     稠密 Cholesky 分解载荷
  dune/                       DUNE 有限元库的计算载荷
  Langlands/                  模形式与 L 函数的计算载荷, 依赖 FLINT、MPFR 与 GMP
  knots/                      Jones 多项式的计算载荷
  cfrac/                      连分数分解载荷, 分配大量短生命周期的小对象
  chibicc/                    C 编译器载荷
  json_calc/                  以多份 JSON 文档为输入的解析载荷, C 与 C++ 两种实现
  nc_udp/                     以 busybox 的 nc 实现 UDP 自收自发的通路载荷
  malloc-churn/               多线程 malloc 压力测试载荷
  stress-ng/                  上游压力测试工具
  bin/                        构建产物
  build/                      构建产物

tests/                        pytest 套件与部分测试项的源码
  test_*.py                   顶层单元与集成测试
  test_*/                     按子系统拆分的测试目录
  src-env/                    汇编测试源码, 经 llvm-mc 与 ld.lld 构建
  src-alg/                    C++ 算法源码
  src-rv8/                    部署到根文件系统 /eval/toy-progs 的测试程序
  instr_encode.py             RV64 指令编码辅助
  loader.py                   把多个 U 模式程序注入内核进程表

mk/                           按子系统拆分的 makefile, 生成 configs_gen.py等文件
examples/                     编程接口使用示例
docs/                         设计文档与 CHANGELOG
tools/                        辅助性脚本工具
vendor/                       第三方子模块, 如 musl
build/                        构建产物
```


## 交互式调试器 (rvdb)

```
rvdbg[0] bt              # GDB 风格栈回溯
rvdbg[0] disasm 0x80000000 +32  # 反汇编
rvdbg[0] step 10         # 单步 10 条指令
rvdbg[0] regs            # 全部寄存器
rvdbg[0] mem 0x80000000 64  # 内存 dump
rvdbg[0] csr satp        # CSR 读写
```

依赖 `prompt_toolkit` 提供 Tab 补全与历史记录，历史记录写入 `~/.pyremu_history`；`rich` 提供彩色格式化的命令行输出。历史文件仅保留最近 5000 行。


## Rust Native 并发执行引擎

`pyremu/_native/` 包含一个 Rust 编写的 **thread-per-hart 并发执行引擎** (`libdecode.so` +
`libtermio.so`)，将 fetch-decode-execute 循环从 Python 移入 Rust，以 OS 线程并发
执行直至 MMIO 设备访问或外部中断退出。

CLINT 定时器/MSIP、UART TX、virtio-blk MMIO、压缩指令、AMO 原子操作等热路径全部在
Rust 内联处理，无需退出到 Python。仅 PLIC claim/complete 等需要 Python 侧设备状态的
操作才触发 FFI 边界。

### 工作原理

Python 侧通过 `marshal_hart()` 将全部 hart 的寄存器文件序列化为 `#[repr(C)]` FFI 结构体,
与 RAM buffer / PMP / CLINT / UART 等执行上下文一并传入 Rust 引擎 (`run_parallel`) 以 thread-per-hart 模型并发执行 fetch-decode-execute 循环。
期间通过共享原子变量同步CLINT 中断与TLB为代表的微架构结构。

### 构建

```bash
make build-native # Release 构建 + 复制到 pyremu/_native/libdecode.so
```
也可以手动构建:

```bash
cargo build --release --manifest-path pyremu/_native/Cargo.toml
cp pyremu/_native/target/release/libdecode.so pyremu/_native/libdecode.so

cargo test --manifest-path pyremu/_native/Cargo.toml   # Rust 侧单元测试
```

构建依赖: Rust 工具链 (edition 2021), 无需 `maturin`/`setuptools-rust`——输出为标准
cdylib，通过 CPython 内置 `ctypes` 加载。

当 `.so` 缺失或不兼容时，模拟器将回退到纯 Python 执行路径。但执行用时较长，且目前缺乏维护。


### 状态结构体布局

`HartState` 和 `BatchResult` 通过 `#[repr(C)]` 锁定内存布局，Rust 和 Python ctypes
两侧必须逐字节一致。修改字段时需同步更新:

| 文件 | 作用 |
|------|------|
| [state.rs](pyremu/_native/cpu/src/state.rs) | Rust `#[repr(C)]` 结构体定义 |
| [hart.py](pyremu/core/hart.py) | Python ctypes `_fields_` + `marshal_hart`/`unmarshal_hart` |

验证命令:
```bash
cargo test --manifest-path pyremu/_native/Cargo.toml   # sizeof/align
uv run pytest tests/test_emulator.py::TestNativeBatchLayout   # Python 侧
```

## 运行模拟器期间可用的快捷键

| 按键 | 功能 |
|------|------|
| **Ctrl+Q** | 运行中暂停，回到调试器 |

#### 代码质量规范
本仓库对所编写代码存在要求：
1. 单个函数内，**缩进在原则上不应超过 $4$ 层**，且行数不应超过 $100$ 行，每列字符不应超过 $100$ 个。
   1. 如果不可避免超过 $4$ 层而小于等于 $5$ 层，则该层内连同注释的行数，不应连续超过 $10$ 行。
   2. 拒绝评阅与通过达到或超过 $6$ 层嵌套的实现逻辑。
2. 对于函数签名/函数定义，传入与传出参数的总和不应超过 $6$ 个。且当定义过长时，应使用下列参考格式而完成你的实现。
3. 在满足前三条约束的情形下，请在发起分支拉取请求前，用本地的lint或代码格式检查器自动格式化你的实现代码。
5. 除自动化生成的代码实现以及单元测试代码，每个文件的代码实现建议不超过 $1500$ 行。
6. 如有必要，可补充文档说明。
