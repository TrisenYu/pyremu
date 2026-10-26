# pyremu 启动流程

启动链沿用 SiFive HiFive Unleashed 的分工: ZSBL -> OpenSBI -> S 模式下一阶段. 下列各点按
执行次序给出启动期间的必要过程. 

地址以默认配置为准: `ram_base = 0x8000_0000`,
`ram_size = 128 MiB`, 固件映像保留区为 `[0x8000_0000, 0x8040_0000)`,
引导只读存储器为 `[0x4000_0000, 0x4002_0000)`.

## 一、模拟器初始化与上电

1. **创建平台.** `Emulator` 按 `PlatformConfig` 创建总线、L2 缓存、CLINT、PLIC 或
   IMSIC+APLIC、UART 等外设, 并为每个 hart 创建寄存器组、TLB 与 PMP. 每个 hart 的 PC
   取 `prog_cnt`, 特权级取 M 模式.

2. **加载固件.** `parse_firmware()` 解析 ELF, `Emulator.load_firmware()` 把各段写入 RAM.
   固件链接在 `0x00000000` 基址而 RAM 起始于 `ram_base` 时, 各段写入 `vaddr + ram_base`, 同时经
   `Bus.apply_pie_ofs()` 登记链接地址区间, 使固件按链接地址给出的取值仍指向同一批字节;
   PC 置为映像入口地址.

3. **注入 ZSBL.** 调试器把 `--preload` 指定的裸二进制注入引导只读存储器, 起始地址取
   `--preload-addr` 即 `rom_base = 0x4000_0000`. 该文件由
   [bsp/kei-boot/kei.sav.S](../bsp/kei-boot/kei.sav.S) 构建, 链接基址为 0, 全部地址取
   PC 相对寻址, 故装入地址即其运行地址. 引导只读存储器的窗口由
   `PlatformConfig.rom_base` 与 `rom_size` 给出, `rom_base` 取 0 时不注册该设备.
   该设备接受读取, 经总线的写入触发访问故障, 装载经 `BootRom.load` 完成. 全部 hart 的
   PC 置为该地址. 随后调试器把每个 hart 的 `misa` 写为 MXL=RV64|S|U, 使固件的特权级
   判定能够通过.

4. **准备设备树.** 未给出 `--no-fdt` 时, 调试器生成 DTB 并写入全部 hart 的 `a1`.
   复位地址落在引导只读存储器的窗口内时, DTB 写入 `rom_base + ROM_DTB_OFFSET` 即
   `0x4100_0000`, 与 ZSBL 经 `lla a1, _start + 0x10000` 算出的地址一致; 其余情形写入
   `ram_base + ram_size - 64 KiB`. 另在 `ram_base + FW_PAYLOAD_FDT_OFFSET` 即
   `0x8220_0000` 写入一份副本, 该地址是固件为下一阶段准备的 FDT 位置. 其后依次预载
   `--kernel` 指定的内核 Image、`--initrd` 指定的 initramfs, 并按 `--disk` 启用
   virtio-blk. 给出 `--no-fdt` 时调试器不生成也不交付 DTB, 其余初始化不变; 此时 ZSBL
   仍执行 `lla a1, _start + 0x10000`, 在默认注入地址下 `a1` 即 `0x4100_0000`, 而该处
   在引导只读存储器内保持为零. 不读取 `a1` 的裸机程序可以照常运行; 本仓库的 OpenSBI
   固件在该地址上读不到合法的 FDT 头, `fw_platform_init` 进入停机循环, 且此时串口
   尚未初始化, 故没有任何输出.

## 二、ZSBL

5. **早期设置.** ZSBL 把 `mtvec` 指向 `AL1S_HANG`, 清零自身 BSS 段.

6. **处理器自检.** 依次执行加、减、乘、除、取余、逻辑、移位、比较、`jal` 与 `jalr`
   各项检查. 任一项结果不符即跳转 `AL1S_CPU_FAIL` 后停机; 通过后进入访存自检.

7. **访存自检.** 读取 `RAM_BASE` 处的 8 字节. 该地址不属于任何存储区时触发访问故障,
   经 `mtvec` 进入停机; 读取成功即通过.

8. **移交 OpenSBI.** 清零 t1 至 t6, 置 `a0 = mhartid`, `a1 = _start + 0x10000`, 即上一步
   写入的 DTB 地址 `0x4100_0000`, 跳转 `RAM_BASE` 处的 OpenSBI.

## 三、OpenSBI 冷启动

9. **选择启动 hart.** `fw_boot_hart()` 返回 -1, 表示不指定启动 hart. 各 hart 经
   `amoswap.w` 竞争 `_boot_lottery`, 未取得者进入 `_wait_for_boot_hart`, 循环读取
   `_boot_status`.

10. **Opensbi重定位.** 启动 hart 遍历 `__rela_dyn_start` 至 `__rela_dyn_end` 的
    `R_RISCV_RELATIVE` 项, 按实际装载地址改写 GOT; 随后清零除 `ra` 与 `a0` 至 `a4`
    之外的寄存器. 本项目的固件以 `FW_SKIP_BSS_ZERO=1` 构建, BSS 清零一步不执行.

11. **平台早期初始化.** 设置临时 `mtvec` 与临时栈, 调用 `fw_platform_init(FDT)`:
    解析根节点与 `/cpus`, 得到 hart 数量与各 hart 编号, 初始化设备树中的 override 驱动,
    并按 `/chosen` 内的配置确定允许冷启动的 hart. 返回值仍为传入的 FDT 地址.

12. **建立 scratch 与重定位 FDT.** `_scratch_init` 为每个 hart 分配 scratch 与栈, 写入
    固件起止地址、`next_arg1`、`next_addr`、`next_mode`、热启动入口与平台描述符.
    随后按 FDT 头部给出的长度把 DTB 复制到 `fw_next_arg1()` 给出的地址
    `0x8220_0000`, 源地址与目标地址相同时跳过. 最后置 `_boot_status` 为
    `BOOT_STATUS_BOOT_HART_DONE`, 进入 `_start_warm`; 其余 hart 读到该状态后同样进入
    `_start_warm`.

## 四、OpenSBI 初始化

13. **每 hart 设置.** `_start_warm` 清空 `mie`, 由 `mhartid` 求出 hart 序号, 据此取
    scratch 与栈, 写入 `mscratch`, 设置正式 `mtvec` 为 `_trap_handler`, 调用 `sbi_init()`.

14. **sbi_init.** 按 `next_mode` 判定下一特权级; 冷启动 hart 经 `coldboot_lottery` 选出,
    执行 `init_coldboot()`, 其余 hart 执行 `init_warmboot()` 并等待冷启动完成.

15. **init_coldboot.** 依次完成: scratch 与堆初始化、域初始化、HSM 初始化、hart 初始化,
    其中包含 PMP 条目数探测; 定时器初始化, 由设备树的 CLINT 节点给出; 平台早期初始化,
    由 `/chosen/stdout-path` 初始化串口; PMU 与 DBTR 初始化; 打印启动横幅; 中断控制器
    初始化, 按构建取 PLIC 或 IMSIC+APLIC; IPI、TLB 与 FWFT 初始化; 域终结; 平台最终
    初始化; SSE 与 ecall 初始化; 打印域与 hart 信息; `sbi_domain_startup` 启动各域;
    `sbi_hart_protection_configure` 配置 PMP 隔离; `init_sm()` 注册飞地 ecall 扩展,
    建立飞地内存位图与宿主请求队列, 并激活宿主域的 lPMP.

16. **次级 hart 初始化.** 其余 hart 经 `init_warm_startup()` 完成每 hart 初始化, 并经
    `init_sm_warm()` 激活自身的 lPMP, 使 S 模式与 U 模式访存不被全局拒绝.

## 五、移交 S 模式

17. **切换特权级.** 各 hart 在初始化末尾调用 `sbi_hsm_hart_start_finish()`, 取 scratch 中
    的 `next_addr`、`next_mode` 与 `next_arg1`, 经 `sbi_hart_switch_mode()` 设置
    `mepc = next_addr` 与 `mstatus.MPP = S`, 执行 `mret`.

18. **下一阶段入口.** 入口地址由固件形态决定: `fw_payload` 为内嵌载荷, 位于
    `_fw_start + FW_PAYLOAD_OFFSET` 即 `0x8040_0000`, 缺省载荷为 OpenSBI test payload,
    可用 `FW_PAYLOAD_PATH` 换成内核或其他镜像; `fw_jump` 为 `FW_JUMP_ADDR`
    即 `0x8040_0000`, 该处由调试器经 `--kernel` 预载内核 Image. 移交时
    `a0 = mhartid`, `a1 = FDT 地址`, 默认为 `0x8220_0000` 处的副本.

19. **S 模式启动.** 下一阶段建立页表, 写 `satp`, 经 `sfence.vma` 后打开地址翻译, 再进入
    内核或载荷自身的启动流程. 
	内嵌的 `.sittim` 段是 S 模式飞地管理器, 启动期间只作为
    数据随固件装入, 
	由 M 模式在 CREATE 调用中复制进飞地内存池后才开始执行.

## 设备树约定

固件与下一阶段依赖以下节点:

- `/chosen`: `stdout-path` 指向串口, 可含 `bootargs` 与 initramfs 范围; 固件据此初始化
  控制台.
- `/cpus`: `timebase-frequency`, 以及每个 `cpu@N` 的 `reg`、`compatible`、`mmu-type`
  与 `interrupt-controller` 子节点; 固件据此得到 hart 数量与编号.
- `/memory`: 物理内存范围.
- `/soc`: CLINT 节点, 经 `interrupts-extended` 引用各 hart 的 `cpu-intc`; 串口节点;
  中断控制器与外设节点.
- `/reserved-memory`: 固件映像保留区与飞地内存池, 均为 `no-map` 区间, 使内核的线性
  映射与分配器避开这两段.

## 启动命令

```bash
make kei-sav
python -m pyremu.debugger \
    --preload=build/firm-bin/kei.sav \
    --preload-addr=0x40000000 \
    --fdt -1 \
    --ram-base=0x80000000 \
    build/elf/custom_opensbi_fw_payload.elf
```

`make emu` 使用内存基址与零阶段引导程序的同一组参数, 并另加 `--harts` 与
`--hart-logs`; 它不传递 `--fdt`, 需要交付设备树时在 `pyargs` 之外另行给出.
零阶段引导程序的路径与装入地址集中在 `mk/pyremu-args.mk` 的 `zsbl_args`, 各启动目标
共用. `make emu-linux-sh` 改用 `fw_jump` 固件, 并追加 `--kernel`、`--sym-dir=build/sym`、
`--fdt -1`、`--disk` 与内核命令行参数. 内核符号不再随命令行自动加载; 启动后执行
`sym ls` 与 `sym load vmlinux pa:<内核加载地址>` 登记.
