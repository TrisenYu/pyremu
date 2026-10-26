# zsbl.mk — 零阶段引导程序 kei.sav

# 链接于 0x0 的零阶段引导程序, 调试器经 --preload 装入引导只读存储器的基址,
# 见 mk/pyremu-args.mk 的 zsbl_args 与 mk/boot-rom.mk.
kei-sav = $(firm_dir)/kei.sav
