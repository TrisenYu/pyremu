# mods.mk — 导入 mk/ 下除本文件外的全部分片

mods_dir := $(dir $(lastword $(MAKEFILE_LIST)))
include $(filter-out $(mods_dir)mods.mk,$(sort $(wildcard $(mods_dir)*.mk)))

export
