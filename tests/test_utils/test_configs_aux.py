#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""配置读取辅助测试.

覆盖: 环境变量与 configs_gen 默认值两者的优先次序 / 各类型的解析 / 环境变量是否被
显式设置的判定 / force_config 上下文对 configs_gen 的临时覆盖与恢复.
"""

from pyremu import configs_gen
from pyremu.configs_aux import cfg_bool, cfg_float, cfg_int, cfg_is_set, cfg_str, force_config

ENV_NAME = "PYREMU_TEST_CFG_AUX"
UNSET_NAME = "PYREMU_TEST_CFG_AUX_UNSET"


class TestCfgStr:
    """字符串配置."""

    def test_env_value_wins(self, monkeypatch):
        """环境变量非空时返回该取值."""
        monkeypatch.setenv(ENV_NAME, "from_env")
        assert cfg_str(ENV_NAME) == "from_env"

    def test_falls_back_to_configs_gen(self, monkeypatch):
        """环境变量未设置时回退到 configs_gen 中的同名取值."""
        monkeypatch.delenv("TLB_ENTRIES", raising=False)
        assert cfg_str("TLB_ENTRIES") == str(configs_gen.TLB_ENTRIES)

    def test_unknown_name_falls_back_to_empty(self, monkeypatch):
        """configs_gen 中没有该名称时回退为空串."""
        monkeypatch.delenv(UNSET_NAME, raising=False)
        assert cfg_str(UNSET_NAME) == ""


class TestCfgBool:
    """布尔配置."""

    def test_env_value_other_than_zero_is_true(self, monkeypatch):
        """环境变量取值非 "0" 时得到 True."""
        monkeypatch.setenv(ENV_NAME, "1")
        assert cfg_bool(ENV_NAME) is True
        monkeypatch.setenv(ENV_NAME, "yes")
        assert cfg_bool(ENV_NAME) is True

    def test_env_zero_is_false(self, monkeypatch):
        """环境变量取值为 "0" 时得到 False."""
        monkeypatch.setenv(ENV_NAME, "0")
        assert cfg_bool(ENV_NAME) is False

    def test_falls_back_to_configs_gen(self, monkeypatch):
        """环境变量未设置时按 configs_gen 的默认值判定."""
        monkeypatch.delenv("PYREMU_AIA", raising=False)
        assert cfg_bool("PYREMU_AIA") is (configs_gen.PYREMU_AIA != 0)


class TestCfgInt:
    """整数配置."""

    def test_decimal_env_value(self, monkeypatch):
        """环境变量的十进制取值被解析."""
        monkeypatch.setenv(ENV_NAME, "1234")
        assert cfg_int(ENV_NAME) == 1234

    def test_hex_env_value(self, monkeypatch):
        """环境变量的 0x 前缀取值按十六进制解析."""
        monkeypatch.setenv(ENV_NAME, "0x200000")
        assert cfg_int(ENV_NAME) == 0x200000

    def test_falls_back_to_configs_gen(self, monkeypatch):
        """环境变量未设置时回退到 configs_gen 中的同名取值."""
        monkeypatch.delenv("FAST_LOAD_THRESHOLD", raising=False)
        assert cfg_int("FAST_LOAD_THRESHOLD") == configs_gen.FAST_LOAD_THRESHOLD

    def test_unknown_name_falls_back_to_zero(self, monkeypatch):
        """configs_gen 中没有该名称时回退为 0."""
        monkeypatch.delenv(UNSET_NAME, raising=False)
        assert cfg_int(UNSET_NAME) == 0


class TestCfgFloat:
    """浮点配置."""

    def test_env_value(self, monkeypatch):
        """环境变量的取值按浮点解析."""
        monkeypatch.setenv(ENV_NAME, "0.125")
        assert cfg_float(ENV_NAME) == 0.125

    def test_falls_back_to_configs_gen(self, monkeypatch):
        """环境变量未设置时回退到 configs_gen 中的同名取值."""
        monkeypatch.delenv("WFI_MAX_SLEEP", raising=False)
        assert cfg_float("WFI_MAX_SLEEP") == float(configs_gen.WFI_MAX_SLEEP)

    def test_unknown_name_falls_back_to_zero(self, monkeypatch):
        """configs_gen 中没有该名称时回退为 0.0."""
        monkeypatch.delenv(UNSET_NAME, raising=False)
        assert cfg_float(UNSET_NAME) == 0.0


class TestCfgIsSet:
    """环境变量是否被显式设置."""

    def test_set_after_putenv(self, monkeypatch):
        """设置之后判定为已显式设置."""
        monkeypatch.delenv(UNSET_NAME, raising=False)
        assert not cfg_is_set(UNSET_NAME)
        monkeypatch.setenv(UNSET_NAME, "anything")
        assert cfg_is_set(UNSET_NAME)

    def test_empty_value_still_counts_as_set(self, monkeypatch):
        """取值为空串仍算已显式设置, 与未设置区分开."""
        monkeypatch.setenv(UNSET_NAME, "")
        assert cfg_is_set(UNSET_NAME)


class TestForceConfig:
    """force_config 上下文的临时覆盖."""

    def test_value_is_overridden_inside_and_restored_outside(self):
        """上下文内取到覆盖值, 退出后恢复原值."""
        original = configs_gen.PYREMU_AIA
        with force_config("PYREMU_AIA", 7):
            assert configs_gen.PYREMU_AIA == 7
        assert configs_gen.PYREMU_AIA == original

    def test_value_is_restored_when_body_raises(self):
        """上下文体内抛异常时同样恢复原值."""
        original = configs_gen.PYREMU_AIA
        try:
            with force_config("PYREMU_AIA", 9):
                raise RuntimeError("body failed")
        except RuntimeError:
            pass
        assert configs_gen.PYREMU_AIA == original
