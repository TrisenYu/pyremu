import os
import sys

import pytest

# Tests assume step() = 1 instruction per hart. Disable the native batch
# engine by default; batch-specific tests enable it via PYREMU_NATIVE_BATCH=1
# around Emulator construction.
os.environ.setdefault("PYREMU_NATIVE_BATCH", "0")


@pytest.fixture(autouse=True)
def _mock_stdin_fileno(monkeypatch):
    """给 sys.stdin 打上 fileno() 桩, 因为 pytest 的 DontReadFromInput
    的 fileno() 抛 UnsupportedOperation, 而 Debugger.__init__ 需要它来捕获 stdin fd.
    """
    monkeypatch.setattr(sys.stdin, "fileno", lambda: 0)


@pytest.fixture(autouse=True)
def _reset_native_batch_env():
    """每个用例结束后把 PYREMU_NATIVE_BATCH 复位为默认 "0", 隔离用例间 env 泄漏。

    多个 batch 相关用例会临时改写该 env; 历史上有的用例用 pop/del 未正确还原,
    导致后续依赖纯 Python 默认的用例在 *全量* 运行时行为漂移 (单独跑却通过)。
    autouse 复位从根本上消除这类顺序相关的隐性泄漏。需要 native 的用例应在自身
    构造 Emulator 前显式设 "1"。
    """
    yield
    os.environ["PYREMU_NATIVE_BATCH"] = "0"
