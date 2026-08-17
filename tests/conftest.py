import sys

import pytest


@pytest.fixture(autouse=True)
def _mock_stdin_fileno(monkeypatch):
    """
    因为 pytest 的 DontReadFromInput
    的 fileno() 抛 UnsupportedOperation, 而 Debugger.__init__ 需要它来捕获 stdin fd.
    """
    monkeypatch.setattr(sys.stdin, "fileno", lambda: 0)
