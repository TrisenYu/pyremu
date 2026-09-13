from datetime import datetime
import io
import select
import sys
import time


def yield_cpu(interval: float = 0.001) -> None:
    """通过 ``select`` 在 stdin 上等待 *interval* 秒以让出 CPU."""
    try:
        select.select([sys.stdin], [], [], interval)
    except (io.UnsupportedOperation, TypeError, OSError):
        time.sleep(interval)


def fmt_now_ms() -> str:
    """取操作系统当前本地时间, 精确到毫秒 (YYYY/MM/DD-HH:MM:SS.mmm)."""
    now = datetime.now()
    return f"{now:%Y/%m/%d-%H:%M:%S}.{now.microsecond // 1000:03d}"
