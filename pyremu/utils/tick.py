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
