import os

# Tests assume step() = 1 instruction per hart. Disable the native batch
# engine by default; batch-specific tests enable it via emu._native_batch = True.
os.environ.setdefault("PYREMU_NATIVE_BATCH", "0")
