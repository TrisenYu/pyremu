"""验证 amoadd.w.aqrl — OpenSBI atomic_sub_return 的底层指令.

tlb_entry_process() 用 atomic_sub_return() ->amoadd.w.aqrl 递减 tlb_sync.
若 .aq/.rl 位导致 funct5 提取错误, 则 AMO 操作会被错误路由或产生非法指令陷态.
"""

import struct

import pytest

from pyremu._native import decode_fields
from pyremu.core.hart import RiscvMode
from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig


# ============================================================
#  编码: amoadd.w rd, rs2, (rs1)  — funct5=00000, func7 含 aq/rl
# ============================================================

def _encode_amoadd_w(rd=5, rs1=10, rs2=11, aq=0, rl=0):
    return (
        (0b00000 << 27)  # funct5 = AMOADD
        | (aq << 26)
        | (rl << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (0b010 << 12)  # funct3 = .W (32-bit)
        | (rd << 7)
        | 0b0101111       # opcode = AMO
    )


class TestAmoaddWFunct5:
    """amoadd.w 的 funct5 掩码剥离 .aq/.rl 位."""

    @pytest.mark.parametrize("aq,rl", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_funct5_strips_aq_rl(self, aq, rl):
        instr = _encode_amoadd_w(aq=aq, rl=rl)
        f = decode_fields(instr)
        funct5 = f.func7 >> 2  # Rust handler 的做法: func7 >> 2
        assert funct5 == 0b00000, (
            f"aq={aq} rl={rl}: funct5 expected 0 (AMOADD), got {funct5}"
        )

    def test_func7_with_aqrl_is_0x03(self):
        """amoadd.w.aqrl 编码后 func7 = 0b0000011."""
        instr = _encode_amoadd_w(aq=1, rl=1)
        f = decode_fields(instr)
        assert f.func7 == 0b0000011


# ============================================================
#  Python 执行路径: 验证 amoadd.w.aqrl 正确递减 tlb_sync
# ============================================================

@pytest.fixture
def _m_mode_hart():
    cfg = PlatformConfig(num_harts=1, ram_base=0x80000000, ram_size=64 * 1024 * 1024)
    emu = Emulator(cfg, bootargs="")
    hart = emu.harts[0]
    hart.mode = RiscvMode.M
    hart.satp_val = 0
    return emu, hart


class TestAmoaddWExecution:
    """amoadd.w.aqrl 的完整执行语义 (Python 路径)."""

    @staticmethod
    def _exec_amo(emu, hart, instr, pa, rs2_val, initial=1):
        """在给定 PA 执行 AMO 指令, 返回 (old_mem, new_mem, rd_val)."""
        emu.bus.write(pa, struct.pack("<I", initial))
        emu.bus.write(0x80000000, struct.pack("<I", instr))
        hart.gprs[10] = pa       # rs1 = a0 = address
        hart.gprs[11] = rs2_val  # rs2 = a1
        hart.gprs[5] = 0         # rd = t0
        hart.pc = 0x80000000
        old_raw = emu.bus.read(pa, 4)
        old_val = struct.unpack("<I", old_raw)[0]
        hart.exec_instr(instr)
        new_val = struct.unpack("<I", emu.bus.read(pa, 4))[0]
        return old_val, new_val, hart.gprs[5] & 0xFFFFFFFF

    def test_amoadd_w_aqrl_sub_1(self, _m_mode_hart):
        """amoadd.w.aqrl: tlb_sync 从 1→0 (递减 1)."""
        emu, hart = _m_mode_hart
        instr = _encode_amoadd_w(rd=5, rs1=10, rs2=11, aq=1, rl=1)
        old_val, new_val, rd_val = self._exec_amo(
            emu, hart, instr, 0x80001000, 0xFFFFFFFF  # -1 as u32
        )
        assert old_val == 1
        assert new_val == 0, f"new_val={new_val}, expected 0"
        assert rd_val == 1, f"rd should return old value 1, got {rd_val}"

    def test_amoadd_w_no_aqrl_same_behavior(self, _m_mode_hart):
        """不带 .aq/.rl 的 amoadd.w 与 .aqrl 功能等价."""
        emu, hart = _m_mode_hart
        instr = _encode_amoadd_w(rd=5, rs1=10, rs2=11, aq=0, rl=0)
        old_val, new_val, rd_val = self._exec_amo(
            emu, hart, instr, 0x80001000, 0xFFFFFFFF
        )
        assert old_val == 1
        assert new_val == 0
        assert rd_val == 1

    def test_amoadd_w_twice_to_zero(self, _m_mode_hart):
        """连续两次递减: 2→1→0."""
        emu, hart = _m_mode_hart
        instr = _encode_amoadd_w(rd=5, rs1=10, rs2=11, aq=1, rl=1)
        pa = 0x80001000
        emu.bus.write(0x80000000, struct.pack("<I", instr))
        hart.gprs[11] = 0xFFFFFFFF  # -1
        hart.gprs[5] = 0
        hart.pc = 0x80000000
        # 第一次: tlb_sync 从 2 ->1
        emu.bus.write(pa, struct.pack("<I", 2))
        hart.gprs[10] = pa
        hart.exec_instr(instr)
        v1 = struct.unpack("<I", emu.bus.read(pa, 4))[0]
        r1 = hart.gprs[5] & 0xFFFFFFFF
        assert v1 == 1, f"after 1st: expected 1, got {v1}"
        assert r1 == 2, f"rd 1st: expected 2, got {r1}"
        # 第二次: 从 1 ->0 (不重置内存)
        hart.gprs[10] = pa
        hart.exec_instr(instr)
        v2 = struct.unpack("<I", emu.bus.read(pa, 4))[0]
        r2 = hart.gprs[5] & 0xFFFFFFFF
        assert v2 == 0, f"after 2nd: expected 0, got {v2}"
        assert r2 == 1, f"rd 2nd: expected 1, got {r2}"

    def test_amoadd_w_aq_alone(self, _m_mode_hart):
        """仅 .aq (无 .rl): 行为不变."""
        emu, hart = _m_mode_hart
        instr = _encode_amoadd_w(rd=5, rs1=10, rs2=11, aq=1, rl=0)
        _, v, r = self._exec_amo(emu, hart, instr, 0x80001000, 0xFFFFFFFF)
        assert v == 0
        assert r == 1

    def test_amoadd_w_rl_alone(self, _m_mode_hart):
        """仅 .rl (无 .aq): 行为不变."""
        emu, hart = _m_mode_hart
        instr = _encode_amoadd_w(rd=5, rs1=10, rs2=11, aq=0, rl=1)
        _, v, r = self._exec_amo(emu, hart, instr, 0x80001000, 0xFFFFFFFF)
        assert v == 0
        assert r == 1

    def test_amoadd_d_aqrl_64bit(self, _m_mode_hart):
        """amoadd.d.aqrl (64-bit) 递减."""
        emu, hart = _m_mode_hart
        pa = 0x80001000
        emu.bus.write(pa, struct.pack("<Q", 1))

        instr = (
            (0b00000 << 27) | (1 << 26) | (1 << 25)  # AMOADD .aqrl
            | (11 << 20) | (10 << 15)                  # rs2, rs1
            | (0b011 << 12)                             # funct3 = .D
            | (5 << 7) | 0b0101111
        )
        emu.bus.write(0x80000000, struct.pack("<I", instr))
        hart.gprs[10] = pa
        hart.gprs[11] = 0xFFFFFFFFFFFFFFFF  # -1 as u64
        hart.gprs[5] = 0
        hart.pc = 0x80000000

        old = struct.unpack("<Q", emu.bus.read(pa, 8))[0]
        hart.exec_instr(instr)
        new = struct.unpack("<Q", emu.bus.read(pa, 8))[0]
        assert old == 1, f"initial: expected 1, got {old}"
        assert new == 0, f"after amoadd.d.aqrl: expected 0, got {new}"
        assert (hart.gprs[5] & 0xFFFFFFFFFFFFFFFF) == 1, "rd should be 1"


class TestAmoaddWFaultHandling:
    """amoadd.w 在异常条件下的行为."""

    def test_amo_to_device_detected(self, _m_mode_hart):
        """对 MMIO 地址 (如 CLINT) 的 AMO 在 Python 路径下走 Bus 直通路径.

        Python decoder 中的 handle_amo ->mem_write/read 最终走 Bus.write/read,
        对 device MMIO 直接读写设备寄存器 (而非触发 trap).
        这与原生 batch 不同 (原生 batch 中 device AMO 触发 EXIT_MMIO).
        此处仅验证 device 访问不会 panic/crash.
        """
        emu, hart = _m_mode_hart
        instr = _encode_amoadd_w(rd=5, rs1=10, rs2=11, aq=1, rl=1)
        emu.bus.write(0x80000000, struct.pack("<I", instr))
        hart.gprs[10] = 0x02000000  # CLINT MSIP[0]
        hart.gprs[11] = 0
        hart.gprs[5] = 0
        hart.pc = 0x80000000
        # 不应触发未处理异常
        try:
            hart.exec_instr(instr)
        except Exception as exc:
            pytest.fail(f"AMO to CLINT MSIP should not raise: {exc}")
