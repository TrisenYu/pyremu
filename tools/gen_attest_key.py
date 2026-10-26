#!/usr/bin/env python3
# SPDX-LICENSE-IDENTIFIER: MIT
# Author: <kisfg@hotmail.com> 2026
#
# 由 PEM 形式的私钥生成扩展模块映像的签名密钥常量, 供主机目标上的用例在签名映像时恢复。
#
# 调用方式: 以私钥路径为唯一参数执行本脚本, 由调用方重定向标准输出
#
# 生成的字节数组是私钥的编码结果, 逐字节独立:
#
#   t[i] = ((s[i] << 3) | (s[i] >> 5)) & 0xFF      循环左移 3 位
#   t[i] = t[i] ^ MASK[i % 8]                      与位置相关的掩码异或
#
# 恢复即逆运算: 先异或掩码, 再循环右移 3 位, 见 bsp/sittim/tests/ext_mod_host.rs。本脚本由
# bsp/sittim/Makefile 在 make test 之前调用, 生成物写为 bsp/sittim/tests/configs_gen.rs

import base64
import sys

# 异或所用的掩码, 按位置每 8 个字节循环一次。
MASK = bytes([0x5A, 0x3C, 0xE7, 0x91, 0x24, 0xBB, 0x6D, 0xF0])
# 循环移位的位数。
SHIFT = 3
# 私钥标量的字节数。
SCALAR_LEN = 32


def read_tlv(buf: bytes, pos: int) -> tuple[int, bytes, int]:
    """读出一个 DER 项, 返回 (标签, 内容, 下一项的位置)。"""
    if pos + 2 > len(buf):
        raise ValueError("the DER item is truncated")
    tag, lena = buf[pos], buf[pos + 1]
    pos += 2
    if lena & 0x80:
        width = lena & 0x7F
        if width == 0 or pos + width > len(buf):
            raise ValueError("the DER length is not supported")
        lena = int.from_bytes(buf[pos : pos + width], "big")
        pos += width
    if pos + lena > len(buf):
        raise ValueError("the DER item is truncated")
    return tag, buf[pos : pos + lena], pos + lena


def private_scalar(der: bytes) -> bytes:
    """取出 DER 编码的私钥的标量, 兼容 SEC1 与 PKCS#8 两种外层。"""
    outer_tag, body, _ = read_tlv(der, 0)
    if outer_tag != 0x30:
        raise ValueError("the DER document is not a SEQUENCE")
    first_tag, first, pos = read_tlv(body, 0)
    if first_tag != 0x02:
        raise ValueError("the first item of the DER document is not an INTEGER")
    version = int.from_bytes(first, "big")
    if version == 0:
        # PKCS#8 的 PrivateKeyInfo: 算法标识之后是 SEC1 形式的私钥。
        _, _, pos = read_tlv(body, pos)
        inner_tag, inner, _ = read_tlv(body, pos)
        if inner_tag != 0x04:
            raise ValueError("the PKCS#8 document holds no private key")
        return private_scalar(inner)
    if version != 1:
        raise ValueError(f"the private key version {version} is not supported")
    key_tag, key, _ = read_tlv(body, pos)
    if key_tag != 0x04:
        raise ValueError("the SEC1 document holds no private key")
    # DER 的 INTEGER 允许补一个前导 0 字节, 此处按标量的字节数取回。
    if len(key) == SCALAR_LEN + 1 and key[0] == 0:
        key = key[1:]
    if len(key) != SCALAR_LEN:
        raise ValueError(f"the private key is {len(key)} bytes, expected {SCALAR_LEN}")
    return key


def pem_body(pem: str) -> bytes:
    """取出 PEM 文档中的正文并解码。"""
    lines = []
    inside = False
    for raw in pem.splitlines():
        line = raw.strip()
        if line.startswith("-----BEGIN "):
            inside = True
            continue
        if line.startswith("-----END "):
            break
        if inside and line:
            lines.append(line)
    if not lines:
        raise ValueError("the PEM document holds no body")
    return base64.b64decode("".join(lines))


def encode(scalar: bytes) -> bytes:
    """把标量编码为生成的常量的取值。"""
    encoded = bytearray(len(scalar))
    for index, byte in enumerate(scalar):
        rotated = ((byte << SHIFT) | (byte >> (8 - SHIFT))) & 0xFF
        encoded[index] = rotated ^ MASK[index % len(MASK)]
    return bytes(encoded)


def render(encoded: list):
    fields = ", ".join(f"0x{byte:02x}" for byte in encoded)
    return (
        "/// generated from the private key by tools/gen_attest_key.py — do not edit\n\n"
        "/// 受简易异或算法保护的私钥字节取值。\n"
        f"pub const ATTEST_PRIV_KEY_ENCODED: [u8; {len(encoded)}] = [{fields}];\n"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <private key>", file=sys.stderr)
        sys.exit(1)
    key_path = sys.argv[1]
    try:
        with open(key_path, encoding="utf-8") as handle:
            pem = handle.read()
    except OSError as err:
        print(f"cannot read the private key at {key_path}: {err}", file=sys.stderr)
        sys.exit(1)
    try:
        encoded = encode(private_scalar(pem_body(pem)))
    except ValueError as err:
        print(f"cannot decode the private key at {key_path}: {err}", file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(render(encoded))
