
auipc t0, 0x0
addi  a1, t0, // &dtb的地址
csrr  a0, mhartid
lw    t0, 24(t0)
ld    t0, 24(t0)
jr    t0

