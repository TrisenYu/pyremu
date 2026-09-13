// payload_embed.S — 以 .incbin 嵌入待加载的飞地载荷 (cfrac 静态 ELF).
//
// Makefile 先把 $(PAYLOAD) 拷贝为当前目录下的 payload.bin, 再由本文件嵌入
// .rodata.payload. 链接后 payload_blob / payload_blob_end 是载荷在 easy_kern
// 二进制中的起止地址 (Bare 模式下即物理地址), main 将这对地址作为 ENTER
// 的载荷指针与大小传给 M 模式, M 模式据此拷入飞地内存并交给运行时装载.
	.section .rodata.payload,"a",@progbits
	.globl payload_blob
	.globl payload_blob_end
payload_blob:
	.incbin "payload.bin"
payload_blob_end:

	.section .note.GNU-stack,"",@progbits
