/* nc_udp — 飞地内 UDP 自收自发的套接字通路载荷
 *
 * 协议收发由 busybox 的 nc applet 完成, 本载荷用的是 vendor/busybox 的源码, 上游文件
 * 未作改动。本文件是它的启动外壳, 只做两件 busybox 自身做不到的事:
 *
 *   1. 把标准输入换成 ramfs 上的一个文件。nc 的发送方向读自标准输入, 而飞地的标准输入
 *      接在控制台上, 飞地又没有字符输入源, 读到的恒为文件结束, nc 因此不会发出任何
 *      字节。自检标记即写入该文件。
 *   2. 构造 argv。飞地载荷的 argv 由宿主给定, 其中的程序名不是 applet 名, 故载荷不读
 *      自己的 argv[0], 而是把 "busybox" "nc" 与固定参数表交给 busybox 的入口。
 *
 * 标准输入是打开文件表中的一条普通表项, 关闭后编号 0 由随后的 open 复用, 故先
 * close(0) 再打开该文件即取得编号 0; open 返回的编号不是 0 时本载荷立即失败。
 *
 * nc 从标准输入读出标记后经套接字发出, 收到的字节写回描述符 1, 故标记出现在控制台即
 * 表示套接字接口、协议栈与网卡通路全程成立。
 *
 * 运行参数: nc_udp [地址 [端口]]。地址不是点分四段十进制形式时取默认地址, 端口不是
 * 1 至 65535 的十进制数时取默认端口; 本地端口与目的端口取同一个值。
 *
 * 编译: 见同目录 Makefile。
 */

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

/* busybox 的入口, 构建时由 busybox 的 main 改名而来 (见同目录 Makefile)。 */
extern int bb_main(int argc, char **argv);

/* 承载自检标记的文件, 标准输入在其上重新打开。 */
#define MARKER_PATH "udp_msg"

/* 自检标记。本文件不把它写入描述符 1, 它只能经 nc 的接收方向到达控制台。 */
#define MARKER_TEXT "nc_udp: udp round trip ok\n"

#define DEFAULT_ADDR "10.0.0.2"
#define DEFAULT_PORT "5555"

/* nc 的参数: "busybox" "nc" -u -w 1 -p 端口 地址 端口 */
#define NC_ARGC 8

/* *text* 为点分四段十进制地址时返回 1, 其余输入返回 0。 */
static int is_dotted_quad(const char *text)
{
	int parts = 0;

	for (;;) {
		int digits = 0;
		int value = 0;

		while (*text >= '0' && *text <= '9') {
			value = value * 10 + (*text - '0');
			digits++;
			text++;
		}
		if (digits == 0 || digits > 3 || value > 255) {
			return 0;
		}
		parts++;
		if (*text != '.') {
			break;
		}
		text++;
	}
	return parts == 4 && *text == '\0';
}

/* *text* 为 1 至 65535 的十进制数时返回 1, 其余输入返回 0。 */
static int is_port(const char *text)
{
	int digits = 0;
	int value = 0;

	while (*text >= '0' && *text <= '9') {
		value = value * 10 + (*text - '0');
		digits++;
		text++;
	}
	return digits > 0 && digits <= 5 && value >= 1 && value <= 65535 && *text == '\0';
}

/* 把 *text* 写入描述符 *fd*。 */
static void say(int fd, const char *text)
{
	const char *end = text;

	while (*end) {
		end++;
	}
	write(fd, text, (size_t)(end - text));
}

/* 报告返回编号 *value* 的调用, 供失败路径使用。 */
static void report_call(const char *call, int value)
{
	char line[96];
	int len = snprintf(line, sizeof(line), "nc_udp: %s returned %d\n", call, value);

	if (len > 0) {
		write(2, line, (size_t)len);
	}
}

int main(int argc, char **argv)
{
	const char *addr = DEFAULT_ADDR;
	const char *port = DEFAULT_PORT;
	char line[96], *bb_argv[NC_ARGC + 2];
	int len, fd, rc;

	if (argc > 1 && is_dotted_quad(argv[1])) {
		addr = argv[1];
		if (argc > 2 && is_port(argv[2])) {
			port = argv[2];
		}
	}

	/* 标准输入换为 ramfs 上的文件: nc 的发送方向读自标准输入。 */
	close(0);
	fd = open(MARKER_PATH, O_CREAT | O_RDWR, 0600);
	if (fd != 0) {
		report_call("open " MARKER_PATH, fd);
		return 1;
	}
	if (write(0, MARKER_TEXT, sizeof(MARKER_TEXT) - 1) != (ssize_t)(sizeof(MARKER_TEXT) - 1)) {
		say(2, "nc_udp: write " MARKER_PATH " failed\n");
		return 1;
	}
	if (lseek(0, 0, SEEK_SET) != 0) {
		say(2, "nc_udp: lseek " MARKER_PATH " failed\n");
		return 1;
	}

	len = snprintf(line, sizeof(line), "nc_udp: stdin=%s peer=%s:%s\n", MARKER_PATH, addr, port);
	if (len > 0) {
		write(1, line, (size_t)len);
	}

	bb_argv[0] = "busybox";
	bb_argv[1] = "nc";
	bb_argv[2] = "-u";
	bb_argv[3] = "-w";
	bb_argv[4] = "1";
	bb_argv[5] = "-p";
	bb_argv[6] = (char *)port;
	bb_argv[7] = (char *)addr;
	bb_argv[8] = (char *)port;
	bb_argv[9] = NULL;

	/* nc 可能自行退出而不返回, 此时不打印这一行。 */
	rc = bb_main(NC_ARGC + 1, bb_argv);

	len = snprintf(line, sizeof(line), "nc_udp: nc exit=%d\n", rc);
	if (len > 0) {
		write(1, line, (size_t)len);
	}
	return rc;
}
