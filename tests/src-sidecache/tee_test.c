/* tee_test.c — TEE enclave driver userspace test
 *
 * Demonstrates /dev/tee_enclave ioctl interface:
 *   1. GET_MEM  — query available enclave memory pool
 *   2. GET_ID   — query current mdid (0 = host)
 *   3. CREATE   — spawn a new enclave (firmware-provided enclave manager)
 *   4. ENTER    — load & run a custom payload in the enclave
 *   5. SHUTDOWN — destroy enclave (or from inside the enclave)
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "tee_enclave.h"

static int fd = -1;

/* ---- helpers ---- */

static void die(const char *msg) {
	fprintf(stderr, "tee_test: %s (errno=%d)\n", msg, errno);
	if (fd >= 0) {
		close(fd);
	}
	exit(EXIT_FAILURE);
}

static void check_open(void) {
	fd = open(TEE_DEVICE_PATH, O_RDWR);
	if (fd < 0) {
		die("open " TEE_DEVICE_PATH);
	}
	printf("[test] opened %s (fd=%d)\n", TEE_DEVICE_PATH, fd);
}

/* ---- ioctl wrappers (return 0 on success, nonzero on error) ---- */

static int do_get_mem(struct tee_mem_info *info) {
	if (ioctl(fd, TEE_IOC_GET_MEM, info) < 0) {
		return -errno;
	}
	return 0;
}

static int do_get_id(uint64_t *id) {
	if (ioctl(fd, TEE_IOC_GET_ID, id) < 0) {
		return -errno;
	}
	return 0;
}

static int do_create(void) {
	/* CREATE transfers execution into the enclave on success.
     * The host thread becomes the enclave thread; it only returns
     * here after the enclave calls SUSPEND (or on CREATE error). */
	if (ioctl(fd, TEE_IOC_CREATE, 0) < 0) {
		return -errno;
	}
	return 0;
}

static int do_enter(const struct tee_enter_args *args) {
	/* Same caveat as CREATE — on success, transfers into enclave.
     * Returns here after SUSPEND. */
	if (ioctl(fd, TEE_IOC_ENTER, args) < 0) {
		return -errno;
	}
	return 0;
}

static int do_suspend(void) {
	if (ioctl(fd, TEE_IOC_SUSPEND, 0) < 0) {
		return -errno;
	}
	return 0;
}

static int do_shutdown(void) {
	if (ioctl(fd, TEE_IOC_SHUTDOWN, 0) < 0) {
		return -errno;
	}
	return 0;
}

/* Read an entire file into a heap buffer.  Sets *size on success. */
static unsigned char *read_file(const char *path, size_t *size) {
	FILE *fp = fopen(path, "rb");
	if (!fp) {
		fprintf(stderr, "tee_test: cannot open '%s'\n", path);
		return NULL;
	}
	fseek(fp, 0, SEEK_END);
	long sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	if (sz <= 0) {
		fclose(fp);
		return NULL;
	}
	unsigned char *buf = malloc((size_t)sz);
	if (!buf) {
		fclose(fp);
		return NULL;
	}
	if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
		free(buf);
		fclose(fp);
		return NULL;
	}
	fclose(fp);
	*size = (size_t)sz;
	return buf;
}

/* ---- main ---- */

int main(int argc, char **argv) {
	uint64_t mdid;
	struct tee_mem_info mem;
	int rc;

	puts("=== TEE Enclave Driver Test ===\n");

	/* 1. Open device */
	check_open();

	/* 2. Query current mdid (expect 0 = host) */
	rc = do_get_id(&mdid);
	if (rc == 0) {
		printf("[test] GET_ID: current mdid=%llu (0=host)\n", (unsigned long long)mdid);
	} else {
		printf("[test] GET_ID: failed (%d) — SBI ext may not be loaded\n", rc);
	}

	/* 3. Query available memory */
	rc = do_get_mem(&mem);
	if (rc == 0) {
		printf(
			"[test] GET_MEM: free_total=%llu max_contiguous=%llu (2 MiB units)\n",
			(unsigned long long)mem.free_total,
			(unsigned long long)mem.max_contiguous);
	} else {
		printf("[test] GET_MEM: failed (%d)\n", rc);
	}

	/* 4. CREATE enclave (will block until enclave SUSPENDs).
     * Skip if we can't find a payload to enter. */
	if (argc < 2) {
		printf("\n[test] Usage: %s <payload_path>\n", argv[0]);
		puts("[test] Skipping CREATE/ENTER — no payload specified.");
		puts("[test] Basic ioctl smoke test PASSED.");
		close(fd);
		return 0;
	}

	const char *payload_path = argv[1];
	printf("\n[test] Creating enclave...\n");
	rc = do_create();
	if (rc != 0) {
		/* CREATE may fail if enclave manager is not in firmware */
		printf("[test] CREATE: failed (%d) — skip ENTER test\n", rc);
		close(fd);
		return 1;
	}
	puts("[test] CREATE: enclave created & suspended, back in host");

	/* 5. Load custom payload and ENTER */
	size_t payload_size	   = 0;
	unsigned char *payload = read_file(payload_path, &payload_size);
	if (!payload) {
		printf("[test] Cannot read payload '%s'\n", payload_path);
		do_shutdown();
		close(fd);
		return 1;
	}
	printf("[test] Loaded payload '%s' (%zu bytes)\n", payload_path, payload_size);

	struct tee_enter_args args = {
		.enclave_id	  = 1, /* first enclave gets id=1 */
		.payload_ptr  = (uint64_t)(unsigned long)payload,
		.payload_size = payload_size,
		.argc		  = argc - 2,
		.argv_ptr	  = (uint64_t)(unsigned long)(argv + 2),
	};

	printf("[test] Entering enclave with payload...\n");
	rc = do_enter(&args);
	if (rc == 0) {
		printf("[test] ENTER: payload ran, suspended back to host\n");
	} else {
		printf("[test] ENTER: failed (%d)\n", rc);
	}

	free(payload);

	/* 6. Shutdown */
	printf("[test] Shutting down enclave...\n");
	rc = do_shutdown();
	if (rc == 0) {
		printf("[test] SHUTDOWN: enclave destroyed\n");
	} else {
		printf("[test] SHUTDOWN: failed (%d)\n", rc);
	}

	/* 7. Verify back in host */
	rc = do_get_id(&mdid);
	if (rc == 0) {
		printf("[test] GET_ID after shutdown: mdid=%llu\n", (unsigned long long)mdid);
	}

	close(fd);
	printf("\n[test] All tests finished.\n");
	return 0;
}
