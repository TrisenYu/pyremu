/* tee_enclave.h — Linux TEE enclave driver userspace interface
 *
 * /dev/tee_enclave: ioctl-based enclave lifecycle management.
 * SBI ecall (ext_id=0x20221222) bridges to OpenSBI M-mode enclave extension.
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#ifndef TEE_ENCLAVE_H
#define TEE_ENCLAVE_H

#include <stdint.h>
#include <sys/ioctl.h>

#define TEE_DEVICE_PATH  "/dev/tee_enclave"

/* ---- SBI function IDs (from custom-opensbi-rs) ---- */

#define SBI_ENCLAVE_CREATE           400
#define SBI_ENCLAVE_ENTER            401
#define SBI_ENCLAVE_SHUTDOWN         403
#define SBI_ENCLAVE_SUSPEND          404
#define SBI_ENCLAVE_RESUME           405
#define SBI_ENCLAVE_GET_ID           407
#define SBI_ENCLAVE_GET_HARTID       408
#define SBI_ENCLAVE_GET_AVAILABLE_MEM 409
#define SBI_ENCLAVE_MEM_ALLOC        500

/* ---- ioctl command codes ---- */

#define TEE_IOC_MAGIC  'T'

/* Create a new enclave.  Returns enclave_id; does NOT return if successful
 * (execution transfers into enclave).  Caller sees return only on error. */
#define TEE_IOC_CREATE     _IO(TEE_IOC_MAGIC, 0)

/* Enter an existing enclave with a payload.
 * On success, thread enters enclave and does not return until SUSPEND. */
#define TEE_IOC_ENTER      _IOW(TEE_IOC_MAGIC, 1, struct tee_enter_args)

/* Query current mdid (0 = host). */
#define TEE_IOC_GET_ID     _IOR(TEE_IOC_MAGIC, 2, uint64_t)

/* Query available memory pool (2 MiB units). */
#define TEE_IOC_GET_MEM    _IOR(TEE_IOC_MAGIC, 3, struct tee_mem_info)

/* Suspend current enclave, returning to host.  Only valid inside enclave. */
#define TEE_IOC_SUSPEND    _IO(TEE_IOC_MAGIC, 4)

/* Resume a previously suspended enclave by id. */
#define TEE_IOC_RESUME     _IOW(TEE_IOC_MAGIC, 5, uint64_t)

/* Shutdown current enclave (clears memory, frees slot, mfence.did). */
#define TEE_IOC_SHUTDOWN   _IO(TEE_IOC_MAGIC, 6)

/* ---- parameter structures ---- */

struct tee_enter_args {
    uint64_t enclave_id;   /* target enclave id (from CREATE) */
    uint64_t payload_ptr;  /* userspace pointer to payload data */
    uint64_t payload_size; /* payload size in bytes */
    uint64_t argc;         /* argument count for enclave entry */
    uint64_t argv_ptr;     /* userspace pointer to argv array */
};

struct tee_mem_info {
    uint64_t free_total;      /* free 2 MiB partitions */
    uint64_t max_contiguous;  /* largest contiguous run */
};

#endif /* TEE_ENCLAVE_H */
