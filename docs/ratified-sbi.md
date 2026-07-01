## Chapter 2. Terms and Abbreviations
This specification uses the following terms and abbreviations:
-----------------------------------------------
Term Meaning
-----------------------------------------------
ACPI Advanced Configuration and Power Interface
ASID Address Space Identifier
BMC Baseboard Management Controller
CPPC Collaborative Processor Performance Control
EID Extension ID
FID Function ID
HSM Hart State Management
IPI Inter Processor Interrupt
PMU Performance Monitoring Unit
SBI Supervisor Binary Interface
SEE Supervisor Execution Environment
VMID Virtual Machine Identifier
-----------------------------------------------

## Chapter 3. Binary Encoding
ll SBI functions share a single binary encoding, which facilitates the mixing of SBI extensions. The SBI
specification follows the below calling convention.
- An ECALL is used as the control transfer instruction between the supervisor and the SEE.
- `a7` encodes the SBI extension ID (EID).
- `a6` encodes the SBI function ID (FID) for a given extension ID encoded in a7 for any SBI extension
defined in or after SBI v0.2.
- `a0` through a5 contain the arguments for the SBI function call. Registers that are not defined in the SBI function call are not reserved.
- All registers except `a0 & a1` must be preserved across an SBI call by the callee.
- SBI functions must return a pair of values in `a0` and `a1`, with `a0` returning an error code. This is analogous to returning the C structure

```c
struct sbiret {
	long error;
	union {
		long value;
		unsigned long uvalue;
	};
};
```
Data type long in C pseudocode is XLEN bits wide.
In the name of compatibility, SBI extension IDs (EIDs) and SBI function IDs (FIDs) are encoded as signed
32-bit integers. When passed in registers these follow the standard above calling convention rules.
The Table 1 below provides a list of Standard SBI error codes.

Table 1. Standard SBI Errors
-----------------------------------------------
Error Type Value Description
-----------------------------------------------
SBI_SUCCESS                0 Completed successfully
SBI_ERR_FAILED            -1 Failed
SBI_ERR_NOT_SUPPORTED     -2 Not supported
SBI_ERR_INVALID_PARAM     -3 Invalid parameter(s)
SBI_ERR_DENIED            -4 Denied or not allowed
SBI_ERR_INVALID_ADDRESS   -5 Invalid address(s)
SBI_ERR_ALREADY_AVAILABLE -6 Already available
SBI_ERR_ALREADY_STARTED   -7 Already started
SBI_ERR_ALREADY_STOPPED   -8 Already stopped
SBI_ERR_NO_SHMEM          -9 Shared memory not available
SBI_ERR_INVALID_STATE    -10 Invalid state
SBI_ERR_BAD_RANGE        -11 Bad (or invalid) range
SBI_ERR_TIMEOUT          -12 Failed due to timeout
SBI_ERR_IO               -13 Input/Output error
SBI_ERR_DENIED_LOCKED    -14 Denied or not allowed due to lock status
-----------------------------------------------

An ECALL with an unsupported SBI extension ID (EID) or an unsupported SBI function ID (FID) must
return the error code SBI_ERR_NOT_SUPPORTED.
If an SBI function call returns an error code other than SBI_SUCCESS, the value returned in a1 is
unspecified unless explicitly defined for that SBI function.
Every SBI function should prefer unsigned long as the data type. It keeps the specification simple and
easily adaptable for all RISC-V ISA types. In case the data is defined as 32bit wide, higher privilege software
must ensure that it only uses 32 bit data. Parameters that are 2×XLEN bits wide are passed in a pair of
argument registers, with the low-order XLEN bits in the lower-numbered register and the high-order XLEN
bits in the higher-numbered register.
