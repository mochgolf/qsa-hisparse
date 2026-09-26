# NUMA memory placement for GPU workers

Set `SGLANG_NUMA_INTERLEAVE=1` to interleave memory allocated by SGLang worker
subprocesses across the NUMA nodes allowed by their cpuset. This option applies
to the NUMA V2 binding path, so keep `SGLANG_NUMA_BIND_V2=1`. The default is
`SGLANG_NUMA_INTERLEAVE=0`, which preserves the existing per-GPU memory binding.

The option changes only the worker memory policy. Workers keep the CPU binding
to the GPU-local NUMA node, limited to CPUs in the process's allowed CPU set.
`--interleave=all` is constrained by the nodes allowed to the process. It does
not change model configuration or precision.

The environment variable configures SGLang worker subprocesses. Start the
server parent under an outer `numactl --interleave=all` when its own memory
allocations should use the same policy.

If the kernel rejects the interleave policy, SGLang warns and retries with the
existing CPU-only binding. `SGLANG_CRASH_ON_NUMA_BIND_FAILURE=1` controls the
existing fatal behavior for NUMA binding failures, such as an empty CPU
intersection or a rejected CPU-only binding; it does not make a rejected
memory-only policy fatal when CPU-only binding succeeds.
