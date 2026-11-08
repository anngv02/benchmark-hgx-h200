# Benchmark Environment

[Overview](../README.md) · [Results summary](results.md) · [Benchmark environment](environment.md)

| Component | Details |
|---|---|
| GPU | NVIDIA H200 SXM 141 GB |
| Benchmark GPU scope | GPU0 for the detailed baseline; 8 GPUs for concurrent BF16; GPU0–GPU7 for HBM/PCIe validation |
| PCIe BDF | `18:00.0` |
| Driver | 580.159.03 |
| CUDA Toolkit | 13.0 |
| CUTLASS | v4.8.0dev-6-g59e3a333 |
| nvbandwidth | v0.10.0 |
| NUMA binding for Host↔GPU | Bind to the local NUMA node: GPU0–1→0, GPU2–3→1, GPU4–5→2, GPU6–7→3 |
| GPU power limit | 700 W |

The GEMM tests use:

```text
M = N = K = 16384
Warm-up iterations = 50
Profiling iterations = 200
```

All compute benchmarks use the same `16384³` shape. For Tensor Core tests, CUTLASS Profiler first searches for the **best kernel/configuration among the compiled kernels** for this fixed shape. The best kernel is then rerun with 50 warm-up iterations and 200 profiling iterations to obtain sustained results.

For PCIe tests, each GPU is bound to its nearest NUMA node:

```text
GPU0, GPU1 → NUMA0
GPU2, GPU3 → NUMA1
GPU4, GPU5 → NUMA2
GPU6, GPU7 → NUMA3
```

NVIDIA provides CUTLASS Profiler to run and profile GEMM kernels:

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html

`nvbandwidth` is NVIDIA’s tool for measuring bandwidth between the host and GPUs, and between GPU memory regions:

- https://github.com/NVIDIA/nvbandwidth

H200 SXM reference specifications used for comparison:

- https://www.nvidia.com/en-us/data-center/h200/

NVIDIA lists TF32/BF16/FP16/FP8 Tensor Core figures on the H200 page with the qualifier **“With sparsity”**. For the dense GEMMs in this report, the dense reference is approximated as half the sparse peak. This conversion follows Hopper’s 2:4 structured sparsity mechanism, which can double Tensor Core throughput relative to equivalent dense operations:

- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
