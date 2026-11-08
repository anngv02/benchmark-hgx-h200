# Results Summary

[Overview](../README.md) · [Results summary](results.md) · [Benchmark environment](environment.md)

## Single-GPU Baseline Compared with NVIDIA Specifications

| Category | Measured result | NVIDIA specification | Comparison reference | Ratio |
|---|---:|---:|---:|---:|
| FP32 SIMT | **48.12 TFLOPS** | 67 TFLOPS | 67 TFLOPS | **71.8%** |
| TF32 Tensor | **424.12 TFLOPS** | 989 TFLOPS sparse | ~494.5 TFLOPS dense | **85.8%** |
| BF16 Tensor | **702.19 TFLOPS** | 1,979 TFLOPS sparse | ~989.5 TFLOPS dense | **71.0%** |
| FP16 Tensor | **734.53 TFLOPS** | 1,979 TFLOPS sparse | ~989.5 TFLOPS dense | **74.2%** |
| FP8 Standard | **1,242.54 TFLOPS** | 3,958 TFLOPS sparse | ~1,979 TFLOPS dense | **62.8%** |
| FP8 FastAccum | **1,252.80 TFLOPS** | 3,958 TFLOPS sparse | ~1,979 TFLOPS dense | **63.3%** |
| HBM read | **4.548 TB/s** | 4.8 TB/s | 4.8 TB/s | **94.7%** |
| HBM write | **4.333 TB/s** | 4.8 TB/s | 4.8 TB/s | **90.3%** |
| PCIe H2D | **55.46 GB/s** | 128 GB/s bidirectional | ~64 GB/s one-way | **86.7%** |
| PCIe D2H | **55.27 GB/s** | 128 GB/s bidirectional | ~64 GB/s one-way | **86.4%** |
| PCIe concurrent H2D | **38.64 GB/s** | 128 GB/s bidirectional | ~64 GB/s one-way | **60.4%** |
| PCIe concurrent D2H | **52.52 GB/s** | 128 GB/s bidirectional | ~64 GB/s one-way | **82.1%** |
| Tensor workload power | **~685–700 W** | Up to 700 W | 700 W | Near the power cap |

> Tensor Core ratios use the **dense theoretical reference** derived from NVIDIA’s headline sparse figures, rather than comparing directly with sparse TFLOPS.

## Multi-GPU validation

| Category | Baseline | 8 GPUs concurrently | Ratio to baseline |
|---|---:|---:|---:|
| BF16 average/GPU | **702.19 TFLOPS** single-GPU tuned | **711.87 TFLOPS/GPU** | **101.4%** |
| BF16 aggregate | **5.618 PFLOPS** = 8 × single-GPU baseline | **5.695 PFLOPS** | **101.4%** |
| HBM Read average/GPU | **4.536 TB/s** | **4.535 TB/s** | **99.99%** |
| HBM Read aggregate | **36.286 TB/s** | **36.283 TB/s** | **99.99%** |
| PCIe H2D average/GPU | **55.45 GB/s** | **54.25 GB/s** | **97.8%** |
| PCIe H2D aggregate | **443.62 GB/s** | **434.00 GB/s** | **97.8%** |
| PCIe D2H average/GPU | **55.27 GB/s** | **37.36 GB/s** | **67.6%** |
| PCIe D2H aggregate | **442.16 GB/s** | **298.84 GB/s** | **67.6%** |

For BF16, the baseline is the **tuned single-GPU** result on GPU0. The 101.4% figure should not be interpreted as superlinear scaling; it indicates no throughput degradation with 8 GPUs running concurrently.

For HBM and PCIe, the baseline is the average of the individual GPU measurements taken before running all 8 GPUs concurrently.

H200 reference specifications:

- https://www.nvidia.com/en-us/data-center/h200/

Aggregate references are sums of per-GPU values used to evaluate scaling. They do not represent a single shared compute, memory, or PCIe fabric.
