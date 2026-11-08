# Compute Benchmarks

[Overview](../README.md) · [Results summary](results.md) · [Benchmark environment](environment.md)

## Kernel Selection Methodology

### Why Select Kernels Before Measurement

GEMM performance depends on precision, tile size, cluster shape, scheduler, epilogue, `raster_order`, and `swizzle_size`. To obtain representative performance, the report follows this procedure:

```text
1. Fix the shape at M=N=K=16384
2. Search compatible kernels compiled into CUTLASS Profiler
3. Select the kernel/configuration with the highest GFLOPs
4. Keep the kernel + raster_order + swizzle_size fixed
5. Run two sustained measurements with 50 warm-up + 200 profiling iterations
6. Report the average
```

CUTLASS Profiler supports kernel searches for a fixed shape:

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html

Example search command:

```bash
CUDA_VISIBLE_DEVICES=0 \
$CUTLASS_PROFILER \
  --operation=gemm \
  --kernels='<PATTERN>' \
  --enable-best-kernel-for-fixed-shape \
  --sort-results-flops-per-sec \
  --m=16384 --n=16384 --k=16384 \
  --verification-enabled=false \
  --output=<OUTPUT>
```

> Here, “best kernel” means the best among **the kernels compiled into the current CUTLASS binary**. It does not claim to be the absolute best of all kernels CUTLASS can generate.

---

## FP32 SIMT

### Why Measure This

FP32 SIMT measures the basic floating-point capability of CUDA Cores, independently of Tensor Cores.

### Kernel Selection

Search SGEMM SIMT kernels with the same layout:

```bash
CUDA_VISIBLE_DEVICES=0 \
$CUTLASS_PROFILER \
  --operation=gemm \
  --kernels='cutlass_simt_sgemm_*_nt_align1' \
  --m=16384 --n=16384 --k=16384 \
  --sort-results-flops-per-sec \
  --profiling-iterations=20 \
  --verification-enabled=false \
  --output=/root/h200-benchmark/logs/fp32_simt_search
```

The current binary contains two matching kernels:

```text
48.1184 TFLOPS  cutlass_simt_sgemm_256x128_8x5_nt_align1
41.9062 TFLOPS  cutlass_simt_sgemm_128x128_8x2_nt_align1
```

Selected kernel:

```text
cutlass_simt_sgemm_256x128_8x5_nt_align1
```

### Sustained Measurement Command

```bash
CUDA_VISIBLE_DEVICES=0 \
$CUTLASS_PROFILER \
  --operation=gemm \
  --kernels=cutlass_simt_sgemm_256x128_8x5_nt_align1 \
  --m=16384 --n=16384 --k=16384 \
  --warmup-iterations=50 \
  --profiling-iterations=200
```

### Results

```text
48.1184 TFLOPS
```

### Comparison with NVIDIA Specifications

NVIDIA H200 SXM FP32 peak:

```text
67 TFLOPS
```

Sources:

- https://www.nvidia.com/en-us/data-center/h200/

Efficiency:

```text
48.1184 / 67 ≈ 71.8%
```

**Assessment:** FP32 SIMT achieves approximately **71.8% of theoretical peak**.

---

## TF32 Tensor Core

### Why Measure This

TF32 enables workloads using FP32 data to use Tensor Cores for higher throughput than FP32 SIMT.

### Kernel Selection

The fixed-shape search selected:

```text
Kernel:
cutlass3x_sm90_tensorop_tf32gemm_f32_f32_f32_f32_f32_256x128x32_1x2x1_0_tnn_align4_stream_k_warpspecialized_cooperative_epi_nosmem

raster_order = along_m
swizzle_size = 8
```

### Sustained Measurement Command

```bash
CUDA_VISIBLE_DEVICES=0 \
$CUTLASS_PROFILER \
  --operation=gemm \
  --kernels=cutlass3x_sm90_tensorop_tf32gemm_f32_f32_f32_f32_f32_256x128x32_1x2x1_0_tnn_align4_stream_k_warpspecialized_cooperative_epi_nosmem \
  --m=16384 --n=16384 --k=16384 \
  --raster_order=along_m \
  --swizzle_size=8 \
  --warmup-iterations=50 \
  --profiling-iterations=200 \
  --verification-enabled=false
```

### Results

```text
Run 1: 424.555 TFLOPS
Run 2: 423.686 TFLOPS
Average: 424.121 TFLOPS
```

The difference between the two runs is approximately **0.2%**.

### Comparison with NVIDIA Specifications

NVIDIA specifies:

```text
TF32 Tensor Core = 989 TFLOPS (with sparsity)
```

Dense reference:

```text
989 / 2 ≈ 494.5 TFLOPS
```

Efficiency:

```text
424.121 / 494.5 ≈ 85.8%
```

Sources:

- https://www.nvidia.com/en-us/data-center/h200/
- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

**Assessment:** Tuned sustained TF32 achieves approximately **85.8% of the dense theoretical reference**.

---

## BF16 Tensor Core

### Why Measure This

BF16 is widely used in training and inference because it preserves a dynamic range close to FP32 while providing high Tensor Core throughput.

### Kernel Selection

The fixed-shape search selected:

```text
Kernel:
cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_f32_f32_128x128x64_1x2x1_0_ttn_align8_warpspecialized_cooperative_epi_tma

raster_order = along_m
swizzle_size = 8
```

### Sustained Measurement Command

```bash
CUDA_VISIBLE_DEVICES=0 \
$CUTLASS_PROFILER \
  --operation=gemm \
  --kernels=cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_f32_f32_128x128x64_1x2x1_0_ttn_align8_warpspecialized_cooperative_epi_tma \
  --m=16384 --n=16384 --k=16384 \
  --raster_order=along_m \
  --swizzle_size=8 \
  --warmup-iterations=50 \
  --profiling-iterations=200 \
  --verification-enabled=false
```

### Results

```text
Run 1: 701.930 TFLOPS
Run 2: 702.455 TFLOPS
Average: 702.193 TFLOPS
```

The difference between the two runs is approximately **0.07%**.

### Comparison with NVIDIA Specifications

NVIDIA specifies:

```text
BF16 Tensor Core = 1,979 TFLOPS (with sparsity)
```

Dense reference:

```text
1979 / 2 ≈ 989.5 TFLOPS
```

Efficiency:

```text
702.193 / 989.5 ≈ 71.0%
```

Sources:

- https://www.nvidia.com/en-us/data-center/h200/
- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

**Assessment:** Tuned sustained BF16 achieves approximately **71.0% of the dense theoretical reference**.

---

## FP16 Tensor Core

### Why Measure This

FP16 is a common precision for deep learning and enables high Tensor Core throughput.

### Kernel Selection

The fixed-shape search selected:

```text
Kernel:
cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_ttn_align8_stream_k_warpspecialized_cooperative_epi_tma

raster_order = along_n
swizzle_size = 8
```

### Sustained Measurement Command

```bash
CUDA_VISIBLE_DEVICES=0 \
$CUTLASS_PROFILER \
  --operation=gemm \
  --kernels=cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_ttn_align8_stream_k_warpspecialized_cooperative_epi_tma \
  --m=16384 --n=16384 --k=16384 \
  --raster_order=along_n \
  --swizzle_size=8 \
  --warmup-iterations=50 \
  --profiling-iterations=200 \
  --verification-enabled=false
```

### Results

```text
Run 1: 733.754 TFLOPS
Run 2: 735.312 TFLOPS
Average: 734.533 TFLOPS
```

The difference between the two runs is approximately **0.21%**.

### Comparison with NVIDIA Specifications

```text
FP16 Tensor Core = 1,979 TFLOPS (with sparsity)
Dense reference ≈ 989.5 TFLOPS
```

Efficiency:

```text
734.533 / 989.5 ≈ 74.2%
```

Sources:

- https://www.nvidia.com/en-us/data-center/h200/
- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

**Assessment:** Tuned sustained FP16 achieves approximately **74.2% of the dense theoretical reference**.

---

## FP8 E4M3 Standard

### Why Measure This

FP8 is a key precision on Hopper for high AI throughput. Standard accumulation is measured separately to establish a baseline with numerical characteristics distinct from FastAccum.

### Kernel Selection

The Standard FP8 search excludes `fastaccum` kernels. The best configuration for the `16384³` shape is:

```text
Kernel:
cutlass3x_sm90_tensorop_gemm_e4m3_e4m3_f32_f32_f32_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma

raster_order = along_m
swizzle_size = 8
```

### Sustained Measurement Command

```bash
CUDA_VISIBLE_DEVICES=0 \
$CUTLASS_PROFILER \
  --operation=gemm \
  --kernels=cutlass3x_sm90_tensorop_gemm_e4m3_e4m3_f32_f32_f32_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma \
  --m=16384 --n=16384 --k=16384 \
  --raster_order=along_m \
  --swizzle_size=8 \
  --warmup-iterations=50 \
  --profiling-iterations=200 \
  --verification-enabled=false
```

### Results

```text
Run 1: 1243.18 TFLOPS
Run 2: 1241.89 TFLOPS
Average: 1242.54 TFLOPS ≈ 1.243 PFLOPS
```

The difference is approximately **0.10%**.

### Comparison with NVIDIA Specifications

```text
FP8 Tensor Core = 3,958 TFLOPS (with sparsity)
Dense reference ≈ 1,979 TFLOPS
```

Efficiency:

```text
1242.54 / 1979 ≈ 62.8%
```

Sources:

- https://www.nvidia.com/en-us/data-center/h200/
- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

**Assessment:** Tuned sustained FP8 Standard achieves approximately **62.8% of the dense theoretical reference**.

---

## FP8 E4M3 FastAccum

### Why Measure This

FastAccum is measured separately because its accumulation mechanism has different numerical trade-offs from Standard accumulation.

### Measurement Command

```bash
CUDA_VISIBLE_DEVICES=0 \
$CUTLASS_PROFILER \
  --operation=gemm \
  --kernels=cutlass3x_sm90_tensorop_gemm_e4m3_e4m3_f32_f32_f32_128x128x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmem \
  --m=16384 --n=16384 --k=16384 \
  --warmup-iterations=50 \
  --profiling-iterations=200
```

### Results

```text
Average: 1252.80 TFLOPS ≈ 1.253 PFLOPS
```

### Comparison with NVIDIA Specifications

Dense FP8 reference:

```text
3,958 / 2 = 1,979 TFLOPS
```

Efficiency:

```text
1252.80 / 1979 ≈ 63.3%
```

Sources:

- https://www.nvidia.com/en-us/data-center/h200/
- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

Compared with tuned Standard accumulation:

```text
1252.80 / 1242.54 - 1 ≈ 0.8%
```

**Assessment:** For this benchmark shape and configuration, FastAccum is only about **0.8%** faster than tuned Standard accumulation. The main distinction between these modes is their accumulation and numerical behavior, rather than a large throughput gap.

---

## BF16 with 8 GPUs Running Concurrently

### Why Measure This

After selecting the optimal BF16 kernel on GPU0, running the same kernel/configuration concurrently on all 8 GPUs checks:

- whether throughput drops when the entire node is under compute load;
- consistency across all 8 GPUs under the same load conditions.

### Measurement Command

```bash
KERNEL='cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_f32_f32_128x128x64_1x2x1_0_ttn_align8_warpspecialized_cooperative_epi_tma'

for g in {0..7}; do
  CUDA_VISIBLE_DEVICES=$g \
  $CUTLASS_PROFILER \
    --operation=gemm \
    --kernels="$KERNEL" \
    --m=16384 --n=16384 --k=16384 \
    --raster_order=along_m \
    --swizzle_size=8 \
    --warmup-iterations=50 \
    --profiling-iterations=200 \
    --verification-enabled=false \
    > /root/h200-benchmark/logs/8gpu_bf16_tuned/gpu${g}.txt 2>&1 &
done

wait
```

Monitor the entire node:

```bash
nvidia-smi dmon -s pucmt -d 1
```

### Results

| GPU | BF16 concurrent |
|---|---:|
| GPU0 | 700.499 TFLOPS |
| GPU1 | 723.194 TFLOPS |
| GPU2 | 702.109 TFLOPS |
| GPU3 | 717.712 TFLOPS |
| GPU4 | 729.263 TFLOPS |
| GPU5 | 704.871 TFLOPS |
| GPU6 | 713.122 TFLOPS |
| GPU7 | 704.181 TFLOPS |
| **Average** | **711.869 TFLOPS/GPU** |

Aggregate:

```text
5.695 PFLOPS
```

Single-GPU tuned baseline:

```text
702.193 TFLOPS/GPU
```

Multiplying the single-GPU baseline by 8:

```text
8 × 702.193 ≈ 5.618 PFLOPS
```

Ratio of measured concurrent throughput to the baseline:

```text
5.695 / 5.618 ≈ 101.4%
```

A value above 100% should not be interpreted as “superlinear scaling”; the difference may result from clock/power states and run-to-run variation. The key finding is that **no throughput degradation was observed with all 8 GPUs running concurrently**.

The per-GPU range under concurrent load is:

```text
700.499 – 729.263 TFLOPS
```

This corresponds to approximately 4% between the minimum and maximum.

**Assessment:** BF16 compute performance is sustained under load across all 8 H200 GPUs, reaching approximately **5.695 PFLOPS** in aggregate.
