# HBM3e Benchmarks

[Overview](../README.md) · [Results summary](results.md) · [Benchmark environment](environment.md)

## Why Measure This

HBM bandwidth determines performance for memory-bound workloads. NVIDIA specifies **141 GB of HBM3e with 4.8 TB/s of memory bandwidth** for the H200 SXM.

NVIDIA reference source:

- https://www.nvidia.com/en-us/data-center/h200/

Measurement tool:

- https://github.com/NVIDIA/nvbandwidth

## Measurement Command with a 512 MiB Buffer

```bash
CUDA_VISIBLE_DEVICES=0 \
./build/nvbandwidth \
  -t device_local_read_sm device_local_write_sm device_local_copy_sm \
  -b 512 \
  -i 10 \
  | tee /root/h200-benchmark/logs/hbm_512m.txt
```

### Results

| Test | Result |
|---|---:|
| HBM read SM | 4580.49 GB/s |
| HBM write SM | 4371.65 GB/s |
| Local copy SM | 1990.39 GB/s |

## Sustained Measurement Command with a 4 GiB Buffer

```bash
CUDA_VISIBLE_DEVICES=0 \
./build/nvbandwidth \
  -t device_local_read_sm device_local_write_sm device_local_copy_sm \
  -b 4096 \
  -i 10 \
  | tee /root/h200-benchmark/logs/hbm_4g.txt
```

### Results

| Test | Result | NVIDIA HBM reference | Ratio |
|---|---:|---:|---:|
| Read SM | **4547.69 GB/s** | 4800 GB/s | **94.7%** |
| Write SM | **4332.61 GB/s** | 4800 GB/s | **90.3%** |
| Local Copy SM | 1983.55 GB/s | Not directly comparable | — |

`device_local_copy_sm` reads the source and writes the destination, so its reported value is not directly comparable to the 4.8 TB/s read-only/write-only reference.

### Assessment

Increasing the buffer size from 512 MiB to 4 GiB:

```text
Read : 4580.49 → 4547.69 GB/s
Write: 4371.65 → 4332.61 GB/s
```

The decrease is very small, indicating stable sustained HBM3e bandwidth. Reads achieve nearly **95%**, and writes exceed **90%**, of NVIDIA’s theoretical memory bandwidth.

---


## HBM Reads on Each GPU, GPU0–GPU7

### Why Measure This

Measure HBM reads sequentially on all 8 GPUs to check HBM3e consistency and identify any GPU with unusually low memory subsystem performance.

NVIDIA reference specification: the H200 SXM provides **4.8 TB/s of HBM3e bandwidth**.

- https://www.nvidia.com/en-us/data-center/h200/

### Measurement Command

```bash
NVBW=/root/h200-benchmark/src/nvbandwidth/build/nvbandwidth

for g in {0..7}; do
  CUDA_VISIBLE_DEVICES=$g \
  $NVBW \
    -t device_local_read_sm \
    -b 4096 \
    -i 10 \
    | tee /root/h200-benchmark/logs/per_gpu/gpu${g}_hbm_read.txt
done
```

### Results

| GPU | HBM Read | Relative to 4.8 TB/s |
|---|---:|---:|
| GPU0 | 4549.50 GB/s | 94.8% |
| GPU1 | 4448.46 GB/s | 92.7% |
| GPU2 | 4549.35 GB/s | 94.8% |
| GPU3 | 4548.60 GB/s | 94.8% |
| GPU4 | 4544.23 GB/s | 94.7% |
| GPU5 | 4547.84 GB/s | 94.7% |
| GPU6 | 4548.60 GB/s | 94.8% |
| GPU7 | 4549.05 GB/s | 94.8% |
| **Average** | **4535.70 GB/s** | **94.5%** |

GPU1 is approximately 2.2% below the others, but this result is repeatable and has not indicated a fault.
