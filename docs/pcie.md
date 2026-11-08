# PCIe Host ↔ GPU Benchmarks

[Overview](../README.md) · [Results summary](results.md) · [Benchmark environment](environment.md)

## Why Measure This

PCIe bandwidth determines data transfer speed between system memory and GPU memory. It also provides a baseline for evaluating data ingestion pipelines, storage→host→GPU transfers, and other P2P paths.

NVIDIA specifies the following for the H200 SXM:

```text
PCIe Gen5: 128 GB/s
```

Sources:

- https://www.nvidia.com/en-us/data-center/h200/

The 128 GB/s figure on NVIDIA’s page is the total/bidirectional PCIe Gen5 bandwidth. For a one-way measurement, this report uses an approximate reference of:

```text
128 / 2 ≈ 64 GB/s per direction
```

GPU0 is closest to NUMA0, so the Host↔GPU benchmark binds both CPU execution and memory allocation to NUMA0.

---

## PCIe unidirectional – buffer 512 MiB

### Measurement Command

```bash
CUDA_VISIBLE_DEVICES=0 \
numactl --cpunodebind=0 --membind=0 \
./build/nvbandwidth \
  -d \
  -t host_to_device_memcpy_ce device_to_host_memcpy_ce \
  -b 512 \
  -i 10 \
  | tee /root/h200-benchmark/logs/pcie_512m.txt
```

### Results

```text
H2D: 55.42 GB/s
D2H: 55.26 GB/s
```

---

## PCIe unidirectional – buffer 4 GiB

### Measurement Command

```bash
CUDA_VISIBLE_DEVICES=0 \
numactl --cpunodebind=0 --membind=0 \
./build/nvbandwidth \
  -d \
  -t host_to_device_memcpy_ce device_to_host_memcpy_ce \
  -b 4096 \
  -i 10 \
  | tee /root/h200-benchmark/logs/pcie_4g.txt
```

### Results and Comparison

| Direction | Measured | NVIDIA-derived one-way reference | Ratio |
|---|---:|---:|---:|
| Host → GPU | **55.46 GB/s** | ~64 GB/s | **86.7%** |
| GPU → Host | **55.27 GB/s** | ~64 GB/s | **86.4%** |

The 512 MiB and 4 GiB results are nearly identical, indicating a stable Host↔GPU path as buffer size increases.

> Using the PCIe 5.0 x16 theoretical payload bandwidth of approximately 63 GB/s per direction, instead of halving NVIDIA’s published 128 GB/s figure, gives approximately 88% utilization. The main table retains NVIDIA’s 128 GB/s figure for a consistent reference source.

---

## PCIe bidirectional

### Why Measure This

Measure H2D and D2H while both directions are active to evaluate full-duplex capability and bandwidth degradation from internal contention.

### Measurement Command

```bash
CUDA_VISIBLE_DEVICES=0 \
numactl --cpunodebind=0 --membind=0 \
./build/nvbandwidth \
  -d \
  -t host_to_device_bidirectional_memcpy_ce \
     device_to_host_bidirectional_memcpy_ce \
  -b 4096 \
  -i 10 \
  | tee /root/h200-benchmark/logs/pcie_bidirectional_4g.txt
```

### Results

| Test | Measured | Compared with unidirectional |
|---|---:|---:|
| H2D with concurrent D2H | **38.64 GB/s** | ~30.3% decrease |
| D2H with concurrent H2D | **52.52 GB/s** | ~5.0% decrease |

Relative to the approximate reference of 64 GB/s per direction:

```text
H2D concurrent ≈ 60.4%
D2H concurrent ≈ 82.1%
```

There is clear asymmetry: D2H retains most of its original throughput, while H2D drops more sharply when both directions are active.

The two test cases run separately, and each reports only one direction while generating traffic in the other. Therefore, **38.64 + 52.52 GB/s must not be treated as a single aggregate measurement**.

---


## PCIe H2D/D2H on Each GPU, GPU0–GPU7

### Why Measure This

Measuring each GPU with the correct NUMA locality checks consistency of the Host↔GPU PCIe paths across the node and helps rule out unusually low performance on a PCIe/NUMA branch.

NVIDIA reference source:

- https://www.nvidia.com/en-us/data-center/h200/

### Measurement Command

```bash
NUMA=(0 0 1 1 2 2 3 3)

for g in {0..7}; do
  n=${NUMA[$g]}
  CUDA_VISIBLE_DEVICES=$g \
  numactl --cpunodebind=$n --membind=$n \
  $NVBW \
    -d \
    -t host_to_device_memcpy_ce device_to_host_memcpy_ce \
    -b 4096 \
    -i 10 \
    | tee /root/h200-benchmark/logs/per_gpu/gpu${g}_pcie.txt
done
```

### Results

| GPU | NUMA | H2D | D2H |
|---|---:|---:|---:|
| GPU0 | 0 | 55.45 GB/s | 55.27 GB/s |
| GPU1 | 0 | 55.44 GB/s | 55.27 GB/s |
| GPU2 | 1 | 55.46 GB/s | 55.27 GB/s |
| GPU3 | 1 | 55.45 GB/s | 55.27 GB/s |
| GPU4 | 2 | 55.44 GB/s | 55.27 GB/s |
| GPU5 | 2 | 55.45 GB/s | 55.27 GB/s |
| GPU6 | 3 | 55.46 GB/s | 55.27 GB/s |
| GPU7 | 3 | 55.47 GB/s | 55.27 GB/s |
| **Average** | — | **55.45 GB/s** | **55.27 GB/s** |

Results are nearly identical across all 8 GPUs, indicating highly consistent PCIe path performance with correct NUMA binding.

---

## PCIe H2D/D2H with 8 GPUs Running Concurrently

### Why Measure This

When all 8 GPUs exchange data with the host, the PCIe root complex, memory controllers, and host DRAM become shared resources. This test checks how the Host↔GPU paths scale across the node.

### Measurement Command

Use a 1 GiB buffer; the earlier 512 MiB and 4 GiB tests produced nearly identical single-GPU results.

```bash
NUMA=(0 0 1 1 2 2 3 3)

for g in {0..7}; do
  n=${NUMA[$g]}
  CUDA_VISIBLE_DEVICES=$g \
  numactl --cpunodebind=$n --membind=$n \
  $NVBW \
    -d \
    -t host_to_device_memcpy_ce device_to_host_memcpy_ce \
    -b 1024 \
    -i 10 \
    > /root/h200-benchmark/logs/8gpu/gpu${g}_pcie.txt 2>&1 &
done
wait
```

### Results

| GPU | H2D single | H2D 8-GPU | D2H single | D2H 8-GPU |
|---|---:|---:|---:|---:|
| GPU0 | 55.45 | 54.24 | 55.27 | 38.04 GB/s |
| GPU1 | 55.44 | 54.11 | 55.27 | 37.88 GB/s |
| GPU2 | 55.46 | 54.31 | 55.27 | 37.43 GB/s |
| GPU3 | 55.45 | 54.22 | 55.27 | 36.18 GB/s |
| GPU4 | 55.44 | 54.27 | 55.27 | 37.72 GB/s |
| GPU5 | 55.45 | 54.82 | 55.27 | 37.04 GB/s |
| GPU6 | 55.46 | 54.04 | 55.27 | 37.34 GB/s |
| GPU7 | 55.47 | 53.99 | 55.27 | 37.21 GB/s |
| **Average** | **55.45** | **54.25** | **55.27** | **37.36 GB/s** |

### Scaling Comparison

Host → Device:

```text
54.25 / 55.45 ≈ 97.8%
```

Aggregate H2D:

```text
8 GPU simultaneous ≈ 434.0 GB/s
```

Device → Host:

```text
37.36 / 55.27 ≈ 67.6%
```

Aggregate D2H:

```text
8 GPU simultaneous ≈ 298.84 GB/s
```

The resulting decreases are:

```text
H2D decrease ≈ 2.2%
D2H decrease ≈ 32.4%
```
