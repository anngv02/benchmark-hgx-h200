# NVIDIA HGX H200 Hardware Benchmarks

Hardware benchmark results for an **8 × NVIDIA H200 SXM 141 GB** system, covering GPU compute, HBM3e memory bandwidth, PCIe host transfers, and a separate GDS vs. CPU → GPU storage transfer comparison.

The report establishes a single-GPU baseline and checks performance across all eight GPUs. Detailed commands, kernel configurations, measurements, and charts are available in the documents below.

## System at a Glance

| Component | Configuration |
|---|---|
| GPUs | 8 × NVIDIA H200 SXM 141 GB, 700 W power limit per GPU |
| CPUs | 2 × Intel Xeon Platinum 8558; 192 logical CPUs |
| NUMA | 4 nodes |
| GPU interconnect | Fourth-generation NVLink and 4 third-generation NVSwitches |
| GPU-local devices | ConnectX-7 adapters and Samsung NVMe drives |
| Benchmark software | Driver 580.159.03, CUDA 13.0, CUTLASS, nvbandwidth |

See [system topology](docs/topology.md) for NVIDIA reference architecture images and the node’s device mapping, and [benchmark environment](docs/environment.md) for software versions and measurement settings.

## Key Results

| Measurement | Result |
|---|---|
| Single-GPU tuned BF16 | **702.19 TFLOPS** |
| Concurrent BF16 across 8 GPUs | **5.695 PFLOPS** aggregate; no observed throughput degradation |
| Single-GPU HBM read / write | **4.548 / 4.333 TB/s** |
| Single-GPU PCIe H2D / D2H | **55.46 / 55.27 GB/s** |
| Concurrent PCIe across 8 GPUs | **434.00 GB/s H2D**, **298.84 GB/s D2H** aggregate |
| GDS vs. CPU → GPU, 8 GPUs at 1M blocks | Approximately **42–43 GiB/s**; GDS uses fewer CPU cores |

Tensor Core comparisons use dense theoretical references derived from NVIDIA’s sparse specifications. The GDS figures are approximate readings from the supplied charts and use GiB/s; the PCIe measurements use GB/s. See the [full results summary](docs/results.md) and [GDS comparison](docs/gds.md) for context and limitations.

## Detailed Documentation

| Document | Contents |
|---|---|
| [System topology](docs/topology.md) | Reference architecture images, NVLink/NVSwitch connections, GPU–NIC–NVMe–NUMA mapping |
| [Benchmark environment](docs/environment.md) | Software versions, GEMM shape, warm-up settings, NUMA binding, reference specifications |
| [Compute benchmarks](docs/compute.md) | Kernel selection, FP32/TF32/BF16/FP16/FP8 results, concurrent BF16 measurements |
| [HBM3e benchmarks](docs/hbm.md) | Read, write, and copy bandwidth; buffer sizes; per-GPU validation |
| [PCIe benchmarks](docs/pcie.md) | Host↔GPU transfers, bidirectional tests, per-GPU results, concurrent scaling |
| [Results summary](docs/results.md) | Complete comparison tables and scaling interpretation |
| [GDS vs. CPU → GPU](docs/gds.md) | Eight throughput and CPU usage charts for 1, 2, 4, and 8 GPUs |

Images are stored in [`assets/architecture/`](assets/architecture/) and [`assets/benchmarks/`](assets/benchmarks/). NVIDIA image credits and download links are listed in the [architecture source notes](assets/architecture/SOURCES.md).
