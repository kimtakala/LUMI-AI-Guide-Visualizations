# The following are the specifications for the laptop and lumi partitions used:

---

## Laptop

### Architecture:

    x86_64
    CPU op-mode(s): 32-bit, 64-bit

### CPUs:

    Core(s) per socket: 4
    Thread(s) per core: 2
    Total: 8

### Model:

    11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz

### Memory:

    Total: 15Gi

### Caches (sum of all):

    L1d: 192 KiB (4 instances)
    L1i: 128 KiB (4 instances)
    L2: 5 MiB (4 instances)
    L3: 12 MiB (1 instance)

### GPU (unused):

    description: VGA compatible controller
    product: Intel Corporation TigerLake-LP GT2 [Iris Xe Graphics]
    prefetchable memory: 256M (64-bit)
    non-prefetchable memory: 16M (64-bit)

## LUMI-G:

### LUMI-G specifications per node:

### Architecture:

    x86_64
    CPU op-mode(s): 32-bit, 64-bit

### CPUs:

    Core(s) per socket: 64
    Thread(s) per core: 1 (SMT disabled)
    Socket(s): 1
    Total: 64

### Model:

    AMD EPYC 7A53 "Trento" CPU (Zen 3 cores, optimized for HPC)

### Memory:

    Total: 512 GiB DDR4
    Configuration: 4 NUMA nodes (NPS4)
    Memory per NUMA node: 128 GiB

### Caches (per CPU):

    L1d: 32 KiB per core
    L1i: 32 KiB per core
    L2: 512 KiB per core
    L3: 256 MiB total (32 MiB per 8-core group)

### GPUs:

    Count: 4 x AMD MI250X modules (8 GCDs total)
    Memory per MI250X: 128 GB HBM2e total (64 GB per GCD)
    Compute Units per GCD: 110 (out of 112 physical)
    Matrix Cores per CU: 4
    Peak Performance: 256 FP64 FLOP/cycle/CU

### GPU Memory Hierarchy:

    L1 cache per CU: 16 KiB
    L2 cache per GCD: 8 MB (32 slices, 128 B/clock/slice)
    Local Data Share (LDS) per CU: 64 KiB
    Vector General Purpose Registers per SIMD: 512 x 64-wide x 4-byte

### Interconnect:

    Intra-MI250X (GCD-to-GCD): up to 400 GB/s bidirectional
    Inter-MI250X: 100-200 GB/s bidirectional (single/double Infinity Fabric links)
    Network: Slingshot-11 (up to 25+25 GB/s peak bandwidth per MI250X)

### Grid/Block Limits:

    Grid dimensions: (2.1B, 2.1B, 2.1B) [2.147.483.647 each dimension]
    Block dimensions: (1024, 1024, 1024) with max 1024 threads per block
    Wavefront size: 64 threads (compared to 32 in NVIDIA warps)
