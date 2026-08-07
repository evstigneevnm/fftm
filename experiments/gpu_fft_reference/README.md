# Same-System GPU-FFT Reference Diagnostic

This isolated experiment tests whether FFTM's multi-node throughput loss is
specific to FFTM or is also visible in the published slab-based GPU-FFT
implementation on the same cluster. It does not include or link FFTM.

The image fetches the BSD-3-Clause upstream repository at the pinned commit
`ff4d84bbc6e0ecbcd52246c9d7e83198dadb2426`. Upstream source is compiled
without edits. The local benchmark driver adds only:

- node-local rank to CUDA-device binding;
- deterministic device-side initialization and relative-L2 validation;
- repeated double-precision R2C/normalize/C2R pair timing;
- maximum-rank wall-time aggregation and compact CSV output;
- FFTM's independently validated GPU-to-HCA affinity launcher.

The upstream project is [Manthan-Verma/GPU_FFT](https://github.com/Manthan-Verma/GPU_FFT).
The corresponding paper is [Verma et al., *Scaling real-to-complex FFTs on
multiple GPUs*, SN Computer Science (2023)](https://doi.org/10.1007/s42979-023-02109-0).

## Matrix

The published-size comparison uses `2048^3` on 8, 16, and 32 GPUs. GPU-FFT's
equal slab decomposition requires the size to divide the MPI rank count, so
`2048^3` cannot run on 24 ranks. A second `1920^3` series covers exactly 8,
16, 24, and 32 GPUs, providing the requested 1/2/3/4-node strong-scaling
diagnostic.

Each case defaults to five warmups and 30 measured pairs. Timing includes the
normalization needed between the forward and inverse transforms. Effective
throughput is `10*N*log2(N)/time`, where `N` is the total number of points.

## Build

Build the independent image locally from the repository root:

```bash
GPUFFT_REF_BASE_IMAGE=nvcr.io/nvidia/hpc-benchmarks:24.06 \
GPUFFT_REF_SQSH_NAME=fftm_gpu_fft_reference_a100 \
experiments/gpu_fft_reference/build_image.sh
```

The build uses `enroot_workdir` by default and removes its temporary directory
after producing the SQSH. Copy `fftm_gpu_fft_reference_a100.sqsh` and its
`.sha256` file to the cluster.

## Cluster Runs

The most queue-friendly approach launches each allocation independently in the
foreground. For example, the two-node target is:

```bash
GPUFFT_REF_CONTAINER_IMAGE=/scratch/evstigneevnm/fftm/fftm_gpu_fft_reference_a100.sqsh \
GPUFFT_REF_TIMES=30 \
GPUFFT_REF_WARMUP=5 \
GPUFFT_REF_AFFINITY_MODE=hca \
GPUFFT_REF_SALLOC_EXTRA_ARGS='--exclude=cn13' \
GPUFFT_REF_STOP_ON_FAILURE=0 \
experiments/gpu_fft_reference/run_slurm_pyxis.sh node2
```

Use `node1`, `node2`, `node3`, and `node4` as positional targets for
independent allocations. Result directories are named automatically.
`published` runs only `2048^3` on 1/2/4 nodes in one four-node allocation;
`full` runs all seven cases in one four-node allocation. `smoke` runs a small
8-GPU numerical preflight.

Every production case uses one MPI rank per GPU and the same HCA pinning used
for FFTM paper runs. The output includes per-case logs, per-iteration CSVs,
summary CSVs, a status table, image/source provenance, and an automatically
generated `analysis/` comparison against the paper's reported 2048-cubed
throughputs.

## Decision

- If upstream GPU-FFT shows a comparable 8-to-16 GPU discontinuity, the common
  cluster transport/topology path is the dominant explanation.
- If GPU-FFT approaches its published 16/32-GPU rates while FFTM remains lower,
  FFTM's decomposition or schedule remains responsible.
- Compare both absolute throughput and the normalized 8-to-16 and 8-to-32
  speedups; software and CUDA versions can shift the absolute rates.
