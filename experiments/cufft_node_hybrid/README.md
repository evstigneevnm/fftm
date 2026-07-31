# cuFFT Node-Hybrid Diagnostic

This project tests two independent assumptions behind a possible node-grouped
FFTM backend:

1. A single process can execute a large 3D real transform efficiently across
   the GPUs visible on one node using cuFFT Xt.
2. That process can exchange the resulting shuffled per-GPU buffers with a
   matching process on another node without losing the node's available
   InfiniBand bandwidth.

It is deliberately separate from FFTM production code. CUDA and cuFFT Xt types
are confined to `src/cuda_cufft_xt_backend.cu`. The benchmark sees only an
opaque FFT backend and device-segment views, and all MPI operations use the
SCFD communicator.

The same project also contains a standalone 4D candidate. It performs batched
multi-GPU YZW D2Z/Z2D transforms over X-owned batches, transposes those batches
through CUDA peer access into FFTM-compatible native `[x,z,w,y]` spectral
storage, and executes the remaining contiguous X transforms on each GPU.
CUDA, cuFFT, streams, descriptors, and peer kernels remain confined to
`src/cuda_cufft_xt_4d_backend.cu`.

## Scope

The `none` mode measures an actual cuFFT Xt D2Z/Z2D round trip. The `ring` mode
adds a spectrum exchange between corresponding ranks on adjacent nodes before
the inverse transform. The inverse therefore reconstructs the previous node's
input and validates against that input.

Repeated timing iterations normalize the inverse result outside the measured
interval. Numerical validation uses one fresh, untimed round trip after the
measurements, so cuFFT's unnormalized inverse does not accumulate across
attempts. Host setup and validation use the required in-place real padding
`2*(Z/2+1)` and copy the natural X-slab segments inside the CUDA backend.

The ring composition is a transport diagnostic, not a globally distributed
3D FFT. A node-local FFT of a slab does not replace the global transform along
the distributed dimension. We should implement a full hierarchical FFT only
if these component measurements show enough margin.

## Build Locally

```bash
make -C experiments/cufft_node_hybrid \
    CUDA_ARCH="-gencode arch=compute_75,code=sm_75"
```

cuFFT Xt requires at least two visible GPUs at runtime. A one-GPU machine can
compile the binary and run `--help`, but cannot execute the transform.

## Build The Test Image

```bash
FFTM_NODE_HYBRID_BASE_IMAGE=nvcr.io/nvidia/hpc-benchmarks:24.06 \
FFTM_NODE_HYBRID_SQSH_NAME=fftm_node_hybrid_a100 \
experiments/cufft_node_hybrid/build_image.sh
```

The builder creates a per-run temporary area under the repository's
`enroot_workdir` and removes it after the SQSH is complete. Set
`FFTM_NODE_HYBRID_KEEP_ENROOT_WORKDIR=1` only when diagnosing an Enroot build.

## Cluster Targets

- `smoke`: one small two-GPU correctness case.
- `local`: 1024 cubed on 2/4/8 GPUs and 2048 cubed on 8 GPUs.
- `multinode`: 2048 cubed on two 8-GPU nodes with UCX auto, one HCA pair, and
  all four HCA pairs.
- `selection`: the 8-GPU 2048 local case plus the three two-node transport
  variants.
- `all`: local and multinode matrices.

The launcher requests one allocation and continues after failed cases by
default:

```bash
FFTM_NODE_HYBRID_CONTAINER_IMAGE=/scratch/evstigneevnm/fftm/fftm_node_hybrid_a100.sqsh \
FFTM_NODE_HYBRID_DATA_DIR=/scratch/evstigneevnm/fftm/data_cufft_node_hybrid_$(date +%Y%m%d_%H%M%S) \
FFTM_NODE_HYBRID_TARGET=selection \
FFTM_NODE_HYBRID_SALLOC_EXTRA_ARGS='--exclude=cn13' \
experiments/cufft_node_hybrid/run_slurm_pyxis.sh
```

## Standalone 4D Candidate

The 4D binary does not include or link `fftm.hpp`. It measures the complete
forward/inverse algorithm:

1. Batched YZW D2Z across the selected GPUs.
2. X-slab to Z-slab peer transpose into native `xzwy`.
3. Per-GPU batched X Z2Z.
4. Inverse X Z2Z.
5. Z-slab to X-slab peer transpose.
6. Batched YZW Z2D.

The result CSV reports every stage, complete-pair wall time, plan time,
descriptor bytes, shared workspace bytes, and device-side numerical
validation. The 320-to-the-fourth candidate must beat 196 ms before production
integration is justified.

After rebuilding the diagnostic SQSH, run the complete selection matrix with:

```bash
FFTM_NODE_HYBRID_4D_CONTAINER_IMAGE=/scratch/evstigneevnm/fftm/fftm_node_hybrid_a100.sqsh \
FFTM_NODE_HYBRID_4D_DATA_DIR=/scratch/evstigneevnm/fftm/data_cufft_node_hybrid_4d_$(date +%Y%m%d_%H%M%S) \
FFTM_NODE_HYBRID_4D_TARGET=selection \
FFTM_NODE_HYBRID_4D_TIMES=20 \
FFTM_NODE_HYBRID_4D_WARMUP=5 \
FFTM_NODE_HYBRID_4D_SALLOC_EXTRA_ARGS='--exclude=cn13' \
experiments/cufft_node_hybrid/run_slurm_pyxis_4d.sh
```

Targets are `smoke` for 2-GPU 64-to-the-fourth validation, `selection` for
2-GPU 64, 8-GPU 160, and 8-GPU 320, and `production` for only the final
8-GPU 320 case. Failures are recorded and the matrix continues by default.

## Decision Rule

Compare the local 8-GPU cuFFT Xt round trip against the established FFTM
single-node 2048-cubed result. Then compare the ring's complete exchange time
across HCA modes.

- Continue toward a hierarchical FFT only if the projected complete transform
  improvement is at least 10%.
- Treat 5-10% as inconclusive unless memory or scaling also improves.
- Stop below 5%; the ownership, layout, and portability cost is not justified.
