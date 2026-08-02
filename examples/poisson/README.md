# Autotuned 3D periodic Poisson example

This directory contains the minimal SCFD/FFTM manufactured-solution example
for a periodic three-dimensional Poisson problem.

Build it directly:

```bash
make -C examples/poisson
```

It remains part of the aggregate examples build:

```bash
make -C examples poisson_periodic_3d_autotuned.bin
```

Run it with one MPI rank per GPU. The positional arguments are `Nx Ny Nz`,
cache path, measured application iterations, and application warmup:

```bash
FFTM_WRAP_PROCS_GPUS=0 \
mpiexec -n 4 ./examples/build/poisson_periodic_3d_autotuned.bin \
  256 256 256 fftm_poisson_256_4g.env 3 1
```

The first run measures the C++ 3D candidate matrix and writes a schema-v2
hardware-validated cache. A second identical run loads the measured winner.
The candidate measurement defaults are two warmups and five timed transform
pairs. They can be changed without Python:

```bash
FFTM_CPP_AUTOTUNE_WARMUP=3 \
FFTM_CPP_AUTOTUNE_TIMES=10 \
mpiexec -n 4 ./examples/build/poisson_periodic_3d_autotuned.bin \
  256 256 256 fftm_poisson_256_4g.env 3 1
```

Set `FFTM_CPP_AUTOTUNE_MEASURE=0` to create/load only the known production
policy. A cache whose size, rank count, hardware, MPI, topology, or transport
signature does not match is rejected rather than silently reused.

An application can constrain selection without rerunning measurements. The
example maps the following environment variables to the equivalent C++ cache
constraints:

```bash
FFTM_CPP_AUTOTUNE_STRATEGY_3D=pencil-pencil \
FFTM_CPP_AUTOTUNE_MODE=p2p-waitany \
FFTM_CPP_AUTOTUNE_PENCIL_LAYOUT=opt0 \
mpiexec -n 8 ./examples/build/poisson_periodic_3d_autotuned.bin \
  2048 2048 2048 fftm_poisson_2048_8g.env 3 1
```

`FFTM_CPP_AUTOTUNE_GRID_3D`, `FFTM_CPP_AUTOTUNE_BACKEND_3D`, and
`FFTM_CPP_AUTOTUNE_STRICT_DEVICE_IDENTITY` provide the remaining expert and
cache-validation controls. Optional candidate expansion is available through
`FFTM_CPP_AUTOTUNE_INCLUDE_ALLTOALLV`,
`FFTM_CPP_AUTOTUNE_INCLUDE_HIGH_MEMORY_OPT0`, and
`FFTM_CPP_AUTOTUNE_MAX_PENCIL_GRIDS`. Set
`FFTM_CPP_AUTOTUNE_LOG_CANDIDATE_MEMORY=1` to report free device memory before
and after each measured candidate; it is disabled by default. Measured
candidates reuse SCFD-owned workspace and spectrum pools because CUDA-aware MPI
stacks can retain large registered allocations after a free. The real input
keeps normal SCFD ownership. The selected plan and Poisson spectrum reuse the
same retained pools. Each recovery
record separates this intentional pool growth from unexplained retained
memory. The allowed unexplained runtime residue is 1024 MiB and can be changed with
`FFTM_CPP_AUTOTUNE_MEMORY_RECOVERY_TOLERANCE_MIB`; set
`FFTM_CPP_AUTOTUNE_VERIFY_MEMORY_RECOVERY=0` only for diagnosis.

By default, measurement compares the validated production pencil candidate
against slab-pencil. This bounds device-memory use while retaining both the
best constrained pencil choice and the global-fastest alternative. Set
`FFTM_CPP_AUTOTUNE_PRODUCTION_CANDIDATES_ONLY=0` to request the broad strategy,
layout, and grid sweep; large problems may require substantially more memory.

The cluster validation launcher checks measured creation, exact cache reuse,
and constrained selection from the same cache. It has a short smoke target and
a production `2048^3` target for 6, 7, and 8 GPUs:

```bash
scripts/run_cpp_autotune_validation.sh smoke
scripts/run_cpp_autotune_validation.sh production
```

Each target requests one Slurm allocation unless it is already running inside
one. The result directory contains one short subdirectory per GPU count,
`status.csv`, the generated caches, and a compact selected-configuration log.
