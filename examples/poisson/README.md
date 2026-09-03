# Periodic Poisson examples

This directory contains two SCFD/FFTM manufactured-solution applications:

- `poisson_periodic_3d_autotuned.bin` demonstrates measured C++ autotuning and
  cache reuse for a distributed 3D transform.
- `poisson_periodic_4d_autotuned.bin` demonstrates measured 4D strategy, transport, and
  spectral-layout selection. Its Fourier-space operator supports both the
  public `yzwx` and native `xzwy` contracts.

Both sources are backend-neutral `.cpp` translation units. They use FFTM's
backend facade and SCFD device containers and kernels; CUDA and HIP names do
not appear in application code. Because the files contain device functors,
the Makefile compiles them explicitly as CUDA or HIP rather than using a
host-only C++ compiler.

Build it directly:

```bash
make -C examples/poisson
```

CUDA and MPI installations default to `/usr/local/cuda` and `/usr/local/mpi`.
They and the target GPU architecture are normal Make variables, so a remote
machine can override them without editing project files:

```bash
make -C examples/poisson \
  cuda_dir=/opt/cuda mpi_dir=/opt/openmpi \
  CUDA_ARCH='-gencode arch=compute_80,code=sm_80'
```

Build the ROCm-aware HIP binaries with:

```bash
make -C examples/poisson hip \
  ROCM_DIR=/opt/rocm \
  mpi_dir=$HOME/opt/rocm-mpi/openmpi \
  HIP_ARCH=gfx1102
```

For MPI installations that cannot consume device pointers, build the
host-staged variants instead:

```bash
make -C examples/poisson hip-nca \
  ROCM_DIR=/opt/rocm \
  mpi_dir=/opt/openmpi \
  HIP_ARCH=gfx1102
```

These targets produce `_hip.bin` and `_hip_nca.bin` binaries alongside the
stable CUDA binary names. The historical SCFD CUDA-aware macro appears only in
the build boundary; FFTM application code treats it as a backend-neutral MPI
capability.

It remains part of the aggregate examples build:

```bash
make -C examples poisson_periodic_3d_autotuned.bin
make -C examples poisson_periodic_4d_autotuned.bin
```

## Autotuned 3D solve

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
pairs. They can be changed through environment variables:

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
`FFTM_CPP_AUTOTUNE_CONTINUE_ON_CANDIDATE_ERROR=0` to make an unsupported or
failed candidate abort the sweep. By default, failed candidates are recorded
as invalid in the cache and measurement continues; a valid candidate is still
required before the cache is written. Set
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

For a small reader-facing check on a two-GPU CUDA machine with CUDA-aware MPI,
build and run the policy-cache, measured-cache, and 4D examples with:

```bash
CUDA_ARCH='-gencode arch=compute_80,code=sm_80' \
MPIEXEC=/opt/openmpi/bin/mpiexec \
bash examples/tests/run_reader_smoke.sh
```

The launcher validates the compatibility signature by default. This permits a
cache to be reused when Slurm assigns a different physical subset of equivalent
GPUs on the same hardware class. Set
`FFTM_CPP_TUNE_STRICT_DEVICE_IDENTITY=1` only when the allocation pins the same
GPU UUIDs and PCI bus IDs across every measurement and reuse step. Strict
identity rejection is covered separately by the hardware-signature unit test.

Each target requests one Slurm allocation unless it is already running inside
one. The result directory contains one short subdirectory per GPU count,
`status.csv`, the generated caches, and a compact selected-configuration log.

## Autotuned 4D solve

The 4D application solves

```text
-Laplacian(u) = f
u = sin(x) cos(2y) sin(3z) cos(w)
```

on the periodic domain `[0,2*pi)^4`. It stores all fields in SCFD tensors,
measures the supported 4D strategy/communication candidates, applies the
diagonal operator in the selected Fourier-space layout, and verifies the
relative L2 error.

The full positional form is `Nx Ny Nz Nw`, followed by the cache path,
repetition count, and application warmup. A single `N` remains a shorthand
for an `N x N x N x N` problem:

```bash
mpiexec -n 4 ./examples/build/poisson_periodic_4d_autotuned.bin \
  64 80 96 128 fftm_poisson_4d_64x80x96x128_4g.env 3 1
```

The example declares that its Fourier-space kernel accepts both implemented
physical contracts:

```cpp
autotune_options.accepted_spectral_layouts = {
    fftm::fftm_4d_spectral_layout::public_yzwx,
    fftm::fftm_4d_spectral_layout::native_xzwy
};
```

Consequently, complete forward/backward pairs for both layouts enter the
measured candidate set. The selected layout is serialized with every candidate
and returned in `selected.config`. Changing the accepted-layout set invalidates
the measured cache so that an unmeasured representation cannot win by stale
timing. Applications that require one fixed representation should leave
`accepted_spectral_layouts` empty and set `requested_spectral_layout`.

Override the example's two-layout set with a comma-separated list:

```bash
FFTM_CPP_AUTOTUNE_ACCEPTED_SPECTRAL_LAYOUTS_4D=native-xzwy \
mpiexec -n 4 ./examples/build/poisson_periodic_4d_autotuned.bin \
  64 64 64 64 fftm_poisson_4d_native_64_4g.env 3 1
```

The first execution measures and writes the cache; the second loads its
winner. `FFTM_CPP_AUTOTUNE_WARMUP`, `FFTM_CPP_AUTOTUNE_TIMES`, and the 4D
strategy/layout constraint variables use the same semantics as their 3D
counterparts.

For a local one-GPU correctness check with multiple wrapped MPI ranks:

```bash
FFTM_WRAP_PROCS_GPUS=1 \
mpiexec -n 2 ./examples/build/poisson_periodic_4d_autotuned.bin \
  16 16 16 16 fftm_poisson_4d_smoke.env 1 0
```

## Reader smoke suite

The source checkout provides a compact correctness suite for both
applications. It builds the public examples and runs each on one and two MPI
ranks wrapped over the locally visible GPU. Each 3D case runs twice and
requires cache creation followed by exact cache reuse:

```bash
make -C examples reader-smoke
```

Set `FFTM_READER_BUILD_DIR`, `CUDA_ARCH`, or `MPIEXEC` to override the local
build directory, CUDA architecture, or MPI launcher. For HIP, use:

```bash
FFTM_READER_BACKEND=hip \
FFTM_READER_DEVICE_AWARE_MPI=0 \
ROCM_DIR=/opt/rocm \
HIP_ARCH=gfx1102 \
mpi_dir=/opt/openmpi \
MPIEXEC=/opt/openmpi/bin/mpiexec \
bash examples/tests/run_reader_smoke.sh
```

Set `FFTM_READER_DEVICE_AWARE_MPI=1` only when Open MPI and its transport have
been built with ROCm pointer support. The smoke suite checks backend identity,
transport capability, cache creation/reuse, numerical accuracy, and both 4D
layout candidates.
