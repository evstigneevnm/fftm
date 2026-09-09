# Multi-architecture capsule validation, 2026-09-09

This report covers the `portable-multiarch-20260909` Poisson images. These are
correctness and reproducibility checks, not paper performance measurements.
The earlier [NGPU validation](VALIDATION_NGPU.md) describes different images and
must not be substituted for the hardware coverage recorded here.

## Compiled targets

```makefile
CUDA_ARCH_LIST = 60 61 70 75 80 86 89 90 100 103 120
HIP_ARCH_LIST = gfx906 gfx908 gfx90a gfx942 gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201
```

CUDA contains all eleven SASS targets and a `compute_60` PTX fallback in the
four Poisson binaries and the probe. HIP contains all twelve code-object targets
in the four Poisson binaries and the rebuilt active rocFFT library. The HIP
probe contains no device kernels: it calls the vendor runtime's allocation and
copy APIs. The architecture audit records this runtime-only exception explicitly.

The images retain the inspection listings and `metadata/architectures.json`.
The profiles supply the flags to both solver and probe builds. These checks do
not establish vendor support, driver compatibility, or runtime correctness on
all listed architectures. Both images target x86-64 CPUs, not ARM64.

The stock ROCm 6.4.1 rocFFT library omits embedded code objects for
gfx906/gfx1031/gfx1032. The HIP capsule therefore rebuilds rocFFT 1.0.32 from the
pinned ROCm 6.4.1 source commit
`058ba87fdcfdae334dbc8dbe048955b248e9328a`, using all twelve configured targets.
Runtime compilation is enabled by default and the shipped kernel cache is
retained. Missing cache entries may increase first-plan latency. This does not
modify FFTM, hipFFT's public interface, or the host ROCm installation.
The pinned source archives, upstream license, CMake cache and install manifest
are included. The rocFFT dependency has its own cached Docker build layer.

## Results

| Backend and host | Selected GPUs | Main matrix | Maximum relative L2 error |
| --- | ---: | ---: | --- |
| CUDA, threadripper, V100-SXM2-32GB | 1 | 18/18 passed | `7.675053e-16` |
| CUDA, threadripper, V100-SXM2-32GB | 2 | 36/36 passed | `7.675053e-16` |
| HIP, amdcluster, RX 7600 XT gfx1102 | 1 | 18/18 passed | `7.502343e-16` |

The matrices use 3D `32 x 40 x 48` and 4D `16 x 20 x 24 x 32`, both device-aware
and host-staged MPI, measured cache creation, cache reuse, expected wrong-size
rejection, and both explicit 4D spectral layouts. Every MPI invocation has a
180-second timeout. An expected nonzero wrong-size rejection counts as a passed
check, not a successful solve.

Additional checks:

- CUDA: fresh-container two-GPU 4D preflight and cache reuse, both passed.
- HIP: fresh-container one-GPU 4D preflight and cache reuse, both passed.
- CUDA: forced PTX JIT (`CUDA_FORCE_PTX_JIT=1`) for one-GPU host-staged 3D and
  4D solves, including both preflights, all four checks passed.

Together these are **80 passed GPU checks**. Both fresh solves reported
`source=cache`, with unchanged cache bytes. Before/after cache SHA-256 values:

```text
CUDA 54806bfef03d123f6f3182927d64d52170f90a7bc0cd0e587cf06ea99e15e6d1
HIP  fd8f971525f9b9d1021bd9fcdd9d4c39eace184f440df0bd9890b1d434f91fba
```

Host validation passed: 32 capsule tests, 16 build-configuration tests, shell
syntax checks, the backend abstraction-boundary check, and a GPU-free integration
check of the ELF inspector on real previous-release HIP binaries. Fixtures cover
missing targets, runtime-only probe handling, rocFFT configuration validation,
profile overrides, backend-specific dependency selection, and NGPU behavior.

### Limits and safety

The second AMD GPU repeatedly reported 99-100% utilization, despite no KFD
compute processes being listed. It was not used. **Two-GPU HIP validation of
these rebuilt images is deferred**, not passed. GPU 0 was idle with ample free
memory and was selected using `ROCR_VISIBLE_DEVICES=0`. Both AMD render nodes
remain exposed for ROCr enumeration; only the selected compute GPU is used.

Other GPU architectures and counts above two were not hardware-tested. The local
workstation GPU was busy with limited free memory and was not used for GPU tests.
No results establish multinode behavior or performance.

CUDA device-aware runs use UCX automatic transport selection. HIP device-aware
runs use `self,sm,tcp,rocm_copy`; accepting device pointers does not prove direct
ROCm IPC, and UCX may stage through host memory. The host-staged executables use
FFTM's separate staging path and OB1 MPI.

All remote tests used the existing rootless daemons in dedicated directories.
No drivers, permissions, global contexts, services or unrelated workloads were
modified. Threadripper's system Docker service remained active, and all test
containers exited.

## Images and provenance

| Backend | Image tag | Compressed bytes | Release parts | Increase over NGPU image |
| --- | --- | ---: | ---: | ---: |
| CUDA | `fftm/poisson:cuda-portable-multiarch-20260909` | 417330870 | 1 | 2.15 MiB |
| HIP | `fftm/poisson:hip-portable-multiarch-20260909` | 567795150 | 1 | 9.72 MiB |

These are approximately 398.00 MiB and 541.49 MiB; each part is below 2 GiB.

```text
CUDA image ID:
sha256:989cb95cf8375f212c742a7d47f6874677e45f01a285574280d7f877ec82bc05
CUDA archive SHA-256:
94bd70306f18bcf1ae40dcb5d4fa4d0d8a2c1b28757fc4399aaca9030e7d981b
CUDA source manifest SHA-256:
8045a9f4b758d0ee1a9aeb49f6ff25d3f05b9b1dd8cff7a8259e8712ab4882dd

HIP image ID:
sha256:890497e5f3323e1426138939cc5d2c423dd3dbf767536e1f0065e9703fdfcf8f
HIP archive SHA-256:
564553ff999d60ffe666bb70ab6b4e18207eed5afd7f148ac96975fc523545a5
HIP source manifest SHA-256:
24257e8bde1cd091d8d4285f676b83934e44c528acfb4f49d6cdb67157fa5a8b

FFTM base commit: dd1222573d8c690ee5a3573d1ee09bcccf49ee20
SCFD commit:      8e4a5cb0e04adf1c1ee81d91851c4d91c49904ac
```

Both contexts contain 595 source files and record the uncommitted capsule
changes. They are not clean builds of the base commit alone. The contexts differ
in HIP build/audit files and documentation corrected after the CUDA build.
The 570 library, SCFD, Poisson example and shared build-configuration files are
identical in both images and match the committed baseline. The cluster Dockerfile
and FFT execution code were not changed. This report and final documentation
were added after building and are distributed with tools/evidence.

The toolchains remain CUDA 12.9.1, ROCm 6.4.1, Open MPI 5.0.7 and UCX 1.18.1.
Images were exported locally, transferred, checksum-verified and loaded remotely.
The remote image-inspection records match the archive manifests.

## Reproduce and deliver

Follow [README.md](README.md) for prerequisites and the release one-liner. With
fresh output directories and the corresponding image loaded:

```bash
NGPU=1 bash Docker_config/poisson/run.sh cuda \
    fftm/poisson:cuda-portable-multiarch-20260909 "$PWD/cuda_one" verify --timeout 180
NGPU=2 bash Docker_config/poisson/run.sh cuda \
    fftm/poisson:cuda-portable-multiarch-20260909 "$PWD/cuda_two" verify --timeout 180
NGPU=1 bash Docker_config/poisson/run.sh hip \
    fftm/poisson:hip-portable-multiarch-20260909 "$PWD/hip_one" verify --timeout 180
```

The local delivery directory is `build/poisson_capsule_multiarch_20260909/`.
Its `RELEASE_ASSETS.md` lists the nine replacement draft assets. The tools archive
contains this report and the current capsule tools. The evidence archive retains
build records, metadata, command lines, numerical results, cache files, availability
checks and the validation drivers. Earlier unsuccessful build logs remain in
their separate local build directories, not in the successful result set.
No Git commit, tag, upload or release publication was performed.
