# General NGPU capsule validation, 2026-09-09

This report covers the refreshed single-node Poisson images with `NGPU=N`
support. It supersedes the earlier two-rank capsule for new release downloads;
the original images and [their validation](VALIDATION.md) remain archived.
These are correctness/reproducibility checks, not performance measurements.

## Results

| Backend and host | Selected GPUs | Matrix | Maximum relative L2 error |
| --- | ---: | ---: | --- |
| CUDA, threadripper, V100-SXM2-32GB | 1 | 18/18 passed | `7.675053e-16` |
| CUDA, threadripper, V100-SXM2-32GB | 2 | 36/36 passed | `7.675053e-16` |
| HIP, amdcluster, RX 7600 XT gfx1102 | 1 | 18/18 passed | `7.502343e-16` |
| HIP, amdcluster, RX 7600 XT gfx1102 | 2 | 36/36 passed | `7.502343e-16` |

All 108 matrix invocations passed. Each backend also passed a fresh-container
preflight and two-GPU 4D cache reuse, for 112 successful GPU test invocations
in total. Both fresh solves reported `source=cache`, with cache bytes unchanged.
Cache SHA-256 values before and after fresh-container reuse were:

```text
CUDA 6292d21d8ea14deef0138b699c65740aae5d435a08f6b3d9b29bfd0844f9c946
HIP  76dc2b1791ab2c79ed99d559198a7d39688be3790dc25d31675a8bc1a1b6e6c1
```

The full matrices used 3D `32 x 40 x 48` and 4D `16 x 20 x 24 x 32`, both
device-aware and host-staged executables, measured cache creation, unchanged
cache reuse, expected wrong-size rejection, and both explicit 4D spectral
layouts. The launcher supplied the rank lists; tests did not pass `--ranks`.
Requests, GPU mappings, cache decisions, errors, and MPI environments are
retained in the evidence. Matrices use a 180-second timeout per MPI invocation.

Both hosts rejected `NGPU=3` with exit code 2 before starting Docker because only
two physical GPUs were available. Both images, run locally without GPU access,
rejected the default shape at 32 ranks with exit code 2 before launching MPI.
The host-side checks passed: 25 capsule tests, 16 build-configuration tests,
shell syntax checks, and the backend abstraction-boundary check. Fixtures cover
3-, 4-, and 8-GPU launch construction, an above-two-rank matrix, custom shapes,
half-spectrum bounds, the input-size cap, missing devices, and failure handling.

More than two physical GPUs were not available for this validation. Larger GPU
counts are supported by the launcher/runner but are not hardware-verified here.
Both physical hosts have two GPUs: one-GPU tests select one compute GPU, and
discovery on a physically single-GPU HIP machine is covered by fixtures only.
No results establish multinode behavior or performance on these machines.

## Images and provenance

| Backend | Image tag | Compressed bytes | Release parts |
| --- | --- | ---: | ---: |
| CUDA | `fftm/poisson:cuda-portable-ngpu-20260909` | 415076566 | 1 |
| HIP | `fftm/poisson:hip-portable-ngpu-20260909` | 557605170 | 1 |

```text
CUDA image ID:
sha256:3ee4345b0469b14aac814072a6e90d32ebe285902554cc2a8f1f567e913d6103
CUDA archive SHA-256:
65e06a2bdf43159d635a2f9a6dc693f3b7cfb0460f346160eb99cb8bc9a725d2

HIP image ID:
sha256:5a4c6a8dea5d5ddecdd831b9923970b2897c12df2f10e200f528d78bf7460025
HIP archive SHA-256:
339b82125b95c65ef987f8b340dc593caf2397dafbb9146456c5d3f152d0b0e2

FFTM base commit: 15699db9661fac777651f3c4e7de02b7517e2b93
SCFD commit:      8e4a5cb0e04adf1c1ee81d91851c4d91c49904ac
Source manifest SHA-256 (both images, 591 files):
57d4ef982f937e32bf957d3c72947a30518393b365d06906347ee0e1346f5165
```

The build context includes the uncommitted NGPU capsule changes and records
them explicitly. These are not clean builds of the base commit alone. The
FFTM library, SCFD, Poisson C++ sources and cluster Dockerfile were not modified.
Only capsule orchestration, tests and documentation changed. This report was
added after building and is distributed with the tools/evidence, not included
in the image's source manifest.

The build used two jobs per image, pinned CUDA 12.9.1 / ROCm 6.4.1 base images,
Open MPI 5.0.7 and UCX 1.18.1. Runtime architectures remain sm70/75/80/86 for
CUDA and gfx1102 for HIP. Archives were exported locally, transferred, verified
on the remote hosts, and loaded into rootless Docker. Loaded image IDs and all
matrix `image-inspect.json` records match the manifests above.

## Transport and safety

CUDA device-aware runs use automatic UCX transport selection. HIP device-aware
runs explicitly use `self,sm,tcp,rocm_copy`: MPI accepts device pointers, but
UCX can stage through host memory. This does not validate direct ROCm IPC.
Host-staged binaries use the separate FFTM host-staging path and OB1 MPI.

CUDA device selection uses host `nvidia-smi` inventory and Docker CDI. HIP
discovers AMD render nodes and selects compute devices with
`ROCR_VISIBLE_DEVICES`. All needed render nodes remain exposed for ROCr
initialization even when one compute GPU is selected.

GPUs were idle with ample free memory before testing. No drivers, permissions,
global Docker contexts or services were modified. Threadripper's system Docker
service remained active with PID 1694, and all validation containers exited.

The capsule checks a sufficient partition bound and a default 64 MiB global
real-input cap before running FFTs. The bound includes the R2C half-spectrum;
it does not duplicate production grid selection. The size cap is not a total
workspace estimate. Larger shapes require an explicit cap change and a device
memory check; unsupported configurations must not be reported as passed.

## Reproduce

Follow [README.md](README.md) for rootless Docker prerequisites, archive loading,
GPU selection, shape overrides, and the release one-liner. On each appropriate
GPU host, with a loaded image and a fresh output directory, run:

```bash
NGPU=1 bash Docker_config/poisson/run.sh cuda \
    fftm/poisson:cuda-portable-ngpu-20260909 "$PWD/cuda_one" verify --timeout 180
NGPU=2 bash Docker_config/poisson/run.sh cuda \
    fftm/poisson:cuda-portable-ngpu-20260909 "$PWD/cuda_two" verify --timeout 180

NGPU=2 bash Docker_config/poisson/run.sh cuda \
    fftm/poisson:cuda-portable-ngpu-20260909 "$PWD/cuda_two" poisson4d \
    --transport device-aware --sizes 16 20 24 32 \
    --cache /data/run/d4-r2-device-aware.env --output /data/fresh-reuse4 --timeout 180
```

For HIP, replace the backend, image tag and output directory names. Set
`CAPSULE_DOCKER` to the user-local Docker executable when it is not on PATH.
The fresh solve defaults to NGPU ranks and reuses the preceding matrix cache.

The local delivery is `build/poisson_capsule_general_ngpu_20260909/`.
Its `validation/` directory contains all four matrices, fresh reuse cases,
launcher stdout, negative guards, host test logs and both image build records.
These files are also in `poisson-capsule-evidence.tar.gz`. All nine draft release
assets must be replaced together: both image parts, four manifest files, tools,
evidence, and the auxiliary checksum. Nothing was uploaded or published, and
no Git commit or tag was made during this work.
