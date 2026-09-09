# Poisson capsule validation, 2026-09-08

These are single-node correctness and reproducibility checks, not performance
measurements. Both images were built locally, exported, transferred over SSH,
checksum-verified, loaded into separate rootless Docker daemons, and tested on
the target GPU machines. Nothing was uploaded to GitHub or a registry.

## Results

| Backend | SSH host | GPUs | Matrix | Fresh-container reuse | Maximum relative L2 error in matrix |
| --- | --- | --- | --- | --- | --- |
| CUDA | `threadripper` | 2 x Tesla V100-SXM2-32GB | 36/36 passed | 2/2 passed | `7.675053e-16` |
| HIP | `amdcluster` (`amdtest`) | 2 x Radeon RX 7600 XT, gfx1102, 16 GiB | 36/36 passed | 2/2 passed | `7.502343e-16` |

The CUDA host uses NVIDIA driver 580.126.20. The HIP capsule uses ROCm 6.4.1,
matching the tested host installation. Both remote Docker clients/daemons were
29.8.0; both reported `name=rootless` in their security options. The local GPU
was not used: it did not meet the idle/free-memory readiness criteria.

Each 36-case matrix covers one and two MPI ranks, one rank per GPU, and both
device-aware and FFTM-host-staged executables:

- Four GPU-mapping and MPI-buffer preflights. Device-aware preflight includes
  both send/receive and GPU-buffer `MPI_Alltoallv` self-copy.
- Sixteen measured cache creations, including explicit tests of each 4D layout.
- Eight cache reuses, reporting `source=cache` without modifying the cache.
- Eight deliberately wrong-size cache requests, rejected with the expected
  cache-specific error and without modifying the cache.

The 3D shape is `32 x 40 x 48`; the 4D shape is `16 x 20 x 24 x 32`.
Both `public-yzwx` and `native-xzwy` are tested explicitly, in addition to
measured selection from the accepted pair. All successful solves have finite
relative L2 error below `1e-10`.

After each complete matrix container exited, a fresh container reused its
two-GPU, device-aware, two-layout 4D cache. The extra preflight and solve passed;
the solve reported `source=cache`. Cache SHA-256 values before and after were:

```text
CUDA f4585824dc20b9b63cdbb98fa4834ee3a16a3c4f7d7af8410c5dc09f218345ee
HIP  bb62bac65cef8470a50d201c4beb4e1cf32b09f482ee449f04e890c89b4b6861
```

Host-side checks also passed: 16 build-configuration tests, 8 capsule tests,
the backend abstraction-boundary check, shell syntax checks, and
`git diff --check`. The capsule unit tests deliberately exercise nonzero exit
and timeout paths; their printed failure messages are expected test inputs.

## Transport qualification

CUDA device-aware runs use the UCX PML with automatic transport selection.
HIP device-aware runs explicitly use `UCX_TLS=self,sm,tcp,rocm_copy`: MPI accepts
GPU pointers, while UCX is permitted to stage through host memory. This does
not validate direct GPU-to-GPU ROCm IPC or GPUDirect/RDMA performance.

Both host-staged variants are separate binaries, built without the SCFD
device-aware MPI macro. They use the `ob1` PML and `self,sm,tcp` BTLs.
The per-case command JSON files record these effective settings.

Two packaging/deployment problems were found and addressed before the final
passing run:

1. Open MPI 5.0.7's automatic CUDA library discovery disabled its CUDA
   accelerator component when both toolkit `compat` and `stubs` directories
   existed. UCX send/receive alone did not expose this. The builder now supplies
   the explicit stub-library directory, requires accelerator support at
   configure time, and preflight tests collective GPU self-copy. Driver stubs
   are not included in the runtime image.
2. Automatic ROCm IPC selection failed peer access on the two-Radeon rootless
   deployment. The bounded preflight failed and its dependent cases were
   skipped, with the overall run marked failed. The final HIP image explicitly
   selects the tested copy transport. `CAPSULE_UCX_TLS=all` remains an opt-in
   IPC diagnostic; no silent fallback is performed.

ROCr also needed both render nodes exposed on this host. For one selected
compute GPU, keep the render-node passthrough and use
`CAPSULE_HIP_VISIBLE_DEVICES=0`, as documented in the reader instructions.
No host drivers, GPU settings, ACLs, daemon configuration, or services were
changed during verification. The existing system Docker service on
`threadripper` remained active with PID 1694. All final test containers exited.

## Exact artifacts

The local delivery directory is `build/poisson_capsule_ready_20260908/`.
Only the following image archives are designated by this report; intermediate
images and failed-run directories are not release candidates.

| Backend | Image tag | Compressed archive bytes | Approximate MiB |
| --- | --- | ---: | ---: |
| CUDA | `fftm/poisson:cuda-portable-20260908` | 415108388 | 396 |
| HIP | `fftm/poisson:hip-portable-20260908` | 557599368 | 532 |

Both fit below 2 GiB. Each backend directory contains the full `.tar.gz`, one
identical `.tar.gz.part000`, a manifest JSON and its checksum. For publication,
use the part plus the two manifest files; do not upload both copies.

```text
CUDA image ID:
sha256:f0a04656a82afcd18edd9fd29659404839c32d409113831766dcdecaa42d0ebb
CUDA compressed archive SHA-256:
a06643bb1f475576136df5bc3c3e1d484062c3d4d9de62b12cb323868b4d6eb6

HIP image ID:
sha256:6a0b84e77ef3d80096f9839738980db28007b050983a5ca987a9a6eb65c45cf5
HIP compressed archive SHA-256:
4e86c9b55b729e16055260c8c6476215c3fbac9c733391cc760c6784aa99c163
```

Both images contain the same 589-file source snapshot:

```text
FFTM base Git commit: 94ceee6e4effe6b5064a34958f6b40b74ef9d533
SCFD Git commit:      8e4a5cb0e04adf1c1ee81d91851c4d91c49904ac
Source manifest SHA-256:
242f621a57e1afd6f360763196080ce3ffc9e754bf29e9c430109b821bd0ef26
```

The snapshot includes the uncommitted capsule/build changes. The embedded
provenance explicitly records this; these are not claimed to be clean builds
of the base Git commit alone. This report was added after image verification.
Preserve the tested archives and manifests when associating them with a release.

## Reproduce and inspect

Follow [README.md](README.md) for build prerequisites, archive loading, device
access, and detailed commands. After loading, run the relevant command on its
GPU host with a new output directory:

```bash
NGPU=2 bash Docker_config/poisson/run.sh cuda fftm/poisson:cuda-portable-20260908 \
    "$PWD/build/capsule_cuda_verify" verify --ranks 1,2 --timeout 180
NGPU=2 bash Docker_config/poisson/run.sh hip fftm/poisson:hip-portable-20260908 \
    "$PWD/build/capsule_hip_verify" verify --ranks 1,2 --timeout 180
```

Set `CAPSULE_DOCKER` to the user-local client path when it is not on `PATH`.
The launcher always selects the requested context explicitly.

Collected final evidence is under `cuda_results/` and `hip_results/` in the
local delivery directory. Each contains `image-inspect.json`, the complete
`run/` matrix, and `fresh-reuse4/`. Logs, exact commands and environments,
status records, summaries and caches are retained. Earlier CUDA packaging and
HIP IPC failures remain in separate local `build/poisson_capsule_*` directories.

The FFTM library source and original cluster Dockerfile were unchanged. The
only existing build-path changes add explicitly named CUDA host-staged Poisson
targets and support them in the reader smoke launcher. These tests do not
establish performance, multi-node operation, or support for untested GPU
architectures. Review vendor redistribution terms before publishing the images.

## NGPU launcher follow-up, 2026-09-09

The host launcher now defaults to `NGPU=1`; `NGPU=2` retains the complete
one-/two-rank matrix. The two image IDs and all binaries listed above are
unchanged. These checks used fresh output directories on the same idle machines:

| Backend | NGPU | Passed invocations | Maximum relative L2 error |
| --- | ---: | ---: | --- |
| CUDA, threadripper | 1 | 18/18 | `7.675053e-16` |
| CUDA, threadripper | 2 | 36/36 | `7.675053e-16` |
| HIP, amdcluster | 1 | 18/18 | `7.502343e-16` |
| HIP, amdcluster | 2 | 36/36 | `7.502343e-16` |

For each row the command was:

```bash
NGPU=1 CAPSULE_DOCKER="$HOME/.local/share/docker-rootless/bin/docker" \
    bash Docker_config/poisson/run.sh cuda fftm/poisson:cuda-portable-20260908 \
    "$PWD/results-cuda-one" verify --timeout 180
```

The other rows changed `NGPU`, backend/image, and output directory. No explicit
`--ranks` or device override was passed: the host launcher supplied those.
Both transports, 3D/4D solves, cache creation/reuse, expected cache rejections,
and both 4D spectral layouts passed. CUDA one-GPU preflight reported one visible
device. HIP automatically discovered both AMD render nodes while selecting
only compute GPU 0 for `NGPU=1`; two-rank preflights confirmed distinct devices.

Both physical test machines have two GPUs. A physically single-GPU HIP host was
not available: discovery of exactly one render node and omission of a nonexistent
second node were separately checked with host unit-test fixtures.
All 17 capsule tests, 16 build-configuration tests, abstraction-boundary checks,
launcher shell syntax, 15 README shell-block syntax checks, and
`git diff --check` passed. No driver, device permissions, services or global
Docker contexts changed; all validation containers exited.

Validated host-file SHA-256 values:

```text
5b2479a577aa31488805d39c4985e688f45bb0565677950eda4aaf3d4b41c186  run.sh
8e69e0a42e23a8183559d99be55cb8f9f9690ffc61eeb992a62bc0877307af58  host_devices.py
```

The local follow-up directory is `build/poisson_capsule_ngpu_20260909/`.
Its `validation/{cuda_one,cuda_two,hip_one,hip_two}/` directories retain all new
logs, summaries, commands and caches. Updated `poisson-capsule-tools.tar.gz`
and `auxiliary-assets.sha256` must replace the corresponding draft release
assets before advertising `NGPU`. The evidence archive remains the original
2026-09-08 record; image parts and manifests also remain unchanged. The updated
tools contain this follow-up report. No GitHub assets or tags were changed by
the verification process.
