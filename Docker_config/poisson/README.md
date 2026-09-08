# CUDA/HIP Poisson reproducibility capsule

This single-node correctness capsule uses FFTM's existing C++ Poisson examples.
It is independent of `Docker_config/Dockerfile`, the cluster benchmark image.
Neither the solvers nor FFTM's execution path are replaced by container code.

| Image | Local FFT implementation | Compiled targets | MPI |
| --- | --- | --- | --- |
| CUDA | CUDA/cuFFT 12.9.1 | sm70, sm75, sm80, sm86 | Open MPI 5.0.7, UCX 1.18.1 with CUDA |
| HIP | ROCm 6.4.1, hipFFT 1.0.18, rocFFT 1.0.32 | gfx1102 | Open MPI 5.0.7, UCX 1.18.1 with ROCm |

Both images contain 3D/4D executables for device-aware and host-staged MPI, a
SCFD-backed GPU/MPI buffer probe, source snapshots, dependency source archives
and licenses, build profiles, package lists, binary checksums, and provenance.
Runtime images exclude compilers and unrelated GPU math libraries; rocFFT's
kernel cache and runtime compilation dependencies remain included.

The tags below are **local build examples**, not existing public registry/release
uploads. Publication is separate. Publishers must supply an exact image digest
or tested archive manifest, not just a mutable tag.

The [2026-09-08 validation report](VALIDATION.md) records the tested image IDs,
archive checksums, remote results, and transport limitations.

## Requirements and safety

- Linux x86-64 and Docker. Host Python 3.8+ is needed for build/archive helpers;
  Git, curl and network access are needed for building. Execution uses the image's
  Python; no host scientific Python packages are required.
- One or two compatible GPUs with at least 2 GiB free per selected GPU. Check
  current utilization and postpone testing when GPUs are busy.
- CUDA: a working NVIDIA driver and NVIDIA Container Toolkit with readable CDI
  specifications. The intended V100 validation host uses driver 580.126.20.
  A container cannot install or replace the host driver. Verify driver/toolkit
  compatibility when using a different machine.
- HIP: a working ROCm-compatible AMD kernel driver, `/dev/kfd`, and the selected
  `/dev/dri/renderD*` devices. The default build targets RX 7600 XT (`gfx1102`).
  Other architectures require changing `config_hip.inc` and the matching probe
  compilation in `build_examples.sh`, then rebuilding and validating.
- Reserve about 30 GiB for a runtime load plus archives, and at least 60 GiB for
  building both backends with layer caches. Build one backend at a time on a small
  disk; exports record actual sizes.

Use a rootless daemon explicitly. The scripts do not change the default Docker
context, configure daemons, restart services, modify GPU permissions, or use sudo.
An existing system daemon and its workloads remain independent. Do not add
`--privileged` or mount host GPU toolkits/MPI libraries into the capsule.

```bash
# For a user-local client, if needed:
export PATH="$HOME/.local/share/docker-rootless/bin:$PATH"
docker --context rootless info
docker --context rootless info --format '{{json .SecurityOptions}}'
# SecurityOptions must contain name=rootless for a rootless installation.
```

CUDA readiness, without changing configuration:

```bash
nvidia-smi
nvidia-ctk cdi list
```

The launcher uses native Docker CDI (`--device nvidia.com/gpu=0`), not a global
NVIDIA runtime `no-cgroups` modification. The existing CDI specification must
match the driver. See [NVIDIA's CDI documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).

HIP readiness:

```bash
ls -l /dev/kfd /dev/dri/renderD128 /dev/dri/renderD129
getfacl /dev/kfd /dev/dri/renderD128 /dev/dri/renderD129
```

Host supplementary groups alone may not grant access inside a rootless user
namespace. An administrator may need to grant device ACLs to the host user.
Direct device ACLs can disappear on reboot/device recreation. This capsule does
not modify ACLs or udev rules. See [ROCm's container documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html).

## Build from a release checkout

Use the publisher's release tag/full commit, not a moving branch, and initialize
SCFD. A release checkout should have an empty `git status --short`.

```bash
git submodule update --init --recursive
git status --short
git submodule status

python3 Docker_config/poisson/build.py cuda \
    --tag fftm/poisson:cuda-local --context rootless --jobs 2 \
    --output build/poisson_cuda_build

python3 Docker_config/poisson/build.py hip \
    --tag fftm/poisson:hip-local --context rootless --jobs 2 \
    --output build/poisson_hip_build
```

Every build output directory must be new. Failures retain `build.log` and the exact
context. On retry use a new directory; Docker caches dependency layers.
`--docker /absolute/path/to/docker` selects a non-PATH client.

`build.py` prepares an allowlisted context containing tracked library/example
files, SCFD and capsule files. It excludes result archives, the paper, `.git`,
credentials, local build profiles and unrelated untracked experiments. Included
source modifications are recorded, so a dirty build cannot masquerade as an
unchanged release. `source.sha256` identifies each included file.

The Dockerfiles pin base-image digests. `dependencies.json` pins verified UCX and
Open MPI source SHA-256 values. Ubuntu packages come from package repositories;
installed versions are recorded, but a future independent rebuild is **not
guaranteed bit-for-bit identical**. Archive the tested image for exact delivery.

Direct Dockerfile invocation also uses the prepared context:

```bash
python3 Docker_config/poisson/build.py cuda --prepare-only \
    --tag fftm/poisson:cuda-local --output build/cuda_context
docker --context rootless build \
    -f build/cuda_context/context/src/Docker_config/poisson/Dockerfile.cuda \
    --build-arg BUILD_JOBS=2 -t fftm/poisson:cuda-local build/cuda_context/context
```

## Inspect and run

GPU-free inspection:

```bash
docker --context rootless run --rm fftm/poisson:cuda-local info
docker --context rootless run --rm fftm/poisson:hip-local info
```

Complete verification, each command on its corresponding GPU machine:

```bash
bash Docker_config/poisson/run.sh cuda fftm/poisson:cuda-local "$PWD/build/capsule_cuda_run"
bash Docker_config/poisson/run.sh hip fftm/poisson:hip-local "$PWD/build/capsule_hip_run"
```

One-GPU verification:

```bash
CAPSULE_CUDA_DEVICES=0 bash Docker_config/poisson/run.sh \
    cuda fftm/poisson:cuda-local "$PWD/build/cuda_one" verify --ranks 1
CAPSULE_HIP_VISIBLE_DEVICES=0 bash Docker_config/poisson/run.sh \
    hip fftm/poisson:hip-local "$PWD/build/hip_one" verify --ranks 1
```

Two-device defaults are `CAPSULE_CUDA_DEVICES=0,1` and
`CAPSULE_HIP_DEVICES=/dev/dri/renderD128,/dev/dri/renderD129`. Use the host's actual
device identifiers. `/dev/kfd` is always passed for HIP. `CAPSULE_DOCKER` and
`CAPSULE_DOCKER_CONTEXT` select a client/context without changing global defaults.

On ROCm 6.4.1, hiding one render node on a two-GPU host can cause runtime
initialization to fail even on the remaining GPU. Pass the required render nodes
and use `CAPSULE_HIP_VISIBLE_DEVICES=0` to select only GPU 0 for computation. This
sets `ROCR_VISIBLE_DEVICES` inside the container; it does not modify the host or
select the other GPU for work. `CAPSULE_HIP_DEVICES` controls device passthrough,
not the rank-to-compute-device mask.

Default verification: **36 MPI invocations per image**. For each rank count
(1,2) and transport (device-aware,host-staged), it runs:

| Check | Invocations |
| --- | ---: |
| Distinct GPU mapping and GPU/host buffer round trip | 1 |
| 3D `32 x 40 x 48`: measured creation, reuse, rejected wrong size | 3 |
| 4D `16 x 20 x 24 x 32`: both-layout tuning, reuse, rejected wrong size | 3 |
| 4D explicit `public-yzwx` and explicit `native-xzwy` | 2 |

Device-aware binaries use UCX. Host-staged binaries are compiled without
device-aware MPI and use Open MPI's `ob1` PML with `self,sm,tcp`. This checks
different FFTM transport contracts, not only MPI environment settings. It does
not establish RDMA or multi-node performance on a single-node host.

The HIP image explicitly defaults to `UCX_TLS=self,sm,tcp,rocm_copy` for its
device-aware cases. MPI still accepts device pointers, but UCX can stage them
through host memory. This is not a claim of direct GPU-to-GPU IPC: automatic
`rocm_ipc` selection failed peer access on the two-Radeon validation host.
FFTM's host-staged executable is a separate test of staging inside FFTM itself.
CUDA retains UCX's automatic transport selection. The effective MPI environment
is recorded for every case. See the [UCX transport reference](https://openucx.readthedocs.io/en/master/faq.html).

To explicitly test automatic ROCm transport selection on a host with working
GPU peer access, use a new output directory and cache:

```bash
CAPSULE_UCX_TLS=all bash Docker_config/poisson/run.sh \\
    hip fftm/poisson:hip-local "$PWD/build/hip_ipc_probe" preflight --ranks 2 --transport device-aware
```

This override is never applied to the host-staged MPI variant. A failed IPC
preflight is not silently converted into a successful copy-transport result.

The build requires Open MPI's own CUDA/ROCm accelerator support as well as
UCX's GPU transports. CUDA configuration explicitly selects the toolkit's
build-time `libcuda` stub directory, avoiding Open MPI 5.0.7's ambiguous
auto-detection when both `stubs` and `compat` libraries exist. No driver stubs
are shipped in the runtime image. Preflight tests both GPU send/receive and
GPU-buffer `MPI_Alltoallv` self-copy; UCX send/receive alone is insufficient.

Successful solves must have finite relative L2 error below `1e-10`, and the
requested shape, rank count, backend and transport. Creation must report
`source=measured`; reuse must report `source=cache` and leave the cache unchanged.
Negative tests must fail specifically for a cache mismatch without modifying the
cache. A crash/timeout cannot count as successful rejection. Failed preflights
skip dependent solves and make the complete matrix fail.

## Individual solves and persistent caches

Use a common host directory for caches and a new `--output` subdirectory for each
invocation. Substitute `hip` and its image for AMD:

```bash
bash Docker_config/poisson/run.sh cuda fftm/poisson:cuda-local "$PWD/build/individual" \
    poisson3d --ranks 2 --transport host-staged --sizes 64 80 96 \
    --cache /data/cache3.env --output /data/create3
bash Docker_config/poisson/run.sh cuda fftm/poisson:cuda-local "$PWD/build/individual" \
    poisson3d --ranks 2 --transport host-staged --sizes 64 80 96 \
    --cache /data/cache3.env --output /data/reuse3
bash Docker_config/poisson/run.sh cuda fftm/poisson:cuda-local "$PWD/build/individual" \
    poisson4d --ranks 2 --transport device-aware --sizes 16 20 24 32 \
    --layouts public-yzwx,native-xzwy --cache /data/cache4.env --output /data/create4
```

Only the output directory is mounted. The container uses private 1 GiB shared
memory and a fixed hostname for identity consistency between invocations. MPI
runs inside the container; do not wrap Docker in host `mpirun`.
Strict device identity is enabled. Use a new cache for changed GPUs, shapes,
transports, ranks or accepted layouts rather than disabling cache validation.

## Evidence and failures

The host output root contains `image-inspect.json`. `OUTPUT_DIR/run/` (or the
explicit `--output`) contains `run.json` provenance, per-case `status.jsonl`,
`summary.json`, exact `*.command.json` arguments/environment, `*.log` output,
and `*.env` caches. Keep the entire directory, including failed cases. Git SHA
alone is insufficient provenance: retain the image ID and source manifest too.
Timing fields here are diagnostic, not paper performance measurements.

Each invocation has a 300-second timeout, configurable with `--timeout SECONDS`.
The runner terminates only that invocation's process group and retains its log.
Missing/inaccessible GPUs, insufficient free memory, unsupported architectures,
MPI errors, numerical failures and cache failures all return nonzero. Read the
first failed case's log, not just the summary. GPU-free dependency diagnostics:

```bash
docker --context rootless run --rm --entrypoint /opt/mpi/bin/ompi_info fftm/poisson:hip-local --all
docker --context rootless run --rm --entrypoint /opt/ucx/bin/ucx_info fftm/poisson:hip-local -v
```

If rootful Docker created root-owned output, do not apply broad permission changes;
use the intended rootless context and a new directory. Never prune a shared daemon.

## Export and load release assets

```bash
python3 Docker_config/poisson/archive.py --context rootless export \
    --image fftm/poisson:cuda-local --name poisson-cuda --output build/cuda_delivery
```

This retains `poisson-cuda.tar.gz` and creates ordered parts no larger than
**1900 MiB**, below GitHub's 2 GiB per-asset limit. Upload the parts,
`poisson-cuda.manifest.json`, and `poisson-cuda.manifest.sha256`, not the full
archive. Use `--name poisson-hip` for HIP to avoid asset-name collisions.
Publication is not performed automatically. Checksums establish
integrity, not authenticity: obtain them from the trusted release.

After downloading all files for one backend:

```bash
cd downloaded_cuda_bundle
sha256sum -c poisson-cuda.manifest.sha256
python3 /path/to/fftm/Docker_config/poisson/archive.py verify poisson-cuda.manifest.json
python3 /path/to/fftm/Docker_config/poisson/archive.py --context rootless load poisson-cuda.manifest.json
```

The loader checks sizes, part hashes, order, and combined SHA-256 before Docker
load, then verifies the resulting image ID. Use the manifest's image tag with
`run.sh`. A registry alternative is `docker pull REGISTRY/IMAGE@sha256:...`, using
the publisher's tested digest, not a substituted mutable tag.

Do not add image layers or paper data to Git. Preserve FFTM/SCFD, MPI/UCX and
vendor notices with redistribution; review NVIDIA/AMD runtime license terms
before publishing. Runtime source snapshots plus archived UCX/MPI tarballs are
inside `/opt/fftm`; their checksums and package versions are under `metadata/`.

## Maintainer checks

```bash
python3 -m unittest discover -s scripts/tests -p test_build_configuration.py
python3 -m unittest discover -s scripts/tests -p test_poisson_capsule.py
make -C source/tests check-abstraction-boundaries
```

These host tests do not replace the GPU container matrix. `probe.cpp` uses the
FFTM backend facade and SCFD memory, with no application-level CUDA/HIP calls.
