# Build configuration

The Makefiles in `source/tests`, `examples`, and `examples/poisson` share one
configuration. These files contain **GNU Make syntax**, not shell syntax and
not C++ includes. They configure compilation, not FFTM's runtime autotuning.
Use a trusted configuration file: Make includes can execute commands.

## Configure once

From the repository root, choose a template:

```bash
cp build_configs/config_cuda_release.inc build_configs/config_local.inc
# Or: cp build_configs/config_hip_release.inc build_configs/config_local.inc
```

Edit `config_local.inc` for your compiler, GPU architecture, MPI installation,
and output directory. The file is ignored by Git and excluded from the Docker
build context. Then:

```bash
make -C examples/poisson print-config check-config
make -C examples/poisson -j2
make -C source/tests test_fftm_options.bin
make -C examples reader-smoke
```

Alternatively select an explicit profile in every invocation:

```bash
make -C examples/poisson CONFIG_FILE=build_configs/my_machine.inc -j2
make -C source/tests CONFIG_FILE=/absolute/path/my_machine.inc test_fftm_options.bin
```

Relative `CONFIG_FILE` paths are resolved from the **repository root**, even
with `make -C`. A missing or empty explicitly selected file is an error, never
a silent fallback. Without `CONFIG_FILE`, `build_configs/config_local.inc` is
loaded when present; otherwise historical CUDA defaults apply.

Use `config_local_<machine>.inc` for additional ignored personal profiles.
Keep separate output directories for different machines and toolchains.

## Settings

| Setting | Meaning / default |
| --- | --- |
| `FFTM_BUILD_BACKEND` | `cuda` (default) or `hip`; selects the `all` target family. |
| `FFTM_DEVICE_AWARE_MPI` | `0` or `1`; chooses host-staged or device-aware HIP defaults. Explicit binary target names still determine their transport capability. CUDA Poisson/Taylor-Green binaries require device-aware MPI; CUDA tests also supply `_nca` binaries. |
| `cuda_dir`, `ROCM_DIR`, `mpi_dir` | Toolkit/MPI prefixes; `/usr/local/cuda`, `/opt/rocm`, `/usr/local/mpi`. |
| `NVCC`, `HIPCC`, `MPICXX`, `CXX` | Compiler commands; normally derived from prefixes (`CXX` defaults to `g++`). MPI must be built with a compatible host compiler and GPU-pointer support when used directly. |
| `CUDA_HOST_CXX` | Optional `nvcc -ccbin` host compiler. |
| `CUDA_ARCH_LIST` | Space-separated CUDA capabilities without dots, e.g. `70 80`; default `70 75`. |
| `HIP_ARCH_LIST` | Space-separated AMD targets, e.g. `gfx1102`; empty uses hipcc's default detection. Set explicitly for reproducible or cross-machine builds. |
| `CUDA_ARCH`, `HIP_ARCH` | Legacy overrides: raw nvcc architecture flags, or space-separated AMD targets. Override the corresponding list when set. |
| `CPPSTD` | C++ standard, default `c++14`. |
| `TARGET_GCC`, `TARGET_NVCC`, `TARGET_HIP` | Optimization/debug flags, default `-O3`. No fast-math changes are introduced. |
| `GCCFLAGS`, `NVCCFLAGS`, `HIPFLAGS` | Complete compiler-specific flag overrides; replace the composed defaults, including architecture/standard flags. |
| `CPPFLAGS` | Extra preprocessing flags, common to all compilers; use GPU-compiler-compatible syntax. |
| `CXXFLAGS` | Extra flags for CPU-only C++ test compilation; use `CUDA_HOST_CXX`/`NVCCFLAGS` for nvcc. |
| `LDFLAGS`, `NVCC_LDFLAGS`, `LDLIBS` | Host/HIP linker flags, nvcc linker flags, and extra libraries. `-Wl,...` flags are not valid raw nvcc options; use `-Xlinker` in `NVCC_LDFLAGS` as needed. |
| `SCFD_INCLUDE_DIR` | Defaults to `source/contrib/scfd/include`; initialize the SCFD submodule before building. |
| `include_cuda`, `lib_cuda`, `include_mpi`, `lib_mpi`, `ROCM_LIB_DIR` | Override include/library directories independently of prefixes (e.g. a `lib64` MPI installation). |
| `MPI_CPPFLAGS`, `MPI_LDFLAGS`, `MPI_LIBS` | MPI includes, search paths, and libraries; defaults `-I$(include_mpi)`, `-L$(lib_mpi)`, `-lmpi`. For nonstandard installations inspect the MPI wrapper's compiler/link flags and set these explicitly. |
| `OMP` | Existing host-link threading flags, default `-fopenmp -lpthread`. |
| `BUILD_FOLDER`, `BUILD_DIR` | Output location; `BUILD_DIR` is the effective value. Defaults to the entry Makefile's `build` directory. Prefer an absolute path or `$(PROJECT_ROOT)/build/<profile>`. |
| `MPIEXEC`, `MPIEXEC_FLAGS` | MPI launcher and whitespace-separated launcher options used by `reader-smoke`; this does not configure Slurm allocations. |

Retain `SCFD_FLAGS` unless deliberately changing SCFD compilation. The default
`-DSCFD_ARRAYS_ORDINAL_TYPE=ptrdiff_t` is required by the current storage model.
Reference-only tests remain explicitly optional via
`FFTM_ENABLE_FFTM3D_BACKEND=1` and `EGGER_ROOT`; their external sources are not
required for ordinary applications.

Normal Make precedence applies: command-line assignments override profile
assignments; ordinary profile assignments override inherited environment
values. Defaults use `?=`. For example:

```bash
make -C examples/poisson CONFIG_FILE=build_configs/config_cuda_release.inc \
    CUDA_ARCH_LIST=80 BUILD_FOLDER="$PWD/build/a100" -j2
```

A command-line `BUILD_FOLDER` overrides a profile's `BUILD_DIR`; an explicit
command-line `BUILD_DIR` wins if both are supplied. Profiles should not use
Make's `override` directive. Avoid `make -e`.

Inspect `print-config` before building from an environment that injects flags,
such as Conda. The committed release/container templates clear ambient extra
flags; edit those assignments to add your own. Without a profile, environment
`CPPFLAGS`, `CXXFLAGS`, and linker flags are now honored, so inspect them before
comparing performance. A profile can clear unwanted flags with `CPPFLAGS =` or
`LDFLAGS =`. Output is diagnostic `key=value` text, **not a sourceable shell
script**. `check-config` checks executable availability and SCFD headers; it
does not prove ABI compatibility, installed FFT libraries, or device-aware
MPI correctness. Compilation and the runtime smoke test provide those checks.

## Targets and verification

Existing target names, executable suffixes, and explicit command-line
architecture/path overrides remain valid.

- `examples/poisson`: `cuda`, `hip`, `hip-nca`, and backend-selected `all`.
- `examples`: CUDA `all` includes Poisson and Taylor-Green; HIP `all` includes
  only portable Poisson examples. Taylor-Green remains CUDA-only.
- `source/tests`: `cuda` retains the existing CUDA suite; `hip` builds both
  device-aware and host-staged variants, while `hip-nca` builds only host-staged
  variants. `all` selects the family from the profile.
- `test_fftm_options.bin` and `check-config-cxx` can run without a GPU toolkit.
- `check-abstraction-boundaries` remains a source-only check.

The reader smoke suite inherits the selected profile and uses one/two MPI
ranks on small problems. `FFTM_READER_BUILD_DIR`, `FFTM_READER_BACKEND`, and
`FFTM_READER_DEVICE_AWARE_MPI` remain explicit runner overrides. For example:

```bash
make -C examples CONFIG_FILE=build_configs/config_local.inc reader-smoke
python3 scripts/tests/test_build_configuration.py -v
```

The second command tests configuration loading, precedence, recursive builds,
and incremental rebuilding with compiler fixtures and requires no GPU.
Other benchmark/cluster scripts keep their existing `FFTM_*` launch settings;
the profile is not a replacement for those launch configurations.

Compiler depfiles track included headers and `source/detail/*.inc` files.
An effective-configuration stamp forces rebuilding when flags or toolchain
paths change, including when switching back to a previous profile. Existing
builds are rebuilt once to generate their stamps and depfiles. Replacing a
compiler/library **at the same path** is not detectable from its path: use a
new build directory or `make -B`. No existing build or results directory is
deleted by these Makefiles.

The main Docker build uses the committed `config_container_cuda.inc` profile;
its existing `CUDA_ARCH` and output-directory arguments still override that
profile. Local profiles never enter the image build.
