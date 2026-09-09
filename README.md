# FFTM

`fftm` is an experimental CUDA/HIP MPI FFT project for real-to-complex 3D and
4D transforms on one or many GPUs. CUDA/cuFFT is the production and
performance-qualified backend; HIP/hipFFT currently has correctness coverage.

The project contains two closely related transform frontends:

| Frontend | Purpose |
| --- | --- |
| `ffts` | Single-process, single-GPU FFT wrapper. It is used as the serial/reference implementation and as the one-GPU performance baseline. |
| `fftm` | MPI distributed multi-GPU FFT wrapper. It partitions the domain across MPI ranks and performs local FFTs plus MPI transpositions. |

CUDA and HIP runtime/FFT calls remain inside `source/external_wrap/`.
`source/fftm_backend.hpp` is the single compile-time backend selector.
Higher-level FFT, transpose, and example code uses backend-neutral wrapper and
SCFD interfaces; vendor runtime types, copy descriptors, and FFT complex types
do not cross those boundaries. Verify this rule with
`make -C source/tests check-abstraction-boundaries`.

## Repository Layout

| Path | Contents |
| --- | --- |
| `source/ffts.hpp` | Single-GPU FFT frontend. |
| `source/fftm.hpp` | MPI multi-GPU FFT frontend. |
| `source/fftm_backend.hpp` | Compile-time CUDA/HIP backend selection facade. |
| `source/external_wrap/` | Backend wrappers around cuFFT/CUDA and hipFFT/HIP runtime functionality. |
| `source/detail/` | Transpose kernels, MPI transpose classes, profiling, and shared implementation details. |
| `source/tests/` | Unit tests, Poisson examples, comparison tests, versioned tests, and benchmark binaries. |
| `examples/poisson/` | Minimal periodic 3D autotuning and 4D native-spectral Poisson applications. |
| `examples/turbulence/` | SCFD-based 3D Taylor-Green simulation, vorticity/Q visualization, and 4D spatio-temporal filtering. |
| `scripts/` | Local, Docker, Slurm/Pyxis, and analysis scripts for collecting benchmark data and generating figures/tables. |
| `build_configs/` | Shared CUDA/HIP Make configuration, machine-profile templates, and build-variable reference. |
| `Docker_config/Dockerfile` | Docker image used for local and cluster benchmark runs. |
| `fft_Egger/` | Reference implementation used for comparison experiments. |

## Requirements

The default CUDA build assumes:

| Component | Default path |
| --- | --- |
| CUDA toolkit | `/usr/local/cuda` |
| MPI | `/usr/local/mpi` |
| C++ standard | C++14 |
| Build tool | `make` |
| Python | `python3` for benchmark orchestration and analysis |

The Makefile uses `nvcc` for CUDA compilation and `mpicxx` for final MPI-linked
executables. The shared default CUDA architectures are
`sm_70` and `sm_75`; override `CUDA_ARCH` when building for another GPU family.

### Configure a machine once

The test and example Makefiles share a GNU Make configuration file. Copy
`build_configs/config_cuda_release.inc` or `config_hip_release.inc` to
`build_configs/config_local.inc`, then edit its compiler/toolkit/MPI paths,
GPU architecture list, and optional `BUILD_FOLDER`. The local file is ignored
by Git and excluded from Docker builds.

```bash
cp build_configs/config_cuda_release.inc build_configs/config_local.inc
# Edit build_configs/config_local.inc for this machine, then:
make -C examples/poisson print-config check-config
make -C examples/poisson -j2
make -C examples reader-smoke
```

Use `CONFIG_FILE=build_configs/my_machine.inc` to select another profile;
relative paths are resolved from the repository root. Missing or empty
explicit profiles fail immediately. Without a profile, historical defaults
remain available. Command-line overrides still work. Header dependencies
(including C++ `.inc` files) and changes to effective build settings trigger
recompilation automatically.

See [Build configuration](build_configs/README.md) for the complete variable
reference, CUDA/HIP target selection, precedence, and verification commands.
These build settings are separate from the runtime autotune options below.

### Direct command-line builds

Example for A100:

```bash
make -C source/tests all \
  CUDA_ARCH="-gencode arch=compute_80,code=sm_80" \
  -j8
```

Example for V100/Turing:

```bash
make -C source/tests all \
  CUDA_ARCH="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75" \
  -j8
```

By default binaries are created in `source/tests/build`. Override the output
directory with `BUILD_FOLDER`:

```bash
make -C source/tests all BUILD_FOLDER="$PWD/build/fftm_tests" -j8
```

### HIP/hipFFT correctness targets

HIP targets require ROCm, hipFFT, and an MPI installation. The build is kept
separate from CUDA targets and does not change CUDA compilation flags or code
paths:

```bash
make -C source/tests hip \
  BUILD_FOLDER="$PWD/build/fftm_hip_tests" \
  ROCM_DIR=/opt/rocm \
  mpi_dir=/path/to/mpi \
  HIP_ARCH=gfx90a \
  -j4
```

The reusable correctness runner checks both GPUs individually, then checks 3D
and 4D slab and pencil decompositions with `alltoallv` and `p2p-waitany`.
When device-aware MPI is enabled, it also validates the optimized 4D
`native_xzwy` WZ-plan pipeline with a distributed Poisson solve:

```bash
FFTM_HIP_MPI_DIR=/path/to/mpi \
FFTM_HIP_ARCH=gfx90a \
FFTM_HIP_TEST_HOST_STAGED=1 \
FFTM_HIP_TEST_DEVICE_AWARE=1 \
scripts/run_hip_correctness.sh
```

Set `FFTM_HIP_TEST_DEVICE_AWARE=0` when MPI cannot communicate with ROCm device
pointers. That still exercises the same hipFFT and HIP runtime implementation,
with MPI traffic staged through pinned host buffers.

The HIP build also includes the production benchmark drivers. A compact
same-host comparison against direct FFTW-MPI is available for machines with two
GPUs and 40 physical CPU cores:

```bash
scripts/run_hip_fftw_comparison.sh

python3 scripts/analyze_hip_fftw_comparison.py \
  build_hip/fftm_vs_fftw_TIMESTAMP
```

The default matrix compares `256^3` and `64^4`, which contain the same number
of real samples. FFTM records one- and two-GPU host-staged paths; FFTW records
one, two, and four MPI ranks while keeping the total at 40 physical CPU cores.
Override the paths, sizes, iteration counts, CPU masks, and planner limits with
the `FFTM_COMPARE_*` environment variables in the runner when using a different
host topology.

## CUDA-Aware and Non-CUDA-Aware MPI Builds

Most MPI test targets are built in two variants:

| Suffix | Meaning |
| --- | --- |
| `.bin` | Built with SCFD's legacy-named `SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI` capability; MPI may communicate directly with CUDA or HIP device pointers. |
| `_nca.bin` | Built without CUDA-aware MPI; communication stages through host memory. |

Use `_nca.bin` on clusters where GPU-aware MPI is unavailable or unreliable.

Examples:

```bash
mpiexec -n 2 ./source/tests/build/test_benchmark_fftm_3D.bin \
  --strategy slab-pencil --mode p2p-waitany --times 10 --warmup 3 1024 1024 1024

mpiexec -n 2 ./source/tests/build/test_benchmark_fftm_3D_nca.bin \
  --strategy slab-pencil --mode p2p-waitany --times 10 --warmup 3 1024 1024 1024
```

## `ffts`: Single-GPU FFT

`ffts` is the single-process transform class. It supports:

| Dimension | Transform |
| --- | --- |
| 2D | Real-to-complex forward and complex-to-real backward FFT. |
| 3D | Real-to-complex forward and complex-to-real backward FFT, normally using cuFFT 3D plans. |
| 4D | Real-to-complex forward and complex-to-real backward FFT implemented as staged local transforms and local transposes. |

The 4D `ffts` implementation supports these local strategies:

| Strategy | Description |
| --- | --- |
| `pencil-direct` | 1D FFT stages with direct local transpose kernels. |
| `pencil-memcpy` | 1D FFT stages with runtime-copy-based local transposes. |
| `slab-direct` | 2D FFT stages with direct local transpose kernels. |
| `slab-memcpy` | 2D FFT stages with runtime-copy-based local transposes. |

The `ffts` benchmarks are the preferred one-GPU baseline for performance plots.

## `fftm`: MPI Multi-GPU FFT

`fftm` distributes the transform over MPI ranks. Each MPI rank selects one
backend device through `fftm::device_backend::init_mpi`; the selection facade
delegates to the corresponding SCFD initializer.

### Production configuration

Application code should start from a production preset rather than configuring
transport and plan-execution switches individually:

```cpp
auto options_3d = fftm::production_options_3d(
    fftm::transform_strategy_3d::pencil_pencil,
    comm.num_procs,
    true // device-aware MPI
);

auto options_4d = fftm::production_options_4d(
    fftm::transform_strategy_4d_mpi::slab_slab,
    fftm::fftm_4d_spectral_layout::native_xzwy,
    true // device-aware MPI
);

// The native direct-layout preset requires this communication mode.
using slab_strategy_4d = fftm::strategy_4d_slab_slab_mpi<
    fftm::mpi_transpose_3d_mode::p2p_waitany>;

// For an explicitly selected 4D pencil strategy, provide the rank topology.
// A nodes x ranks-per-node x 1 grid selects the node-aligned WZ pipeline.
fftm::fftm_4d_production_topology topology(
    comm.num_procs, 8, grid.p1, grid.p2, grid.p3
);
auto pencil_options_4d = fftm::production_options_4d(
    fftm::transform_strategy_4d_mpi::pencil_pencil,
    fftm::fftm_4d_spectral_layout::native_xzwy,
    true,
    topology
);
```

Set the topology policy to `fftm::fftm_4d_pencil_pipeline::standard` to retain
the standard pencil path explicitly. An explicitly requested
`node_aligned_wz` policy fails if the process grid, layout, or transport does
not satisfy its prerequisites.

The top-level `fftm_init_options` contains layout selection and three grouped
option sets:

| Group | Purpose |
| --- | --- |
| `reporting` | Explicit profiling, verbose setup messages, and destruction-time summaries. Defaults to silent. |
| `execution` | Internal policy selected by production presets or the autotune cache. Direct use is intended for benchmark construction. |
| `diagnostics` | Unsafe, failed, or measurement-only variants. These are disabled by default and are not production API choices. |

The tables below document the public controls declared in
[`source/fftm_options.hpp`](source/fftm_options.hpp). The
[execution and diagnostic reference](docs/fftm_options_reference.md) lists
every internal field, its default, prerequisites, and compatibility aliases.
An option being listed does not mean it is a measured-autotune candidate.
The [measured autotuning section](#c-measured-autotuning) describes that
separate API and its candidate space.

#### Initialization fields

These are the defaults of `fftm::fftm_init_options{}`, **before** production
presets, measured selection, and initialization-time normalization. Set options
before initializing the plan; editing the original options object afterward
does not reconfigure an initialized transform.

| Field | Default | Meaning and restrictions |
| --- | --- | --- |
| `use_optimized` | `true` | Enables optimized 3D execution where the compile-time strategy supports it. It cannot change a strategy's SCFD array types. An optimized pencil-pencil strategy rejects `false`; use `strategy_3d_pencil_pencil<Mode, false>` for the legacy path. |
| `pencil_layout_3d` | `fftm_3d_pencil_layout::auto_select` | Selects the 3D pencil plan/layout variant; see the enum table below. Relevant to pencil-pencil, not a general tensor-permutation control. |
| `pencil_pipeline_3d` | `fftm_3d_pencil_pipeline::staged` | Selects how the 3D pencil FFT and redistribution stages are executed. `reference_parity` additionally normalizes several transport switches. |
| `spectral_layout_4d` | `fftm_4d_spectral_layout::public_yzwx` | Selects the physical 4D spectral contract. Applications must use the corresponding array view and forward/backward interface; this is not merely a performance hint. |
| `reporting` | `fftm_reporting_options{}` | Timing, memory reporting, and verbosity; all output is off by default. |
| `execution` | `detail::fftm_execution_options{}` | Internal transport, plan, and workspace policy. Let a preset or measured selection populate it; see the [full inventory](docs/fftm_options_reference.md#execution-options). |
| `diagnostics` | `detail::fftm_diagnostic_options{}` | Experimental and measurement controls, not ordinary production settings; see the [full inventory](docs/fftm_options_reference.md#diagnostic-options). |

#### Strategies, layouts, and policies

The values below are C++ enum members, using underscores. CLI/cache spellings
can use hyphens and are defined by their adapters, not by automatic environment
variable parsing in the library. `P` denotes the total MPI rank count.

| Enum | All values | Meaning |
| --- | --- | --- |
| `transform_strategy_3d` | `slab_pencil`, `pencil_slab`, `pencil_pencil` | 3D strategy families, with process grids `P x 1`, `1 x P`, or `P1 x P2`, respectively; `P1*P2=P`. See [3D strategies](#3d-strategies). |
| `transform_strategy_4d_mpi` | `slab_slab`, `pencil_pencil` | 4D strategy families, with `1 x P x 1` for slab-slab or `P1 x P2 x P3` for pencil-pencil; `P1*P2*P3=P`. See [4D strategies](#4d-strategies). |
| `mpi_transpose_3d_mode` | `p2p_waitall`, `p2p_waitany`, `alltoallv`, `alltoallw` | Shared by 3D **and 4D**, despite the historical type name. Selects peer-to-peer completion or variable-count collective exchange. Not every pipeline supports every mode; see [MPI modes](#mpi-communication-modes). |
| `fftm_3d_pencil_layout` | `auto_select`, `opt0`, `opt1`, `legacy` | `opt0` and `opt1` are historical identifiers for two physical FFT/transpose implementations, not compiler optimization levels or a ranking. The optimized native layout resolver uses `opt0` for `P1>P2`, otherwise `opt1`, when left at `auto_select`; this is a heuristic, not a measured sweep. `legacy` requires a non-optimized compile-time pencil strategy. |
| `fftm_3d_pencil_pipeline` | `staged`, `fused`, `reference`, `reference_parity`; aliases `egger`, `egger_parity` | `staged` runs separate stage operations; `fused` uses the combined FFT/transpose executor, without implying all work is one GPU kernel. `reference` uses FFTM's owned reference-style executor; `reference_parity` adds the constrained device-aware byte schedule. Neither requires the external reference project. Deprecated aliases map to `reference` and `reference_parity`. |
| `fftm_4d_spectral_layout` | `public_yzwx`, `native_xzwy` | Public spectral order `(y,z,w,x)` versus native order `(x,z,w,y)`. For native output, use `get_local_spectral_sizes_4d`, `forward_native_spectral_4d`, and `backward_native_spectral_4d` with the matching SCFD view. Raw indexing must follow that view's arranger. |
| `fftm_4d_pencil_pipeline` | `auto_select`, `standard`, `node_aligned_wz` | Topology policy for pencil-pencil. `auto_select` chooses node-aligned WZ only when its grid, native layout, and device-aware prerequisites hold; otherwise standard. `standard` disables that node-aligned choice, but does not disable eligible degenerate local optimizations. Explicit `node_aligned_wz` rejects incompatible inputs. |
| `fftm_4d_slab_backward_credit_policy` | `auto_select`, `disabled` | Selects the production backward-plane credit budget or the unbounded fallback. `disabled` does not disable communication or synchronization. |

Advanced enums `fftm_3d_large_count_p2p_transport` and
`fftm_3d_contiguous_forward_send_mode` are documented with their owning
[internal controls](docs/fftm_options_reference.md#advanced-enums).

Compile-time strategy tags are `strategy_3d_slab_pencil<Mode, UseOptimized>`,
`strategy_3d_pencil_slab<Mode, UseOptimized>`,
`strategy_3d_pencil_pencil<Mode, UseOptimized>`,
`strategy_4d_slab_slab_mpi<Mode>`, and `strategy_4d_pencil_pencil_mpi<Mode>`.
All default to `Mode=mpi_transpose_3d_mode::alltoallv`; the 3D tags also
default to `UseOptimized=true`. The tags fix the strategy/mode used by a
templated plan. A production-options function does **not** replace its strategy
tag or select a different MPI mode. In particular, the fast native direct-layout
4D presets require a `p2p_waitany` tag and a device-aware MPI build/runtime.

#### Reporting fields

All fields below belong to `options.reporting` (`fftm_reporting_options`).
Enabling a destruction-time print flag alone does not enable its profiler.

| Field | Default | Meaning |
| --- | --- | --- |
| `profiling_key` | Empty string | Nonempty key enables the temporal profiler under this name; empty disables it. |
| `memory_profiling_key` | Empty string | Nonempty key enables the memory profiler under this name; empty disables it. |
| `verbose` | `false` | Enables optional setup/status messages; independent of the profiler keys. |
| `print_profile_summary_on_destroy` | `false` | Prints the enabled temporal profiler's detailed summary when the plan is destroyed. |
| `print_profile_totals_on_destroy` | `false` | Prints the enabled temporal profiler's totals on destruction. |
| `print_memory_profile_on_destroy` | `false` | Prints the enabled memory profiler's detailed summary on destruction. |
| `print_memory_totals_on_destroy` | `false` | Prints the enabled memory profiler's totals on destruction. |

`fftm::profiling_reporting_options()` returns keys `"fftm_prof"` and
`"fftm_mem"` with all four destruction-time print flags enabled. It leaves
`verbose=false`. Use `options.reporting = fftm::profiling_reporting_options()`
when these reports are wanted. Fine-grained stage timers are separate
diagnostics and can perturb the measured execution time.

#### Topology and preset behavior

`fftm_4d_production_topology` supplies policy metadata; it neither discovers
nodes nor creates the process grid nor binds ranks to devices. Supply the actual
allocation and the same grid used to initialize the plan. Node-aligned operation
assumes rank placement consistent with that grid.

| Field | Default | Meaning |
| --- | --- | --- |
| `num_procs` | `1` | Positive total MPI rank count. |
| `procs_per_node` | `1` | Positive ranks per node, for a uniform node allocation. |
| `p1` | `1` | First positive process-grid extent. |
| `p2` | `1` | Second positive process-grid extent. |
| `p3` | `1` | Third positive process-grid extent; the grid product must equal `num_procs`. |
| `pencil_pipeline` | `fftm_4d_pencil_pipeline::auto_select` | Node-aligned versus standard pencil policy. |
| `slab_backward_credit_policy` | `fftm_4d_slab_backward_credit_policy::auto_select` | Automatic backward-plane budget versus disabled fallback. |

| Preset/helper | Selection and defaults |
| --- | --- |
| `production_options_3d(strategy, num_procs, device_aware_mpi=true)` | Requires a positive rank count. For more than one rank and pencil-pencil, selects `reference_parity` with device-aware MPI or `reference` otherwise. Selects `opt0` for device-aware counts 7, 8, or at least 64, and `opt1` otherwise. Enables the native opt0 execution bundle for `opt0`. Other strategies and one-rank calls retain the initialization defaults, apart from the device-aware capability field. |
| `select_production_pencil_layout_3d(num_procs, device_aware_mpi=true)` | Returns the same rank-count-based `opt0`/`opt1` policy above without constructing options; distinct from the grid-based `auto_select` resolver. |
| `production_options_4d(strategy, spectral_layout=public_yzwx, device_aware_mpi=true)` | Convenience overload using default topology `(1,1,1,1,1)`. It does not discover multi-node topology. Use the four-argument overload for actual grid-dependent decisions. |
| `production_options_4d(strategy, spectral_layout, device_aware_mpi, topology)` | For native/device-aware slab-slab: direct XW, chunking with window 1, compact staging, work-area aliasing, communication-native WZ, concurrency 4, ready pipeline, and the applicable backward-credit policy. For native pencil-pencil: node-aligned WZ where eligible; otherwise local-transpose/sliced-Z specialization on `1 x P2 x 1`, or the standard path. Public-layout/host-staged calls do not enable the slab native direct-layout bundle. |
| `is_4d_node_aligned_pencil_topology(topology)` | Validates positive extents and grid product. Returns true for a multi-node uniform allocation with grid `nodes x procs_per_node x 1`. |
| `select_production_pencil_pipeline_4d(topology, spectral_layout, device_aware_mpi)` | Applies `pencil_pipeline` and the native/device-aware/node-aligned prerequisites. Explicit incompatible requests throw; `auto_select` falls back to `standard`. |
| `select_production_slab_backward_credit_4d(topology, spectral_layout, device_aware_mpi)` | With `auto_select`, native layout, device-aware MPI, and eight ranks per node: selects `(window, cyclic_peer_order)` as `(8,false)` at 16 ranks, `(6,true)` at 24, and `(4,true)` at 32. Other configurations and `disabled` return `(0,false)`. |

The last helper returns `fftm_4d_slab_backward_credit_selection`, whose two
fields are `window` (default `0`, unbounded fallback) and `cyclic_peer_order`
(default `false`, current peer order). A positive window bounds backward planes
in flight independently of FFT-plan concurrency; it is not a GPU count.

Presets are deterministic policies, not performance guarantees or measured
autotuning. They do not inspect problem-size memory feasibility themselves;
plan initialization and the measured evaluator apply the relevant checks. For
the complete preset-to-execution mapping and initialization overrides, see
[preset expansion and normalization](docs/fftm_options_reference.md#preset-expansion-and-normalization).

String helpers in this header are `mpi_transpose_3d_mode_name`,
`fftm_4d_spectral_layout_name`, `fftm_4d_pencil_pipeline_name`,
`fftm_3d_large_count_p2p_transport_name`, and
`fftm_3d_contiguous_forward_send_mode_name`. They return printable names (for
example `p2p-waitany`, `native-xzwy`, `node-aligned-wz`, or `mpi-count`);
they do not parse strings or change settings.

### C++ measured autotuning

The C++ autotune API has two layers for both 3D and 4D:

- `fftm_autotune.hpp` and `fftm_autotune_4d.hpp` create or load known
  production-policy caches.
- `fftm_autotune_measure.hpp` and `fftm_autotune_measure_4d.hpp` measure
  complete forward/backward pairs for production-safe candidate matrices,
  store every candidate timing, and select the lowest median max-rank wall
  time.

Schema-v2 caches validate the GPU model and architecture, device memory,
CUDA or HIP runtime and driver, MPI implementation, node/rank topology, distinct
devices per node, and transport environment. Exact device UUIDs, PCI IDs, and
node names are recorded but only enforced when
`autotune_options::strict_device_identity` is enabled. This lets an ordinary
cache move between equivalent scheduler nodes while preventing a wrapped
multi-rank/one-GPU run from matching a real multi-GPU allocation. Legacy
hardware signatures are rejected unless explicitly allowed.

The minimal measured setup is:

```cpp
fftm::autotune::autotune_options cache;
cache.cache_file = "fftm_3d.env";

fftm::autotune::measured_3d_options measurements;
measurements.warmup = 2;
measurements.iterations = 5;
measurements.verify_candidate_memory_recovery = true;

using evaluator_t = fftm::autotune::fftm_3d_candidate_evaluator<
    fft_backend_t, comm_t, backend_t, log_t>;
evaluator_t evaluator(comm, sizes, log);

auto selected = fftm::autotune::load_or_measure_3d_config<runtime_api_t>(
    comm, sizes, cache, measurements, evaluator);
```

The 4D interface follows the same protocol. Applications that can consume both
implemented physical contracts allow the layout to participate in measurement:

```cpp
fftm::autotune::autotune_options_4d cache4d;
cache4d.cache_file = "fftm_4d.env";
cache4d.requested_spectral_layout = fftm::fftm_4d_spectral_layout::native_xzwy;
cache4d.accepted_spectral_layouts = {
    fftm::fftm_4d_spectral_layout::public_yzwx,
    fftm::fftm_4d_spectral_layout::native_xzwy
};

fftm::autotune::measured_4d_options measurements4d;
using evaluator_4d_t = fftm::autotune::fftm_4d_candidate_evaluator<
    fft_backend_t, comm_t, backend_t, log_t>;
evaluator_4d_t evaluator4d(comm, sizes4d, log);

auto selected4d = fftm::autotune::load_or_measure_4d_config<runtime_api_t>(
    comm, sizes4d, cache4d, measurements4d, evaluator4d);
```

The measured 4D matrix covers slab-slab and pencil-pencil production paths,
eligible `p2p-waitany` and `alltoallv` modes, the node-aligned pencil grid on
multiple nodes, and the validated slab backward-credit fallback where it is
applicable. It measures each accepted spectral layout end to end and records
the accepted-layout set in the cache. Changing that set requires a new
measurement. Leave `accepted_spectral_layouts` empty to retain the exact
`requested_spectral_layout` contract. Combinations that conflict with a
required native direct layout are removed during candidate preflight instead
of being launched. Use
`init_autotuned_4d_plan` and `init_autotuned_4d_data_arrays` to transfer the
retained SCFD workspace and data pools to the selected application plan.

An expert can constrain a cached selection without rerunning measurements:

```cpp
cache.constraints_3d.strategy_3d = "pencil-pencil";
cache.constraints_3d.pencil_layout = "opt0";
```

The 3D Poisson example maps the same C++ fields to environment variables for
command-line experiments. These variables belong to the example adapter; the
FFTM library API itself uses `autotune_options` and `measured_3d_options`.
Unset string variables leave that dimension unconstrained. Constraint values
are matched exactly against valid measured candidates in the cache.

| 3D Poisson environment variable | Accepted values and effect |
| --- | --- |
| `FFTM_CPP_AUTOTUNE_STRATEGY_3D` | `slab-pencil`, `pencil-slab`, or `pencil-pencil`. Restricts selection to the named decomposition and local-FFT pipeline. |
| `FFTM_CPP_AUTOTUNE_MODE` | The measured tuner currently generates `p2p-waitany` and `alltoallv`. The former uses nonblocking peer transfers and processes receives as they complete; the latter uses the variable-count MPI collective. The runtime can dispatch `p2p-waitall` and `alltoallw`, but they are not current measured candidates. |
| `FFTM_CPP_AUTOTUNE_BACKEND_3D` | Currently `native`. This constrains the FFTM implementation recorded in the cache; it does not select CUDA versus HIP, which is fixed by the build. |
| `FFTM_CPP_AUTOTUNE_GRID_3D` | Exact `P1xP2` process grid, for example `4x2`; `P1*P2` must equal the MPI rank count. It selects the two communicator dimensions for `pencil-pencil`; current slab candidates use `P x 1` or `1 x P`. |
| `FFTM_CPP_AUTOTUNE_PENCIL_LAYOUT` | `opt0` or `opt1` for measured `pencil-pencil` candidates. These are historical names for two physical transpose/FFT layout implementations, not optimization levels or a quality ordering. Leave this unset to compare/select the eligible production layout. The runtime parser also retains `auto` and `legacy`, but the measured pencil matrix does not normally generate them. |
| `FFTM_CPP_AUTOTUNE_STRICT_DEVICE_IDENTITY` | Boolean, default `false`. When enabled, cache validation additionally requires the exact device UUIDs, PCI bus IDs, and node names. The default compatibility signature permits equivalent devices on different scheduler nodes. |
| `FFTM_CPP_AUTOTUNE_ALLOW_LEGACY_SIGNATURE` | Boolean, default `false`. Permits loading a cache with a missing or pre-v2 hardware signature. This weakens hardware validation and is intended only for explicit migration of old caches; regenerating the cache is preferred. |

Boolean variables accept `1/0`, `true/false`, `yes/no`, or `on/off` (with the
documented lower- or upper-case spellings). The first five variables constrain
winner selection; the final two control cache validation. On creation of a new
measured cache, candidate generation is controlled by `measured_3d_options`,
all enabled candidates are measured, and the constraints are then applied when
selecting a valid winner. Consequently, a constraint does not by itself shorten
the initial sweep. On cache reuse, it selects the fastest matching stored
candidate without repeating measurements. An unsatisfied constraint fails
instead of silently selecting a different configuration.

The measured cache retains the global winner and the timings/configuration of
all tested candidates, so removing the constraints restores global-fastest
selection. Plan construction is outside the timed section; the benchmarked
quantity is a synchronized forward/backward transform pair. During a measured
sweep, candidates lease backend-neutral SCFD pools for the FFT workspace and
the communication-visible complex spectrum. These allocations are reused
without freeing and reallocating device-aware MPI-registered storage, then
transferred to the selected application plan and its spectrum view. The real
input retains normal SCFD ownership because it is not registered by the
validated communication paths. Applications using a measured
selection should initialize the transform arrays with
`init_autotuned_3d_data_arrays`. The 4D evaluator additionally retains the
SCFD input pool because its larger candidate matrix can otherwise expose the
same device-aware MPI allocation-lifetime behavior; selected 4D applications
use `init_autotuned_4d_data_arrays`. The memory guard reports intentional pool
growth separately and still rejects unexplained retained memory.

### 3D Strategies

| Strategy | Process grid | Local FFT structure |
| --- | --- | --- |
| `slab-pencil` | `P1 x P2`, currently requiring `P2 = 1` | Optimized path uses a local 2D YZ real transform, one MPI redistribution, then local X complex FFT. |
| `pencil-slab` | `P1 x P2`, currently requiring `P1 = 1` | Optimized path uses local Z real FFT, one MPI redistribution, then local 2D XY complex FFT. |
| `pencil-pencil` | `P1 x P2` | Uses Z, Y, and X FFT stages with two MPI redistributions. Intended for true 2D process grids. |

### 4D Strategies

| Strategy | Process grid | Local FFT structure |
| --- | --- | --- |
| `slab-slab` | `P1 x P2 x P3`, currently requiring `P1 = 1` and `P3 = 1` | Local 2D ZW transform, one MPI redistribution, then local 2D XY transform. |
| `pencil-pencil` | `P1 x P2 x P3` | Uses staged 1D FFTs and multiple MPI redistributions. |

### MPI Communication Modes

| Mode | Description |
| --- | --- |
| `alltoallv` | MPI `Alltoallv` with contiguous buffers. This is usually the safest collective path. |
| `alltoallw` | MPI `Alltoallw` / datatype path. This can fail for very large messages on some MPI stacks. |
| `p2p-waitall` | Peer-to-peer nonblocking sends/receives, completed with `Waitall`. |
| `p2p-waitany` | Peer-to-peer nonblocking sends/receives, unpacking received chunks as they complete. |

For large production measurements, use the C++ production preset or the
benchmark launcher’s `production` policy. The selected byte-transfer and
layout options differ by strategy and process count, so one manual flag list is
not valid for every 3D and 4D path.

Avoid `alltoallw` for very large sizes unless the MPI stack has been tested for
large derived datatypes and displacements.

## Common Binary Options

### FFTM 3D Benchmark

Binary:

```bash
test_benchmark_fftm_3D.bin
test_benchmark_fftm_3D_nca.bin
```

Usage:

```bash
mpiexec -n P ./test_benchmark_fftm_3D.bin \
  [--strategy slab-pencil|pencil-slab|pencil-pencil|all] \
  [--mode p2p-waitall|p2p-waitany|alltoallv|alltoallw] \
  [--grid P1 P2] \
  [--times repeats] [--warmup repeats] \
  [--epsilon eps] [--directory path] \
  [--use-direct-backward-receive|--no-direct-backward-receive] \
  [--direct-p2p-cuda-aware|--no-direct-p2p-cuda-aware] \
  [--use-p2p-send-thread|--no-p2p-send-thread] \
  [--use-p2p-byte-transfer|--no-p2p-byte-transfer] \
  [Nx Ny Nz]
```

### FFTM 4D Benchmark

Binary:

```bash
test_benchmark_fftm_4D.bin
test_benchmark_fftm_4D_nca.bin
```

Usage:

```bash
mpiexec -n P ./test_benchmark_fftm_4D.bin \
  [--strategy pencil-pencil|slab-slab|all] \
  [--mode p2p-waitall|p2p-waitany|alltoallv|alltoallw] \
  [--grid P1 P2 P3] \
  [--times repeats] [--warmup repeats] \
  [--epsilon eps] [--directory path] \
  [--use-direct-backward-receive|--no-direct-backward-receive] \
  [--direct-p2p-cuda-aware|--no-direct-p2p-cuda-aware] \
  [--use-p2p-byte-transfer|--no-p2p-byte-transfer] \
  [Nx Ny Nz Nw]
```

### FFTS 3D Benchmark

Binary:

```bash
test_benchmark_ffts_3D.bin
```

Usage:

```bash
./test_benchmark_ffts_3D.bin \
  [--times repeats] [--warmup repeats] \
  [--epsilon eps] [--directory path] \
  [Nx Ny Nz]
```

### FFTS 4D Benchmark

Binary:

```bash
test_benchmark_ffts_4D.bin
```

Usage:

```bash
./test_benchmark_ffts_4D.bin \
  [--strategy pencil-direct|pencil-memcpy|slab-direct|slab-memcpy|all] \
  [--times repeats] [--warmup repeats] \
  [--epsilon eps] [--directory path] \
  [Nx Ny Nz Nw]
```

### Option Notes

| Option | Meaning |
| --- | --- |
| `--times N` | Number of timed repetitions. Random data generation is outside the timed section in benchmark binaries. |
| `--warmup N` | Untimed warmup repetitions before the timed section. Use at least `3` for stable benchmark data. |
| `--epsilon EPS` | Accuracy tolerance for benchmark reconstruction checks. |
| `--threshold EPS` | Accuracy threshold used by comparison/versioned tests. |
| `--directory PATH` | Output directory for CSV files written by benchmark binaries. Note that some defaults still use the historical `./resutls` spelling. |
| `--grid ...` | Manually selects the MPI process grid. If omitted, tests choose a grid based on strategy and number of ranks. |
| `--use-direct-backward-receive` | Enables optional CUDA-aware direct backward receive paths where implemented. |
| `--direct-p2p-cuda-aware` | Allows CUDA-aware peer paths to receive directly into device-side targets where supported. |
| `--use-p2p-send-thread` | Diagnostic sender-thread experiment for eligible optimized 3D p2p paths; requires `MPI_THREAD_MULTIPLE` when active and is normalized off by `reference_parity`. |
| `--use-p2p-byte-transfer` | Enables eligible device-aware byte-transfer paths; the large-count transport controls datatype/count/chunk representation. This is MPI-stack-sensitive. |

## Examples and Tests

The numerical transform tests are in `source/tests/`. Reader-facing periodic
Poisson applications are documented in `examples/poisson/README.md`. The
complete Taylor-Green application and 4D analysis workflow is documented in
`examples/turbulence/README.md`.

| Test family | Files |
| --- | --- |
| MPI transposition checks | `test_mpi_transpose_3D.cu`, `test_mpi_transpose_4D.cu` |
| FFTM vs FFTS comparisons | `test_fftm_3D_compare.cu`, `test_fftm_4D_compare.cu` |
| Manufactured Poisson tests | `test_3D_poisson_mpi.cu`, `test_4D_poisson_mpi.cu` |
| Tutorial Poisson examples | `test_3D_poisson_mpi_tutorial.cu`, `test_4D_poisson_mpi_tutorial.cu` |
| C++ autotune caches | `test_fftm_autotune_hardware.cu`, `test_fftm_autotune_4d.cu`, `test_fftm_autotune_4d_runtime.cu` |
| Performance benchmarks | `test_benchmark_ffts_3D.cu`, `test_benchmark_ffts_4D.cu`, `test_benchmark_fftm_3D.cu`, `test_benchmark_fftm_4D.cu` |
| Versioned analytical tests | `test_ffts_v0_3D.cu` through `test_ffts_v4_4D.cu`, and `test_fftm_v0_3D.cu` through `test_fftm_v4_4D.cu` |

Typical development checks:

```bash
make -C source/tests test_fftm_3D_compare.bin test_fftm_4D_compare.bin -j4

mpiexec -n 2 ./source/tests/build/test_fftm_3D_compare.bin \
  --strategy slab-pencil --mode p2p-waitany 128 128 128

mpiexec -n 4 ./source/tests/build/test_fftm_4D_compare.bin \
  --strategy slab-slab --mode p2p-waitany 64 64 64 64
```

Reader-facing 3D/4D application smoke test:

```bash
make -C examples reader-smoke
```

### Final Paper Verification

The final verification launcher separates API correctness, single-node
production performance, and HCA-pinned multinode performance into independent
Slurm allocations:

```bash
FFTM_FINAL_CONTAINER_IMAGE=/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh \
FFTM_FINAL_DATA_DIR=/scratch/evstigneevnm/fftm/data_final_paper_$(date +%Y%m%d_%H%M%S) \
FFTM_FINAL_SALLOC_EXTRA_ARGS='--exclude=cn13' \
scripts/run_final_paper_verification.sh all
```

The targets can also be queued separately against the same data root:

```bash
export FFTM_FINAL_CONTAINER_IMAGE=/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh
export FFTM_FINAL_DATA_DIR=/scratch/evstigneevnm/fftm/data_final_paper_YYYYMMDD_HHMMSS
export FFTM_FINAL_SALLOC_EXTRA_ARGS='--exclude=cn13'

scripts/run_final_paper_verification.sh api
scripts/run_final_paper_verification.sh production
scripts/run_final_paper_verification.sh multinode
scripts/run_final_paper_verification.sh validate
```

`api` checks public presets, quiet defaults, the backend abstraction boundary,
small CUDA-aware and host-staged 3D/4D numerical transforms, hardware-aware
C++ autotuning, resource release, cache reuse, and the reader-facing 3D/4D
Poisson examples.
`production` runs the validated 6-8 GPU `2048^3` and `320^4` configurations.
`multinode` runs `2048^3` on the explicit `4x4` 3D grid and `320^4` on the 4D
slab-native path using 16 GPUs and HCA rank affinity. Each target writes a
short status file and a `PASSED` marker only after numerical, configuration,
telemetry, and regression-limit checks succeed.

Build the SQSH from the clean commit being verified. `make_docker.sh` records
the Git commit and tracked dirty state in `fftm_build_info.txt`. The launcher
checks the host checkout when Git metadata is available. On a cluster copy
without `.git`, it instead requires a valid embedded commit, requires the
embedded dirty state to be zero, and records the SQSH SHA-256. A plan can be
inspected without requesting an allocation by setting
`FFTM_FINAL_DRY_RUN=1 FFTM_FINAL_REQUIRE_CLEAN=0`.
The `all` target stops after the first failed target by default so an invalid
image cannot consume additional allocations. Set
`FFTM_FINAL_CONTINUE_ON_TARGET_FAILURE=1` only when later targets are known to
be independent and should still run.

Poisson tutorial timing example:

```bash
mpiexec -n 4 ./source/tests/build/test_3D_poisson_mpi_tutorial.bin \
  --strategy slab-pencil --mode p2p-waitany --times 100 --warmup 3 256 256 256
```

## Benchmark Data Collection

The benchmark scripts collect stdout logs, C++ CSV files, hardware metadata,
run matrices, and summary JSON files. The analysis script consumes that output.

### Local Native Runs

Use `scripts/run_local_paper_benchmarks.py` when binaries are available on the
local filesystem:

```bash
python3 scripts/run_local_paper_benchmarks.py \
  --tests-root source/tests/build \
  --data-directory build/paper_data/local_run \
  --gpu-counts 1,2 \
  --measure-times 10 \
  --warmup 3 \
  --extra-sizes-3d 540,686,729,900
```

### Local Docker Runs

Use `run_local_docker_fftm.sh` when testing the same container image intended
for the cluster:

```bash
FFTM_DOCKER_IMAGE=fftm/bench:v100-mpi \
FFTM_DATA_DIR=build/paper_data/docker_v100x2 \
FFTM_GPU_COUNTS=1,2 \
FFTM_BENCHMARK_TIMES=10 \
FFTM_WARMUP=3 \
./run_local_docker_fftm.sh
```

If Docker requires sudo and the NVIDIA runtime:

```bash
sudo FFTM_DOCKER_IMAGE=fftm/bench:v100-mpi \
  FFTM_DATA_DIR=build/paper_data/docker_v100x2 \
  ./run_local_docker_fftm.sh
```

If the Docker installation uses `--gpus all` without `--runtime=nvidia`:

```bash
DOCKER_RUNTIME=none FFTM_DOCKER_IMAGE=fftm/bench:v100-mpi \
  FFTM_DATA_DIR=build/paper_data/docker_v100x2 \
  ./run_local_docker_fftm.sh
```

### Slurm/Pyxis Cluster Runs

Use `scripts/run_slurm_pyxis_benchmarks.sh` on clusters where jobs run through
Slurm and NVIDIA Pyxis/enroot:

```bash
FFTM_CONTAINER_IMAGE=/path/to/fftm_bench_a100.sqsh \
FFTM_DATA_DIR=/scratch/$USER/fftm_cluster_data \
FFTM_NODE_COUNTS=1 \
FFTM_GPU_COUNTS=auto \
FFTM_GPUS_PER_NODE=8 \
FFTM_DEVICE_MEMORY_MIB=81920 \
FFTM_AUTO_MEMORY_FRACTION=0.80 \
FFTM_AUTO_REFERENCE_SIZE_3D=1200 \
FFTM_AUTO_REFERENCE_SIZE_4D=200 \
FFTM_BENCHMARK_TIMES=3 \
FFTM_WARMUP=3 \
FFTM_MODES=alltoallv,p2p-waitall,p2p-waitany \
FFTM_USE_DIRECT_BACKWARD_RECEIVE=0 \
FFTM_DIRECT_P2P_CUDA_AWARE=1 \
FFTM_USE_P2P_SEND_THREAD=0 \
FFTM_USE_P2P_BYTE_TRANSFER=0 \
scripts/run_slurm_pyxis_benchmarks.sh
```

For fixed-size strong scaling across GPU counts, add:

```bash
FFTM_FIXED_SCALING_SIZES_3D=2048
FFTM_FIXED_SCALING_SIZES_4D=256
```

After changing production configuration or transpose ownership, rebuild the
container and run the focused six-case guard:

```bash
FFTM_CONTAINER_IMAGE=/scratch/$USER/fftm_bench_a100.sqsh \
FFTM_DATA_DIR=/scratch/$USER/data_cleanup_prod_guard_$(date +%Y%m%d_%H%M%S) \
scripts/run_cleanup_production_guard.sh
```

It checks the validated 6/7/8-GPU 3D `2048^3` pencil configurations and 4D
`320^4` native-spectral slab configurations. Node `cn13` is excluded by
default and can be overridden with `FFTM_CLEANUP_SRUN_EXTRA_ARGS`.

### Orchestration Script Controls

`scripts/run_slurm_pyxis_benchmarks.sh`,
`scripts/run_container_local_benchmarks.sh`, and `run_local_docker_fftm.sh`
are controlled mainly through environment variables. The most important ones
are:

| Environment variable | Meaning |
| --- | --- |
| `FFTM_CONTAINER_IMAGE` | `.sqsh` image path or Pyxis image URI for Slurm/Pyxis runs. |
| `FFTM_DOCKER_IMAGE` | Docker image name for local container runs. |
| `FFTM_DATA_DIR` | Host directory where logs, CSV files, metadata, and summaries are written. |
| `FFTM_NODE_COUNTS` | Comma-separated node counts for cluster sweeps. |
| `FFTM_GPU_COUNTS` | GPU counts to test, or `auto` for dense sweeps up to the available maximum. |
| `FFTM_MAX_GPUS` | Upper bound for automatic GPU-count sweeps. |
| `FFTM_GPUS_PER_NODE` | Number of GPUs per node, usually `8` on A100 nodes. |
| `FFTM_DEVICE_MEMORY_MIB` | Device memory per GPU in MiB. May be a single value or per-GPU list for local runs. |
| `FFTM_GPU_NAME` | Label written to collected hardware/config metadata. |
| `FFTM_SRUN_TIME` | Slurm time limit passed to each `srun` step. |
| `FFTM_SRUN_EXTRA_ARGS` | Extra arguments appended to every `srun` command. |
| `FFTM_BENCHMARK_SIZES_3D` | Comma-separated 3D side lengths, `auto`, or `none`. |
| `FFTM_BENCHMARK_SIZES_4D` | Comma-separated 4D side lengths, `auto`, or `none`. |
| `FFTM_AUTO_MEMORY_FRACTION` | Fraction of device memory used by automatic size fitting. |
| `FFTM_AUTO_RESERVE_MEMORY_MIB` | Memory reserve subtracted from each GPU during automatic fitting. |
| `FFTM_AUTO_REFERENCE_SIZE_3D` | Measured one-GPU 3D reference side length for memory-scaled fitting. |
| `FFTM_AUTO_REFERENCE_SIZE_4D` | Measured one-GPU 4D reference side length for memory-scaled fitting. |
| `FFTM_AUTO_MIN_SIZE_3D`, `FFTM_AUTO_MAX_SIZE_3D` | Bounds for automatic 3D side-length selection. |
| `FFTM_AUTO_MIN_SIZE_4D`, `FFTM_AUTO_MAX_SIZE_4D` | Bounds for automatic 4D side-length selection. |
| `FFTM_EXTRA_SIZES_3D` | Extra 3D side lengths appended to selected fitted sizes. |
| `FFTM_EXTRA_SIZES_3D_BY_GPU` | Per-GPU-count extra 3D sizes, for example `1:1050;2:1344;4:1680`. |
| `FFTM_FIXED_SCALING_SIZES_3D` | Fixed 3D side lengths added to every FFTM GPU count for strong scaling. |
| `FFTM_FIXED_SCALING_SIZES_4D` | Fixed 4D side lengths added to every FFTM GPU count for strong scaling. |
| `FFTM_BENCHMARK_TIMES` | Timed benchmark iterations. |
| `FFTM_WARMUP` or `FFTM_BENCHMARK_WARMUP` | Untimed warmup iterations. |
| `FFTM_VALIDATION_TIMES` | Repetitions for validation/versioned tests. |
| `FFTM_MODES` | Communication modes to test, for example `alltoallv,p2p-waitall,p2p-waitany`. |
| `FFTM_TRANSPORTS` | `cuda_aware`, `non_cuda_aware`, or both. |
| `FFTM_INCLUDE_VERSIONED` | Set to `1` to run versioned analytical tests. |
| `FFTM_VERSIONED_FULL_MATRIX` | Set to `1` to run versioned FFTM tests for all selected modes. |
| `FFTM_USE_DIRECT_BACKWARD_RECEIVE` | Enables or disables direct backward receive optimization. |
| `FFTM_DIRECT_P2P_CUDA_AWARE` | Legacy compatibility name for direct device-aware p2p receive targets on CUDA or HIP. |
| `FFTM_USE_P2P_SEND_THREAD` | Requests the diagnostic sender-thread p2p experiment; not a production preset. |
| `FFTM_USE_P2P_BYTE_TRANSFER` | Enables or disables eligible device-aware byte-transfer variants; representation depends on the large-count transport. |
| `FFTM_P2P_VARIANTS` | `configured`, `all`, or a comma-separated subset of `value-packed,datatype-direct,byte-packed,byte-direct`. |
| `FFTM_TIMEOUT_SECONDS` | Per-command timeout used by the Python runner. |
| `FFTM_CUDA_ARCH` | CUDA architecture flags used when the local Docker helper builds an image. |
| `FFTM_BUILD_JOBS` | Build parallelism used by the local Docker helper. |

### Analysis

After data is collected:

```bash
python3 scripts/analyze_paper_results.py \
  --data-directory build/paper_data/local_run
```

Outputs are written to `<data-directory>/analysis` unless
`--output-directory` is provided. The analysis produces direct PDF/PNG figures,
CSV summaries, LaTeX tables, and a GitHub-oriented `README.md` inside the
analysis directory.

Useful figure formatting options:

```bash
python3 scripts/analyze_paper_results.py \
  --data-directory build/paper_data/local_run \
  --font-size 9 \
  --axis-font-size 8 \
  --legend-font-size 7 \
  --figure-width-cm 16 \
  --figure-height-cm 10
```

## Docker Image

For reader-facing reproduction of the 3D and 4D Poisson examples, see the
[CUDA/HIP Poisson capsule](Docker_config/poisson/README.md). It includes separate
Dockerfiles, rootless launch instructions, cache/correctness checks, retained
logs, and checksummed split image archives. Set `NGPU=N` to verify rank counts
1 through N on N local GPUs (`NGPU=1` by default); an explicit `--ranks` list
selects a subset. GPU validation currently covers one and two GPUs. It is independent of
the cluster benchmark image below.

The Dockerfile is `Docker_config/Dockerfile`. It builds all test binaries into
`/opt/fftm/bin` and copies the benchmark scripts into `/opt/fftm/scripts`.

Build the Docker image:

```bash
FFTM_DOCKER_TAG=fftm/bench:a100 ./make_docker.sh
```

The image build accepts these Docker build arguments through the Dockerfile:

| Build arg | Default | Meaning |
| --- | --- | --- |
| `CUDA_ARCH` | `-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70` | CUDA architectures used by `make`. |
| `BUILD_JOBS` | `8` | Parallel build jobs inside the image. |

To customize them manually:

```bash
sudo docker build -f Docker_config/Dockerfile . \
  -t fftm/bench:v100 \
  --build-arg CUDA_ARCH="-gencode arch=compute_70,code=sm_70" \
  --build-arg BUILD_JOBS=16
```

Run a quick shell in the container:

```bash
sudo docker run --rm --gpus all -v "$PWD:/data" fftm/bench:a100 bash
```

If your Docker installation requires the NVIDIA runtime explicitly:

```bash
sudo docker run --rm --runtime=nvidia --gpus all -v "$PWD:/data" fftm/bench:a100 bash
```

## Enroot/SquashFS Image

Create an enroot/Pyxis `.sqsh` image from a local Docker image:

```bash
FFTM_DOCKER_TAG=fftm/bench:a100 ./make_enroot.sh
```

By default the Docker tag is converted into a filesystem-safe name:

```text
fftm/bench:a100 -> fftm_bench_a100.sqsh
```

Override the output name:

```bash
FFTM_DOCKER_TAG=fftm/bench:a100 \
FFTM_SQSH_NAME=fftm_a100_latest \
./make_enroot.sh
```

The resulting `.sqsh` file is passed to Pyxis through `--container-image` or
through the `FFTM_CONTAINER_IMAGE` environment variable used by
`scripts/run_slurm_pyxis_benchmarks.sh`.

## Profiling

Both `ffts` and `fftm` include optional temporal and memory profiling. The
benchmark binaries print profiler summaries and memory categories by default.

Memory profiler categories include:

| Category | Meaning |
| --- | --- |
| `device` | Internal device allocations owned by FFTM/FFTS. |
| `host_pinned` | Internal pinned host buffers used mainly by non-CUDA-aware MPI paths. |
| `external_test` | Test-owned arrays tracked by benchmark drivers. |
| `other` | Any remaining tracked category. |

Temporal profiler entries separate local FFT time, MPI transpose time, packing,
posting, waiting, host staging, and unpacking where instrumentation exists.

## Notes for Large Runs

Use `--warmup 3` or higher for performance measurements. First-use CUDA and MPI
costs can otherwise distort effective GFLOP/s.

Prefer FFT-friendly problem sizes: powers of two or products of small primes
`2`, `3`, `5`, and `7`.

The following is a historical configured-transport control from the tested
A100 cluster, **not** the current universal production preset. In particular,
`reference_parity` normalizes byte transfer to `true`, so this list must not be
used to override a preset or a measured winner:

```bash
FFTM_USE_DIRECT_BACKWARD_RECEIVE=0
FFTM_DIRECT_P2P_CUDA_AWARE=1
FFTM_USE_P2P_SEND_THREAD=0
FFTM_USE_P2P_BYTE_TRANSFER=0
FFTM_P2P_VARIANTS=configured
```

On local systems where CUDA-aware MPI performs poorly, compare against `_nca.bin`
or set `FFTM_TRANSPORTS=cuda_aware,non_cuda_aware` in the orchestration scripts.

## License

This project is distributed under the license in `LICENSE`.
