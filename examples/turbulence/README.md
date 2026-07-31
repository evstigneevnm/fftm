# FFTM Taylor-Green turbulence example

This example evolves decaying incompressible Taylor-Green flow on the periodic
domain \([0,2\pi]^3\):

\[
\partial_t u = P(u \times \omega) + \nu \nabla^2u,\qquad
\omega=\nabla\times u,\qquad \nabla\cdot u=0,
\]

with \(\nu=1/Re\) and

\[
u_x=\sin x\cos y\cos z,\qquad
u_y=-\cos x\sin y\cos z,\qquad u_z=0.
\]

The implementation takes the initial condition and projection idea from the
Taylor-Green operator in `deflated_continuation`, but is an independent FFTM
application. The external project is not included or modified.

## Numerical method

- Fourier pseudospectral discretization with the rectangular two-thirds rule.
- Rotational nonlinear form, requiring three inverse velocity transforms,
  three inverse vorticity transforms, and three forward nonlinear transforms
  per right-hand-side evaluation.
- Three-stage Williamson low-storage RK3 with explicit diagonal diffusion.
- Optional adaptive CFL stepping. The controller reuses velocity already
  materialized by the first RK stage and adds no FFTs. Physical-time snapshot
  periods are hit exactly so the 4D temporal grid remains uniform.
- FFTM C++ cache selection for the 3D distributed transform.
- SCFD tensors, device iteration, reductions, memory copies, communicator
  operations, and MPI-IO. CUDA and cuFFT calls remain inside FFTM/SCFD
  abstractions.
- Snapshot-time \(Q\)-criterion evaluation using
  \(Q=-\tfrac12\operatorname{tr}[(\nabla u)^2]\). Velocity gradients are
  evaluated spectrally and accumulated through reusable SCFD work tensors.

The inverse FFTM operation is destructive. The solver therefore copies each
spectral component into a reusable SCFD work tensor before every inverse.

## Build

```bash
make -C examples taylor_green_3d_autotuned.bin
make -C examples taylor_green_spacetime_4d.bin
```

The default executable uses double precision. A single-precision binary can be
built explicitly:

```bash
make -C examples taylor_green_3d_autotuned_float.bin
```

## Local smoke test

The smoke test runs one and two MPI ranks on the locally visible GPU, validates
the analytic initial vorticity, and requires bitwise-identical distributed
snapshot files:

```bash
bash examples/turbulence/tests/run_local_smoke.sh
```

## Small run

```bash
FFTM_WRAP_PROCS_GPUS=1 mpiexec -n 2 \
  examples/build/taylor_green_3d_autotuned.bin \
  --size 128 \
  --reynolds 1600 \
  --dt 0.001 \
  --final-time 0.1 \
  --cache fftm_taylor_green_128.env \
  --output output_tg128 \
  --diagnostics-every 10
```

`diagnostics.csv` contains kinetic energy, enstrophy, maximum velocity,
divergence RMS, CFL number, and wall time. At \(t=0\), the implementation
checks \(E=1/8\) and the divergence-free constraint.

## Adaptive time stepping

Enable CFL-controlled stepping while preserving snapshots at exact physical
times:

```bash
FFTM_WRAP_PROCS_GPUS=1 mpiexec -n 2 \
  examples/build/taylor_green_3d_autotuned.bin \
  --size 128 \
  --reynolds 1600 \
  --dt 0.001 --dt-max 0.02 --dt-growth 1.1 \
  --target-cfl 0.35 --cfl-fail 0.4 \
  --final-time 10 \
  --snapshot-period 0.5 --viz-period 0.5 \
  --snapshot-size 128 --write-initial-snapshot 1 \
  --cache fftm_taylor_green_128.env \
  --output output_tg128_adaptive
```

`--dt` is the initial adaptive step, `--dt-max` is an optional upper bound,
and `--dt-growth` limits accepted-step growth. Use a target below
`--cfl-fail` to leave room for velocity growth over a complete RK step.
`adaptive_timesteps.csv` records every accepted step and controller CFL.
Do not combine `--snapshot-period` with `--snapshot-every`, or
`--viz-period` with `--viz-every`.

## Slurm/Pyxis runs

The focused launcher records the exact `srun` command in `config.env`. It uses
one MPI rank per GPU, rejects uneven rank placement, excludes `cn13` by
default, and enables the validated GPU-local HCA mapping automatically for
multi-node runs.

Run the 3D simulation:

```bash
FFTM_TURBULENCE_CONTAINER_IMAGE=/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh \
FFTM_TURBULENCE_DATA_DIR=/scratch/evstigneevnm/fftm/data_tg_512_$(date +%Y%m%d_%H%M%S) \
FFTM_TURBULENCE_NODES=1 \
FFTM_TURBULENCE_GPUS=8 \
FFTM_TURBULENCE_GPUS_PER_NODE=8 \
FFTM_TURBULENCE_SIZE=512 \
FFTM_TURBULENCE_REYNOLDS=1600 \
FFTM_TURBULENCE_DT=0.001 \
FFTM_TURBULENCE_TARGET_CFL=0.35 \
FFTM_TURBULENCE_CFL_FAIL=0.4 \
FFTM_TURBULENCE_DT_MAX=0.02 \
FFTM_TURBULENCE_DT_GROWTH=1.1 \
FFTM_TURBULENCE_FINAL_TIME=10 \
FFTM_TURBULENCE_STEPS=0 \
FFTM_TURBULENCE_DIAGNOSTICS_EVERY=1 \
FFTM_TURBULENCE_SNAPSHOT_SIZE=128 \
FFTM_TURBULENCE_SNAPSHOT_PERIOD=0.5 \
FFTM_TURBULENCE_VIZ_PERIOD=0.5 \
FFTM_TURBULENCE_SRUN_TIME=00:30:00 \
examples/turbulence/scripts/run_slurm_pyxis_turbulence.sh simulate
```

Run the 4D analysis after the requested number of signed-vorticity snapshots
has been collected:

```bash
FFTM_TURBULENCE_CONTAINER_IMAGE=/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh \
FFTM_TURBULENCE_INPUT_DIR=/scratch/evstigneevnm/fftm/data_tg_2048/simulation \
FFTM_TURBULENCE_DATA_DIR=/scratch/evstigneevnm/fftm/data_tg_4d_$(date +%Y%m%d_%H%M%S) \
FFTM_TURBULENCE_NODES=1 \
FFTM_TURBULENCE_GPUS=8 \
FFTM_TURBULENCE_GPUS_PER_NODE=8 \
FFTM_TURBULENCE_ANALYSIS_SIZE=512 \
FFTM_TURBULENCE_ANALYSIS_FRAMES=80 \
FFTM_TURBULENCE_MODE_MIN=1 \
FFTM_TURBULENCE_MODE_MAX=4 \
FFTM_TURBULENCE_WRITE_EVERY=10 \
FFTM_TURBULENCE_SRUN_TIME=01:00:00 \
examples/turbulence/scripts/run_slurm_pyxis_turbulence.sh analyze
```

Render a three-dimensional Q-criterion isosurface inside the SQSH:

```bash
FFTM_TURBULENCE_CONTAINER_IMAGE=/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh \
FFTM_TURBULENCE_SNAPSHOT=/scratch/evstigneevnm/fftm/data_tg_2048/simulation/q_criterion_s00005000.raw \
FFTM_TURBULENCE_DATA_DIR=/scratch/evstigneevnm/fftm/data_tg_q_render_$(date +%Y%m%d_%H%M%S) \
FFTM_TURBULENCE_RENDER_MODE=isosurface \
FFTM_TURBULENCE_RENDER_FIELD=q \
examples/turbulence/scripts/run_slurm_pyxis_turbulence.sh visualize
```

## Large run and 4D-analysis sampling

For a \(2048^3\) simulation, use `--snapshot-size 512`. This performs a
spectral low-pass to \(|k_i|\leq 512/3\) before exact factor-four decimation.
An arbitrary \(500^3\) target would require interpolation because 500 does not
divide 2048; it is therefore not the preferred scientific path.

The production workflow writes 80 signed-vorticity samples at a physical-time
interval of 0.2, including both \(t=0\) and \(t=15.8\). The temporal length 80
is FFT-friendly (\(80=2^4\cdot5\)) and covers the enstrophy growth, peak, and
decay. Visualization fields use a separate \(256^3\) output grid:

```text
--snapshot-period 0.2 --viz-period 0.5
--snapshot-size 512 --viz-snapshot-size 256
```

`omega_z` is the signed scalar intended for subsequent \(512^3\times80\) 4D
spatio-temporal FFT analysis. `vorticity_magnitude` and `q_criterion` are
written under `simulation/visualization` at each visualization checkpoint.
The latter is spectrally low-pass-filtered after its nonlinear gradient
products and before decimation.

Each field is one global float32 raw file written through SCFD MPI-IO. The
matching XDMF file can be opened directly in ParaView. The storage order is
x-fastest, and `layout.json` records the source resolution, filter cutoff, and
distributed ownership.

## 4D spatio-temporal filtering

The second executable reads uniformly spaced signed-vorticity snapshots,
subtracts the temporal mean, applies an RMS-normalized Hann window, and runs
the preferred FFTM 4D configuration:

- slab-slab, `p2p-waitany`;
- native `xzwy` spectral layout;
- direct, chunked XW transport with a bounded window and compact staging;
- plane-owned WZ communication scheduler with concurrency four.

It then retains a temporal Fourier band, optionally applies an additional
spatial cutoff, performs the inverse 4D transform, and writes selected
reconstructed frames:

```bash
FFTM_WRAP_PROCS_GPUS=1 mpiexec -n 8 \
  examples/build/taylor_green_spacetime_4d.bin \
  --input output_tg2048 \
  --output output_tg_4d_modes_1_4 \
  --field omega_z \
  --size 512 \
  --frames 80 \
  --mode-min 1 \
  --mode-max 4 \
  --write-every 10
```

`analysis.json` records the physical temporal-frequency band, retained
spectral-power fraction, preprocessing time, and forward/inverse 4D FFT times.
The reconstructed fields also include XDMF metadata.

### Local 4D validation

The analysis size must match the raw files in its input directory. To analyze a
smaller spatial grid, first form a new snapshot set with a Fourier low-pass and
spectral resampling; changing only `--size` is invalid. The preparation tool
selects a uniform time interval, applies a rectangular spatial cutoff, and
writes a self-contained snapshot directory:

```bash
python3 examples/turbulence/scripts/prepare_spacetime_subset.py \
  output_tg64 output_tg32_t2_13p5 \
  --field omega_z --frame-offset 4 --frames 24 --target-size 32
```

Run the complete local check in one command:

```bash
bash examples/turbulence/tests/run_local_spacetime_validation.sh \
  output_tg64 build/local_spacetime_validation
```

The harness performs two independent FFTM analyses on the same
\(32^3\times24\) input:

1. a no-window, full-temporal-band forward/inverse round trip;
2. an RMS-normalized Hann, temporal-mean-subtracted reconstruction retaining
   modes 1 through 3.

`plot_spacetime_analysis.py` compares the FFTM outputs against an independent
NumPy temporal-filter reference, checks the retained spectral-power fraction,
and writes 600-DPI PNG and vector PDF figures plus CSV/JSON metrics. The
selected default interval is \(t=2.0\) through \(13.5\), covering the
Taylor-Green enstrophy growth, peak, and decay.

### First cluster-scale 4D validation

The focused cluster pipeline evolves a \(512^3\), \(Re=1600\) Taylor-Green
case to \(t=15.75\), writes exactly 64 anti-aliased \(256^3\) signed-vorticity
snapshots at \(\Delta t=0.25\), and performs:

- a full-band \(256^3\times64\) round trip;
- a mean-subtracted, Hann-windowed modes 1-4 reconstruction;
- independent numerical validation and 600-DPI figures;
- the kinetic-energy/enstrophy literature comparison;
- 600-DPI vorticity-magnitude and positive-\(Q\) isosurfaces for every
  visualization snapshot, plus per-field contact sheets.

```bash
FFTM_TURBULENCE_CONTAINER_IMAGE=/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh \
FFTM_TURBULENCE_PIPELINE_ROOT=/scratch/evstigneevnm/fftm/data_tg512_4d256x64_$(date +%Y%m%d_%H%M%S) \
FFTM_TURBULENCE_USE_SINGLE_ALLOCATION=1 \
FFTM_TURBULENCE_ALLOCATION_TIME=08:00:00 \
examples/turbulence/scripts/run_512_256x64_cluster_validation.sh
```

The pipeline rejects missing, incorrectly sized, nonuniform, or incomplete
snapshot sets before starting the 4D transform. It uses eight A100 GPUs for
the simulation and transforms and one GPU allocation for in-container
plotting. In single-allocation mode, the plotting process retains the complete
eight-GPU GRES request because some Slurm configurations cannot subdivide a
job-level GPU allocation between steps; only one Python process is launched.

Resume a partially completed pipeline without repeating earlier stages by
setting `FFTM_TURBULENCE_PIPELINE_START_STAGE` to `2` through `6` and reusing
the original `FFTM_TURBULENCE_PIPELINE_ROOT`. Stage-4 through Stage-6 resumes
request only one GPU and 16 CPUs. When Stage 6 is added by rebuilding an older
pipeline image, set `FFTM_TURBULENCE_ALLOW_RESUME_IMAGE_MISMATCH=1`; this
exception is accepted only for Stage 6 and records the rendering image
checksum separately.

### Paper-scale 2048 run

The production pipeline evolves the \(2048^3\), \(Re=1600\) case to \(t=15.8\)
and performs a sparse-output \(512^3\times80\) full-band round trip and
modes-1-4 analysis. It defaults to 64 GPUs for the simulation and one
eight-GPU node for each 4D transform. The independent NumPy check traverses the
full \(512^3\times80\) field in bounded z-slabs and validates the eight written
frames without allocating a full host-side 4D array.

```bash
FFTM_TURBULENCE_CONTAINER_IMAGE=/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh \
FFTM_TURBULENCE_PIPELINE_ROOT=/scratch/evstigneevnm/fftm/data_tg2048_4d512x80_$(date +%Y%m%d_%H%M%S) \
examples/turbulence/scripts/run_2048_512x80_cluster_production.sh
```

Stages are independently queued and the result directory is resumable. Set
`FFTM_TURBULENCE_PIPELINE_STOP_STAGE=1` for the long simulation-only job.
Then set `FFTM_TURBULENCE_PIPELINE_START_STAGE=2`, keep the original
`FFTM_TURBULENCE_PIPELINE_ROOT`, and run Stages 2-6. Any contiguous stage range
from 1 through 6 can be selected this way. The pipeline checks the image
checksum, exact 80-frame time grid, payload sizes, sparse reconstruction count,
validation result, and rendered artifacts before reporting success.

## Paper visualization

Create three-dimensional vorticity isosurfaces:

```bash
python3 examples/turbulence/scripts/render_turbulence.py \
  output_tg/vorticity_magnitude_s00000500.raw \
  --output figures/tg_vorticity_isosurfaces.png \
  --mode isosurface --field-kind vorticity \
  --dpi 600 --width-inches 7.2 --height-inches 5.4
```

Create positive-\(Q\) isosurfaces:

```bash
python3 examples/turbulence/scripts/render_turbulence.py \
  output_tg/q_criterion_s00000500.raw \
  --output figures/tg_q_isosurfaces.png \
  --mode isosurface --field-kind q \
  --dpi 600 --width-inches 7.2 --height-inches 5.4
```

The renderer also supports `--mode volume` and `--mode slice`. Three-dimensional
renders include the \([0,2\pi]^3\) domain box and use a `0.8` camera zoom by
default. Use `--camera-zoom FACTOR` to change the framing or
`--no-domain-box` for an unframed close-up. The defaults are publication-sized
at 7.2 by 5.4 inches and 600 DPI, producing a 4320 by 3240 PNG with matching
DPI metadata. PyVista, VTK, Pillow, NumPy, SciPy, Matplotlib, and the EGL/OpenGL
runtime libraries are installed in an isolated environment inside the
benchmark SQSH. No Python installation is required on the cluster host. The
pinned visualization requirements remain in
`requirements-visualization.txt` for local use.

## Literature validation

Plot FFTM kinetic energy and enstrophy against up to three \(Re=1600\),
\(512^3\)-class literature datasets. The two redistributable datasets work
immediately after cloning the repository. Fetch the optional spectral reference
with:

```bash
python3 examples/turbulence/scripts/fetch_reference_data.py \
  --dataset hiocfd_spectral_512
```

Then generate the figures:

```bash
python3 examples/turbulence/scripts/plot_taylor_green_validation.py \
  output_tg/diagnostics.csv \
  --output-prefix figures/taylor_green_validation \
  --dpi 600
```

The command writes a 600-DPI PNG, vector PDF and SVG, a machine-readable
metrics CSV/JSON, and plot metadata. FFTM is a continuous line; literature
values are unconnected scatter markers. The default references are:

- the optional HiOCFD C3.5 dealiased pseudospectral \(512^3\) benchmark,
  downloaded from its authoritative host and validated by van Rees et al.
  (2011), DOI
  [10.1016/j.jcp.2010.11.031](https://doi.org/10.1016/j.jcp.2010.11.031);
- the OpenSBLI finite-difference \(512^3\) dataset, DOI
  [10.5258/SOTON/401892](https://doi.org/10.5258/SOTON/401892);
- the GALAEXI/FLEXI degree-7 DG result on \(64^3\) elements, DOI
  [10.18419/DARUS-4139](https://doi.org/10.18419/DARUS-4139).

All source locations, checksums, redistribution states, licenses, column
mappings, and normalization rules are recorded in
`reference_data/manifest.json`. Missing optional references are skipped unless
explicitly requested. FFTM and the references use
\(E=\langle|u|^2\rangle/2\) and
\(\Omega=\langle|\omega|^2\rangle/2\). For sources that report only
incompressible dissipation, the script applies
\(\Omega=Re\,\epsilon/2\).

The same plot can be generated inside the SQSH:

```bash
FFTM_TURBULENCE_CONTAINER_IMAGE=/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh \
FFTM_TURBULENCE_DIAGNOSTICS=/scratch/evstigneevnm/fftm/data_tg_2048/simulation/diagnostics.csv \
FFTM_TURBULENCE_DATA_DIR=/scratch/evstigneevnm/fftm/data_tg_validation_$(date +%Y%m%d_%H%M%S) \
FFTM_TURBULENCE_VALIDATION_DPI=600 \
examples/turbulence/scripts/run_slurm_pyxis_turbulence.sh validate
```

## Current scope

The solver supports fixed and adaptive CFL time stepping but does not yet
implement restart checkpoints. The 4D example performs band filtering and
reconstruction, but does not yet export a fully binned spatial/temporal power
spectrum. Those features can be added without changing the validated snapshot
contract.
