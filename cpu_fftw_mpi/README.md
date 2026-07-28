# Direct FFTW MPI CPU Baseline

This project is intentionally independent of FFTM and SCFD. The executable calls
the FFTW MPI and FFTW threads APIs directly and measures an out-of-place real
forward transform followed by its matching inverse.

The forward plan uses `FFTW_MPI_TRANSPOSED_OUT`; the inverse uses
`FFTW_MPI_TRANSPOSED_IN`. The distributed real arrays use FFTW's required
last-dimension padding, including for these out-of-place transforms. Planning,
input generation, validation, and CSV output are excluded from measured
execution time.

Example:

```bash
mpiexec -n 2 ./fftw_mpi_cpu_benchmark.bin \
    --dimension 3 \
    --size 128 \
    --threads 4 \
    --planner measure \
    --planner-time-limit 120 \
    --warmup 2 \
    --times 10 \
    --output-dir results
```

The executable writes:

- `benchmark_fftw_mpi_cpu.csv`: one aggregate row per configuration.
- `fftw_mpi_cpu_iterations.csv`: individual forward, inverse, and pair samples.

`FFTW_MEASURE` can spend a long time planning very large transforms. The
cluster launcher passes a per-plan limit of 120 seconds by default through
`FFTW_CPU_PLANNER_TIME_LIMIT_SECONDS`. Planning time is recorded separately
and remains excluded from the measured transform time.

Build the dedicated image from the repository root:

```bash
FFTW_CPU_SQSH_NAME=fftm_fftw_mpi_cpu \
./build_fftw_mpi_image.sh
```

The image contains FFTW 3.3.10 and this benchmark only; it does not build or
link FFTM, SCFD, CUDA, cuFFT, or the diagnostic `fftm3d`/Egger code.

The launcher uses Slurm's `--cpus-per-task`/`--cpu-bind=cores` cpuset as the
rank boundary, but leaves OpenMP threads unbound inside that cpuset. Do not set
`OMP_PROC_BIND`, `OMP_PLACES`, or `GOMP_CPU_AFFINITY`: FFTW's pthread workers
inherit the calling thread's mask, and an OpenMP-bound initialization loop can
otherwise collapse the FFTW worker pool onto one core. The aggregate CSV
records both launch-time and post-OpenMP CPU-mask sizes.

Run the Slurm/Pyxis matrix with:

```bash
FFTW_CPU_CONTAINER_IMAGE=/scratch/evstigneevnm/fftm/fftm_fftw_mpi_cpu.sqsh \
FFTW_CPU_DATA_DIR=/scratch/evstigneevnm/fftm/data_fftw_mpi_cpu_$(date +%Y%m%d_%H%M%S) \
scripts/run_fftw_mpi_cpu_benchmarks.sh
```

When invoked outside Slurm, the launcher requests one shared allocation sized
for the largest configured node count and runs all matrix entries as sequential
job steps. Set `FFTW_CPU_USE_SINGLE_ALLOCATION=0` to restore independent
allocations. `FFTW_CPU_ALLOCATION_TIME` controls the shared allocation wall
limit, and `FFTW_CPU_SALLOC_EXTRA_ARGS` carries allocation constraints such as
`--exclude=cn13`.

After copying the data back, compare it with an existing selected FFTM scaling
table:

```bash
python3 scripts/analyze_fftw_mpi_cpu_comparison.py \
    --cpu-data-dir build/resutls_stats/data_fftw_mpi_cpu_TIMESTAMP \
    --gpu-selected-csv build/resutls_stats/data_scale_hca_TIMESTAMP/analysis/hca_scaling_selected.csv
```
