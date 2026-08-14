#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PACKAGE=fftm_paper_performance_data_20260808
OUTPUT=${1:-"${ROOT}/assets/${PACKAGE}.tar.gz"}
STAGE=$(mktemp -d /tmp/fftm_paper_performance_assets.XXXXXX)
PACKAGE_ROOT="${STAGE}/${PACKAGE}"
DATA_ROOT="${PACKAGE_ROOT}/repo-data"
trap 'rm -rf "${STAGE}"' EXIT

mkdir -p "${DATA_ROOT}" "$(dirname "${OUTPUT}")"

copy_repo_file()
{
    local relative=$1
    if [[ ! -f "${ROOT}/${relative}" ]]; then
        echo "Missing paper-data input: ${relative}" >&2
        exit 1
    fi
    mkdir -p "${DATA_ROOT}/$(dirname "${relative}")"
    cp "${ROOT}/${relative}" "${DATA_ROOT}/${relative}"
}

copy_matches()
{
    local base=$1
    local pattern=$2
    local found=0
    while IFS= read -r path; do
        found=1
        copy_repo_file "${path#"${ROOT}/"}"
    done < <(find "${ROOT}/${base}" -type f -name "${pattern}" | sort)
    if [[ ${found} -eq 0 ]]; then
        echo "No paper-data inputs matched ${base}/${pattern}" >&2
        exit 1
    fi
}

# Exact inputs consumed by plot_final_performance_suite.py.
FINAL_INPUTS=(
    build/resutls_stats/data_final_multinode_paper_3d4d_complete_20260808_124136/paper_results_summary.csv
    build/resutls_stats/data_release_fitted_20260805_092652/benchmarks/cpp_csv/benchmark_fftm_3d.csv
    fft_Egger/results/egger_cluster_data_20260521_111434/measurements.csv
    build/resutls_stats/data_scale_hca_96_120_fixed_20260725_163251/analysis/hca_scaling_selected.csv
    build/resutls_stats/data_final_high_count_20260808_164725/selected_results.csv
    build/resutls_stats/data_gpu_fft_ref_full_20260807_093212/analysis/gpu_fft_reference_comparison.csv
)
for path in "${FINAL_INPUTS[@]}"; do
    copy_repo_file "${path}"
done

for item in \
    32:20260805_102745 \
    64:20260805_103730 \
    96:20260805_115614 \
    120:20260805_152130; do
    gpu=${item%%:*}
    stamp=${item#*:}
    for dimension in 3 4; do
        copy_repo_file \
            "build/resutls_stats/data_release_scale-${gpu}_${stamp}/benchmarks/cpp_csv/benchmark_fftm_${dimension}d.csv"
    done
done

copy_matches \
    build/resutls_stats/data_4d_weak_96_20260806_094507/full \
    benchmark_fftm_4d.csv
copy_matches \
    build/resutls_stats/data_4d_weak_120_20260806_145512/full \
    benchmark_fftm_4d.csv
copy_matches \
    build/resutls_stats/data_gpu_fft_ref_full_20260807_093212/raw \
    'c*_summary.csv'
copy_matches \
    build/resutls_stats/data_gpu_fft_ref_full_20260807_093212/raw \
    'c*_iterations.csv'

# Direct FFTW-MPI inputs used for the CPU/GPU comparison.
for campaign in \
    data_fftw_mpi_cpu_prod_1n2n_20260728_090353 \
    data_fftw_mpi_cpu_prod_4n_20260728_113419; do
    for name in benchmark_fftw_mpi_cpu.csv fftw_mpi_cpu_iterations.csv config.env status.csv; do
        copy_repo_file "build/resutls_stats/${campaign}/${name}"
    done
done

# Compact two-GPU HIP and host FFTW-MPI comparison inputs.
copy_matches \
    build/resutls_stats/data_hip_fftw_minimal_20260803_150054 \
    '*.csv'
for name in metadata.txt status.csv; do
    copy_repo_file "build/resutls_stats/data_hip_fftw_minimal_20260803_150054/${name}"
done

# Keep the curated data-only reference outputs for exact selection audits.
for name in \
    README.md \
    analysis_manifest.json \
    coverage_audit.csv \
    matched_external_comparisons.csv \
    remaining_cluster_cases.csv \
    selected_performance.csv \
    strong_candidate_audit.csv \
    table_best_strong_scaling.tex; do
    copy_repo_file "build/resutls_stats/paper_final_performance_20260808_complete/${name}"
done

# Small provenance records associated with the final high-count and summary sets.
for path in \
    build/resutls_stats/data_final_high_count_20260808_164725/git_commit.txt \
    build/resutls_stats/data_final_high_count_20260808_164725/git_status.txt \
    build/resutls_stats/data_final_high_count_20260808_164725/image.sha256 \
    build/resutls_stats/data_final_high_count_20260808_164725/launcher.sha256 \
    build/resutls_stats/data_final_high_count_20260808_164725/matrix.csv \
    build/resutls_stats/data_final_high_count_20260808_164725/status.csv \
    build/resutls_stats/data_final_multinode_paper_3d4d_complete_20260808_124136/source_manifest.json \
    build/resutls_stats/data_final_multinode_paper_3d4d_complete_20260808_124136/verification.json \
    build/resutls_stats/data_gpu_fft_ref_full_20260807_093212/analysis/manifest.json \
    build/resutls_stats/data_gpu_fft_ref_full_20260807_093212/config.env; do
    copy_repo_file "${path}"
done

analysis_commit=$(git -C "${ROOT}" rev-parse HEAD)
cat > "${PACKAGE_ROOT}/PROVENANCE.json" <<EOF
{
  "archive_schema": 1,
  "data_snapshot": "2026-08-08",
  "analysis_checkout_commit": "${analysis_commit}",
  "tested_sqsh_sha256": "717c426d90107c39323617b33d6edd34887237c90edc2c6ccf36c87ad36ebf4b",
  "metric": "complete forward/backward pair using maximum MPI-rank wall time",
  "effective_flops": "10 * total_points * log2(total_points) per transform pair",
  "analysis_scripts": [
    "scripts/plot_final_performance_suite.py",
    "scripts/analyze_fftw_mpi_cpu_comparison.py",
    "scripts/analyze_hip_fftw_comparison.py"
  ]
}
EOF

cat > "${PACKAGE_ROOT}/README.md" <<'EOF'
# FFTM paper performance data

This archive is a compact, data-only snapshot of the numerical inputs used for
the FFTM performance figures. Paths below `repo-data/` intentionally match the
paths expected by the tracked analysis scripts. The archive excludes generated
figures, raw cluster logs, container images, and turbulence field snapshots.

From an FFTM checkout, extract the archive anywhere and run:

```bash
DATA_ROOT=/path/to/fftm_paper_performance_data_20260808/repo-data
OUT=/tmp/fftm-paper-figures

python3 scripts/plot_final_performance_suite.py \
  --repo-root "$DATA_ROOT" \
  --output-dir "$OUT/final" \
  --dpi 600

python3 scripts/analyze_fftw_mpi_cpu_comparison.py \
  --cpu-data-dir "$DATA_ROOT/build/resutls_stats/data_fftw_mpi_cpu_prod_1n2n_20260728_090353" \
  --cpu-data-dir "$DATA_ROOT/build/resutls_stats/data_fftw_mpi_cpu_prod_4n_20260728_113419" \
  --gpu-selected-csv "$DATA_ROOT/build/resutls_stats/data_scale_hca_96_120_fixed_20260725_163251/analysis/hca_scaling_selected.csv" \
  --output-dir "$OUT/cpu-vs-gpu"

python3 scripts/analyze_hip_fftw_comparison.py \
  "$DATA_ROOT/build/resutls_stats/data_hip_fftw_minimal_20260803_150054" \
  --output-dir "$OUT/hip-vs-fftw"
```

The final suite writes six PDF/PNG figure sets for fitted 3D performance,
strong 3D and 4D scaling, weak 3D and 4D scaling, and same-system external
comparisons. `MANIFEST.sha256` covers every file in this archive except itself.
EOF

(
    cd "${PACKAGE_ROOT}"
    find . -type f ! -name MANIFEST.sha256 -print0 \
        | sort -z \
        | xargs -0 sha256sum > MANIFEST.sha256
)

tar \
    --sort=name \
    --mtime='UTC 2026-08-08 00:00:00' \
    --owner=0 --group=0 --numeric-owner \
    -C "${STAGE}" -cf - "${PACKAGE}" \
    | gzip -n -9 > "${OUTPUT}"

(
    cd "$(dirname "${OUTPUT}")"
    sha256sum "$(basename "${OUTPUT}")" > "$(basename "${OUTPUT}").sha256"
)
printf 'Wrote %s (%s)\n' "${OUTPUT}" "$(du -h "${OUTPUT}" | cut -f1)"
