# Release data assets

`fftm_paper_performance_data_20260808.tar.gz` contains the compact numerical
inputs used by FFTM's paper-performance plotting scripts. It intentionally
contains CSV data and provenance metadata, not generated figures, cluster raw
logs, container images, or the journal manuscript.

Verify it before extraction:

```bash
sha256sum -c assets/fftm_paper_performance_data_20260808.tar.gz.sha256
```

The archive's own `README.md` gives commands for regenerating the final CUDA,
FFTW-MPI, and HIP comparison figures. Rebuild the archive from the ignored
local result collection with:

```bash
scripts/build_paper_performance_assets.sh
```
