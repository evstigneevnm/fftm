ARG FFTM_BASE_IMAGE=nvcr.io/nvidia/hpc-benchmarks:24.06

FROM ${FFTM_BASE_IMAGE} AS build

ARG FFTM_BASE_IMAGE
ARG FFTM_GIT_COMMIT=unknown
ARG FFTM_GIT_DIRTY=unknown

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCH="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70"
ARG BUILD_JOBS=8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    g++ \
    make \
    python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fftm_src
COPY source /opt/fftm_src/source
COPY examples /opt/fftm_src/examples
COPY scripts /opt/fftm_src/scripts
# Reference-only inputs. Uncomment these together with the reference build block
# below when rebuilding an image for Egger/fftm3d comparison runs.
# COPY fft_Egger/include /opt/fftm_src/fft_Egger/include
# COPY fft_Egger/src /opt/fftm_src/fft_Egger/src
# COPY fft_Egger/LICENSE /opt/fftm_src/fft_Egger/LICENSE
# COPY fftm3d /opt/fftm_src/fftm3d
COPY README.md LICENSE /opt/fftm_src/

RUN make -C source/tests test_benchmark_fftm_3D.bin test_benchmark_fftm_3D_nca.bin \
    test_benchmark_fftm_4D.bin test_benchmark_fftm_4D_nca.bin \
    test_fftm_options.bin test_fftm_quiet_default.bin \
    test_fftm_autotune_hardware.bin test_fftm_resource_lifecycle.bin \
    test_scfd_mpi_transport.bin \
    test_4D_poisson_mpi.bin test_4D_poisson_mpi_tutorial.bin \
    BUILD_FOLDER=/opt/fftm/bin \
    CUDA_ARCH="${CUDA_ARCH}" \
    -j"${BUILD_JOBS}" && \
    make -C examples all \
    BUILD_FOLDER=/opt/fftm/bin \
    CUDA_ARCH="${CUDA_ARCH}" \
    -j"${BUILD_JOBS}" && \
    test -x /opt/fftm/bin/test_benchmark_fftm_3D.bin && \
    test -x /opt/fftm/bin/test_benchmark_fftm_3D_nca.bin && \
    test -x /opt/fftm/bin/test_benchmark_fftm_4D.bin && \
    test -x /opt/fftm/bin/test_benchmark_fftm_4D_nca.bin && \
    test -x /opt/fftm/bin/test_fftm_options.bin && \
    test -x /opt/fftm/bin/test_fftm_quiet_default.bin && \
    test -x /opt/fftm/bin/test_fftm_autotune_hardware.bin && \
    test -x /opt/fftm/bin/test_fftm_resource_lifecycle.bin && \
    test -x /opt/fftm/bin/test_scfd_mpi_transport.bin && \
    test -x /opt/fftm/bin/test_4D_poisson_mpi.bin && \
    test -x /opt/fftm/bin/test_4D_poisson_mpi_tutorial.bin && \
    test -x /opt/fftm/bin/poisson_periodic_3d_autotuned.bin && \
    test -x /opt/fftm/bin/poisson_periodic_4d.bin && \
    test -x /opt/fftm/bin/taylor_green_3d_autotuned.bin && \
    test -x /opt/fftm/bin/taylor_green_spacetime_4d.bin && \
    printf 'git_commit=%s\ngit_dirty=%s\n' \
        "${FFTM_GIT_COMMIT}" "${FFTM_GIT_DIRTY}" \
        > /opt/fftm/bin/fftm_build_info.txt

# Reference/comparison image build. Uncomment together with the fft_Egger/fftm3d
# COPY lines above when running Egger or fftm3d A/B comparisons.
# RUN make -C source/tests all \
#         BUILD_FOLDER=/opt/fftm/bin \
#         CUDA_ARCH="${CUDA_ARCH}" \
#         -j"${BUILD_JOBS}" && \
#     test -x /opt/fftm/bin/test_egger_pencil_comm_pattern_3D.bin && \
#     make -C fftm3d all \
#         BUILD_FOLDER=/opt/fftm/bin \
#         CUDA_ARCH="${CUDA_ARCH}" \
#         -j"${BUILD_JOBS}" && \
#     make -C examples all \
#         BUILD_FOLDER=/opt/fftm/bin \
#         CUDA_ARCH="${CUDA_ARCH}" \
#         -j"${BUILD_JOBS}" && \
#     test -x /opt/fftm/bin/fftm3d_benchmark_3d.bin && \
#     test -x /opt/fftm/bin/fftm3d_benchmark_3d_wrapped.bin && \
#     test -x /opt/fftm/bin/poisson_periodic_3d_autotuned.bin

FROM ${FFTM_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG FFTM_BASE_IMAGE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libegl1 \
    libgl1 \
    libopengl0 \
    libxrender1 \
    python3 \
    python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fftm
COPY --from=build /opt/fftm/bin /opt/fftm/bin
COPY scripts /opt/fftm/scripts
COPY examples/turbulence/requirements-visualization.txt /opt/fftm/requirements-visualization.txt
COPY examples/turbulence/scripts/render_turbulence.py /opt/fftm/scripts/render_turbulence.py
COPY examples/turbulence/scripts/render_vorticity.py /opt/fftm/scripts/render_vorticity.py
COPY examples/turbulence/scripts/render_snapshot_series.py /opt/fftm/scripts/render_snapshot_series.py
COPY examples/turbulence/scripts/plot_taylor_green_validation.py /opt/fftm/scripts/plot_taylor_green_validation.py
COPY examples/turbulence/scripts/fetch_reference_data.py /opt/fftm/scripts/fetch_reference_data.py
COPY examples/turbulence/scripts/prepare_spacetime_subset.py /opt/fftm/scripts/prepare_spacetime_subset.py
COPY examples/turbulence/scripts/plot_spacetime_analysis.py /opt/fftm/scripts/plot_spacetime_analysis.py
COPY examples/turbulence/scripts/plot_spacetime_analysis_streaming.py /opt/fftm/scripts/plot_spacetime_analysis_streaming.py
COPY examples/turbulence/reference_data /opt/fftm/reference_data
# Reference-only helper scripts. Uncomment when the image includes fftm3d.
# COPY fftm3d/scripts /opt/fftm/scripts
COPY README.md LICENSE /opt/fftm/

RUN python3 -m venv /opt/fftm/visualization-venv && \
    /opt/fftm/visualization-venv/bin/python -m pip install \
        --no-cache-dir \
        --requirement /opt/fftm/requirements-visualization.txt && \
    PYVISTA_OFF_SCREEN=true /opt/fftm/visualization-venv/bin/python -c \
        "import pathlib, pyvista as pv; p=pv.Plotter(off_screen=True); p.add_mesh(pv.Sphere()); p.show(screenshot='/tmp/pyvista-smoke.png'); assert pathlib.Path('/tmp/pyvista-smoke.png').stat().st_size > 0" && \
    rm -f /tmp/pyvista-smoke.png

RUN test -x /opt/fftm/bin/test_scfd_mpi_transport.bin && \
    test -x /opt/fftm/bin/test_fftm_options.bin && \
    test -x /opt/fftm/bin/test_fftm_quiet_default.bin && \
    test -x /opt/fftm/bin/test_fftm_autotune_hardware.bin && \
    test -x /opt/fftm/bin/test_fftm_resource_lifecycle.bin && \
    test -x /opt/fftm/bin/poisson_periodic_3d_autotuned.bin && \
    test -x /opt/fftm/bin/poisson_periodic_4d.bin && \
    test -s /opt/fftm/bin/fftm_build_info.txt && \
    test -x /opt/fftm/scripts/run_scfd_mpi_transport_diagnostic.sh && \
    test -x /opt/fftm/scripts/run_cpp_autotune_validation.sh && \
    test -x /opt/fftm/bin/taylor_green_3d_autotuned.bin && \
    test -x /opt/fftm/bin/taylor_green_spacetime_4d.bin && \
    test -x /opt/fftm/scripts/run_mpi_rank_affinity.sh && \
    test -x /opt/fftm/scripts/render_turbulence.py && \
    test -x /opt/fftm/scripts/render_snapshot_series.py && \
    test -x /opt/fftm/scripts/plot_taylor_green_validation.py && \
    test -x /opt/fftm/scripts/fetch_reference_data.py && \
    test -x /opt/fftm/scripts/prepare_spacetime_subset.py && \
    test -x /opt/fftm/scripts/plot_spacetime_analysis.py && \
    test -x /opt/fftm/scripts/plot_spacetime_analysis_streaming.py && \
    /opt/fftm/visualization-venv/bin/python \
        /opt/fftm/scripts/render_snapshot_series.py --help >/dev/null && \
    /opt/fftm/visualization-venv/bin/python \
        /opt/fftm/scripts/fetch_reference_data.py --help >/dev/null && \
    /opt/fftm/visualization-venv/bin/python \
        /opt/fftm/scripts/prepare_spacetime_subset.py --help >/dev/null && \
    /opt/fftm/visualization-venv/bin/python \
        /opt/fftm/scripts/plot_spacetime_analysis.py --help >/dev/null && \
    /opt/fftm/visualization-venv/bin/python \
        /opt/fftm/scripts/plot_spacetime_analysis_streaming.py --help >/dev/null && \
    test -s /opt/fftm/reference_data/manifest.json && \
    test ! -e /opt/fftm/reference_data/hiocfd_spectral_512.gdiag

ENV PATH="/opt/fftm/visualization-venv/bin:/opt/fftm/bin:/opt/fftm/scripts:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/mpi/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV FFTM_BASE_IMAGE="${FFTM_BASE_IMAGE}"
ENV PYVISTA_OFF_SCREEN="true"

CMD ["/bin/bash"]
