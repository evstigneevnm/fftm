ARG FFTM_BASE_IMAGE=nvcr.io/nvidia/hpc-benchmarks:24.06

FROM ${FFTM_BASE_IMAGE} AS build

ARG FFTM_BASE_IMAGE

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
    test -x /opt/fftm/bin/test_scfd_mpi_transport.bin && \
    test -x /opt/fftm/bin/test_4D_poisson_mpi.bin && \
    test -x /opt/fftm/bin/test_4D_poisson_mpi_tutorial.bin && \
    test -x /opt/fftm/bin/poisson_periodic_3d_autotuned.bin

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
    python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fftm
COPY --from=build /opt/fftm/bin /opt/fftm/bin
COPY scripts /opt/fftm/scripts
# Reference-only helper scripts. Uncomment when the image includes fftm3d.
# COPY fftm3d/scripts /opt/fftm/scripts
COPY README.md LICENSE /opt/fftm/

RUN test -x /opt/fftm/bin/test_scfd_mpi_transport.bin && \
    test -x /opt/fftm/scripts/run_scfd_mpi_transport_diagnostic.sh

ENV PATH="/opt/fftm/bin:/opt/fftm/scripts:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/mpi/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV FFTM_BASE_IMAGE="${FFTM_BASE_IMAGE}"

CMD ["/bin/bash"]
