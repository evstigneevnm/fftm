FROM nvcr.io/nvidia/hpc-benchmarks:24.06 AS build

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
COPY scripts /opt/fftm_src/scripts
COPY README.md LICENSE /opt/fftm_src/

RUN make -C source/tests all \
        BUILD_FOLDER=/opt/fftm/bin \
        CUDA_ARCH="${CUDA_ARCH}" \
        -j"${BUILD_JOBS}" && \
    test -x /opt/fftm/bin/test_egger_pencil_comm_pattern_3D.bin

FROM nvcr.io/nvidia/hpc-benchmarks:24.06

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fftm
COPY --from=build /opt/fftm/bin /opt/fftm/bin
COPY scripts /opt/fftm/scripts
COPY README.md LICENSE /opt/fftm/

ENV PATH="/opt/fftm/bin:/opt/fftm/scripts:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/mpi/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

CMD ["/bin/bash"]
