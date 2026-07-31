ARG FFTM_BASE_IMAGE=nvcr.io/nvidia/hpc-benchmarks:24.06

FROM ${FFTM_BASE_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCH="-gencode arch=compute_80,code=sm_80"

RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ make && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fftm_src
COPY source/contrib/scfd /opt/fftm_src/source/contrib/scfd
COPY experiments/cufft_node_hybrid /opt/fftm_src/experiments/cufft_node_hybrid

RUN make -C experiments/cufft_node_hybrid all \
        BUILD_DIR=/opt/fftm_node_hybrid/bin \
        CUDA_ARCH="${CUDA_ARCH}" && \
    test -x /opt/fftm_node_hybrid/bin/fftm_cufft_node_hybrid.bin && \
    test -x /opt/fftm_node_hybrid/bin/fftm_cufft_node_hybrid_4d.bin

FROM ${FFTM_BASE_IMAGE}

WORKDIR /opt/fftm_node_hybrid
COPY --from=build /opt/fftm_node_hybrid/bin /opt/fftm_node_hybrid/bin

ENV PATH="/opt/fftm_node_hybrid/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/mpi/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

CMD ["/bin/bash"]
