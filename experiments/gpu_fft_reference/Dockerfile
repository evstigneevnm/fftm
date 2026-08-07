ARG FFTM_BASE_IMAGE=nvcr.io/nvidia/hpc-benchmarks:24.06

FROM ${FFTM_BASE_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG GPU_FFT_REPOSITORY=https://github.com/Manthan-Verma/GPU_FFT.git
ARG GPU_FFT_COMMIT=ff4d84bbc6e0ecbcd52246c9d7e83198dadb2426
ARG CUDA_ARCH="-gencode arch=compute_80,code=sm_80"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates g++ git make && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/gpu_fft_upstream
RUN git init source && \
    cd source && \
    git remote add origin "${GPU_FFT_REPOSITORY}" && \
    git fetch --depth 1 origin "${GPU_FFT_COMMIT}" && \
    git checkout --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${GPU_FFT_COMMIT}" && \
    test -s LICENSE && \
    test -s GPU_FFT/src/GPU_FFT/GPU_FFT.h

WORKDIR /opt/gpu_fft_reference_src
COPY experiments/gpu_fft_reference /opt/gpu_fft_reference_src

RUN make all \
        GPU_FFT_DIR=/opt/gpu_fft_upstream/source/GPU_FFT \
        BUILD_DIR=/opt/gpu_fft_reference \
        GPU_FFT_UPSTREAM_COMMIT="${GPU_FFT_COMMIT}" \
        CUDA_ARCH="${CUDA_ARCH}" && \
    test -x /opt/gpu_fft_reference/bin/gpu_fft_reference_benchmark.bin && \
    test -s /opt/gpu_fft_reference/lib/libGPU_FFT.so

RUN mkdir -p /opt/gpu_fft_reference/share/licenses/GPU_FFT && \
    cp /opt/gpu_fft_upstream/source/LICENSE \
        /opt/gpu_fft_reference/share/licenses/GPU_FFT/LICENSE && \
    printf 'repository=%s\ncommit=%s\n' \
        "${GPU_FFT_REPOSITORY}" "${GPU_FFT_COMMIT}" \
        > /opt/gpu_fft_reference/share/upstream.env

FROM ${FFTM_BASE_IMAGE}

COPY --from=build /opt/gpu_fft_reference /opt/gpu_fft_reference
COPY scripts/run_mpi_rank_affinity.sh \
    /opt/gpu_fft_reference/bin/run_mpi_rank_affinity.sh

ENV PATH="/opt/gpu_fft_reference/bin:/usr/local/mpi/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/gpu_fft_reference/lib:/usr/local/mpi/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /opt/gpu_fft_reference

RUN test -x /opt/gpu_fft_reference/bin/gpu_fft_reference_benchmark.bin && \
    test -x /opt/gpu_fft_reference/bin/run_mpi_rank_affinity.sh && \
    test -s /opt/gpu_fft_reference/share/licenses/GPU_FFT/LICENSE && \
    /opt/gpu_fft_reference/bin/gpu_fft_reference_benchmark.bin --help >/dev/null

CMD ["/bin/bash"]
