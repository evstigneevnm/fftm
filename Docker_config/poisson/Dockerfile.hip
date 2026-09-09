ARG BUILDER_IMAGE=rocm/dev-ubuntu-22.04:6.4.1-complete@sha256:a3850e6638c6c390436ef1aacd72fd1359af36083ac823d5136818206998c484
ARG RUNTIME_IMAGE=ubuntu:22.04@sha256:2edbbc5dc405e9612ba3584ce95480277e3eb374407b5505fe26f17df77c7dbc
FROM ${BUILDER_IMAGE} AS builder
ARG BUILD_JOBS=2
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    ca-certificates python3 pkg-config libnuma-dev libelf-dev autoconf automake libtool \
    flex bison && rm -rf /var/lib/apt/lists/*
COPY deps/ /deps/
COPY src/Docker_config/poisson/build_mpi.sh /build_mpi.sh
RUN bash /build_mpi.sh hip
# Required only for the multi-architecture rocFFT rebuild.
RUN apt-get update && apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/mpi/bin:/opt/ucx/bin:/opt/rocm/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/mpi/lib:/opt/ucx/lib:/opt/rocm/lib:/opt/amdgpu/lib/x86_64-linux-gnu
COPY src/build_configs/common.mk /opt/fftm/src/build_configs/common.mk
COPY src/Docker_config/poisson/Makefile.probe src/Docker_config/poisson/config_hip.inc \
    src/Docker_config/poisson/build_rocfft.sh /opt/fftm/src/Docker_config/poisson/
RUN make -f /opt/fftm/src/Docker_config/poisson/Makefile.probe \
    CONFIG_FILE=Docker_config/poisson/config_hip.inc rocfft
COPY src/ /opt/fftm/src/
RUN bash /opt/fftm/src/Docker_config/poisson/build_examples.sh hip && \
    bash /opt/fftm/src/Docker_config/poisson/collect_hip_runtime.sh

FROM ${RUNTIME_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates \
    libnuma1 libelf1 libgomp1 libdrm2 libdrm-amdgpu1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /hip-runtime/ /
COPY --from=builder /opt/mpi /opt/mpi
COPY --from=builder /opt/ucx /opt/ucx
COPY --from=builder /opt/fftm /opt/fftm
ENV PATH=/opt/mpi/bin:/opt/ucx/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/mpi/lib:/opt/ucx/lib:/opt/rocm/lib:/opt/amdgpu/lib/x86_64-linux-gnu
ENV ROCM_PATH=/opt/rocm
# Portable GPU-buffer MPI for Radeon hosts without working cross-GPU IPC.
ENV CAPSULE_UCX_TLS=self,sm,tcp,rocm_copy
WORKDIR /opt/fftm
RUN sha256sum -c /opt/fftm/metadata/binaries.sha256 && \
    python3 /opt/fftm/src/Docker_config/poisson/capsule.py info && \
    dpkg-query -W > /opt/fftm/metadata/runtime-packages.txt
ENTRYPOINT ["python3", "/opt/fftm/src/Docker_config/poisson/capsule.py"]
CMD ["--help"]
