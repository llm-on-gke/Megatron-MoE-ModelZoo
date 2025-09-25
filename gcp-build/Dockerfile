# Dockerfile for vLLM development
# Use a CUDA base image.
FROM docker.io/nvidia/cuda:12.9.1-devel-ubuntu22.04 AS base

WORKDIR /app

ENV CUDA_MAJOR=12
ENV CUDA_MINOR=9
ENV PYTHON_VERSION=3.12
ENV UCX_VERSION=1.19.0
ENV UCX_HOME=/opt/ucx
ENV CUDA_HOME=/usr/local/cuda/
ENV GDRCOPY_VERSION=2.4
ENV GDRCOPY_HOME=/usr/local
ENV NIXL_VERSION="0.5.0"
ENV NIXL_SOURCE_DIR=/opt/nixl
ENV NIXL_PREFIX=/usr/local/nixl
ENV NVSHMEM_VERSION=3.3.9
ENV NVSHMEM_PREFIX=/usr/local/nvshmem
ENV TORCH_CUDA_ARCH_LIST="9.0a 10.0"
ENV CMAKE_CUDA_ARCHITECTURES="90a;100"
# Work around https://github.com/vllm-project/vllm/issues/18859 and mount gIB if they
# are found for NCCL.
ENV LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}:

ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select New_York' | debconf-set-selections \
    && apt-get -qq update \
    && apt-get -qq install -y ccache software-properties-common git wget curl \
    && for i in 1 2 3; do \
        add-apt-repository -y ppa:deadsnakes/ppa && break || \
        { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
    done \
    # NVSHMEM
    #&& wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    #&& dpkg -i cuda-keyring_1.1-1_all.deb \
    # Mellanox OFED
    && wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add - \
    && cd /etc/apt/sources.list.d/ && wget https://linux.mellanox.com/public/repo/mlnx_ofed/24.10-0.7.0.0/ubuntu22.04/mellanox_mlnx_ofed.list \
    # Update all
    && apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
      # Python and related tools
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python${PYTHON_VERSION}-venv \
      python${PYTHON_VERSION}-dbg \
      ca-certificates \
      htop \
      iputils-ping net-tools \
      vim ripgrep bat clangd fuse fzf \
      nodejs npm clang fd-find xclip \
      zsh \
      # Build tools for UCX, NVSHMEM, etc.
      build-essential \
      autoconf automake libtool pkg-config \
      ninja-build cmake \
      # Other dependencies
      libnuma1 libsubunit0 libpci-dev libibverbs-dev \
      # MPI / PMIx / libfabric for NVSHMEM
      libopenmpi-dev openmpi-bin \
      libpmix-dev libfabric-dev \
      datacenter-gpu-manager \
      # Debugging tools
      kmod pciutils binutils \
      gdb strace lsof \
      # Should be included for GCP setup, uncomment if they go missing
      libnl-3-200 libnl-route-3-200 \
      # NVSHMEM - disabled due to link errors on DeepEP python package
      # nvshmem-cuda-${CUDA_MAJOR} \
      # Allow NVSHMEM to build nvshmem4py
      python3.10-venv python3.10-dev \
      # Mellanox OFED
      ibverbs-utils libibverbs-dev libibumad3 libibumad-dev librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \

    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && python${PYTHON_VERSION} -m ensurepip --upgrade \
    && python${PYTHON_VERSION} -m pip install --upgrade pip setuptools wheel

# --- Build and Install GDRCopy from Source ---
RUN apt-get update && apt-get install -y check
RUN cd /tmp && \
    git clone https://github.com/NVIDIA/gdrcopy.git && \
    cd gdrcopy && \
    git checkout tags/v${GDRCOPY_VERSION} && \
    make prefix=${GDRCOPY_HOME} lib_install exes_install && \
    ldconfig && \
    rm -rf /tmp/gdrcopy

ENV PATH=${GDRCOPY_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${GDRCOPY_HOME}/lib:${LD_LIBRARY_PATH}
ENV CPATH=${GDRCOPY_HOME}/include:${CPATH}
ENV LIBRARY_PATH=${GDRCOPY_HOME}/lib:${LIBRARY_PATH}

# --- Build and Install UCX from Source ---
RUN cd /tmp \
    && wget https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz \
    && tar -zxf ucx-${UCX_VERSION}.tar.gz \
    && cd ucx-${UCX_VERSION} \
    && ./contrib/configure-release      \
        --prefix=${UCX_HOME}            \
        --with-cuda=${CUDA_HOME}        \
        --with-gdrcopy=${GDRCOPY_HOME}  \
        --enable-shared         \
        --disable-static        \
        --disable-doxygen-doc   \
        --enable-optimizations  \
        --enable-cma            \ 
        --enable-devel-headers  \
        --with-verbs            \
        --with-dm               \ 
        --enable-mt             \
    && make -j$(nproc) && make install-strip \
    && rm -rf /tmp/ucx-${UCX_VERSION}*

ENV PATH=${UCX_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${UCX_HOME}/lib:${LD_LIBRARY_PATH}
ENV CPATH=${UCX_HOME}/include:${CPATH}
ENV LIBRARY_PATH=${UCX_HOME}/lib:${LIBRARY_PATH}
ENV PKG_CONFIG_PATH=${UCX_HOME}/lib/pkgconfig:${PKG_CONFIG_PATH}

# --- Build and Install NIXL from Source ---
# Grab meson from pip, since the 22.04 version of meson is not new enough.
RUN python${PYTHON_VERSION} -m pip install 'meson>=0.64.0' pybind11 \
    && cd /tmp \
    && wget "https://github.com/ai-dynamo/nixl/archive/refs/tags/${NIXL_VERSION}.tar.gz" \
        -O "nixl-${NIXL_VERSION}.tar.gz" \
    && mkdir -p ${NIXL_SOURCE_DIR} \
    && tar --strip-components=1 -xzf "nixl-${NIXL_VERSION}.tar.gz" -C ${NIXL_SOURCE_DIR} \
    && rm "nixl-${NIXL_VERSION}.tar.gz" \
    \
    # create an out-of-source build directory
    && mkdir -p ${NIXL_SOURCE_DIR}/build \
    && cd ${NIXL_SOURCE_DIR}/build \
    \
    # configure, compile, install
    && meson setup .. \
         --prefix=${NIXL_PREFIX} \
         -Dbuildtype=release \
    && ninja -j$(nproc) -C . \
    && ninja -j$(nproc) -C . install \
    # \
    # TODO: install wheel later
    # && cd .. \
    # source ${VIRTUAL_ENV}/bin/activate && \
    # python -m build --no-isolation --wheel -o /wheels && \
    # uv pip install --no-cache-dir . && \
    # rm -rf build
    # cleanup
    && rm -rf ${NIXL_SOURCE_DIR}/build

ENV LD_LIBRARY_PATH=${NIXL_PREFIX}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV NIXL_PLUGIN_DIR=${NIXL_PREFIX}/lib/x86_64-linux-gnu/plugins

# --- Prepare an NVSHMEM directory to support DeepEP compilation ---
# ENV NVSHMEM_DIR=${NVSHMEM_PREFIX}
# RUN mkdir -p "${NVSHMEM_DIR}/include" \
#     && cp -R /usr/include/nvshmem_${CUDA_MAJOR}/* "${NVSHMEM_DIR}/include/"
# TODO: Generates link errors like:
#       nvlink error   : Undefined reference to 'nvshmemi_ibgda_device_state_d' in '/app/deepep/build/temp.linux-x86_64-cpython-312/csrc/kernels/internode.o' (target: sm_100)

# --- Build and Install NVSHMEM from Source ---
ENV MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
ENV CPATH=${MPI_HOME}/include:${CPATH}
RUN export CC=/usr/bin/mpicc CXX=/usr/bin/mpicxx \
    && cd /tmp \
    && wget https://developer.download.nvidia.com/compute/redist/nvshmem/${NVSHMEM_VERSION}/source/nvshmem_src_cuda${CUDA_MAJOR}-all-all-${NVSHMEM_VERSION}.tar.gz \
    && tar -xzf nvshmem_src_cuda${CUDA_MAJOR}-all-all-${NVSHMEM_VERSION}.tar.gz \
    && cd nvshmem_src \
    && mkdir -p build \
    && cd build \
    && cmake \
      -G Ninja \
      -DNVSHMEM_PREFIX=${NVSHMEM_PREFIX} \
      -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
      -DNVSHMEM_PMIX_SUPPORT=0           \
      -DNVSHMEM_LIBFABRIC_SUPPORT=0      \
      -DNVSHMEM_IBRC_SUPPORT=1           \
      -DNVSHMEM_IBGDA_SUPPORT=1          \
      -DNVSHMEM_IBDEVX_SUPPORT=1         \
      -DNVSHMEM_SHMEM_SUPPORT=0          \
      -DNVSHMEM_USE_GDRCOPY=1            \
      -DNVSHMEM_USE_NCCL=0               \
      -DNVSHMEM_BUILD_TESTS=1            \
      -DNVSHMEM_BUILD_EXAMPLES=0         \
      -DNVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
      -DLIBFABRIC_HOME=/usr              \
      -DGDRCOPY_HOME=${GDRCOPY_HOME}     \
      -DNVSHMEM_MPI_SUPPORT=1            \
      -DNVSHMEM_DISABLE_CUDA_VMM=1       \
      .. \
    && ninja -j$(nproc) \
    && ninja -j$(nproc) install \
    && rm -rf /tmp/nvshmem_src*

ENV PATH=${NVSHMEM_PREFIX}/bin:${PATH}
ENV LD_LIBRARY_PATH=${NVSHMEM_PREFIX}/lib:${LD_LIBRARY_PATH}
ENV CPATH=${NVSHMEM_PREFIX}/include:${CPATH}
ENV LIBRARY_PATH=${NVSHMEM_PREFIX}/lib:${LIBRARY_PATH}
ENV PKG_CONFIG_PATH=${NVSHMEM_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="/usr/local/bin/" sh

# Install dependencies and NIXL (python)
COPY install-scripts/ /install-scripts/
RUN chmod +x /install-scripts/*.sh \
    && cd /install-scripts \
    && ./base-deps.sh

# For neovim.appimage
ENV APPIMAGE_EXTRACT_AND_RUN=1

# ENTRYPOINT ["/app/code/venv/bin/vllm", "serve"]

#==============================================================================

FROM base AS deepep

# Install specific versions
SHELL ["/bin/bash", "-ec"]
RUN DEEPEP_COMMIT=9af0e0d0e74     /install-scripts/deepep.sh 
    #&& DEEPGEMM_COMMIT=ea9c5d92   /install-scripts/deepgemm.sh 
    #&& /install-scripts/flashinfer.sh \
    #&& VLLM_USE_PRECOMPILED=0 MAX_JOBS=$(( "$(nproc)" * 3 / 4 )) /install-scripts/vllm.sh

WORKDIR /home/
#ENTRYPOINT ["/app/code/venv/bin/vllm", "serve"]
