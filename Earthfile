VERSION 0.5
ARG BASE_IMAGE=oneflowinc/manylinux2014_x86_64_cuda11.2
ARG ONEFLOW_CI_CMAKE_INIT_CACHE=cmake/caches/international/cuda.cmake
ARG ONEFLOW_CI_BUILD_SCRIPT=ci/manylinux/build-gcc7.sh
ARG ONEFLOW_CI_PYTHON_EXE=/opt/python/cp37-cp37m/bin/python3
ARG ONEFLOW_CI_BUILD_PARALLEL=8
FROM ${BASE_IMAGE}

WORKDIR /code
ENV ONEFLOW_CI_SRC_DIR=/code/src
ENV ONEFLOW_CI_BUILD_DIR=/code/build

code:
  COPY . src

build:
  FROM +code
  # cache cmake temp files to prevent rebuilding .o files
  # when the .cpp files don't change
  WORKDIR /code/src
  RUN --mount=type=cache,target=/code/build/CMakeFiles \
      --mount=type=cache,target=/root/.ccache \
      --mount=type=cache,target=/root/.local \
      --mount=type=cache,target=/root/.cache \
     cmake -B /code/build -C ${ONEFLOW_CI_CMAKE_INIT_CACHE} -DPython3_EXECUTABLE=${ONEFLOW_CI_PYTHON_EXE}
  RUN --mount=type=cache,target=/code/build/CMakeFiles \
      --mount=type=cache,target=/root/.ccache \
      --mount=type=cache,target=/root/.local \
      --mount=type=cache,target=/root/.cache \
     cmake --build /code/build --parallel ${ONEFLOW_CI_BUILD_PARALLEL}
  RUN auditwheel repair python/dist/* --wheel-dir wheelhouse
  SAVE ARTIFACT wheelhouse AS LOCAL wheelhouse
