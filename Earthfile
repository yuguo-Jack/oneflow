VERSION 0.5
ARG BASE_IMAGE=oneflowinc/manylinux2014_x86_64_cuda11.2
ARG ONEFLOW_CI_CMAKE_INIT_CACHE=cmake/caches/international/cuda.cmake
ARG ONEFLOW_CI_BUILD_SCRIPT=ci/manylinux/build-gcc7.sh
FROM ${BASE_IMAGE}

WORKDIR /code
ENV ONEFLOW_CI_PYTHON_EXE=/opt/python/cp37-cp37m/bin/python3
ENV ONEFLOW_CI_SRC_DIR=/code/src
ENV ONEFLOW_CI_BUILD_DIR=/code/build
ENV ONEFLOW_CI_CMAKE_INIT_CACHE=${ONEFLOW_CI_CMAKE_INIT_CACHE}

code:
  COPY . src

build:
  FROM +code
  # cache cmake temp files to prevent rebuilding .o files
  # when the .cpp files don't change
  WORKDIR /code/src
  RUN --mount=type=cache,target=/code/build/CMakeFiles bash ${ONEFLOW_CI_BUILD_SCRIPT}
  RUN auditwheel repair python/dist/* --wheel-dir wheelhouse
  SAVE ARTIFACT wheelhouse AS LOCAL wheelhouse
