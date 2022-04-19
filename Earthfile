VERSION 0.5
ARG BASE_IMAGE=oneflowinc/manylinux2014_x86_64_cuda11.2
ARG CMAKE_INIT_CACHE=cmake/caches/international/cuda.cmake
FROM ${BASE_IMAGE}

WORKDIR /code
ENV ONEFLOW_CI_PYTHON_EXE=/opt/python/cp37-cp37m/bin/python3
code:
  COPY . src

build:
  FROM +code
  WORKDIR /code/src
  RUN cmake -B /code/build -C ${CMAKE_INIT_CACHE} -DPython3_EXECUTABLE=${ONEFLOW_CI_PYTHON_EXE}
  WORKDIR /code
  # cache cmake temp files to prevent rebuilding .o files
  # when the .cpp files don't change
  RUN --mount=type=cache,target=/code/build/CMakeFiles ninja
  RUN auditwheel repair src/python/dist/* --wheel-dir wheelhouse
  SAVE ARTIFACT wheelhouse AS LOCAL wheelhouse
