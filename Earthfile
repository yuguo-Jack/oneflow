VERSION 0.5
FROM registry.cn-beijing.aliyuncs.com/oneflow/manylinux2014_x86_64_cuda10.2

WORKDIR /code
ENV ONEFLOW_CI_PYTHON_EXE=/opt/python/cp37-cp37m/bin/python3
code:
  COPY . src

build:
  FROM +code
  RUN cmake -B build -C cmake/caches/ci/release/cuda.cmake -DPython3_EXECUTABLE=${ONEFLOW_CI_PYTHON_EXE} src
  # cache cmake temp files to prevent rebuilding .o files
  # when the .cpp files don't change
  RUN --mount=type=cache,target=/code/build/CMakeFiles ninja
  RUN auditwheel repair src/python/dist/* --wheel-di wheelhouse
  SAVE ARTIFACT wheelhouse AS LOCAL wheelhouse
