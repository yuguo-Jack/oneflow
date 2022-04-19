set -e
earthly \
    --build-arg BASE_IMAGE=registry.cn-beijing.aliyuncs.com/oneflow/manylinux2014_x86_64_cuda10.2 \
    --build-arg ONEFLOW_CI_CMAKE_INIT_CACHE=cmake/caches/ci/release/cuda.cmake \
    +build
