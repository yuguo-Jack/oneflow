cmake_minimum_required(VERSION 3.18.0)
include(FetchContent)
project(oneflow_deps DESCRIPTION "this is used by Cargo")

set(CMAKE_DEBUG_POSTFIX "" CACHE STRING "" FORCE)

# glog
FetchContent_Declare(glog URL ${GLOG_URL} URL_HASH MD5=${GLOG_HASH})
set(WITH_UNWIND OFF CACHE BOOL "" FORCE)
set(WITH_GFLAGS OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(WITH_GTEST OFF CACHE BOOL "")
FetchContent_MakeAvailable(glog)

# protobuf
set(PROTOBUF_URL "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip")
set(PROTOBUF_MD5 cf02c32870a1f78c860039e0f63a6343)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "")
set(protobuf_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "")
set(protobuf_BUILD_TESTS OFF CACHE BOOL "")
set(protobuf_DEBUG_POSTFIX "" CACHE STRING "")
FetchContent_Declare(protobuf URL ${PROTOBUF_URL} URL_HASH MD5=${PROTOBUF_MD5} SOURCE_SUBDIR cmake)
FetchContent_MakeAvailable(protobuf)

# half

set(HALF_URL https://github.com/Oneflow-Inc/half/archive/v2.1.0-fix-cuda-raise.zip)
set(HALF_URL_HASH 30b0dc289729f9e85ddf6995f2e6968f)
FetchContent_Declare(half URL ${HALF_URL} URL_HASH MD5=${HALF_URL_HASH} SOURCE_SUBDIR cmake)
FetchContent_MakeAvailable(half)
install(
FILES ${half_SOURCE_DIR}/include/half.hpp
TYPE INCLUDE
COMPONENT half_headers)
add_custom_target(
install_half_headers ALL
COMMAND
    "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=half_headers
    -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
DEPENDS ${half_SOURCE_DIR}/include/half.hpp)
