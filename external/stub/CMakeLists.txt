cmake_minimum_required(VERSION 3.18.0)
include(FetchContent)
project(oneflow_deps DESCRIPTION "this is used by Cargo")

FetchContent_Declare(glog URL ${GLOG_URL} URL_HASH MD5=${GLOG_HASH})
set(WITH_UNWIND OFF CACHE BOOL "" FORCE)
set(WITH_GFLAGS OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(WITH_GTEST OFF CACHE BOOL "")
FetchContent_MakeAvailable(glog)

set(PROTOBUF_URL "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip")
set(PROTOBUF_MD5 cf02c32870a1f78c860039e0f63a6343)

set(protobuf_WITH_ZLIB OFF CACHE BOOL "")
set(protobuf_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "")
set(protobuf_BUILD_TESTS OFF CACHE BOOL "")
set(protobuf_DEBUG_POSTFIX "" CACHE STRING "")

FetchContent_Declare(protobuf URL ${PROTOBUF_URL} URL_HASH MD5=${PROTOBUF_MD5})
FetchContent_MakeAvailable(protobuf)
