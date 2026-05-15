#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Refactored for CUDA 13.2 / sm_132
# - Updated architecture from sm_131 to sm_132
# - C++20 standard for CUDA 13.2 compatibility
#

cmake_minimum_required(VERSION 3.24)

project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(CudaRasterizer
	cuda_rasterizer/backward.h
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.h
	cuda_rasterizer/forward.cu
	cuda_rasterizer/auxiliary.h
	cuda_rasterizer/rasterizer_impl.cu
	cuda_rasterizer/rasterizer_impl.h
	cuda_rasterizer/rasterizer.h
)

# [CUDA 13.2 change] Target sm_132 for CUDA 13.2 architecture
set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "132")

target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Enable extended lambda and relaxed constexpr for cooperative_groups
target_compile_options(CudaRasterizer PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:
	--expt-relaxed-constexpr
	--expt-extended-lambda
	-use_fast_math
>)
