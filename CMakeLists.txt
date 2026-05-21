#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Refactored for CUDA 13.2 / sm_120
# - Targeting sm_120 architecture (RTX 5090 / Blackwell)
# - C++20 standard for CUDA 13.2 compatibility
#

cmake_minimum_required(VERSION 3.24)

project(DiffRast LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)

# CUDA should already be enabled by the parent project
# If not, find and enable it
if(NOT CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit REQUIRED)
    if(EXISTS "${CUDAToolkit_NVCC_EXECUTABLE}")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
    elseif(EXISTS "${CUDAToolkit_ROOT_DIR}/bin/nvcc.exe")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT_DIR}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
    enable_language(CUDA)
endif()

# Find PyTorch via python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Get torch include dirs from python
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(TORCH_CMAKE_PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
endif()

find_package(Torch REQUIRED)

add_library(CudaRasterizer
	cuda_rasterizer/backward.h
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.h
	cuda_rasterizer/forward.cu
	cuda_rasterizer/auxiliary.h
	cuda_rasterizer/config.h
	cuda_rasterizer/rasterizer_impl.cu
	cuda_rasterizer/rasterizer_impl.h
	cuda_rasterizer/rasterizer.h
	rasterize_points.h
	rasterize_points.cu
	ext.cpp
)

# Target sm_120 (RTX 5090 / Blackwell)
set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "120")

target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Link with torch and CUDA runtime
target_link_libraries(CudaRasterizer PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart)
target_include_directories(CudaRasterizer PRIVATE ${TORCH_INCLUDE_DIRS})

# Enable extended lambda and relaxed constexpr for cooperative_groups
target_compile_options(CudaRasterizer PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:
	--expt-relaxed-constexpr
	--expt-extended-lambda
	-use_fast_math
>
)

# MSVC: standard conforming preprocessor required by CUDA 13.2 CCCL
if(MSVC)
	target_compile_options(CudaRasterizer PRIVATE
		$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;/Zc:preprocessor>
	)
endif()
