#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Refactored for CUDA 13.2 / sm_120
# - Updated gencode flags to sm_120 (RTX 5090 / Blackwell)
# - Added -std=c++20 for CUDA 13.2 compatibility
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "nvcc": [
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                    # Target sm_120 architecture (RTX 5090 / Blackwell)
                    "-gencode", "arch=compute_120,code=sm_120",
                    "-gencode", "arch=compute_120,code=compute_120",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    # C++20 standard for modern CUDA features
                    "-std=c++20",
                    "-use_fast_math",
                    # CUDA 13.2 CCCL requires MSVC standard-conforming preprocessor
                    "-Xcompiler", "/Zc:preprocessor",
                ],
                "cxx": ["/std:c++20", "/Zc:preprocessor"],
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
