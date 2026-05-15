#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Refactored for CUDA 13.2 / sm_132
# - Updated gencode flags from sm_131 to sm_132
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
                    # [CUDA 13.2 change] Target sm_132 architecture (next-gen after sm_131)
                    "-gencode", "arch=compute_132,code=sm_132",
                    "-gencode", "arch=compute_132,code=compute_132",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    # [CUDA 13.2 change] C++20 standard for modern CUDA features
                    "-std=c++20",
                    "-use_fast_math",
                ],
                "cxx": ["-std=c++20"],
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
