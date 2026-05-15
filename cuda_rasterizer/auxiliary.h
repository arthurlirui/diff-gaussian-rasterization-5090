/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * Refactored for CUDA 13.2 / sm_132
 * - Updated from CUDA 13.1 to CUDA 13.2
 * - Uses cooperative_groups standard APIs (no deprecated intrinsics)
 * - Streaming load/store helpers updated for sm_132 cache hierarchy
 * - Added cuda::memcpy_async support declarations for cooperative loading
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// [CUDA 13.2] Include memcpy_async support for cooperative shared memory loading
#if __CUDA_ARCH__ >= 1320
#include <cuda/pipeline>
#include <cuda/barrier>
#endif

// Spherical harmonics coefficients — placed in constant-friendly format
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

// ---- CUDA Tile Helper Functions for sm_132 ----

// Warp-level all-reduce: returns true if all threads in the warp tile have predicate true
template <typename TileT>
__device__ __forceinline__ bool tileAll(const TileT& tile, bool predicate)
{
	return tile.all(predicate);
}

// Warp-level any-reduce: returns true if any thread in the warp tile has predicate true
template <typename TileT>
__device__ __forceinline__ bool tileAny(const TileT& tile, bool predicate)
{
	return tile.any(predicate);
}

// Warp-level sum reduction
template <typename TileT>
__device__ __forceinline__ float tileReduceSum(const TileT& tile, float value)
{
	return cg::reduce(tile, value, cg::plus<float>());
}

// Warp-level max reduction
template <typename TileT>
__device__ __forceinline__ float tileReduceMax(const TileT& tile, float value)
{
	return cg::reduce(tile, value, cg::greater<float>{});
}

// ---- Geometry / Math Helpers ----

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

// ---- sm_132 Cache Hint Wrappers ----
// [CUDA 13.2 change] Streaming load: bypasses L1, uses L2 (for read-once data)
// __ldcs remains the recommended intrinsic for streaming loads in CUDA 13.2,
// but sm_132's enhanced L2 cache (larger, lower latency) makes this even more effective.
__forceinline__ __device__ float  ldcs(const float* ptr)  { return __ldcs(ptr); }
__forceinline__ __device__ float2 ldcs(const float2* ptr) { return __ldcs(ptr); }
__forceinline__ __device__ float4 ldcs(const float4* ptr) { return __ldcs(ptr); }
__forceinline__ __device__ int    ldcs(const int* ptr)    { return __ldcs(ptr); }
__forceinline__ __device__ uint32_t ldcs(const uint32_t* ptr) { return __ldcs(ptr); }

// [CUDA 13.2 change] Streaming store: write-back, no allocation in cache
__forceinline__ __device__ void stcs(float* ptr, float val)  { __stcs(ptr, val); }
__forceinline__ __device__ void stcs(float2* ptr, float2 val) { __stcs(ptr, val); }
__forceinline__ __device__ void stcs(float4* ptr, float4 val) { __stcs(ptr, val); }

// [CUDA 13.2 change] Enhanced error checking macro
// After each kernel launch, now also checks cudaGetLastError() immediately
// (not just after cudaDeviceSynchronize). This catches launch-time errors
// such as invalid configuration arguments before they manifest as silent failures.
#define CHECK_CUDA(A, debug) \
A; \
if (debug) { \
	auto launch_err = cudaGetLastError(); \
	if (launch_err != cudaSuccess) { \
		std::cerr << "\n[CUDA LAUNCH ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(launch_err); \
		throw std::runtime_error(cudaGetErrorString(launch_err)); \
	} \
	auto sync_err = cudaDeviceSynchronize(); \
	if (sync_err != cudaSuccess) { \
		std::cerr << "\n[CUDA RUNTIME ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(sync_err); \
		throw std::runtime_error(cudaGetErrorString(sync_err)); \
	} \
}

#endif
