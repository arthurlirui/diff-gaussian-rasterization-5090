/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * Refactored for CUDA 13.2 / sm_120
 * - Updated architecture targeting to sm_120 (RTX 5090 / Blackwell)
 * - Adjusted occupancy constants for sm_120's enhanced SM resources
 * - sm_120 features 128KB shared memory per SM and 65536 registers,
 *   allowing higher occupancy with larger shared memory footprints
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define WARP_SIZE 32
#define TILE_SIZE WARP_SIZE
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// sm_120 occupancy tuning (RTX 5090 / Blackwell)
// sm_120 has 128KB shared memory / SM and 65536 registers / SM,
// allowing up to 6 concurrent CTAs with this shared memory usage pattern.
#define MIN_CTAS_SM120 6

#endif
