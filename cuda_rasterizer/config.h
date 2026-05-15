/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * Refactored for CUDA 13.2 / sm_132
 * - Updated architecture targeting from sm_131 to sm_132
 * - Adjusted occupancy constants for sm_132's enhanced SM resources
 * - sm_132 features 128KB shared memory per SM and 65536 registers,
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

// sm_132 occupancy tuning
// [CUDA 13.2 change] sm_132 has 128KB shared memory / SM and 65536 registers / SM,
// allowing up to 6 concurrent CTAs with this shared memory usage pattern.
// Increased from MIN_CTAS=4 on sm_131 to MIN_CTAS=6 on sm_132.
#define MIN_CTAS_SM132 6

// Backward compatibility alias (deprecated, use MIN_CTAS_SM132)
#define MIN_CTAS_SM131 MIN_CTAS_SM132

#endif
