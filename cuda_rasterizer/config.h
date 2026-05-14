/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * Refactored for CUDA 13.1 / sm_131 with CUDA Tile support
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

// sm_131 occupancy tuning
#define MIN_CTAS_SM131 4

#endif
