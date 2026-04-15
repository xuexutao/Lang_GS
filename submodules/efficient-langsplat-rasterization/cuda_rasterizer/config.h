/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define NUM_CHANNELS_language_feature_BASE 64 // Base length (global codebook) per semantic level
// CUDA packed local-global (scheme-1): output per-pixel packed weight map once.
// packed channels = BASE * (1 + MAX_LOCAL_REGIONS)
#ifndef MAX_LOCAL_REGIONS
#define MAX_LOCAL_REGIONS 8
#endif
#define NUM_CHANNELS_language_feature_PACKED (NUM_CHANNELS_language_feature_BASE * (1 + MAX_LOCAL_REGIONS))
#define NUM_CHANNELS_quick_render 12 // Spare Coefficient Length of Three Semantic Levels 
#define BLOCK_X 16
#define BLOCK_Y 16

#endif
