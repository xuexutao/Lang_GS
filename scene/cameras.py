#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import pickle
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from typing import Optional

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # Lazy mmap cache for language features (avoid per-iter disk IO + meshgrid overhead).
        self._lf_cached_dir: Optional[str] = None
        self._lf_seg_mmap = None
        self._lf_feat_mmap = None
        self._lf_feat_gpu = None

    def get_language_feature(self, language_feature_dir, feature_level):
        language_feature_name = os.path.join(language_feature_dir, self.image_name)

        # Cache mmap handles (memory friendly; lets OS page-cache do the work).
        if self._lf_cached_dir != language_feature_dir or self._lf_seg_mmap is None or self._lf_feat_mmap is None:
            self._lf_cached_dir = language_feature_dir
            self._lf_seg_mmap = np.load(language_feature_name + '_s.npy', mmap_mode='r')
            self._lf_feat_mmap = np.load(language_feature_name + '_f.npy', mmap_mode='r')
            # feature_map is typically small; keep it on GPU for fast indexing.
            # NOTE: mmap buffer can be non-writable -> copy to avoid PyTorch warning/UB.
            self._lf_feat_gpu = torch.from_numpy(np.asarray(self._lf_feat_mmap).copy()).to(self.data_device)

        lvl = int(feature_level)
        if lvl < 0 or lvl > 3:
            raise ValueError("feature_level=", feature_level)

        # seg_map: [4, H, W] (with -1 for invalid)
        seg_lvl = torch.from_numpy(self._lf_seg_mmap[lvl]).to(self.data_device).long()
        mask = (seg_lvl != -1).unsqueeze(0)
        seg_safe = seg_lvl.clamp(min=0)

        # feature_map: [num_segments, 512] -> gather to [H, W, 512]
        point_feature = self._lf_feat_gpu[seg_safe]
        point_feature = point_feature.permute(2, 0, 1).contiguous()

        # Zero out invalid pixels to keep behavior consistent.
        point_feature = point_feature * mask
        return point_feature, mask

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
