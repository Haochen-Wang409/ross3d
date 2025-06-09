#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import math
import re
import time
import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from einops import rearrange
from copy import deepcopy

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import (
    build_vision_projector,
    build_inv_projector,
)
from .pixel_decoder.builder import build_pixel_decoder

from ross3d.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from ross3d.mm_utils import get_anyres_image_grid_shape
from ross3d.utils import rank0_print, rank_print
import random


class Ross3DMetaModel:

    def __init__(self, config):
        super(Ross3DMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            self.mask_token = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
        
        if hasattr(self.config, 'world_position_embedding_type'):
            from ross3d.model.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingMLP

            if "sample9" in self.config.world_position_embedding_type:
                n_points = 9
            elif "sample5" in self.config.world_position_embedding_type:
                n_points = 5
            elif "minmax" in self.config.world_position_embedding_type:
                n_points = 2
            else:
                n_points = 1
        
            if "mlp" in self.config.world_position_embedding_type:
                self.world_position_embedding = PositionEmbeddingMLP(config.hidden_size, n_points=n_points)
            elif "sin3d" in self.config.world_position_embedding_type:
                self.world_position_embedding = PositionEmbeddingSine3D(config.hidden_size, n_points=n_points)
            # elif "slp" in self.config.world_position_embedding_type:
            #     self.world_position_embedding = PositionEmbeddingSine3DMLP(config.hidden_size, n_points=n_points)

        # self.mm_inv_projector = build_inv_projector(self.config)
        # self.mm_pixel_decoder = build_pixel_decoder(self.config)
        # # other necessary information for reconstruction
        # self.image_embed_len = math.ceil(
        #     (self.vision_tower.config.image_size // self.vision_tower.config.patch_size)
        #     / float(self.config.mm_spatial_pool_stride)) ** 2


    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        # for pixel_decoder
        mm_pixel_decoder = model_args.mm_pixel_decoder
        pretrain_mm_inv_adapter = model_args.pretrain_mm_inv_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")
        self.config.mm_pixel_decoder = mm_pixel_decoder

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.image_embed_len = math.ceil(
            (self.vision_tower.config.image_size // self.vision_tower.config.patch_size)
            / float(model_args.mm_spatial_pool_stride)) ** 2
        self.config.image_embed_len = self.image_embed_len
        self.config.image_mean = self.vision_tower.image_processor.image_mean
        self.config.image_std = self.vision_tower.image_processor.image_std
        self.config.decode_image_size = self.vision_tower.config.image_size // self.vision_tower.config.patch_size * 8  # 336 -> 192; 384 -> 216

        ### build CLIP-LLM projector
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
            self.mask_token = nn.Parameter(torch.randn(config.hidden_size, dtype=self.dtype) * embed_std,
                                           requires_grad=True)

            if "unpad" in mm_patch_merge_type:
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std,
                                                  requires_grad=True)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

        self.config.ross_enable = False
        if getattr(model_args, 'mm_pixel_decoder', False):
            self.config.ross_enable = True
            ### build pixel decoder
            self.mm_pixel_decoder = build_pixel_decoder(self.config)
            self.config.mm_inv_hidden_size = self.mm_pixel_decoder.latent_dim

            ### build LLM-CLIP projector
            self.config.use_mm_inv_proj = True
            self.config.mm_inv_projector_type = getattr(model_args, 'mm_inv_projector_type', 'linear')

            if getattr(self, 'mm_inv_projector', None) is None:
                self.mm_inv_projector = build_inv_projector(self.config)
            else:
                # In case it is frozen by LoRA
                for p in self.mm_inv_projector.parameters():
                    p.requires_grad = True

            if pretrain_mm_inv_adapter is not None:
                rank0_print(f"=> loading pretrain_mm_inv_adapter from {pretrain_mm_inv_adapter} ...")
                mm_inv_projector_weights = torch.load(pretrain_mm_inv_adapter, map_location='cpu')

                def get_w(weights, keyword):
                    new_weights = {}
                    for k, v in weights.items():
                        if keyword in k:
                            new_k = k.split(keyword + '.')[1]
                            new_weights[new_k] = v

                    return new_weights

                # interpolate positional embeddings if necessary
                old_pos_embed = mm_inv_projector_weights["model.mm_inv_projector.net.pos_embed"]
                old_h = old_w = int(math.sqrt(old_pos_embed.shape[1]))
                cur_h = cur_w = int(math.sqrt(self.mm_inv_projector.net.pos_embed_view.shape[1]))
                if old_h != cur_h:
                    rank0_print(f"=> interpolated pos_embed from {old_h}x{old_w} to {cur_h}x{cur_w}")
                    old_pos_embed = rearrange(old_pos_embed, 'b (h w) c -> b c h w', h=old_h, w=old_w)
                    new_pos_embed = torch.nn.functional.interpolate(
                        old_pos_embed,
                        size=(cur_h, cur_w),
                        mode='bilinear',
                    )
                    mm_inv_projector_weights["model.mm_inv_projector.net.pos_embed_view"] = rearrange(
                        new_pos_embed, 'b c h w -> b (h w) c',
                    )
                else:
                    mm_inv_projector_weights["model.mm_inv_projector.net.pos_embed_view"] = (
                        mm_inv_projector_weights)["model.mm_inv_projector.net.pos_embed"]

                # for shared weights
                cur_h = cur_w = int(math.sqrt(self.mm_inv_projector.net.pos_embed_bev.shape[1]))
                if old_h != cur_h:
                    rank0_print(f"=> interpolated pos_embed from {old_h}x{old_w} to {cur_h}x{cur_w}")
                    # old_pos_embed = rearrange(old_pos_embed, 'b (h w) c -> b c h w', h=old_h, w=old_w)
                    new_pos_embed = torch.nn.functional.interpolate(
                        old_pos_embed,
                        size=(cur_h, cur_w),
                        mode='bilinear',
                    )
                    mm_inv_projector_weights["model.mm_inv_projector.net.pos_embed_bev"] = rearrange(
                        new_pos_embed, 'b c h w -> b (h w) c',
                    )
                else:
                    mm_inv_projector_weights["model.mm_inv_projector.net.pos_embed_bev"] = (
                        mm_inv_projector_weights)["model.mm_inv_projector.net.pos_embed"]

                # rename weights
                old_weights = deepcopy(mm_inv_projector_weights)
                for k, v in old_weights.items():
                    if "net.x_embedder." in k:
                        k_view = k.replace(".x_embedder.", ".x_embedder_view.")
                        k_bev = k.replace(".x_embedder.", ".x_embedder_bev.")
                        mm_inv_projector_weights[k_view] = v
                        mm_inv_projector_weights[k_bev] = v
                    if "net.z_embedder." in k:
                        k_view = k.replace(".z_embedder.", ".z_embedder_view.")
                        k_bev = k.replace(".z_embedder.", ".z_embedder_bev.")
                        mm_inv_projector_weights[k_view] = v
                        mm_inv_projector_weights[k_bev] = v

                msg = self.mm_inv_projector.load_state_dict(get_w(mm_inv_projector_weights, 'mm_inv_projector'), strict=False)
                print(msg)

            self.config.ross_multi_task = False
            if getattr(model_args, 'ross_multi_task', False):
                self.config.ross_multi_task = True
                # share weights
                

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class Ross3DMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        # bchw
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        # if self.config.mm_spatial_pool_mode == "average":
        #     image_feature = nn.functional.avg_pool2d(image_feature, stride)
        # elif self.config.mm_spatial_pool_mode == "max":
        #     image_feature = nn.functional.max_pool2d(image_feature, stride)
        # elif self.config.mm_spatial_pool_mode == "bilinear":
        #     height, width = image_feature.shape[2:]
        #     scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        #     image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        # else:
        #     raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        height, width = image_feature.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature


    def average_coordinate_in_patch(self, world_coords, patch_size=27):

        V, H, W, D = world_coords.size() # D = 3

        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]    # [32, 378, 378, 3]
        world_coords = world_coords.permute(0, 3, 1, 2)   # [V, D, 378, 378]
        world_coords_avg = torch.nn.functional.avg_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        patch_num = world_coords_avg.shape[-1]
        world_coords_avg = world_coords_avg.permute(0, 2, 3, 1)     # [32, 14, 14, 3]

        return world_coords_avg


    def minmax_coordinate_in_patch(self, world_coords, patch_size=27):

        V, H, W, D = world_coords.size() # D = 3

        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]    # [32, 378, 378, 3]
        world_coords = world_coords.permute(0, 3, 1, 2)   # [V, D, 378, 378]

        world_coords_max = torch.nn.functional.max_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        world_coords_max = world_coords_max.permute(0, 2, 3, 1)     # [32, 14, 14, 3]

        world_coords_min = - torch.nn.functional.max_pool2d(-world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        world_coords_min = world_coords_min.permute(0, 2, 3, 1)     # [32, 14, 14, 3]
        world_coords = torch.stack([world_coords_min, world_coords_max], dim=3) # [32, 14, 14, 2, 3]

        return world_coords


    def sample_n_points(self, world_coords, n_points=9):

        V, H, W, D = world_coords.size() # D = 3
        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :] 
        world_coords = world_coords.view(-1, 14, 27, 14, 27, 3).permute(0, 1, 3, 2, 4, 5)
        if n_points == 9:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
        elif n_points == 5:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
            world_coords_sample = world_coords_sample[:, :, :, 0::2, :].reshape(V, 14, 14, 5, 3)
        elif n_points == 1:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
            world_coords_sample = world_coords_sample[:, :, :, 4, :].reshape(V, 14, 14, 3)
        else:
            raise NotImplementedError
        
        return world_coords_sample


    def discrete_coords(self, world_coords, xyz_min):

        # V, H, W, D = world_coords.size() # D = 3
        # world_coords_discrete = (world_coords.view(-1, 3) - xyz_min.view(1, 3)) / self.config.voxel_size

        min_xyz_range = torch.tensor(self.config.min_xyz_range).to(world_coords.device)
        max_xyz_range = torch.tensor(self.config.max_xyz_range).to(world_coords.device)

        world_coords = torch.maximum(world_coords, min_xyz_range)
        world_coords = torch.minimum(world_coords, max_xyz_range)
        world_coords_discrete = (world_coords - min_xyz_range) / self.config.voxel_size
        world_coords_discrete = world_coords_discrete.round()

        return world_coords_discrete.detach()


    def encode_images(self, images, world_coords=None):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)

        return image_features


    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat, cur_mm_spatial_pool_stride)
            if slower_img_feat != 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)

        return all_videos_or_images_features, all_faster_video_features


    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        num_tokens = image_feature.shape[1]
        feature_dim = image_feature.shape[-1]
        old_image_feature = image_feature.clone().detach()
        boi_ids = [None for _ in range(num_frames)]
        eoi_ids = [None for _ in range(num_frames)]

        # [32, 196, 3584] --> [32, 1, 14, 14, 3584]
        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        # [32, 1, 14, 14, 3584] --> [3584, 32, 14, 1, 14]
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        # [3584, 32, 14, 1, 14] --> [3584, 448, 1, 14] --> [3584, 448, 14]
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        # [3584, 448, 15]
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        newline_ids = [resize_h + i * (resize_h + 1) for i in range(num_frames * num_tokens // resize_h)]

        for image_id in range(old_image_feature.shape[0]):
            old_boi_id = image_id * num_tokens
            old_eoi_id = old_boi_id + num_tokens - 1
            boi_ids[image_id] = int(old_boi_id + old_boi_id // resize_h)
            eoi_ids[image_id] = int(old_eoi_id + old_eoi_id // resize_h)
            assert (old_image_feature[image_id, 0] == image_feature[boi_ids[image_id]]).sum() == feature_dim
            assert (old_image_feature[image_id, -1] == image_feature[eoi_ids[image_id]]).sum() == feature_dim

        return image_feature, boi_ids, eoi_ids, old_image_feature, newline_ids

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def replace_with_mask_token(self, x, mask_ratio):
        # x: [num_frames, num_patches, embed_dim]
        num_frames, num_patches, embed_dim = x.shape
        len_keep = int(num_frames * (1 - mask_ratio))
        len_mask = num_frames - len_keep

        noise = torch.rand(num_frames, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([num_frames], device=x.device)
        mask[:len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=0, index=ids_restore)

        # keep the first subset
        ids_keep = ids_shuffle[:len_keep]
        x_masked = torch.gather(x, dim=0, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, num_patches, embed_dim))

        # append mask tokens
        mask_tokens = self.get_model().mask_token.unsqueeze(0).repeat(len_mask, num_patches, 1)
        x_ = torch.cat([x_masked, mask_tokens], dim=0)
        x_ = torch.gather(x_, dim=0, index=ids_restore.unsqueeze(-1).unsqueeze(-1).repeat(1, num_patches, embed_dim))  # unshuffle

        return x_, mask


    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images: List[torch.FloatTensor],
        modalities=["image"], 
        image_sizes=None, 
        video_dict=None,
        use_object_proposals: bool = False,
        replace_with_mask_token: bool = False,
    ):
        object_boxes = None
        if use_object_proposals:
            object_boxes = video_dict["objects"][0]
            object_boxes_center = object_boxes[:, :3]
            object_features = []
            obj_num = len(object_boxes)

            object_patch = []
            # ignore the batch dimension here
            world_coords = video_dict["world_coords"][0]

            for l in range(obj_num):
                box = object_boxes[l]
                min_xyz = box[:3] - box[3:] / 2
                max_xyz = box[:3] + box[3:] / 2
                
                if "patch27" in self.config.object_feature_type:
                    world_coords_new = world_coords[:, :378, :378, :].reshape(-1, 14, 27, 14, 27, 3).transpose(2, 3).flatten(3, 4)  # [32, 14, 14, 27*27, 3]
                    cur_object_patch = torch.all((min_xyz <= world_coords_new) & (world_coords_new <= max_xyz), dim=-1)     # [32, 14, 14, 27*27]
                    cur_object_patch = cur_object_patch.sum(dim=3) >= int(27 * 27 * 0.25)
                    object_patch.append(cur_object_patch)
                elif "patch14" in self.config.object_feature_type:
                    world_coords_new = world_coords[:, :378, :378, :].reshape(-1, 27, 14, 27, 14, 3).transpose(2, 3).flatten(3, 4)  # [32, 14, 14, 27*27, 3]
                    cur_object_patch = torch.all((min_xyz <= world_coords_new) & (world_coords_new <= max_xyz), dim=-1)     # [32, 14, 14, 27*27]
                    cur_object_patch = cur_object_patch.sum(dim=3) >= int(14 * 14 * 0.5)
                    object_patch.append(cur_object_patch)
                else:
                    raise NotImplementedError


        use_mrope_position_embedding = False
        use_sin3d_pe = False
        use_mlp_pe = False
        if hasattr(self.config, 'world_position_embedding_type') and past_key_values is None:
            B = input_ids.shape[0]
            world_coords = video_dict['world_coords']
            xyz_min = world_coords.view(B, -1, 3).min(dim=1)[0]

            if len(video_dict['box_input']):
                box_input = video_dict['box_input']     # [1, 3]
            else:
                box_input = None

            n_points = 1
            if 'avg' in self.config.world_position_embedding_type:
                world_coords = [self.average_coordinate_in_patch(coords) for coords in world_coords]
            elif "sample9" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=9) for coords in world_coords]
                n_points = 9
            elif "sample5" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=5) for coords in world_coords]
                n_points = 5
            elif "sample1" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=1) for coords in world_coords]
            elif "minmax" in self.config.world_position_embedding_type:
                world_coords = [self.minmax_coordinate_in_patch(coords) for coords in world_coords]
                n_points = 2

            if n_points > 1:
                if box_input is not None:
                    box_input = box_input[:, None, :].repeat(1, n_points, 1)
                if object_boxes is not None:
                    object_boxes_center = object_boxes_center[:, None, :].repeat(1, n_points, 1)

            if 'discrete' in self.config.world_position_embedding_type or use_mrope_position_embedding:
                world_coords_discrete = [self.discrete_coords(coords, xyz_min[i]) for i, coords in enumerate(world_coords)]
                if box_input is not None:
                    box_input = self.discrete_coords(box_input, None)
                if object_boxes is not None:
                    object_boxes_center = self.discrete_coords(object_boxes_center, None)

            if 'mrope' in self.config.world_position_embedding_type:
                use_mrope_position_embedding = True
            
            if "sin3d" in self.config.world_position_embedding_type:
                use_sin3d_pe = True
            
            if "mlp" in self.config.world_position_embedding_type:
                use_mlp_pe = True


        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None, None, None, None

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images)  # [num_frames, num_tokens, embed_dim]

            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)
            assert len(image_features) == 1 # only support batch_size=1
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if use_object_proposals:
                object_features = []
                valid_obj_num = 0
                for l in range(obj_num):
                    # print(f"image_featurs: {image_features[0].shape}")
                    # print(f"object_patch: {object_patch[l].shape}")
                    if "patch27" in self.config.object_feature_type:
                        cur_object_features = image_features[0][object_patch[l].view(-1, 196)]
                    elif "patch14" in self.config.object_feature_type:
                        cur_object_features = encoded_image_features[0][object_patch[l].view(-1, 729)]
                    else:
                        raise NotImplementedError

                    if len(cur_object_features) == 0:
                        cur_object_features = torch.zeros(image_features[0].shape[-1]).to(image_features[0].device)
                    else:
                        cur_object_features = cur_object_features.mean(dim=0)
                        valid_obj_num += 1
                    object_features.append(cur_object_features)
                object_features = torch.stack(object_features)
                if use_mlp_pe or use_sin3d_pe:
                    box_center_features = self.get_model().world_position_embedding(object_boxes_center.unsqueeze(0)).squeeze(0)      
                    object_features += box_center_features
            else:
                object_features =  None
            
            if use_sin3d_pe or use_mlp_pe:
                new_image_features = []
                masks = []
                for idx, image_feat in enumerate(image_features):
                    if "discrete" in self.config.world_position_embedding_type:
                        coords = world_coords_discrete[idx].flatten(1, 2)
                    else:
                        coords = world_coords[idx].flatten(1, 2)

                    # replace with mask token
                    if replace_with_mask_token:
                        image_feat, mask = self.replace_with_mask_token(image_feat, getattr(self.config, "view_mask_ratio", 0.))
                    else:
                        image_feat, mask = self.replace_with_mask_token(image_feat, 0.)

                    # coords: num_frames, num_tokens, 3
                    image_feat = image_feat + self.get_model().world_position_embedding(coords.detach())
                    new_image_features.append(image_feat)
                    masks.append(mask)

                image_features = new_image_features

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            (
                                image_feature,
                                boi_ids,
                                eoi_ids,
                                old_image_feature,
                                newline_ids,
                            ) = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_world_coords = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            assert num_images == 1

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]

            cat_cur_input_ids_noim = torch.cat(cur_input_ids_noim)
            cur_input_embeds = self.get_model().embed_tokens(cat_cur_input_ids_noim)

            # Add input coord PE
            if hasattr(self.config, "coord_token_ids") and (use_sin3d_pe or use_mlp_pe):
                query_coord_tokens = (cat_cur_input_ids_noim == self.config.coord_token_ids[0])
                if query_coord_tokens.sum() != 0:
                    cur_input_embeds[query_coord_tokens] += self.get_model().world_position_embedding(box_input.unsqueeze(0).detach())[:, 0]

            
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_world_coords = []
            cur_pos_index = 0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                text_len = cur_input_embeds_no_im[i].shape[0]
                cur_new_labels.append(cur_labels_noim[i])
                if use_mrope_position_embedding:
                    cur_new_world_coords.append(
                        torch.arange(cur_pos_index, cur_pos_index + len(cur_input_embeds_no_im[i])).to(cur_input_embeds_no_im[i].device).unsqueeze(1).repeat(1, 3)
                    )
                    cur_pos_index += len(cur_input_embeds_no_im[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                        boi_ids = list(map(lambda x: x + text_len, boi_ids))
                        eoi_ids = list(map(lambda x: x + text_len, eoi_ids))
                        newline_ids = list(map(lambda x: x + text_len, newline_ids))
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    
                    if use_mrope_position_embedding:
                        coords = world_coords_discrete[batch_idx]
                        V, H, W, D = coords.shape
                        new_coords = torch.zeros(V*H*(W+1), 3).to(cur_input_embeds_no_im[i].device).view(V, H, W+1, 3)
                        new_coords[:, :, :W, :] = coords
                        new_coords = new_coords.view(-1, 3)
                        cur_pos_index += V * H * (W + 1)
                        cur_new_world_coords.append(new_coords)

                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)

            first_match = torch.all(old_image_feature[:, 0] == cur_new_input_embeds[torch.LongTensor(boi_ids)], dim=1)
            last_match = torch.all(old_image_feature[:, -1] == cur_new_input_embeds[torch.LongTensor(eoi_ids)], dim=1)
            newline_match = torch.all(self.model.image_newline.unsqueeze(0).repeat(len(newline_ids), 1)
                                      == cur_new_input_embeds[torch.LongTensor(newline_ids)], dim=1)
            assert torch.all(first_match) and torch.all(last_match) and torch.all(newline_match)

            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

            if use_mrope_position_embedding:
                cur_new_world_coords = torch.cat(cur_new_world_coords, dim=0)
                new_world_coords.append(cur_new_world_coords)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        mrope_position_ids = torch.zeros((batch_size, max_len, 3), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    if use_mrope_position_embedding:
                        mrope_position_ids[i, -cur_len:, :] = new_world_coords[i][-cur_len:, :]

            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    if use_mrope_position_embedding:
                        mrope_position_ids[i, :cur_len, :] = new_world_coords[i][:cur_len, :]

        # mrope_position_ids = mrope_position_ids.permute(2, 0, 1)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        
        if use_mrope_position_embedding:
            position_ids = mrope_position_ids

        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        try:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, object_features, object_boxes, boi_ids, eoi_ids, newline_ids, masks[0]
        except:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, object_features, object_boxes, boi_ids, eoi_ids, newline_ids, None

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def compute_vm_loss(
        self,
        images: torch.Tensor,
        hidden_states: torch.Tensor,
        boi_ids: List[int],
        eoi_ids: List[int],
        newline_ids: torch.Tensor,
        mask: torch.Tensor,
    ):
        batch_size = hidden_states.shape[0]
        assert batch_size == 1 and len(images) == 1
        images = images[0]
        boi_ids = torch.LongTensor(boi_ids)
        eoi_ids = torch.LongTensor(eoi_ids)

        num_frames = boi_ids.shape[0]
        patch_h = math.ceil(math.sqrt(self.model.image_embed_len))
        image_hidden_states = torch.zeros((num_frames, self.model.image_embed_len, hidden_states.shape[-1]),
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        for frame_index, (cur_boi_id, cur_eoi_id) in enumerate(zip(boi_ids, eoi_ids)):
            if (cur_boi_id is not None) and (cur_eoi_id is not None):
                # need to remove image_newline tokens
                # rank0_print(cur_boi_id, cur_eoi_id, newline_ids[frame_index * patch_h : (frame_index + 1) * patch_h - 1])

                cur_hidden_states = [hidden_states[0][cur_boi_id : newline_ids[frame_index * patch_h]]]
                for k in range(frame_index * patch_h + 1, (frame_index + 1) * patch_h):
                    cur_hidden_states.append(
                        hidden_states[0][newline_ids[k - 1] + 1 : newline_ids[k]]
                    )
                cur_hidden_states.append(hidden_states[0][newline_ids[(frame_index + 1) * patch_h - 1] + 1 : cur_eoi_id])
                image_hidden_states[frame_index] = torch.cat(cur_hidden_states)

        images_std = torch.tensor(self.config.image_std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        images_mean = torch.tensor(self.config.image_mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        images_vae = ((images * images_std + images_mean - 0.5) / 0.5).clamp(-1., 1.)
        images_vae = nn.functional.interpolate(images_vae, size=(self.config.decode_image_size, self.config.decode_image_size), mode='bilinear')

        if mask.sum() > 0:
            # recon only masked images
            images_vae = images_vae[mask.bool()]
            image_hidden_states = image_hidden_states[~mask.bool()]
            repeat_factor = int(num_frames // mask.sum().item())
        else:
            return torch.tensor(0.)

        with torch.no_grad():
            posterior = self.model.mm_pixel_decoder.encode(images_vae).latent_dist
            z_q = (posterior.sample() - self.model.mm_pixel_decoder.shift_factor) * self.model.mm_pixel_decoder.scaling_factor
            if z_q.shape[-1] % 2 == 1:
                z_q = nn.functional.interpolate(z_q, size=(z_q.shape[-2] + 1, z_q.shape[-1] + 1), mode='bilinear')
            # group each (2x2) window
            z_q = z_q.unfold(2, 2, 2).unfold(3, 2, 2)
            z_q = rearrange(z_q, 'b c h w p1 p2 -> b (c p1 p2) h w').contiguous()

        with torch.amp.autocast('cuda', dtype=torch.float32):
            # image_hidden_states = self.model.mm_inv_projector.ln_pre(
            #     image_hidden_states) + self.model.mm_inv_projector.pos_embed
            image_hidden_states = self.model.mm_inv_projector.ln_pre(image_hidden_states)
            h = w = int(image_hidden_states.shape[1] ** 0.5)
            image_hidden_states = rearrange(image_hidden_states, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            vm_loss = self.model.mm_inv_projector(
                z=image_hidden_states.repeat(repeat_factor, 1, 1, 1).contiguous().float(),
                target=z_q.repeat(repeat_factor, 1, 1, 1).contiguous().float(),
                bev=False,
            )
        vm_loss = vm_loss.float().mean()
        return vm_loss


    def compute_vm_loss_v2(
        self,
        images: torch.Tensor,
        hidden_states: torch.Tensor,
        boi_ids: List[int],
        eoi_ids: List[int],
        newline_ids: torch.Tensor,
        mask: torch.Tensor,
    ):
        batch_size = hidden_states.shape[0]
        assert batch_size == 1 and len(images) == 1
        images = images[0]
        boi_ids = torch.LongTensor(boi_ids)
        eoi_ids = torch.LongTensor(eoi_ids)

        num_frames = boi_ids.shape[0]
        patch_h = math.ceil(math.sqrt(self.model.image_embed_len))
        image_hidden_states = torch.zeros((num_frames, self.model.image_embed_len, hidden_states.shape[-1]),
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        for frame_index, (cur_boi_id, cur_eoi_id) in enumerate(zip(boi_ids, eoi_ids)):
            if (cur_boi_id is not None) and (cur_eoi_id is not None):
                # need to remove image_newline tokens
                # rank0_print(cur_boi_id, cur_eoi_id, newline_ids[frame_index * patch_h : (frame_index + 1) * patch_h - 1])

                cur_hidden_states = [hidden_states[0][cur_boi_id : newline_ids[frame_index * patch_h]]]
                for k in range(frame_index * patch_h + 1, (frame_index + 1) * patch_h):
                    cur_hidden_states.append(
                        hidden_states[0][newline_ids[k - 1] + 1 : newline_ids[k]]
                    )
                cur_hidden_states.append(hidden_states[0][newline_ids[(frame_index + 1) * patch_h - 1] + 1 : cur_eoi_id])
                image_hidden_states[frame_index] = torch.cat(cur_hidden_states)

        images_std = torch.tensor(self.config.image_std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        images_mean = torch.tensor(self.config.image_mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        images_vae = ((images * images_std + images_mean - 0.5) / 0.5).clamp(-1., 1.)
        images_vae = nn.functional.interpolate(images_vae, size=(self.config.decode_image_size, self.config.decode_image_size), mode='bilinear')

        with torch.no_grad():
            posterior = self.model.mm_pixel_decoder.encode(images_vae).latent_dist
            z_q = (posterior.sample() - self.model.mm_pixel_decoder.shift_factor) * self.model.mm_pixel_decoder.scaling_factor
            if z_q.shape[-1] % 2 == 1:
                z_q = nn.functional.interpolate(z_q, size=(z_q.shape[-2] + 1, z_q.shape[-1] + 1), mode='bilinear')
            # group each (2x2) window
            z_q = z_q.unfold(2, 2, 2).unfold(3, 2, 2)
            z_q = rearrange(z_q, 'b c h w p1 p2 -> b (c p1 p2) h w').contiguous()

        with torch.amp.autocast('cuda', dtype=torch.float32):
            # image_hidden_states = self.model.mm_inv_projector.ln_pre(
            #     image_hidden_states) + self.model.mm_inv_projector.pos_embed
            image_hidden_states = self.model.mm_inv_projector.ln_pre(image_hidden_states)
            h = w = int(image_hidden_states.shape[1] ** 0.5)
            image_hidden_states = rearrange(image_hidden_states, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            vm_loss = self.model.mm_inv_projector(
                z=image_hidden_states.float(),
                x0=z_q.float(),
            )
        vm_loss = vm_loss.float().mean()
        return vm_loss

    def compute_vm_loss_bev(
        self,
        images: torch.Tensor,
        hidden_states: torch.Tensor,
        boi_ids: List[int],
        eoi_ids: List[int],
        newline_ids: torch.Tensor,
        bev_image: torch.Tensor,
        mask: torch.Tensor,
    ):
        batch_size = hidden_states.shape[0]
        assert batch_size == 1 and len(images) == 1
        images = images[0]
        boi_ids = torch.LongTensor(boi_ids)
        eoi_ids = torch.LongTensor(eoi_ids)

        num_frames = boi_ids.shape[0]
        patch_h = math.ceil(math.sqrt(self.model.image_embed_len))
        image_hidden_states = torch.zeros((num_frames, self.model.image_embed_len, hidden_states.shape[-1]),
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        for frame_index, (cur_boi_id, cur_eoi_id) in enumerate(zip(boi_ids, eoi_ids)):
            if (cur_boi_id is not None) and (cur_eoi_id is not None):
                # need to remove image_newline tokens
                # rank0_print(cur_boi_id, cur_eoi_id, newline_ids[frame_index * patch_h : (frame_index + 1) * patch_h - 1])

                cur_hidden_states = [hidden_states[0][cur_boi_id : newline_ids[frame_index * patch_h]]]
                for k in range(frame_index * patch_h + 1, (frame_index + 1) * patch_h):
                    cur_hidden_states.append(
                        hidden_states[0][newline_ids[k - 1] + 1 : newline_ids[k]]
                    )
                cur_hidden_states.append(hidden_states[0][newline_ids[(frame_index + 1) * patch_h - 1] + 1 : cur_eoi_id])
                image_hidden_states[frame_index] = torch.cat(cur_hidden_states)

        images_vae = ((bev_image - 0.5) / 0.5).clamp(-1., 1.)

        if mask.sum() > 0:
            # take only [unmasked] hidden states as conditions
            image_hidden_states = image_hidden_states[~mask.bool()]

        with torch.no_grad():
            posterior = self.model.mm_pixel_decoder.encode(images_vae).latent_dist
            z_q = (posterior.sample() - self.model.mm_pixel_decoder.shift_factor) * self.model.mm_pixel_decoder.scaling_factor
            if z_q.shape[-1] % 2 == 1:
                z_q = nn.functional.interpolate(z_q, size=(z_q.shape[-2] + 1, z_q.shape[-1] + 1), mode='bilinear')
            # group each (2x2) window
            z_q = z_q.unfold(2, 2, 2).unfold(3, 2, 2)
            z_q = rearrange(z_q, 'b c h w p1 p2 -> b (c p1 p2) h w').contiguous()
            # filter
            bev_downsample = torch.nn.functional.interpolate(bev_image, size=(z_q.shape[-2], z_q.shape[-1]), mode='bilinear').mean(dim=1)
            loss_mask = (bev_downsample.unsqueeze(1) > 0).bool().repeat(1, z_q.shape[1], 1, 1)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            # image_hidden_states = self.model.mm_inv_projector.ln_pre(
            #     image_hidden_states) + self.model.mm_inv_projector.pos_embed
            image_hidden_states = self.model.mm_inv_projector.ln_pre(image_hidden_states)
            h = w = int(image_hidden_states.shape[1] ** 0.5)
            image_hidden_states = rearrange(image_hidden_states, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            vm_loss = self.model.mm_inv_projector(
                z=image_hidden_states.float(),
                target=z_q.float(),
                bev=True,
            )
        vm_loss = (vm_loss.float() * loss_mask).sum() / loss_mask.sum()
        # vm_loss = vm_loss.float().mean()
        return vm_loss


@dataclass
class CausalLMOutputWithPastRoss(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    vm_loss: Optional[torch.FloatTensor] = None
    bev_loss: Optional[torch.FloatTensor] = None
    n_tokens: Optional[torch.LongTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    scores: Optional[torch.FloatTensor] = None
