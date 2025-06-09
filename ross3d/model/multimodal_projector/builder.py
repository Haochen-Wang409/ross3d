import torch
import torch.nn as nn
import re

from .pooler_projector import PoolerProjector
from ..multimodal_denoiser.denoiser_dit_multi import Ross3DDenoiserMulti


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == "pooler":
        return PoolerProjector(config, kwargs["vision_cfg"])

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")


def build_inv_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_inv_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.hidden_size, config.mm_inv_hidden_size)

    if projector_type.startswith("denoiser"):
        vit_match = re.match(r'^denoiser_vit(\d+)x$', projector_type)
        depth = int(vit_match.group(1))

        if depth == 8:
            width = 1280
        elif depth == 12:
            width = 1536
        else:
            width = 1024

        return Ross3DDenoiserMulti(
            x_channel=config.mm_inv_hidden_size,
            z_channel=config.hidden_size,
            embed_dim=width,
            depth=depth,
            timesteps='1000',
            learn_sigma=False,
            n_patches_view=config.image_embed_len,
            n_patches_bev=729,
        )

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.hidden_size, config.hidden_size)]
        if mlp_depth > 2:
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        modules.append(nn.GELU())
        modules.append(nn.Linear(config.hidden_size, config.mm_inv_hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
