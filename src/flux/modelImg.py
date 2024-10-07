from dataclasses import dataclass
from einops import repeat
import torch
from torch import Tensor, nn

from flux.modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)


@dataclass
class FluxImgParams:
    in_channels: int
    cond_channels: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    height: int
    width: int
    qkv_bias: bool
    guidance_embed: bool


class FluxImg(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxImgParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.cond_channels = params.cond_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.img_cond_in = nn.Linear(self.cond_channels, self.hidden_size, bias=True)

        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        model_size = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dtype = next(self.parameters()).dtype
        print(f"Number of parameters: {model_size}, {model_size / (dtype.itemsize * 1e6):.2f}MB")

    def forward(
        self,
        img: Tensor,
        img_cond: Tensor,
        pe: Tensor,
        timesteps: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        print(f'-img shape: {img.shape}, txt shape: {img_cond.shape}')
        if img.ndim != 3 or img_cond.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

        img_cond = self.img_cond_in(img_cond)
        print(f'\tpe shape: {pe.shape}, img shape: {img.shape}, txt shape: {img_cond.shape}')
        for block in self.double_blocks:
            img, img_cond = block(img=img, txt=img_cond, vec=vec, pe=pe)

        img = torch.cat((img_cond, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, img_cond.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img