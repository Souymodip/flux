import torch
from torch import Tensor
from einops import rearrange, repeat
from flux.model import Flux, FluxParams
from flux.modules.conditioner import HFEmbedder
from flux.sampling import denoise, get_schedule, prepare
import math


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
feature_size = 2
H = 32
W = 32


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        1,
        # allow for packing
        feature_size * math.ceil(height / feature_size),
        feature_size * math.ceil(width / feature_size),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def get_t5(max_length=256):
    return HFEmbedder("google-t5/t5-small", max_length=max_length, torch_dtype=dtype)


def get_clip(max_length=77):
    return HFEmbedder("openai/clip-vit-base-patch16", max_length=max_length, torch_dtype=dtype)


params = FluxParams(
            in_channels=feature_size*feature_size,
            vec_in_dim=512,
            context_in_dim=512,
            hidden_size=1024,
            mlp_ratio=4.0,
            num_heads=8,
            depth=7,
            depth_single_blocks=14,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        )


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / feature_size),
        w=math.ceil(width / feature_size),
        ph=2,
        pw=2,
    )


def infer(inp, timesteps, model, pe):
    img = inp["img"].to(device)
    img_cond = inp["img_cond"].to(device)
    pe = pe.to(device)
    t_vec = torch.full((img.shape[0],), timesteps[0], dtype=img.dtype, device=img.device)
    model = model.to(device)
    pred = model(
        img=img,
        img_cond=img_cond,
        pe=pe,
        timesteps=t_vec,
        guidance=4.0,
    )
    nx = unpack(pred, 32, 32)
    print(f'Output Shape: {pred.shape}, Unpacked Shape: {nx.shape}')


def run_denoise():
    t5 = get_t5()
    clip = get_clip()
    img = get_noise(1, 32, 32, device=device, dtype=dtype, seed=1)
    prompt = "one"
    inp = prepare(t5, clip, img, prompt)

    timesteps = get_schedule(4, inp["img"].shape[1], shift=False)
    model = Flux(params)
    print(f'Image Shape: {inp["img"].shape},\n'
          f'Timesteps: {timesteps},\n'
          f'T5 Shape: {inp["txt"].shape}\n'
          f'Clip Shape: {inp["vec"].shape}\n'
          f'Image IDs Shape: {inp["img_ids"].shape}\n'
          f'Text IDs Shape: {inp["txt_ids"].shape}\n')

    model = model.to(device)
    inp["img"] = inp["img"].to(device)
    inp["img_ids"] = inp["img_ids"].to(device)
    inp["txt"] = inp["txt"].to(device)
    inp["txt_ids"] = inp["txt_ids"].to(device)
    inp["vec"] = inp["vec"].to(device)

    x = denoise(model, **inp, timesteps=timesteps, guidance=4)
    print(f'Output Shape: {x.shape}')
