import torch
from torch import Tensor
from einops import rearrange, repeat
from flux.model import Flux, FluxParams
from flux.modelImg import FluxImg, FluxImgParams
from flux.modules.conditioner import HFEmbedder
from flux.sampling import denoise, get_schedule
from flux.modules.layers import EmbedND
import math

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
feature_size = 2
height = 32
width = 32


def get_t5(max_length=256):
    return HFEmbedder("google-t5/t5-small", max_length=max_length, torch_dtype=dtype)


def get_clip(max_length=77):
    return HFEmbedder("openai/clip-vit-base-patch16", max_length=max_length, torch_dtype=dtype)


params_img=FluxImgParams(
            in_channels=feature_size*feature_size,
            hidden_size=1024,
            mlp_ratio=4.0,
            num_heads=8,
            depth=7,
            depth_single_blocks=14,
            axes_dim=[16, 56, 56],
            theta=10_000,
            height=height,
            width=width,
            qkv_bias=True,
            guidance_embed=False,
        )


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


def main(inp, timesteps, model, pe):
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


def main2(inp, timesteps, model):
    x = denoise(model, **inp, timesteps=timesteps, guidance=4)
    print(f'Output Shape: {x.shape}')


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / feature_size),
        w=math.ceil(width / feature_size),
        ph=2,
        pw=2,
    )


def prepare_img(img:Tensor, img_cond:Tensor):
    assert img_cond.shape == img.shape, f'Image and Image Conditioner shape mismatch: {img.shape} != {img_cond.shape}'

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=feature_size, pw=feature_size)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=feature_size, pw=feature_size)

    return {
        "img": img,
        "img_cond": img_cond
    }


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    import pdb; pdb.set_trace()
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=feature_size, pw=feature_size)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // feature_size, w // feature_size, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // feature_size)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // feature_size)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def get_pe(height:int, width:int, batch_size: int,
                pe_dim: int, theta:int, axes_dim: list[int]) -> Tensor:
    ids0 = torch.zeros(height//feature_size, width//feature_size, 3)
    ids0[..., 1] = ids0[..., 1] + torch.arange(height//feature_size)[:, None]
    ids0[..., 2] = ids0[..., 2] + torch.arange(width//feature_size)[None, :]

    ids0 = repeat(ids0, "h w c -> b (h w) c", b=batch_size)
    ids1 = ids0.clone()

    ids = torch.cat((ids0, ids1), dim=1)
    pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
    return pe_embedder(ids)


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
    x = denoise(model, **inp, timesteps=timesteps, guidance=4)
    print(f'Output Shape: {x.shape}')


def run():
    rng = torch.Generator(device="cpu")
    seed = rng.seed()
    x = get_noise(1, height, width, device=device, dtype=dtype, seed=seed)
    img_cond = get_noise(1, height, width, device=device, dtype=dtype, seed=seed)

    model = FluxImg(params_img)
    import pdb; pdb.set_trace()

    inp = prepare_img(x, img_cond)
    timesteps = get_schedule(4, inp["img"].shape[1], shift=False)

    pe_dim = params_img.hidden_size // params_img.num_heads
    pe = get_pe(height, width, 1, pe_dim, params_img.theta, params_img.axes_dim)

    print(f'Input Shape: {inp["img"].shape},\n'
          f'Timesteps: {timesteps},\n'
          f'Image Conditioner Shape: {inp["img_cond"].shape}\n'
          f'Positional Encoding Shape: {pe.shape}\n')

    main(inp, timesteps, model, pe)


if __name__ == "__main__":
    run()