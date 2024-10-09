import torch
from torch import Tensor
from einops import rearrange, repeat
from flux.modelImg import FluxImg, FluxImgParams
from flux.modules.layers import EmbedND
import math

device = torch.device("cuda") if torch.cuda.is_available() else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
feature_size = 2
H = 32
W = 32
in_channels = 1
cond_channels = 3

params_img=FluxImgParams(
            in_channels=feature_size*feature_size * in_channels,
            cond_channels=feature_size*feature_size * cond_channels,
            hidden_size=1024,
            mlp_ratio=4.0,
            num_heads=8,
            depth=7,
            depth_single_blocks=14,
            axes_dim=[16, 56, 56],
            theta=10_000,
            height=H,
            width=W,
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
        in_channels,
        height,
        width,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
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


def denoise(model:FluxImg, x:Tensor, y:Tensor, pe:Tensor, guidance, timesteps:list[float]):
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)
        v = model(img=x, img_cond=y, pe=pe, timesteps=t_vec, guidance=guidance)
        x = x + v * ( t_prev - t_curr)
    return x


def generate(model:FluxImg, y:Tensor, pe:Tensor, guidance:float,
             timesteps:list[float], seed:int) -> Tensor:
    x = get_noise(16, H, W, device, dtype, seed=seed)
    x = denoise(model, x, y, pe, guidance, timesteps)
    return x




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


def prepare_img(img:Tensor, img_cond:Tensor):
    assert img_cond.shape == img.shape, f'Image and Image Conditioner shape mismatch: {img.shape} != {img_cond.shape}'

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=feature_size, pw=feature_size)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=feature_size, pw=feature_size)

    return {
        "img": img,
        "img_cond": img_cond
    }


def pack(x):
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=feature_size, pw=feature_size)


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


def get_pe2(height:int, width:int, batch_size: int,
                pe_dim: int, theta:int, axes_dim: list[int]) -> Tensor:
    ids0 = torch.zeros(height//feature_size, width//feature_size, 3)
    ids0[..., 1] = ids0[..., 1] + torch.arange(height//feature_size)[:, None]
    ids0[..., 2] = ids0[..., 2] + torch.arange(width//feature_size)[None, :]
    ids0 = repeat(ids0, "h w c -> b (h w) c", b=batch_size)

    ids1 = torch.zeros(height//feature_size, width//feature_size, 3)
    ids1[..., 1] = ids1[..., 1] + torch.arange(height//feature_size, 2*height//feature_size)[:, None]
    ids1[..., 2] = ids1[..., 2] + torch.arange(width//feature_size, 2*width//feature_size)[None, :]
    ids1 = repeat(ids1, "h w c -> b (h w) c", b=batch_size)

    ids = torch.cat((ids0, ids1), dim=1)
    print(f'Positional Encoding IDs Shape: {ids.shape} -> {ids0.shape} + {ids1.shape}')

    pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
    return pe_embedder(ids)


def get_ts(batch_size, is_stratified, device):
    if is_stratified:
        quantiles = torch.linspace(0, 1, batch_size + 1).to(device)
        z = quantiles[:-1] + torch.rand((batch_size,)).to(device) / batch_size
        z = torch.erfinv(2 * z - 1) * math.sqrt(2)
        t = torch.sigmoid(z)
    else:
        nt = torch.randn((batch_size,)).to(device)
        t = torch.sigmoid(nt)
    return t


def forward(model:FluxImg, x, y, rope, guidance):
    assert x.ndim == 3, f'Expected 3D Tensor, got {x.ndim}D Tensor'
    assert x.device == rope.device, f'Image and Positional Encoding device mismatch: {x.device} != {rope.device}'

    b, n, d = x.shape
    ts = get_ts(b, True, x.device)
    texp = ts.view([b, *([1] * len(x.shape[1:]))])
    z1 = torch.randn_like(x)
    zt = (1 - texp) * x + texp * z1

    zt = zt.to(x.device)
    vtheta = model(img=zt, img_cond=y, pe=rope, timesteps=ts, guidance=guidance)
    batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
    loss = batchwise_mse.mean()

    tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
    ttloss = [(tv, tloss) for tv, tloss in zip(ts, tlist)]
    return loss, ttloss


def get_random_imgs(batch_size, height, width, channels, device, dtype):
    seed = torch.randint(0, 1000, (1,)).item()
    # img = get_noise(batch_size, height, width, device=device, dtype=dtype, seed=seed)
    img = torch.randn(
        batch_size,
        channels,
        # allow for packing
        feature_size * math.ceil(height / feature_size),
        feature_size * math.ceil(width / feature_size),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    return img


def run():
    rng = torch.Generator(device="cpu")
    seed = rng.seed()
    # x = get_noise(16, H, W, device=device, dtype=dtype, seed=seed)
    img_cond = get_noise(16, H, W, device=device, dtype=dtype, seed=seed)

    model = FluxImg(params_img)

    # timesteps = get_schedule(4, inp["img"].shape[1], shift=False)

    pe_dim = params_img.hidden_size // params_img.num_heads
    pe = get_pe2(H, W, 16, pe_dim, params_img.theta, params_img.axes_dim)

    x = get_random_imgs(16, H, W, in_channels, device, dtype)
    y = get_random_imgs(16, H, W, cond_channels, device, dtype)
    token_x = pack(x)
    token_y = pack(y)
    # inp = prepare_img(x, img_cond)

    print(f'token_x Shape: {token_x.shape}\n'
          f'token_y Shape: {token_y.shape}\n'
          f'Positional Encoding Shape: {pe.shape}\n')

    model = model.to(device)
    token_x = token_x.to(device)
    pe = pe.to(device)

    guidance = 4.0
    guidance_vec = torch.full((token_x.shape[0],), guidance, device=token_x.device, dtype=token_x.dtype)
    loss, ttloss = forward(model, token_x, token_y, pe, guidance_vec)
    print(f'Loss: {loss}')
    # main(inp, timesteps, model, pe)


if __name__ == "__main__":
    run()