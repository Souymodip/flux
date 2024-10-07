import torch
from flux.model import Flux, FluxParams
from flux.modules.conditioner import HFEmbedder
from flux.sampling import denoise, get_noise, prepare, get_schedule


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def get_t5(max_length=256):
    return HFEmbedder("google-t5/t5-small", max_length=max_length, torch_dtype=dtype)


def get_clip(max_length=77):
    return HFEmbedder("openai/clip-vit-base-patch16", max_length=max_length, torch_dtype=dtype)


params=FluxParams(
            in_channels=64,
            vec_in_dim=512,
            context_in_dim=512,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        )


def main(inp, timesteps, model):
    x = denoise(model, **inp, timesteps=timesteps, guidance=4)


if __name__ == "__main__":
    t5 = get_t5()
    model = Flux(params)
    clip = get_clip()

    print(f'Flux Model created')
    rng = torch.Generator(device="cpu")
    seed = rng.seed()

    x = get_noise(1, 32, 32, device=device, dtype=dtype, seed=seed)
    prompt = "one"

    inp = prepare(t5, clip, x, prompt=prompt)
    timesteps = get_schedule(4, inp["img"].shape[1], shift=False)

    import pdb; pdb.set_trace()
    main(inp, timesteps, model)



