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


@torch.no_grad()
def test_model_img():
    head = '/Users/souymodip/GIT/pythonProject/'
    val_path = f'{head}/data_seg_test'
    # ckpt_path ='/Users/souymodip/GIT/flux/CKPT/checkpoints/epoch=124-step=128000.ckpt'
    # ckpt_path = '/Users/souymodip/GIT/flux/src/FMSeg/sstep=00077625-ltrain_loss=0.005.ckpt'
    ckpt_path = '/Users/souymodip/Downloads/sstep=00077625-ltrain_loss=0.005.ckpt'
    from dlSeg import get_train_loader as loader
    import lt
    from flux.modelImg import FluxImg
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from tqdm import tqdm
    import imageio

    fparams = lt.fparams
    train_params = lt.train_params

    rf = lt.RF.load_from_checkpoint(ckpt_path, model=FluxImg(fparams), fluxparams=fparams, trainparams=train_params)
    rf.setup("eval")
    rf.model.eval()

    batch_size = 4
    train_loader = loader(folder_path=val_path, size=32,
                                    batch_size=batch_size, num_workers=4, dtype=torch.float32)
    print(f'Number of mini batche: {len(train_loader)}')


    def to_img(l):
        l = l - l.min()
        l = l / l.max()
        return rf.unpack(l).permute(0, 2, 3, 1).cpu().numpy()

    def prepare(in_img, out_img):
        # z1 = torch.randn_like(out_img)
        token_y = rf.pack(in_img)  # b, 3, H, W
        token_x = rf.pack(out_img)  # b, 1, H, W
        token_z1 = torch.randn_like(token_x) #rf.pack(z1)  # b, 1, H, W

        b, n, d = token_x.shape
        # import pdb; pdb.set_trace()
        # ts = torch.tensor([timestep], dtype=token_x.dtype, device=token_x.device)
        # texp = ts.view([b, *([1] * len(token_x.shape[1:]))])

        pe = repeat(rf.pe, "... -> b ...", b=b).squeeze(1)

        # zt = (1 - texp) * token_x + texp * token_z1
        return token_z1, token_x, token_y, pe

    timesteps = torch.linspace(1, 0, 1000)
    timesteps = torch.sigmoid(11 * (timesteps - 0.5))

    plt.plot(timesteps)
    plt.show()

    for k, (x, y) in enumerate(train_loader):
        print(f'x Shape: {x.shape}, range: [{x.min()}, {x.max()}]\n'
              f'y Shape: {y.shape}, range: [{y.min()}, {y.max()}]')

        # _y0 = y[0].squeeze().cpu().numpy()*0.5 + 0.5
        # _x0 = x[0].permute(1, 2, 0).cpu().numpy()*0.5 + 0.5
        # fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
        # ax1.imshow(_x0)
        # ax2.imshow(_y0)
        # plt.show()


        # import pdb;
        # pdb.set_trace()

        token_z1, token_x, token_y, pe = prepare(x, y)
        z1 = token_z1.clone()

        folder = f'output{k}'
        os.makedirs(folder, exist_ok=True)
        losses = []

        token_x_img = to_img(token_x)
        token_y_img = to_img(token_y)

        device = torch.device("mps")
        rf = rf.to(device)
        token_y = token_y.to(device)
        token_x = token_x.to(device)
        token_z1 = token_z1.to(device)
        z1 = z1.to(device)
        pe = pe.to(device)

        image_paths_per_batch = [[] for _ in range(batch_size)]
        for i in tqdm(range(len(timesteps) - 1)):
            t_vec = torch.full((token_x.shape[0],), timesteps[i], dtype=token_x.dtype, device=token_x.device)
            v = rf.model(img=z1, img_cond=token_y, pe=pe, timesteps=t_vec)

            loss = ((token_z1 - token_x - v) ** 2).mean()
            losses.append(loss.item())
            # import pdb; pdb.set_trace()
            z1 = z1 + v * (timesteps[i+1] - timesteps[i])

            if i % 10 == 0 or i == len(timesteps) - 2:
                z1_img = to_img(z1)

                for j in range(batch_size):
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)
                    ax1.imshow(z1_img[j])
                    ax2.imshow(token_x_img[j])
                    ax3.imshow(token_y_img[j])
                    fig.suptitle(f'Timestep: {timesteps[i] :.3f}, loss: {losses[-1]:.3f}')
                    plt.savefig(os.path.join(folder, f'{j}_{i}.png'))
                    plt.close()
                    image_paths_per_batch[j].append(os.path.join(folder, f'{j}_{i}.png'))

                # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)
                # ax1.imshow(to_img(z1))
                # ax2.imshow(token_x_img)
                # ax3.imshow(token_y_img)
                # fig.suptitle(f'Timestep: {timesteps[i] :.3f}, loss: {losses[-1]:.3f}')
                # plt.savefig(os.path.join(folder, f'{i}.png'))
                # plt.close()
                # img_paths.append(os.path.join(folder, f'{i}.png'))

        plt.plot(losses)
        plt.savefig(os.path.join(folder, 'loss.png'))
        plt.close()

        # convert images to gif
        for b, image_paths in enumerate(image_paths_per_batch):
            images = [imageio.v2.imread(img_path) for img_path in image_paths]
            imageio.mimsave(f'{folder}/output{b}.gif', images)
            for img_path in image_paths[:-1]:
                os.remove(img_path)

        # images = [imageio.v2.imread(img_path) for img_path in img_paths]
        # imageio.mimsave(f'output/output{k}.gif', images)

        # z1, zt, token_y, token_x, ts, pe = rf.prepare((x, y), 0)
        # vtheta = rf.model(img=zt, img_cond=token_y, pe=pe, timesteps=ts)

        # break


if __name__ == "__main__":
    # run_denoise()
    test_model_img()