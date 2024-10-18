import lightning as l
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import torch
from torch import Tensor
from einops import rearrange, repeat
from flux.modelImg import FluxImg, FluxImgParams
from flux.modules.layers import EmbedND
from dataclasses import dataclass
import math
import torchvision
import dlSeg as dl
import matplotlib.pyplot as plt
# from lightning.pytorch.loggers import WandbLogger
# import wandb
from tqdm import tqdm
from io import BytesIO
import numpy as np
from PIL import Image
from flux.modules.autoencoder import AutoEncoder
import flux.util as util


torch.set_float32_matmul_precision('medium')


@torch.no_grad()
def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    ae = util.load_ae(name, device)
    # iterate over all parameters and set requires_grad to False
    for param in ae.parameters():
        param.requires_grad = False
    return ae


@dataclass
class TrainParams:
    batch_size: int
    epochs: int
    lr: float
    H: int
    W: int
    Z: int # latent dim
    seed: int
    in_channels: int
    cond_channels: int
    feature_size: int
    device: torch.device
    dtype: torch.dtype


def get_noise(
        num_samples: int,
        in_channels: int,
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


def plot_line(x, y, label_x, label_y):
    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5))
    ax.plot(x, y, 'o-')
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # create numpy array of buf
    img = np.array(Image.open(buf))
    buf.close()
    plt.close()
    return img


def get_pe(height: int, width: int, feature_size: int,
           pe_dim: int, theta: int, axes_dim: list[int]) -> Tensor:
    ids0 = torch.zeros(height // feature_size, width // feature_size, 3)
    ids0[..., 1] = ids0[..., 1] + torch.arange(height // feature_size)[:, None]
    ids0[..., 2] = ids0[..., 2] + torch.arange(width // feature_size)[None, :]
    ids0 = repeat(ids0, "h w c -> b (h w) c", b=1)
    # ids0 = ids0.unsqueeze(0)

    ids1 = torch.zeros(height // feature_size, width // feature_size, 3)
    ids1[..., 1] = ids1[..., 1] + torch.arange(height // feature_size, 2 * height // feature_size)[:, None]
    ids1[..., 2] = ids1[..., 2] + torch.arange(width // feature_size, 2 * width // feature_size)[None, :]
    ids1 = repeat(ids1, "h w c -> b (h w) c", b=1)
    # ids1 = ids1.unsqueeze(0)

    ids = torch.cat((ids0, ids1), dim=1)
    # print(f'Positional Encoding IDs Shape: {ids.shape} -> {ids0.shape} + {ids1.shape}')

    pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
    return pe_embedder(ids)


def get_random_imgs(batch_size, height, width, channels, feature_size, device, dtype):
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


def get_ts(batch_size, device, is_stratified, focus_ends):
    if is_stratified:
        quantiles = torch.linspace(0, 1, batch_size + 1).to(device)
        z = quantiles[:-1] + torch.rand((batch_size,)).to(device) / batch_size
        z = torch.erfinv(2 * z - 1) * math.sqrt(2)
        k = torch.randint(low=1, high=4,size=(1,)).item() if focus_ends else 1
        t = torch.sigmoid(k*z)
    else:
        nt = torch.randn((batch_size,)).to(device)
        t = torch.sigmoid(nt)
    return t


class RF(l.LightningModule):
    def __init__(self, model: FluxImg, vae:AutoEncoder|None,
                 fluxparams: FluxImgParams, trainparams: TrainParams):
        super(RF, self).__init__()
        self.model: FluxImg = model
        self.fparams: FluxImgParams = fluxparams
        self.tparams: TrainParams = trainparams

        self.pe = None
        self.vae = vae.eval()

    def setup(self, stage: str) -> None:
        if self.vae is not None:
            H, W, F = self.tparams.Z, self.tparams.Z, 1
        else:
            H, W, F = self.tparams.H, self.tparams.W, self.tparams.feature_size

        self.pe = get_pe(H, W, F, self.fparams.hidden_size,
                         self.fparams.theta, self.fparams.axes_dim)

        self.pe.requires_grad = False
        self.pe.to(self.tparams.dtype)
        self.pe = self.pe.to(self.tparams.device)
        # import pdb; pdb.set_trace()

    def pack(self, x):
        return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                         ph=self.tparams.feature_size, pw=self.tparams.feature_size)

    def pack_vae(self, x):
        """
            x shape b,c,h,w
        """
        assert self.vae is not None, "VAE is not loaded!"
        assert not self.vae.training, "VAE is still training!"
        b, c, h, w = x.shape
        assert c == 3, "Input must be RGB Image"
        x = self.vae.encode(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x

    def unpack(self, x: Tensor) -> Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(self.tparams.H / self.tparams.feature_size),
            w=math.ceil(self.tparams.W / self.tparams.feature_size),
            ph=self.tparams.feature_size,
            pw=self.tparams.feature_size,
        )

    def unpack_vae(self, x: Tensor) -> Tensor:
        """
            x shape b, n, d
        """
        assert self.vae is not None, "VAE is not loaded!"
        assert not self.vae.training, "VAE is still training!"
        b, n, d = x.shape
        Z = self.tparams.Z
        x = rearrange(x, "b (h w) d -> b d h w", h=Z, w=Z)
        return self.vae.decode(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.tparams.lr)

    def prepare(self, batch, batch_idx):
        in_img, out_img = batch
        if self.vae is not None:
            out_img = out_img.repeat(1, 3, 1, 1)
            token_y = self.pack_vae(in_img)
            token_x = self.pack_vae(out_img)
        else:
            token_y = self.pack(in_img)  # b, 3, H, W
            token_x = self.pack(out_img)  # b, 1, H, W

        curr_epoch = self.current_epoch
        b, n, d = token_x.shape
        ts = get_ts(b, token_x.device, True, curr_epoch > 100)
        texp = ts.view([b, *([1] * len(token_x.shape[1:]))])

        z1 = torch.randn_like(token_x)
        zt = (1 - texp) * token_x + texp * z1
        zt = zt.to(token_x.device)

        assert self.pe.shape[0] == 1, f'pe shape : {self.pe.shape}'
        pe = repeat(self.pe, "... -> b ...", b=b).squeeze(1)
        assert len(pe.shape) == len(self.pe.shape), f'{pe.shape} ~ {self.pe.shape}'

        return z1, zt, token_y, token_x, ts, pe

    def training_step(self, batch, batch_idx):
        assert self.model.training, "Model is not in training!"
        z1, zt, token_y, token_x, ts, pe = self.prepare(batch, batch_idx)

        vtheta = self.model(img=zt, img_cond=token_y, pe=pe, timesteps=ts)

        batchwise_mse = ((z1 - token_x - vtheta) ** 2).mean(dim=list(range(1, len(token_x.shape))))
        loss = batchwise_mse.mean()

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss


class GenerateCallback(Callback):
    def __init__(self, imgs: Tensor, out_imgs: Tensor, timesteps: list[float] | None, seed: int):
        self.num = 16
        self.x = out_imgs[:self.num]
        self.y = imgs[:self.num]  # torchvision.transforms.Normalize(mean=[0.5]*c, std=[0.5]*c)(imgs)
        self.timesteps = timesteps
        self.seed = seed

    @staticmethod
    def prepare(rf, cond_img, out_img):
        if rf.vae is not None:
            out_img = out_img.repeat(1, 3, 1, 1)
            token_y = rf.pack_vae(cond_img)
            token_x = rf.pack_vae(out_img)
        else:
            token_y = rf.pack(cond_img)  # b, 3, H, W
            token_x = rf.pack(out_img)  # b, 1, H, W
        token_z1 = torch.randn_like(token_x) #rf.pack(z1)  # b, 1, H, W

        b, _, _ = token_x.shape
        pe = repeat(rf.pe, "... -> b ...", b=b).squeeze(1)
        return token_z1, token_x, token_y, pe

    @staticmethod
    def get_timesteps():
        timesteps = torch.linspace(1, 0, 100)
        timesteps = torch.sigmoid(11 * (timesteps - 0.5))
        return timesteps

    @staticmethod
    def to_img(rf, l):
        if rf.vae is None:
            l = l - l.min()
            l = l / l.max()
            return rf.unpack(l)
        else:
            l = rf.unpack_vae(l).clip(0, 1)
        return l

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.model.eval()
        epoch = trainer.current_epoch
        device_id = pl_module.device.index if torch.cuda.is_available() else "cpu"
        print(f'Device: {pl_module.device}, {device_id}')
        with torch.no_grad():
            if trainer.current_epoch % 10 == 0:
                x, y = self.x.to(pl_module.device), self.y.to(pl_module.device)
                token_z1, token_x, token_y, pe = self.prepare(pl_module, cond_img=y, out_img=x)
                zt = token_z1.clone()
                timesteps = self.get_timesteps()
                losses = []

                for i in tqdm(range(len(timesteps) - 1)):
                    t_vec = torch.full((token_x.shape[0],), timesteps[i], dtype=token_x.dtype, device=token_x.device)
                    v = pl_module.model(img=zt, img_cond=token_y, pe=pe, timesteps=t_vec)

                    loss = ((token_z1 - token_x - v) ** 2).mean()
                    losses.append(loss.item())

                    zt = zt + v * (timesteps[i + 1] - timesteps[i])

                zt_img = self.to_img(pl_module, zt).cpu() # b, 1, H, W
                y = y.cpu() * 0.5 + 0.5 # b, 3, H, W
                x = x.cpu() * 0.5 + 0.5 # b, 1, H, W

                zt_img = repeat(zt_img, "b c h w -> b (k c) h w", k=3)
                x = repeat(x, "b c h w -> b (k c) h w", k=3)
                grid = torch.cat((y, x, zt_img), dim=0)
                out = torchvision.utils.make_grid(grid, nrow=grid.shape[0] // 3, pad_value=1)

                # rand_id = x.get_device()
                # trainer.logger.experiment.log({f"Gen:{epoch}:{device_id}": wandb.Image(out, caption="Generated Images")})
                # # # log losses plot in wandb
                # plt_img = plot_line(timesteps[:-1], losses, "Time", "Loss")
                # trainer.logger.experiment.log({f"Losses:{epoch}:{device_id}": wandb.Image(plt_img, caption="Losses Plot")})

        pl_module.model.train()


device = torch.device("cuda") if torch.cuda.is_available() else (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
dtype = torch.float32  # torch.bfloat16 if torch.cuda.is_available() else torch.float32
head = '/Users/souymodip/GIT/SvgDataGen' #'/home/souchakr/sensei-fs-symlink/users/souchakr/localssd'  #
folder_path = f'{head}/data_seg_val'
val_path = f'{head}/data_seg_val'
num_workers = 8  # 1

train_params = TrainParams(
    batch_size=2,
    epochs=420000,
    lr=1e-4,
    H=128,
    W=128,
    Z=16,
    seed=42,
    in_channels=16,
    cond_channels=16,
    feature_size=1,
    device=device,
    dtype=dtype
)

fparams = FluxImgParams(
    in_channels=train_params.feature_size * train_params.feature_size * train_params.in_channels,
    cond_channels=train_params.feature_size * train_params.feature_size * train_params.cond_channels,
    hidden_size=1024,
    mlp_ratio=4.0,
    num_heads=8,
    depth=7,
    depth_single_blocks=14,
    axes_dim=[16, 56, 56],
    theta=10_000,
    height=train_params.H,
    width=train_params.W,
    qkv_bias=True,
    guidance_embed=False,
)


def train():
    data = dl.get_train_loader(folder_path, train_params.H, train_params.dtype, train_params.batch_size,
                               num_workers=num_workers)
    val = dl.get_val_loader(val_path, train_params.H, train_params.dtype, train_params.batch_size,
                            num_workers=num_workers)

    mc = ModelCheckpoint(monitor='train_loss', save_last=True, every_n_epochs=5,
                         filename='s{step:08d}-l{train_loss:.3f}')

    x, y = next(iter(val))
    cb = [
        mc,
        GenerateCallback(x, y, None, 42)
    ]
    # wandb_logger = WandbLogger(project="DiTEdge")

    trainer = l.Trainer(accelerator="gpu", devices="auto", strategy="auto",
                        callbacks=cb,
                        # logger=wandb_logger, max_epochs=train_params.epochs, accumulate_grad_batches=8,
                        max_epochs=1, limit_train_batches=1,
                        # val_check_interval=0.5
                        )
    vae = load_ae("flux-schnell", device)
    vae.eval()

    ckpt_path = None # '/Users/souymodip/Downloads/sstep=00077625-ltrain_loss=0.005.ckpt' #'/sensei-fs/users/souchakr/RF/src/DiTEdge/2pgkchff/checkpoints/epoch=124-step=128000.ckpt'  # '/Users/souymodip/GIT/flux/CKPT/checkpoints/epoch=124-step=128000.ckpt'
    if ckpt_path:
        print(f"Loading Checkpoint from {ckpt_path}")
        rf = RF.load_from_checkpoint(ckpt_path, model=FluxImg(fparams), vae=vae, fluxparams=fparams, trainparams=train_params)
        rf.setup("train")
    else:
        rf = RF(model=FluxImg(fparams), vae=vae,
                fluxparams=fparams, trainparams=train_params)

    trainer.fit(rf, data)
    print('Training Completed!')


if __name__ == '__main__':
    train()

    # ckpt_path ='/sensei-fs/users/souchakr/RF/src/DiTEdge/2pgkchff/checkpoints/epoch=124-step=128000.ckpt'
    # if ckpt_path:
    #     print(f"Loading Checkpoint from {ckpt_path}")
    #     rf = RF.load_from_checkpoint(ckpt_path, model=FluxImg(fparams), fluxparams=fparams, trainparams=train_params)
    # import pdb; pdb.set_trace()


