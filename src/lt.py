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
import dl
# import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger
import wandb
from tqdm import tqdm

torch.set_float32_matmul_precision('medium')


@dataclass
class TrainParams:
    batch_size: int
    epochs: int
    lr: float
    H: int
    W: int
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


def get_pe(height: int, width: int, batch_size: int, feature_size: int,
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


class RF(l.LightningModule):
    def __init__(self, model: FluxImg, fluxparams: FluxImgParams, trainparams: TrainParams):
        super(RF, self).__init__()
        self.model: FluxImg = model
        self.fparams: FluxImgParams = fluxparams
        self.tparams: TrainParams = trainparams
        self.pe = get_pe(self.tparams.H, self.tparams.W, self.tparams.batch_size,
                         self.tparams.feature_size, self.fparams.hidden_size,
                         self.fparams.theta, self.fparams.axes_dim)
        self.pe.requires_grad = False
        self.pe.to(self.tparams.dtype)
        self.pe = self.pe.to(self.tparams.device)

    def setup(self, stage: str) -> None:
        self.pe = get_pe(self.tparams.H, self.tparams.W, self.tparams.batch_size,
                         self.tparams.feature_size, self.fparams.hidden_size,
                         self.fparams.theta, self.fparams.axes_dim)
        self.pe.requires_grad = False
        self.pe = self.pe.to(self.tparams.device)
        # import pdb; pdb.set_trace()

    def pack(self, x):
        return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                         ph=self.tparams.feature_size, pw=self.tparams.feature_size)

    def unpack(self, x: Tensor) -> Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(self.tparams.H / self.tparams.feature_size),
            w=math.ceil(self.tparams.W / self.tparams.feature_size),
            ph=self.tparams.feature_size,
            pw=self.tparams.feature_size,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.tparams.lr)

    def prepare(self, batch, batch_idx):
        in_img, out_img = batch
        token_y = self.pack(in_img)  # b, 3, H, W
        token_x = self.pack(out_img)  # b, 1, H, W

        b, n, d = token_x.shape
        ts = get_ts(b, True, token_x.device)
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

    def validation_step(self, batch, batch_idx):
        assert not self.model.training, "Model is still training!"
        z1, zt, token_y, token_x, ts, pe = self.prepare(batch, batch_idx)
        epoch = self.current_epoch
        with torch.no_grad():
            vtheta = self.model(img=zt, img_cond=token_y, pe=pe, timesteps=ts)
            batchwise_mse = ((z1 - token_x - vtheta) ** 2).mean(dim=list(range(1, len(token_x.shape))))
            loss = batchwise_mse.mean()
            self.log('val_loss', loss.item(), prog_bar=True, sync_dist=True)

            # if epoch % 10 == 0:
            #     tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
            #     bins = ts.detach().cpu().numpy().tolist()

            #     data = list(zip(bins, tlist))
            #     table = wandb.Table(data=data, columns=["ts", "loss"])
            #     self.logger.experiment.log({f"Losses:{epoch}": wandb.plot.line(
            #         table, "ts", "loss", title=f"time vs loss {epoch}"
            #     )})
        return loss

    @torch.no_grad()
    def denoise(self, x: Tensor, y: Tensor, guidance, timesteps: list[float]):
        for i in tqdm(range(len(timesteps) - 1), desc="Denoising"):
            t_curr, t_prev = timesteps[i], timesteps[i + 1]
            # for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)
            v = self.model(img=x, img_cond=y, pe=self.pe, timesteps=t_vec, guidance=guidance)
            x = x + v * (t_prev - t_curr)
        return x

    @torch.no_grad()
    def generate(self, y: Tensor, timesteps: list[float] | None, seed: int):
        b, c, h, w = y.shape
        assert c == 3, "Input must be RGB Image"
        assert h == self.tparams.H and w == self.tparams.W, "Input shape must match model input shape"
        x = get_noise(b, self.tparams.in_channels, h, w, y.device, y.dtype, seed)

        token_x = self.pack(x)
        token_y = self.pack(y)

        if timesteps is None:
            timesteps = torch.linspace(1, 0, 100)
            timesteps = torch.sigmoid(11 * (timesteps - 0.5))  # do we need this ?

        x = self.denoise(token_x, token_y, None, timesteps)
        return self.unpack(x)


class GenerateCallback(Callback):
    def __init__(self, imgs: Tensor, out_imgs: Tensor, timesteps: list[float] | None, seed: int):
        b, c, h, w = imgs.shape
        self.num = 16
        self.x = out_imgs[:self.num]
        self.y = imgs[:self.num]  # torchvision.transforms.Normalize(mean=[0.5]*c, std=[0.5]*c)(imgs)
        self.timesteps = timesteps
        self.seed = seed

    @staticmethod
    def to_img(l: Tensor):
        # get min value in a batch
        b, c, h, w = l.shape
        min_val = l.view(b, -1).min(dim=-1).values.view(b, 1, 1, 1)
        max_val = l.view(b, -1).max(dim=-1).values.view(b, 1, 1, 1)
        l = l - min_val
        l = l / max_val
        return l.clamp(0, 1)

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.model.eval()
        epoch = trainer.current_epoch
        with torch.no_grad():
            if trainer.current_epoch % 10 == 0:
                y = self.y.to(pl_module.device)
                out = pl_module.generate(y, self.timesteps, self.seed)
                out = self.to_img(out).cpu()
                y = y.cpu() * 0.5 + 0.5
                x = self.x.cpu() * 0.5 + 0.5

                out = repeat(out, "b c h w -> b (k c) h w", k=3)
                x = repeat(x, "b c h w -> b (k c) h w", k=3)

                grid = torch.cat((y, x, out), dim=0)
                out = torchvision.utils.make_grid(grid, nrow=grid.shape[0] // 3)
                trainer.logger.experiment.log({f"Gen:{epoch}": wandb.Image(out, caption="Generated Images")})
        pl_module.model.train()


device = torch.device("cuda") if torch.cuda.is_available() else (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
dtype = torch.float32  # torch.bfloat16 if torch.cuda.is_available() else torch.float32
head = '/home/souchakr/sensei-fs-symlink/users/souchakr/localssd'  # '/Users/souymodip/GIT/pythonProject/'
folder_path = f'{head}/data4'
val_path = f'{head}/data0'
num_workers = 11  # 1

train_params = TrainParams(
    batch_size=24,
    epochs=420000,
    lr=1e-4,
    H=32,
    W=32,
    seed=42,
    in_channels=1,
    cond_channels=3,
    feature_size=2,
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
    wandb_logger = WandbLogger(project="DiTEdge")

    trainer = l.Trainer(accelerator="gpu", devices="auto", strategy="auto",
                        callbacks=cb,
                        logger=wandb_logger,
                        # max_epochs=1, limit_train_batches=2, limit_val_batches=2,
                        # val_check_interval=0.5
                        )

    ckpt_path = '/sensei-fs/users/souchakr/RF/src/DiTEdge/2pgkchff/checkpoints/epoch=124-step=128000.ckpt'  # '/Users/souymodip/GIT/flux/CKPT/checkpoints/epoch=124-step=128000.ckpt'
    if ckpt_path:
        print(f"Loading Checkpoint from {ckpt_path}")
        rf = RF.load_from_checkpoint(ckpt_path, model=FluxImg(fparams), fluxparams=fparams, trainparams=train_params)
        rf.setup("train")
    else:
        rf = RF(FluxImg(fparams), fparams, train_params)

    trainer.fit(rf, data)
    print('Training Completed!')


if __name__ == '__main__':
    train()

    # ckpt_path ='/sensei-fs/users/souchakr/RF/src/DiTEdge/2pgkchff/checkpoints/epoch=124-step=128000.ckpt'
    # if ckpt_path:
    #     print(f"Loading Checkpoint from {ckpt_path}")
    #     rf = RF.load_from_checkpoint(ckpt_path, model=FluxImg(fparams), fluxparams=fparams, trainparams=train_params)
    # import pdb; pdb.set_trace()


