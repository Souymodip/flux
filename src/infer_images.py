import PIL.Image
from torchvision import transforms
import torch
from einops import repeat
from dlSeg import get_train_loader as loader
import lt
from flux.modelImg import FluxImg
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import imageio


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")



@torch.no_grad()
def images2tensor(img_paths):
    def load_image(img_path):
        img = PIL.Image.open(img_path)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
        return img

    imgs = [load_image(img_path) for img_path in img_paths]
    return torch.stack(imgs)


@torch.no_grad()
def to_img(rf, l):
    l = l - l.min()
    l = l / l.max()
    return rf.unpack(l).permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def prepare(rf, in_img, out_img):
    # z1 = torch.randn_like(out_img)
    token_y = rf.pack(in_img)  # b, 3, H, W
    token_x = rf.pack(out_img)  # b, 1, H, W
    token_z1 = torch.randn_like(token_x) #rf.pack(z1)  # b, 1, H, W

    b, n, d = token_x.shape
    pe = repeat(rf.pe, "... -> b ...", b=b).squeeze(1)

    return token_z1, token_x, token_y, pe


@torch.no_grad()
def infer_tensor_images(rf, tensor_images, timesteps, batch_size, folder):
    b, _, _, _ = tensor_images.shape
    device = rf.device
    print(f'Number of images: {b}, device: {device}')
    for batch in range(0, b, batch_size):
        _tensor_images = tensor_images[batch:batch+batch_size]
        b, c, h, w = _tensor_images.shape # b, 3, H, W
        z1 = torch.randn(b, 1, h, w, device=_tensor_images.device, dtype=_tensor_images.dtype) # b, 1, H, W
        token_z1 = rf.pack(z1)
        token_y = rf.pack(_tensor_images)
        pe = repeat(rf.pe, "... -> b ...", b=b).squeeze(1)
        zt = token_z1

        zt = zt.to(device)
        token_y = token_y.to(device)
        pe = pe.to(device)

        out_folder = os.path.join(folder, f'batch_{batch}')
        os.makedirs(out_folder, exist_ok=True)

        # import pdb; pdb.set_trace()

        image_paths_per_batch = [[] for _ in range(b)]
        for i in tqdm(range(len(timesteps) - 1)):
            t_vec = torch.full((b, ), timesteps[i], dtype=token_y.dtype, device=token_y.device)
            v = rf.model(img=zt, img_cond=token_y, pe=pe, timesteps=t_vec)

            zt = zt + v * (timesteps[i+1] - timesteps[i])

            if i % 10 == 0 or i == len(timesteps) - 2:
                zt_img = to_img(rf, zt)
                y_img = to_img(rf, token_y)
                for j in range(b):
                    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
                    ax1.imshow(zt_img[j])
                    ax2.imshow(y_img[j])
                    fig.suptitle(f'Timestep: {timesteps[i] :.3f}')
                    plt.savefig(os.path.join(out_folder, f'{j}_{i}.png'))
                    plt.close()
                    image_paths_per_batch[j].append(os.path.join(out_folder, f'{j}_{i}.png'))

        # convert images to gif
        for b, image_paths in enumerate(image_paths_per_batch):
            images = [imageio.v2.imread(img_path) for img_path in image_paths]
            imageio.mimsave(f'{out_folder}/output{b}.gif', images)
            for img_path in image_paths[:-1]:
                os.remove(img_path)


@torch.no_grad()
def get_RF():
    # ckpt_path = '/Users/souymodip/GIT/flux/CKPT/SEG/seg=00077625-ltrain_loss=0.005.ckpt'
    ckpt_path = '/Users/souymodip/GIT/flux/CKPT/SEG/last.ckpt'
    fparams = lt.fparams
    train_params = lt.train_params

    rf = lt.RF.load_from_checkpoint(ckpt_path, model=FluxImg(fparams), fluxparams=fparams, trainparams=train_params)
    rf.setup("eval")
    rf = rf.eval()
    rf.model.eval()
    device = get_device()
    rf = rf.to(device)
    return rf


@torch.no_grad()
def get_time_steps(deb=True):
    timesteps = torch.linspace(1, 0, 100)
    timesteps = torch.sigmoid(11 * (timesteps - 0.5))
    if deb:
        plt.plot(timesteps)
        plt.show()
    return timesteps


@torch.no_grad()
def test_model_img():
    head = '/Users/souymodip/GIT/pythonProject/'
    val_path = f'{head}/data_seg_val'
    # ckpt_path ='/Users/souymodip/GIT/flux/CKPT/checkpoints/epoch=124-step=128000.ckpt'
    # ckpt_path = '/Users/souymodip/GIT/flux/src/FMSeg/sstep=00077625-ltrain_loss=0.005.ckpt'
    # ckpt_path = '/Users/souymodip/GIT/flux/CKPT/seg=00077625-ltrain_loss=0.005.ckpt'
    rf = get_RF()

    batch_size = 8
    train_loader = loader(folder_path=val_path, size=32,
                                    batch_size=batch_size, num_workers=4, dtype=torch.float32)
    print(f'Number of mini batche: {len(train_loader)}')

    timesteps = get_time_steps()

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

        token_z1, token_x, token_y, pe = prepare(rf, x, y)
        z1 = token_z1.clone()

        folder = f'output{k}'
        os.makedirs(folder, exist_ok=True)
        losses = []

        token_x_img = to_img(rf, token_x)
        token_y_img = to_img(rf, token_y)

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

            N = max(len(timesteps) // 20, 1)
            if i % N == 0 or i == len(timesteps) - 2:
                z1_img = to_img(rf, z1)

                for j in range(batch_size):
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)
                    ax1.imshow(z1_img[j])
                    ax2.imshow(token_x_img[j])
                    ax3.imshow(token_y_img[j])
                    fig.suptitle(f'Timestep: {timesteps[i] :.3f}, loss: {losses[-1]:.3f}')
                    plt.savefig(os.path.join(folder, f'{j}_{i}.png'))
                    plt.close()
                    image_paths_per_batch[j].append(os.path.join(folder, f'{j}_{i}.png'))

        plt.plot(losses)
        plt.savefig(os.path.join(folder, 'loss.png'))
        plt.close()

        # convert images to gif
        for b, image_paths in enumerate(image_paths_per_batch):
            images = [imageio.v2.imread(img_path) for img_path in image_paths]
            imageio.mimsave(f'{folder}/output{b}.gif', images)
            for img_path in image_paths[:-1]:
                os.remove(img_path)

        if k >= 10:
            break


def infer_images(folder):
    img_paths = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith('.png')]
    tensor_images = images2tensor(img_paths)
    result_folder = os.path.join(folder, 'results')
    os.makedirs(result_folder, exist_ok=True)

    infer_tensor_images(get_RF(), tensor_images, get_time_steps(), 8, result_folder)



if __name__ == "__main__":
    # run_denoise()
    # test_model_img()
    infer_images('/Users/souymodip/GIT/flux/assets/crops32')
