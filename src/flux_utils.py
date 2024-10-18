import flux.util as util
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
from flux.sampling import get_noise
import matplotlib.pyplot as plt


def image_to_tensor(img_path):
    img = Image.open(img_path)
    img = transforms.ToTensor()(img)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return img.unsqueeze(0)


@torch.no_grad()
def test(img):
    name = "flux-schnell"
    device = "cpu"
    ae_model = util.load_ae(name, device)
    # img = torch.randn(1, 3, 128, 128)
    import pdb; pdb.set_trace()
    b, c, h, w = img.shape
    x = get_noise(1, h, w, device=img.device, dtype=img.dtype, seed=42)
    encoded = ae_model.encode(img)
    decoded = ae_model.decode(encoded)
    d_img = decoded[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
    ax1.imshow(img[0].permute(1, 2, 0).cpu().numpy())
    ax2.imshow(d_img)
    plt.show()


if __name__ == "__main__":
    img_path = "/Users/souymodip/GIT/flux/assets/crop128/3.png"
    img = image_to_tensor(img_path)
    test(img)
