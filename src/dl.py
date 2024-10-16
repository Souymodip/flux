from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from einops import rearrange
import torch
import numpy as np


class DS(Dataset):
    def __init__(self, folder_path, shuffle, size, dtype):
        self.folder_path = folder_path
        x_paths, y_paths = [], []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if 'f' in file:
                    x_paths.append(os.path.join(root, file))
                if 's' in file:
                    y_paths.append(os.path.join(root, file))
        assert len(x_paths) == len(y_paths)
        x_paths.sort()
        y_paths.sort()
        self.path_pairs = list(zip(x_paths, y_paths))
        self.shuffle = shuffle
        self.size = size
        self.dtype = dtype
        self.mult = 32
        self.num = len(self.path_pairs)

    def __len__(self):
        return self.mult * self.num

    def __getitem__(self, idx):
        idx = idx % self.num
        x_path, y_path = self.path_pairs[idx]
        x = Image.open(x_path).convert('RGB')
        y = Image.open(y_path).convert('L')

        height, width = x.size
        if self.shuffle:
            i, j, _, _ = transforms.RandomCrop.get_params(x, output_size=(self.size, self.size))
        else:
            i, j = height // 2 - self.size // 2, width // 2 - self.size // 2

        x = x.crop((i, j, i + self.size, j + self.size))
        y = y.crop((i, j, i + self.size, j + self.size))

        x = torch.as_tensor(np.array(x).astype('float')/255.)
        y = torch.as_tensor(np.array(y).astype('float')/255.)
        x = rearrange(x, 'h w c -> c h w')
        y = y.unsqueeze(0)

        # x = transforms.ToTensor()(x)
        # y = transforms.ToTensor()(y)

        x = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(x)
        y = transforms.Normalize(mean=[0.5], std=[0.5])(y)

        x = x.to(self.dtype)
        y = y.to(self.dtype)
        return x, y


class DS2(Dataset):
    def __init__(self, folder_path, shuffle, size, dtype):
        self.folder_path = folder_path
        x_paths, y_paths = [], []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if 'f' in file:
                    x_paths.append(os.path.join(root, file))
                if 's' in file:
                    y_paths.append(os.path.join(root, file))
        assert len(x_paths) == len(y_paths)
        x_paths.sort()
        y_paths.sort()
        self.path_pairs = list(zip(x_paths, y_paths))
        self.shuffle = shuffle
        self.size = size
        self.dtype = dtype
        self.mult = 32
        self.num = len(self.path_pairs)

    def __len__(self):
        return self.mult * self.num

    def __getitem__(self, idx):
        idx = idx % self.num
        x_path, y_path = self.path_pairs[idx]
        x = Image.open(x_path).convert('RGB')
        y = Image.open(y_path).convert('L')

        height, width = x.size
        if self.shuffle:
            i, j, _, _ = transforms.RandomCrop.get_params(x, output_size=(self.size, self.size))
        else:
            i, j = height // 2 - self.size // 2, width // 2 - self.size // 2

        i2 = 2 * i
        j2 = 2 * j
        size2 = 2 * self.size

        x = x.crop((i, j, i + self.size, j + self.size))
        y = y.crop((i2, j2, i2 + size2, j2 + size2))

        x = torch.as_tensor(np.array(x).astype('float')/255.)
        y = torch.as_tensor(np.array(y).astype('float')/255.)
        x = rearrange(x, 'h w c -> c h w')
        y = y.unsqueeze(0)

        # x = transforms.ToTensor()(x)
        # y = transforms.ToTensor()(y)

        x = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(x)
        y = transforms.Normalize(mean=[0.5], std=[0.5])(y)

        x = x.to(self.dtype)
        y = y.to(self.dtype)
        return x, y


def get_train_loader(folder_path, size, dtype, batch_size, num_workers):
    assert os.path.exists(folder_path), f"Folder path {folder_path} does not exist"
    ds = DS2(folder_path, shuffle=True, size=size, dtype=dtype)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)


def get_val_loader(folder_path, size, dtype, batch_size, num_workers):
    assert os.path.exists(folder_path), f"Folder path {folder_path} does not exist"
    ds = DS(folder_path, shuffle=False, size=size, dtype=dtype)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

def test():
    import matplotlib.pyplot as plt
    head = '/Users/souymodip/GIT/pythonProject/' # '/home/souchakr/sensei-fs-symlink/users/souchakr/localssd' #
    folder_path = f'{head}/_data'
    val_path = f'{head}/data0'
    train_loader = get_train_loader(folder_path=folder_path, size=32,
                                    batch_size=1, num_workers=4, dtype=torch.float32)
    print(f'Number of mini batche: {len(train_loader)}')
    import pdb; pdb.set_trace()
    for img_cond, img_out in train_loader:
        print(f'img_out Shape: {img_out.shape}, range: [{img_out.min()}, {img_out.max()}]\n'
              f'img_cond Shape: {img_cond.shape}, range: [{img_cond.min()}, {img_cond.max()}]')
        fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
        x0 = img_cond[0].permute(1, 2, 0) * 0.5 + 0.5
        y0 = img_out[0].squeeze() * 0.5 + 0.5
        ax1.imshow(x0)
        ax2.imshow(y0)
        plt.show()
        # import pdb; pdb.set_trace()
        # break


if __name__ == '__main__':
    test()