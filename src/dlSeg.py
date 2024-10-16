from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from einops import rearrange
import torch
import numpy as np


class SegDS(Dataset):
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

    @staticmethod
    def shuffle_classes(out, num_classes):
        perm = torch.randperm(num_classes)
        # print(f'perm: {perm}')
        out_int = perm.gather(0, out.flatten())
        out_int = out_int.reshape(out.size())
        return out_int

    def __getitem__(self, idx):
        idx = idx % self.num
        cond_path, out_path = self.path_pairs[idx]
        cond = Image.open(cond_path).convert('RGB')
        out = Image.open(out_path).convert('L')

        height, width = cond.size
        if self.shuffle:
            i, j, _, _ = transforms.RandomCrop.get_params(cond, output_size=(self.size, self.size))
        else:
            i, j = height // 2 - self.size // 2, width // 2 - self.size // 2

        cond = cond.crop((i, j, i + self.size, j + self.size))
        out = out.crop((i, j, i + self.size, j + self.size))

        # out_int = torch.as_tensor(np.array(out).astype('int'))//40
        # print(f'Unique values: {np.unique(out_int)}')
        # out_int = self.shuffle_classes(out_int, 6) * 40

        cond = torch.as_tensor(np.array(cond).astype('float')/255.)
        out = torch.as_tensor(np.array(out).astype('float')/255.) # out_int.to(torch.float32) / 255. #
        cond = rearrange(cond, 'h w c -> c h w')
        out = out.unsqueeze(0)

        # x = transforms.ToTensor()(x)
        # y = transforms.ToTensor()(y)

        cond = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(cond)
        out = transforms.Normalize(mean=[0.5], std=[0.5])(out)

        cond = cond.to(self.dtype)
        out = out.to(self.dtype)
        return cond, out


def get_train_loader(folder_path, size, dtype, batch_size, num_workers):
    assert os.path.exists(folder_path), f"Folder path {folder_path} does not exist"
    ds = SegDS(folder_path, shuffle=True, size=size, dtype=dtype)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)


def get_val_loader(folder_path, size, dtype, batch_size, num_workers):
    assert os.path.exists(folder_path), f"Folder path {folder_path} does not exist"
    ds = SegDS(folder_path, shuffle=False, size=size, dtype=dtype)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)


def test():
    import matplotlib.pyplot as plt
    head = '/Users/souymodip/GIT/pythonProject/' # '/home/souchakr/sensei-fs-symlink/users/souchakr/localssd' #
    folder_path = f'{head}/data_seg_val'
    val_path = f'{head}/data_seg_val'
    train_loader = get_train_loader(folder_path=folder_path, size=32,
                                    batch_size=1, num_workers=1, dtype=torch.float32)
    print(f'Number of mini batche: {len(train_loader)}')
    # import pdb; pdb.set_trace()
    counter = 0
    for img_cond, img_out in train_loader:
        print(f'img_out Shape: {img_out.shape}, range: [{img_out.min()}, {img_out.max()}]\n'
              f'img_cond Shape: {img_cond.shape}, range: [{img_cond.min()}, {img_cond.max()}]')
        fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
        x0 = img_cond[0].permute(1, 2, 0) * 0.5 + 0.5
        y0 = img_out[0].squeeze() * 0.5 + 0.5
        ax1.imshow(x0)
        ax2.imshow(torch.round(y0*255)//40)
        plt.show()

        counter += 1
        if counter > 20:
            break
        # import pdb; pdb.set_trace()
        # break


if __name__ == '__main__':
    test()