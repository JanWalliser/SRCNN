# dataset_sr.py
import os, random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

def _paired_filelist(lr_dir, hr_dir):
    lr_names = set(os.listdir(lr_dir))
    hr_names = set(os.listdir(hr_dir))
    # Schnittmenge, sortiert – sichert 1:1 Paare
    names = sorted(list(lr_names & hr_names))
    if not names:
        raise RuntimeError("Keine überlappenden Dateien in LR/HR gefunden.")
    return names

class SRPairedDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale=4, patch_hr=192, augment=True, rgb=True):
        self.lr_dir, self.hr_dir = lr_dir, hr_dir
        self.names = _paired_filelist(lr_dir, hr_dir)
        self.scale = scale
        self.patch_hr = patch_hr - (patch_hr % scale)  # sicher scale-multiple
        self.patch_lr = self.patch_hr // scale
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.rgb = rgb

    def __len__(self): return len(self.names)

    def _random_crop_pair(self, lr, hr):
        w_lr, h_lr = lr.size
        if w_lr < self.patch_lr or h_lr < self.patch_lr:
            # minimaler Fallback: zentrierter Crop mit Pad
            lr = TF.pad(lr, padding=(
                0, 0, max(0, self.patch_lr - w_lr), max(0, self.patch_lr - h_lr)
            ), fill=0)
            hr = TF.pad(hr, padding=(
                0, 0, max(0, self.patch_hr - hr.size[0]), max(0, self.patch_hr - hr.size[1])
            ), fill=0)
            w_lr, h_lr = lr.size

        x_lr = random.randint(0, w_lr - self.patch_lr)
        y_lr = random.randint(0, h_lr - self.patch_lr)
        x_hr, y_hr = x_lr * self.scale, y_lr * self.scale

        lr_patch = lr.crop((x_lr, y_lr, x_lr + self.patch_lr, y_lr + self.patch_lr))
        hr_patch = hr.crop((x_hr, y_hr, x_hr + self.patch_hr, y_hr + self.patch_hr))
        return lr_patch, hr_patch

    def _maybe_augment(self, lr, hr):
        if not self.augment: return lr, hr
        if random.random() < 0.5:
            lr = TF.hflip(lr); hr = TF.hflip(hr)
        if random.random() < 0.5:
            lr = TF.vflip(lr); hr = TF.vflip(hr)
        if random.random() < 0.5:
            # 0°, 90°, 180°, 270°
            k = random.choice([0,1,2,3])
            if k: 
                lr = TF.rotate(lr, 90*k, expand=True)
                hr = TF.rotate(hr, 90*k, expand=True)
        return lr, hr

    def __getitem__(self, idx):
        name = self.names[idx]
        lr = Image.open(os.path.join(self.lr_dir, name)).convert('RGB')
        hr = Image.open(os.path.join(self.hr_dir, name)).convert('RGB')

        # Optional: nur Y-Kanal (klassisch bei SR)
        if not self.rgb:
            lr = lr.convert('YCbCr'); hr = hr.convert('YCbCr')
            lr, _, _ = lr.split(); hr, _, _ = hr.split()

        lr, hr = self._random_crop_pair(lr, hr)
        lr, hr = self._maybe_augment(lr, hr)

        lr = self.to_tensor(lr)
        hr = self.to_tensor(hr)
        return lr, hr

def get_dataloader(lr_dir, hr_dir, batch_size=16, scale=4, patch_hr=192, augment=True,
                   num_workers=4, pin_memory=True, shuffle=True, rgb=True):
    ds = SRPairedDataset(lr_dir, hr_dir, scale, patch_hr, augment, rgb)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
