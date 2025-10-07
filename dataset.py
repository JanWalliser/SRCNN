import os

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform
        self.image_names = os.listdir(low_res_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        low_res_path = os.path.join(self.low_res_dir, self.image_names[idx])
        high_res_path = os.path.join(self.high_res_dir, self.image_names[idx])

        low_res = Image.open(low_res_path).convert('RGB')  # Schwarz-Weiß
        high_res = Image.open(high_res_path).convert('RGB')

        if self.transform:
            low_res = self.transform(low_res)
            high_res = self.transform(high_res)

        return low_res, high_res


def get_dataloader(low_res_dir, high_res_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((800, 1200)),  # Einheitliche Größe für alle Bilder
        transforms.ToTensor(),
    ])
    dataset = ImageDataset(low_res_dir, high_res_dir, transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)