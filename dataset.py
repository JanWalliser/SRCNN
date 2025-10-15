# dataset.py
from __future__ import annotations

import os
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random


def _list_images(folder: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([f for f in os.listdir(folder) if os.path.splitext(f.lower())[1] in exts])


def _center_crop(img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    return img.crop((x, y, x + w, y + h))


def _pair_center_crop(lr_img: Image.Image, hr_img: Image.Image, scale: int,
                      lr_w: int, lr_h: int) -> Tuple[Image.Image, Image.Image]:
    """Zentriert HR und LR so, dass Größe exakt (lr_w, lr_h) und (lr_w*scale, lr_h*scale) ist."""
    hr_w, hr_h = lr_w * scale, lr_h * scale
    lr_x = max((lr_img.width  - lr_w) // 2, 0)
    lr_y = max((lr_img.height - lr_h) // 2, 0)
    hr_x = max((hr_img.width  - hr_w) // 2, 0)
    hr_y = max((hr_img.height - hr_h) // 2, 0)
    lr_c = _center_crop(lr_img, lr_x, lr_y, lr_w, lr_h)
    hr_c = _center_crop(hr_img, hr_x, hr_y, hr_w, hr_h)
    return lr_c, hr_c


def _align_pair_by_center_crop(lr_img: Image.Image, hr_img: Image.Image, scale: int) -> Tuple[Image.Image, Image.Image]:
    """
    Stellt sicher, dass LR und HR zueinander passen (HR = LR * scale).
    Cropt zentriert auf die maximal mögliche gemeinsame Größe (verhindert Off-by-one-Mismatches).
    """
    lrW, lrH = lr_img.size
    hrW, hrH = hr_img.size

    # maximal mögliche LR-Größe, die zu HR passt
    max_lr_w = min(lrW, hrW // scale)
    max_lr_h = min(lrH, hrH // scale)
    if max_lr_w <= 0 or max_lr_h <= 0:
        # Fallback: nutze kleinste sinnvolle zentrale Fläche (verhindert leere Crops)
        max_lr_w = max(1, min(lrW, hrW // scale))
        max_lr_h = max(1, min(lrH, hrH // scale))

    # Zentrierte Crops auf passende Größen
    lr_c, hr_c = _pair_center_crop(lr_img, hr_img, scale, max_lr_w, max_lr_h)
    return lr_c, hr_c


class SRPairedDataset(Dataset):
    """
    Lädt LR/HR-Paare aus zwei Ordnern.
    - Erwartet, dass Dateinamen übereinstimmen (Schnittmenge wird verwendet).
    - Sicherer Größenabgleich: zentrierter Crop auf zueinander passende Dimensionen.
    - Train: zufälliger koordinierter Crop + optionale Augmentierung.
    - Val/Test: deterministischer Center-Crop (oder Full-Image falls zu klein).
    Rückgabe: (lr_tensor, hr_tensor) in [0,1], float32.
    """

    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        *,
        scale: int = 4,
        patch_hr: int = 192,
        augment: bool = True,
        rgb: bool = True,
        center_crop: bool = False,  # True -> deterministisch (für Val/Test)
    ):
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = int(scale)
        self.patch_hr = int(patch_hr)
        self.patch_lr = self.patch_hr // self.scale
        self.augment = bool(augment)
        self.rgb = bool(rgb)
        self.center_crop = bool(center_crop)

        lr_files = set(_list_images(lr_dir))
        hr_files = set(_list_images(hr_dir))
        self.files = sorted(list(lr_files & hr_files))
        if not self.files:
            raise RuntimeError(f"Keine gemeinsamen Dateien in\nLR: {lr_dir}\nHR: {hr_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _load_pair(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        name = self.files[idx]
        lr_path = os.path.join(self.lr_dir, name)
        hr_path = os.path.join(self.hr_dir, name)

        lr_img = Image.open(lr_path)
        hr_img = Image.open(hr_path)

        if self.rgb:
            lr_img = lr_img.convert("RGB")
            hr_img = hr_img.convert("RGB")

        # Größen sauber aufeinander ausrichten (zentrierter Crop, kein Padding)
        lr_img, hr_img = _align_pair_by_center_crop(lr_img, hr_img, self.scale)
        return lr_img, hr_img

    def _random_crop_pair(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Zufälliger koordinierter Crop. Fällt auf Center-Crop zurück, wenn das Bild zu klein ist."""
        if lr_img.width < self.patch_lr or lr_img.height < self.patch_lr \
           or hr_img.width < self.patch_hr or hr_img.height < self.patch_hr:
            # zu klein -> deterministic fallback
            return self._center_crop_pair(lr_img, hr_img)

        # Zufällige obere linke Ecke im LR
        x_lr = random.randint(0, lr_img.width  - self.patch_lr)
        y_lr = random.randint(0, lr_img.height - self.patch_lr)
        x_hr, y_hr = x_lr * self.scale, y_lr * self.scale

        lr_patch = lr_img.crop((x_lr, y_lr, x_lr + self.patch_lr, y_lr + self.patch_lr))
        hr_patch = hr_img.crop((x_hr, y_hr, x_hr + self.patch_hr, y_hr + self.patch_hr))
        return lr_patch, hr_patch

    def _center_crop_pair(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Deterministischer koordinierter Center-Crop.
        - Wenn das Bild zu klein ist → benutze Full-Image (kein Padding!).
        """
        if lr_img.width < self.patch_lr or lr_img.height < self.patch_lr \
           or hr_img.width < self.patch_hr or hr_img.height < self.patch_hr:
            # Full-Image (bereits zueinander ausgerichtet), vermeidet schwarze Patches
            return lr_img, hr_img

        # normaler koordinierter Center-Crop
        lr_x = (lr_img.width  - self.patch_lr) // 2
        lr_y = (lr_img.height - self.patch_lr) // 2
        hr_x = lr_x * self.scale
        hr_y = lr_y * self.scale

        lr_patch = lr_img.crop((lr_x, lr_y, lr_x + self.patch_lr, lr_y + self.patch_lr))
        hr_patch = hr_img.crop((hr_x, hr_y, hr_x + self.patch_hr, hr_y + self.patch_hr))
        return lr_patch, hr_patch

    def _maybe_augment(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Einfache Flip-Augmentierungen, synchron auf LR/HR."""
        if not self.augment:
            return lr_img, hr_img
        # Horizontaler Flip (50%)
        if random.random() < 0.5:
            lr_img = TF.hflip(lr_img)
            hr_img = TF.hflip(hr_img)
        # Vertikaler Flip (50%)
        if random.random() < 0.5:
            lr_img = TF.vflip(lr_img)
            hr_img = TF.vflip(hr_img)
        return lr_img, hr_img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_img, hr_img = self._load_pair(idx)

        # Crop-Strategie
        if self.center_crop:
            lr_img, hr_img = self._center_crop_pair(lr_img, hr_img)
        else:
            lr_img, hr_img = self._random_crop_pair(lr_img, hr_img)
            lr_img, hr_img = self._maybe_augment(lr_img, hr_img)

        # ToTensor → float32 in [0,1]
        lr_t = TF.to_tensor(lr_img)
        hr_t = TF.to_tensor(hr_img)

        # Sicherheits-Clamp (numerische Stabilität, Bicubic-Overshoot abbügeln falls vorher resampled)
        lr_t = lr_t.clamp(0.0, 1.0)
        hr_t = hr_t.clamp(0.0, 1.0)
        return lr_t, hr_t


def get_dataloader(
    lr_dir: str,
    hr_dir: str,
    *,
    batch_size: int,
    scale: int,
    patch_hr: int,
    augment: bool,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    rgb: bool = True,
    center_crop: bool = False,
) -> DataLoader:
    """
    Erzeugt einen DataLoader für SRPairedDataset.

    center_crop:
      - Train: False (random crops + Augment)
      - Val/Test: True  (deterministische Center-Crops oder Full-Image, kein Padding)
    """
    ds = SRPairedDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        scale=scale,
        patch_hr=patch_hr,
        augment=augment,
        rgb=rgb,
        center_crop=center_crop,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
