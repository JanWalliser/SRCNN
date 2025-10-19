# dataset.py — Robust LR/HR pairing (by basename) and alignment
from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# -----------------------------
# Helpers
# -----------------------------

def _list_images(folder: str) -> List[str]:
    """List image filenames in a folder (no subdirs). Case-insensitive by extension."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f.lower())[1] in exts]
    )


def _basename_noext(path: str) -> str:
    """Return filename without extension (case-insensitive)."""
    return os.path.splitext(os.path.basename(path))[0]


def _build_basename_index(folder: str) -> Dict[str, str]:
    """Map basename (without extension, lowercased) -> filename with extension found in folder.
    If multiple files share the same basename with different extensions, the first alphabetic filename wins.
    """
    idx: Dict[str, str] = {}
    for fname in _list_images(folder):
        base = _basename_noext(fname).lower()
        if base not in idx:
            idx[base] = fname
    return idx


# -----------------------------
# Alignment utilities
# -----------------------------

def _pair_center_crop(lr_img: Image.Image, hr_img: Image.Image, scale: int,
                      lr_w: int, lr_h: int) -> Tuple[Image.Image, Image.Image]:
    """Centered crop of LR to (lr_w,lr_h) and HR to (lr_w*scale, lr_h*scale)."""
    hr_w, hr_h = lr_w * scale, lr_h * scale
    lr_x = max((lr_img.width  - lr_w) // 2, 0)
    lr_y = max((lr_img.height - lr_h) // 2, 0)
    hr_x = max((hr_img.width  - hr_w) // 2, 0)
    hr_y = max((hr_img.height - hr_h) // 2, 0)
    lr_c = lr_img.crop((lr_x, lr_y, lr_x + lr_w, lr_y + lr_h))
    hr_c = hr_img.crop((hr_x, hr_y, hr_x + hr_w, hr_y + hr_h))
    return lr_c, hr_c


def _align_pair_by_center_crop(lr_img: Image.Image, hr_img: Image.Image, scale: int) -> Tuple[Image.Image, Image.Image]:
    """Align LR and HR by centered cropping to the largest mutually consistent sizes (HR = LR * scale)."""
    lrW, lrH = lr_img.size
    hrW, hrH = hr_img.size

    max_lr_w = min(lrW, hrW // scale)
    max_lr_h = min(lrH, hrH // scale)
    if max_lr_w <= 0 or max_lr_h <= 0:
        max_lr_w = max(1, min(lrW, hrW // scale))
        max_lr_h = max(1, min(lrH, hrH // scale))

    lr_c, hr_c = _pair_center_crop(lr_img, hr_img, scale, max_lr_w, max_lr_h)
    return lr_c, hr_c


def _align_pair_by_top_left(lr_img: Image.Image, hr_img: Image.Image, scale: int) -> Tuple[Image.Image, Image.Image]:
    """Align LR and HR by *top-left* cropping (0,0). Matches pipelines that did mod-crop + downscale from top-left.
    We crop both images from (0,0) to mutually consistent sizes so that HR = LR * scale exactly.
    """
    lrW, lrH = lr_img.size
    hrW, hrH = hr_img.size

    max_lr_w = min(lrW, hrW // scale)
    max_lr_h = min(lrH, hrH // scale)
    if max_lr_w <= 0 or max_lr_h <= 0:
        max_lr_w = max(1, min(lrW, hrW // scale))
        max_lr_h = max(1, min(lrH, hrH // scale))

    lr_c = lr_img.crop((0, 0, max_lr_w, max_lr_h))
    hr_c = hr_img.crop((0, 0, max_lr_w * scale, max_lr_h * scale))
    return lr_c, hr_c


# -----------------------------
# Dataset
# -----------------------------

class SRPairedDataset(Dataset):
    """
    Lädt LR/HR-Paare aus zwei Ordnern und richtet sie pixelgenau aus.

    Matching:
      - Paart nach *Basename* (ohne Dateiendung). Damit dürfen LR/HR unterschiedliche Extensions haben.
        Beispiel: `IMG_001.png` (LR) ↔ `IMG_001.jpg` (HR).

    Alignment:
      - `align = 'topleft'`  → passgenauer Top-Left-Crop (empfohlen, wenn LR via mod-crop + Downscale aus HR erzeugt wurde)
      - `align = 'center'`   → zentrierter Crop auf kompatible Größen

    Cropping & Augment:
      - Train: zufällige, koordinierte Crops (HR-Patch-Größe `patch_hr`; LR-Patch = `patch_hr // scale`)
      - Val/Test (`center_crop=True`): deterministischer Center-Crop, oder Full-Image wenn kleiner als Patch.

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
        center_crop: bool = False,   # True → deterministisch (Val/Test)
        align: str = "topleft",      # 'topleft' | 'center'
    ):
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = int(scale)
        # Ensure patch_hr is divisible by scale; otherwise, round down to nearest multiple
        if patch_hr % self.scale != 0:
            patch_hr = (patch_hr // self.scale) * self.scale
        self.patch_hr = int(patch_hr)
        self.patch_lr = self.patch_hr // self.scale
        self.augment = bool(augment)
        self.rgb = bool(rgb)
        self.center_crop = bool(center_crop)
        self.align = (align or "topleft").lower()
        if self.align not in {"topleft", "center"}:
            raise ValueError(f"Unknown align='{align}'. Use 'topleft' or 'center'.")

        # Build basename indices and pair files
        lr_idx = _build_basename_index(lr_dir)
        hr_idx = _build_basename_index(hr_dir)
        common_keys = sorted(set(lr_idx.keys()) & set(hr_idx.keys()))
        self.pairs: List[Tuple[str, str]] = [
            (lr_idx[k], hr_idx[k]) for k in common_keys
        ]

        if not self.pairs:
            # Fallback message that hints common pitfalls (e.g., different extensions or wrong scale folders)
            raise RuntimeError(
                "Keine gemeinsamen LR/HR-Dateien gefunden (Basename-Matching).\n"
                f"LR-Ordner: {lr_dir}\nHR-Ordner: {hr_dir}\n"
                "Prüfe: gleiche Basenames (unabhängig von .png/.jpg), richtigen 'scale' und Ordnerzuordnung."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_pair(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        lr_name, hr_name = self.pairs[idx]
        lr_path = os.path.join(self.lr_dir, lr_name)
        hr_path = os.path.join(self.hr_dir, hr_name)

        lr_img = Image.open(lr_path)
        hr_img = Image.open(hr_path)

        if self.rgb:
            lr_img = lr_img.convert("RGB")
            hr_img = hr_img.convert("RGB")

        # Geometrisches Alignment (Top-Left oder Center)
        if self.align == "topleft":
            lr_img, hr_img = _align_pair_by_top_left(lr_img, hr_img, self.scale)
        else:
            lr_img, hr_img = _align_pair_by_center_crop(lr_img, hr_img, self.scale)
        return lr_img, hr_img

    def _random_crop_pair(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Zufälliger koordinierter Crop. Fällt auf Center-Crop zurück, wenn das Bild zu klein ist."""
        if (
            lr_img.width < self.patch_lr
            or lr_img.height < self.patch_lr
            or hr_img.width < self.patch_hr
            or hr_img.height < self.patch_hr
        ):
            return self._center_crop_pair(lr_img, hr_img)

        # Wähle linke obere Ecke im LR-Raum und skaliere Koordinaten für HR
        x_lr = random.randint(0, lr_img.width - self.patch_lr)
        y_lr = random.randint(0, lr_img.height - self.patch_lr)
        x_hr, y_hr = x_lr * self.scale, y_lr * self.scale

        lr_patch = lr_img.crop((x_lr, y_lr, x_lr + self.patch_lr, y_lr + self.patch_lr))
        hr_patch = hr_img.crop((x_hr, y_hr, x_hr + self.patch_hr, y_hr + self.patch_hr))
        return lr_patch, hr_patch

    def _center_crop_pair(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Deterministischer koordinierter Center-Crop.
        Wenn das Bild kleiner als der gewünschte Patch ist → benutze Full-Image (kein Padding).
        """
        if (
            lr_img.width < self.patch_lr
            or lr_img.height < self.patch_lr
            or hr_img.width < self.patch_hr
            or hr_img.height < self.patch_hr
        ):
            return lr_img, hr_img

        lr_x = (lr_img.width - self.patch_lr) // 2
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
        if random.random() < 0.5:
            lr_img = TF.hflip(lr_img)
            hr_img = TF.hflip(hr_img)
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
        lr_t = TF.to_tensor(lr_img).clamp(0.0, 1.0)
        hr_t = TF.to_tensor(hr_img).clamp(0.0, 1.0)
        return lr_t, hr_t


# -----------------------------
# Factory for DataLoader
# -----------------------------

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
    align: str = "topleft",
) -> DataLoader:
    """
    Erzeugt einen DataLoader für SRPairedDataset.

    Parameter-Hinweise:
      - `align='topleft'` → standardmäßig aktiv, um mit Top-Left-Mod-Crop/Downscale-Pipelines kompatibel zu sein.
      - Für Validierung/Tests setze `center_crop=True` (deterministisch) – siehe train_baseline.py.
    """
    ds = SRPairedDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        scale=scale,
        patch_hr=patch_hr,
        augment=augment,
        rgb=rgb,
        center_crop=center_crop,
        align=align,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
