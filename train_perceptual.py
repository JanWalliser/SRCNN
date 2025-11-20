# train_perceptual.py
import os
import math
import time
import yaml
import argparse

import torch
import torch.nn.functional as F
from torch import optim
from torchvision.utils import save_image
from torchvision import models

from dataset import get_dataloader
from srcnn_model import build_model
from utils import save_checkpoint, load_checkpoint, psnr


# -----------------------------
# Hilfsfunktionen
# -----------------------------
def upsample_to_hr(lr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """Bicubic-Upsampling von LR auf exakt die HR-Shape (HR→HR-Refinement-Setup)."""
    return F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """BT.601 Luma (Y) in [0,1], erwartet x=[B,3,H,W] in [0,1]."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.257 * r + 0.504 * g + 0.098 * b + (16.0 / 255.0)


# -----------------------------
# VGG19 Feature-Extractor & Perceptual Loss
# -----------------------------
class VGGFeatures(torch.nn.Module):
    """
    Extrahiert Feature-Maps aus VGG19 (ImageNet-pretrained) an bestimmten 'relu'-Indices.
    Erwartet Eingaben in [0,1]; wendet ImageNet-Normierung intern an.
    """
    def __init__(self, layers=(8, 17, 26)):  # relu2_2, relu3_3, relu4_3 in torchvision vgg19.features
        super().__init__()
        # Kompatibel mit neueren & älteren torchvision-Versionen:
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        except Exception:
            vgg = models.vgg19(pretrained=True)
        self.features = vgg.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

        self.layers = set(int(i) for i in layers)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        feats = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.layers:
                feats.append(x)
        return feats


class PerceptualLoss(torch.nn.Module):
    """
    L_total = λ_l1 * L1(sr, hr) + λ_perc * Σ L1(VGG(sr)_k, VGG(hr)_k) + λ_tv * TV(sr)
    """
    def __init__(self, vgg_layers=(8, 17, 26), w_l1=1.0, w_perc=0.1, w_tv=0.0):
        super().__init__()
        self.vgg = VGGFeatures(layers=vgg_layers)
        self.w_l1 = float(w_l1)
        self.w_perc = float(w_perc)
        self.w_tv = float(w_tv)

    @staticmethod
    def tv_loss(x: torch.Tensor) -> torch.Tensor:
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return (dx.abs().mean() + dy.abs().mean())

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.w_l1 > 0:
            loss = loss + self.w_l1 * F.l1_loss(sr, hr)

        if self.w_perc > 0:
            sr_feats = self.vgg(sr)
            hr_feats = self.vgg(hr)
            perc = sum(F.l1_loss(a, b) for a, b in zip(sr_feats, hr_feats))
            loss = loss + self.w_perc * perc

        if self.w_tv > 0:
            loss = loss + self.w_tv * self.tv_loss(sr)

        return loss


# -----------------------------
# Training & Validierung
# -----------------------------
def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn: PerceptualLoss,
    device,
    *,
    amp=True,
    grad_clip=1.0,
    device_type="cuda",
):
    from torch.amp import GradScaler, autocast

    model.train()
    scaler = GradScaler(device_type, enabled=amp)
    loss_meter = 0.0

    for it, (lr, hr) in enumerate(loader):
        lr = lr.to(device, non_blocking=True).float()
        hr = hr.to(device, non_blocking=True).float()

        lr_up = upsample_to_hr(lr, hr).clamp(0, 1)
        hr = hr.clamp(0, 1)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type, enabled=amp):
            sr = model(lr_up).clamp(0, 1)
            loss = loss_fn(sr, hr)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_meter += float(loss.detach())

        if it == 0:
            print(
                f"[TRAIN] lr_up min/max: {lr_up.min():.4f}/{lr_up.max():.4f} | "
                f"hr min/max: {hr.min():.4f}/{hr.max():.4f} | "
                f"sr min/max: {sr.min():.4f}/{sr.max():.4f} | "
                f"nan(sr)={torch.isnan(sr).any().item()}"
            )

    return loss_meter / max(1, len(loader))


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    *,
    epoch=None,
    out_dir="debug_val",
    use_y_psnr=True,
):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    total_psnr, total_bicubic = 0.0, 0.0
    n = 0

    for i, (lr, hr) in enumerate(loader):
        lr = lr.to(device, non_blocking=True).float()
        hr = hr.to(device, non_blocking=True).float()

        lr_up = upsample_to_hr(lr, hr).clamp(0, 1)
        hr = hr.clamp(0, 1)
        sr = model(lr_up).clamp(0, 1)

        if i == 0:
            print(
                f"[VAL]   lr_up min/max: {lr_up.min():.4f}/{lr_up.max():.4f} | "
                f"hr min/max: {hr.min():.4f}/{hr.max():.4f} | "
                f"sr min/max: {sr.min():.4f}/{sr.max():.4f} | "
                f"nan(sr)={torch.isnan(sr).any().item()}"
            )

        if use_y_psnr:
            sr_m, hr_m, lr_m = rgb_to_y(sr), rgb_to_y(hr), rgb_to_y(lr_up)
        else:
            sr_m, hr_m, lr_m = sr, hr, lr_up

        total_psnr += psnr(sr_m, hr_m).item()
        total_bicubic += psnr(lr_m, hr_m).item()
        n += 1

        if i == 0:
            tag = f"ep{epoch:03d}" if epoch is not None else "epX"
            save_image(lr_up, f"{out_dir}/{tag}_lrup.png")
            save_image(sr,    f"{out_dir}/{tag}_sr.png")
            save_image(hr,    f"{out_dir}/{tag}_hr.png")

    return total_psnr / max(1, n), total_bicubic / max(1, n)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    torch.manual_seed(cfg.get("seed", 42))

    # ------------------ Dataloader ------------------
    train_loader = get_dataloader(
        cfg["lr_train"], cfg["hr_train"],
        batch_size=cfg["batch_size"],
        scale=cfg["scale"],
        patch_hr=cfg["patch_hr"],
        augment=True,
        num_workers=cfg.get("workers", 4),
        pin_memory=True,
        shuffle=True,
        rgb=True,
        center_crop=False
    )

    val_loader = get_dataloader(
        cfg["lr_val"], cfg["hr_val"],
        batch_size=1,
        scale=cfg["scale"],
        patch_hr=cfg["patch_hr"],
        augment=False,
        num_workers=cfg.get("workers", 4),
        pin_memory=True,
        shuffle=False,
        rgb=True,
        center_crop=True
    )

    # ------------------ Modell, Optimizer, Scheduler ------------------
    model = build_model(cfg["variant"]).to(device)  # low|medium|high bleiben identisch :contentReference[oaicite:1]{index=1}

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("wd", 0.0)
    )

    if cfg.get("cosine", True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg.get("lr_min", 0.0)
        )
    else:
        milestones = [int(cfg["epochs"] * 0.5), int(cfg["epochs"] * 0.8)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    # ------------------ Perceptual-Loss-Setup ------------------
    loss_fn = PerceptualLoss(
        vgg_layers=tuple(cfg.get("vgg_layers", (8, 17, 26))),  # relu2_2, relu3_3, relu4_3
        w_l1=float(cfg.get("w_l1", 1.0)),
        w_perc=float(cfg.get("w_perc", 0.1)),
        w_tv=float(cfg.get("w_tv", 0.0)),
    ).to(device)

    # ------------------ Resume ------------------
    start_epoch, best_psnr = 1, -math.inf
    if cfg.get("resume"):
        ckpt = load_checkpoint(cfg["resume"], model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        if isinstance(ckpt, dict):
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_psnr = float(ckpt.get("best_psnr", -math.inf) or -math.inf)

    out_dir = cfg["out_dir"]
    run_name = cfg["run_name"]
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # ------------------ Trainingsloop ------------------
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            amp=bool(cfg.get("amp", True)),
            grad_clip=cfg.get("grad_clip", 1.0),
            device_type=device_type
        )

        scheduler.step()

        # ------------------ Validierung ------------------
        if epoch % cfg.get("val_every", 1) == 0:
            val_psnr, bicubic_psnr = validate(
                model, val_loader, device,
                epoch=epoch, out_dir=debug_dir,
                use_y_psnr=bool(cfg.get("use_y_psnr", True))
            )

            elapsed = time.time() - t0
            print(
                f"[{epoch:04d}/{cfg['epochs']}] loss={train_loss:.4f} | "
                f"val_psnr={val_psnr:.2f} dB (bicubic={bicubic_psnr:.2f} dB) | "
                f"best={max(best_psnr, val_psnr):.2f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
            )

            # ------------------ Checkpoints ------------------
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    os.path.join(out_dir, f"{run_name}_best.pt"),
                    epoch, model, optimizer=optimizer, scheduler=scheduler, best_psnr=best_psnr,
                    extra={"val_psnr_bicubic": bicubic_psnr}
                )

            save_checkpoint(
                os.path.join(out_dir, f"{run_name}_latest.pt"),
                epoch, model, optimizer=optimizer, scheduler=scheduler, best_psnr=best_psnr,
                extra={"val_psnr_bicubic": bicubic_psnr}
            )


if __name__ == "__main__":
    main()
