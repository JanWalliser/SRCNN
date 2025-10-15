# train_baseline.py
import os
import math
import time
import yaml
import argparse

import torch
import torch.nn.functional as F
from torch import optim
from torchvision.utils import save_image

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
    """Optional: Luma (BT.601) für Y-PSNR. Erwartet x in [0,1], [B,3,H,W]."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.257 * r + 0.504 * g + 0.098 * b + (16.0 / 255.0)


# -----------------------------
# Training & Validierung
# -----------------------------
def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    amp=True,
    grad_clip=1.0,
    use_y=False,
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
            loss = F.l1_loss(rgb_to_y(sr), rgb_to_y(hr)) if use_y else F.l1_loss(sr, hr)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_meter += loss.item()

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
    use_y=False,
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

        # optional Y-Kanal für PSNR
        if use_y:
            sr_m, hr_m, lr_m = rgb_to_y(sr), rgb_to_y(hr), rgb_to_y(lr_up)
        else:
            sr_m, hr_m, lr_m = sr, hr, lr_up

        total_psnr += psnr(sr_m, hr_m).item()
        total_bicubic += psnr(lr_m, hr_m).item()
        n += 1

        # erstes Beispiel speichern
        if i == 0:
            tag = f"ep{epoch:03d}" if epoch is not None else "epX"
            save_image(lr_up, f"{out_dir}/{tag}_lrup.png")
            save_image(sr, f"{out_dir}/{tag}_sr.png")
            save_image(hr, f"{out_dir}/{tag}_hr.png")

    return total_psnr / max(1, n), total_bicubic / max(1, n)


# -----------------------------
# Main-Trainingsloop
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
        center_crop=True   # wichtig!
    )

    # ------------------ Modell, Optimizer, LR-Sched ------------------
    model = build_model(cfg["variant"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 0.0))

    if cfg.get("cosine", True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg.get("lr_min", 0.0)
        )
    else:
        milestones = [int(cfg["epochs"] * 0.5), int(cfg["epochs"] * 0.8)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

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
            model, train_loader, optimizer, device,
            amp=bool(cfg.get("amp", True)),
            grad_clip=cfg.get("grad_clip", 1.0),
            use_y=False,
            device_type=device_type
        )

        scheduler.step()

        # ------------------ Validierung ------------------
        if epoch % cfg.get("val_every", 1) == 0:
            val_psnr, bicubic_psnr = validate(
                model, val_loader, device,
                epoch=epoch, out_dir=debug_dir, use_y=False
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
