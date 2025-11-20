# train_gan.py
import os, math, time, yaml, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.utils import save_image

from dataset import get_dataloader
from srcnn_model import build_model
from utils import save_checkpoint, load_checkpoint, psnr

# ---------- Helper ----------
def upsample_to_hr(lr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    return F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)

def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.257 * r + 0.504 * g + 0.098 * b + (16.0 / 255.0)

# ---------- PatchGAN Discriminator ----------
class PatchDiscriminator(nn.Module):
    """
    70x70-PatchGAN-ähnlich: klassifiziert Bild-Patches als real/fake.
    Erwartet Eingabe in [0,1], Shape [B,3,H,W].
    """
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        def block(ci, co, ks=3, s=2, p=1, bn=True):
            layers = [nn.Conv2d(ci, co, ks, s, p)]
            if bn: layers.append(nn.BatchNorm2d(co))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            block(base_ch, base_ch, 3, 2, 1, bn=True),
            block(base_ch, base_ch*2, 3, 1, 1, bn=True),
            block(base_ch*2, base_ch*2, 3, 2, 1, bn=True),
            block(base_ch*2, base_ch*4, 3, 1, 1, bn=True),
            block(base_ch*4, base_ch*4, 3, 2, 1, bn=True),
            nn.Conv2d(base_ch*4, 1, 3, 1, 1)  # Patch-Logits
        )

    def forward(self, x):
        return self.net(x)

# ---------- Train / Val ----------
def train_one_epoch_gan(
    G, D, loader, optG, optD, device, *,
    amp=True, grad_clip=1.0,
    l1_weight=1.0, adv_weight=0.005,
    use_y=False, device_type="cuda",
):
    from torch.amp import GradScaler, autocast
    G.train(); D.train()
    scaler_g = GradScaler(device_type, enabled=amp)
    scaler_d = GradScaler(device_type, enabled=amp)

    l1_meter, d_meter, g_meter = 0.0, 0.0, 0.0
    bce = nn.BCEWithLogitsLoss()

    for it, (lr, hr) in enumerate(loader):
        lr = lr.to(device, non_blocking=True).float()
        hr = hr.to(device, non_blocking=True).float()
        lr_up = upsample_to_hr(lr, hr).clamp(0,1); hr = hr.clamp(0,1)

        # ---------- Update D ----------
        optD.zero_grad(set_to_none=True)
        with autocast(device_type, enabled=amp):
            with torch.no_grad():
                sr_det = G(lr_up).clamp(0,1)
            pred_real = D(hr)
            pred_fake = D(sr_det)
            # real=1, fake=0
            d_loss_real = bce(pred_real, torch.ones_like(pred_real))
            d_loss_fake = bce(pred_fake, torch.zeros_like(pred_fake))
            d_loss = (d_loss_real + d_loss_fake) * 0.5
        scaler_d.scale(d_loss).backward()
        scaler_d.step(optD); scaler_d.update()

        # ---------- Update G ----------
        optG.zero_grad(set_to_none=True)
        with autocast(device_type, enabled=amp):
            sr = G(lr_up).clamp(0,1)
            # Content loss
            if use_y:
                c_loss = F.l1_loss(rgb_to_y(sr), rgb_to_y(hr))
            else:
                c_loss = F.l1_loss(sr, hr)
            # Adversarial loss (Generator möchte "real" sein)
            pred_fake_for_g = D(sr)
            g_adv = bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            g_loss = l1_weight * c_loss + adv_weight * g_adv

        scaler_g.scale(g_loss).backward()
        if grad_clip is not None:
            scaler_g.unscale_(optG)
            nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
        scaler_g.step(optG); scaler_g.update()

        l1_meter += c_loss.item()
        d_meter  += d_loss.item()
        g_meter  += g_loss.item()

        if it == 0:
            print(
                f"[TRAIN] lr_up min/max: {lr_up.min():.4f}/{lr_up.max():.4f} | "
                f"hr min/max: {hr.min():.4f}/{hr.max():.4f} | "
                f"sr min/max: {sr.min():.4f}/{sr.max():.4f} | "
                f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f}"
            )

    n = max(1, len(loader))
    return (l1_meter/n, d_meter/n, g_meter/n)

@torch.no_grad()
def validate_g_only(G, loader, device, *, epoch=None, out_dir="debug_val", use_y=False):
    G.eval()
    os.makedirs(out_dir, exist_ok=True)
    total_psnr, total_bicubic, n = 0.0, 0.0, 0

    for i, (lr, hr) in enumerate(loader):
        lr = lr.to(device, non_blocking=True).float()
        hr = hr.to(device, non_blocking=True).float()
        lr_up = upsample_to_hr(lr, hr).clamp(0,1); hr = hr.clamp(0,1)
        sr = G(lr_up).clamp(0,1)

        if i == 0:
            print(
                f"[VAL]   lr_up min/max: {lr_up.min():.4f}/{lr_up.max():.4f} | "
                f"hr min/max: {hr.min():.4f}/{hr.max():.4f} | "
                f"sr min/max: {sr.min():.4f}/{sr.max():.4f}"
            )

        if use_y:
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

    return total_psnr/max(1,n), total_bicubic/max(1,n)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    torch.manual_seed(cfg.get("seed", 42))

    # Data
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

    # Models
    G = build_model(cfg["variant"]).to(device)           # Generator (SRCNN)
    D = PatchDiscriminator(in_ch=3, base_ch=cfg.get("disc_channels", 64)).to(device)

    # Opt / Sched
    optG = optim.AdamW(G.parameters(), lr=cfg.get("lr_g", 1e-4), weight_decay=cfg.get("wd_g", 0.0))
    optD = optim.Adam(D.parameters(),  lr=cfg.get("lr_d", 1e-4), betas=(0.5, 0.999), weight_decay=cfg.get("wd_d", 0.0))

    if cfg.get("cosine", True):
        schG = optim.lr_scheduler.CosineAnnealingLR(optG, T_max=cfg["epochs"], eta_min=cfg.get("lr_min", 0.0))
        schD = optim.lr_scheduler.CosineAnnealingLR(optD, T_max=cfg["epochs"], eta_min=cfg.get("lr_min", 0.0))
    else:
        milestones = [int(cfg["epochs"]*0.5), int(cfg["epochs"]*0.8)]
        schG = optim.lr_scheduler.MultiStepLR(optG, milestones=milestones, gamma=0.5)
        schD = optim.lr_scheduler.MultiStepLR(optD, milestones=milestones, gamma=0.5)

    # Resume (optional)
    start_epoch, best_psnr = 1, -math.inf
    if cfg.get("resume_g"):
        ckptg = load_checkpoint(cfg["resume_g"], G, optimizer=optG, scheduler=schG, map_location=device)
        start_epoch = int(ckptg.get("epoch", 0)) + 1
        best_psnr = float(ckptg.get("best_psnr", -math.inf) or -math.inf)
    if cfg.get("resume_d"):
        load_checkpoint(cfg["resume_d"], D, optimizer=optD, scheduler=schD, map_location=device)

    out_dir = cfg["out_dir"]; run_name = cfg["run_name"]
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, "debug"); os.makedirs(debug_dir, exist_ok=True)

    # Train
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()
        l1_w   = float(cfg.get("l1_weight", 1.0))
        adv_w  = float(cfg.get("adv_weight", 0.005))
        use_y  = bool(cfg.get("use_y", False))

        l1_avg, d_avg, g_avg = train_one_epoch_gan(
            G, D, train_loader, optG, optD, device,
            amp=bool(cfg.get("amp", True)),
            grad_clip=cfg.get("grad_clip", 1.0),
            l1_weight=l1_w, adv_weight=adv_w,
            use_y=use_y, device_type=device_type
        )

        schG.step(); schD.step()

        # Validation (Generator only)
        if epoch % cfg.get("val_every", 1) == 0:
            val_psnr, bicubic_psnr = validate_g_only(
                G, val_loader, device, epoch=epoch, out_dir=debug_dir, use_y=use_y
            )
            elapsed = time.time() - t0
            print(
                f"[{epoch:04d}/{cfg['epochs']}] L1={l1_avg:.4f} | D={d_avg:.4f} | G={g_avg:.4f} | "
                f"val_psnr={val_psnr:.2f} dB (bicubic={bicubic_psnr:.2f} dB) | "
                f"best={max(best_psnr, val_psnr):.2f} | "
                f"lrG={optG.param_groups[0]['lr']:.2e} lrD={optD.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
            )

            # Checkpoints: speichere G und D separat
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(os.path.join(out_dir, f"{run_name}_G_best.pt"),
                                epoch, G, optimizer=optG, scheduler=schG, best_psnr=best_psnr,
                                extra={"val_psnr_bicubic": bicubic_psnr})
                save_checkpoint(os.path.join(out_dir, f"{run_name}_D_best.pt"),
                                epoch, D, optimizer=optD, scheduler=schD, best_psnr=best_psnr)

            save_checkpoint(os.path.join(out_dir, f"{run_name}_G_latest.pt"),
                            epoch, G, optimizer=optG, scheduler=schG, best_psnr=best_psnr,
                            extra={"val_psnr_bicubic": bicubic_psnr})
            save_checkpoint(os.path.join(out_dir, f"{run_name}_D_latest.pt"),
                            epoch, D, optimizer=optD, scheduler=schD, best_psnr=best_psnr)

if __name__ == "__main__":
    main()
