import os, torch, math
from torch.nn import functional as F

def save_checkpoint(path, epoch, model, optimizer=None, scheduler=None, best_psnr=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_psnr": best_psnr,
        "extra": extra or {},
    }
    torch.save(ckpt, path)
    print(f"Checkpoint @ epoch {epoch} -> {path}")

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer and ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Loaded checkpoint from {path} (epoch={ckpt.get('epoch')})")
    return ckpt

@torch.no_grad()
def infer_single_image(model, img_tensor, device="cpu", tile=None, tile_overlap=16):
    model.eval(); img = img_tensor.unsqueeze(0).to(device)
    if tile is None:
        out = model(img)
    else:
        _, _, h, w = img.shape
        ph = pw = tile
        out_chunks = torch.zeros((1, 3, h*model.scale, w*model.scale), device=device)
        count_map  = torch.zeros_like(out_chunks)
        for y in range(0, h, ph - tile_overlap):
            for x in range(0, w, pw - tile_overlap):
                y2 = min(y+ph, h); x2 = min(x+pw, w)
                patch = img[:, :, y:y2, x:x2]
                out_p = model(patch)
                ys, xs = y*model.scale, x*model.scale
                ye, xe = ys + out_p.shape[2], xs + out_p.shape[3]
                out_chunks[:, :, ys:ye, xs:xe] += out_p
                count_map[:, :, ys:ye, xs:xe] += 1
        out = out_chunks / count_map.clamp_min(1)
    out = out.squeeze(0).clamp_(0, 1).cpu()
    return out



@torch.no_grad()
def infer_single_image(model, img_tensor, device="cuda", tile=None, tile_overlap=16):
    """
    img_tensor: [3,H,W] in [0,1]
    Gibt [3,H,W] zurück (gleich groß wie Eingabe), passend für HR->HR-Refinement-Modelle.
    """
    model.eval()
    img = img_tensor.unsqueeze(0).to(device)  # [1,3,H,W]

    if tile is None:
        out = model(img)
    else:
        _, _, h, w = img.shape
        ph = pw = int(tile)
        step_y = max(1, ph - int(tile_overlap))
        step_x = max(1, pw - int(tile_overlap))

        out_acc = torch.zeros((1, 3, h, w), device=device)
        cnt_acc = torch.zeros((1, 3, h, w), device=device)

        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                y2 = min(y + ph, h)
                x2 = min(x + pw, w)
                patch = img[:, :, y:y2, x:x2]
                out_p = model(patch)
                out_acc[:, :, y:y2, x:x2] += out_p
                cnt_acc[:, :, y:y2, x:x2] += 1

        out = out_acc / cnt_acc.clamp_min(1)

    return out.squeeze(0).clamp_(0, 1).to(device)

def psnr(pred, target, eps=1e-10):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))



def bicubic_up_by_factor(x_4d, scale: int):
    """
    x_4d: [B,3,H,W] in [0,1]
    scale: 2/3/4/6
    """
    return F.interpolate(x_4d, scale_factor=scale, mode="bicubic", align_corners=False)



import torch
from srcnn_model import build_model
from utils import load_checkpoint

def load_sr_model(
    arch: str = "base",      # "base", "perc", "gan"
    variant: str = "high",   # "low", "medium", "high"
    scale: int = 3,          # 2, 3, 4, 6 ...
    device: str = "cuda",
):
    """
    Lädt ein SR-Modell für eine gegebene Architektur (base/perc/gan),
    Netzwerk-Variante (low/medium/high) und Skalenfaktor.
    """
    arch = arch.lower()
    variant = variant.lower()

    # --- Mapping: (arch, variant, scale) -> Checkpoint-Pfad ---
    # Hier deine echten Pfade eintragen!
    ckpt_map = {
        # BASELINE (L1)
        ("base", "high", 3): "checkpoints/high/x3_base/srcnn_base_high_x3_best.pt",
        ("base", "high", 4): "checkpoints/high/x4_base/srcnn_base_high_x4_best.pt",

        # PERCEPTUAL
        ("perc", "high", 3): "checkpoints/high/x3_perc/srcnn_perceptual_high_x3_best.pt",
        ("perc", "high", 4): "checkpoints/high/x4_perc/srcnn_perceptual_high_x4_best.pt",

        # GAN
        ("gan", "high", 3):  "checkpoints/high/x3_gan/srcnn_gan_high_x3_G_latest.pt",
        ("gan", "high", 4):  "checkpoints/high/x4_gan/srcnn_gan_high_x4_G_best.pt",
    }

    key = (arch, variant, scale)
    if key not in ckpt_map:
        raise ValueError(f"Kein Checkpoint für Kombination {key} definiert.")

    ckpt_path = ckpt_map[key]

    # Modell bauen und Checkpoint laden
    model = build_model(variant).to(device).eval()
    load_checkpoint(ckpt_path, model, map_location=device)

    return model
