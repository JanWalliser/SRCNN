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

def psnr(pred, target, eps=1e-10):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))
