import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from srcnn_model import build_model
from utils import infer_single_image, load_checkpoint, psnr

# --- Einstellungen ---
model_variant   = "high"   # 'low', 'medium', 'high'
checkpoint_path = "checkpoints/high/x3/srcnn_baseline_best.pt"
input_path      = "results/Sara.jpg"      # LR (klein)
highres_path    = "../../Data/train/high_res/1.jpg"       # HR (Ziel)
output_path     = "results/Sara_high.png"
device          = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1) Modell laden ---
model = build_model(model_variant).to(device)
_ = load_checkpoint(checkpoint_path, model, map_location=device)
model.eval()

# --- 2) Bild(e) laden ---
to_tensor = transforms.ToTensor()
to_pil    = transforms.ToPILImage()

lr_img = Image.open(input_path).convert("RGB")
hr_img = Image.open(highres_path).convert("RGB")

# --- 3) LR auf HR-Größe hochskalieren (wichtig für HR->HR-Refinement) ---
#    Falls Größen nicht übereinstimmen, wird LR bikubisch auf HR-Größe gebracht.
if lr_img.size != hr_img.size:
    lr_up = lr_img.resize(hr_img.size, resample=Image.BICUBIC)
else:
    lr_up = lr_img  # bereits HR-Größe

inp_tensor = to_tensor(lr_up)       # [C,H,W], in [0,1]
hr_tensor  = to_tensor(hr_img)      # [C,H,W], in [0,1]

# --- 4) Inferenz (ohne AMP, um NaNs/Underruns zu vermeiden) ---
with torch.no_grad():
    sr_tensor = infer_single_image(model, inp_tensor, device=device)  # gibt [C,H,W] in [0,1] zurück

# --- 5) Sanity-Checks: Wertebereich & Größen ---
print(f"SR stats -> min={sr_tensor.min().item():.6f}, max={sr_tensor.max().item():.6f}, "
      f"nan={torch.isnan(sr_tensor).any().item()}")
assert sr_tensor.shape == hr_tensor.shape, \
    f"SR und HR haben unterschiedliche Shapes: {sr_tensor.shape} vs {hr_tensor.shape}"

# --- 6) PSNR berechnen (auf [0,1]-Basis) ---
val_psnr = psnr(sr_tensor, hr_tensor).item()
print(f"PSNR (SR vs HR): {val_psnr:.2f} dB")

# --- 7) Speichern & Anzeigen ---
os.makedirs(os.path.dirname(output_path), exist_ok=True)
sr_img = to_pil(sr_tensor.clamp(0,1))
sr_img.save(output_path)

