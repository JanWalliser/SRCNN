# infer_main.py (neu anlegen)
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from srcnn_model import build_model
from utils import load_checkpoint, infer_single_image, bicubic_up_by_factor

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Modell laden
model = build_model("high").to(device).eval()
load_checkpoint("checkpoints/high/x3/srcnn_baseline_best.pt", model, map_location=device)

# 2) Eingabebild laden (z. B. 300x200)
img = Image.open("../../Data/val/low_res_x3/1121.png").convert("RGB")
lr = to_tensor(img).unsqueeze(0).to(device)     # [1,3,H,W]

# 3) Bicubic Upscale per Faktor (kein size=(...))
s = 3  # z.B. x3
lr_up = bicubic_up_by_factor(lr, s)             # [1,3,H*s,W*s]

# 4) Refinement (mit/ohne Tiling)
sr = infer_single_image(model, lr_up.squeeze(0), device=device, tile=None)

# 5) Speichern (PNG)
to_pil_image(sr).save("Test3.png")
