import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from utils import infer_single_image, bicubic_up_by_factor
from utils import load_sr_model   # oder direkt aus infer_main importiert

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    scale = 3
    arch  = "perc"   # "base", "perc" oder "gan" 
    variant = "high" # "low", "medium", "high"

    # 1) Modell entsprechend arch/variant/scale laden
    model = load_sr_model(arch=arch, variant=variant, scale=scale, device=device)

    # 2) LR-Bild laden
    img = Image.open("fabrik.jpg").convert("RGB")
    lr = to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W]

    # 3) Bicubic Upscale
    lr_up = bicubic_up_by_factor(lr, scale)

    # 4) SR-Inferenz
    sr = infer_single_image(model, lr_up.squeeze(0), device=device, tile=None)

    # 5) Speichern
    out = to_pil_image(sr.cpu())
    out.save(f"Fabrik_sr_x{scale}_{arch}.png")

if __name__ == "__main__":
    main()
