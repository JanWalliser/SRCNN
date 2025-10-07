import os
from PIL import Image

def mod_crop_size(w, h, scale):
    return (w - (w % scale), h - (h % scale))

def process_one_image(img: Image.Image, scale: int):
    # 1) RGB & mod-crop
    img = img.convert("RGB")
    w, h = img.size
    w_c, h_c = mod_crop_size(w, h, scale)
    if (w_c, h_c) != (w, h):
        img = img.crop((0, 0, w_c, h_c))  # top-left mod-crop (reproduzierbar)

    # 2) Downscale by factor s (bicubic), 3) Upscale back to cropped HR size (bicubic)
    lr = img.resize((w_c // scale, h_c // scale), Image.BICUBIC)
    lr_up = lr.resize((w_c, h_c), Image.BICUBIC)
    return lr_up

def create_low_res_folders(
    high_res_dir: str,
    out_root: str,
    scale_factors=(2, 3, 4, 6),
    save_format="PNG"
):
    os.makedirs(out_root, exist_ok=True)
    names = [n for n in os.listdir(high_res_dir) if not n.startswith(".")]

    for s in scale_factors:
        out_dir = os.path.join(out_root, f"low_res_x{s}")
        os.makedirs(out_dir, exist_ok=True)

        for name in names:
            in_path = os.path.join(high_res_dir, name)
            out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".png" if save_format.upper()=="PNG" else name)

            try:
                with Image.open(in_path) as img:
                    lr_up = process_one_image(img, s)
                    # PNG vermeidet mehrfache JPEG-Rekompression
                    lr_up.save(out_path, format=save_format)
            except Exception as e:
                print(f"[x{s}] Error processing {name}: {e}")

if __name__ == "__main__":
    create_low_res_folders(
        high_res_dir="data/high_res",
        out_root="data",
        scale_factors=(2, 3, 4, 6),
        save_format="PNG"
    )
