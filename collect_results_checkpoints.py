#!/usr/bin/env python3
"""
collect_results_checkpoints.py

Extrahiert PSNR & einfache Zeitschätzer aus deinen Debug-Bildern in der Struktur:

checkpoints/
  high/
    x2/
      debug/ep001_sr.png, ep001_hr.png, [ep001_lrup.png, ...]
    x2_perc/
      debug/...
    x2_gan/
      debug/...
    x3/, x3_perc/, x3_gan/, ...
  medium/
    ...
  low/
    ... (z.B. kein *_gan)

Dieses Skript ist loss-spezifisch:
    --loss base  -> nutzt Ordner x2, x3, x4, x6
    --loss perc  -> nutzt x2_perc, x3_perc, ...
    --loss gan   -> nutzt x2_gan, x3_gan, ...

Pro Kombination (scale, arch) werden berechnet:
- PSNR_best_dB          (Best-PSNR über Epochen)
- Bicubic_dB            (PSNR der Bicubic-Referenz)
- Delta_dB              (PSNR_best - Bicubic)
- e_best                (Epoche mit Best-PSNR)
- e_conv                (erste Epoche mit >= (1-epsilon)*PSNR_best)
- t_ep_avg_s            (grob: mittlere Epoche-Zeit aus Datei-mtime)
- T_conv_s              (e_conv * t_ep_avg_s)

Outputs im --outdir:
  summary_x2.csv, summary_x3.csv, ...
  summary_all_scales.csv

Beispielaufrufe:
    # L1 / Baseline
    python collect_results_checkpoints.py \
        --root checkpoints \
        --outdir results_base \
        --loss base

    # Perceptual
    python collect_results_checkpoints.py \
        --root checkpoints \
        --outdir results_perc \
        --loss perc

    # GAN
    python collect_results_checkpoints.py \
        --root checkpoints \
        --outdir results_gan \
        --loss gan
"""

from __future__ import annotations
import argparse, re, math, sys
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from PIL import Image

EPOCH_RE = re.compile(r"ep(\d{3,5})_([a-z]+)\.png$", re.IGNORECASE)

# ---------------------- Hilfsfunktionen ---------------------- #

def rgb_to_y(img: np.ndarray) -> np.ndarray:
    # img: float32 in [0,1], shape (H,W,3)
    # BT.601 luma (digital full-range)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def psnr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    # a,b: float32 in [0,1], same shape
    mse = np.mean((a - b) ** 2)
    if mse <= eps:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)

def load_image(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr

def find_epoch_files(dirpath: Path, suffix: str) -> Dict[int, Path]:
    """
    Sammelt Dateien ep####_<suffix>.png und gibt {epoch: path} zurück.
    suffix ∈ {'sr','hr','lrup'}.
    """
    out: Dict[int, Path] = {}
    if not dirpath.exists():
        return out
    for p in dirpath.glob(f"ep*_{suffix}.png"):
        m = EPOCH_RE.search(p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        out[epoch] = p
    return out

def estimate_epoch_time_from_mtime(sr_files: Dict[int, Path]) -> Optional[float]:
    """Grobe Schätzung: mittlere Zeit zwischen SR-Dateien (sek)."""
    if not sr_files:
        return None
    items = sorted(sr_files.items())
    deltas: List[float] = []
    prev_t = None
    for _, p in items:
        try:
            t = p.stat().st_mtime
        except Exception:
            continue
        if prev_t is not None:
            dt = t - prev_t
            if dt > 0:
                deltas.append(dt)
        prev_t = t
    if len(deltas) >= 3:
        return float(np.median(deltas))
    elif deltas:
        return float(np.mean(deltas))
    return None

def bicubic_from_hr(hr: np.ndarray, scale: int) -> np.ndarray:
    """
    Erzeugt Bicubic-Referenz aus HR:
      HR -> downscale by factor -> upsample back (BICUBIC).
    Gibt float32 in [0,1] mit gleicher Form wie hr zurück.
    """
    H, W, C = hr.shape
    assert C == 3, "Expect RGB image for bicubic_from_hr."
    im = Image.fromarray((hr * 255.0).clip(0, 255).astype("uint8"), mode="RGB")

    lr_w, lr_h = W // scale, H // scale
    if lr_w < 1 or lr_h < 1:
        raise ValueError(f"Scale {scale} zu groß für Bildgröße {W}x{H}.")

    lr = im.resize((lr_w, lr_h), Image.BICUBIC)
    hr_up = lr.resize((W, H), Image.BICUBIC)
    arr = np.asarray(hr_up, dtype=np.float32) / 255.0
    return arr

# ---------------------- Main-Logik ---------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Extract PSNR & times from checkpoints/*/*/debug structure.")
    ap.add_argument("--root", type=str, default="checkpoints",
                    help="Root der Checkpoints (enthält low/, medium/, high/).")
    ap.add_argument("--outdir", type=str, default="results_base",
                    help="Wohin CSVs geschrieben werden.")
    ap.add_argument("--scales", nargs="+", default=["x2", "x3", "x4", "x6"],
                    help="Liste der Skalen (x2, x3, x4, x6).")
    ap.add_argument("--arches", nargs="+", default=["low", "medium", "high"],
                    help="Architektur-Ordner unter root (z.B. low medium high).")
    ap.add_argument("--epsilon", type=float, default=0.01,
                    help="Konvergenzkriterium: 1-epsilon vom Best-PSNR.")
    ap.add_argument("--y-psnr", action="store_true",
                    help="PSNR auf Y-Kanal statt RGB.")
    ap.add_argument("--loss", type=str, default="base",
                    choices=["base", "perc", "gan"],
                    help="Loss-Variante: base (L1), perc (perceptual), gan (GAN).")
    return ap.parse_args()

def main():
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    loss_suffix_map = {
        "base": "",
        "perc": "_perc",
        "gan": "_gan",
    }
    loss_suffix = loss_suffix_map[args.loss]

    all_summary_rows = []

    for scale in args.scales:
        summary_rows = []
        factor = int(scale.replace("x", ""))

        for arch in args.arches:
            # Pfad: checkpoints/<arch>/<scale><suffix>/debug
            debug_dir = root / arch / f"{scale}{loss_suffix}" / "debug"
            if not debug_dir.exists():
                print(f"[INFO] Überspringe {arch}/{scale}{loss_suffix}: {debug_dir} existiert nicht.",
                      file=sys.stderr)
                continue

            sr_files = find_epoch_files(debug_dir, "sr")
            hr_files = find_epoch_files(debug_dir, "hr")
            lrup_files = find_epoch_files(debug_dir, "lrup")  # optional

            epochs = sorted(set(sr_files.keys()) & set(hr_files.keys()))
            if not epochs:
                print(f"[WARN] Keine SR/HR-Paare für {arch}/{scale}{loss_suffix} gefunden.",
                      file=sys.stderr)
                continue

            psnrs = []
            bic_psnrs = []

            for ep in epochs:
                try:
                    sr = load_image(sr_files[ep])
                    hr = load_image(hr_files[ep])

                    if args.y_psnr:
                        sr_y = rgb_to_y(sr)
                        hr_y = rgb_to_y(hr)
                        cur_psnr = psnr(sr_y, hr_y)
                    else:
                        cur_psnr = psnr(sr, hr)
                    psnrs.append((ep, cur_psnr))

                    # Bicubic: bevorzugt lrup-Datei; sonst aus HR erzeugen
                    if ep in lrup_files:
                        lrup = load_image(lrup_files[ep])
                        if args.y_psnr:
                            cur_bpsnr = psnr(rgb_to_y(lrup), rgb_to_y(hr))
                        else:
                            cur_bpsnr = psnr(lrup, hr)
                    else:
                        # on-the-fly Bicubic aus HR
                        bic = bicubic_from_hr(hr, factor)
                        if args.y_psnr:
                            cur_bpsnr = psnr(rgb_to_y(bic), rgb_to_y(hr))
                        else:
                            cur_bpsnr = psnr(bic, hr)

                    bic_psnrs.append((ep, cur_bpsnr))
                except Exception as e:
                    print(f"[WARN] Fehler bei Epoche {ep} in {arch}/{scale}{loss_suffix}: {e}",
                          file=sys.stderr)

            if not psnrs:
                print(f"[WARN] Keine PSNR-Werte für {arch}/{scale}{loss_suffix}.", file=sys.stderr)
                continue

            # DataFrames
            df = pd.DataFrame(psnrs, columns=["epoch", "psnr"]).set_index("epoch")
            df_b = pd.DataFrame(bic_psnrs, columns=["epoch", "psnr_bic"]).set_index("epoch")

            # Best-PSNR
            best_idx = int(df["psnr"].idxmax())
            best_val = float(df.loc[best_idx, "psnr"])

            # Konvergenz-Epoche (>= (1-eps)*best)
            thr = (1.0 - args.epsilon) * best_val
            e_conv = None
            for ep in sorted(df.index):
                if df.loc[ep, "psnr"] >= thr:
                    e_conv = int(ep)
                    break

            # mittlere Epoche-Zeit
            t_ep = estimate_epoch_time_from_mtime(sr_files)  # Sekunden oder None
            T_conv = None
            if (t_ep is not None) and (e_conv is not None):
                T_conv = float(t_ep * e_conv)

            # Bicubic-Referenz: nehme Wert bei e_best
            psnr_bic = None
            if best_idx in df_b.index:
                psnr_bic = float(df_b.loc[best_idx, "psnr_bic"])
            else:
                # Fallback: Median über alle Bicubic-Werte
                psnr_bic = float(df_b["psnr_bic"].median())

            delta = None
            if (psnr_bic is not None) and (best_val is not None):
                delta = best_val - psnr_bic

            summary_rows.append({
                "scale": scale,
                "model": arch,          # low / medium / high
                "PSNR_best_dB": best_val,
                "Bicubic_dB": psnr_bic,
                "Delta_dB": delta,
                "e_best": best_idx,
                "e_conv": e_conv,
                "t_ep_avg_s": t_ep,
                "T_conv_s": T_conv,
                "n_epochs_analyzed": len(df),
            })

        # pro-Scale CSV
        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            sum_path = outdir / f"summary_{scale}.csv"
            df_sum.to_csv(sum_path, index=False)
            print(f"[OK] {sum_path}")
            all_summary_rows.extend(summary_rows)

    # Gesamt-Summary über alle Scales
    if all_summary_rows:
        df_all = pd.DataFrame(all_summary_rows)
        all_path = outdir / "summary_all_scales.csv"
        df_all.to_csv(all_path, index=False)
        print(f"[OK] {all_path}")
    else:
        print("[WARN] Keine Daten gesammelt – prüfe Pfade und Argumente.", file=sys.stderr)

if __name__ == "__main__":
    main()
