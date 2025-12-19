#!/usr/bin/env python3
# collect_results_perc.py
#
# Extrahiert PSNR & einfache Zeitschätzer aus deinen Debug-Bildern in:
#
# checkpoints/
#   low/x2_perc/debug/ep####_{sr,hr}.png
#   medium/x2_perc/debug/ep####_{sr,hr}.png
#   high/x2_perc/debug/ep####_{sr,hr}.png
#   ... (x3_perc, x4_perc, x6_perc)
#
# Bicubic wird on-the-fly aus HR erzeugt:
#   HR -> downscale (factor s) -> bicubic upsample -> HR-PSNR
#
# Aufrufbeispiel:
#   python collect_results_perc.py \
#       --root checkpoints \
#       --outdir Res_perc/csv \
#       --scales x2 x3 x4 \
#       --models low medium high
#
# (für Modelle/Scales, die es nicht gibt, wird einfach gewarnt und übersprungen)

from __future__ import annotations
import argparse, re, math, sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image

EPOCH_RE = re.compile(r"ep(\d{3,5})_([a-z]+)\.png$", re.IGNORECASE)

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
    suffix ∈ {'sr','hr'}.
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

def parse_args():
    ap = argparse.ArgumentParser(description="Extract PSNR & times for L1 SR from checkpoints/*/x*_L1/debug.")
    ap.add_argument("--root", type=str, default="checkpoints",
                    help="Root der Ergebnisse (Ordner mit low/ medium/ high/).")
    ap.add_argument("--outdir", type=str, default="Res_L1/csv",
                    help="Wohin CSVs geschrieben werden.")
    ap.add_argument("--scales", nargs="+", default=["x2","x3","x4","x6"],
                    help="Liste der Skalenordner (x2, x3, x4, x6).")
    ap.add_argument("--models", nargs="+", default=["low","medium","high"],
                    help="Modelle/Architekturordner (low, medium, high).")
    ap.add_argument("--epsilon", type=float, default=0.01,
                    help="Konvergenzkriterium: 1-epsilon vom Best-PSNR.")
    ap.add_argument("--y-psnr", action="store_true",
                    help="PSNR auf Y-Kanal statt RGB.")
    return ap.parse_args()

def main():
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []

    for scale in args.scales:
        factor = int(scale.replace("x", ""))

        # Für Learning-Curve CSV: eine gemeinsame Epochenliste (Vereinigung)
        all_epochs = set()
        per_model_series = {}

        # Für learnspeed CSV sammeln wir gezielte Epochen
        key_epochs = [10, 100,1000,2500]
        learnspeed_rows = []

        # Ausgabe-Zeilen (summary_<scale>.csv)
        summary_rows = []

        for model in args.models:
            # checkpoints/<model>/<scale>_L1/debug
            mod_dir = root / model / f"{scale}" / "debug"
            sr_files = find_epoch_files(mod_dir, "sr")
            hr_files = find_epoch_files(mod_dir, "hr")

            epochs = sorted(set(sr_files.keys()) & set(hr_files.keys()))
            if not epochs:
                print(f"[WARN] Keine passenden SR/HR-Paare für {model}/{scale} gefunden.", file=sys.stderr)
                continue

            # PSNR-Reihen berechnen
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

                    # Bicubic PSNR: on-the-fly aus HR
                    bic = bicubic_from_hr(hr, factor)
                    if args.y_psnr:
                        cur_bpsnr = psnr(rgb_to_y(bic), rgb_to_y(hr))
                    else:
                        cur_bpsnr = psnr(bic, hr)
                    bic_psnrs.append((ep, cur_bpsnr))

                except Exception as e:
                    print(f"[WARN] Fehler bei Epoche {ep} in {model}/{scale}: {e}", file=sys.stderr)

            # In DataFrame bringen
            if not psnrs:
                print(f"[WARN] Keine PSNRs für {model}/{scale}.", file=sys.stderr)
                continue

            df = pd.DataFrame(psnrs, columns=["epoch", f"psnr_{model}"]).set_index("epoch")
            per_model_series[model] = df
            all_epochs.update(df.index.tolist())

            df_b = None
            if bic_psnrs:
                df_b = pd.DataFrame(bic_psnrs, columns=["epoch", "psnr_bicubic"]).set_index("epoch")

            # best, e_best
            best_idx = int(df[f"psnr_{model}"].idxmax())
            best_val = float(df.loc[best_idx, f"psnr_{model}"])

            # e_conv (99% vom Best-PSNR standardmäßig)
            thr = (1.0 - args.epsilon) * best_val
            e_conv = None
            for ep in sorted(df.index):
                if df.loc[ep, f"psnr_{model}"] >= thr:
                    e_conv = int(ep)
                    break

            # mittlere Zeit/Epoche (grob) aus Datei-mtime
            t_ep = estimate_epoch_time_from_mtime(sr_files)  # Sekunden oder None
            T_conv = None
            if (t_ep is not None) and (e_conv is not None):
                T_conv = float(t_ep * e_conv)

            # Bicubic Vergleichswert (nimm bicubic@e_best, sonst Median)
            psnr_bic = None
            if df_b is not None:
                if best_idx in df_b.index:
                    psnr_bic = float(df_b.loc[best_idx, "psnr_bicubic"])
                else:
                    psnr_bic = float(df_b["psnr_bicubic"].median())

            delta = None
            if (psnr_bic is not None) and (best_val is not None):
                delta = best_val - psnr_bic

            summary_rows.append({
                "scale": scale,
                "model": model,
                "PSNR_best_dB": best_val,
                "Bicubic_dB": psnr_bic,
                "Delta_dB": delta,
                "e_best": best_idx,
                "e_conv": e_conv,
                "t_ep_avg_s": t_ep,
                "T_conv_s": T_conv,
                "n_epochs_analyzed": len(df),
            })

            # learnspeed row
            row_ls = {"model": model}
            for ke in key_epochs:
                row_ls[f"ep{ke}_dB"] = float(df.loc[ke, f"psnr_{model}"]) if ke in df.index else np.nan
            row_ls["bicubic_dB"] = psnr_bic if psnr_bic is not None else np.nan
            row_ls["delta_at_2500_dB"] = (
                (row_ls.get("ep2500_dB", np.nan) - row_ls["bicubic_dB"])
                if not (math.isnan(row_ls.get("ep2500_dB", np.nan)) or math.isnan(row_ls["bicubic_dB"]))
                else np.nan
            )
            learnspeed_rows.append(row_ls)

        # Learning-curve CSV (vereinigte Epochen + alle Modelle + Bicubic-Spalte)
        if per_model_series:
            epochs_sorted = sorted(all_epochs)
            df_curve = pd.DataFrame(index=epochs_sorted)
            for model in args.models:
                if model in per_model_series:
                    df_curve = df_curve.join(per_model_series[model], how="left")

            # Bicubic-Spalte: pro Epoche HR aus irgendeinem Modell nehmen
            bic_vals = []
            for ep in epochs_sorted:
                hr_path = None
                for model in args.models:
                    mod_dir = root / model / f"{scale}" / "debug"
                    hr_map = find_epoch_files(mod_dir, "hr")
                    if ep in hr_map:
                        hr_path = hr_map[ep]
                        break
                if hr_path is None:
                    bic_vals.append(np.nan)
                    continue
                try:
                    hr = load_image(hr_path)
                    bic = bicubic_from_hr(hr, factor)
                    if args.y_psnr:
                        v = psnr(rgb_to_y(bic), rgb_to_y(hr))
                    else:
                        v = psnr(bic, hr)
                    bic_vals.append(v)
                except Exception:
                    bic_vals.append(np.nan)

            df_curve["psnr_bicubic"] = bic_vals

            curve_path = outdir / f"learning_curve_{scale}.csv"
            df_curve.reset_index(names="epoch").to_csv(curve_path, index=False)
            print(f"[OK] {curve_path}")

        # Summary CSV pro Scale
        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            sum_path = outdir / f"summary_{scale}.csv"
            df_sum.to_csv(sum_path, index=False)
            print(f"[OK] {sum_path}")
            all_summary_rows.extend(summary_rows)

        # Learnspeed CSV pro Scale
        if learnspeed_rows:
            df_ls = pd.DataFrame(learnspeed_rows)
            ls_path = outdir / f"learnspeed_{scale}.csv"
            df_ls.to_csv(ls_path, index=False)
            print(f"[OK] {ls_path}")

    # Gesamtsummary (alle Scales)
    if all_summary_rows:
        df_all = pd.DataFrame(all_summary_rows)
        all_path = outdir / "summary_all_scales.csv"
        df_all.to_csv(all_path, index=False)
        print(f"[OK] {all_path}")

if __name__ == "__main__":
    main()
