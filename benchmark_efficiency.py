#!/usr/bin/env python3
"""
benchmark_efficiency.py
-----------------------
Measure model efficiency metrics for your SRCNN variants and save to CSV:
- Parameter count (M)
- Model size (MB) in memory and checkpoint size on disk (if provided)
- Inference latency (ms): mean / p50 / p90 / p99
- Throughput (MPix/s)
- (GPU) Peak VRAM (MB)
- Approx. FLOPs (GMAC) per forward (Conv2d-only) via runtime hooks

Assumptions:
- Your models are built via srcnn_model.build_model(name: str) with names: low|medium|high
- Checkpoints are optional. If present, they live under: checkpoints/<arch>/<scale>/srcnn_baseline_best.pt (or *_latest.pt)
- The network is HR->HR refinement; input & output have the same HxW.

Usage examples:
    python benchmark_efficiency.py --device cuda --sizes 256 512 768 1024
    python benchmark_efficiency.py --device cuda --precision fp16 --sizes 512 --repeat 200
    python benchmark_efficiency.py --device cpu  --sizes 256 512 --repeat 50
    # If checkpoints elsewhere:
    python benchmark_efficiency.py --ckpt_root C:/path/to/checkpoints

Outputs:
    results/efficiency_metrics.csv
"""
import os, time, math, statistics, argparse
from pathlib import Path
import torch
from torch import nn
from typing import Dict, List, Tuple, Optional

from srcnn_model import build_model  # expects your project file
# Optional: if you want to override model names, edit MODEL_NAMES below.

MODEL_NAMES = ["low", "medium", "high"]
SCALES = ["x2", "x3", "x4", "x6"]

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sizeof_state_dict_mb(sd: Dict[str, torch.Tensor]) -> float:
    total_bytes = 0
    for t in sd.values():
        if isinstance(t, torch.Tensor):
            total_bytes += t.numel() * t.element_size()
    return total_bytes / (1024 * 1024)

def find_checkpoint(ckpt_root: Path, arch: str, scale: str) -> Optional[Path]:
    cand = [
        ckpt_root / arch / scale / "srcnn_baseline_best.pt",
        ckpt_root / arch / scale / "srcnn_baseline_latest.pt",
    ]
    for p in cand:
        if p.exists():
            return p
    # fallback: any *.pt in folder
    folder = ckpt_root / arch / scale
    if folder.exists():
        pts = sorted(folder.glob("*.pt"))
        if pts:
            return pts[-1]
    return None

# ---- FLOPs (GMAC) via forward hooks (Conv2d only) ----
class ConvFlopCounter:
    def __init__(self):
        self.gmac = 0.0
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, module: nn.Module, inp, out):
        # Count MACs for Conv2d: Cout * Hout * Wout * (Cin/groups * Kh * Kw)
        if not isinstance(module, nn.Conv2d):
            return
        # Input shape: (B, Cin, Hin, Win); Output: (B, Cout, Hout, Wout)
        # We use output tensor to get Hout, Wout, Cout, and module to get kernel and Cin/groups
        try:
            out_t: torch.Tensor = out if isinstance(out, torch.Tensor) else out[0]
            B, Cout, Hout, Wout = out_t.shape
            Cin = module.in_channels
            Kh, Kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            groups = module.groups
            macs = B * Cout * Hout * Wout * (Cin // groups) * Kh * Kw
            self.gmac += macs / 1e9
        except Exception:
            pass

    def add(self, model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self.handles.append(m.register_forward_hook(self._hook))

    def clear(self):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles.clear()

def percentile(vals: List[float], p: float) -> float:
    if not vals: return float("nan")
    vals_sorted = sorted(vals)
    k = (len(vals_sorted)-1) * (p/100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c: return vals_sorted[int(k)]
    return vals_sorted[f] * (c-k) + vals_sorted[c] * (k-f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu","cuda"], help="Device for benchmarking.")
    ap.add_argument("--precision", default="fp32", choices=["fp32","fp16","bf16"], help="Inference precision.")
    ap.add_argument("--sizes", nargs="+", type=int, default=[256, 512], help="Square HR sizes (HxW).")
    ap.add_argument("--repeat", type=int, default=100, help="Number of timed iterations.")
    ap.add_argument("--warmup", type=int, default=20, help="Warmup iterations (not timed).")
    ap.add_argument("--batch", type=int, default=1, help="Batch size.")
    ap.add_argument("--ckpt_root", type=str, default="checkpoints", help="Root folder of checkpoints.")
    ap.add_argument("--out_csv", type=str, default="results/efficiency_metrics.csv", help="Output CSV path.")
    args = ap.parse_args()

    device = torch.device(args.device)
    prec = args.precision
    use_autocast = (prec in ["fp16","bf16"]) and device.type == "cuda"

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    for arch in MODEL_NAMES:
        for scale in SCALES:
            # Build model
            model = build_model(arch)
            model.eval().to(device)

            # Load checkpoint if exists
            ckpt = None
            ckpt_path = find_checkpoint(Path(args.ckpt_root), arch, scale)
            ckpt_size_mb = None
            if ckpt_path is not None and ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    if isinstance(ckpt, dict) and "model" in ckpt:
                        model.load_state_dict(ckpt["model"], strict=False)
                    elif isinstance(ckpt, dict):
                        model.load_state_dict(ckpt, strict=False)
                    ckpt_size_mb = ckpt_path.stat().st_size / (1024*1024)
                except Exception as e:
                    print(f"[WARN] Could not load checkpoint {ckpt_path}: {e}")

            # Basic stats
            params = count_params(model)
            # size in memory (parameters only)
            try:
                state = model.state_dict()
                model_size_mb = sizeof_state_dict_mb(state)
            except Exception:
                model_size_mb = float("nan")

            for size in args.sizes:
                H = W = size
                x = torch.rand(args.batch, 3, H, W, device=device)

                # Autocast dtype
                amp_dtype = torch.float16 if prec=="fp16" else (torch.bfloat16 if prec=="bf16" else torch.float32)

                # Warmup
                with torch.inference_mode():
                    for _ in range(args.warmup):
                        if device.type == "cuda": torch.cuda.synchronize()
                        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_autocast else torch.cuda.amp.autocast(enabled=False)
                        with ctx:
                            y = model(x)
                        if device.type == "cuda": torch.cuda.synchronize()

                # FLOPs single pass via hooks
                flop_counter = ConvFlopCounter()
                flop_counter.add(model)
                with torch.inference_mode():
                    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_autocast else torch.cuda.amp.autocast(enabled=False)
                    with ctx:
                        _ = model(x)
                flop_counter.clear()
                gmac = flop_counter.gmac  # GMAC per forward

                # Timed loop
                lat_ms: List[float] = []
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_autocast else torch.cuda.amp.autocast(enabled=False)
                    for _ in range(args.repeat):
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            with ctx:
                                _ = model(x)
                            torch.cuda.synchronize()
                            t1 = time.perf_counter()
                        else:
                            t0 = time.perf_counter()
                            with ctx:
                                _ = model(x)
                            t1 = time.perf_counter()
                        lat_ms.append((t1 - t0) * 1000.0)

                lat_mean = statistics.mean(lat_ms) if lat_ms else float("nan")
                lat_p50  = percentile(lat_ms, 50)
                lat_p90  = percentile(lat_ms, 90)
                lat_p99  = percentile(lat_ms, 99)
                # throughput in megapixels/s
                pix_per_iter = args.batch * H * W
                thr_mpix_s = (pix_per_iter / 1e6) / (lat_mean / 1000.0) if lat_mean > 0 else float("nan")

                vram_mb = float("nan")
                if device.type == "cuda":
                    vram_mb = torch.cuda.max_memory_allocated() / (1024*1024)

                rows.append({
                    "arch": arch,
                    "scale": scale,
                    "size_hw": f"{H}x{W}",
                    "device": device.type,
                    "precision": prec,
                    "batch": args.batch,
                    "params_M": round(params / 1e6, 3),
                    "model_size_MB": round(model_size_mb, 3) if model_size_mb==model_size_mb else "",
                    "ckpt_path": str(ckpt_path) if ckpt_path else "",
                    "ckpt_size_MB": round(ckpt_size_mb, 3) if ckpt_size_mb else "",
                    "FLOPs_GMAC": round(gmac, 3),
                    "lat_mean_ms": round(lat_mean, 3),
                    "lat_p50_ms": round(lat_p50, 3),
                    "lat_p90_ms": round(lat_p90, 3),
                    "lat_p99_ms": round(lat_p99, 3),
                    "throughput_MPix_s": round(thr_mpix_s, 3),
                    "VRAM_peak_MB": round(vram_mb, 1) if vram_mb==vram_mb else "",
                })

            # cleanup to free VRAM between models
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Write CSV
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "arch","scale","size_hw","device","precision","batch",
        "params_M","model_size_MB","ckpt_path","ckpt_size_MB",
        "FLOPs_GMAC","lat_mean_ms","lat_p50_ms","lat_p90_ms","lat_p99_ms",
        "throughput_MPix_s","VRAM_peak_MB"
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[OK] Wrote {out_csv} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
