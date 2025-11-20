import os
import sys
import glob
import yaml
import argparse
import subprocess
from datetime import datetime

def read_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def list_configs(config_dir: str, pattern: str):
    """Return sorted list of YAML files, ordered by 'scale' if present."""
    paths = sorted(glob.glob(os.path.join(config_dir, pattern)))
    items = []
    for p in paths:
        try:
            data = read_yaml(p)
            scale = int(data.get("scale", 10**9))
        except Exception:
            scale = 10**9
        items.append((scale, p))
    items.sort(key=lambda t: (t[0], t[1]))  # by scale then name
    return [p for _, p in items]

def pick_trainer(config_path: str, default_trainer: str) -> str:
    """
    Decide which training script to use.
    Priority:
      1) explicit cfg['trainer']
      2) heuristic: GAN keys -> train_gan.py
      3) heuristic: perceptual keys -> train_perceptual.py
      4) fallback -> default_trainer
    """
    cfg = read_yaml(config_path)

    # 1) Explicit override via YAML
    t = str(cfg.get("trainer", "")).strip().lower()
    alias = {
        "baseline": "train_baseline.py",
        "l1": "train_baseline.py",
        "perceptual": "train_perceptual.py",
        "perc": "train_perceptual.py",
        "gan": "train_gan.py",
        "adversarial": "train_gan.py",
        # allow direct filenames too:
        "train_baseline.py": "train_baseline.py",
        "train_perceptual.py": "train_perceptual.py",
        "train_gan.py": "train_gan.py",
    }
    if t in alias:
        return alias[t]

    # 2) Heuristic: GAN
    gan_keys = {"adv_weight", "disc_channels", "lr_d", "resume_d"}
    if any(k in cfg for k in gan_keys):
        return "train_gan.py"

    # 3) Heuristic: Perceptual
    perceptual_keys = {"w_perc", "vgg_layers", "use_y_psnr", "w_tv", "perceptual_weight", "vgg_name"}
    if any(k in cfg for k in perceptual_keys):
        return "train_perceptual.py"

    # 4) Fallback
    return default_trainer

def resolve_trainer_path(project_root: str, trainer_name: str) -> str:
    candidate = os.path.join(project_root, trainer_name)
    if os.path.isfile(candidate):
        return candidate
    return trainer_name  # maybe already an absolute/relative path

def run_one(python_exe: str, trainer: str, cfg_path: str, log_dir: str, project_root: str, dry_run: bool=False) -> int:
    os.makedirs(log_dir, exist_ok=True)
    run_name = os.path.splitext(os.path.basename(cfg_path))[0]
    log_file = os.path.join(log_dir, f"{run_name}.log")

    trainer_path = resolve_trainer_path(project_root, trainer)
    cfg_abs = os.path.abspath(cfg_path)
    cmd = [python_exe, trainer_path, "--config", cfg_abs]

    print(f"\n=== Running: {cmd} ===")
    print(f"    CWD: {project_root}")
    print(f"    Log: {log_file}")

    if dry_run:
        return 0

    with open(log_file, "wb") as lf:
        proc = subprocess.Popen(cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(proc.stdout.readline, b""):
            if not line:
                break
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            lf.write(line)
        return proc.wait()

def main():
    ap = argparse.ArgumentParser(description="Run multiple SR configs sequentially (with project-root CWD).")
    ap.add_argument("--config_dir", required=True, help="Folder that contains YAML configs")
    ap.add_argument("--pattern", default="*.yaml", help="Glob to pick configs (default: *.yaml)")
    ap.add_argument("--trainer", default="train_baseline.py", help="Default training script to use (fallback)")
    ap.add_argument("--python", dest="python_exe", default=sys.executable, help="Python executable to call")
    ap.add_argument("--project_root", default=None, help="Project root where train_*.py live; defaults to this script's parent")
    ap.add_argument("--stop_on_error", action="store_true", help="Abort on first non-zero exit code")
    ap.add_argument("--dry_run", action="store_true", help="Only print what would run")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = args.project_root or script_dir

    configs = list_configs(args.config_dir, args.pattern)
    if not configs:
        print(f"No configs found in {args.config_dir!r} with pattern {args.pattern!r}", file=sys.stderr)
        sys.exit(2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(project_root, "logs", f"multi_run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Found {len(configs)} configs:")
    for c in configs:
        print(f"  - {c}")

    exit_code = 0
    for cfg in configs:
        trainer = pick_trainer(cfg, args.trainer)
        rc = run_one(args.python_exe, trainer, cfg, log_dir, project_root, dry_run=args.dry_run)
        if rc != 0:
            print(f"[ERROR] {trainer} failed for {cfg} with exit code {rc}")
            exit_code = rc
            if args.stop_on_error:
                break

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
