#!/usr/bin/env python3
"""
DEIMv2 log (JSON-lines) -> CSV converter with optional YOLO-style schema + plotting.

Usage examples:
  python deimv2_log_to_csv.py log.txt --outdir out/
  python deimv2_log_to_csv.py log.txt --outdir out/ --schema yolo
  python deimv2_log_to_csv.py log.txt --outdir out/ --map mapping.json

What it does:
1) Parses DEIMv2 training logs where each epoch prints a JSON object per line (or nearly-JSON key:value pairs).
2) Writes two CSVs by default:
    - deimv2_raw.csv  : all numeric keys per epoch (wide table).
    - deimv2_yolo.csv : a YOLO-style, human-friendly subset (if keys exist; otherwise best-effort).
3) Optionally accepts a mapping file (JSON) to define your own CSV columns:
    - mapping.json = {"train_loss": "train/box_loss", "metrics_map50": "metrics/mAP50"}
4) Plots loss curves (all keys containing "loss") and AP/mAP curves (keys containing "AP" or "mAP").
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def try_parse_json(line: str) -> Optional[dict]:
    """Try parse as JSON first; if it fails, fallback to extracting key:float pairs."""
    s = line.strip()
    if not s:
        return None
    # Try strict JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to coerce single quotes to double quotes if it's JSON-like
    # (keeps numbers as-is). If still fails, fall back to regex pairs.
    if ("{" in s and "}" in s) and ("'" in s and '"' not in s):
        s2 = s.replace("'", '"')
        try:
            return json.loads(s2)
        except Exception:
            pass

    # Fallback: regex for key:value numeric pairs (floats or ints)
    # e.g., {"train_loss": 26.4, "train_lr": 2.5e-5, ...}
    kv = dict()
    for k, v in re.findall(r'["\']?([A-Za-z0-9_.\-]+)["\']?\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', s):
        try:
            kv[k] = float(v)
        except ValueError:
            continue
    return kv or None


def load_log(path: Path) -> pd.DataFrame:
    rows = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f, start=1):
            data = try_parse_json(line)
            if not data:
                continue
            # Attach epoch index if not present
            epoch = None
            # common candidates
            for k in ["epoch", "Epoch", "ep", "iter", "iteration"]:
                if k in data:
                    epoch = int(data[k])
                    break
            if epoch is None:
                epoch = idx  # fallback: line index as epoch

            # Flatten: keep numeric keys only; expand known list metrics
            flat = {"epoch": epoch}
            for k, v in data.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    flat[str(k)] = float(v)
                elif isinstance(v, list) and k == "test_coco_eval_bbox":
                    # Standard COCO order: AP, AP50, AP75, APs, APm, APl, AR@1, AR@10, AR@100, ARs, ARm, ARl
                    names = [
                        "coco/AP", "coco/AP50", "coco/AP75", "coco/APs", "coco/APm", "coco/APl",
                        "coco/AR@1", "coco/AR@10", "coco/AR@100", "coco/ARs", "coco/ARm", "coco/ARl"
                    ]
                    for i, val in enumerate(v[:len(names)]):
                        try:
                            val = float(val)
                        except Exception:
                            continue
                        flat[names[i]] = val
            rows.append(flat)

    if not rows:
        raise SystemExit("No parseable lines found in log file.")
    # Combine; later rows with same epoch will be reduced by averaging duplicates if any
    df = pd.DataFrame(rows).groupby("epoch", as_index=False).mean(numeric_only=True)
    df = df.sort_values("epoch").reset_index(drop=True)
    return df


def best_effort_yolo(df: pd.DataFrame) -> pd.DataFrame:
    """
    YOLO-style friendly columns (best-effort):
    epoch, lr, train/loss, val/loss, metrics/mAP50, metrics/mAP50-95
    Also include a few common detailed losses if present.
    """
    out = pd.DataFrame()
    out["epoch"] = df["epoch"]

    # lr candidates
    lr_cols = [c for c in df.columns if "lr" in c.lower()]
    out["lr"] = df[lr_cols[0]] if lr_cols else np.nan

    # train total loss
    tloss = [c for c in df.columns if c.lower() in ("train_loss", "loss") or c.lower().endswith("/loss")]
    out["train/loss"] = df[tloss[0]] if tloss else np.nan

    # val total loss
    vloss = [c for c in df.columns if c.lower() in ("val_loss", "valid_loss", "validation_loss") or "val" in c.lower() and "loss" in c.lower()]
    out["val/loss"] = df[vloss[0]] if vloss else np.nan

    # AP / mAP metrics
    # mAP50-95
    map_5095 = [c for c in df.columns if re.search(r'(map|ap).*50[-_/]?[95]', c, re.I)]
    if map_5095:
        out["metrics/mAP50-95"] = df[map_5095[0]]
    else:
        out["metrics/mAP50-95"] = np.nan

    # mAP50
    map50 = [c for c in df.columns if re.search(r'(map|ap).*50([^0-9]|$)', c, re.I)]
    if map50:
        out["metrics/mAP50"] = df[map50[0]]
    else:
        out["metrics/mAP50"] = np.nan

    # Common detailed train losses if present (box/cls/dfl/giou/fgl etc.)
    candidates = [
        ("train/box_loss", r"train.*box"),
        ("train/cls_loss", r"train.*cls"),
        ("train/dfl_loss", r"train.*dfl"),
        ("train/giou_loss", r"train.*giou"),
        ("train/fgl_loss", r"train.*fgl"),
        ("val/box_loss", r"val.*box"),
        ("val/cls_loss", r"val.*cls"),
        ("val/dfl_loss", r"val.*dfl"),
        ("val/giou_loss", r"val.*giou"),
        ("val/fgl_loss", r"val.*fgl"),
    ]
    for col_name, pattern in candidates:
        cols = [c for c in df.columns if re.search(pattern, c, re.I)]
        out[col_name] = df[cols[0]] if cols else np.nan

    return out


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    mapping: {source_key_in_df: desired_output_column_name}
    Only includes mapped keys that exist in df; missing keys become NaN columns.
    """
    out = pd.DataFrame()
    # Always include epoch if present
    if "epoch" in df.columns:
        out["epoch"] = df["epoch"]
    for src, dst in mapping.items():
        if src in df.columns:
            out[dst] = df[src]
        else:
            out[dst] = np.nan
    return out


def plot_curves(df: pd.DataFrame, outdir: Path) -> List[Path]:
    """Plot loss curves (columns containing 'loss') and AP curves (columns containing 'AP' or 'mAP')."""
    out_paths = []
    epoch = df["epoch"].values

    # Loss curves
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    if loss_cols:
        plt.figure()
        for c in loss_cols:
            plt.plot(epoch, df[c].values, label=c)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss Curves")
        plt.legend(loc="best")
        p = outdir / "loss_curves.png"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close()
        out_paths.append(p)

    # AP/mAP curves
    ap_cols = [c for c in df.columns if re.search(r'(^|[^a-z])(ap|map)($|[^a-z])', c, re.I)]
    if ap_cols:
        plt.figure()
        for c in ap_cols:
            plt.plot(epoch, df[c].values, label=c)
        plt.xlabel("epoch")
        plt.ylabel("AP / mAP")
        plt.title("AP / mAP Curves")
        plt.legend(loc="best")
        p = outdir / "ap_curves.png"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close()
        out_paths.append(p)

    return out_paths


def main():
    parser = argparse.ArgumentParser(description="Convert DEIMv2 log to CSV (YOLO-style optional) and plots.")
    parser.add_argument("logfile", type=str, help="Path to DEIMv2 log.txt")
    parser.add_argument("--outdir", type=str, default="out_csv", help="Output directory")
    parser.add_argument("--schema", type=str, default="auto", choices=["auto", "raw", "yolo"],
                        help="Which CSV to write primarily; always writes raw; yolo if possible")
    parser.add_argument("--map", type=str, default=None, help="JSON file mapping {source_key: output_name}")
    args = parser.parse_args()

    log_path = Path(args.logfile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_log(log_path)
    # Always save raw wide CSV
    raw_csv = outdir / "deimv2_raw.csv"
    df.to_csv(raw_csv, index=False)

    # Decide secondary CSV
    yolo_df = None
    if args.map:
        with open(args.map, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        mapped_df = apply_mapping(df, mapping)
        mapped_csv = outdir / "deimv2_mapped.csv"
        mapped_df.to_csv(mapped_csv, index=False)
        print(f"[OK] Wrote mapped CSV: {mapped_csv}")
        yolo_df = mapped_df

    if args.schema in ("auto", "yolo") and yolo_df is None:
        yolo_df = best_effort_yolo(df)

    if yolo_df is not None:
        yolo_csv = outdir / "deimv2_yolo.csv"
        yolo_df.to_csv(yolo_csv, index=False)
        print(f"[OK] Wrote YOLO-style CSV: {yolo_csv}")

    # Plots
    outs = plot_curves(df, outdir)
    for p in outs:
        print(f"[OK] Wrote plot: {p}")

    print(f"[OK] Wrote raw CSV: {raw_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()
