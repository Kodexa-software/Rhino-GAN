"""Plot average-loss curves across multiple inversions and transfer tests.

W+/FS inversion CSVs are read from each per-image folder in
``images/output/<name>/{W,FS}_losses.csv``.

Mixing/Preservation CSVs are read from ``transfer_tests/{mixing,preservation}/*.csv``
where each CSV is one test run.

For every source, all CSVs are concatenated and averaged per iteration,
then plotted with one line per CSV column (auto-detected).

Output PNGs land in ``loss_charts/``.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BACKEND_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "images" / "output"
DEFAULT_TRANSFER_DIR = BACKEND_DIR / "transfer_tests"
DEFAULT_TESTS_DIR = BACKEND_DIR / "tests"
DEFAULT_CHART_DIR = BACKEND_DIR / "loss_charts"


def _read_csv(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [{k: float(v) for k, v in row.items()} for row in reader]
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def _collect_nested(parent_dir, filename):
    """Read every <parent_dir>/*/<filename>."""
    runs, fieldnames = [], []
    for folder in sorted(parent_dir.iterdir()):
        if not folder.is_dir():
            continue
        csv_path = folder / filename
        if csv_path.is_file():
            cols, rows = _read_csv(csv_path)
            if not fieldnames:
                fieldnames = cols
            runs.append(rows)
    return fieldnames, runs


def _collect_flat(folder):
    """Read every *.csv directly inside folder."""
    runs, fieldnames = [], []
    if not folder.is_dir():
        return fieldnames, runs
    for csv_path in sorted(folder.glob("*.csv")):
        cols, rows = _read_csv(csv_path)
        if not fieldnames:
            fieldnames = cols
        runs.append(rows)
    return fieldnames, runs


def _average(fieldnames, runs):
    if not runs:
        return None
    loss_cols = [c for c in fieldnames if c != "iter"]
    bucket = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for row in run:
            it = int(row["iter"])
            for col in loss_cols:
                if col in row:
                    bucket[it][col].append(row[col])
    iters = sorted(bucket.keys())
    out = {"iter": np.array(iters, dtype=int)}
    for col in loss_cols:
        out[col] = np.array(
            [float(np.mean(bucket[i][col])) if bucket[i][col] else np.nan for i in iters]
        )
    return out


def _plot(avg, title, out_path, xlim, ylim=None, exclude=None, rename=None, scale=None):
    if avg is None or len(avg["iter"]) == 0:
        print("[SKIP] no data for " + title)
        return

    exclude = set(exclude or ())
    rename = rename or {}
    scale = scale or {}

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    for col, values in avg.items():
        if col == "iter" or col in exclude:
            continue
        factor = scale.get(col, 1.0)
        plotted = values * factor
        label = rename.get(col, col)
        ax.plot(avg["iter"], plotted, label=label, linewidth=1.6, marker="o", markersize=3)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK]   saved: " + str(out_path))


def plot_loss_charts(
    output_dir=DEFAULT_OUTPUT_DIR,
    transfer_dir=DEFAULT_TRANSFER_DIR,
    tests_dir=DEFAULT_TESTS_DIR,
    chart_dir=DEFAULT_CHART_DIR,
    w_xlim=(0, 1100),
    fs_xlim=(0, 250),
    mixing_xlim=None,
    preservation_xlim=None,
    tunning_xlim=None,
):
    output_dir = Path(output_dir)
    transfer_dir = Path(transfer_dir)
    tests_dir = Path(tests_dir)
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    # ---- W+ / FS inversion (per-image subfolders) ----
    if output_dir.is_dir():
        w_cols, w_runs = _collect_nested(output_dir, "W_losses.csv")
        fs_cols, fs_runs = _collect_nested(output_dir, "FS_losses.csv")
        _plot(_average(w_cols, w_runs), "W+ Inversion - Average Loss",
              chart_dir / "W_loss_avg.png", w_xlim)
        _plot(_average(fs_cols, fs_runs), "FS Inversion - Average Loss",
              chart_dir / "FS_loss_avg.png", fs_xlim)
    else:
        print("[MISS] inversion output dir not found: " + str(output_dir))

    # ---- Image Mixing / Preservation (flat folder of test CSVs) ----
    mix_cols, mix_runs = _collect_flat(transfer_dir / "mixing")
    pres_cols, pres_runs = _collect_flat(transfer_dir / "preservation")
    _plot(_average(mix_cols, mix_runs), "Image Mixing - Average Loss",
          chart_dir / "mixing_loss_avg.png", mixing_xlim, ylim=(0, 0.05),
          exclude={"style"}, rename={"total": "mixing loss"})
    _plot(_average(pres_cols, pres_runs), "Preservation - Average Loss",
          chart_dir / "preservation_loss_avg.png", preservation_xlim,
          exclude={"total"})

    # ---- Tunning evaluation (single chart averaging all datasets + variants) ----
    if tests_dir.is_dir():
        all_runs, all_cols = [], []
        for dataset_dir in sorted(tests_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            csv_root = dataset_dir / "csv"
            if not csv_root.is_dir():
                continue
            for variant_dir in sorted(csv_root.iterdir()):
                if not variant_dir.is_dir():
                    continue
                cols, runs = _collect_flat(variant_dir)
                if not all_cols and cols:
                    all_cols = cols
                all_runs.extend(runs)

        _plot(_average(all_cols, all_runs),
              "Nose Shape Refinement (NSR) - Average Loss",
              chart_dir / "nsr_loss_avg.png", tunning_xlim,
              ylim=(0, 0.25),
              exclude={"total"},
              scale={"Segmentation loss": 1.0 })
    else:
        print("[MISS] tests dir not found: " + str(tests_dir))


if __name__ == "__main__":
    plot_loss_charts()
