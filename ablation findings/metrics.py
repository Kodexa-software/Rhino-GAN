"""
Per-case metrics for the ablation runs, reusing the evaluation code from the Research repo
(C:/projects/Research/nose/quantative/ttests.py -> class Ttest). Nothing new is defined here:
every number comes from a Ttest method (or the repo's BiSeNet / FAN helpers used by Ttest).

Runs inside the Research container (ai-container-server, /ai = C:/projects/Research) because
insightface (ArcFace) is only installed there:

    docker cp "ablation findings" ai-container-server:/tmp/ablation
    docker exec ai-container-server bash -lc "cd /ai/nose && python /tmp/ablation/metrics.py /tmp/ablation"
    docker cp ai-container-server:/tmp/ablation/results_*.csv "ablation findings/"

Metrics (all vs the ORIGINAL identity image unless stated):
  arcface_identity  Ttest.compute_arcface_score      cosine similarity, 1.0 = same person   (higher better)
  lpips_non_nose    Ttest.compute_identity_score     LPIPS with the union nose mask blacked out (lower = identity kept)
  lpips_nose        Ttest.compute_nose_score         LPIPS inside the union nose mask       (NSB: vs target image's nose)
  lpips_nose_RA     Ttest.compute_RA_lpips_score     row-aligned grayscale nose-crop LPIPS  (NSB only, vs target image)
  nose_area_ratio   BiSeNet nose pixel count / original nose pixel count; nose_area_target_ratio = same for the target mask (1.2 / 0.8)
  lm_l1_to_target   L1 of FAN nose landmarks (27..35) to the NSR target landmarks, px in 1024 space (NSR only)
"""
import os
import sys
import csv
import json
import copy
import numpy as np
import torch

sys.path.insert(0, "/ai/nose")
os.chdir("/ai/nose")
from setting import setting
from quantative.ttests import Ttest, NOSE, nose_idxs
from utils.fan import extract_landmarks_from_tensor

ROOT = sys.argv[1]
RUNS = os.path.join(ROOT, "runs")
ORIG = {"4": os.path.join(ROOT, "originals", "4.png"), "5": os.path.join(ROOT, "originals", "5.png")}

te = Ttest(copy.deepcopy(setting))
# insightface's default det_thresh=0.5 misses image 4 (eyes squeezed shut, wide grin) at the 640 px the
# Ttest code feeds it; 0.3 detects it (det_score 0.374). Only the detector threshold changes, not the metric.
te.arcface.prepare(ctx_id=0, det_thresh=0.3)


def nose_area(img):
    return float(te.get_nose_mask(img).sum().item())


def nose_landmarks(img):
    return extract_landmarks_from_tensor(img.detach().requires_grad_(True), device="cuda")[nose_idxs][:, :2].detach()


def nsb_metrics(identity_png, target_png, result_png):
    a, t, r = te.load_png(identity_png), te.load_png(target_png), te.load_png(result_png)
    mask_a, mask_t = te.get_nose_mask(a), te.get_nose_mask(t)
    return {
        "arcface_identity": te.compute_arcface_score(te.downsample_to_640(a), te.downsample_to_640(r)),
        "lpips_non_nose": te.compute_identity_score(a, r, mask_a).item(),
        "lpips_nose_vs_target": te.compute_nose_score(t, r, mask_t).item(),
        "lpips_nose_RA_vs_target": te.compute_RA_lpips_score(t, r, mask_t).item(),
        "lpips_nose_vs_identity": te.compute_nose_score(a, r, mask_a).item(),
    }


def nsr_metrics(orig_png, result_png, target):
    a, r = te.load_png(orig_png), te.load_png(result_png)
    mask_a = te.get_nose_mask(a)
    tgt_lm = torch.tensor(target["landmarks"], dtype=torch.float32, device="cuda")[nose_idxs][:, :2]
    tgt_area = float((np.array(target["segmentation"]) == NOSE).sum()) * 4.0   # 512^2 label grid -> 1024^2 mask grid
    return {
        "arcface_identity": te.compute_arcface_score(te.downsample_to_640(a), te.downsample_to_640(r)),
        "lpips_non_nose": te.compute_identity_score(a, r, mask_a).item(),
        "lpips_nose": te.compute_nose_score(a, r, mask_a).item(),
        "nose_area_ratio": nose_area(r) / nose_area(a),
        "nose_area_target_ratio": tgt_area / nose_area(a),
        "lm_l1_to_target_px": (nose_landmarks(r) - tgt_lm).abs().mean().item(),
        "lm_l1_orig_to_target_px": (nose_landmarks(a) - tgt_lm).abs().mean().item(),
    }


rows_by_cfg = {}
for case in sorted(os.listdir(RUNS)):
    d = os.path.join(RUNS, case)
    if not os.path.isdir(d):
        continue
    parts = case.split("_")
    for f in sorted(os.listdir(d)):
        if not f.endswith(".png"):
            continue
        cfg = f[:-4]
        res = os.path.join(d, f)
        if parts[0] == "nsb":                       # nsb_<identity>_from_<target>
            m = nsb_metrics(ORIG[parts[1]], ORIG[parts[3]], res)
        else:                                        # nsr_<img>_<mode>
            tgt = json.load(open(os.path.join(ROOT, "targets", f"{parts[1]}_{parts[2]}.json")))
            m = nsr_metrics(ORIG[parts[1]], res, tgt)
        rows_by_cfg.setdefault(cfg, []).append({"case": case, **m})
        print(case, cfg, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})

summary = []
for cfg, rows in sorted(rows_by_cfg.items()):
    keys = [k for k in rows[0] if k != "case"]
    with open(os.path.join(ROOT, f"results_{cfg}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case"] + keys)
        w.writeheader()
        w.writerows(rows)
    for k in keys:
        vals = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
        summary.append({"config": cfg, "metric": k, "n": len(vals), "mean": vals.mean() if len(vals) else "", "std": vals.std() if len(vals) else ""})
with open(os.path.join(ROOT, "summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["config", "metric", "n", "mean", "std"])
    w.writeheader()
    w.writerows(summary)
print("done")
