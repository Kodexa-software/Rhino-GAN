"""
Generate NSR (nose-shape-refinement) targets for the ablation study.

Run inside the rhinogan container from /nose-ai/backend:
    python "/nose-ai/ablation findings/make_targets.py" 4 5

For every image name given it loads output/<name>/FS.npz, regenerates the face with the
StyleGAN2 FS generator (exactly what Tunning.nose_tunning sees), and produces:

  * reference 512x512 BiSeNet label map (argmax)              -> targets/<name>_ref_seg.npy
  * reference 68x3 FAN landmarks (1024 space)                  -> targets/<name>_ref_landmarks.json
  * enlarge target: nose label grown into skin until the nose area is >= +20 %
  * reduce  target: nose label eroded (-> skin) until the nose area is <= -20 %
    (same dilate/erode-by-pixel-count procedure as Research/nose/resize_nose.py, but at 20 %)
  * landmark targets: nose landmarks 27..35 scaled about their centroid by sqrt(1.2) / sqrt(0.8)
    (area-consistent with the mask change), z untouched, all other landmarks unchanged
  * <name>_<enlarge|reduce>.json  = {"segmentation": [[...512x512...]], "landmarks": [[x,y,z,flag]...]}
    which is exactly the payload the frontend sends to POST /fine-tune (JSON-stringified there).
  * <name>_targets_preview.png: visual check of both masks.
"""
import os
import sys
import json
import copy
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

sys.path.insert(0, "/nose-ai/backend")
os.chdir("/nose-ai/backend")

from setting import setting
from models.Net import Net
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.data_utils import load_FS_latent
from utils.fan import extract_landmarks_from_tensor
from utils.helpers import dilate_mask, NOSE_REGION, SKIN_REGION

OUT_DIR = "/nose-ai/ablation findings/targets"
PERCENT = 0.20
os.makedirs(OUT_DIR, exist_ok=True)

opts = copy.deepcopy(setting)
net = Net(opts)
seg_net = BiSeNet(n_classes=16).to(opts.device)
seg_net.load_state_dict(torch.load(opts.seg_ckpt))
seg_net.eval()
downsample = BicubicDownSample(factor=opts.size // 512)


def erode_mask(mask, kernel_size):
    return 1.0 - dilate_mask(1.0 - mask, kernel_size=kernel_size)


def modify_nose(seg_map, enlarge):
    nose = (seg_map == NOSE_REGION)
    nose_f = nose.float().unsqueeze(0).unsqueeze(0)
    n0 = nose.sum().item()
    out = seg_map.clone()
    if enlarge:
        skin = seg_map == SKIN_REGION
        target = n0 * (1 + PERCENT)
        for k in range(3, 151, 2):
            grown = dilate_mask(nose_f, kernel_size=k).squeeze() > 0.5
            add = grown & ~nose & skin
            if n0 + add.sum().item() >= target:
                break
        out[add] = NOSE_REGION
        info = dict(kernel=k, before=int(n0), after=int((out == NOSE_REGION).sum().item()))
    else:
        target = n0 * (1 - PERCENT)
        for k in range(3, 151, 2):
            shrunk = erode_mask(nose_f, kernel_size=k).squeeze() > 0.5
            if shrunk.sum().item() <= target:
                break
        removed = nose & ~shrunk
        out[removed] = SKIN_REGION
        info = dict(kernel=k, before=int(n0), after=int((out == NOSE_REGION).sum().item()))
    return out, info


def scale_nose_landmarks(lm, factor):
    lm = np.array(lm, dtype=np.float32).copy()
    idx = np.arange(27, 36)
    c = lm[idx, :2].mean(axis=0, keepdims=True)
    lm[idx, :2] = c + (lm[idx, :2] - c) * factor
    return lm


def with_flags(lm):
    # frontend format: [x, y, z, flag]; flag=1 for nose indices
    rows = []
    for i, (x, y, z) in enumerate(lm.tolist()):
        rows.append([x, y, z, 1 if 27 <= i <= 35 else 0])
    return rows


for name in sys.argv[1:]:
    fs_path = f"/nose-ai/backend/images/output/{name}/FS.npz"
    S, Fl = load_FS_latent(fs_path, opts.device)
    with torch.no_grad():
        gen, _ = net.generator([S], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=Fl)
        im = (downsample((gen + 1) / 2).clamp(0, 1) - seg_mean) / seg_std
        seg_out, _, _ = seg_net(im)
        seg_map = torch.argmax(seg_out, dim=1).long().squeeze(0)      # 512x512
    # fan.py asserts the landmarks are differentiable, so run it with grad enabled
    lm = extract_landmarks_from_tensor(gen.detach().requires_grad_(True), device=opts.device)  # 68x3, 1024 space

    np.save(f"{OUT_DIR}/{name}_ref_seg.npy", seg_map.cpu().numpy())
    lm_np = lm.detach().cpu().numpy()
    json.dump(with_flags(lm_np), open(f"{OUT_DIR}/{name}_ref_landmarks.json", "w"))

    previews = [seg_map]
    for mode, enlarge, factor in [("enlarge", True, np.sqrt(1 + PERCENT)), ("reduce", False, np.sqrt(1 - PERCENT))]:
        tgt_seg, info = modify_nose(seg_map, enlarge)
        tgt_lm = scale_nose_landmarks(lm_np, factor)
        payload = {
            "segmentation": tgt_seg.cpu().numpy().astype(int).tolist(),
            "landmarks": with_flags(tgt_lm),
            "info": dict(mode=mode, percent=PERCENT, landmark_scale=float(factor), **info),
        }
        json.dump(payload, open(f"{OUT_DIR}/{name}_{mode}.json", "w"))
        print(name, mode, info, "lm scale", round(float(factor), 4))
        previews.append(tgt_seg)

    # preview: original face + nose masks (ref / enlarge / reduce) as overlays
    face = ((gen[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    face = (Image.fromarray((face * 255).astype(np.uint8)).resize((512, 512)))
    tiles = []
    for s in previews:
        m = (s == NOSE_REGION).cpu().numpy()
        arr = np.array(face).astype(np.float32)
        arr[m] = arr[m] * 0.4 + np.array([0, 80, 255]) * 0.6
        tiles.append(Image.fromarray(arr.astype(np.uint8)))
    canvas = Image.new("RGB", (512 * 4, 512))
    canvas.paste(face, (0, 0))
    for i, t in enumerate(tiles):
        canvas.paste(t, (512 * (i + 1), 0))
    canvas.save(f"{OUT_DIR}/{name}_targets_preview.png")

print("done")
