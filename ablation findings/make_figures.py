"""
Build the two deliverable figures from the run outputs (host side, PIL only).

    python "ablation findings/make_figures.py"

fig_nsb.png : rows = NSB cases (base image 4, nose from 5); cols = Identity, Target, Full, No mixing,
              No preservation, No nose loss
fig_nsr.png : rows = NSR cases on image 4 (enlarge +20 %, reduce -20 %); cols = Original, Full,
              No segmentation, No landmark, No nose style, No identity

Every panel is the full 1024x1024 face resized with the same LANCZOS filter to PANEL px, so crop,
scale and resolution are identical across a figure. Width = 6*PANEL + gaps ~= 2480 px = A4 at 300 dpi.
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
ORIG = os.path.join(ROOT, "originals")
PANEL, GAP, HEADER, MARGIN = 400, 8, 64, 16
FONT = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 34)

NSB_ROWS = [("4", "5")]                                   # (identity/base, target/nose donor)
NSB_COLS = ["Identity", "Target", "Full", "No mixing", "No preservation", "No nose loss"]
NSR_ROWS = [("4", "enlarge"), ("4", "reduce")]
NSR_COLS = ["Original", "Full", "No segmentation", "No landmark", "No nose style", "No identity"]


def grid(rows_of_paths, headers, out):
    n_rows, n_cols = len(rows_of_paths), len(headers)
    W = 2 * MARGIN + n_cols * PANEL + (n_cols - 1) * GAP
    H = MARGIN + HEADER + n_rows * PANEL + (n_rows - 1) * GAP + MARGIN
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for c, h in enumerate(headers):
        x = MARGIN + c * (PANEL + GAP)
        tw = d.textlength(h, font=FONT)
        d.text((x + (PANEL - tw) / 2, MARGIN + (HEADER - 40) / 2), h, fill="black", font=FONT)
    for r, paths in enumerate(rows_of_paths):
        for c, p in enumerate(paths):
            im = Image.open(p).convert("RGB")
            assert im.size == (1024, 1024), p
            im = im.resize((PANEL, PANEL), Image.LANCZOS)
            canvas.paste(im, (MARGIN + c * (PANEL + GAP), MARGIN + HEADER + r * (PANEL + GAP)))
    canvas.save(out, dpi=(300, 300))
    print("wrote", out, canvas.size)


grid([[f"{ORIG}/{a}.png", f"{ORIG}/{b}.png"] + [f"{RUNS}/nsb_{a}_from_{b}/B{i}.png" for i in range(4)] for a, b in NSB_ROWS],
     NSB_COLS, os.path.join(ROOT, "fig_nsb.png"))
grid([[f"{ORIG}/{a}.png"] + [f"{RUNS}/nsr_{a}_{m}/N{i}.png" for i in range(5)] for a, m in NSR_ROWS],
     NSR_COLS, os.path.join(ROOT, "fig_nsr.png"))
