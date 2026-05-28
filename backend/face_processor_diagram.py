"""Generate a simple workflow diagram for face_processor.py."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUTPUT_PATH = Path(__file__).with_name("face_processor_workflow.png")

C_INPUT = "#E8F1FA"
C_INPUT_EDGE = "#3A6FA0"
C_PROC = "#FFF3DC"
C_PROC_EDGE = "#B38600"
C_DECISION = "#F4E3FA"
C_DECISION_EDGE = "#7A3E8E"
C_ERROR = "#FBE3E3"
C_ERROR_EDGE = "#A83232"
C_OUTPUT = "#E1F4E5"
C_OUTPUT_EDGE = "#2E7D4F"
TEXT = "#1A1A1A"

fig, ax = plt.subplots(figsize=(9, 11), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fontsize=12, weight="normal"):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=1.6,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=TEXT, weight=weight)


def diamond(x, y, w, h, text, fc, ec, fontsize=12):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    poly = mpatches.Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.6)
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=TEXT)


def arrow(x1, y1, x2, y2, label=None, label_offset=(0.18, 0)):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color="#333333",
    )
    ax.add_patch(a)
    if label:
        ax.text(
            (x1 + x2) / 2 + label_offset[0],
            (y1 + y2) / 2 + label_offset[1],
            label, fontsize=10, color="#333333", style="italic",
        )


# Blocks
box(5, 11.0, 5.0, 0.9, "Input image", C_INPUT, C_INPUT_EDGE, fontsize=13, weight="bold")
box(5, 9.4,  5.0, 0.9, "Detect face", C_PROC, C_PROC_EDGE, fontsize=13)
box(5, 7.8,  5.0, 0.9, "Center and align the face", C_PROC, C_PROC_EDGE, fontsize=13)
diamond(5, 6.0, 3.6, 1.3, "Face found?", C_DECISION, C_DECISION_EDGE, fontsize=12)
box(8.6, 6.0, 2.4, 0.9, "Stop with error", C_ERROR, C_ERROR_EDGE, fontsize=12, weight="bold")
box(5, 4.0,  5.0, 0.9, "Resize to 1024×1024", C_PROC, C_PROC_EDGE, fontsize=13)
box(5, 2.4,  5.0, 0.9, "Save aligned face", C_PROC, C_PROC_EDGE, fontsize=13)
box(5, 0.8,  5.0, 0.9, "Output image", C_OUTPUT, C_OUTPUT_EDGE, fontsize=13, weight="bold")

# Arrows
arrow(5, 10.55, 5, 9.85)
arrow(5, 8.95,  5, 8.25)
arrow(5, 7.35,  5, 6.65)
arrow(6.8, 6.0, 7.4, 6.0, label="No", label_offset=(-0.05, 0.22))
arrow(5, 5.35,  5, 4.45, label="Yes", label_offset=(0.18, 0))
arrow(5, 3.55,  5, 2.85)
arrow(5, 1.95,  5, 1.25)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTPUT_PATH}")
