# Ablation study — prompt for the repo agent

Paste everything below the line into the agent running in the RhinoGAN code repository, with the
paper attached.

---

I've attached a paper describing the method implemented in this repository. Your task is an
ablation study. **The deliverable is two figures.** Everything else exists to produce them.

Read the paper first. The names in it will not match the names in this code — it was written for
a different audience and uses presentation terminology. Mapping the paper's concepts onto the
actual implementation is the first and most important part of this task. Do not assume any name
below appears verbatim in the code.

## Phase 1 — Map the paper onto the code

The method has two editing modules that operate on inverted latent codes. In the paper they are
called NSB and NSR. Find both in this repository and identify, for each, the individual loss
terms and optimization stages.

**NSB** transfers a nose from a target image onto an identity image. The paper describes three
stages:

1. An optimization over a mixed structure tensor, ~10 iterations, driven by a perceptual (LPIPS)
   loss between row-aligned nose crops.
2. A masked blend combining the identity and mixed tensors using a nose mask. No loss, no
   optimization — pure tensor arithmetic.
3. A "preservation" optimization, ~40 iterations, with two loss terms: one perceptual loss on the
   nose region, one perceptual loss on everything outside the nose.

**NSR** reshapes the nose in a single image toward a target segmentation mask and target
landmarks. One optimization, ~50 iterations, with four loss terms summed together:

- a cross-entropy term against a target segmentation mask
- an L1 term against target landmark coordinates
- a perceptual term on the row-aligned nose crop, against the original image
- a perceptual term on the non-nose region, against the original image

**Hints for locating these.** The loss weights in the paper are 0.3, 0.01, 0.1 and 1.0 for the
four NSR terms, and 0.1 for two of the NSB terms — grepping for those literals is likely faster
than reading. Iteration counts of 10, 40 and 50 are similarly distinctive. This code is derived
from BarbershopGAN, so some names may still reflect hair editing rather than nose editing.

**Output of this phase:** print a table mapping each paper concept to its file, function or
config key in this repo. If anything is ambiguous or you cannot find it, stop and ask me rather
than guessing. Getting this wrong makes everything downstream worthless.

## Phase 2 — Sanity check before ablating

Run the unmodified full model on a few cases from the II2S dataset and confirm the output is
consistent with what the paper reports for II2S. If it clearly isn't, stop and tell me — there is
a setup problem, and ablating a broken baseline produces nothing.

## Phase 3 — Run the configurations

Nine runs total. **30 cases per part**, selected with a fixed seed that you record.

**Part A — NSB.** 30 identity/target image pairs from II2S.

| ID  | Configuration                                                                    |
| --- | -------------------------------------------------------------------------------- |
| B0  | Unmodified full model                                                            |
| B1  | Skip stage 1 — no mixing optimization; use the target structure tensor directly  |
| B2  | Skip stage 3 entirely — output comes straight from the masked blend, unoptimized |
| B3  | Run stage 3 but remove its nose-region loss term; keep the non-nose term         |

**Part B — NSR.** 30 images from II2S, 15 enlarged by 20% and 15 reduced by 20%. If this repo
already has a stored set of target segmentation masks and target landmarks from a previous
evaluation, reuse them. Do not regenerate targets.

| ID  | Configuration                          |
| --- | -------------------------------------- |
| N0  | Unmodified full model                  |
| N1  | Remove the segmentation term           |
| N2  | Remove the landmark term               |
| N3  | Remove the nose-region perceptual term |
| N4  | Remove the non-nose perceptual term    |

**How to implement the ablations.** Add a runtime flag or config override. Do not comment out
lines in the main code path, do not change any default, and do not commit modifications to the
default behaviour. After the run, the repository must still produce identical results to before
when invoked normally.

**What must not change.** Iteration counts, learning rates, model checkpoints, the dataset, the
targets, and the seed are identical across every configuration. Removing a term means dropping it
from the sum — do not rescale or renormalize the remaining weights to compensate, and do not
re-tune anything. No hyperparameter sweeps. No step-count experiments. The only difference
between any two runs is the presence or absence of one term or one stage.

**Reuse inversions.** If cached inverted latents exist for these images, use them. If not,
compute them once and cache — all nine runs share one set. Never invert per configuration.

**Determinism.** Fix torch and numpy seeds. Enable deterministic algorithms if it runs clean;
if a kernel refuses, leave it off and note that.

## Phase 4 — Build the two figures

This is the actual deliverable.

**Case selection.** Take the 1st, 15th and 30th case in your seeded order. Fix this _before_
looking at which cases make the ablations look worst. If one is degenerate for a reason unrelated
to the ablation — a failed inversion, a broken segmentation — substitute the next in order and
record that you did. Do not choose cases for dramatic failure.

**`ablation/fig_nsb.png`** — a 3 × 6 grid. Rows are the three cases. Columns in order:
identity image, target image, B0, B1, B2, B3.

**`ablation/fig_nsr.png`** — a 3 × 6 grid. Rows are the three cases. Columns in order:
original image, N0, N1, N2, N3, N4.

**Rendering requirements:**

- Identical crop, scale and resolution in every panel of a figure. A reader comparing nose shape
  must never be comparing two different zoom levels.
- Full face in every panel — identity preservation cannot be judged from a cropped nose.
- Column headers on the top row only. Use short labels: "Identity", "Target", "Full", "No mixing",
  "No preservation", "No nose loss" for the first figure; "Original", "Full", "No segmentation",
  "No landmark", "No nose style", "No identity" for the second.
- Sized to stay legible at A4 print width.
- No annotations, arrows or zoom insets.

## Also produce

Per-case metrics, using whatever evaluation code already exists in this repo — do not write new
metrics. At minimum: identity similarity (ArcFace or equivalent), perceptual distance on the
non-nose region, and perceptual distance on the nose region. Write one CSV per configuration plus
a `summary.csv` with mean and standard deviation per metric per configuration.

These are not going into the write-up. They exist so I can confirm the three cases in each figure
are typical rather than exceptional.

## Report honestly

Each removed term is expected to visibly damage something. If one of them doesn't — if a
configuration is indistinguishable from the full model — say so plainly. That is a real result and
I need to know it. The landmark term in particular has a very small weight and may contribute
little. Do not tune, reweight or reselect cases to produce a more convincing failure.

## Deliverables, all under `ablation/`

1. `fig_nsb.png` and `fig_nsr.png`
2. `results_B0.csv` … `results_B3.csv`, `results_N0.csv` … `results_N4.csv`, `summary.csv`
3. `README.md` containing: the Phase 1 mapping table, exact commands run, git commit hash, GPU,
   seed, the seeded case order, which three cases went into each figure and any substitutions,
   wall-clock per configuration, and anything you did differently from these instructions

Expect roughly 25–30 minutes of GPU time if cached latents exist.
