# RhinoGAN ablation study — findings

Deliverables in this folder:

| File | What |
|---|---|
| `fig_nsb.png` | NSB figure, 1 × 6: Identity (4), Target (5), Full, No mixing, No preservation, No nose loss |
| `fig_nsr.png` | NSR figure, 2 × 6 (row 1 = enlarge +20 %, row 2 = reduce −20 %, both on image 4): Original, Full, No segmentation, No landmark, No nose style, No identity |
| `results_B0..B3.csv`, `results_N0..N4.csv` | per-case metrics (one row per case: `nsb_4_from_5`, `nsr_4_enlarge`, `nsr_4_reduce`) |
| `results_B0rep / B0_s7 / B0_s99`, `results_N0rep / N0_s7 / N0_s99` | full-model repeats used as the **noise floor** (see §5) |
| `summary.csv` | mean / std per metric per configuration over the cases in `runs/` (NSB n=1, NSR n=2) |
| `runs/<case>/<config>.png` | every output image; `runs/timing.csv` wall-clock; `runs/cases.json` seeded case order |
| `targets/` | generated NSR targets (`4_enlarge.json`, `4_reduce.json`, reference seg/landmarks) + `4_targets_preview.png` |
| `_work/` | scratch previews: `full_*.png` (uncropped strips) and `crop_*.png` (same fixed nose crop, for reading the differences) |
| `make_targets.py`, `run_ablation.py`, `metrics.py`, `make_figures.py` | the scripts, in the order they were run |

Scope as requested by the author (deviations from `ablation-command.md` are in §7): **one base image, `output/4`**.
NSB = nose of `output/5` transferred onto `4`; NSR = `4` enlarged / reduced by 20 %. Both images were already
inverted (`FS.npz` cached) — no inversion was run.

---

## 1. Phase 1 — mapping the paper onto the code

| Paper concept | Repo location | Notes |
|---|---|---|
| **NSB** (nose structure transfer) | `backend/models/StructureTransfer.py` → `StructureTransfer.transfer(I1, I2, …)`; entry `Tunning.nose_transfer` | `I1` = identity (`fullPath`), `I2` = target (`noseStyle`) |
| NSB stage 1 — mixing optimisation, ~10 it, LPIPS on row-aligned nose crops | `transfer()` loop over `opts.Transfer_restructure_steps = 10`; loss = `lpips(aligned_gen, aligned_ref)` × `Transfer_restructure_perceptual_lambda = 0.1`; optimises `F_mix` only | **Code differs from the paper text**: `F_mix = swap_F_noses(...)` at line 156 is immediately overwritten by `F_mix = F2.clone()` (line 160), so stage 1 optimises the *target's* structure tensor F2 directly, not a mixed tensor. The first `swap_F_noses` call is dead code. |
| NSB stage 2 — masked blend, no loss | `swap_F_noses(ref_seg, F1, nose_seg, F2)` called after the loop (line ~209): dilated nose masks (k=21) minus dilated "red line" non-skin/non-nose regions (k=19), `F_mix = F1·(1−m) + F2·m` at F resolution | pure tensor arithmetic, as described |
| NSB stage 3 — preservation, ~40 it, nose term + non-nose term | `transfer()` second loop, `opts.Transfer_perceptual_steps = 40`, optimises `F_fixed, S_fixed`; `nose_image_loss` (LPIPS, mask = union of dilated noses, k=3) × `Transfer_perceptual_nose_lambda = 0.1`; `rest_image_loss` (LPIPS on the complement) × `Transfer_perceptual_face_lambda = 1.0` | both at 256 px |
| **NSR** (nose shape refinement) | `backend/models/Tunning.py` → `Tunning.nose_tunning(item)`; `opts.Tune_steps = 50`, optimises `F, S` | reached via `POST /fine-tune` with `noseStyle: "self"` |
| NSR cross-entropy vs target mask (0.3) | `ce_loss = cross_entropy(down_seg, item.segmentation) * opts.Tunning_segmentation_lambda (0.3)` | 512×512 BiSeNet logits vs user label map |
| NSR L1 vs target landmarks (0.01) | `landmark_loss = l1(lm_x, lm_y) * opts.Tunning_landmarks_lambda (0.01)`; `lm_x` from differentiable FAN (`utils/fan.py`), only `nose_idxs = 27..35` | |
| NSR perceptual on row-aligned nose crop vs original (0.1) | `style_loss = opts.Tunning_nose_perceptual_lambda (0.1) * lpips(aligned_gen, aligned_ref)`; crops via `extract_and_align_noses` (grayscale, masked) | reference = original FS image regenerated each step |
| NSR perceptual on non-nose region vs original (1.0) | `face_style = compare_LPIPS(self, ref_im_L, gen_L, nose_out_mask)`; `percept_lambda = 1.0`; mask = outside (original nose ∪ target nose) | |
| Learning rate / optimiser | `opts.learning_rate = 0.01`, Adam, for all stages | unchanged |
| Checkpoints | `pretrained_models/ffhq.pt`, `seg.pth`, FAN (face_alignment), LPIPS vgg | unchanged |

Nothing in the mapping was ambiguous except the dead `swap_F_noses` call noted above; the ablation "B1 — use the
target structure tensor directly" therefore coincides with what the code already starts stage 1 from, so **B1 = B0
minus the 10 stage-1 iterations**.

### How the ablation switch is implemented (no default changed)

`POST /fine-tune` now accepts an optional `ablation` object (`backend/main.py` → `FineTuneRequest.ablation`,
`backend/models/Tunning.py` → `AblationConfig`):

```json
{"name": "<run name>", "drop": ["mixing" | "preservation" | "nose_loss" | "segmentation" | "landmarks" | "nose_style" | "face_style"],
 "seed": 2026, "deterministic": true}
```

* When the field is absent every code path is the original one (the guards are `if not ablation_dropped(...)`).
* `drop` removes a term from the sum / skips a stage; nothing is rescaled.
* Results go to `output/<img>/tuned/ablation/<name>/FS.png` instead of `tuned/FS.png`, so normal results are untouched.
* `seed` seeds `random`, `numpy`, `torch`, `torch.cuda`; `deterministic` sets `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`.
  Caveat: those two settings are **process-global** and stay on until the backend is restarted (it was restarted afterwards).

Diff: `backend/main.py` (+5), `backend/models/Tunning.py` (guards + `AblationConfig` + `save_result(subdir=)`),
`backend/models/StructureTransfer.py` (`transfer(..., ablation=None)` + three `if`s). Not committed.

## 2. Phase 2 — sanity check

Unmodified NSB 5 → 4 through the normal endpoint (`{"fullPath":"4/4.png","noseStyle":"5/5.png"}`, no `ablation`
field): 50 steps in 15 s, output `output/4/tuned/FS.png`. Identity kept (hair, expression, lips, background),
nose ring removed, bridge narrower / straighter like image 5 — consistent with the paper's NSB behaviour
(`_work/sanity_nsb.png`). No II2S set is present in this repo (only `output/4`, `output/5`), so the check is on
these two cases.

## 3. Environment

* git commit `0b93e0e4b248759aa852ab138b808a327bbe2d59` (branch `main`, ablation edits uncommitted on top)
* GPU: NVIDIA GeForce RTX 4080 SUPER 16 GB, driver 591.86; container `rhinogan-container`, torch 1.13.1+cu117, Python 3.7
* metrics: container `ai-container-server` (Research repo mounted at `/ai`), insightface 0.7.3 (ArcFace buffalo_l), same LPIPS vgg
* seed **2026** (torch / numpy / random) for every configuration; extra seeds 7 and 99 only for the B0/N0 noise-floor repeats
* determinism: `torch.use_deterministic_algorithms(True, warn_only=True)` — three kernels refused
  (`grid_sampler_2d_backward_cuda`, `reflection_pad2d_backward_cuda`, cuBLAS without `CUBLAS_WORKSPACE_CONFIG`), so it ran
  warn-only. Empirically NSB is bit-reproducible with a fixed seed (B0 = B0rep, max pixel diff 0); NSR is not
  (N0 vs N0rep: mean |Δ| 1.5 / 255, from the FAN + grid_sample backward).
* **Important:** the StyleGAN2 generator is called with `randomize_noise=True` (default in `models/stylegan2/model.py`),
  so every forward draws fresh noise. Two unseeded runs of the *same* config differ by mean |Δ| ≈ 3.4 / 255 (max ≈ 70).
  Configurations that skip generator calls (B1, B2, N3) therefore also shift the noise stream; pixel differences for
  them cannot be attributed to the dropped term alone — see §5 for the seed-spread noise floor used instead.

## 4. Exact commands

```bash
# backend (inside rhinogan-container)
cd /nose-ai/backend && python main.py            # started after the code edits, restarted at the end

# targets (inside rhinogan-container)
cd /nose-ai/backend && python "/nose-ai/ablation findings/make_targets.py" 4 5

# sanity baseline (host)
curl -X POST localhost:8001/fine-tune -H "Content-Type: application/json" -d '{"fullPath":"4/4.png","noseStyle":"5/5.png"}'

# all configurations (host; POST /fine-tune per run, polls /processes and the output file)
python "ablation findings/run_ablation.py" nsb
python "ablation findings/run_ablation.py" nsr
# noise-floor repeats: run_ablation.run(...) called with names B0rep / N0rep (seed 2026) and B0_s7, B0_s99, N0_s7, N0_s99

# metrics (Research container, reuses Research/nose/quantative/ttests.py::Ttest)
docker cp "ablation findings" ai-container-server:/tmp/ablation
docker exec ai-container-server bash -lc "cd /ai/nose && python /tmp/ablation/metrics.py /tmp/ablation"
docker cp ai-container-server:/tmp/ablation/summary.csv "ablation findings/"   # + results_*.csv

# figures (host)
python "ablation findings/make_figures.py"
```

### Cases and seeded order

Base image is `4` throughout (author's instruction). NSB: one pair, `4 ← 5` (base 4, nose from 5). NSR: `4 reduce`,
`4 enlarge` — this is the `random.Random(2026).shuffle` order (`runs/cases.json`). Figure rows follow that order except
that the NSR figure shows enlarge first, then reduce, for readability. No substitutions were needed (no failed
inversion / broken segmentation).

The reverse pair (`5 ← 4`) and NSR on image 5 were run first by mistake and **deleted at the author's request** so
the findings stay focused on image 4; nothing from them is used below.

### NSR targets (generated — none were stored in the repo)

Same procedure as the paper's NSR evaluation script (`Research/nose/resize_nose.py`) but at 20 %: the BiSeNet nose
label is dilated into *skin* (enlarge, kernel 7 → 5421 → 6573 px = +21 %) or eroded to skin (reduce, kernel 9 →
5421 → 4005 px = −26 %; erosion is quantised by kernel size, the first kernel reaching ≤ −20 %). Landmark targets:
nose landmarks 27–35 scaled about their centroid by √1.2 = 1.095 / √0.8 = 0.894 (area-consistent), z unchanged.
Reference seg/landmarks were computed on the FS-regenerated image inside the container, exactly as the loss sees them.

### Wall-clock per configuration (image-4 cases, `runs/timing.csv`)

| B0 | B1 | B2 | B3 | N0 | N1 | N2 | N3 | N4 |
|---|---|---|---|---|---|---|---|---|
| 7.1 s | 6.0 s | 3.0 s | 6.1 s | 16–21 s | 15.3 s | 11.3 s | 12.3 s | 15.3 s |

Whole study incl. repeats and metrics ≈ 12 min GPU time.

## 5. Results

### 5a. NSB — 5's nose onto 4 (`results_B*.csv`, row `nsb_4_from_5`)

| config | ArcFace ↑ | LPIPS non-nose ↓ | LPIPS nose vs target ↓ | LPIPS nose RA vs target ↓ | LPIPS nose vs identity | mean |Δpx| vs B0 |
|---|---|---|---|---|---|---|
| **B0 Full** (seed 2026) | 0.740 | 0.0363 | 0.0096 | 0.289 | 0.0086 | 0 |
| B0 seed 7 / seed 99 (noise floor) | 0.727 / 0.695 | 0.0396 / 0.0410 | 0.0083 / 0.0081 | 0.291 / 0.287 | 0.0086 / 0.0090 | 3.4 |
| **B1 No mixing** | 0.714 | 0.0388 | 0.0104 | 0.297 | 0.0084 | 3.2 |
| **B2 No preservation** | **0.650** | 0.0401 | 0.0082 | **0.270** | 0.0090 | 3.9 |
| **B3 No nose loss** | 0.748 | 0.0365 | 0.0096 | 0.296 | **0.0066** | 0.8 |

* **B2 (skip stage 3)** is the only NSB ablation with a clearly visible effect: the raw blend gives a paler, flatter,
  waxy-looking nose with over-smoothed skin around it and a weakened nasolabial fold (`_work/crop_nsb_4_from_5.png`,
  `_work/zoom_check.png` = B0 | B2 | B1 | B3 at 1:1) — not a hard seam in this pair — and ArcFace drops below the seed
  spread (0.650 vs 0.695–0.740). Its RA nose distance to the target is *lower* than
  Full — the unoptimised blend keeps more of the donor F, and stage 3 trades some of that back for identity.
* **B1 (skip stage 1)** is **indistinguishable from Full within the noise floor**: every metric sits inside the seed-to-seed
  spread of B0, and the pixel difference (3.2) equals the difference between two seeds of the same config (3.4).
  Given that stage 1 starts from F2 and runs only 10 iterations at λ = 0.1 (loss moves 0.256 → 0.235), this is expected.
* **B3 (no nose term in stage 3)** has a small but real effect: same noise stream as B0 (same generator-call count),
  pixel diff 0.8, and the nose drifts back towards the identity's own nose (LPIPS nose-vs-identity 0.0066 vs 0.0086,
  RA-vs-target 0.296 vs 0.289). Visually it is a subtle softening of the transferred bridge, not a failure.

### 5b. NSR on image 4 (`results_N*.csv`, rows `nsr_4_enlarge`, `nsr_4_reduce`)

Target nose-area ratio: 1.23 (enlarge) / 0.75 (reduce). Landmark L1 of the *original* to the target: 3.2 px / 4.5 px.

| config | ArcFace ↑ (enl / red) | LPIPS non-nose ↓ | LPIPS nose (vs orig) | nose area ratio (enl / red) | landmark L1 to target px (enl / red) |
|---|---|---|---|---|---|
| **N0 Full** | 0.937 / 0.890 | 0.041 / 0.041 | 0.0031 / 0.0031 | 1.10 / 0.86 | 2.06 / 2.10 |
| N0 repeats (seed 2026 rep, 7, 99) | 0.92–0.95 / 0.88–0.88 | 0.040–0.042 / 0.042–0.043 | 0.003–0.003 / 0.003–0.005 | 1.07–1.11 / 0.82–0.85 | 1.84–2.16 / 2.13–2.24 |
| **N1 No segmentation** | 0.945 / 0.937 | 0.040 / 0.038 | **0.0016 / 0.0021** | **1.05 / 0.90** | 1.93 / 2.10 |
| **N2 No landmark** | 0.923 / 0.896 | 0.039 / 0.039 | 0.0038 / 0.0049 | 1.10 / 0.89 | **3.65 / 5.42** |
| **N3 No nose style** | 0.908 / 0.891 | 0.041 / 0.041 | **0.0050 / 0.0061** | 1.08 / 0.82 | 2.23 / 1.96 |
| **N4 No identity** | **0.748 / 0.707** | **0.276 / 0.279** | 0.0044 / 0.0052 | 1.08 / 0.84 | 1.88 / 2.02 |

* **N4 (no non-nose perceptual term)** is catastrophic and obvious at figure scale: skin tone, eyes/eyelids, lips,
  wrinkles and the background all drift (LPIPS non-nose ×7, ArcFace 0.94 → 0.75). The nose target is still met.
* **N1 (no segmentation term)** roughly halves the shape change (area 1.05 vs 1.10 when enlarging, 0.90 vs 0.86 when
  reducing — both outside the seed spread) and the nose stays closer to the original (LPIPS nose halves). The landmark
  term alone still moves the nose part-way. Visually: a weaker version of Full.
* **N2 (no landmark term)** — the term does something measurable but small: the nose landmarks no longer move toward
  the target (L1 3.7 / 5.4 px, i.e. no better — or worse — than the untouched original at 3.2 / 4.5 px, versus 2.1 px
  with the term), while the mask-driven area change is unaffected (1.10 / 0.89). In the figure it is
  **not distinguishable from Full by eye**; the full-face pixel difference (2.3) is close to the NSR run-to-run noise (1.5).
  This is the "small-weight term contributes little" outcome the instructions anticipated, and it is real.
* **N3 (no nose-style term)** — the nose texture/shading drifts more from the original (LPIPS nose ≈ ×1.7, above the
  seed spread) and ArcFace drops slightly (0.908 vs 0.92–0.95 when enlarging). Visually a subtle change in bridge shading;
  not a failure. Note N3 also shifts the noise stream (one generator call fewer per step).
* Neither Full nor any ablation reaches the full ±20 % area target in 50 steps (Full: +10 % / −14 %). That is a property
  of the unmodified model, not of the ablations.

### 5c. Are the figure cases typical?

The figure rows are the only cases in the study (one NSB pair, two NSR edits on image 4), so the per-case CSVs *are*
the figure cases; there is no larger population to compare against. The two NSR rows (enlarge / reduce) agree with
each other on every ordering above.

## 6. Honest summary

| Removed | Visible damage in the figure? | Measurable? |
|---|---|---|
| NSB stage 1 (mixing) — B1 | **No** | No — inside the seed noise floor |
| NSB stage 3 (preservation) — B2 | Yes — flat, waxy nose and over-smoothed surrounding skin; identity score drops | Yes (ArcFace, pixel diff) |
| NSB stage-3 nose term — B3 | Barely (slight loss of the transferred bridge) | Yes but small (same noise stream, Δ = 0.8 px) |
| NSR segmentation term — N1 | Weakly — edit is about half as strong | Yes (nose area) |
| NSR landmark term — N2 | **No** | Yes on landmark error only (2 → 4–5 px); visually nil |
| NSR nose-style term — N3 | Barely (bridge shading) | Yes, small (nose LPIPS) |
| NSR non-nose (identity) term — N4 | **Yes, severe** — face changes entirely | Yes, large |

## 7. Deviations from `ablation-command.md`

1. **Cases**: one base image (`4`) instead of 30 per part, per the author's instruction; NSB = 5 → 4 only.
   Figure grids are therefore 1 × 6 and 2 × 6 rather than 3 × 6. "1st / 15th / 30th case" does not apply.
   (A reverse pair and image-5 NSR edits were run first by mistake and deleted.)
2. **Targets** were generated (none stored in the repo); procedure in §4. The mask change is dilate/erode-by-count
   (as in the paper's evaluation script) rather than a geometric 20 % scale.
3. **Metrics** reuse the Research repo's `Ttest` methods (ArcFace cosine, masked LPIPS, RA LPIPS) plus two
   target-attainment numbers (nose area ratio from BiSeNet, landmark L1 from FAN) that use the repo's own models.
   `nose_area_ratio` scaling and the ArcFace `det_thresh` were the only tweaks: insightface at its default threshold
   0.5 does not detect image 4 at all (eyes shut, wide grin); 0.3 detects it (score 0.37). The metric itself is unchanged.
   `summary.csv` lists the noise-floor repeats as separate "configs" next to B0–B3 / N0–N4.
4. **Noise floor**: extra B0/N0 runs with seeds 7 and 99 and a same-seed repeat were added so that "indistinguishable"
   claims are made against measured run-to-run variation, not against a single number.
5. **Determinism** could only be enabled warn-only (§3).
6. The ablation switch is an added optional request field rather than a CLI flag (the repo is API-driven and the author
   asked for API triggers). The backend was restarted afterwards so the process-global deterministic setting is cleared.
