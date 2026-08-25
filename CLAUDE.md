# CLAUDE.md — RhinoGAN

Full project context for Claude Code. Read this instead of re-exploring the codebase.

> **MAINTENANCE RULE (always follow):** This file must stay 100% accurate. Whenever you change anything in this repo — or a hook/session notice reports files modified after CLAUDE.md — update the affected sections of this file in the same turn (endpoints, settings values, components, commands, flow). If you verify a reported change requires no doc edit, still make sure this file's content matches the code before finishing.

## What this project is

**RhinoGAN** (GitHub: kodo-yousif/RhinoGAN) is a surgeon-guided nasal editing system built on the **FS latent space of StyleGAN2**. It supports:

- **NSB — Nose Structure transfer**: transfer a nose from another face onto the patient's face
- **NSR — Nose Shape Refinement**: refine the nose using user-edited segmentation masks and draggable landmarks

All edits preserve facial identity. Built for clinical visualization (surgeon evaluation ~7.5/10). Demo: https://www.youtube.com/watch?v=NBWAY4dVwSM — Contact: kodo.yousif@gmail.com

## Repo layout

```
backend/     Python FastAPI + PyTorch server (port 3001) — all ML logic
frontend/    Vite + React 18 + TS + antd 5 + Tailwind + zustand (dev port 3000)
environment/environment.yml   conda env "nose-ai" (Python 3.7, torch 1.13.1)
Dockerfile   pytorch 2.0.1-cuda11.7 base + CUDA toolkit 11.7 + Miniconda + Node 22 (nvm)
images/      README assets only
process.json Runtime job-status state (also backend/process.json); reset to {} on backend start
alignment-phase.md/.svg/.png  Flowchart (mermaid + rendered image) + worked numeric example of the upload-time FFHQ face-alignment phase
k-inversion.mp4  20 s recording of the latest inversion (18 s of W+ then FS, 2 s black gap between); overwritten by every inversion, git-ignored
alignment-phase-simple.svg/.png  Seminar slide: the alignment phase as 4 simple illustrated steps (find / measure / frame / straighten)
```

No tests, no CI, no lint hooks in the repo.

## Build & run (everything runs inside Docker)

```bash
docker build -t rhinogan-image .
docker run -d --gpus all --privileged -it -p 8000:3000 -p 8001:3001 -v .:/nose-ai --name rhinogan-container rhinogan-image
docker exec -it rhinogan-container bash
# terminal 1:
cd frontend && yarn && yarn dev        # vite, host 0.0.0.0, strictPort 3000
# terminal 2:
cd backend && python main.py           # uvicorn on 0.0.0.0:3001
```

- Host access: frontend on **8000**, API on **8001** (frontend hardcodes `${protocol}//${hostname}:8001` as axios baseURL in `src/main.tsx`).
- `backend/setting.py` uses absolute container paths (`/nose-ai/backend/...`) — the backend will **not** run directly on the Windows host.
- Frontend scripts: `yarn dev`, `yarn build` (tsc && vite build), `yarn lint`, `yarn language` (syncs locale JSONs via `language-adjusting.js`).

## Pretrained models

| File | How it gets there |
|---|---|
| `backend/pretrained_models/ffhq.pt` | StyleGAN2 FFHQ — **manual download** (Google Drive link in README / note.txt) |
| `backend/pretrained_models/seg.pth` | BiSeNet face parsing — auto-download via gdown (`utils/model_utils.py` `weight_dic`) |
| `backend/cache/*shape_predictor_68*.dat` | dlib landmark predictor — auto-download via `utils/drive.open_url` |
| `ffhq_PCA.npz` (next to ckpt) | Built on first `Net()` init (1M-sample PCA of W space) — slow one-time step |
| `backend/fail-safe-models/` | Local backup of all weights; `backend/failsafe_models.py` copies them there (restore source if upstream 404s) |

---

# Backend (`backend/`)

FastAPI app in `main.py` (uvicorn `0.0.0.0:3001`, CORS `*`, loguru → `backend/app.log`).

## Config — `setting.py`

Single `Namespace` called `setting`, imported everywhere as `opts`. Key values:

- `size=1024`, `device='cuda'`, `learning_rate=0.01` (Adam), `ckpt=/nose-ai/backend/pretrained_models/ffhq.pt`, `seg_ckpt=.../seg.pth`
- Step counts: `W_steps=1100`, `FS_steps=250`, `Tune_steps=50`, `Transfer_restructure_steps=10`, `Transfer_perceptual_steps=40`
- Lambdas: `percept=1.0`, `l2=1.0`, `p_norm=0.001`, `l_F=0.1`, `Tunning_segmentation=0.3`, `Tunning_landmarks=0.01`, `Tunning_nose_perceptual=0.1`, `Transfer_restructure_perceptual=0.1`, `Transfer_perceptual_nose=0.1`, `Transfer_perceptual_face=1.0`
- Dirs (created on import): `images/inputs` (aligned faces), `images/output` (results), `images/unprocessed` (raw uploads); `root_dir=/nose-ai`
- Inversion video: `capture_video=True` — the on/off boolean, declared as a module-level constant at the top of `setting.py`; flip it to `False` to skip recording entirely. Companion values: `video_path=/nose-ai/k-inversion.mp4`, `video_size=512`, `video_fps=30`, `video_seconds=18` (frame content), `video_gap_seconds=2` → 20 s total

## HTTP endpoints (`main.py`)

- `POST /upload-image` — multipart png/jpeg → save to unprocessed → `face_processor.process_face` (dlib align → 1024×1024 PNG in inputs) → queue inversion job. Returns `"Started"` / `"Queued"`
- `POST /fine-tune` — body `{fullPath, noseStyle, landmarks?, segmentation?}`; **landmarks/segmentation arrive as JSON strings**. `noseStyle == 'self'` or `== fullPath` → tuning job named `Tunning-<stem>`; otherwise transfer job `Transfer-<refstem><nosestem>`. Paths are relative to `output_dir`
- `GET /get-image?im_name=` — raw image bytes from output_dir
- `GET /image-data?im_name=` — base64 image + BiSeNet 512×512 label matrix (+ `COLOR_MAP`, region ids nose=2 / skin=1) + 68 FAN 3D landmarks
- `GET /get-images` — recursive list of all `.png` under output_dir (relative paths)
- `GET /processes` — contents of process.json (job progress)
- `GET /`, `GET /health`

## Job system

Module-level in `main.py`: `maxWorkers=1`, `currentWorkers`, `imageQueue` (`queue.Queue`). Jobs run on plain `threading.Thread`; when one finishes it recursively pops the next queue item. **Caution:** inversion and fine-tune jobs share the same queue, but their workers expect different item types (str vs `FineTuneContext`) — a mixed queue can hand the wrong type to a worker.

Progress lives in `process.json` via `file_process.py` (`set_inversion_image`, `increment_inversion_step`, `error_field` sets `current_step=-1`; thread-locked; file reset to `{}` at import). Keys are `<name>_inversion`.

## ML pipeline (`models/`)

- **`Net.py`** — wraps StyleGAN2 `Generator` (`models/stylegan2/model.py`, supports `start_layer`/`end_layer`/`layer_in` partial forward). Loads `ffhq.pt` `g_ema` + `latent_avg`, frozen/eval. `layer_num=18` @1024, `S_index=7` (layers ≥7 form the S half of FS). Holds the PCA model for p-norm regularization (`cal_p_norm_loss`, `cal_l_F`).
- **`Embedding.py`** — image inversion (II2S-style): `invert_images_in_W` (W+ optimization; L2 + LPIPS + p-norm) then `invert_images_in_FS` (optimize F = layer-3 feature map + S = W+ layers 7–17; L2 + LPIPS + F-structure). Saves `output/<name>/FS.npz` (`latent_in`=S, `latent_F`=F) + `<name>.png`. Also records both phases into `k-inversion.mp4` via `utils/video.InversionVideo` (opened in `invert_image`, closed in a `finally`).
- **`Tunning.py`** — `Tunning.tune_image(FineTuneContext)`; pydantic `FineTuneContext {ref_path, inversion_name, nose_path?, landmarks?, segmentation?}`. Self-tuning path `nose_tunning`: optimizes F+S with (1) cross-entropy seg loss vs user-edited 512×512 mask, (2) L1 landmark loss on nose landmarks (indices 27–35, `nose_idxs`), (3) LPIPS nose-style loss on aligned nose crops vs original, (4) masked LPIPS face-preservation loss outside the nose. Saves to `output/<name>/tuned/FS.npz` + `tuned/FS.png`.
- **`StructureTransfer.py`** — NSB: loads both images' FS.npz; `swap_F_noses` blends F maps inside dilated nose masks minus "red line" (non-skin/non-nose) forbidden regions; then two optimization phases — nose-style LPIPS on grayscale aligned crops (10 steps, F only), then nose/rest LPIPS preservation (40 steps, F+S). Called via `Tunning.nose_transfer`, same save path. Local seg constants: NOSE=2, SKIN=1, HAIR=10.
- **`face_parsing/`** — BiSeNet (16 classes), input 512, normalized by `seg_mean`/`seg_std` from `face_parsing/model.py`.
- **`stylegan2/op/`** — fused CUDA ops (`fused_act`, `upfirdn2d`) compiled at runtime; needs CUDA toolkit + ninja.

## Utils / losses

- `losses/lpips/` — LPIPS perceptual loss (net-lin vgg; weights bundled in repo under `losses/lpips/weights/`)
- `utils/helpers.py` — `COLOR_MAP` (16 RGB rows), region ids (`NOSE_REGION=2`, `SKIN_REGION=1`, `HAIR_REGION=10`), `nose_idxs = arange(27,36)`, `verbose` (tqdm desc), `compare_LPIPS` (optional mask), `save_as_image`, `dilate_mask` (max_pool2d), `extract_and_align_noses` / `crop_to_bbox` / `row_align_gen_to_ref` (nose crop alignment used by style losses)
- `utils/fan.py` — **differentiable** FAN landmarks: face_alignment 3D model, `soft_argmax_2d` over heatmaps → (68,3); z = heatmap max. Model instantiated at module import (cuda).
- `utils/shape_predictor.py` + `face_processor.py` — dlib face alignment; predictor cached in `backend/cache/`
- `utils/model_utils.py` — gdown `weight_dic` (ffhq.pt, seg.pth, afhq*, metfaces)
- `utils/video.py` — `InversionVideo`: cv2 `mp4v` writer for the inversion recording (`open(total_steps)`/`add`/`add_black`/`close`). `keeps_frame` samples `video_seconds × video_fps` frames evenly across `W_steps + FS_steps`, so each phase keeps its share (1100/250 → 440/100 frames) and the file always lands at `video_seconds`. Silently no-ops when `opts.capture_video` is `False` or the writer can't open
- `utils/bicubic.py` — `BicubicDownSample`; `utils/data_utils.py` — `load_FS_latent`; `datasets/image_dataset.py` — `ImagesDataset` (returns im_H 1024 + im_L 256, normalized to [-1,1])

## Backend gotchas

- Nearly everything assumes CUDA; several helpers hardcode `.cuda()` / `device="cuda"`.
- Heavy work happens at import time: `fan.py` builds FaceAlignment on import; `main.py` builds BiSeNet + `Embedding` + `Tunning` at startup.
- Existing naming/spelling ("Tunning", "Perpetual", "prepropess") is intentional legacy — match it, don't rename.
- `process.json` is written in the CWD (`backend/` when run normally).

---

# Frontend (`frontend/`)

Vite 5 + React 18 + TypeScript, antd 5 (dark/light via `useDarkMode` + ConfigProvider), Tailwind 3, zustand 4 (persisted), react-router-dom 6 (lazy routes), axios, i18next, react-compare-slider. Path alias `@/` → `src/` (vite-tsconfig-paths). Package name is "react-k-template" (template leftover).

## API wiring

`src/main.tsx` sets `axios.defaults.baseURL = ${protocol}//${hostname}:8001`. Several components additionally build raw `<img src>` URLs with the same `:8001/get-image?im_name=...` pattern (AppLayout, ResultViewer).

## Routing (`src/routes.tsx` + `src/lib/route.ts`)

Root (ErrorBoundary = `pages/Error`) → `/` `AppLayout` → index `Home`, `/video` `video-player`; `*` `NotFound`. `lazyPageBuilder` adapts dynamic imports to router lazy objects.

## State — `src/global/useImageNodes.ts` (the core store; zustand persist key `"ImageNodeStates"`)

- `nodes[2]` (`RawNode {key,title,fullPath,type}`) — index 0 = selected input image (arrays are 2-slot but only index 0 is really used)
- `noseStyle: string` — `"self"` or output-relative path of the style image
- `segmentationDatas[2]` — `{segmentedImage: number[][] (512×512 labels), segmentedImageRef (pristine copy), COLOR_MAP, regions:{nose,skin}}`
- `nodeLandmarksInfo[2]` — `{landmarks, referenceLandmarks}`, each Landmark = `[x,y,z]` in 1024-space
- `showSegmentation[]`, `showNodeLandmarks[]`, `modules {noseStyle, landmarks, segmentation, doStyle}`
- `drawSegmentationByIndex` / `undrawSegmentationByIndex` — circular brush (`paintArray`); draw sets nose label; undraw restores from `segmentedImageRef` (ref-nose cells become skin)
- `setNodeByIndex` resets seg data + landmarks and sets `noseStyle → "self"`
- `IMAGE_ROWS = IMAGE_COLS = 512`

## Key components

- **`containers/AppLayout.tsx`** — header: select-image modal + select-style modal (both list `GET /get-images` thumbnails); in self mode shows landmark/segmentation toggles, in style mode a **Transfer** button (`POST /fine-tune {fullPath, noseStyle}`); plus `ServerStatus`, `UploadImage`, refresh button. Content renders `<Outlet/>`.
- **`pages/Home.tsx`** — self mode: `ImageViewer` (editing) + `ResultViewer`; style mode: `ResultViewer(imagePath=selected)` + `ResultViewer` (result).
- **`containers/ImageViewer.tsx`** — fetches `/image-data`; shows base64 image, draggable nose landmarks (only indices 27–35 rendered; moved ones draw a dark reference dot + gradient line), `SegmentationCanvas` overlay. **Style** button posts `/fine-tune` with JSON-stringified segmentation + landmarks (each landmark gains a 4th flag: 1 if nose index or moved vs reference).
- **`components/SegmentationCanvas.tsx`** — 512×512 canvas overlay painting the nose diff (added nose = blue, removed = black, alpha 100); pointer events → draw/undraw with brush-size input + draw/undraw switch.
- **`components/ResultViewer.tsx`** — polls every 1 s for `output/<name>/tuned/FS.png` via `/get-image` (cache-busted with `_t=Date.now()`); compare-slider modal vs the original.
- **`components/ServerStatus.tsx`** — polls `/processes` every 3 s; progress bars from `current_step/total_steps`; `-1` renders red/exception.
- **`components/UploadImage.tsx`** — hidden file input → `POST /upload-image` multipart.
- `global/` also has `useDarkMode`, `useCollapse`, `useFilePath`, `useLoading`, `useUser` — template leftovers, mostly unused.

## i18n

i18next + http-backend + browser-languagedetector; locales in `frontend/public/locales/{en,ar,ckb}.json`; `t("dir")` returns rtl/ltr and is applied to antd ConfigProvider + a wrapper div. Fonts in `src/assets/fonts` (Speda for Kurdish, Roboto).

---

# End-to-end flow (mental model)

1. User uploads a photo → dlib aligns/crops to 1024×1024 → `images/inputs/<name>.png`
2. Inversion job (~1350 steps): W+ then FS optimization → `images/output/<name>/FS.npz` + `<name>.png`, plus `k-inversion.mp4` at the repo root (overwritten each run)
3. User selects the inverted image in the UI, then either:
   - **NSR (self)**: edits the 512×512 nose segmentation with a brush and/or drags nose landmarks (27–35) → **Style** → `/fine-tune` → `Tunning.nose_tunning` (50 steps)
   - **NSB (transfer)**: picks another inverted image as nose style → **Transfer** → `/fine-tune` → `StructureTransfer.transfer` (10 + 40 steps)
4. Result written to `images/output/<name>/tuned/FS.png`; frontend polls until it appears and offers a before/after compare slider.
