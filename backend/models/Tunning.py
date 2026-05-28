import csv
import os
import shutil
import torch
import numpy as np
from argparse import Namespace
from torch import nn
from tqdm import tqdm
from pathlib import Path
from losses import lpips
from models.Net import Net
from pydantic import BaseModel
from typing import Optional, List
from utils.bicubic import BicubicDownSample
from utils.data_utils import load_FS_latent
from utils.model_utils import download_weight
from models.face_parsing.model import BiSeNet
from file_process import increment_inversion_step, set_inversion_image
from torch.nn import functional as torch_functions
from utils.fan import extract_landmarks_from_tensor
from models.StructureTransfer import StructureTransfer
from models.face_parsing.model import seg_mean, seg_std
from utils.helpers import save_as_image, get_face_segmentation_region, NOSE_REGION, SKIN_REGION, nose_idxs, extract_and_align_noses, compare_LPIPS, verbose, dilate_mask


BACKEND_DIR = Path(__file__).resolve().parent.parent
INVERTS_DIR = BACKEND_DIR / "inverts"
TESTS_DIR = BACKEND_DIR / "tests"
NOSE_SCALE_REDUCE = 0.8
NOSE_SCALE_ENLARGE = 1.2


def _write_loss_csv(path, history):
    if not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

class FineTuneContext(BaseModel):
    ref_path: str
    inversion_name: str
    nose_path: Optional[str] = None
    landmarks: Optional[List[List[float]]] = None
    segmentation: Optional[List[List[float]]] = None


class Tunning(nn.Module):
    def __init__(self, opts):
        super(Tunning, self).__init__()
        self.opts = opts
        self.net = Net(self.opts)
        self.load_downsampling()
        self.load_loss_functions()

    def load_downsampling(self):
        self.downsample = BicubicDownSample(factor=self.opts.size // 512)
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256)

    def load_loss_functions(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)
        if not os.path.exists(self.opts.seg_ckpt):
            download_weight(self.opts.seg_ckpt)
        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

        self.transferModel = StructureTransfer(self.opts)

        self.l1 = torch.nn.L1Loss()

        self.lpips = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
        self.lpips.eval()


    def save_result(self,F,S, ref_path):
            gen_im, _ = self.net.generator([S], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F)

            output_folder_path = os.path.join(Path(ref_path).parent, "tuned")

            os.makedirs(output_folder_path, exist_ok=True)

            FS_path = os.path.join(output_folder_path, "FS.npz")
            image_result_path = os.path.join(output_folder_path, f"FS.png")

            np.savez(
                FS_path,
                latent_in=S.detach().cpu().numpy(),
                latent_F=F.detach().cpu().numpy()
            )

            save_as_image(gen_im, image_result_path)

    def nose_transfer(self, item: FineTuneContext):
        S, F = self.transferModel.transfer(item.ref_path, item.nose_path, item.inversion_name)
        self.save_result(F, S, item.ref_path)
        


    def nose_tunning(self, item, csv_path=None, image_output_path=None):
        FS_path = os.path.join(Path(item.ref_path).parent, "FS.npz")
        S, F = load_FS_latent(FS_path, self.opts.device)

                
        ref_im_H, _ = self.net.generator([S], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F)

        ref_im_L = self.downsample_256(ref_im_H).detach().clone().to(self.opts.device)

        ref_im_H = ref_im_H.detach().clone().requires_grad_(False)
        
        S_ref = S.detach().clone().requires_grad_(False).to(self.opts.device)
        F_ref = F.detach().clone().requires_grad_(False).to(self.opts.device)

        S = S.detach().clone().requires_grad_(True).to(self.opts.device)
        F = F.detach().clone().requires_grad_(True).to(self.opts.device)

        ref_im_seg = get_face_segmentation_region(self, ref_im_H)

        nose_out_mask = torch.where(ref_im_seg == NOSE_REGION, 0, 1).float()

        if item.segmentation is not None:
            nose_out_mask = torch.where(item.segmentation == NOSE_REGION, 0, nose_out_mask).float()

        new_latent_optimizer = torch.optim.Adam([F, S], lr=self.opts.learning_rate)

        pbar = tqdm(range(self.opts.Tune_steps), desc="FS Inversion", leave=False)

        loss_history = []

        for step in pbar:
            new_latent_optimizer.zero_grad()

            loss = 0.0
            loss_dict = {}

            gen_H, _ = self.net.generator([S], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F)
            gen_L = self.downsample_256(gen_H)

            # Segmentation loss
            gen_im_0_1 = (gen_H + 1) / 2
            im = (self.downsample(gen_im_0_1) - seg_mean) / seg_std
            down_seg, _, _ = self.seg(im)
            ce_loss = torch_functions.cross_entropy(down_seg, item.segmentation) * self.opts.Tunning_segmentation_lambda
            loss_dict['Segmentation loss'] = ce_loss.item()
            loss += ce_loss

            # Landmarks loss
            lm_x = extract_landmarks_from_tensor(tensor_image = gen_H, device= self.opts.device)[nose_idxs]
            target_landmark_tensor = torch.tensor(item.landmarks, dtype=torch.float32, device= self.opts.device)[nose_idxs]
            # lm_mask = target_landmark_tensor[:, 3].unsqueeze(1)  # [N, 1] to broadcast correctly
            lm_y = target_landmark_tensor[:, :3]
            landmark_loss = self.l1(lm_x, lm_y) * self.opts.Tunning_landmarks_lambda
            # if lm_mask is None:
            #     landmark_loss = self.l1(lm_x, lm_y) * self.opts.Tunning_landmarks_lambda
            # else:
            #     landmark_loss = self.l1(lm_x * lm_mask, lm_y * lm_mask) * self.opts.Tunning_landmarks_lambda
            loss_dict['Landmarks loss'] = landmark_loss.item()
            loss += landmark_loss

            # compare styles
            gen_im_0_1 = (gen_H + 1) / 2
            im = (self.downsample(gen_im_0_1) - seg_mean) / seg_std
            gen_image_seg, _, _ = self.seg(im)
            gen_image_seg = torch.argmax(gen_image_seg, dim=1).long()
            gen_image_nose_mask = torch.where(gen_image_seg == NOSE_REGION, 1, 0).float()
            gen_image_nose_mask = torch_functions.interpolate(gen_image_nose_mask.unsqueeze(1).float(), size=1024, mode="bilinear")

            ref_im, _ = self.net.generator([S_ref], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F_ref)
            gen_im_0_1 = (ref_im + 1) / 2
            im = (self.downsample(gen_im_0_1) - seg_mean) / seg_std
            ref_image_seg, _, _ = self.seg(im)
            ref_image_seg = torch.argmax(ref_image_seg, dim=1).long()
            ref_image_nose_mask = torch.where(ref_image_seg == NOSE_REGION, 1, 0).float()
            ref_image_nose_mask = torch_functions.interpolate(ref_image_nose_mask.unsqueeze(1).float(), size=1024, mode="bilinear")

            aligned_ref , aligned_gen = extract_and_align_noses(ref_im, ref_image_nose_mask, gen_H, gen_image_nose_mask, to_grayscale=True, do_mask=True)
            style_loss = self.opts.Tunning_nose_perceptual_lambda * self.lpips(aligned_gen, aligned_ref).sum()
            loss_dict['Nose Style loss'] = style_loss.item()
            loss += style_loss

            # Face style loss
            face_style =  compare_LPIPS(self, ref_im_L, gen_L, nose_out_mask)
            loss_dict['Face Style loss'] = face_style.item()
            loss += face_style

            if csv_path is not None:
                loss_history.append({"iter": step, **loss_dict, "total": loss.item()})

            verbose(self, "Tunning:", loss_dict, loss, pbar)
            increment_inversion_step(item.inversion_name)
            loss.backward()
            new_latent_optimizer.step()

        if csv_path is not None:
            _write_loss_csv(csv_path, loss_history)

        if image_output_path is not None:
            with torch.no_grad():
                final_im, _ = self.net.generator([S], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F)
            Path(image_output_path).parent.mkdir(parents=True, exist_ok=True)
            save_as_image(final_im, str(image_output_path))
        else:
            self.save_result(F, S, item.ref_path)

    # ---- Evaluation helpers (temporary; replace tune_image to trigger sweep) ----

    def _compute_segmentation(self, image):
        """Run BiSeNet on a generated image and return a label map of shape [1, H, W]."""
        im_01 = (image + 1) / 2
        im = (self.downsample(im_01) - seg_mean) / seg_std
        logits, _, _ = self.seg(im)
        return torch.argmax(logits, dim=1).long()

    def _scale_nose_landmarks(self, landmarks, scale):
        """Scale nose landmarks (x, y) around their centroid by `scale`."""
        new_lm = landmarks.clone()
        nose_pts = new_lm[nose_idxs, :2]
        centroid = nose_pts.mean(dim=0, keepdim=True)
        new_lm[nose_idxs, :2] = centroid + (nose_pts - centroid) * scale
        return new_lm

    def _modify_nose_seg(self, seg_labels, scale):
        """Erode the nose region when scale<1, dilate it when scale>1.

        Kernel size is proportional to the smaller side of the nose bbox so
        the visual change targets the requested percentage.
        """
        nose_mask = (seg_labels == NOSE_REGION).float().unsqueeze(0)  # [1,1,H,W]

        ys, xs = torch.where(nose_mask[0, 0] > 0)
        if ys.numel() == 0:
            return seg_labels.clone()

        nose_h = int(ys.max() - ys.min() + 1)
        nose_w = int(xs.max() - xs.min() + 1)
        delta = abs(1.0 - scale)
        k = max(3, int(min(nose_h, nose_w) * delta))
        if k % 2 == 0:
            k += 1

        new_seg = seg_labels.clone()

        if scale < 1.0:
            eroded = 1.0 - dilate_mask(1.0 - nose_mask, kernel_size=k)
            removed = ((nose_mask > 0.5) & (eroded < 0.5))[0, 0]
            new_seg[0][removed] = SKIN_REGION
        else:
            dilated = dilate_mask(nose_mask, kernel_size=k)
            added = ((dilated > 0.5) & (nose_mask < 0.5))[0, 0]
            new_seg[0][added] = NOSE_REGION

        return new_seg

    def _evaluate_image(self, dataset_name, image_dir):
        image_name = image_dir.name

        # Copy into images/output so the model can find FS.npz the usual way
        dst_folder = Path(self.opts.output_dir) / f"{dataset_name}__{image_name}"
        if dst_folder.exists():
            shutil.rmtree(dst_folder)
        shutil.copytree(image_dir, dst_folder)

        fs_path = dst_folder / "FS.npz"
        if not fs_path.is_file():
            print(f"[SKIP] no FS.npz in {image_dir}")
            return

        ref_path = str(dst_folder / "FS.png")

        # Generate the canonical image and compute landmarks + segmentation once
        S, F = load_FS_latent(str(fs_path), self.opts.device)
        with torch.no_grad():
            gen_im, _ = self.net.generator([S], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F)
        landmarks = extract_landmarks_from_tensor(tensor_image=gen_im, device=self.opts.device).detach()
        with torch.no_grad():
            seg_labels = self._compute_segmentation(gen_im)

        for variant, scale in (("reduce", NOSE_SCALE_REDUCE), ("enlarge", NOSE_SCALE_ENLARGE)):
            modified_lm = self._scale_nose_landmarks(landmarks, scale)
            modified_seg = self._modify_nose_seg(seg_labels, scale)

            item = Namespace(
                ref_path=ref_path,
                inversion_name=f"{dataset_name}__{image_name}__{variant}",
                landmarks=modified_lm.detach().cpu().tolist(),
                segmentation=modified_seg,  # already a CUDA long tensor [1,H,W]
            )

            image_out = TESTS_DIR / dataset_name / "images" / variant / f"{image_name}.png"
            csv_out = TESTS_DIR / dataset_name / "csv" / variant / f"{image_name}.csv"

            # Register the inversion entry in process.json so increment_inversion_step works
            set_inversion_image(item.inversion_name, {"current_step": 0, "total_steps": self.opts.Tune_steps})

            print(f"  [{variant:7s}] {image_name}")
            self.nose_tunning(item, csv_path=csv_out, image_output_path=image_out)

    def evaluate_datasets(self):
        if not INVERTS_DIR.is_dir():
            print(f"[MISS] inverts dir not found: {INVERTS_DIR}")
            return

        for dataset_dir in sorted(INVERTS_DIR.iterdir()):
            if not dataset_dir.is_dir():
                continue
            print(f"\n=== Dataset: {dataset_dir.name} ===")
            for image_dir in sorted(dataset_dir.iterdir()):
                if image_dir.is_dir():
                    self._evaluate_image(dataset_dir.name, image_dir)

    def tune_image(self, item: FineTuneContext):
        # TEMPORARY: API trigger runs the full evaluation sweep instead of
        # the normal tune/transfer flow. Revert this method to its previous
        # body when evaluation is complete.
        self.evaluate_datasets()

