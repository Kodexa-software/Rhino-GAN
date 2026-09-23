import os
import torch
import numpy as np
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
from file_process import increment_inversion_step
from torch.nn import functional as torch_functions
from utils.fan import extract_landmarks_from_tensor
from models.StructureTransfer import StructureTransfer
from models.face_parsing.model import seg_mean, seg_std
from utils.helpers import save_as_image, get_face_segmentation_region, NOSE_REGION, nose_idxs, extract_and_align_noses, compare_LPIPS, verbose

class AblationConfig(BaseModel):
    # Ablation-study override (only used when explicitly sent; default behaviour is untouched)
    name: str                                 # output goes to <image>/tuned/ablation/<name>/
    drop: List[str] = []                      # NSB: mixing | preservation | nose_loss ; NSR: segmentation | landmarks | nose_style | face_style
    seed: Optional[int] = None                # torch/numpy/random seed applied before the run
    deterministic: bool = False               # torch.use_deterministic_algorithms(True, warn_only=True)

    def dropped(self, key: str) -> bool:
        return key in self.drop

    def out_subdir(self) -> str:
        return os.path.join("tuned", "ablation", self.name)


def ablation_dropped(ablation: Optional[AblationConfig], key: str) -> bool:
    return ablation is not None and ablation.dropped(key)


class FineTuneContext(BaseModel):
    ref_path: str
    inversion_name: str
    nose_path: Optional[str] = None
    landmarks: Optional[List[List[float]]] = None
    segmentation: Optional[List[List[float]]] = None
    ablation: Optional[AblationConfig] = None


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


    def save_result(self,F,S, ref_path, subdir="tuned"):
            gen_im, _ = self.net.generator([S], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F)

            output_folder_path = os.path.join(Path(ref_path).parent, subdir)

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
        S, F = self.transferModel.transfer(item.ref_path, item.nose_path, item.inversion_name, ablation=item.ablation)
        self.save_result(F, S, item.ref_path, subdir=item.ablation.out_subdir() if item.ablation else "tuned")
        


    def nose_tunning(self, item: FineTuneContext):
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

        for step in pbar:
            new_latent_optimizer.zero_grad()

            loss = 0.0
            loss_dict = {}

            gen_H, _ = self.net.generator([S], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F)
            gen_L = self.downsample_256(gen_H)

            # Segmentation loss
            if not ablation_dropped(item.ablation, "segmentation"):
                gen_im_0_1 = (gen_H + 1) / 2
                im = (self.downsample(gen_im_0_1) - seg_mean) / seg_std
                down_seg, _, _ = self.seg(im)
                ce_loss = torch_functions.cross_entropy(down_seg, item.segmentation) * self.opts.Tunning_segmentation_lambda
                loss_dict['Segmentation loss'] = ce_loss.item()
                loss += ce_loss

            # Landmarks loss
            if not ablation_dropped(item.ablation, "landmarks"):
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
            if not ablation_dropped(item.ablation, "nose_style"):
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
            if not ablation_dropped(item.ablation, "face_style"):
                face_style =  compare_LPIPS(self, ref_im_L, gen_L, nose_out_mask)
                loss_dict['Face Style loss'] = face_style.item()
                loss += face_style

            verbose(self, "Tunning:", loss_dict, loss, pbar)
            increment_inversion_step(item.inversion_name)
            loss.backward()
            new_latent_optimizer.step()

        self.save_result(F, S, item.ref_path, subdir=item.ablation.out_subdir() if item.ablation else "tuned")

    def apply_ablation_seed(self, ablation: Optional[AblationConfig]):
        if ablation is None:
            return
        if ablation.seed is not None:
            import random
            random.seed(ablation.seed)
            np.random.seed(ablation.seed)
            torch.manual_seed(ablation.seed)
            torch.cuda.manual_seed_all(ablation.seed)
        if ablation.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

    def tune_image(self, item: FineTuneContext):
        self.apply_ablation_seed(item.ablation)
        if item.nose_path == "self" or item.ref_path == item.nose_path or item.nose_path is None:
            if item.segmentation is not None:
                item.segmentation = torch.tensor(item.segmentation, dtype=torch.long, device= self.opts.device).unsqueeze(0)
            self.nose_tunning(item)
        else:
            self.nose_transfer(item)

