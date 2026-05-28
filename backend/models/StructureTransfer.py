import csv
import os
import torch
from datetime import datetime
from torch import nn
from tqdm import tqdm
from pathlib import Path
from losses import lpips
from models.Net import Net
from torch.nn import functional as F
from utils.bicubic import BicubicDownSample
from utils.data_utils import load_FS_latent
from models.face_parsing.model import BiSeNet
from utils.model_utils import download_weight
from file_process import increment_inversion_step
from models.face_parsing.model import seg_mean, seg_std
from utils.helpers import dilate_mask, extract_and_align_noses, to_grayscale_tensor, verbose


SAVE_TRANSFER_LOSS_CSV = True

TRANSFER_TESTS_DIR = Path(__file__).resolve().parent.parent / "transfer_tests"
MIXING_DIR = TRANSFER_TESTS_DIR / "mixing"
PRESERVATION_DIR = TRANSFER_TESTS_DIR / "preservation"


def _write_loss_csv(path, history):
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

def extract_bbox(mask):
    ys, xs = torch.where(mask[0,0] > 0)
    if ys.numel() == 0 or xs.numel() == 0:
        return None
    return ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()

def crop_region(F, bbox):
    y1, y2, x1, x2 = bbox
    return F[:, :, y1:y2, x1:x2]


NECK = 14
HAIR = 10
LOWER_LIP = 9 
UPPER_LIP = 8
MOUTH = 7
EYE_BROWS = 5
EYES = 4
NOSE = 2
SKIN = 1
BACKGROUND = 0

class StructureTransfer(nn.Module):
    def __init__(self, opts):
        super(StructureTransfer, self).__init__()  
        self.opts = opts
        self.net = Net(self.opts)
        self.load_downsampling()
        self.load_networks()

    def load_downsampling(self):
        self.downsample = BicubicDownSample(factor=self.opts.size // 512)
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256)

    def load_networks(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)
        self.lpips = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
        self.lpips.eval()

        if not os.path.exists(self.opts.seg_ckpt):
            download_weight(self.opts.seg_ckpt)
        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

    def getFullFSPath(self, file_path):
        path = Path(file_path)
        return f"{path.parent}/FS.npz"

    def get_im_seg(self, im):
        im_01 = (im + 1) / 2
        seg, _, _ = self.seg(self.downsample(im_01))
        return seg

    def gen_image(self, S, F):
        im, _ = self.net.generator([S], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F)
        return im

    def swap_F_noses(self, ref_seg, F1, nose_seg, F2):
        with torch.no_grad():
            old_seg = torch.argmax(ref_seg, dim=1) 
            old_nose = (old_seg == NOSE).float().unsqueeze(1) 
            old_nose = dilate_mask(old_nose, kernel_size=21)
            
            new_seg = torch.argmax(nose_seg, dim=1) 
            new_nose = (new_seg == NOSE).float().unsqueeze(1) 
            new_nose = dilate_mask(new_nose, kernel_size=21) 

            bbox1 = extract_bbox(old_nose)
            bbox2 = extract_bbox(new_nose)

            if bbox1 and bbox2:
                # Crop new_nose
                nose2_crop = crop_region(new_nose, bbox2)

                # Resize it to fit F1 nose dimensions
                target_h = bbox1[1] - bbox1[0]
                target_w = bbox1[3] - bbox1[2]
                nose2_resized = F.interpolate(nose2_crop, size=(target_h, target_w), mode="nearest")

                # Place into F1 nose box
                aligned_new_nose = torch.zeros_like(old_nose)
                aligned_new_nose[:, :, bbox1[0]:bbox1[1], bbox1[2]:bbox1[3]] = nose2_resized

                # Final nose mask
                nose_mask = torch.clamp(old_nose + aligned_new_nose, 0, 1)
            else:
                nose_mask = old_nose


            # forbiden touch areas 
            old_red_line = ((old_seg != SKIN) & (old_seg != NOSE)).float().unsqueeze(1)
            old_red_line = dilate_mask(old_red_line, kernel_size=19)

            new_red_line = ((new_seg != SKIN) & (new_seg != NOSE)).float().unsqueeze(1)
            new_red_line = dilate_mask(new_red_line, kernel_size=19)
            
            red_line = torch.clamp(old_red_line + new_red_line, 0, 1)


            changeable_area = (nose_mask - red_line).clamp(min=0)
            
            # save_colored_segmentation(changeable_area.squeeze(1).bool() , "test_seg.png")
            
            # nose_mask = torch.clamp(old_nose + new_nose, 0, 1)
            
            changeable_area = F.interpolate(changeable_area, size=F2.shape[-2:], mode="nearest")

        changeable_area_exp = changeable_area.expand(-1, F2.shape[1], -1, -1)

        F_mix = F1 * (1 - changeable_area_exp) + F2 * changeable_area_exp

        return F_mix

    def transfer(self, I1, I2, inversion_name):
        FS_1_path = self.getFullFSPath(I1)
        FS_2_path = self.getFullFSPath(I2)
        
        S1, F1 = load_FS_latent(FS_1_path, device=self.opts.device)
        S2, F2 = load_FS_latent(FS_2_path, device=self.opts.device)

        with torch.no_grad():
            orginal_image = self.gen_image(S1, F1)
            orginal_seg = self.get_im_seg(orginal_image)    

            target_im = self.gen_image(S2, F2)
            
            im_nose = self.gen_image(S1, F2)
            im_nose_seg = self.get_im_seg(im_nose)



        F_mix = self.swap_F_noses(orginal_seg.clone(), F1, im_nose_seg.clone(), F2)

        pbar = tqdm(range(self.opts.Transfer_restructure_steps), desc="Transfer Structure", leave=False)

        F_mix = F2.clone().detach().requires_grad_(True)

        new_latent_optimizer = torch.optim.Adam([F_mix], lr=self.opts.learning_rate)

        mixing_history = []

        for step in pbar:
            new_latent_optimizer.zero_grad()

            loss_dict = {}

            loss = 0.0

            target_im = target_im.requires_grad_(False)

            gen_im_0_1 = (target_im + 1) / 2
            im = (self.downsample(gen_im_0_1) - seg_mean) / seg_std
            style_seg, _, _ = self.seg(im)
            style_seg = torch.argmax(style_seg, dim=1).long()
            
            style_nose_mask = torch.where(style_seg == NOSE, 1, 0).float()
            style_nose_mask  = F.interpolate(style_nose_mask.unsqueeze(1).float(), size=1024, mode="nearest")

            target_im_gray = to_grayscale_tensor(target_im).detach().clone().requires_grad_(False)

            gen_im = self.gen_image(S1, F_mix)

            gen_im_0_1 = (gen_im + 1) / 2
            im = (self.downsample(gen_im_0_1) - seg_mean) / seg_std
            gen_style_seg, _, _ = self.seg(im)
            gen_style_seg = torch.argmax(gen_style_seg, dim=1).long()
            
            gen_style_nose_mask = torch.where(gen_style_seg == NOSE, 1, 0).float()
            gen_style_nose_mask  = F.interpolate(gen_style_nose_mask.unsqueeze(1).float(), size=1024, mode="nearest")


            aligned_ref , aligned_gen = extract_and_align_noses(target_im_gray, style_nose_mask, gen_im, gen_style_nose_mask, to_grayscale=True, do_mask=True)

            style_loss = self.lpips(aligned_gen, aligned_ref).sum()

            loss_dict['style'] = style_loss.item()

            loss += style_loss * self.opts.Transfer_restructure_perceptual_lambda

            if SAVE_TRANSFER_LOSS_CSV:
                mixing_history.append({"iter": step, **loss_dict, "total": loss.item()})

            increment_inversion_step(inversion_name)
            verbose(self, "Transferring", loss_dict, loss, pbar)

            loss.backward()
            new_latent_optimizer.step()

        if SAVE_TRANSFER_LOSS_CSV:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _write_loss_csv(MIXING_DIR / f"{inversion_name}_{ts}.csv", mixing_history)

        F_mix = self.swap_F_noses(orginal_seg.clone(), F1, self.get_im_seg(gen_im), F_mix)

        pbar = tqdm(range(self.opts.Transfer_perceptual_steps), desc="Preserve Structure", leave=False)

        mix_im_fixed = self.gen_image(S1, F_mix)

        # start fixing inversion    

        F_fixed = F_mix.clone().detach().requires_grad_(True)
        S_fixed = S1.clone().detach().requires_grad_(True)

        new_latent_optimizer = torch.optim.Adam([F_fixed, S_fixed], lr=self.opts.learning_rate)


        ref_image = orginal_image.detach().clone().requires_grad_(False)
        orginal_seg = self.get_im_seg(ref_image).requires_grad_(False)

        ref_seg = torch.argmax(orginal_seg, dim=1) 
        ref_nose = (ref_seg == NOSE).float().unsqueeze(1) 
        ref_nose = dilate_mask(ref_nose, kernel_size=3)


        nose_image = mix_im_fixed.detach().clone().requires_grad_(False)
        nose_seg = self.get_im_seg(nose_image).requires_grad_(False)
        
        nose_seg = torch.argmax(nose_seg, dim=1)
        nose_image_nose = (nose_seg == NOSE).float().unsqueeze(1)
        nose_image_nose = dilate_mask(nose_image_nose, kernel_size=3)


        nose_mask = torch.clamp(nose_image_nose + ref_nose, 0, 1)
        image_mask = 1.0 - nose_mask

        nose_mask = F.interpolate(nose_mask, size=256, mode="bilinear")
        image_mask = F.interpolate(image_mask, size=256, mode="bilinear")

        ref_image = self.downsample_256(ref_image)
        nose_image = self.downsample_256(nose_image)

        ref_image_target = ref_image * image_mask
        nose_image_target = nose_image * nose_mask

        preservation_history = []

        for step in pbar:
            new_latent_optimizer.zero_grad()

            loss_dict = {}
            
            loss = 0.0

            gen_im = self.gen_image(S_fixed, F_fixed)

            gen_im = self.downsample_256(gen_im)

            gen_image_nose = gen_im * nose_mask
            gen_image_rest = gen_im * image_mask

            nose_image_loss = self.lpips(gen_image_nose, nose_image_target).sum()
            loss_dict['nose'] = nose_image_loss.item()
            loss += nose_image_loss * self.opts.Transfer_perceptual_nose_lambda

            rest_image_loss = self.lpips(gen_image_rest, ref_image_target).sum()
            loss_dict['rest'] = rest_image_loss.item()
            loss += rest_image_loss * self.opts.Transfer_perceptual_face_lambda

            if SAVE_TRANSFER_LOSS_CSV:
                preservation_history.append({"iter": step, **loss_dict, "total": loss.item()})

            increment_inversion_step(inversion_name)
            verbose(self, "Preserving", loss_dict, loss, pbar)

            loss.backward()
            new_latent_optimizer.step()

        if SAVE_TRANSFER_LOSS_CSV:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _write_loss_csv(PRESERVATION_DIR / f"{inversion_name}_{ts}.csv", preservation_history)

        return S_fixed.detach().clone(), F_fixed.detach().clone()
