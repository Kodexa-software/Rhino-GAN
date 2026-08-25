import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from losses import lpips
from models.Net import Net
from utils.video import InversionVideo
from utils.bicubic import BicubicDownSample
from datasets.image_dataset import ImagesDataset
from file_process import increment_inversion_step
from utils.helpers import verbose, compare_LPIPS, save_as_image

class Embedding(nn.Module):
    def __init__(self, opts):
        super(Embedding, self).__init__()
        self.opts = opts
        self.net = Net(self.opts)
        self.video = InversionVideo(self.opts)
        self.load_downsampling()
        self.setup_loss_functions()

    def load_downsampling(self):
        factor = self.opts.size // 256
        self.downsample = BicubicDownSample(factor=factor)


    def setup_loss_functions(self):
        self.l2 = nn.MSELoss()
        self.lpips = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
        self.lpips.eval()
        
    def invert_image(self, image_name=str):
        img_path = os.path.join(self.opts.input_dir, f"{image_name}.png")

        ref_H, ref_L, _ = ImagesDataset(opts=self.opts, image_path=img_path)[0]

        ref_H = ref_H.unsqueeze(0).to(self.opts.device)
        ref_L = ref_L.unsqueeze(0).to(self.opts.device)

        self.video.open(self.opts.W_steps + self.opts.FS_steps)

        try:
            W_plus =  self.invert_images_in_W(image_name, ref_H, ref_L)

            # black gap that separates the W+ phase from the FS phase
            self.video.add_black()

            gen_im, F, S = self.invert_images_in_FS(image_name, ref_H, ref_L, W_plus)
        finally:
            self.video.close()

        output_folder = os.path.join(self.opts.output_dir, image_name)

        os.makedirs(output_folder, exist_ok=True)

        FS_path = os.path.join(output_folder, "FS.npz")
        image_result_path = os.path.join(output_folder, f"{image_name}.png")

        np.savez(
            FS_path,
            latent_in=S.detach().cpu().numpy(),
            latent_F=F.detach().cpu().numpy()
        )

        save_as_image(gen_im, image_result_path)

    def invert_images_in_W(self, image_name, ref_H, ref_L):
        pbar = tqdm(range(self.opts.W_steps), desc="W+ Inversion", leave=False)

        latent = []

        if (self.opts.tile_latent):
            tmp = self.net.latent_avg.clone().detach().cuda()
            tmp.requires_grad = True
            for i in range(self.net.layer_num):
                latent.append(tmp)
            optimizer_W = torch.optim.Adam([tmp], lr=self.opts.learning_rate)
        else:
            for i in range(self.net.layer_num):
                tmp = self.net.latent_avg.clone().detach().cuda()
                tmp.requires_grad = True
                latent.append(tmp)
            optimizer_W = torch.optim.Adam(latent, lr=self.opts.learning_rate)

        
        for step in pbar:
            optimizer_W.zero_grad()
            latent_in = torch.stack(latent).clone().unsqueeze(0)

            loss = 0.0
            loss_dict = {}

            img_H, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False)
            img_L = self.downsample(img_H)

            # L2 loss
            l2_loss = self.l2(img_H, ref_H) * self.opts.l2_lambda
            loss_dict["L2"] = l2_loss.item()
            loss += l2_loss

            # Perpetual loss
            perpetual_loss = compare_LPIPS(self, ref_L, img_L)
            loss_dict["Perpetual"] = perpetual_loss.item()
            loss += perpetual_loss

            # P-norm loss
            p_norm_loss = self.net.cal_p_norm_loss(latent_in)
            loss_dict["P-norm"] = p_norm_loss.item()
            loss += p_norm_loss

            loss.backward()
            optimizer_W.step()
            verbose(self, "W+ inversion", loss_dict, loss, pbar)
            self.video.add(img_H)
            increment_inversion_step(image_name)

        return latent_in.detach().clone()

    def invert_images_in_FS(self, image_name, ref_H, ref_L, W_plus):
        pbar = tqdm(range(self.opts.FS_steps), desc="FS Inversion", leave=False)

        F_init, _ = self.net.generator([W_plus], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)

        latent_F = F_init.clone().detach().requires_grad_(True)
        latent_S = []

        for i in range(self.net.layer_num):
            tmp = W_plus[0, i].clone()
            if i < self.net.S_index:
                tmp.requires_grad = False
            else:
                tmp.requires_grad = True

            latent_S.append(tmp)

        optimizer_FS = torch.optim.Adam(latent_S[self.net.S_index:] + [latent_F], lr=self.opts.learning_rate)

        for step in pbar:
            optimizer_FS.zero_grad()
            latent_in = torch.stack(latent_S).unsqueeze(0)

            loss = 0.0
            loss_dict = {}

            img_H , _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=latent_F)
            img_L = self.downsample(img_H)

            # L2 loss
            l2_loss = self.l2(img_H, ref_H) * self.opts.l2_lambda
            loss_dict["L2"] = l2_loss.item()
            loss += l2_loss

            # Perpetual loss
            perpetual_loss = compare_LPIPS(self, ref_L, img_L)
            loss_dict["Perpetual"] = perpetual_loss.item()
            loss += perpetual_loss

            # Perpetual loss
            F_structure_loss = self.net.cal_l_F(latent_F, F_init)
            loss_dict["F-structure"] = F_structure_loss.item()
            loss += F_structure_loss

            loss.backward()
            optimizer_FS.step()
            verbose(self, "FS inversion", loss_dict, loss, pbar)
            self.video.add(img_H)
            increment_inversion_step(image_name)
        
        return img_H.detach().clone(), latent_F, latent_in

