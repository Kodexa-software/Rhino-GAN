import torch
import torchvision
import numpy as np
from torch.nn import functional as F
from models.face_parsing.model import seg_mean, seg_std

toPIL = torchvision.transforms.ToPILImage()

HAIR_REGION = 10
NOSE_REGION = 2
SKIN_REGION = 1
IGNORE_CODE = 255
nose_idxs = torch.arange(27, 36, device="cuda")

COLOR_MAP = np.array([
    [0, 0, 0],         # 0: background - black
    [255, 0, 0],       # 1: red
    [0, 255, 0],       # 2: green
    [0, 0, 255],       # 3: blue
    [255, 255, 0],     # 4: yellow
    [255, 0, 255],     # 5: magenta
    [0, 255, 255],     # 6: cyan
    [255, 128, 0],     # 7: orange
    [128, 0, 255],     # 8: violet
    [0, 128, 255],     # 9: sky blue
    [128, 255, 0],     # 10: lime
    [255, 0, 128],     # 11: pink-red
    [0, 255, 128],     # 12: aqua green
    [128, 128, 0],     # 13: olive
    [0, 128, 128],     # 14: teal
    [192, 192, 192],   # 15: light gray
], dtype=np.uint8)


def get_face_segmentation_region(ctx, im_tensor):
    im = ((im_tensor[0] + 1) / 2).detach().cpu().clamp(0, 1)
    im = (ctx.downsample(im_tensor).clamp(0, 1) - seg_mean) / seg_std
    seg, _, _ = ctx.seg(im)
    seg = torch.argmax(seg, dim=1).long()
    return seg

def verbose(ctx, title, dict, loss, pbar):
    if ctx.opts.verbose:
        desc_items = [f"{k}: {v:.3f}" for k, v in dict.items()]
        pbar.set_description(f"{title}: {loss:.3f}, " + ", ".join(desc_items))


def compare_LPIPS(ctx, ref_L, gen_L, foreign_mask= None):
    if foreign_mask is not None :
        mask = F.interpolate(foreign_mask.unsqueeze(1), size=(256, 256), mode='nearest')
        lpips_loss = ctx.lpips(gen_L * mask, ref_L * mask).sum() * ctx.opts.percept_lambda
    else:
        lpips_loss = ctx.lpips(gen_L, ref_L).sum() * ctx.opts.percept_lambda

    return lpips_loss


def save_as_image(latent, path):
    save_im = toPIL(((latent[0] + 1) / 2).detach().cpu().clamp(0, 1))
    save_im.save(path)

def dilate_mask(mask, kernel_size=15):
    if mask.dtype != torch.float32:
        mask = mask.float()

    if not mask.is_cuda:
        mask = mask.cuda()

    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated > 0.5).float()   # keep it binary 0/1

def to_grayscale_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return image
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

def crop_to_bbox(image, mask, target_size, to_grayscale=True, do_mask=False):
    if to_grayscale and image.shape[1] == 3:
        image = to_grayscale_tensor(image)

    if do_mask:
        image = image * mask
    
    mask_bin = mask[0,0] > 0
    ys, xs = torch.where(mask_bin)
    if ys.numel() == 0 or xs.numel() == 0:
        raise ValueError("Mask is empty!")

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    cropped = image[:, :, y1:y2, x1:x2]
    cropped_mask = mask[:, :, y1:y2, x1:x2]

    cropped = F.interpolate(cropped, size=(target_size, target_size), mode="nearest")
    cropped_mask = F.interpolate(cropped_mask, size=(target_size, target_size), mode="nearest")
    
    return cropped, cropped_mask


def extract_and_align_noses(ref_image, ref_mask, gen_image, gen_mask, target_size=512, to_grayscale=True, do_mask=False):
    
    ref_mask = dilate_mask(ref_mask.to("cuda"), kernel_size=15)
    gen_mask = dilate_mask(gen_mask.to("cuda"), kernel_size=15)
    
    ref_crop, ref_cropped_mask = crop_to_bbox(ref_image, ref_mask, target_size, to_grayscale, do_mask)
    gen_crop, gen_cropped_mask = crop_to_bbox(gen_image, gen_mask, target_size, to_grayscale, do_mask)

    if do_mask:
        aligned_ref, aligned_gen = row_align_gen_to_ref(ref_crop, ref_cropped_mask, gen_crop, gen_cropped_mask)
        return aligned_ref, aligned_gen
    
    return ref_crop, gen_crop

def row_align_gen_to_ref(aligned_ref, aligned_ref_mask, aligned_gen, aligned_gen_mask):
    device = aligned_ref.device
    B, C, H, W = aligned_ref.shape
    assert B == 1, "Batch size must be 1"

    ref_m = (aligned_ref_mask[0, 0] > 0.5)
    gen_m = (aligned_gen_mask[0, 0] > 0.5)

    ref_has = ref_m.any(dim=1)
    gen_has = gen_m.any(dim=1)
    both_has = ref_has & gen_has

    ref_first = ref_m.float().argmax(dim=1)
    ref_last = (W - 1) - ref_m.flip(1).float().argmax(dim=1)
    gen_first = gen_m.float().argmax(dim=1)
    gen_last = (W - 1) - gen_m.flip(1).float().argmax(dim=1)

    neg1 = torch.full_like(ref_first, -1)
    ref_first = torch.where(ref_has, ref_first, neg1)
    ref_last = torch.where(ref_has, ref_last, neg1)
    gen_first = torch.where(gen_has, gen_first, neg1)
    gen_last = torch.where(gen_has, gen_last, neg1)

    ref_w = (ref_last - ref_first + 1).clamp(min=1)
    gen_w = (gen_last - gen_first + 1).clamp(min=1)

    cols = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
    rows = torch.arange(H, device=device)
    region = (cols >= ref_first.unsqueeze(1)) & (cols <= ref_last.unsqueeze(1)) & both_has.unsqueeze(1)

    denom = (ref_w - 1).clamp(min=1)
    t = (cols - ref_first.unsqueeze(1)).clamp(min=0).float() / denom.unsqueeze(1).float()
    t = t.clamp(0, 1)

    gen_span = (gen_w - 1).float()
    x_in = gen_first.unsqueeze(1).float() + t * gen_span.unsqueeze(1)

    x_in = torch.where(region, x_in, torch.full_like(x_in, -10.0))

    x_norm = ((x_in + 0.5) / W) * 2 - 1
    y_vals = ((rows.float() + 0.5) / H) * 2 - 1
    y_norm = y_vals.unsqueeze(1).expand(H, W)
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)

    aligned_new = F.grid_sample(
        aligned_gen, grid, mode='nearest',
        padding_mode='zeros', align_corners=False
    )

    return aligned_ref, aligned_new
