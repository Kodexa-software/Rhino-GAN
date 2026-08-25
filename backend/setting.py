import os
from argparse import Namespace

size = 1024
cached_dir_name = 'cache'
root_dir_name = '/nose-ai'

# master switch for the inversion recording (root/k-inversion.mp4) — set to False to skip it
capture_video = False

input_dir_name ='/nose-ai/backend/images/inputs'
output_dir_name ='/nose-ai/backend/images/output'
unprocessed_dir_name ='/nose-ai/backend/images/unprocessed'
predictor_url = "https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx"

for d in [input_dir_name, output_dir_name, unprocessed_dir_name]:
    os.makedirs(d, exist_ok=True)

setting = Namespace(
    capture_video=capture_video,
    video_path=os.path.join(root_dir_name, 'k-inversion.mp4'),
    video_size=512,
    video_fps=30,
    video_seconds=18,
    video_gap_seconds=2,
    FS_path = "",
    root_dir=root_dir_name,
    input_dir=input_dir_name,
    output_dir=output_dir_name,
    unprocessed_dir=unprocessed_dir_name,
    landmark_lambda=0.1,
    sign='realistic',
    smooth=5,
    size=size,
    ckpt="/nose-ai/backend/pretrained_models/ffhq.pt",
    channel_multiplier=2,
    latent=512,
    n_mlp=8,
    device='cuda',
    seed=None,
    tile_latent=False,
    opt_name='adam',
    learning_rate=0.01,
    lr_schedule='fixed',
    save_intermediate=False,
    save_interval=300,
    verbose=True,
    seg_ckpt='/nose-ai/backend/pretrained_models/seg.pth',
    percept_lambda=1.0,
    l2_lambda=1.0,
    p_norm_lambda=0.001,
    l_F_lambda=0.1,
    ce_lambda=1.0,
    style_lambda=4e4,
    align_steps1=140,
    align_steps2=100,
    face_lambda=1.0,
    hair_lambda=1.0,
    blend_steps=400,
    W_steps=1100,
    FS_steps=250,
    Tune_steps=50,
    Tunning_segmentation_lambda=0.3,
    Tunning_landmarks_lambda=0.01,
    Tunning_nose_perceptual_lambda=0.1,
    Transfer_restructure_steps=10,
    Transfer_restructure_perceptual_lambda=0.1,
    Transfer_perceptual_steps=40,
    Transfer_perceptual_nose_lambda=0.1,
    Transfer_perceptual_face_lambda=1.0, 

)