import os
import cv2
import numpy as np
from loguru import logger
from torch.nn import functional as F


class InversionVideo:
    """Records the inversion process into a single, always-overwritten mp4.

    Frames come in as generator output tensors (1, 3, H, W) in [-1, 1].
    W+ and FS phases are written into the same file, separated by a black gap.
    """

    def __init__(self, opts):
        self.opts = opts
        self.writer = None
        self.frame_index = 0
        self.total_steps = 0
        self.target_frames = 0

    @property
    def enabled(self):
        return bool(getattr(self.opts, "capture_video", False))

    def open(self, total_steps=0):
        """`total_steps` is how many `add` calls the whole run will make; frames are
        sampled evenly out of them so the recording lasts `video_seconds`."""
        if not self.enabled:
            self.close()
            return

        self.close()
        self.frame_index = 0
        self.total_steps = int(total_steps)
        self.target_frames = int(round(self.opts.video_seconds * self.opts.video_fps))

        path = self.opts.video_path
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        size = self.opts.video_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.opts.video_fps, (size, size))

        if not writer.isOpened():
            logger.warning(f"could not open video writer for {path}, skipping recording")
            return

        self.writer = writer
        logger.info(f"recording inversion video to {path}")

    def add(self, latent):
        if self.writer is None:
            return

        index = self.frame_index
        self.frame_index += 1

        if not self.keeps_frame(index):
            return

        self.writer.write(self.to_frame(latent))

    def keeps_frame(self, index):
        """Spread `target_frames` picks evenly over `total_steps` optimization steps,
        so each phase keeps its share of the recording. Keeps everything when the run
        is shorter than the target."""
        if not self.total_steps or self.target_frames >= self.total_steps:
            return True

        target, total = self.target_frames, self.total_steps

        return ((index + 1) * target) // total > (index * target) // total

    def add_black(self, seconds=None):
        if self.writer is None:
            return

        if seconds is None:
            seconds = self.opts.video_gap_seconds

        size = self.opts.video_size
        black = np.zeros((size, size, 3), dtype=np.uint8)

        for _ in range(int(round(seconds * self.opts.video_fps))):
            self.writer.write(black)

    def to_frame(self, latent):
        size = self.opts.video_size
        im = latent[:1].detach().float()

        if im.shape[-1] != size or im.shape[-2] != size:
            im = F.interpolate(im, size=(size, size), mode="area")

        im = ((im[0] + 1) / 2).clamp(0, 1)
        im = (im.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        return np.ascontiguousarray(im[:, :, ::-1])  # RGB -> BGR

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
