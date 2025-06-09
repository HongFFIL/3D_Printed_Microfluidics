# utils/enhancement.py

import torch
import numpy as np

class Enhancement:
    def __init__(self, imgsz=(576, 768), stack_size=20, mean_intensity=105, scale_factor=2):
        self.BK_COUNT = 0
        self.BK_NUM = stack_size
        self.SCALE_FACTOR = scale_factor
        self.MEAN_INTENSITY = mean_intensity

        self.imgsz = imgsz

        self.BK_STACK = torch.zeros(
            [self.BK_NUM, self.imgsz[1], self.imgsz[0]],
            dtype=torch.float32,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def moving_window_enhance(self, image, to_cpu=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(image, np.ndarray):
            image = torch.tensor(image, device=device, dtype=torch.float32)

        if self.BK_COUNT < self.BK_NUM:
            self.BK_STACK[self.BK_COUNT, :, :] = image
            self.BK_COUNT += 1
            enh_gpu = torch.zeros((self.imgsz[1], self.imgsz[0]), device=device)
        else:
            self.BK_STACK = torch.roll(self.BK_STACK, -1, 0)
            self.BK_STACK[-1, :, :] = image

            enh_gpu = torch.abs(
                (self.BK_STACK[-1, :, :] - torch.mean(self.BK_STACK, axis=0))
                * self.SCALE_FACTOR
                + self.MEAN_INTENSITY
            )

        if to_cpu:
            enh_gpu = enh_gpu.cpu().numpy()
            enh_gpu = enh_gpu.astype(np.uint8)

        return enh_gpu
