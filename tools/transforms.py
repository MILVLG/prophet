# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Preprocessing images to be fed into the model, the script is
#              adapted from the code of CLIP (github.com/openai/CLIP)
# ------------------------------------------------------------------------------ #

from math import ceil
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import ImageOps

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def Pad():
    def _pad(image):
        W, H = image.size # debugged
        if H < W:
            pad_H = ceil((W - H) / 2)
            pad_W = 0
        else:
            pad_H = 0
            pad_W = ceil((H - W) / 2)
        img = ImageOps.expand(image, border=(pad_W, pad_H, pad_W, pad_H), fill=0)
        # print(img.size)
        return img
    return _pad

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def identity(x):
    return x

def _transform(n_px, pad=False, crop=False):
    return Compose([
        Pad() if pad else identity,
        Resize([n_px, n_px], interpolation=BICUBIC),
        CenterCrop(n_px) if crop else identity,
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


if __name__ == '__main__':
    img = np.random.rand(100, 333, 3).astype('uint8')
    img = Image.fromarray(img)
    img = _transform(32 * 14)(img)
    img = torch.Tensor(img)
    print(img.size())
