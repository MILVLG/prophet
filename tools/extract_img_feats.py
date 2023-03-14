# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Tool for extracting image features
# ------------------------------------------------------------------------------ #

import os, sys
sys.path.append(os.getcwd())

import glob, re, math, time, datetime
import numpy as np
import torch
from torch import nn
from PIL import Image
import clip
from tqdm import tqdm
import argparse
from pathlib import Path

from configs.task_cfgs import Cfgs
from configs.task_to_split import *
from tools.transforms import _transform


@torch.no_grad()
def _extract_feat(img_path, net, T, save_path):
    # print(img_path)
    img = Image.open(img_path)
    # W, H = img.size
    img = T(img).unsqueeze(0).cuda()
    clip_feats = net(img).cpu().numpy()[0]
    clip_feats = clip_feats.transpose(1, 2, 0)
    # print(clip_feats.shape, save_path)
    # return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        x=clip_feats,
    )


class ExtractModel:
    def __init__(self, encoder) -> None:
        encoder.attnpool = nn.Identity()
        self.backbone = encoder

        self.backbone.cuda().eval()
    
    @torch.no_grad()
    def __call__(self, img):
        x = self.backbone(img)
        return x


def main(__C, dataset):
    # find imgs
    img_dir_list = []
    for split in SPLIT_TO_IMGS:
        if split.startswith(dataset):
            img_dir_list.append(
                __C.IMAGE_DIR[SPLIT_TO_IMGS[split]]
            )
    print('image dirs:', img_dir_list)
    img_path_list = []
    for img_dir in img_dir_list:
        img_path_list += glob.glob(img_dir + '*.jpg')
    print('total images:', len(img_path_list))

    # load model
    clip_model, _ = clip.load(__C.CLIP_VERSION, device='cpu')
    img_encoder = clip_model.visual

    model = ExtractModel(img_encoder)
    T = _transform(__C.IMG_RESOLUTION)

    for img_path in tqdm(img_path_list):
        img_path_sep = img_path.split('/')
        img_path_sep[-3] += '_feats'
        save_path = '/'.join(img_path_sep).replace('.jpg', '.npz')
        _extract_feat(img_path, model, T, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tool for extracting CLIP image features.')
    parser.add_argument('--dataset', dest='dataset', help='dataset name, e.g., ok, aok', type=str, required=True)
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default='0')
    parser.add_argument('--clip_model', dest='CLIP_VERSION', help='clip model name or local model checkpoint path', type=str, default='RN50x64')
    parser.add_argument('--img_resolution', dest='IMG_RESOLUTION', help='image resolution', type=int, default=512)
    args = parser.parse_args()
    __C = Cfgs(args)
    main(__C, args.dataset)