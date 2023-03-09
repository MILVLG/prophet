# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: the definition of the improved MCAN
# ------------------------------------------------------------------------------ #

import torch
from torch import nn
from torch.nn import functional as F
import math
from transformers import AutoModel, logging
logging.set_verbosity_error()

from .net_utils import *
from .layers import *


class MCA_ED(nn.Module):
    """
    The definition of the encoder-decoder backbone of MCAN.
    """
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        enc = __C.ARCH_CEIL['enc'] * __C.LAYER
        dec = __C.ARCH_CEIL['dec'] * __C.LAYER
        self.enc_list = nn.ModuleList([eval(layer)(__C) for layer in enc])
        self.dec_list = nn.ModuleList([eval(layer)(__C) for layer in dec])

    def forward(self, x, y, x_mask, y_mask):
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y



class MCAN(nn.Module):
    """
    The definition of the complete network of the improved MCAN, mainly includes:
    1. A pretrained BERT model used to encode questions (already represented as tokens)
    2. A linear layer to project CLIP vision features (extracted beforehand, so the CLIP
        model is not included) to a common embedding space
    3. An encoder-decoder backbone to fuse question and image features in depth
    4. A classifier head based on `AttFlat`
    """
    def __init__(self, __C, answer_size):
        super().__init__()

        # answer_size = trainset.ans_size

        self.__C = __C

        self.bert = AutoModel.from_pretrained(__C.BERT_VERSION)

        # self.clip_visual = trainset.clip_model.visual
        # self.clip_visual.layer4 = Identity()
        # self.clip_visual.float()

        # for p in self.clip_visual.parameters():
        #     p.requires_grad = False

        self.img_feat_linear = nn.Sequential(
            nn.Linear(__C.IMG_FEAT_SIZE, __C.HIDDEN_SIZE, bias=False),
        )
        self.lang_adapt = nn.Sequential(
            nn.Linear(__C.LANG_FEAT_SIZE, __C.HIDDEN_SIZE),
            nn.Tanh(),
        )

        self.backbone = MCA_ED(__C)
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.proj_norm = nn.LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def forward(self, input_tuple, output_answer_latent=False):
        img_feat, ques_ix = input_tuple

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = None#self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.bert(
            ques_ix, 
            attention_mask= ~lang_feat_mask.squeeze(1).squeeze(1)
        )[0]
        lang_feat = self.lang_adapt(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)


        # Backbone Framework
        # img_feat = flatten(img_feat)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )
        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = lang_feat + img_feat
        answer_latent = self.proj_norm(proj_feat)
        proj_feat = self.proj(answer_latent)

        if output_answer_latent:
            return proj_feat, answer_latent

        return proj_feat

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
