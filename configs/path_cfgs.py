# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: set const paths and dirs
# ------------------------------------------------------------------------------ #

import os

class PATH:
    def __init__(self):

        self.LOG_ROOT = 'outputs/logs/'
        self.CKPT_ROOT = 'outputs/ckpts/'
        self.RESULTS_ROOT = 'outputs/results/'
        self.DATASET_ROOT = 'datasets/'
        self.ASSETS_ROOT = 'assets/'


        self.IMAGE_DIR = {
            'train2014': self.DATASET_ROOT + 'coco2014/train2014/',
            'val2014': self.DATASET_ROOT + 'coco2014/val2014/',
            # 'test2015': self.DATASET_ROOT + 'coco2015/test2015/',
            'train2017': self.DATASET_ROOT + 'coco2017/train2017/',
            'val2017': self.DATASET_ROOT + 'coco2017/val2017/',
            'test2017': self.DATASET_ROOT + 'coco2017/test2017/',
        }

        self.FEATS_DIR = {
            'train2014': self.DATASET_ROOT + 'coco2014_feats/train2014/',
            'val2014': self.DATASET_ROOT + 'coco2014_feats/val2014/',
            'train2017': self.DATASET_ROOT + 'coco2017_feats/train2017/',
            'val2017': self.DATASET_ROOT + 'coco2017_feats/val2017/',
            'test2017': self.DATASET_ROOT + 'coco2017_feats/test2017/',
        }

        self.QUESTION_PATH = {
            'v2train': self.DATASET_ROOT + 'vqav2/v2_OpenEnded_mscoco_train2014_questions.json',
            'v2val': self.DATASET_ROOT + 'vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
            'vg': self.DATASET_ROOT + 'vqav2/vg_questions.json',
            'v2valvg_no_ok': self.DATASET_ROOT + 'vqav2/v2valvg_no_ok_questions.json',
            'oktrain': self.DATASET_ROOT + 'okvqa/OpenEnded_mscoco_train2014_questions.json',
            'oktest': self.DATASET_ROOT + 'okvqa/OpenEnded_mscoco_val2014_questions.json',
            'aoktrain': self.DATASET_ROOT + 'aokvqa/aokvqa_v1p0_train.json',
            'aokval': self.DATASET_ROOT + 'aokvqa/aokvqa_v1p0_val.json',
            'aoktest': self.DATASET_ROOT + 'aokvqa/aokvqa_v1p0_test.json',
        }

        self.ANSWER_PATH = {
            'v2train': self.DATASET_ROOT + 'vqav2/v2_mscoco_train2014_annotations.json',
            'v2val': self.DATASET_ROOT + 'vqav2/v2_mscoco_val2014_annotations.json',
            'vg': self.DATASET_ROOT + 'vqav2/vg_annotations.json',
            'v2valvg_no_ok': self.DATASET_ROOT + 'vqav2/v2valvg_no_ok_annotations.json',
            'oktrain': self.DATASET_ROOT + 'okvqa/mscoco_train2014_annotations.json',
            'oktest': self.DATASET_ROOT + 'okvqa/mscoco_val2014_annotations.json',
            'aoktrain': self.DATASET_ROOT + 'aokvqa/aokvqa_v1p0_train.json',
            'aokval': self.DATASET_ROOT + 'aokvqa/aokvqa_v1p0_val.json',
        }

        self.ANSWER_DICT_PATH = {
            'v2': self.ASSETS_ROOT + 'answer_dict_vqav2.json',
            'ok': self.ASSETS_ROOT + 'answer_dict_okvqa.json',
            'aok': self.ASSETS_ROOT + 'answer_dict_aokvqa.json',
        }


