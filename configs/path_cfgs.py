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
        self.MPLUG_IMAGE_DIR = {
            'ok': self.DATASET_ROOT + 'coco2014',
            'aok_val': self.DATASET_ROOT,
            'aok_test': self.DATASET_ROOT,
            'science': self.DATASET_ROOT + '/science/images',
            'text_val': self.DATASET_ROOT + 'TextVQA/',
            'text_test': self.DATASET_ROOT + 'TextVQA/',
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
            'science': self.DATASET_ROOT + 'mplug/sicenceqa/test_label2.json',
            'text':self.DATASET_ROOT + 'mplug/textvqa/text_val_labels.json'
        }

        self.ANSWER_DICT_PATH = {
            'v2': self.ASSETS_ROOT + 'answer_dict_vqav2.json',
            'ok': self.ASSETS_ROOT + 'answer_dict_okvqa.json',
            'aok': self.ASSETS_ROOT + 'answer_dict_aokvqa.json',
        }

        
        self.MPLUG_TRAIN_PATH={
            'ok': [self.DATASET_ROOT + 'mplug/okvqa/vqa_train_ama_ocr.json'],
            'aok_val': [self.DATASET_ROOT + 'mplug/aokvqa/aok_train_ocr.json'],
            'aok_test': [self.DATASET_ROOT + 'mplug/aokvqa/aok_train_ocr.json', self.DATASET_ROOT + 'mplug/okvqa/aok_val_ocr.json'],
            'text_val': [self.DATASET_ROOT + 'mplug/textvqa/text_train.json',self.DATASET_ROOT + 'mplug/textvqa/ST_train_ocr.json',self.DATASET_ROOT + 'mplug/textvqa/ST_val_ocr.json'],
            'text_test': [self.DATASET_ROOT + 'mplug/textvqa/text_train.json',self.DATASET_ROOT + 'mplug/textvqa/ST_train_ocr.json',self.DATASET_ROOT + 'mplug/textvqa/text_val.json',self.DATASET_ROOT + 'mplug/textvqa/ST_val_ocr.json'],
            'science':[self.DATASET_ROOT + 'mplug/sicenceqa/train_ocr.json',self.DATASET_ROOT + 'mplug/sicenceqa/val_ocr.json'],
        }
        
        self.MPLUG_TEST_PATH={
            'ok': [self.DATASET_ROOT +'mplug/okvqa/vqa_val_ama_ocr.json'],
            'aok_val': [self.DATASET_ROOT + 'mplug/aokvqa/aok_val_ocr.json'],
            'aok_test': [self.DATASET_ROOT + 'mplug/aokvqa/aok_test_ocr.json'],
            'text_val': [self.DATASET_ROOT + 'mplug/textvqa/text_val.json'],
            'text_test': [self.DATASET_ROOT + 'mplug/textvqa/text_test.json'],
            'science': [self.DATASET_ROOT + 'mplug/sicenceqa/test_ocr.json'],
        }

