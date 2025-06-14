# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Object that manages the configuration of the experiments.
# ------------------------------------------------------------------------------ #

import os
import random
import torch
import numpy as np
from datetime import datetime

from .path_cfgs import PATH
from .task_to_split import *


class Cfgs(PATH):
    
    def __init__(self, args):
        super(Cfgs, self).__init__()
        self.set_silent_attr()

        self.GPU = getattr(args, 'GPU', None)
        if self.GPU is not None:
            self.GPU_IDS = [int(i) for i in self.GPU.split(',')]
            # print(f'Avaliable GPUs: {torch.cuda.device_count()}')
            # print(f'Using GPU {self.GPU}')
            self.CURRENT_GPU = self.GPU_IDS[0]
            torch.cuda.set_device(f'cuda:{self.CURRENT_GPU}')
            self.N_GPU = len(self.GPU_IDS)
            self.SEED = getattr(args, 'SEED', 1111)
            torch.manual_seed(self.SEED)
            # torch.manual_seed_all(self.SEED)
            if self.N_GPU < 2:
                torch.cuda.manual_seed(self.SEED)
            else:
                torch.cuda.manual_seed_all(self.SEED)
            torch.backends.cudnn.deterministic = True
            np.random.seed(self.SEED)
            random.seed(self.SEED)
            torch.set_num_threads(2)

        # -------------------------
        # ---- Version Control ----
        # -------------------------
        self.TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')
        self.VERSION = getattr(args, 'VERSION', self.TIMESTAMP)
        
        # paths and dirs
        self.CKPTS_DIR = os.path.join(self.CKPT_ROOT, self.VERSION)
        self.LOG_PATH = os.path.join(
            self.LOG_ROOT, 
            self.VERSION, 
            f'log_{self.TIMESTAMP}.txt'
        )
        self.RESULT_DIR = os.path.join(self.RESULTS_ROOT, self.VERSION)
        self.RESULT_PATH = os.path.join(
            self.RESULTS_ROOT,
            self.VERSION,
            'result_' + self.TIMESTAMP + '.json'
        )

        # about resume
        self.RESUME = getattr(args, 'RESUME', False)
        if self.RESUME and self.RUN_MODE == 'pretrain':
            self.RESUME_VERSION = getattr(args, 'RESUME_VERSION', self.VERSION)
            self.RESUME_EPOCH = getattr(args, 'RESUME_EPOCH', None)
            resume_path = getattr(args, 'RESUME_PATH', None)
            self.RESUME_PATH = os.path.join(
                self.CKPTS_DIR, 
                self.RESUME_VERSION, 
                f'epoch_{self.RESUME_EPOCH}.pkl'
            ) if resume_path is None else resume_path
        
        # for testing and heuristics generation
        self.CKPT_PATH = getattr(args, 'CKPT_PATH', None)

        # ----------------------
        # ---- Task Control ----
        # ----------------------

        self.TASK = getattr(args, 'TASK', 'ok')
        assert self.TASK in ['ok', 'aok_val', 'aok_test']

        self.RUN_MODE = getattr(args, 'RUN_MODE', 'finetune')
        assert self.RUN_MODE in ['pretrain', 'finetune', 'finetune_test', 'heuristics', 'prompt']

        if self.RUN_MODE == 'pretrain':
            self.DATA_TAG = 'v2'  # used to config answer dict
            self.DATA_MODE = 'pretrain'
        else:
            self.DATA_TAG = self.TASK.split('_')[0]  # used to config answer dict
            self.DATA_MODE = 'finetune'

        
        # config pipeline...
        self.EVAL_NOW = True
        if self.RUN_MODE == 'pretrain' or self.TASK == 'aok_test':
            self.EVAL_NOW = False
        # print(f'Eval Now: {self.EVAL_NOW}')

        # ------------------------
        # ---- Model Training ----
        # ------------------------

        self.NUM_WORKERS = 8
        self.PIN_MEM = True

        # --------------------------------
        # ---- Heuristics Generations ----
        # --------------------------------

        self.CANDIDATE_NUM = getattr(args, 'CANDIDATE_NUM', None)
        if self.CANDIDATE_NUM is not None:
            self.CANDIDATE_FILE_PATH = os.path.join(
                self.RESULTS_ROOT,
                self.VERSION,
                'candidates.json'
            )
            self.EXAMPLE_FILE_PATH = os.path.join(
                self.RESULTS_ROOT,
                self.VERSION,
                'examples.json'
            )
            self.ANSWER_LATENTS_DIR = os.path.join(
                self.RESULTS_ROOT,
                self.VERSION,
                'answer_latents'
            ) # where answer latents will be saved


        # write rest arguments to self
        for attr in args.__dict__:
            setattr(self, attr, getattr(args, attr))
    
    def __repr__(self):
        _str = ''
        for attr in self.__dict__:
            if attr in self.__silent or getattr(self, attr) is None:
                continue
            _str += '{ %-17s }-> %s\n' % (attr, getattr(self, attr))
        
        return _str
    
    def override_from_dict(self, dict_):
        for key, value in dict_.items():
            setattr(self, key, value)
    
    def set_silent_attr(self):
        self.__silent = []
        for attr in self.__dict__:
            self.__silent.append(attr)
        
    @property
    def TRAIN_SPLITS(self):
        return TASK_TO_SPLIT[self.TASK][self.DATA_MODE]['train_split']
    
    @property
    def EVAL_SPLITS(self):
        return TASK_TO_SPLIT[self.TASK][self.DATA_MODE]['eval_split']
        
    @property
    def FEATURE_SPLIT(self):
        FEATURE_SPLIT = []
        for split in self.TRAIN_SPLITS + self.EVAL_SPLITS:
            feat_split = SPLIT_TO_IMGS[split]
            if feat_split not in FEATURE_SPLIT:
                FEATURE_SPLIT.append(feat_split)
        return FEATURE_SPLIT
    
    @property
    def EVAL_QUESTION_PATH(self):
        # if not self.EVAL_NOW:
        #     return []
        return self.QUESTION_PATH[self.EVAL_SPLITS[0]]
    
    @property
    def EVAL_ANSWER_PATH(self):
        if not self.EVAL_NOW:
            return []
        return self.ANSWER_PATH[self.EVAL_SPLITS[0]]