# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: The goal of this file is to define the mapping from task and data
# mode to dataset splits.
# ------------------------------------------------------------------------------ #

class DictSafe(dict):

    def __init__(self, data={}):
        dict.__init__(self, data)
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = DictSafe(value)

    def __getitem__(self, key):
        return self.get(key, [])

# TASK_TO_SPLIT[TASK][DATA_MODE]['train_split'] is a list of dataset split name for training
# TASK_TO_SPLIT[TASK][DATA_MODE]['eval_split'] is a list of dataset split name for evaluation
# 'pretrain' mode is used for pretrain, so it does not have 'eval_split'
# 'finetune' mode is used for finetune, heuristics generation and prompting
TASK_TO_SPLIT = {
    'ok': {
        'pretrain': {
            'train_split': ['v2train', 'v2valvg_no_ok'],
            # As the testing set of okvqa uses a subset of MSCOCO val2014 as the input images,
            # we remove this subset from the training set of pretraining to avoid data leakage.
        },
        'finetune': {
            'train_split': ['oktrain'],
            'eval_split': ['oktest'],
        }
    },
    'aok_val': {
        'pretrain': {
            'train_split': ['v2train'],
        },
        'finetune': {
            'train_split': ['aoktrain'],
            'eval_split': ['aokval'],
        }
    },
    'aok_test': {
        'pretrain': {
            'train_split': ['v2train', 'v2val', 'vg'],
        },
        'finetune': {
            'train_split': ['aoktrain', 'aokval'],
            'eval_split': ['aoktest'],
        }
    },
}
TASK_TO_SPLIT = DictSafe(TASK_TO_SPLIT)

SPLIT_TO_IMGS = {
    'v2train': 'train2014',
    'v2val': 'val2014',
    'v2valvg_no_ok': 'val2014',
    'vg': 'val2014',
    'oktrain': 'train2014',
    'oktest': 'val2014',
    'aoktrain': 'train2017',
    'aokval': 'val2017',
    'aoktest': 'test2017',
}


if __name__ == '__main__':
    print(TASK_TO_SPLIT['okvqa']['test']['train_split'])