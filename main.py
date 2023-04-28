import argparse
import yaml
import torch

from evaluation.okvqa_evaluate import OKEvaluater
from evaluation.aokvqa_evaluate import AOKEvaluater
from configs.task_cfgs import Cfgs
from prophet import get_args, get_runner

# parse cfgs and args
args = get_args()
__C = Cfgs(args)
with open(args.cfg_file, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
__C.override_from_dict(yaml_dict)
print(__C)

# build runner
if __C.RUN_MODE == 'pretrain':
    evaluater = None
elif 'aok' in __C.TASK:
    evaluater = AOKEvaluater(
        __C.EVAL_ANSWER_PATH,
        __C.EVAL_QUESTION_PATH,
    )
else:
    evaluater = OKEvaluater(
        __C.EVAL_ANSWER_PATH,
        __C.EVAL_QUESTION_PATH,
    )

runner = get_runner(__C, evaluater)

# run
runner.run()
