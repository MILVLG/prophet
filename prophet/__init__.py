__author__ = 'Zhenwei Shao'
__version__ = '1.0'

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', dest='TASK', help="task name, one of ['ok', 'aok_val', 'aok_test']", type=str, required=True)
    parser.add_argument('--run_mode', dest='RUN_MODE', help="run mode, one of ['pretrain', 'finetune', 'finetune_test', 'heuristics', 'prompt']", type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='config file', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name, output folder will be named as version name', type=str, required=True)
    parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for test', type=str, default=None)
    parser.add_argument('--pretrained_model', dest='PRETRAINED_MODEL_PATH', help='pretrained model path', type=str, default=None)
    parser.add_argument('--debug', dest='DEBUG', help='debug mode', action='store_true')
    parser.add_argument('--resume', dest='RESUME', help='resume previous run', action='store_true')
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default=None)
    parser.add_argument('--grad_accu', dest='GRAD_ACCU_STEPS', help='random seed', type=int, default=None)
    parser.add_argument('--seed', dest='SEED', help='random seed', type=int, default=99)
    parser.add_argument('--candidate_num', dest='CANDIDATE_NUM', help='topk candidates', type=int, default=None)
    parser.add_argument('--example_num', dest='EXAMPLE_NUM', help='number of most similar examples to be searched, default: 200', type=int, default=None)
    parser.add_argument('--examples_path', dest='EXAMPLES_PATH', help='answer-aware example file path, default: "assets/answer_aware_examples_for_ok.json"', type=str, default=None)
    parser.add_argument('--candidates_path', dest='CANDIDATES_PATH', help='candidates file path, default: "assets/candidates_for_ok.json"', type=str, default=None)
    parser.add_argument('--captions_path', dest='CAPTIONS_PATH', help='captions file path, default: "assets/captions_for_ok.json"', type=str, default=None)
    parser.add_argument('--openai_key', dest='OPENAI_KEY', help='openai api key', type=str, default=None)
    args = parser.parse_args()
    return args



def get_runner(__C, evaluater):
    if __C.RUN_MODE == 'pretrain':
        from .stage1.pretrain import Runner
    elif __C.RUN_MODE == 'finetune':
        from .stage1.finetune import Runner
    elif __C.RUN_MODE == 'finetune_test':
        from .stage1.finetune import Runner
    elif __C.RUN_MODE == 'heuristics':
        from .stage1.heuristics import Runner
    elif __C.RUN_MODE == 'prompt':
        from .stage2.prompt import Runner
    else:
        raise NotImplementedError
    runner = Runner(__C, evaluater)
    return runner