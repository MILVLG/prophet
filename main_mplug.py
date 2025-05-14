import argparse
import yaml
import torch,deepspeed

from evaluation.okvqa_evaluate import OKEvaluater
from evaluation.aokvqa_evaluate import AOKEvaluater
from evaluation.textvqa_evaluate import TEXTEvaluater
from evaluation.sciencevqa_evaluate import SCIEvaluater
from configs.task_cfgs import Cfgs
from prophet import  get_runner


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', dest='TASK', help="task name, one of ['ok', 'aok_val', 'aok_test','text_val','text_test','science']", type=str, required=True)
    parser.add_argument('--run_mode', dest='RUN_MODE', help="run mode, one of ['pretrain', 'finetune', 'finetune_test', 'heuristics', 'prompt','finetune_mplug_test','finetune_mplug','heuristics_mplug']", type=str, required=True)
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
    parser.add_argument('--example_num', dest='EXAMPLE_NUM', help='number of most similar examples to be searched, default: 200', type=int, default=100)
    parser.add_argument('--examples_path', dest='EXAMPLES_PATH', help='answer-aware example file path, default: "assets/answer_aware_examples_for_ok.json"', type=str, default=None)
    parser.add_argument('--candidates_path', dest='CANDIDATES_PATH', help='candidates file path, default: "assets/candidates_for_ok.json"', type=str, default=None)
    parser.add_argument('--captions_path', dest='CAPTIONS_PATH', help='captions file path, default: "assets/captions_for_ok.json"', type=str, default=None)
    # parser.add_argument('--openai_key', dest='OPENAI_KEY', help='openai api key', type=str, default=None)
    #new mplug argument
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument('--max_length', default=20, type=int) #10
    parser.add_argument('--beam_size', default=20, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--gpu_nums', dest='GPU_NUMS', help='gpu nums', type=int, default=1)
    parser.add_argument('--mplug', dest='MPLUG', help='whether use mplug model', action='store_true')
    parser.add_argument('--cot_path', dest='CoT_PATH', help='openai api key', type=str, default=None)
    parser.add_argument('--mc_path', dest='MC_PATH', help='for aokvqa and scienceqa task', required=False,type=str, default='',nargs='?',)
    parser.add_argument('--ocr_path', dest='OCR_PATH', help='only for textvqa', type=str, required=False,default='',nargs='?',)
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args



def convert_to_floats(d):
    for key, value in d.items():
        if isinstance(value, str) and 'e' in value:
            try:
                d[key] = float(value)
            except ValueError:
                pass
        elif isinstance(value, dict):
            convert_to_floats(value)

# parse cfgs and args
args = get_args()
__C = Cfgs(args)
with open(args.cfg_file, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
convert_to_floats(yaml_dict)
__C.override_from_dict(yaml_dict)
print(__C)

# build runner
if __C.DATA_TAG == 'aok':
    evaluater = AOKEvaluater(
        __C.EVAL_ANSWER_PATH,
        __C.EVAL_QUESTION_PATH,
    )
elif __C.DATA_TAG == 'ok':
    evaluater = OKEvaluater(
        __C.EVAL_ANSWER_PATH,
        __C.EVAL_QUESTION_PATH,
    )
elif __C.DATA_TAG == 'text':
    evaluater = TEXTEvaluater(
        __C.ANSWER_PATH[__C.DATA_TAG]
    )
elif __C.DATA_TAG == 'science':
    evaluater = SCIEvaluater(
        __C.ANSWER_PATH[__C.DATA_TAG]
    )

runner = get_runner(__C, evaluater)
# run
runner.run()
