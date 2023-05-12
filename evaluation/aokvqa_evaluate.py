# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Evaluation script for A-OKVQA
# ------------------------------------------------------------------------------ #

import json
from evaluation.aok_utils.eval_predictions import eval_aokvqa
from evaluation.aok_utils.remap_predictions import map_to_choices
from .ans_punct import prep_ans
import argparse

class AOKEvaluater:
    def __init__(self, annotation_path: str, question_path: str):
        self.annotation_path = annotation_path
        self.question_path = question_path
        self.dataset = json.load(open(question_path, 'r'))
        self.result_file = {}
        self.result_path = None
        self.multiple_choice = False
        self.map_to_mc = True
    
    def init(self):
        self.result_file = []
    
    def set_mode(self, multiple_choice=None, map_to_mc=None):
        if multiple_choice is not None:
            self.multiple_choice = multiple_choice
        if map_to_mc is not None:
            self.map_to_mc = map_to_mc
    
    def prep_ans(self, answer):
        return prep_ans(answer)
    
    def add(self, qid, answer):
        if self.multiple_choice:
            self.result_file[qid] = {
                'multiple_choice': answer,
            }
        else:
            self.result_file[qid] = {
                'direct_answer': answer,
            }
    
    def save(self, result_path: str):
        self.result_path = result_path
        if not self.multiple_choice and self.map_to_mc:
            predictions = {qid: item['direct_answer'] for qid, item in self.result_file.items()}
            predictions = map_to_choices(self.dataset, predictions, 'cuda:0')
            for qid, answer in predictions.items():
                self.result_file[qid]['multiple_choice'] = answer
        json.dump(self.result_file, open(self.result_path, 'w'))
    
    def evaluate(self, logfile=None):
        assert self.result_path is not None, "Please save the result file first."

        direct_answer = not self.multiple_choice
        multiple_choice = self.multiple_choice or self.map_to_mc
        eval_str = _evaluate(self.dataset, self.result_file, direct_answer=direct_answer, multiple_choice=multiple_choice)
        print(eval_str)
        if logfile is not None:
            print(eval_str + '\n', file=logfile)


def _evaluate(dataset, results, direct_answer=True, multiple_choice=True):
    result_str = ''

    if direct_answer:
        # Direct Answer Evaluation
        da_predictions = {}
        for qid, item in results.items():
            da_predictions[qid] = item['direct_answer']

        da_acc = eval_aokvqa(
            dataset,
            da_predictions,
            multiple_choice=False,
            strict=False
        )
        result_str += f'DA: {da_acc: .2f}\n'
        
    if multiple_choice:
        # Multiple Choice Evaluation
        mc_predictions = {}
        for qid, item in results.items():
            mc_predictions[qid] = item['multiple_choice']

        mc_acc = eval_aokvqa(
            dataset,
            mc_predictions,
            multiple_choice=True,
            strict=False
        )
        result_str += f'MC: {mc_acc: .2f}\n'
    return result_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate A-OKVQA result file.')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--direct_answer', action='store_true')
    parser.add_argument('--multiple_choice', action='store_true')
    args = parser.parse_args()
    dataset = json.load(open(args.dataset_path, 'r'))
    result = json.load(open(args.result_path, 'r'))
    result_str = _evaluate(dataset, result, direct_answer=args.direct_answer, multiple_choice=args.multiple_choice)
    print(result_str)