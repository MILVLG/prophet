# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Evaluation script for OK-VQA
# ------------------------------------------------------------------------------ #

import json
from evaluation.vqa_utils.vqa import VQA
from evaluation.vqa_utils.vqaEval import VQAEval
from .ans_punct import prep_ans
import argparse

class TEXTEvaluater:
    def __init__(self, label_pth:str ):
        self.label_pth = label_pth
        self.result_file = []
        self.result_path = None
    
    def prep_ans(self, answer):
        return prep_ans(answer)
    
    def add(self, qid, answer):
        qid = int(qid)
        self.result_file.append({
            'question_id': qid,
            'answer': answer
        })
    
    def save(self, result_path: str):
        self.result_path = result_path
        json.dump(self.result_file, open(self.result_path, 'w'))
    
    def evaluate(self, logfile=None):
        assert self.result_path is not None, "Please save the result file first."

        eval_str = _evaluate(self.label_pth, self.result_path)
        print()
        print(eval_str)
        if logfile is not None:
            print(eval_str + '\n', file=logfile)


def _evaluate(label_pth: str, result_file: str):
    # print(f'== Annotation file: {annotation_file}')
    # print(f'== Question file: {question_file}')
    with open(label_pth, "r") as f:
        data_list = json.load(f)
    with open(result_file, "r") as f:
        ans_list = json.load(f)
    
    id2datum = {}
    for each in data_list:
        id2datum[each["question_id"]] = each["label"]
    score = 0.
    for each in ans_list:
        quesid = each["question_id"]
        ans = each["answer"]
        label = id2datum[quesid]
        if ans in label:
            score += label[ans]
    score=score / len(ans_list)
    print('score',score)

    result_str = ''
    result_str += "Overall Accuracy is: %.02f\n" % (score)
    result_str += "{'This is a reference accuracy, if you want to get more accurate accuracy you can submit the results to the textvqa challenge\n"
    return result_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate OK-VQA result file.')
    parser.add_argument('--annotation_path', type=str, required=True)
    parser.add_argument('--question_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    args = parser.parse_args()
    result_str = _evaluate(args.annotation_path, args.question_path, args.result_path)
    print(result_str)