# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Evaluation script for OK-VQA
# ------------------------------------------------------------------------------ #

import json
from evaluation.vqa_utils.vqa import VQA
from evaluation.vqa_utils.vqaEval import VQAEval
from .ans_punct import prep_ans
import argparse

class OKEvaluater:
    def __init__(self, annotation_path: str, question_path: str):
        self.annotation_path = annotation_path
        self.question_path = question_path
        # print(f'== Annotation file: {self.annotation_path}')
        # print(f'== Question file: {self.question_path}')
        self.result_file = []
        self.result_path = None

    def init(self):
        self.result_file = []

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

        eval_str = _evaluate(self.annotation_path, self.question_path, self.result_path)
        print()
        print(eval_str)
        if logfile is not None:
            print(eval_str + '\n', file=logfile)


def _evaluate(annotation_file: str, question_file: str, result_file: str):
    # print(f'== Annotation file: {annotation_file}')
    # print(f'== Question file: {question_file}')
    vqa = VQA(annotation_file, question_file)
    vqaRes_prophet = vqa.loadRes(result_file, question_file)
    vqaEval_prophet = VQAEval(vqa, vqaRes_prophet, n=2)
    vqaEval_prophet.evaluate()

    question_types = {
        "eight": "Plants and Animals",
        "nine": "Science and Technology",
        "four": "Sports and Recreation",
        "six": "Geography, History, Language and Culture",
        "two": "Brands, Companies and Products",
        "one": "Vehicles and Transportation",
        "five": "Cooking and Food",
        "ten": "Weather and Climate",
        "seven": "People and Everyday life",
        "three": "Objects, Material and Clothing"
        # "other": "Other",
    }

    result_str = ''
    result_str += "Overall Accuracy is: %.02f\n" % (vqaEval_prophet.accuracy['overall'])
    result_str += f"{'Question Type':40s}\t{'Prophet'}\n"
    for quesType in question_types:
        result_str += "%-40s\t%.02f\n" % (question_types[quesType], vqaEval_prophet.accuracy['perQuestionType'][quesType])
    # print(result_str)
    return result_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate OK-VQA result file.')
    parser.add_argument('--annotation_path', type=str, required=True)
    parser.add_argument('--question_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    args = parser.parse_args()
    result_str = _evaluate(args.annotation_path, args.question_path, args.result_path)
    print(result_str)