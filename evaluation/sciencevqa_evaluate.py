# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Evaluation script for OK-VQA
# ------------------------------------------------------------------------------ #

import json
from evaluation.vqa_utils.vqa import VQA
from evaluation.vqa_utils.vqaEval import VQAEval
from .ans_punct import prep_ans
import argparse

class SCIEvaluater:
    def __init__(self, label_pth:str):
        self.label_pth = label_pth
        #self.choices_path = choices_path
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
        id2datum[each["question_id"]] = each["label"]#.lower()
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
    return result_str

def levenshtein_distance(s1, s2):

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n]

def similarity_score(s1, s2):
    """
    计算两个字符串之间的相似度分数（使用Levenshtein距离的倒数）
    """
    distance = levenshtein_distance(s1, s2)
    max_length = max(len(s1), len(s2))
    similarity = 1 - distance / max_length
    return similarity



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate OK-VQA result file.')
    parser.add_argument('--annotation_path', type=str, required=True)
    parser.add_argument('--question_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    args = parser.parse_args()
    result_str = _evaluate(args.annotation_path, args.question_path, args.result_path)
    print(result_str)