# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Runner that handles the prompting process
# ------------------------------------------------------------------------------ #
#1.适配最新openai api
#2.适配四种数据集
import os, sys
# sys.path.append(os.getcwd())

import pickle
import json, time
import math
import random
import argparse
from datetime import datetime
from copy import deepcopy
import yaml
from pathlib import Path
import openai
import base64
from openai import OpenAI
from .utils.fancy_pbar import progress, info_column
from .utils.data_utils import Qid2Data
from configs.task_cfgs import Cfgs

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class Runner:
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater
        # openai.api_key = __C.OPENAI_KEY
        self.client = OpenAI(
            base_url='https://api.key77qiqi.cn/v1',
            api_key='sk-qoc1zRByuvCzlrZFgBaA6wqRwkUbyHQUUXnIKXgNXnbIGOXj'
            )#Setting by yourself
        self.MC_choices=None
        self.text_ocr=None
        
    
    def gpt_infer(self, prompt_text, _retry=0):
        # print(prompt_text)
        # exponential backoff
        if _retry > 0:
            print('retrying...')
            st = 2 ** _retry
            time.sleep(st)
        
        if self.__C.DEBUG:
            # print(prompt_text)
            time.sleep(0.05)
            return 0, 0

        try:
            # print('calling gpt...')#需要修改
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ],
            )
            # print('gpt3 called.')
        except Exception as e:
            print(type(e), e)
            if str(e) == 'You exceeded your current quota, please check your plan and billing details.':
                exit(1)
            return self.gpt_infer(prompt_text, _retry + 1)

        response_txt = response.choices[0].message.content
        
        return response_txt
    
    def sample_make(self, qid,ques, capt, cands, ans=None,choices=None,ocr=None):
        line_prefix = self.__C.LINE_PREFIX
        cands = cands[:self.__C.K_CANDIDATES]
        cands_with_conf = [f'{cand["answer"]}({cand["confidence"]:.2f})' for cand in cands]
        cands = ', '.join(cands_with_conf)
        
        prompt_text = line_prefix + f'Context: {capt}\n'
        if ocr!=None:
            prompt_text = line_prefix + f'OCR: {ocr}\n'
        prompt_text += line_prefix + f'Question: {ques}\n'
        if qid in self.cot_dict:
            cot=self.cot_dict[qid]
            prompt_text += line_prefix + f'Rationale: {cot}\n'
        prompt_text += line_prefix + f'Candidates: {cands}\n'
        if choices!=None:
            prompt_text += line_prefix + f'Choices: {choices}\n'
        prompt_text += line_prefix + 'Answer:'
        if ans is not None:
            prompt_text += f' {ans}'
            temp_list=[]
        else:
            image_path=self.image_dict[qid]
            image_path=os.path.join(self.__C.DATA_ROOT,image_path.replace('_img', ''))
            base64_image = encode_image(image_path)
            temp_list=[
                {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                },]
        temp_list+=[{"type": "text", "text": f'{prompt_text}'} ]
        return temp_list

    def get_context(self, example_qids):
        # making context text for one testing input
        if self.__C.DATA_TAG=='science':
            prompt_text = self.__C.PROMPT_HEAD_MC
        else:
            prompt_text = self.__C.PROMPT_HEAD
        temp_list=[{"type": "text", "text": prompt_text},]
        # examples = []
        for key in example_qids:
            ques = self.trainset.get_question(key)
            caption = self.trainset.get_caption(key)
            cands = self.trainset.get_topk_candidates(key)
            gt_ans = self.trainset.get_most_answer(key)
            # examples.append((ques, caption, cands, gt_ans))
            if self.__C.DATA_TAG=='science':
                choices= self.MC_choices[str(key)]["str"]
                #choices=MC_choices[str(key)]
                examples = self.sample_make(key,ques, caption, cands, ans=gt_ans,choices=choices)
            elif self.__C.DATA_TAG=='text':
                ocr=self.text_ocr[str(key)]
                examples = self.sample_make(key,ques, caption, cands, ans=gt_ans,ocr=ocr)
            else:
                examples = self.sample_make(key,ques, caption, cands, ans=gt_ans)
            # prompt_text += '\n\n'
            temp_list=temp_list+examples
        return temp_list
    
    def run(self):
        ## where logs will be saved
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')
        ## where results will be saved
        Path(self.__C.RESULT_DIR).mkdir(parents=True, exist_ok=True)
        
        self.cache = {}
        self.cache_file_path = os.path.join(
            self.__C.RESULT_DIR,
            'cache.json'
        )
        if self.__C.RESUME:
            self.cache = json.load(open(self.cache_file_path, 'r'))
        
        print('Note that the accuracies printed before final evaluation (the last printed one) are rough, just for checking if the process is normal!!!\n')
        self.trainset = Qid2Data(
            self.__C, 
            self.__C.TRAIN_SPLITS,
            True
        )
        self.valset = Qid2Data(
            self.__C, 
            self.__C.EVAL_SPLITS,
            self.__C.EVAL_NOW,
            json.load(open(self.__C.EXAMPLES_PATH, 'r'))
        )

        # if 'aok' in self.__C.TASK:
        #     from evaluation.aokvqa_evaluate import AOKEvaluater as Evaluater
        # else:
        #     from evaluation.okvqa_evaluate import OKEvaluater as Evaluater
        # evaluater = Evaluater(
        #     self.valset.annotation_path,
        #     self.valset.question_path
        # )

        infer_times = self.__C.T_INFER
        N_inctx = self.__C.N_EXAMPLES
        
        print()
        test_data = json.load(open(self.__C.MPLUG_TEST_PATH[self.__C.TASK][0],'r'))
        self.image_dict={str(x['question_id']):x['image'] for x in test_data}
        self.cot_dict=json.load(open(self.__C.CoT_PATH))
        if self.__C.DATA_TAG=='science':
            self.MC_choices = json.load(open('/home/ouyangxc/labs/mPLUG_fix_1/data_science/forgpt/MC_choices.json', 'r'))
        elif self.__C.DATA_TAG=='aok_mc':
            self.MC_choices = json.load(open('/home/ouyangxc/labs/mPLUG_fix_1/data_science/forgpt/MC_choices.json', 'r'))
        elif self.__C.DATA_TAG=='text':
            self.text_ocr=json.load(open('/home/ouyangxc/labs/mPLUG_fix_1/data_text/forgpt/ocr_line_10.json', 'r'))

        for qid in progress.track(self.valset.qid_to_data, description="Working...  "):
            if qid in self.cache:
                continue
            ques = self.valset.get_question(qid)
            caption = self.valset.get_caption(qid)
            cands = self.valset.get_topk_candidates(qid, self.__C.K_CANDIDATES)

            prompt_query = self.sample_make(qid, ques, caption, cands)
            
            example_qids = self.valset.get_similar_qids(qid, k=infer_times * N_inctx)
            random.shuffle(example_qids)
            
            ans_pool = {}
            # multi-times infer
            for t in range(infer_times):
                # print(f'Infer {t}...')
                prompt_in_ctx = self.get_context(example_qids[(N_inctx * t):(N_inctx * t + N_inctx)])
                prompt_text = prompt_in_ctx + prompt_query
                # print(prompt_in_ctx[-1])
                # quit()
                # input={"messages":prompt_text}
                gen_text= self.gpt_infer(prompt_text)

                gen_text=gen_text.split('Answer:')[-1].lower()
                ans = self.evaluater.prep_ans(gen_text)
                if ans != '':
                    if ans in ans_pool:
                        ans_pool[ans] += 1.
                    else:
                        ans_pool[ans]= 1.

                time.sleep(self.__C.SLEEP_PER_INFER)
            
            # vote
            if len(ans_pool) == 0:
                answer = self.valset.get_topk_candidates(qid, 1)[0]['answer']
            else:
                answer = sorted(ans_pool.items(), key=lambda x: x[1], reverse=True)[0][0]
            
            self.evaluater.add(qid, answer)
            self.cache[qid] = {
                'question_id': qid,
                'answer': answer,
                # 'prompt_info': prompt_info_list
            }
            json.dump(self.cache, open(self.cache_file_path, 'w'))

            ll = len(self.cache)
            if self.__C.EVAL_NOW and not self.__C.DEBUG:
                if ll > 21 and ll % 10 == 0:
                    rt_accuracy = self.valset.rt_evaluate(self.cache.values())
                    info_column.info = f'Acc: {rt_accuracy}'

        self.evaluater.save(self.__C.RESULT_PATH)
        if self.__C.EVAL_NOW:
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                self.evaluater.evaluate(logfile)
        
def prompt_login_args(parser):
    parser.add_argument('--debug', dest='DEBUG', help='debug mode', action='store_true')
    parser.add_argument('--resume', dest='RESUME', help='resume previous run', action='store_true')
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, default='configs/prompt.yml')
    parser.add_argument('--examples_path', dest='EXAMPLES_PATH', help='answer-aware example file path, default: "assets/answer_aware_examples_for_ok.json"', type=str, default=None)
    parser.add_argument('--candidates_path', dest='CANDIDATES_PATH', help='candidates file path, default: "assets/candidates_for_ok.json"', type=str, default=None)
    parser.add_argument('--cot_path', dest='CoT_PATH', help='cot file path, default: "assets/candidates_for_ok.json"', type=str, default=None)
    parser.add_argument('--captions_path', dest='CAPTIONS_PATH', help='captions file path, default: "assets/captions_for_ok.json"', type=str, default=None)
    # parser.add_argument('--openai_key', dest='OPENAI_KEY', help='openai api key', type=str, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heuristics-enhanced Prompting')
    prompt_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)

    runner = Runner(__C)
    runner.run()
