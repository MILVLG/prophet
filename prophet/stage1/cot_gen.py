# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Runner that handles the heuristics generations process
# ------------------------------------------------------------------------------ #

import os, sys,copy
# sys.path.append(os.getcwd())

from datetime import datetime
import pickle, random, math, time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as Data
import argparse
from pathlib import Path
import yaml
from copy import deepcopy
from tqdm import tqdm
import requests
import base64
import json
from configs.task_cfgs import Cfgs
from openai import OpenAI

class Runner(object):
    def __init__(self, __C, *args, **kwargs):
        self.__C = __C
        self.client=OpenAI(
            base_url='https://api.key77qiqi.cn/v1',
            api_key='sk-qoc1zRByuvCzlrZFgBaA6wqRwkUbyHQUUXnIKXgNXnbIGOXj'
            )
        #self.net = Non
    
    def encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def inference(self,data,_retry=0):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": data
                    }
                ],
            )
        except Exception as e:
            print(type(e), e)
            return self.inference(data, _retry + 1)
        if response.status_code != 200:
            print('error, retry')
            print(response.status_code)
            if _retry<=3:
                return self.inference(data, _retry + 1)
            else:
                return None
        response_txt = response.choices[0].message.content
        return response_txt
        

    def run(self):
        self.COT_DIR=os.path.join('outputs/results',self.__C.VERSION)
        self.COT_PATH=os.path.join(self.COT_DIR,'cot.json')
        Path(self.COT_DIR).mkdir(parents=True, exist_ok=True)
        
        data_set=[]
        for file in self.__C.COT_DATA_PATH[self.__C.TASK]:
            data_set+=json.load(open(file,'r'))
        prompt=self.__C.PROMPT_HEAD
        
        self.cache = {}
        self.cache_file_path = os.path.join(
            self.COT_DIR,
            'cot_cache.json'
        )
        if self.__C.RESUME:
            self.cache = json.load(open(self.cache_file_path, 'r'))
        for item in tqdm(data_set):
            if str(item['question_id']) in self.cache:
                continue
            image_path=os.path.join(self.__C.DATA_ROOT,item['image'])
            base64_image=self.encode_image(image_path)
            question=item['question'].split('\n')[0].strip()
            inputs=f"""<|user|> {question}\n<|assitant|> Reason: """
            data=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            {"type": "text", "text": inputs},]
                    }
                ]
            result=self.inference(data)
            if not result:
                result=''
            elif "I'm sorry" in result:
                result=''
            self.cache[str(item['question_id'])]=result
        
        with open(self.COT_PATH, 'w') as f:
            json.dump(self.cache, f)


def heuristics_login_args(parser):
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for pretraining')
    heuristics_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)
    runner = Runner(__C)
    runner.run()
