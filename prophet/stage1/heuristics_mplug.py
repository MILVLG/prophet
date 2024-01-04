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
from .model.mplug_model_utils.tokenization_bert import BertTokenizer
from .model.mPLUG import MPLUG
from .utils.mplug_utils import AttrDict, create_two_optimizer,create_scheduler
import multiprocessing as mp

from configs.task_cfgs import Cfgs
from .utils.load_data import CommonData, DataSet,Mplug_DataSet,vqa_collate_fn
#from .model.mcan_for_finetune import MCANForFinetune
from .utils.optim import get_optim_for_finetune as get_optim
import deepspeed

class Runner(object):
    def __init__(self, __C, *args, **kwargs):
        self.__C = __C
        self.train_feat_results=[]
        #self.net = None

    # heuristics generation
    @torch.no_grad()
    def eval(self,net,tokenizer,dataset,device,split=None):
        net.eval()
        
        
        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )
        answer_list = [answer+self.__C.eos for answer in dataloader.dataset.answer_list]
        answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
            
            

        topk_results = {}
        feat_list=[]
        feat_dict=[]
        k = self.__C.CANDIDATE_NUM

        for step, input_tuple in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (step, int(len(dataloader))), end='          ')
            if split=='test':
                image, question, question_id=input_tuple
                file_path='/data2/ouyangxc/NEW_P/out/heuristics_okvqa/test_laten_dict.json'
            elif split=='train':
                image, question, answer, question_id=input_tuple
                feat_answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device)
                file_path='/data2/ouyangxc/NEW_P/out/heuristics_okvqa/train_laten_dict.json'

            image= image.to(device, non_blocking=True)
            question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)
            
            if self.__C.Generate_mode=='topk':#生成topk答案
                topk_ids, topk_probs= net(image, question_input, answer_input, train=False, k=self.__C.k_test, mode='topk')
            else :
                topk_ids, topk_probs = net(image, question_input, answer_input, train=False, k=self.__C.k_test, mode='greedy')
                
            answers_top1 = []
            for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs ):#存储topk答案
                ques_id = int(ques_id.item()) #OK
                answers = []
                for i in range(len(topk_id)):
                    ans = tokenizer.decode(topk_id[i]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
                    answers.append({"answer": ans, "confidence": topk_prob[i].item()})
                    if(i==0):
                        answers_top1.append(ans+'[SEP]')
                topk_results[ques_id] = answers[:k]
            if split=='test':
                feat_answer_input = tokenizer(answers_top1, padding='longest', return_tensors="pt").to(device)
            feats = net(image, question_input, feat_answer_input, train=False, k=k, mode='feat').double().cpu().numpy()
            for ques_id, feat in zip(question_id, feats ):
                ques_id = int(ques_id.item()) #OK
                feat_path=os.path.join(self.__C.ANSWER_LATENTS_DIR, f'{ques_id}.npy')
                #temp_feat=copy.deepcopy(feat)
                feat_list.append({"question_id":ques_id, "feature":feat})
                feat_dict.append({"question_id":ques_id, "path":feat_path})
                np.save(feat_path,feat)
        print()
        json.dump(feat_dict, open(file_path, 'w'),indent=4)
        
        return topk_results, feat_list


    def load_model(self):
        
        tokenizer=BertTokenizer.from_pretrained('/data1/ouyangxc/bert_model/bert-base-uncased')
        net = MPLUG(self.__C,tokenizer) #oy
        print('model done') #

        # Define the optimizer
        arg_opt = AttrDict(self.__C.mplug_optimizer) #oy
        optimizer = create_two_optimizer(arg_opt, net) #oy
        arg_sche = AttrDict(self.__C.mplug_schedular)
        lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
        print('lr_scheduler done')
        ## load the model
        print(f'Loading pretrained model from {self.__C.CKPT_PATH}')
        checkpoint = torch.load(self.__C.CKPT_PATH, map_location='cpu')
        try:
            state_dict = checkpoint['module']
        except:
            state_dict = checkpoint
        msg = net.load_state_dict(state_dict, strict=False)
        print('Finish loading.')
        if self.__C.deepspeed:
            net, optimizer, _, _ = deepspeed.initialize(
                model=net,
                optimizer=optimizer,
                args=self.__C,
                lr_scheduler=lr_scheduler,
                dist_init_required=True)
        print('deepspeed done')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('load model done')
        return net,tokenizer,optimizer,lr_scheduler,device
        
        

    def run(self):
        # Set ckpts and log path
        ## where checkpoints will be saved
        Path(self.__C.CKPTS_DIR).mkdir(parents=True, exist_ok=True)
        ## where the result file of topk candidates will be saved
        Path(self.__C.CANDIDATE_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)
        ## where answer latents will be saved
        Path(self.__C.ANSWER_LATENTS_DIR).mkdir(parents=True, exist_ok=True)
        
        #load model
        net,tokenizer,_,_,device=self.load_model()

        # build dataset entities      
        #在finetune的时候生成一个top1 jsonfile，用于生成feat  
        train_set = Mplug_DataSet(self.__C,'test',self.__C.MPLUG_TRAIN_PATH[self.__C.TASK],feat=True)
        test_set = Mplug_DataSet(self.__C,'test',self.__C.MPLUG_TEST_PATH[self.__C.TASK])

        # forward VQA model
        train_topk_results, train_feat_results = self.eval(net,tokenizer,train_set,device,split='train')
        test_topk_results, test_feat_results = self.eval(net,tokenizer,test_set,device,split='test')
        self.train_feat_results=train_feat_results

        # save topk candidates
        topk_results = train_topk_results | test_topk_results
        json.dump(
            topk_results,
            open(self.__C.CANDIDATE_FILE_PATH, 'w'),
            indent=4
        )

        # search similar examples
        E = self.__C.EXAMPLE_NUM
        print(len(test_feat_results))
        self.similar_qids = {}
        print(f'\ncompute top-{E} similar examples for each testing input')
        with mp.Pool() as pool:
            results = pool.map(self.calculate_similarity, test_feat_results)
        
        """for qid,test_feat in test_feat_results.items():
            test_feat = test_feat / np.linalg.norm(test_feat, axis=-1, keepdims=True)
            test_temp_dict = {}
            for train_qid, train_feat in train_feat_results.items():
                train_feat = train_feat / np.linalg.norm(train_feat, axis=-1, keepdims=True)
                sim = np.matmul(train_feat, test_feat.T)
                sim_temp = (np.sum(np.mean(sim, axis=0)) + np.sum(np.mean(sim, axis=1))) / (sim.shape[0] + sim.shape[1])
                test_temp_dict[train_qid] = sim_temp
            sorted_items = dict(sorted(test_temp_dict.items(), key=lambda x: x[1], reverse=True))
            self.similar_qids[qid]= list(sorted_items.keys())[:E]    """   
        # save similar qids
        print(len(self.similar_qids))
        with open(self.__C.EXAMPLE_FILE_PATH, 'w') as f:
            json.dump(self.similar_qids, f)


def heuristics_login_args(parser):
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for heuristics', type=str, default=None)
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default=None)
    parser.add_argument('--candidate_num', dest='CANDIDATE_NUM', help='topk candidates', type=int, default=None)
    parser.add_argument('--example_num', dest='EXAMPLE_NUM', help='number of most similar examples to be searched, default: 200', type=int, default=None)


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
