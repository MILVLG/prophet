# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Runner that handles the finetuning and evaluation process
# ------------------------------------------------------------------------------ #

import os, sys
# sys.path.append(os.getcwd())

from datetime import datetime
import pickle, random, math, time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import argparse
from pathlib import Path
from copy import deepcopy
import yaml
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from configs.task_cfgs import Cfgs
from .utils.load_data import CommonData, DataSet,Mplug_DataSet,vqa_collate_fn
from .model.mPLUG import MPLUG
from .model.mplug_model_utils.tokenization_bert import BertTokenizer
from .utils.optim import get_optim_for_finetune as get_optim
from .utils.mplug_utils import AttrDict, create_two_optimizer,create_scheduler,get_world_size,get_rank,create_sampler
#from .utils import mplug_utils
import deepspeed

class Runner(object):
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater
        self.start_epoch=0
        #self.lr_scheduler=None
        #self.optimizer=None
        
    def train(self, net, train_set, eval_set=None, tokenizer=None,optimizer=None,lr_scheduler=None,device=None,):

        # Define the binary cross entropy loss
        #loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        epoch_loss = 0   
        if self.__C.distributed:
            num_tasks = get_world_size()
            global_rank = get_rank()
            sampler = create_sampler(train_set, True, num_tasks, global_rank)
        else:
            sampler = None
        dataloader = Data.DataLoader(
            train_set,
            sampler=sampler,
            batch_size=self.__C.BATCH_SIZE,
            shuffle=(sampler is None),
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=self.__C.PIN_MEM,
            collate_fn=vqa_collate_fn,
            drop_last=True
        )
        warmup_steps = self.__C.mplug_schedular['warmup_epochs']

        # Training script
        for epoch in range(self.start_epoch, self.__C.MAX_EPOCH): #self.__C.MAX_EPOCH=config['schedular']['epochs']
            net.train()
            if epoch > 0:
                lr_scheduler.step(epoch + warmup_steps)
            if self.__C.distributed:
                dataloader.sampler.set_epoch(epoch)
            step_size = 100
            warmup_iterations = warmup_steps * step_size
            # Save log information
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                logfile.write(
                    f'nowTime: {datetime.now():%Y-%m-%d %H:%M:%S}\n'
                )

            time_start = time.time()
            # Iteration
            for step, input_tuple in enumerate(dataloader):
                iteration_loss = 0
                if epoch > 0 or not self.__C.warm_up:
                    alpha = self.__C.alpha
                else:
                    alpha = self.__C.alpha * min(1, step / len(dataloader))

                image, question, answer, weights, n=input_tuple
                image, weights = image.to(device, non_blocking=True), weights.to(device, non_blocking=True)
                question_input = tokenizer(question, padding='longest', truncation=True, max_length=self.__C.max_input_length, return_tensors="pt").to(device)
                answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device)

                loss = net(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)
                loss=loss/self.__C.GRAD_ACCU_STEPS

                net.backward(loss)
                #net.step(epoch=epoch)
                net.step()
                loss_item = loss.item()
                #iteration_loss += loss_item
                iteration_loss = loss_item
                epoch_loss += loss_item# * self.__C.GRAD_ACCU_STEPS
                if epoch == 0 and step % step_size == 0 and step <= warmup_iterations:
                    lr_scheduler.step(step // step_size)

                print("\r[version %s][epoch %2d][step %4d/%4d][Task %s][Mode %s] loss: %.4f, lr1: %.2e, lr2: %.2e" % (
                    self.__C.VERSION,
                    epoch,
                    step,
                    #int(len(dataloader) / self.__C.BATCH_SIZE),
                    int(len(dataloader)),
                    self.__C.TASK,
                    self.__C.RUN_MODE,
                    iteration_loss, #/ self.__C.BATCH_SIZE,
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[2]["lr"]
                ), end='\n')
                del image,weights, question_input,answer_input, loss

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end - time_start)))

            # Logging
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                logfile.write(f'epoch = {epoch + 1}  loss = {epoch_loss / len(dataloader)}\nlr1 = {optimizer.param_groups[0]["lr"]}\nlr2 = {optimizer.param_groups[2]["lr"]}\n\n')
            
            # Save checkpoint
            net.save_checkpoint(os.path.join(self.__C.CKPTS_DIR), tag='{}.pt'.format(epoch))
            # Eval after every epoch
            if eval_set is not None:
                self.eval(net,eval_set,tokenizer=tokenizer,eval_now=True,device=device)
            
            epoch_loss = 0

    # Evaluation
    @torch.no_grad()
    def eval(self, net, dataset, tokenizer=None, eval_now=False,device=None):
        net.eval()

        
        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        qid_idx = 0
        answer_list = [answer+self.__C.eos for answer in dataloader.dataset.answer_list]
        answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

        for step, input_tuple in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(len(dataloader)),# / self.__C.EVAL_BATCH_SIZE),
            ), end='          ')
            image, question, question_id=input_tuple
            image= image.to(device, non_blocking=True)
            #question_input = tokenizer(question, padding='longest', truncation=True, max_length=self.__C.max_input_length, return_tensors="pt").to(device)
            question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)


            if self.__C.Generate_mode=='topk':
                topk_ids, topk_probs, _ = net(image, question_input, answer_input, train=False, k=self.__C.k_test, mode='topk')
                for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
                    ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
                    self.evaluater.add(ques_id, ans)
            else :
                topk_ids, topk_probs, _, confi = net(image, question_input, answer_input, train=False, k=self.__C.k_test, mode='greedy')
                for ques_id, topk_id, topk_prob, feat in zip(question_id, topk_ids, topk_probs, confi):
                    ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
                    self.evaluater.add(ques_id, ans)

        self.evaluater.save(self.__C.RESULT_PATH)
        # evaluate if eval_now is True
        if eval_now:
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                self.evaluater.evaluate(logfile)

    def load_model(self):
        seed = self.__C.SEED + get_rank()
        # print(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True
        
        tokenizer=BertTokenizer.from_pretrained('/data1/ouyangxc/bert_model/bert-base-uncased')
        net = MPLUG(self.__C,tokenizer) #oy
        print('model done') #

        # Define the optimizer
        if self.__C.RESUME:
            raise NotImplementedError('Resume training is not needed as the finetuning is fast')
        else:
            arg_opt = AttrDict(self.__C.mplug_optimizer) #oy
            optimizer = create_two_optimizer(arg_opt, net) #oy
            #optim = get_optim(self.__C, net)
        self.start_epoch = 0
        arg_sche = AttrDict(self.__C.mplug_schedular)
        lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
        print('lr_scheduler done')
        ## load the pretrained model
        if self.__C.PRETRAINED_MODEL_PATH is not None:
            print(f'Loading pretrained model from {self.__C.PRETRAINED_MODEL_PATH}')
            checkpoint = torch.load(self.__C.PRETRAINED_MODEL_PATH, map_location='cpu')
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
        ## where logs will be saved
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        ## where eval results will be saved
        Path(self.__C.RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')

        # build dataset entities        
        net,tokenizer,optimizer,lr_scheduler,device=self.load_model()

        if self.__C.RUN_MODE == 'finetune_mplug':
            train_set = Mplug_DataSet(self.__C, 'train')
            valid_set = None
            if self.__C.EVAL_NOW:
                valid_set = Mplug_DataSet(self.__C,'test')
            self.train(net,train_set, valid_set,tokenizer=tokenizer,optimizer=optimizer,lr_scheduler=lr_scheduler,device=device)
        elif self.__C.RUN_MODE == 'finetune_mplug_test':
            test_set = Mplug_DataSet(self.__C,'test')
            self.eval(net,test_set,tokenizer=tokenizer, eval_now=self.__C.EVAL_NOW,device=device)
        else:
            raise ValueError('Invalid run mode')

def finetune_login_args(parser):
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test,text_val,text_test,science', type=str, required=True)
    parser.add_argument('--run_mode', dest='RUN_MODE', help='run mode', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--resume', dest='RESUME', help='resume training', type=bool, default=False)
    parser.add_argument('--resume_version', dest='RESUME_VERSION', help='checkpoint version name', type=str, default='')
    parser.add_argument('--resume_epoch', dest='RESUME_EPOCH', help='checkpoint epoch', type=int, default=1)
    parser.add_argument('--resume_path', dest='RESUME_PATH', help='checkpoint path', type=str, default='')
    parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for test', type=str, default=None)
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default=None)
    parser.add_argument('--seed', dest='SEED', help='random seed', type=int, default=None)
    parser.add_argument('--grad_accu', dest='GRAD_ACCU_STEPS', help='random seed', type=int, default=None)
    parser.add_argument('--pretrained_model', dest='PRETRAINED_MODEL_PATH', help='pretrained model path', type=str, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for pretraining')
    finetune_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    runner = Runner(__C)
    runner.run()
