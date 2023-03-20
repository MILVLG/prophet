# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Runner that handles the pretraining process
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

from configs.task_cfgs import Cfgs
from .utils.load_data import CommonData, DataSet
from .model.mcan import MCAN
from .utils.optim import get_optim

class Runner(object):
    def __init__(self, __C, *args, **kwargs):
        self.__C = __C

    def train(self, train_set, eval_set=None):
        data_size = train_set.data_size

        # Define the MCAN model
        net = MCAN(self.__C, train_set.ans_size)

        # Define the optimizer
        # Load checkpoint if resume training
        if self.__C.RESUME:
            print(' ========== Resume training')

            path = self.__C.RESUME_PATH

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path, map_location='cpu')
            print('Finish loading.')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.__C, net)
            optim.warmup_lr_scale = ckpt['warmup_lr_scale']
            optim.decay_lr_scale = ckpt['decay_lr_scale']
            optim.optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = self.__C.CKPT_EPOCH

        else:
            optim = get_optim(self.__C, net)
            start_epoch = 0

        # load to gpu
        net.cuda()
        # Define the multi-gpu training if needed
        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.GPU_IDS)

        # Define the binary cross entropy loss
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        epoch_loss = 0

        # Define multi-thread dataloader
        dataloader = Data.DataLoader(
            train_set,
            batch_size=self.__C.BATCH_SIZE,
            shuffle=True,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=self.__C.PIN_MEM,
            drop_last=True
        )

        # Training script
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):
            net.train()
            # Save log information
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                logfile.write(
                    f'nowTime: {datetime.now():%Y-%m-%d %H:%M:%S}\n'
                )

            time_start = time.time()

            # Iteration
            for step, input_tuple in enumerate(dataloader):
                iteration_loss = 0
                optim.zero_grad()
                input_tuple = [x.cuda() for x in input_tuple]
                SUB_BATCH_SIZE = self.__C.BATCH_SIZE // self.__C.GRAD_ACCU_STEPS
                for accu_step in range(self.__C.GRAD_ACCU_STEPS):

                    sub_tuple = [x[accu_step * SUB_BATCH_SIZE:
                        (accu_step + 1) * SUB_BATCH_SIZE] for x in input_tuple]
                    
                    sub_ans_iter = sub_tuple[-1]
                    pred = net(sub_tuple[:-1])
                    loss = loss_fn(pred, sub_ans_iter)
                    loss.backward()
                    loss_item = loss.item()
                    iteration_loss += loss_item
                    epoch_loss += loss_item# * self.__C.GRAD_ACCU_STEPS

                print("\r[version %s][epoch %2d][step %4d/%4d][Task %s][Mode %s] loss: %.4f, lr: %.2e" % (
                    self.__C.VERSION,
                    epoch + 1,
                    step,
                    int(data_size / self.__C.BATCH_SIZE),
                    self.__C.TASK,
                    self.__C.RUN_MODE,
                    iteration_loss / self.__C.BATCH_SIZE,
                    optim.current_lr(),
                ), end='          ')

                optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end - time_start)))

            # Logging
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                logfile.write(f'epoch = {epoch + 1}  loss = {epoch_loss / data_size}\nlr = {optim.current_lr()}\n\n')
            
            optim.schedule_step(epoch)

            # Save checkpoint
            state = {
                'state_dict': net.state_dict() if self.__C.N_GPU == 1 \
                    else net.module.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'warmup_lr_scale': optim.warmup_lr_scale,
                'decay_lr_scale': optim.decay_lr_scale,
            }
            torch.save(
                state,
                f'{self.__C.CKPTS_DIR}/epoch{epoch + 1}.pkl'
            )

            epoch_loss = 0

    def run(self):
        # Set ckpts and log path
        Path(self.__C.CKPTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')
        
        common_data = CommonData(self.__C)
        train_set = DataSet(
            self.__C, 
            common_data,
            self.__C.TRAIN_SPLITS
        )
        valid_set = None
        self.train(train_set, valid_set)

def pretrain_login_args(parser):
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--resume', dest='RESUME', help='resume training', type=bool, default=False)
    parser.add_argument('--resume_version', dest='RESUME_VERSION', help='checkpoint version name', type=str, default=None)
    parser.add_argument('--resume_epoch', dest='RESUME_EPOCH', help='checkpoint epoch', type=int, default=None)
    parser.add_argument('--resume_path', dest='RESUME_PATH', help='checkpoint path', type=str, default=None)
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default=None)
    parser.add_argument('--seed', dest='SEED', help='random seed', type=int, default=None)
    parser.add_argument('--grad_accu', dest='GRAD_ACCU_STEPS', help='random seed', type=int, default=None)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for pretraining')
    pretrain_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)
    runner = Runner(__C)
    runner.run()
