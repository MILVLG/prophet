# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Utilities for optimization
# ------------------------------------------------------------------------------ #

import torch
import torch.optim as Optim
from torch.nn.utils import clip_grad_norm_

class OptimizerWrapper(object):
    """
    A Wrapper for optimizer to support learning rate warmup and decay.
    It also support multiple optimizers and switching at different steps.
    """
    def __init__(self, optimizers, 
                 warmup_schd_steps,
                 decay_schd_step_list,
                 decay_rate, 
                 cur_schd_step=-1,
                 change_optim_step_list=None
        ):
        self.optimizer_list = optimizers
        self.groups_lr_list = []
        for _optim in self.optimizer_list:
            self.groups_lr_list.append([])
            for group in _optim.param_groups:
                self.groups_lr_list[-1].append(group['lr'])
        self.curr_optim_id = 0
        self.optimizer = self.optimizer_list[self.curr_optim_id]
        self.change_optim_step_list = change_optim_step_list
        # self.total_schd_steps = total_schd_steps
        self.warmup_schd_steps = warmup_schd_steps
        self.decay_schd_step_list = decay_schd_step_list
        self.decay_rate = decay_rate
        self._step = 0
        self.warmup_lr_scale = 1.0
        self.decay_lr_scale = 1.0
        self.schedule_step(cur_schd_step)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, step=None, schd_step=False):
        if step is None:
            step = self._step
        if schd_step:
            self.schedule_step(step)
        
        for group in self.optimizer.param_groups:
            if '_grad_norm_clip' in group:
                if group['_grad_norm_clip'] > 0:
                    clip_grad_norm_(group['params'], group['_grad_norm_clip'])
        
        self.optimizer.step()
        self._step += 1
    
    def schedule_step(self, schd_step):
        schd_step += 1
        self.warmup_lr_scale = min(1., float(schd_step + 1) / float(self.warmup_schd_steps + 1))
        if schd_step in self.decay_schd_step_list:
            self.decay_lr_scale = self.decay_lr_scale * self.decay_rate
        lr_scale = self.warmup_lr_scale * self.decay_lr_scale
        # lr actually changes in following lines
        if self.change_optim_step_list is not None:
            if schd_step in self.change_optim_step_list:
                self.curr_optim_id += 1
                self.optimizer = self.optimizer_list[self.curr_optim_id]
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr_scale * self.groups_lr_list[self.curr_optim_id][i]

    def current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
    

def get_optim(__C, model):
    optim_class = eval('Optim.' + __C.OPT)
    params = [
        {'params': [], 'lr': __C.LR_BASE, '_grad_norm_clip': __C.GRAD_NORM_CLIP},
        {'params': [], 'lr': __C.LR_BASE * __C.BERT_LR_MULT, '_grad_norm_clip': __C.GRAD_NORM_CLIP},
    ]
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bert' in name:
                params[1]['params'].append(param)
            else:
                params[0]['params'].append(param)
    hyper_params = {k: eval(v) for k, v in __C.OPT_PARAMS.items()}
    return OptimizerWrapper(
        [optim_class(
            params,
            **hyper_params
        ),],
        warmup_schd_steps=__C.WARMUP_EPOCH,
        decay_schd_step_list=__C.LR_DECAY_LIST,
        decay_rate=__C.LR_DECAY_R,
    )


def get_optim_for_finetune(__C, model, new_params_name='proj1'):
    # optimizer for finetuning warmup
    optim_class1 = eval('Optim.' + __C.OPT_FTW)
    params1 = []
    for name, param in model.named_parameters():
        if new_params_name in name and param.requires_grad:
            params1.append(param)
    hyper_params1 = {k: eval(v) for k, v in __C.OPT_PARAMS_FTW.items()}
    optimizer1 = optim_class1(
        params1,
        lr=__C.LR_BASE_FTW,
        **hyper_params1
    )

    optim_class2 = eval('Optim.' + __C.OPT)
    params2 = [
        {'params': [], 'lr': __C.LR_BASE, '_grad_norm_clip': __C.GRAD_NORM_CLIP},
        {'params': [], 'lr': __C.LR_BASE * __C.BERT_LR_MULT, '_grad_norm_clip': __C.GRAD_NORM_CLIP},
    ]
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bert' in name:
                params2[1]['params'].append(param)
            else:
                params2[0]['params'].append(param)
    hyper_params2 = {k: eval(v) for k, v in __C.OPT_PARAMS.items()}
    optimizer2 = optim_class2(
        params2,
        **hyper_params2
    )
    return OptimizerWrapper(
        [optimizer1, optimizer2],
        warmup_schd_steps=__C.WARMUP_EPOCH,
        decay_schd_step_list=__C.LR_DECAY_LIST,
        decay_rate=__C.LR_DECAY_R,
        change_optim_step_list=[__C.EPOPH_FTW,]        
    )
