# --------------------------------------------------------------------------------- #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Data loading and preprocessing. Note that for the sake of simplicity,
#              the code only supports the following datasets for now:
#              * VQA 2.0
#              * OK-VQA
#              * A-OKVQA
#              Transferring to other datasets is easy. You may need to modify a few 
#              line of code in this file.
# --------------------------------------------------------------------------------- #

import numpy as np
import glob, json, pickle, random
import torch
import torch.utils.data as Data
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import os,re
from .randaugment import RandomAugment

from evaluation.ans_punct import prep_ans
# from .transforms import _transform


def soft_target(answers, ans_to_ix, preprocess=True):
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    for ans in answers:
        if preprocess:
            ans = prep_ans(ans)
        if ans in ans_to_ix:
            ans_score[ans_to_ix[ans]] = min(1.0, ans_score[ans_to_ix[ans]] + 0.3)
    return ans_score


class CommonData:
    """
    load common data for all dataset objects:
    * imgid_to_path
    * bert tokenizer
    * ans_to_ix, ix_to_ans
    """
    def __init__(self, __C) -> None:
        print('Loading common data...')
        
        # load imgid_to_path
        self.img_feat_path_list = []
        for split in __C.FEATURE_SPLIT:
            feats_dir = __C.FEATS_DIR[split]
            self.img_feat_path_list += glob.glob(feats_dir + '*.npz')
        self.imgid_to_path = {}
        for feat_path in self.img_feat_path_list:
            img_id = int(feat_path.split('/')[-1].split('_')[-1].split('.')[0])
            self.imgid_to_path[img_id] = feat_path
        # self.preprocess = _transform(__C.RESOLUTION)
        print(f'== Total image number: {len(self.imgid_to_path)}')

        # load bert tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(__C.BERT_VERSION)
        self.token_size = self.tokenizer.vocab_size
        print(f'== BertTokenizer loaded, vocab size: {self.token_size}')

        # load ans_to_ix, ix_to_ans
        ans_dict_path = __C.ANSWER_DICT_PATH[__C.DATA_TAG]
        self.ix_to_ans = json.load(open(ans_dict_path, 'r'))
        self.ans_to_ix = {ans: ix for ix, ans in enumerate(self.ix_to_ans)}
        self.ans_size = len(self.ans_to_ix)
        print(f'== Answer vocab size: {self.ans_size}')

        print('Common data process is done.\n')
        

class DataSet(Data.Dataset):
    def __init__(self, __C, common_data, split_name_list):
        self.__C = __C
        print(f'Loading dataset for {self.__C.TASK}|{self.__C.RUN_MODE}({split_name_list})')
        self.split_name_list = split_name_list

        # load all attributes from common data
        self.imgid_to_path = common_data.imgid_to_path
        self.tokenizer = common_data.tokenizer
        self.token_size = common_data.token_size
        self.ans_to_ix = common_data.ans_to_ix
        self.ix_to_ans = common_data.ix_to_ans
        self.ans_size = common_data.ans_size

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        for split_name in split_name_list:
            ques_list = json.load(open(__C.QUESTION_PATH[split_name], 'r'))
            if 'questions' in ques_list:
                ques_list = ques_list['questions']
            self.ques_list += ques_list
            if split_name in __C.ANSWER_PATH:
                ans_list = json.load(open(__C.ANSWER_PATH[split_name], 'r'))
                if 'annotations' in ans_list:
                    ans_list = ans_list['annotations']
                self.ans_list += ans_list

        # indexing data, note that all question_id is set to str,
        # and all image_id is set to int
        if len(self.ans_list) == len(self.ques_list):
            self.annotated = True
            self.qids = [str(ans['question_id']) for ans in self.ans_list]
        elif len(self.ans_list) < len(self.ques_list):
            self.annotated = False
            self.qids = [str(ques['question_id']) for ques in self.ques_list]
        else:
            raise ValueError('Answer list is longer than question list!')

        self.data_size = len(self.qids)
        print(f'== data size: {self.data_size}\n')

        self.qid_to_ques = {str(ques['question_id']): ques for ques in self.ques_list}
        self.qid_to_ans = {str(ans['question_id']): ans for ans in self.ans_list}


    def __getitem__(self, idx):
        # get question in token ids, image in features,
        # and answer in binary-label vector

        __C = self.__C

        # For code safety
        img_feat  = np.zeros(1)
        ques_ids  = np.zeros(1)
        ans_vec   = np.zeros(1)

        qid = self.qids[idx]
        ques_info = self.qid_to_ques[qid]
        
        # Process question
        ques_str = ques_info['question']
        ques_ids = self.bert_tokenize(ques_str, __C.MAX_TOKEN)

        # Process image feature
        img_id = int(ques_info['image_id'])
        img_feat = np.load(self.imgid_to_path[img_id])['x']
        assert img_feat.shape == (__C.IMG_FEAT_GRID, __C.IMG_FEAT_GRID, __C.IMG_FEAT_SIZE)
        img_feat = img_feat.reshape(-1, __C.IMG_FEAT_SIZE)

        # Process answer
        # The code is compatible with VQA v2, OK-VQA, and A-OKVQA.
        # It is no guarantee that it works for other datasets. If
        # you want to use other datasets, please modify following
        # code to fit your dataset.
        if self.annotated:
            ans_info = self.qid_to_ans[qid]
            if 'answers' in ans_info:
                ans_list = [ans['answer'] for ans in ans_info['answers']]
            elif 'direct_answers' in ans_info:
                ans_list = ans_info['direct_answers']
            else:
                raise ValueError('Error: annotation format is not supported!')
            assert type(ans_list[0]) == str, 'Error: answer format is not supported!'
            ans_vec = soft_target(ans_list, self.ans_to_ix)

        return  torch.tensor(img_feat, dtype=torch.float), \
                torch.tensor(ques_ids, dtype=torch.long), \
                torch.tensor(ans_vec, dtype=torch.float)


    def __len__(self):
        return self.data_size

    def bert_tokenize(self, text, max_token):
        text = text.lower().replace('?', '')
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_token - 2:
            tokens = tokens[:max_token-2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids + [0] * (max_token - len(ids))
        ids = np.array(ids, np.int64)

        return ids
    


class Mplug_DataSet(Data.Dataset):
    def __init__(self, __C, split_name_list,feat_test_files=None,feat=False):
        self.__C = __C
        self.feat=feat
        print(f'Loading dataset for {self.__C.TASK}|{self.__C.RUN_MODE}({split_name_list})')
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.ann = []
        self.split = split_name_list
        if self.split=='train':
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(__C.image_res,scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness','ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,])
            self.max_ques_words=30
            for f in self.__C.MPLUG_TRAIN_PATH[self.__C.TASK]:
                self.ann += json.load(open(f,'r')) 
        elif self.split=='test':
            self.transform = transforms.Compose([
                transforms.Resize((__C.image_res,__C.image_res),interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,])   
            self.max_ques_words=50
            if feat_test_files is not None:
                for f in feat_test_files:
                    self.ann += json.load(open(f,'r'))
            else:
                for f in self.__C.MPLUG_TEST_PATH[self.__C.TASK]:
                    self.ann += json.load(open(f,'r')) 
            self.answer_list = json.load(open(__C.answer_list_path,'r'))   
        self.root=__C.data_root
        self.add_ocr=__C.add_ocr
        self.eos = '[SEP]'
        self.add_object = __C.add_object
        

    def __getitem__(self, idx):
        # get question in token ids, image in features,
        # and answer in binary-label vector

        #__C = self.__C
        ann = self.ann[idx]
        if self.__C.DATA_TAG=='text':
            image_path = ann['image']
        else:
            image_path = os.path.join(self.root,ann['image'].replace('_img', ''))
        #image_path = os.path.join(self.root,ann['image'].replace('_img', ''))
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        question = ann['question']
        
        if self.__C.DATA_TAG=='science':
            choices = ann['choices']
            if len(choices) > 0:
                question = question + " [SEP] "
                for num,choice in enumerate(choices):
                    question= question + "choice " + str(num) +' ' +choice +'. '
            if 'hint' in ann:
                question=question+' '+ann['hint']

        if self.add_ocr and "ocr" in ann:
            ocrs = ann['ocr']
            ocr_tokens = []
            poses = []
            for ocr in ocrs:
                pos, token = ocr
                ocr_tokens.append(token)
                poses.append(pos)
            if len(ocr_tokens) > 0:
                ocr_string = pre_question(" ".join(ocr_tokens), self.max_ques_words)
                question = question + " [SEP] " + ocr_string
        
        if self.add_object and "object_label" in ann:
            objects = ann["object_label"]
            question = question + " [SEP] " + " ".join(objects.split("&&"))
        if self.split == 'test':
            if self.feat==True:
                question_id = ann['question_id']
                answer=ann['answer'][0] +self.eos 
                return image, question, answer, question_id
            else:
                question_id = ann['question_id']            
                return image, question, question_id
        
        elif self.split == 'train':
            answer_weight = {}
            for answer in ann['answer']:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1/len(ann['answer'])
                else:
                    answer_weight[answer] = 1/len(ann['answer'])

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())
            answers = [answer+self.eos for answer in answers]
                    
            return image, question, answers, weights


    def __len__(self):
        return len(self.ann)


def pre_question(question,max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')  
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n