import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .mplug_model_utils.modeling_mplug import BertConfig, BertModel, BertLMHeadModel, FusionModel
from .mplug_model_utils.tokenization_bert import BertTokenizer
from .mplug_model_utils.visual_transformers import initialize_clip
#from models.predictor import TextGenerator
from .mplug_model_utils.predictor2 import TextGenerator


class MPLUG(nn.Module):
    def __init__(self, __C , tokenizer):
        super().__init__()
        
        self.tokenizer = tokenizer #BertTokenizer.from_pretrained('/data1/ouyangxc/bert_model/bert-base-uncased') #tokenizer 
        self.module_setting(__C)
        self.visual_encoder, _ = initialize_clip(__C)
        self.text_encoder = BertModel.from_pretrained(__C.text_encoder, config=self.config_encoder, add_pooling_layer=False)  
        self.fusion_encoder = FusionModel.from_pretrained(__C.text_decoder, config=self.config_fusion, add_pooling_layer=False)  
        self.text_decoder = BertLMHeadModel.from_pretrained(__C.text_decoder, config=self.config_decoder)    
        self.init_distill(__C)
        self.beam_generator = TextGenerator(__C, self.text_decoder) 
            
        
    def forward(self, image, question, answer=None, alpha=0, k=None, weights=None, train=True, mode='greedy'):

        image = image.to(dtype=next(self.parameters()).dtype) 
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        if train:               
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''          
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      
            text_output = self.text_encoder(question.input_ids, attention_mask=question.attention_mask, return_dict=True)
            text_embeds = text_output.last_hidden_state
            fusion_output = self.fusion_encoder(encoder_embeds=text_embeds, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts, return_dict=False)
            
            image_output, question_output = fusion_output
            
            question_output = torch.cat([image_output, question_output], 1)
            merge_text_attention = torch.cat([image_atts, question.attention_mask], 1)
            
            question_states = []                
            question_atts = []  
            for b, n in enumerate(k):
                question_states += [question_output[b]]*n
                question_atts += [merge_text_attention[b]]*n 
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0)     

            if self.distill:                    
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m.visual(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint) 
                    if self.large:
                        image_embeds_m = self.dropout_m(self.visn_layer_norm_m(self.visn_fc_m(image_embeds_m)))
                    text_output_m = self.text_encoder_m(question.input_ids, attention_mask=question.attention_mask,
                                                return_dict=True)
                    text_embeds_m = text_output_m.last_hidden_state
                    fusion_output_m = self.fusion_encoder_m(encoder_embeds=text_embeds_m, 
                                                            attention_mask = question.attention_mask, 
                                                            encoder_hidden_states = image_embeds_m,
                                                            encoder_attention_mask = image_atts, return_dict=False) 

                         
                    image_output_m, question_output_m = fusion_output_m
                    question_output_m = torch.cat([image_output_m, question_output_m], 1)
                    
                    question_states_m = []                
                    for b, n in enumerate(k):
                        question_states_m += [question_output_m[b]]*n
                    question_states_m = torch.stack(question_states_m,0)    

                    logits_m = self.text_decoder_m(answer.input_ids, 
                                                   attention_mask = answer.attention_mask, 
                                                   encoder_hidden_states = question_states_m,
                                                   encoder_attention_mask = question_atts,                                  
                                                   return_logits = True,
                                                   )                       

                answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = question_states,
                                                  encoder_attention_mask = question_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  soft_labels = F.softmax(logits_m,dim=-1),
                                                  reduction = 'none',
                                                 )   
            else:
                answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = question_states,
                                                  encoder_attention_mask = question_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  reduction = 'none',
                                                 )                      
            loss = weights * answer_output.loss         
            loss = loss.sum()/image.size(0)

            return loss
        
        else: 
            # while True:
            text_output = self.text_encoder(question.input_ids, attention_mask=question.attention_mask,
                                                return_dict=True)
            text_embeds = text_output.last_hidden_state
            fusion_output = self.fusion_encoder(encoder_embeds=text_embeds, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = False) 
            image_output, question_output = fusion_output 
            question_output = torch.cat([image_output, question_output], 1)
            encoding = question_output.mean(1)
            merge_text_attention = torch.cat([image_atts, question.attention_mask], 1)
            if mode == 'topk':
                topk_ids, topk_probs, feats = self.rank_answer(question_output, merge_text_attention, answer.input_ids, answer.attention_mask, 128)
                
                return topk_ids, topk_probs
            elif mode == 'feat':
                feats = self.get_feat(question_output, merge_text_attention, answer.input_ids, answer.attention_mask, 128, question.input_ids)
                return feats
            elif mode == 'greedy':
                #topk_ids, topk_probs = self.generation(question_output, merge_text_attention) 
                topk_ids, topk_probs,losses = self.generation(question_output, merge_text_attention, self.tokenizer.pad_token_id) #oyoy
                return topk_ids, losses
                #return topk_ids, topk_probs, encoding #oyoy
            elif mode == 'feat':
                feats = self.get_feat(question_output, merge_text_attention, answer.input_ids, answer.attention_mask, 128, question.input_ids)
                
                return feats
 

    def module_setting(self, __C):
        self.config_encoder = BertConfig.from_json_file(__C.bert_config)   
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.config_fusion = BertConfig.from_json_file(__C.bert_config)   
        self.config_decoder = BertConfig.from_json_file(__C.bert_config)
        self.config_decoder.add_cross_attention = True
        self.config_decoder.num_hidden_layers = self.config_decoder.text_decode_layers
        self.large = False
        if self.config_encoder.hidden_size != __C.vision_width:
            self.visn_fc = nn.Linear(__C.vision_width, self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
        self.use_checkpoint = __C.use_checkpoint 
    
    def init_distill(self, __C):
        self.distill = __C.distill
        if self.distill:
            self.visual_encoder_m, _ = initialize_clip(__C)
            self.text_encoder_m = BertModel.from_pretrained(__C.text_encoder, config=self.config_encoder, add_pooling_layer=False)  
            self.fusion_encoder_m = FusionModel.from_pretrained(__C.text_encoder, config=self.config_fusion, add_pooling_layer=False)  
            self.text_decoder_m = BertLMHeadModel.from_pretrained(__C.text_decoder, config=self.config_decoder)    
            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.text_decoder,self.text_decoder_m],
                               ]
            if self.config_encoder.hidden_size != __C.vision_width:
                self.visn_fc_m = nn.Linear(__C.vision_width, self.config_encoder.hidden_size)
                self.visn_layer_norm_m = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
                self.dropout_m = nn.Dropout(self.config_encoder.hidden_dropout_prob)
                self.model_pairs.extend([[self.visn_fc, self.visn_fc_m], [self.visn_layer_norm, self.visn_layer_norm_m]])
            self.copy_params() 
            self.momentum = 0.995

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    def generation(self, question_states, question_atts, pad_token_id=None):
        encoder_inputs = [question_states, question_atts]
        #topk_ids, topk_scores = self.beam_generator.translate_batch(encoder_inputs)  #out_size=1
        #topk_ids, topk_scores = self.beam_generator.translate_batch(encoder_inputs,out_size=10)  #confi1
        topk_ids, topk_scores,losses = self.beam_generator.translate_batch(encoder_inputs,out_size=10,pad_token_id=pad_token_id)  #confi2
        return topk_ids, topk_scores,losses
        #return topk_ids, topk_scores

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')
        logits = start_output.logits[:,0,:] # first token's logit       
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1)                
        input_ids = []
        input_atts = []
        #input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0) 
        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
        
        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none',
                                   loss1_need=True)                 
        answer_loss = output.loss #loss1 


        last_hidden_states=output.hidden_states[:,0,:].view(num_ques, k, -1)
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        topk_probs = topk_probs.view(-1,1)
        log_probs = -answer_loss

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)
        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs1, topk_ids1 = topk_probs.topk(k,dim=1)

        True_last_hidden_states = torch.gather(last_hidden_states, 1, topk_ids1.unsqueeze(-1).expand(-1, -1, 768))
        topk_ans_ids = topk_ids[
            torch.arange(num_ques).unsqueeze(1).repeat(1,k),
            topk_ids1
        ]
        topk_lm_ids = answer_ids.index_select(dim=0, index=topk_ans_ids.view(-1)).view(num_ques, k, -1)
        return topk_lm_ids, topk_probs1, True_last_hidden_states
    

    def get_feat(self, question_states, question_atts, answer_ids, answer_atts, k, inputs):
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        start_output = self.text_decoder(answer_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')
        
        return start_output.hidden_states
 
    
def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
