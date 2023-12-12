from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint
#from models.med import BertConfig
#from transformers import BertModel, BertLMHeadModel
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np

class BLIP_VQA_MULTI(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_configmulti.json',  #'multi_cased_L-12_H-768_A-12/my_config.json',#
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 lan='multi',
                 config=None    
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer(lan,config['sp'])  

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        encoder_config.vocab_size = len(self.tokenizer)
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False) 
        decoder_config = BertConfig.from_json_file(med_config)   
        decoder_config.vocab_size = len(self.tokenizer)
        self.text_decoder = BertLMHeadModel(config=decoder_config)
        self.max_length = config['max_length']
        self.max_q_length = config['max_q_length']
        #self.img_num = img_num

    def forward(self, image, question, answer=None, n=None, weights=None, train=True, inference='rank', k_test=128,
                sample=False, num_beams=3, max_length=30, top_p=0.9, repetition_penalty=1.0):
        #print(image.shape)
        # print('Q:',question,'ANS:',answer)
        # tmp=torch.zeros(image.shape[2],image.shape[3],image.shape[4]).to(image.device)
        # for i,img_set in enumerate(image):
        #     print(img_set.shape)
        #     real_imgs = torch.unsqueeze(img_set[0], 0)
        #     for ii,img in enumerate(img_set):
        #         if ii==0:
        #             continue
        #         if torch.equal(tmp,img):
        #             break
        #         real_imgs = torch.cat((real_imgs, torch.unsqueeze(img, 0)), 0)
        #     #print(real_imgs.shape)
        #     image_embed = torch.unsqueeze(self.visual_encoder(real_imgs),0)
        #     #print(image_embed.shape)
        #     image_embed = torch.squeeze(torch.mean(image_embed, dim=1, keepdim=True),1)
        #     if i==0:
        #         image_embeds = image_embed
        #         #print(image_embed)
        #         #print(self.visual_encoder(image[:,0,:,:])[0])
        #     else:
        #         image_embeds = torch.cat((image_embeds, image_embed),0)
        # image_embeds=[]
        # # print(image.shape)
        # for i,img_set in enumerate(image):
        #     # print(img_set.shape)
        #     # nonZero = img_set.abs().sum(dim=(1,2,3)).bool()
        #     # # print(nonZero.shape)
        #     # img_set=img_set[nonZero]
        #     # # print(img_set.shape)
        #     # embed_id=self.visual_encoder(img_set)
        #     # # embed_id= F.normalize(embed_id,dim=0)  
        #     # # print(embed_id.shape)
        #     # embed_id=torch.mean(embed_id,axis=0)
        #     # print(embed_id.shape)
        #     embed_id=self.visual_encoder(img_set[:1,:,:,:])
        #     image_embeds.append(embed_id)     
        # image_embeds= torch.stack(image_embeds)
        # image_embeds= F.normalize(image_embeds,dim=0)  
        # image_embeds=self.visual_encoder(image[:,0,:,:,:])
        # print(image_embeds.shape)
        # print('**********************************')
        nonZero = image.abs().sum(dim=(0,2,3,4)).bool()
        image=image[:,nonZero,:,:,:]
        B,N,C,W,H = image.size()
        video_embed = self.visual_encoder(image.view(-1,C,W,H)) #(B*N,)    
        T,W,H =  video_embed.size()
        image_embeds = video_embed.view(B,N,W,H).mean(dim=1)
        # image_embeds = F.normalize(video_embed,dim=(1,2))  
        # print(image_embeds.shape)
        #batch_size*img_num*channel*size*size
        #NUM = image.shape[1]
        ################
        #1. 50s  28801MiB
        #image_ = torch.reshape(image, (image.shape[0]*image.shape[1], image.shape[2], image.shape[3], image.shape[4]))
        #image_embeds = self.visual_encoder(image_)
        #image_embeds = torch.reshape(image_embeds, (int(image_embeds.shape[0]/NUM), NUM, image_embeds.shape[1], image_embeds.shape[2]))
       
        ################
        '''
        ################
        #2. 29173MiB
        image_embeds = torch.unsqueeze(self.visual_encoder(image[:,0,:,:,:]),1)
        for i in range(1, NUM):
            image_embed = torch.unsqueeze(self.visual_encoder(image[:,i,:,:,:]),1)
            image_embeds = torch.cat((image_embeds, image_embed),1)
        ################
        '''
        #print(image_embeds.shape)
        #image_embeds = torch.squeeze(torch.mean(image_embeds, dim=1, keepdim=True),1)
        #print(image_embeds.shape)
        #image_embeds = self.visual_encoder(image[:,0,:,:])
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=self.max_q_length,#35
                                  return_tensors="pt").to(image.device) 
        question.input_ids[:,0] = self.tokenizer.enc_token_id
        
        if train:               
            '''
            n: number of answers for each question
            weights: weight for each answer
            '''                     
            answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device) 
            answer.input_ids[:,0] = self.tokenizer.bos_token_id
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True)    

            question_states = []                
            question_atts = []  
            for b, n in enumerate(n):
                question_states += [question_output.last_hidden_state[b]]*n
                question_atts += [question.attention_mask[b]]*n                
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0)     

            answer_output = self.text_decoder(answer.input_ids, 
                                              attention_mask = answer.attention_mask, 
                                              encoder_hidden_states = question_states,
                                              encoder_attention_mask = question_atts,                  
                                              labels = answer_targets,
                                              return_dict = True,   
                                              reduction = 'none',
                                             )      
            # print('pred: ',answer_output)
            loss = weights * answer_output.loss
            loss = loss.sum()/image.size(0)

            return loss
            
        else: 
            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True) 
            
            if inference=='generate':
                # num_beams = 3
                question_states = question_output.last_hidden_state.repeat_interleave(num_beams,dim=0)
                question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
                model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
                
                bos_ids = torch.full((image.size(0),1),fill_value=self.tokenizer.bos_token_id,device=image.device)
                
                # outputs = self.text_decoder.generate(input_ids=bos_ids,
                #                                     #  max_length=10,
                #                                     max_length=150,
                #                                      min_length=1,
                #                                      num_beams=num_beams,
                #                                      eos_token_id=self.tokenizer.sep_token_id,
                #                                      pad_token_id=self.tokenizer.pad_token_id, 
                #                                      **model_kwargs)
                if sample:
                    #nucleus sampling
                    outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                        max_length=self.max_length,
                                                        min_length=1,
                                                        do_sample=True,
                                                        top_p=top_p,
                                                        num_return_sequences=1,
                                                        eos_token_id=self.tokenizer.sep_token_id,
                                                        pad_token_id=self.tokenizer.pad_token_id, 
                                                        repetition_penalty=1.1,                                            
                                                        **model_kwargs)
                else:
                    #beam search
                    #print()
                    outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                        max_length=self.max_length,
                                                        min_length=1,
                                                        num_beams=num_beams,
                                                        eos_token_id=self.tokenizer.sep_token_id,
                                                        pad_token_id=self.tokenizer.pad_token_id,     
                                                        repetition_penalty=repetition_penalty,
                                                        **model_kwargs)    
                
                answers = []    
                for output in outputs:
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)    
                    answers.append(answer)
                return answers
            
            elif inference=='rank':
                max_ids = self.rank_answer(question_output.last_hidden_state, question.attention_mask, 
                                           answer.input_ids, answer.attention_mask, k_test) 
                return max_ids
 
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
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
                                   reduction = 'none')   
        
        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques,k)

        max_topk_ids = log_probs_sum.argmax(dim=1) 
        max_ids = topk_ids[max_topk_ids>=0,max_topk_ids]

        return max_ids
    
    
def blip_vqa(pretrained='',**kwargs):
    model = BLIP_VQA_MULTI(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
#         assert(len(msg.missing_keys)==0)
    return model  


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
        
        
