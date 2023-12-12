'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import pandas as pd
import argparse
import os
from ruamel.yaml import YAML
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_vqa_multi_images import blip_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result
from pycocoevalcap.eval import eval
from data.utils import pre_caption
import re
import csv
import os

def clean_space(text):
  match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z,.!?:])|\d+ +| +\d+|[a-z A-Z , . ! ? :]+')
  should_replace_list = match_regex.findall(text)
  order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
  for i in order_replace_list:
    if i == u' ':
      continue
    new_i = i.strip()
    text = text.replace(i,new_i)
  text = re.sub(r'\s[,]', ',', text)
  text = re.sub(r'\s[.]', '.', text)
  text = re.sub(r'\s[!]', '!', text)
  text = re.sub(r'\s[;]', ';', text)
  text = re.sub(r'\s[:]', ':', text)
  text = re.sub(r'\s[?]', '?', text)
  text = re.sub(r'\s[-]\s', '-', text)
  return text
def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 5000    
    
    for i,(image, question, answer, weights, _, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # print('question',question)
        # print('answer',answer)
        # print(image.shape)
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      

        loss = model(image, question, answer, train=True, n=n, weights=weights)        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 

@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 100
    
    result = []
    res = {}
    
    if config['inference']=='rank':   
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id
        
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        if config['inference']=='generate':
            # answers = model(image, question, train=False, inference='generate') 
            answers = model(image, question, train=False, inference='generate',num_beams=config['num_beams'], max_length=config['max_length']) 
            for answer, ques_id in zip(answers, question_id):
                # print(ques_id)
                # answer = clean_space(answer)
                # ques_id = int(ques_id.item())      
                ques_id = int(ques_id)   
                result.append({"idx":ques_id, "answer":answer,})
                res[str(ques_id)]=[answer]
    return result, res

def splitdf(df,col='eid_ckd',test_size=.2):

    if col==None:
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=test_size, random_state=42,stratify=None)
    else:
        from sklearn.model_selection import GroupShuffleSplit
        train_inds, test_inds = next(GroupShuffleSplit(test_size=test_size, n_splits=2, random_state = 7).split(df, groups=df[col]))
        train = df.iloc[train_inds].reset_index(drop=True)
        test = df.iloc[test_inds].reset_index(drop=True)
    return train,test

def main(args, config,df):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)
    config['device'] = device

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset #### 
    print("Creating vqa datasets")
    
    if 'split' in df.columns:
        TRAIN=df[df.split=='Train'].reset_index(drop=True)
        VALID=df[df.split=='Validation'].reset_index(drop=True)
        TEST=df[df.split=='Test'].reset_index(drop=True)
    else:
        TRAIN,VALID=splitdf(df ,config['eid'],test_size=config['test_size'])
        VALID,TEST = splitdf(VALID ,config['eid'],0.5)
    #if len(VALID)==0:
    #    VALID=TRAIN[:1000].reset_index(drop=True)
    
    TRAIN.to_csv(os.path.join(args.output_dir,'train.csv'))
    VALID.to_csv(os.path.join(args.output_dir,'valid.csv'))
    VALID=VALID[:20000] ################################################################ to accelerate
    TEST.to_csv(os.path.join(args.output_dir,'test.csv'))
    datasets = create_dataset(TRAIN,VALID,config)   
    print_freq = len(VALID.drop_duplicates('eid'))//10
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)  # mod0529       
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test']],
                                              num_workers=[4,4],is_trains=[True, False], 
                                              collate_fns=[vqa_collate_fn,None],args=args) 
    #### Model #### 
    print("Creating model")
    #config['pretrained'] = "/home/danli/caption/BLIP/output/fa/05-31-1307/checkpoint_00.pth"
    model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'], 
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                       lan=config['language'],
                       config=config)
    
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0 

    print("Start training")
    start_time = time.time()    
    # metric_log = open(os.path.join(args.output_dir, "val_metrics.txt"), "w")
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
    
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])                
            train_stats = train(model, train_loader, optimizer, epoch, device) 

        else:         
            break        
        
        if utils.is_main_process():     
            save_obj = {'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,}
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")    
                                
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  

        if args.distributed:   
            dist.barrier()     
        else:
            vqa_result,res = evaluation(model_without_ddp, test_loader, device, config)
            #print(res)
            gts={}
            gts_dt = pd.read_csv(os.path.join(args.output_dir,'test_mini.csv'))
            # gts_dt = pd.read_csv(os.path.join(args.output_dir,'test.csv')).drop_duplicates('eid')

            for  i,row in gts_dt.iterrows():
                gt=row["ANS"]
                gt = pre_caption(gt,max_words=config['max_length'])
                text=model.tokenizer(gt, padding='longest', truncation=True, max_length=config['max_length'], return_tensors="pt")
                text.input_ids[:,0] = model.tokenizer.bos_token_id
                for gt in text['input_ids']:
                    gt=model.tokenizer.decode(gt, skip_special_tokens=True)    
                gts[str(row["idx"])] = [gt]
                #########################################################
                if i % print_freq ==0:
                # if i in samples:
                    pred=res[str(row["idx"])][0]
                    print('Q: ',row["Q"])
                    print('gt: ',gt)
                    print('pred: ',pred)
                    print(len(gt),len(pred))
                    print('-------------------')
            result_file = save_result(vqa_result, args.result_dir, 'vqa_result'+str(epoch))
            mp = eval(gts,res)
            print("epoch {} Validation:".format(epoch))
            print(list(mp.items())[:3])
            
            log_stats = {'epoch': epoch,}
            for k, v in list(mp.items())[:3]:
                if isinstance(v,list):
                    for i,vv in enumerate(v):
                        log_stats.update({f'{k}_{i+1}':vv})
                else:
                    log_stats.update({k:v})
                
            with open(os.path.join(args.output_dir, "val_summary.csv"), mode='a') as cf:
                dw = csv.DictWriter(cf, fieldnames=log_stats.keys())
                # if write_header:  # first iteration (epoch == 1 can't be used)
                if epoch==0:
                    dw.writeheader()
                dw.writerow(log_stats)   
               
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
if __name__ == '__main__':
    from datetime import datetime,timedelta
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/fa_rep_multi.yaml') 
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=0, type=bool)
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    now = datetime.now()
    yaml = YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))
    args.output_dir = os.path.join(args.output_dir,config['dataset'],now.strftime("%m-%d-%H%M"))
    print(args.output_dir)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config['output_dir'] = args.output_dir
    config['sp'] =''
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    df=pd.read_csv('../csv_data/all.csv')
    df['Q']='Report'
    df['orgid']=df['folder']
    df['eid']=df['orgid']
    df['idx']=df.index.astype(str)
    columns=['Impression', 'HypoF_ExtraFovea','HypoF_Y', 'HyperF_Y','HyperF_ExtraFovea', 'Vascular abnormality (DR)','Pattern',
        'HyperF_Type', 'HyperF_Area(DA)', 'HyperF_Fovea', 'HypoF_Type', 'HypoF_Area(DA)', 'HypoF_Fovea', 'CNV']
    new_df = pd.DataFrame({'Report': df.apply(lambda row: ', '.join([f'{col}: {row[col]}' for col in columns]), axis=1)})
    df['ANS']=new_df['Report']
    print(len(df))
    main(args, config, df)
