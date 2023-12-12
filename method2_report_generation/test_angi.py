import pandas as pd
import argparse
import os
from ruamel.yaml import YAML
import numpy as np
import random
import time
# import datetime
# import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_vqa_multi_images import blip_vqa
import utils
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result
from pycocoevalcap.eval import test_eval
import re
from glob import glob
import json
import csv
from vis_utils import *
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

@torch.no_grad()
def evaluation(model, data_loader, device, config,csvp) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = len(data_loader)//3
    
    result = []
    res = {}   
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        
        if config['inference']=='generate':
            answers = model(image, question, train=False, inference='generate',num_beams=config['num_beams'], max_length=config['max_length']) 
            for answer, ques_id in zip(answers, question_id):
                ques_id = int(ques_id.item())       
                # result.append({"question_id":ques_id, "answer":answer,})
                res[str(ques_id)]=[answer]
                with open(csvp, mode='a') as f:
                    row={'idx':str(ques_id),'ans':answer}
                    dw = csv.DictWriter(f, fieldnames=row.keys())
                    dw.writerow(row)   
    return result, res

def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    if type(args.which_epoch)==int:
        loadfrom=os.path.join(args.ckpt_dir,'checkpoint_%02d.pth'%args.which_epoch)
    else:
        loadfrom=sorted(glob(os.path.join(args.ckpt_dir,'*.pth')))[-1]
        args.which_epoch = int(re.findall('_(\d+).pth',loadfrom)[-1])
        
    csvp=os.path.join(args.result_dir, 'vqa_result'+str(args.which_epoch)+'-'+str(config['max_img_num'])+'.csv')
    if args.testcsv:
        testcsv=args.testcsv
    dt1 = pd.read_csv(os.path.join(args.ckpt_dir,"test.csv"))
    dt2 = pd.read_csv(os.path.join(args.ckpt_dir,"valid.csv"))
    TEST = pd.concat([dt1, dt2])
    print('read csv...')   
    print(TEST.shape,'----------------------------------------------------------------->')
    
    print(TEST.sample(5).Q.value_counts())
    
    if not args.saved_report:
        if os.path.isfile(csvp):
            fi = pd.read_csv(csvp,header=None)
  
    if not args.saved_report:
        datasets = create_dataset(TEST[:1],TEST,config)
        #print(datasets)  
        samplers = [None, None]
        _, test_loader = create_loader(datasets,samplers,
                                                batch_size=[config['batch_size_train'],config['batch_size_test']],
                                                num_workers=[4,4],is_trains=[True, False], 
                                                collate_fns=[vqa_collate_fn,None],args=args) 
    #### Model #### 

    model = blip_vqa(pretrained=loadfrom, image_size=config['image_size'], 
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                       lan=config['language'],
                       config=config)
    
    model = model.to(device)   
    
    start_time = time.time()  
    if not args.saved_report:
        model_without_ddp = model
          
        vqa_result,res = evaluation(model_without_ddp, test_loader, device, config,csvp)  
    else:
        resr = pd.read_csv(csvp,header=None)
        resr.columns=['idx','answer']
        print('read saved predctions from:', resr)
        
        if args.debug:
            resr =resr[:100]   
        res={str(idx):[v] for idx,v in resr[['idx','answer']].values}
        
    gts={}
    gts_lens = []
    res_lens = []           
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str)) 
    
if __name__ == '__main__':
    from datetime import datetime,timedelta
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=0, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=0, type=bool)
    parser.add_argument('--ckpt_dir', default='')
    parser.add_argument("--local_rank", type=int, help="")
    parser.add_argument('--which_epoch', default=None)
    parser.add_argument('--task_type', default="open", help='open/multi_choice/yesno')
    parser.add_argument('--split', default="test")
    parser.add_argument('--debug', default=0)
    parser.add_argument('--saved_report', default=0)
    parser.add_argument('--testcsv', default=None)
    args = parser.parse_args()
    args.ckpt_dir='./checkpoints'
    args.which_epoch=13

    
    config=os.path.join(args.ckpt_dir,'config.yaml')
    yaml = YAML(typ='rt')
    config = yaml.load(open(config, 'r'))
    
    print(args.ckpt_dir)
    print(config)
    args.result_dir = os.path.join(args.ckpt_dir, args.split)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    if 'sp' not in config.keys():
        config['sp']=''
    config['output_dir'] = args.result_dir
    
    main(args, config)
