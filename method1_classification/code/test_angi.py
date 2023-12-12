from show import *
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pipeline.resnet_csra import ResNet_CSRA, ResNet_CSRA_multi
from pipeline.dataset_multi import DataSet
from utils.evaluation.eval import evaluation, print_metrics
from utils.evaluation.warmUpLR import WarmUpLR
from tqdm import tqdm

import os
from collections import OrderedDict
import csv
    
def val(epoch, args, model, test_loader, test_df,output_dir):
    model.eval()
    print("Test on Epoch {}".format(epoch))
    # result_list = []
    result_list_pred = []
    result_scores = []
    ids=[]
    targets=[]
    # calculate logit
    for index, data in enumerate(tqdm(test_loader)):
        img = data['img'].cuda()
        # target = data['target'].cuda()
        idx= data['orgid']

        with torch.no_grad():
            logit = model(img)

        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()
        for k in range(len(idx)):
            res = ""
            scores = result[k]
            orgid=idx[k]
            for i,s in enumerate(scores):
                if s>0.5:
                    if res == "":
                        res+=args.classes[i]
                    else:
                        res+=','+args.classes[i]
            result_list_pred.append(res)
            result_scores.append(scores)
            ids.append(orgid)
            print(orgid)
            #targets.append(data['target'][k].cpu().numpy())

    df1=pd.DataFrame()
    df1['orgid']=ids
    df1["Cls"]=result_list_pred
    df1['scores']=result_scores

    df1=pd.merge(df1,df[['orgid']],).drop_duplicates('orgid')

    return df1

def main(df,args,mp,epoch):
    loadfrom=f'{mp}/ep{epoch}.pth'
    w = torch.load(loadfrom)
    print(w.keys())
    args.classes=w['classes']
    num_cls=len(args.classes)
    if args.model == "resnet101": 
        model = ResNet_CSRA_multi(num_heads=args.num_heads, lam=args.lam, num_classes=num_cls, cutmix=args.cutmix)
    model.load_state_dict(w['state_dict'])   
    model.cuda()
    print("Successfully load weight from {mp}/ep{epoch}.pth")
    test_dataset = DataSet(df, args.test_aug, args.img_size, args.dataset,'test',args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    df1=val(epoch, args, model, test_loader, df,mp)
    return df1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="resnet101")
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam",default=0.1, type=float)
    parser.add_argument("--cutmix", default=None, type=str) # the path to load cutmix-pretrained backbone
    # dataset
    parser.add_argument("--dataset", default="fundus", type=str)
    parser.add_argument("--test_aug", default=[], type=list)
    # parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--img_size", default=384, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    # optimizer, default SGD
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--w_d", default=0.0001, type=float, help="weight_decay")
    parser.add_argument("--classes", default=None, type=list)

    args = parser.parse_args()

    mp = '../checkpoints'
    epoch=13
    split='test'
    df=pd.read_csv(os.path.join(mp,'valid.csv'))
    df2 = pd.read_csv(os.path.join(mp,'test.csv'))
    df = pd.concat([df, df2])
    print(df.head())
    print(len(df.drop_duplicates('folder')))
    
    out=os.path.join(mp,split)
    mkfile(out)
    
    for n in range(12, 13):
        args.max_img_num = n
        df1=main(df,args,mp,epoch)
        df1.to_csv(os.path.join(out,str(epoch)+'-'+str(n)+'_pred.csv'))
        
