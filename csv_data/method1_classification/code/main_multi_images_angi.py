import pandas as pd
import argparse
import time
from sqlalchemy import exists
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pipeline.resnet_csra import ResNet_CSRA, ResNet_CSRA_multi
# from pipeline.vit_csra import VIT_B16_224_CSRA, VIT_L16_224_CSRA, VIT_CSRA
from pipeline.dataset_multi import DataSet
from utils.evaluation.eval import evaluation
from utils.evaluation.warmUpLR import WarmUpLR
from show import *
from lg_utils import *

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from collections import OrderedDict
import csv
# modify for wider dataset and vit models

def train(i, args, model, train_loader, optimizer, warmup_scheduler):
    print()
    model.train()
    epoch_begin = time.time()
    for index, data in enumerate(train_loader):
        batch_begin = time.time() 
        img = data['img'].cuda()
        target = data['target'].cuda()
        weight = data['weights'].cuda()
        #print(data['img_path'])

        optimizer.zero_grad()
        logit, loss = model(img, target, weight)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        t = time.time() - batch_begin

        if index % args.print_freq == 0:
            print("Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, time:{:.4f}".format(
                i, 
                args.batch_size * (index + 1),
                len(train_loader.dataset),
                loss,
                optimizer.param_groups[0]["lr"],
                float(t)
            ))

        if warmup_scheduler and i <= args.warmup_epoch:
            warmup_scheduler.step()
        
    
    t = time.time() - epoch_begin
    print("Epoch {} training ends, total {:.2f}s".format(i, t))


def val(i, args, model, test_loader, test_df,output_dir):
    model.eval()
    print("Test on Epoch {}".format(i))
    result_list = []
    results=[]
    targets=[]
    # calculate logit
    for index, data in enumerate(test_loader):
        img = data['img'].cuda()
        # target = data['target'].cuda()
        orgid = data['orgid']

        with torch.no_grad():
            logit = model(img)

        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()
        for k in range(len(orgid)):
            result_list.append(
                {
                    "orgid": orgid,#[k].split("/")[-1].split(".")[0],
                    "scores": result[k],
                    'target': data['target'][k]
                })
            results.append(result[k])
            targets.append(data['target'][k])
            
            
    # cal_mAP OP OR
    res=evaluation(targets, results,args.classes)
    filename= os.path.join(output_dir, 'summary.csv')
    rowd = OrderedDict(epoch=i)
    rowd.update([(k, v) for k, v in res.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        # if write_header:  # first iteration (epoch == 1 can't be used)
        if i==1:
            dw.writeheader()
        dw.writerow(rowd)
def splitdf(df,col='eid',test_size=.1):
    from sklearn.model_selection import GroupShuffleSplit
    train_inds, test_inds = next(GroupShuffleSplit(test_size=test_size, n_splits=2, random_state = 7).split(df, groups=df[col]))
    train = df.iloc[train_inds].reset_index(drop=True)
    test = df.iloc[test_inds].reset_index(drop=True)
    return train,test
def main(df,args):
    print('Number of classes:',len(args.classes))
    print(args.classes)
    args.num_cls=len(args.classes)
    # model
    if args.model == "resnet101": 
        model = ResNet_CSRA_multi(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls, cutmix=args.cutmix)
    if args.checkpoint:
        model_dict = model.state_dict()
        w = torch.load(args.checkpoint)['state_dict']
        # pretrained_dict = {k: v for k, v in w.items() if k in model_dict}
        pretrained_dict = { k:v for k,v in w.items() if k in model_dict and v.size() == model_dict[k].size() }
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        # model.load_state_dict(w,strict=False)
        print('load from:',args.checkpoint)
        
    model.cuda()
    step_size = 4

    if 'split' in df.columns:
        TRAIN=df[df.split=='Train'].reset_index(drop=True)
        VALID=df[df.split=='Validation'].reset_index(drop=True)
        TEST=df[df.split=='Test'].reset_index(drop=True)
    else:
        pid='orgid'
        TRAIN,TEST=splitdf(df ,pid,.3)
        VALID,TEST=splitdf(TEST ,pid,2/3)
        # valid_df=TEST

    print('***** TRAIN:', len(TRAIN),'***** VALID:', len(VALID),'***** TEST:', len(TEST),)
    from datetime import datetime
    now = datetime.now()
    out =  "./checkpoint/"+args.model+'-'+now.strftime("%m-%d-%H%M")+'/'
    os.makedirs(out,exist_ok=True)
    TEST.to_csv(out+'test.csv')
    VALID.to_csv(out+'valid.csv')

    train_dataset = DataSet(TRAIN, args.train_aug, args.img_size, args.dataset, "train", args)
    test_dataset = DataSet(VALID, args.test_aug, args.img_size, args.dataset,"test", args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # optimizer and warmup
    backbone, classifier = [], []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier.append(param)
        else:
            backbone.append(param)
    optimizer = optim.SGD(
        [
            {'params': backbone, 'lr': args.lr},
            {'params': classifier, 'lr': args.lr * 10}
        ],
        momentum=args.momentum, weight_decay=args.w_d)    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    iter_per_epoch = len(train_loader)
    if args.warmup_epoch > 0:
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warmup_epoch)
    else:
        warmup_scheduler = None

    # training and validation

    for i in range(1, args.total_epoch + 1):
        train(i, args, model, train_loader, optimizer, warmup_scheduler)

        torch.save({'state_dict':model.state_dict(),'classes':args.classes}, out+"ep{}.pth".format( i))
        val(i, args, model, test_loader, VALID,out)
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="resnet101")
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam",default=0.1, type=float)
    parser.add_argument("--cutmix", default=None, type=str) # the path to load cutmix-pretrained backbone
    # dataset
    parser.add_argument("--dataset", default="fundus", type=str)
    parser.add_argument("--num_cls", default=10, type=int)
    parser.add_argument("--train_aug", default=["resizedcrop"], type=list) #"randomflip", 
    parser.add_argument("--test_aug", default=[], type=list)
    parser.add_argument("--img_size", default=384, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    # optimizer, default SGD
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--w_d", default=0.0001, type=float, help="weight_decay")
    parser.add_argument("--warmup_epoch", default=2, type=int)
    parser.add_argument("--total_epoch", default=30, type=int)
    parser.add_argument("--print_freq", default=200, type=int)
    parser.add_argument("--classes", default=None, type=list)
    parser.add_argument("--checkpoint", default='')
    parser.add_argument("--target", default='')
    args = parser.parse_args()

    csvp='../../csv_data/all.csv'
    df=pd.read_csv(csvp)
    df['orgid']=df['folder']
    cols = ['HypoF_ExtraFovea','HypoF_Y', 'HyperF_Y','HyperF_ExtraFovea', 'Vascular abnormality (DR)','Pattern',
        'HyperF_Type', 'HyperF_Area(DA)', 'HyperF_Fovea', 'HypoF_Type', 'HypoF_Area(DA)', 'HypoF_Fovea', 'CNV']

    f=df.drop_duplicates('folder')
    f['cap']=f['Impression'] 
    for col in cols:
        f[col]=f[col].fillna('no')
        f['cap']=f['cap'] +','+col+':'+f[col]

    f=f[['cap','folder']]
    classes=[]
    for i in f.index:
        cap=f.loc[i,'cap']
        for c in cap.split(','):
            if c!='':
                f.loc[i,c]=1
                classes.append(c)
    classes=list(set(classes))

    df=pd.merge(df,f,how='left').drop_duplicates('impath')
    print(df.shape,'classes', len(classes))
    args.freq_dt = None
    args.classes=list(classes)
    args.max_img_num = 12
    df[args.classes]=df[args.classes].fillna(0)
    print(df.head())
    main(df,args)
