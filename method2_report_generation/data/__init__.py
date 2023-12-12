import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
# from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval
# from data.nocaps_dataset import nocaps_eval
# from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
# from data.vqa_dataset import vqa_dataset

# from data.nlvr_dataset import nlvr_dataset
from data.pretrain_dataset import pretrain_dataset
from data.mydataset import *
from transform.randaugment import RandomAugment
import random


def shuffle_dataset(df):
    # import pandas as pd
    imgs = list(set(df["impath"].to_list()))
    random.shuffle(imgs)
    #print("dflen", len(df))
    #print("imglen", len(imgs))
    #print(imgs[:10])
    df_r = pd.DataFrame()
    for i in imgs:
        t = df[df["impath"]==i]
        df_r = pd.concat([df_r, t])
    #print("dflen", len(df_r))
    return df_r

def create_dataset(traindf,testdf, config, min_scale=0.5,trainf=None):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(int(config['image_size']),scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            # transforms.RandomHorizontalFlip(), #左右眼问题
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
    datasetname=config['dataset']
    #print(datasetname)
    if datasetname=='pretrain':
        #dataset = pretrain_dataset(config['train_file'], config['laion_path'], transform_train)              
        dataset = pretrain_dataset([trainf], config['laion_path'], transform_train)
        return dataset  
    
    elif datasetname=='bscan': 
        train_dataset = ultrasound_dataset(df=traindf,config=config,transform=transform_train,split='train') 
        test_dataset = ultrasound_dataset(df=testdf,config=config,transform=transform_test,split= 'test') 
        return train_dataset, test_dataset
    elif datasetname == 'bscan_multi':
        train_dataset = ultrasound_multi_dataset(df=traindf,config=config,transform=transform_train,split='train') 
        test_dataset = ultrasound_multi_dataset(df=testdf,config=config,transform=transform_test,split= 'test') 
        return train_dataset, test_dataset
    elif datasetname in ['fa_multi']:
        train_dataset = fa_multi_dataset(df=traindf,config=config,transform=transform_train,split='train') 
        test_dataset = fa_multi_dataset(df=testdf,config=config,transform=transform_test,split= 'test') 
        return train_dataset, test_dataset
    elif datasetname in ['icg_multi']:
        train_dataset = icg_multi_dataset(df=traindf,config=config,transform=transform_train,split='train') 
        test_dataset = icg_multi_dataset(df=testdf,config=config,transform=transform_test,split= 'test') 
        return train_dataset, test_dataset        
    
    elif datasetname in ['slit_multi']:
        train_dataset = slit_multi_dataset(df=traindf,config=config,transform=transform_train,split='train') 
        test_dataset = slit_multi_dataset(df=testdf,config=config,transform=transform_test,split= 'test') 
        return train_dataset, test_dataset      
    else:
        train_dataset = fa_dataset(df=traindf,config=config,transform=transform_train,split='train') 
        test_dataset = fa_dataset(df=testdf,config=config,transform=transform_test,split= 'test') 
        return train_dataset, test_dataset
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns,args):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = True #(sampler is None)# mod0529
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        if args.distributed:
            shuffle = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

