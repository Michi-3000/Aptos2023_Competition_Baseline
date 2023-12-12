import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np
import random
import pandas as pd

# modify for transformation for vit
# modfify wider crop-person images
def encode_label(label, classes_list ): #encoding the classes into a tensor of shape (N classes) with 0 and 1s.
    target = torch.zeros(len(classes_list))
    for l in label:
        idx = classes_list.index(l)
        target[idx] = 1
    return target

class DataSet(Dataset):
    def __init__(self,
                df,
                augs,
                img_size,
                dataset,
                split,
                args
                ):
        self.dataset = dataset
        self.split = split
        self.max_img_num = args.max_img_num
        self.augment = self.augs_function(augs, img_size)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ] 
            # In this paper, we normalize the image data to [0, 1]
            # You can also use the so called 'ImageNet' Normalization method
        )
        self.img_size = img_size
        self.anns = df
        self.groups = df.drop_duplicates('orgid').reset_index(drop=True)
        self.classes=args.classes
        print(self.augment)
        # print(self.classes)
        if (split=='train') and isinstance(args.freq_dt,pd.DataFrame):
            freqs = []
            for c in self.classes:
                #print(c)
                #print(args.freq_dt[args.freq_dt["words"]==c]['freq'].to_list()[0])
                
                freqs.append(args.freq_dt[args.freq_dt["words"]==c]['freq'].to_list()[0])
            mi = np.median(np.array(freqs))
            #print(mi)
            self.weights = mi/np.array(freqs)
        else:self.weights =np.array([1]*len(args.classes))
        print('***********************',self.groups.shape,'*************************')
            
        #print(self.weights)    

    def augs_function(self, augs, img_size):            
        t = []
        if 'randomflip' in augs:
            # t.append(transforms.RandomHorizontalFlip()) # 左右眼会反
            t.append(transforms.RandomVerticalFlip())
        if 'ColorJitter' in augs:
            t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
        if 'resizedcrop' in augs:
            t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.5)))
        if 'RandAugment' in augs:
            t.append(RandAugment())

        t.append(transforms.Resize((img_size, img_size)))

        return transforms.Compose(t)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        # idx = idx % len(self)
        s=self.anns.loc[self.anns.orgid==self.groups.loc[idx,'orgid']]
        s=s.drop_duplicates()
        paths = s['impath'].values.tolist()

        cpaths = []
        if len(paths)>=self.max_img_num:
            if self.split=='Train':
                n = random.randint(1,self.max_img_num)
            else:
                n=self.max_img_num
            cpaths = random.sample(paths, n)
        else:
            cpaths = paths
            
        imgs=[]
        for i, path in enumerate(cpaths):
            img = Image.open(path).convert('RGB')
            img = self.augment(img)
            img = self.transform(img)
            imgs.append(img)
        
        images = torch.stack(imgs)
        if len(imgs) != self.max_img_num:
            padded = torch.zeros([self.max_img_num, 3, self.img_size, self.img_size])
            padded[:len(cpaths),:,:,:]=images
            #print(images.shape,padded.shape)
            images=padded
        #img = Image.open(self.anns.loc[idx,"impath"]).convert("RGB")

        target = self.groups.loc[idx, self.classes]
        message = {
            "orgid": self.groups.loc[idx,'orgid'],
            "target": torch.Tensor(target),
            "img": images,
            "weights":self.weights
        }
        # print(message)
        # print(message['target'].shape)
        return message