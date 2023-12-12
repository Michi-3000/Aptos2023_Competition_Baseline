import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np

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
                args,
                split
                ):
        self.dataset = dataset
        self.augment = self.augs_function(augs, img_size)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ] 
            # In this paper, we normalize the image data to [0, 1]
            # You can also use the so called 'ImageNet' Normalization method
        )
        self.anns = df
        self.classes=args.classes
        print(self.augment)
        print(self.classes)
        if split=='train':
            freqs = []
            for c in self.classes:
                #print(c)
                #print(args.freq_dt[args.freq_dt["words"]==c]['freq'].to_list()[0])
                freqs.append(args.freq_dt[args.freq_dt["words"]==c]['freq'].to_list()[0])
            mi = np.median(np.array(freqs))
            #print(mi)
            self.weights = mi/np.array(freqs)
            #print(self.weights)    
        else:self.weights = None

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
        return len(self.anns)

    def __getitem__(self, idx):
        idx = idx % len(self)
        # ann = self.anns[idx]
        
        img = Image.open(self.anns.loc[idx,"impath"]).convert("RGB")
        target = self.anns.loc[idx,self.classes]
        #print("target")
        #print(target)

        img = self.augment(img)
        img = self.transform(img)
        message = {
            "img_path": self.anns.loc[idx,"impath"],
            "target": torch.Tensor(target),
            "img": img,
            "weights":self.weights
        }

        return message

