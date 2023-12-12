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
                args
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

        # in wider dataset we use vit models
        # so transformation has been changed
        # if self.dataset == "wider":
        #     self.transform = transforms.Compose(
        #         [
        #             transforms.ToTensor(),
        #             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #         ] 
        #     )        

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
    
    # def load_anns(self):
    #     self.anns = []
    #     for ann_file in self.ann_files:
    #         json_data = json.load(open(ann_file, "r"))
    #         self.anns += json_data

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        idx = idx % len(self)
        # ann = self.anns[idx]
        
        img = Image.open(self.anns.loc[idx,"impath"]).convert("RGB")
        target=self.anns.loc[idx,["Modality",'eye']]
        target=encode_label(target,self.classes)

        # if self.dataset == "wider":
        #     x, y, w, h = ann['bbox']
        #     img_area = img.crop([x, y, x+w, y+h])
        #     img_area = self.augment(img_area)
        #     img_area = self.transform(img_area)
        #     message = {
        #         "img_path": ann['img_path'],
        #         "target": torch.Tensor(ann['target']),
        #         "img": img_area
        #     }
        # else: # voc and coco
        img = self.augment(img)
        img = self.transform(img)
        message = {
            "img_path": self.anns.loc[idx,"impath"],
            "target": torch.Tensor(target),
            "img": img
        }

        return message
        # finally, if we use dataloader to get the data, we will get
        # {
        #     "img_path": list, # length = batch_size
        #     "target": Tensor, # shape: batch_size * num_classes
        #     "img": Tensor, # shape: batch_size * 3 * 224 * 224
        # }
