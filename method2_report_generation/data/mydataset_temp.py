
from torch.utils.data import Dataset
from PIL import Image
import torch
import cv2
from data.utils import pre_question,pre_caption
import random

class fa_dataset(Dataset):

    def __init__(self, df,impath,q,a,transform=None,  split='train'):

        self.df = df
        print('************************',split , df.shape,'********************************')
        # self.paths = df.dir.tolist()
        self.paths = df[impath].tolist()
        self.qs = df[q].tolist()
        self.captions = df[a].tolist()

        self.transform = transform
        self.split=split
        self.im_name = ""
        self.img = None

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        cap = self.captions[index]
        cap = pre_caption(cap,max_words=150)
        q = self.qs[index]
        q = pre_question(q,max_ques_words=35)
        path = self.paths[index]
        if path==self.im_name:
            #print("save!")
            if self.split == 'test':           
                # return self.transform(self.img), q, self.df.loc[index,'idx']
                return self.img, q, self.df.loc[index,'idx']
            elif self.split=='train': 
                # cap = pre_caption(cap,50)
                answers = [cap]
                weights = [0.2]
                #print(self.im_name)
                # return self.transform(self.img), q, answers,weights,path
                return self.img, q, answers,weights,path
        
        try:
            image=cv2.imread(path)[:,:,1]

            image=cv2.merge([image]*3)
            image=Image.fromarray(image)
        except:print('*****************',path)
        # if self.transform is not None:
        image = self.transform(image)
        self.im_name = path
        self.img=image
        #answers = [cap]
        if self.split == 'test':           
            return image, q, self.df.loc[index,'idx']

        elif self.split=='train': 
            # cap = pre_caption(cap,50)
            answers = [cap]
            weights = [0.2]  
            return image, q, answers,weights,path

    def __len__(self):
        return len(self.df)
    
class ultrasound_dataset(Dataset):

    def __init__(self, df,impath,q,a,transform=None,  split='train'):

        self.df = df
        print('************************',split , df.shape,'********************************')
        # self.paths = df.dir.tolist()
        self.paths = df[impath].tolist()
        self.qs = df[q].tolist()
        self.captions = df[a].tolist()

        self.transform = transform
        self.split=split
        self.im_name = ""
        self.img = None
        
    def __getitem__(self, index):
        cap = self.captions[index]
        q = self.qs[index]
        path = self.paths[index]
        cap = pre_caption(cap,max_words=150)
        q = self.qs[index]
        q = pre_question(q,max_ques_words=35)
        if path==self.im_name:
            #print("save!")
            if self.split == 'test':           
                return self.transform(self.img), q, self.df.loc[index,'idx']
                #return self.img, q, self.df.loc[index,'idx']
            elif self.split=='train': 
                # cap = pre_caption(cap,50)
                answers = [cap]
                weights = [0.2]
                #print(self.im_name)
                return self.transform(self.img), q, answers,weights,path
                #return self.img, q, answers,weights,path
        try:
            image=cv2.imread(path)[:,:,1]
            if self.split=='train':
                aug=random.choice([0,1,2])
                if aug:
                    h,w=image.shape
                    if aug==2:
                        image = image[:,w//2:]
                    else:
                        image = image[:,:w//2]
            image=cv2.merge([image]*3)
        except:print('*****************',path)
        image=Image.fromarray(image)
        #image = self.transform(image)
        self.im_name = path
        self.img=image
        image = self.transform(image)
        
        if self.split == 'test':           
            return image, q, self.df.loc[index,'idx']

        elif self.split=='train': 
            # cap = pre_caption(cap,50)
            answers = [cap]
            weights = [0.2]
            #print(self.im_name)
            return image, q, answers,weights,path




    def __len__(self):
        return len(self.df)
def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n        

