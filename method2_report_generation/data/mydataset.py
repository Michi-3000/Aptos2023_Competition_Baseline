from torch.utils.data import Dataset
from PIL import Image
import torch
import cv2
from data.utils import pre_question,pre_caption
import random
import os
# from models.blip import create_vit
import numpy as np

class fa_dataset(Dataset):

    def __init__(self, df,config,transform=None,  split='train'):

        self.df = df
        print('************************',split , df.shape,'********************************')
        # self.paths = df.dir.tolist()
        self.paths = df[config['impath']].tolist()
        self.qs = df[config['Q']].tolist()
        self.captions = df[config['ANS']].tolist()

        self.transform = transform
        self.split=split
        self.max_length=config['max_length']
        self.max_q_length=config['max_q_length']
        #self.im_name = ""
        #self.img = None

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        cap = self.captions[index]
        try:cap = pre_caption(cap,max_words=self.max_length)
        except Exception as e:print(e)
        q = self.qs[index]
        q = pre_question(q,max_ques_words=self.max_q_length)
        path = self.paths[index]
        try:
            image=cv2.imread(path)#[:,:,1]

            # image=cv2.merge([image]*3)
            image=Image.fromarray(image)
        except:print('*****************',path)
        # if self.transform is not None:
        image = self.transform(image)
        #answers = [cap]
        if self.split == 'test':   
            # print(self.df.loc[index,'idx'])        
            return image, q, self.df.loc[index,'idx']

        elif self.split=='train': 
            # cap = pre_caption(cap,50)
            answers = [cap]
            weights = [0.2]  
            return image, q, answers,weights,path

    def __len__(self):
        return len(self.df)

class fa_multi_dataset(Dataset):
    def __init__(self, df,config,transform=None,  split='train'):
        self.df = df
        sdf=df.drop_duplicates('eid')
        if split=='train':         
            # sdf=sdf.append(sdf[sdf.qtype=='Report_CN'])
            # print(sdf.qtype.value_counts())
            #sdf=sdf.sample(frac=1)
            
            print(sdf.Q.value_counts())
            # pass
        else:
            #sdf=sdf.sample(frac=1)
            sdf[['orgid','idx','Q','ANS']].to_csv(os.path.join(config['output_dir'],split+'_mini.csv'))
        self.sdf=sdf.reset_index(drop=True)
        self.transform = transform
        self.split=split
        self.max_length=config['max_length']
        self.max_q_length=config['max_q_length']
        self.max_img_num = config['max_img_num']
        self.config=config
        print('************************',split , df.shape,self.sdf.shape, '********************************')

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        cap = self.sdf.loc[index,'ANS']
        # self.max_img_num=random.randint(6,13) 
        try:cap = pre_caption(cap,max_words=self.max_length)
        except Exception as e:print(e)
        q = self.sdf.loc[index,'Q']
        q = pre_question(q,max_ques_words=self.max_q_length)
        eid=self.sdf.loc[index,'eid']
        s=self.df.loc[self.df.eid==eid]
        
        try:
            s = s.groupby(['Phase','P']).sample(3,replace=True)
        except:
            pass
        s=s.drop_duplicates()
        # print(s.shape)
        paths = s['impath'].values.tolist()
        cpaths = []
        # print(len(s),s.Phase)
        if len(paths)>=self.max_img_num:
            if self.split=='train':
                n = random.randint(1,self.max_img_num)
            else:
                n=self.max_img_num
            cpaths = random.sample(paths, n)
        else:
            cpaths = paths
            
        imgs=[]
        
        for i, path in enumerate(cpaths):
            image=Image.open(path).convert('RGB')
            imgs.append(self.transform(image))
            
        # test=[]
        # for i, path in enumerate(cpaths):
        #     test.append(cv2.resize(cv2.imread(path),(512,512)))
        # test=np.vstack(test)   
        # cv2.imwrite('/home/danli/caption/BLIP/output/fa_multi/inputs/'+eid+'.jpg',test)
        
        images = torch.stack(imgs)
        if len(imgs) != self.max_img_num:
            padded = torch.zeros([self.max_img_num,3,self.config['image_size'],self.config['image_size']])
            padded[:len(cpaths),:,:,:]=images
            # print(images.shape,padded.shape)
            images=padded
            
        if self.split == 'test':     
            return images, q, self.sdf.loc[index,'idx']

        elif self.split=='train': 
            answers = [cap]
            weights = [0.2]
            #print(q, answers, path)
            return images, q, answers,weights,path

    def __len__(self):
        return len(self.sdf)
    
class icg_multi_dataset(Dataset):
    def __init__(self, df,config,transform=None,  split='train'):
        df['eid']=df['orgid']+df['Q']
        self.df = df
        sdf=df.drop_duplicates('eid').reset_index(drop=True)
        if split=='train':
            d1=['drusen','retinitis pigmentosa',
            'degeneration', 'hemorrhage', 'detachment',
                'lacquer crack', 'scar','vascular hamartoma',
            'myopia', 'Vogt-Koyanagi-Harada disease',
            'central serous chorioretinopathy','unremarkable','polypoidal vascular abnormality','choroidal neovascularization']
            sdf['Main']=np.nan
            sdf['main']=sdf['Dis'].str.extract('('+'|'.join(d1)+')')
            sdf=sdf.append(sdf[~sdf['main'].isnull()].sample(frac=2,replace=True)) 
            
            d2=['Von Hippel','dystrophy','crystalline','sympathetic','Stargard','choroidal nevus','choroidal mass',
                'vitelliform','coloboma','congenital hypertrophy']
            sdf['Main']=np.nan
            sdf['main']=sdf['Dis'].str.extract('('+'|'.join(d2)+')')
            sdf=sdf.append(sdf[~sdf['main'].isnull()].sample(frac=5,replace=True)) 
        else:
            sdf[['orgid','idx','Q','ANS']].to_csv(os.path.join(config['output_dir'],split+'_mini.csv'))
        self.sdf=sdf.reset_index(drop=True)
        self.transform = transform
        self.split=split
        self.max_length=config['max_length']
        self.max_q_length=config['max_q_length']
        self.max_img_num = config['max_img_num']
        self.config=config
        print('************************',split , df.shape,self.sdf.shape, '********************************')

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        cap = self.sdf.loc[index,'ANS']
        # self.max_img_num=random.randint(6,13) 
        try:cap = pre_caption(cap,max_words=self.max_length)
        except Exception as e:print(e)
        q = self.sdf.loc[index,'Q']
        q = pre_question(q,max_ques_words=self.max_q_length)
        s=self.df.loc[self.df.eid==self.sdf.loc[index,'eid']]
        s = s.groupby(['Phase']).sample(3,replace=True)
        s=s.drop_duplicates()
        paths = s['impath'].values.tolist()
        cpaths = []
        # print(len(s),s.Phase)
        if len(paths)>=self.max_img_num:
            if self.split=='train':
                n = random.randint(1,self.max_img_num)
            else:
                n=self.max_img_num
            cpaths = random.sample(paths, n)
        else:
            cpaths = paths
            
        imgs=[]
        for i, path in enumerate(cpaths):
            image=Image.open(path).convert('RGB')
            imgs.append(self.transform(image))
        images = torch.stack(imgs)
        if len(imgs) != self.max_img_num:
            padded = torch.zeros([self.max_img_num,3,self.config['image_size'],self.config['image_size']])
            padded[:len(cpaths),:,:,:]=images
            # print(images.shape,padded.shape)
            images=padded
            
        if self.split == 'test':     
            return images, q, self.sdf.loc[index,'idx']

        elif self.split=='train': 
            answers = [cap]
            weights = [0.2]
            #print(q, answers, path)
            return images, q, answers,weights,path
    def __len__(self):
        return len(self.sdf)

class slit_multi_dataset(Dataset):
    def __init__(self, df,config,transform=None,  split='train'):
        # df['eid']=df['orgid']+df['Q']
        self.df = df
        df['loc']=df['loc'].fillna('cornea')
        sdf=df.drop_duplicates('eid').reset_index(drop=True)
        # sdf[['orgid','idx','Q','ANS']].to_csv(os.path.join(config['output_dir'],split+'_mini.csv'))
        sdf.to_csv(os.path.join(config['output_dir'],split+'_mini.csv'))
            
        self.sdf=sdf.reset_index(drop=True)
        self.transform = transform
        self.split=split
        self.max_length=config['max_length']
        self.max_q_length=config['max_q_length']
        self.max_img_num = config['max_img_num']
        self.config=config
        print('************************ slit dataset',split , df.shape,self.sdf.shape, '********************************')

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        cap = self.sdf.loc[index,'ANS']
        # self.max_img_num=random.randint(6,13) 
        try:cap = pre_caption(cap,max_words=self.max_length)
        except Exception as e:print(e)
        q = self.sdf.loc[index,'Q']
        q = pre_question(q,max_ques_words=self.max_q_length)
        s=self.df.loc[self.df.eid==self.sdf.loc[index,'eid']]
        
        s = s.groupby(['eye','loc']).sample(1,replace=True)
        # print(s)
        s=s.drop_duplicates()
        print(s.shape)
        paths = s['impath'].values.tolist()
        cpaths = []
        # print(len(s),s.Phase)
        if len(paths)>=self.max_img_num:
            n=self.max_img_num
            # if self.split=='train':
            #     n = random.randint(1,self.max_img_num)
                
            cpaths = random.sample(paths, n)
        else:
            cpaths = paths
            
        imgs=[]
        for i, path in enumerate(cpaths):
            image=Image.open(path).convert('RGB')
            imgs.append(self.transform(image))
        images = torch.stack(imgs)
        if len(imgs) != self.max_img_num:
            padded = torch.zeros([self.max_img_num,3,self.config['image_size'],self.config['image_size']])
            padded[:len(cpaths),:,:,:]=images
            # print(images.shape,padded.shape)
            images=padded
            
        if self.split == 'test':     
            return images, q, self.sdf.loc[index,'idx']

        elif self.split=='train': 
            answers = [cap]
            weights = [0.2]
            #print(q, answers, path)
            return images, q, answers,weights,path

    def __len__(self):
        return len(self.sdf)
    
# class lt_multi_dataset2(Dataset):
    
#     def __init__(self, df,config,transform=None,  split='train'):
        
#         self.df = df
#         print('************************',split , df.shape,'********************************')
#         all_imgs = []
#         imgs = []
#         all_ans = []
#         all_q = []
#         all_id = []
#         all_idx = []
#         for i, row in df.iterrows():
#             if i==0:
#                 now_ans = row[config['ANS']]
#                 now_q = row[config['Q']]
#                 now_id = row[config['eid']]
#             if (row[config['eid']]!=now_id) or (row[config['Q']]!=now_q):
#                 #print(i)
#                 all_ans.append(row[config['ANS']])
#                 all_q.append(row[config['Q']])
#                 all_imgs.append(imgs)
#                 all_id.append(row[config['eid']])
#                 all_idx.append(row['idx'])
#                 imgs = []
#                 imgs.append(row[config['impath']])
#                 now_ans = row[config['ANS']]
#                 now_q = row[config['Q']]
#                 now_id = row[config['eid']]
#             else:
#                 imgs.append(row[config['impath']])
#         if len(all_imgs) == 0:
#             all_ans.append(row[config['ANS']])
#             all_q.append(row[config['Q']])
#             all_imgs.append([row[config['impath']]])
#             all_id.append(row[config['eid']])
#             all_idx.append(row['idx'])

#         # self.paths = df.dir.tolist()
#         self.paths = all_imgs
#         self.qs = all_q
#         self.captions = all_ans
#         self.idxs = all_idx

#         self.transform = transform
#         self.split=split
#         self.max_length=config['max_length']
#         self.max_q_length=config['max_q_length']
#         self.max_img_num = config['max_img_num']
#         #if split =="test":
#         import pandas as pd
#         TEST = pd.DataFrame()
#         TEST[config['eid']] = all_id
#         TEST[config['ANS']] = all_ans
#         TEST[config['Q']] = all_q
#         TEST[config['impath']] = [str(i) for i in all_imgs]
#         TEST['idx'] = all_idx
#         TEST.to_csv(os.path.join(config['output_dir'],self.split+'_mini.csv'))
#         #self.im_name = ""
#         #self.img = None
#         #self.device = config['device']
#         #self.visual_encoder, vision_width = create_vit(config['vit'], config['image_size'], config['vit_grad_ckpt'], config['vit_ckpt_layer'], drop_path_rate=0.1)
#         #self.visual_encoder.to(config['device'])
#         print('After Processing: ************************',split , len(self.captions),'********************************')

#     def __getitem__(self, index):
#         """Returns one data pair (image and caption)."""
#         cap = self.captions[index]
#         try:cap = pre_caption(cap,max_words=self.max_length)
#         except Exception as e:print(e)
#         q = self.qs[index]
#         q = pre_question(q,max_ques_words=self.max_q_length)
#         paths = self.paths[index]
#         cpaths = []
#         if len(paths)>=self.max_img_num:
#             cpaths = random.sample(paths, self.max_img_num)
#         else:
#             cpaths = paths
#         #print(cpaths)
#         '''
#         cpaths = []
#         if len(paths)>=self.img_num:
#             cpaths = random.sample(paths, self.img_num)
#         else:
#             cpaths = paths
#             #print(len(cpaths))
#             while len(cpaths)<self.img_num:
#                 cpaths.append(random.choice(paths))
#         '''
#         for i, path in enumerate(cpaths):
#             if i==0:
#                 try:
#                     image=cv2.imread(path)
#                     image=Image.fromarray(image)
#                 except:print('*****************',path)
#                 # if self.transform is not None:
#                 image = self.transform(image)
#                 images = torch.unsqueeze(image,0)
#             else:
#                 try:
#                     image=cv2.imread(path)
#                     image=Image.fromarray(image)
#                 except:print('*****************',path)
#                 # if self.transform is not None:
#                 image = self.transform(image)
#                 images = torch.cat((images, torch.unsqueeze(image,0)),dim=0)
#         tmp = torch.unsqueeze(torch.zeros(images.shape[1],images.shape[2],images.shape[3]),0)
#         for i in range(self.max_img_num-len(cpaths)):
#             images = torch.cat((images, tmp), dim=0)

#         #NUM = images.shape[1]
#         print(images.shape)
#         #image_ = torch.reshape(image, (image.shape[0]*image.shape[1], image.shape[2], image.shape[3], image.shape[4]))
#         #image_embeds = self.visual_encoder(images)
#         #image_embeds = torch.reshape(image_embeds, (int(image_embeds.shape[0]/NUM), NUM, image_embeds.shape[1], image_embeds.shape[2]))
#         #image_embeds = torch.squeeze(torch.mean(image_embeds, dim=0, keepdim=True),0)
#         #print(image_embeds.shape)
#         #image_embeds = image_embeds.to('cpu')

#         #answers = [cap]
#         if self.split == 'test':   
#             # print(self.df.loc[index,'idx'])
#             #print(q, self.idxs[index])      
#             return images, q, self.idxs[index]#self.df.loc[index,'idx']

#         elif self.split=='train': 
#             # cap = pre_caption(cap,50)
#             answers = [cap]
#             weights = [0.2]
#             #print(q, answers, path)
#             return images, q, answers,weights,path

#     def __len__(self):
#         return len(self.captions)
    
class ultrasound_dataset(Dataset):

    def __init__(self, df,config,transform=None,  split='train',):

        self.df = df
        print('************************',split , df.shape,'********************************')
        # self.paths = df.dir.tolist()
        self.paths = df[config['impath']].tolist()
        self.qs = df[config['Q']].tolist()
        self.captions = df[config['ANS']].tolist()

        self.transform = transform
        self.split=split
        # self.im_name = ""
        # self.img = None
        self.max_length=config['max_length']
        self.max_q_length=config['max_q_length']        
    def __getitem__(self, index):
        cap = self.captions[index]
        q = self.qs[index]
        path = self.paths[index]
        cap = pre_caption(cap,max_words=self.max_length)
        q = self.qs[index]
        q = pre_question(q,max_ques_words=self.max_q_length)
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
        image = self.transform(image)
        # self.im_name = path
        # self.img=image
        
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

# class ultrasound_multi_dataset(Dataset):

#     def __init__(self, df,config,transform=None,  split='train',):

#         self.df = df
#         print('************************',split , df.shape,'********************************')
#         # self.paths = df.dir.tolist()
#         self.paths = df[config['impath']].tolist()
#         self.qs = df[config['Q']].tolist()
#         self.captions = df[config['ANS']].tolist()

#         self.transform = transform
#         self.split=split
#         # self.im_name = ""
#         # self.img = None
#         self.max_length=config['max_length']
#         self.max_q_length=config['max_q_length']        
#     def __getitem__(self, index):
#         cap = self.captions[index]
#         q = self.qs[index]
#         path = self.paths[index]
#         cap = pre_caption(cap,max_words=self.max_length)
#         q = self.qs[index]
#         q = pre_question(q,max_ques_words=self.max_q_length)
#         try:
#             image=cv2.imread(path)[:,:,1]
#             '''
#             if self.split=='train':
#                 aug=random.choice([0,1,2])
#                 if aug:
#                     h,w=image.shape
#                     if aug==2:
#                         image = image[:,w//2:]
#                     else:
#                         image = image[:,:w//2]
#             '''
#         except:print('*****************',path)
#         if self.split=='train':
#             h,w=image.shape
#             image1 = image[:,w//2:]
#             image2 = image[:,:w//2]
#             image1=cv2.merge([image1]*3)
#             image2=cv2.merge([image2]*3)
#             image1=Image.fromarray(image1)
#             image1 = self.transform(image1)
#             image2=Image.fromarray(image2)
#             image2 = self.transform(image2)
#             # self.im_name = path
#             # self.img=image
            
#             images = torch.cat((torch.unsqueeze(image1,0),torch.unsqueeze(image2,0)),dim=0)
#         else:
#             image=cv2.merge([image]*3)
#             image=Image.fromarray(image)
#             image = self.transform(image)
#             images = torch.cat((torch.unsqueeze(image,0),torch.unsqueeze(image,0)),dim=0)
#         if self.split == 'test':           
#             return images, q, self.df.loc[index,'idx']

#         elif self.split=='train': 
#             # cap = pre_caption(cap,50)
#             answers = [cap]
#             weights = [0.2]
#             #print(self.im_name)
#             return images, q, answers,weights,path
#     def __len__(self):
#         return len(self.df)

# def vqa_collate_fn(batch):
#     image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
#     for image, question, answer, weights in batch:
#         image_list.append(image)
#         question_list.append(question)
#         weight_list += weights       
#         answer_list += answer
#         n.append(len(answer))
#     return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n        

