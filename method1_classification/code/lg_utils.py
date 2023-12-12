import re
import numpy as np
from tqdm import tqdm

from collections import Counter
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import metrics
def countfreq(res,col='cap',kind='',savep='summary',cuts=0,cute=150):
    l = []
    # for i in df['dis_en'].dropna().values:
    for i in res[col].dropna().values:
        words = set(i.split(','))
        # words = [x.strip() for x in words]
        l.extend(words)
    c = Counter(l)
    words = []
    count = []
    for k,v in c.items():
        if len(k)>0:
            words.append(k)
            count.append(v)
    words_count = list(zip(count, words))
    os.makedirs(savep,exist_ok=True)
    freq = pd.DataFrame({'words':words,'count':count})
    freq = freq.sort_values('count', ascending=False)
    
    
    top_words = sorted(words_count)[::-1][cuts:cute]
    _=plt.figure(figsize=((cute-cuts)//50*8,8))
    freq['count'].sort_values(ascending=False)[cuts:cute].plot.bar(color=['black', 'red', 'green', 'blue', 'cyan'])
    # plt.xticks(rotation=50)
    _=plt.xticks(ticks=range(len(top_words)), labels=[w for c,w in top_words], rotation=90)#,fontsize=18)
    _=plt.xlabel("Conditions")
    _=plt.ylabel("Number of images")
    # plt.show()
    _=plt.savefig(f'{savep}/freq{kind}.png',bbox_inches='tight')
    
    freq['freq'] = freq['count'] / freq['count'].sum()
    freq['Freq'] = freq['freq'].mul(100).round(1).astype(str) + '%'
    freq['Freq'] =freq['count'].astype(str)+' ('+freq['Freq']+')'
    freq.to_csv(f'{savep}/freq_imp{kind}.csv')

    return freq

def predictLabelForGivenThreshold(y_scores, threshold):
  y_pred=[]
  for sample in  y_scores:
    y_pred.append([1 if i>=threshold else 0 for i in sample ] )
  return np.array(y_pred)

def multilabel_specificity(ms):
    totSpec = 0
    totSupp = 0
    totSpecW = 0
    spec = []
    for m in ms:
        #TN/(FP+TN)
        spec.append(m[0][0]/(m[1][0]+m[0][0]))
        totSpec+=m[0][0]/(m[1][0]+m[0][0])
        totSupp += m[1][0]+m[1][1]
        totSpecW += m[0][0]/(m[1][0]+m[0][0])*(m[1][0]+m[1][1])
    return totSpec/len(ms), totSpecW/totSupp, spec

# def multilabel_accuracy(y_trues, y_preds, ms):
#     totAcc = 0
#     accs = []
#     #print("Accuracy: ########################")
#     #print("Accuracy: ########################", file=f)
#     for i,m in enumerate(ms):
#         accs.append((m[0][0]+m[1][1])/(m[0][0]+m[1][1]+m[0][1]+m[1][0]))
#         totAcc+=(m[0][0]+m[1][1])/(m[0][0]+m[1][1]+m[0][1]+m[1][0])
#     cnt = 0
#     for i in range(len(y_trues)):
#         if np.array_equal(y_trues[i], y_preds[i]):
#             cnt+=1
#     return totAcc/len(ms), cnt/len(y_trues), accs

def multilabel_accuracy(y_trues, y_preds, ms):
    totAcc = 0
    totSupp = 0
    totAccW = 0
    accs = []
    #print("Accuracy: ########################")
    #print("Accuracy: ########################", file=f)
    for i,m in enumerate(ms):
        accs.append((m[0][0]+m[1][1])/(m[0][0]+m[1][1]+m[0][1]+m[1][0]))
        totSupp += m[1][0]+m[1][1]
        totAccW +=(m[0][0]+m[1][1])/(m[0][0]+m[1][1]+m[0][1]+m[1][0])*(m[1][0]+m[1][1])
        totAcc+=(m[0][0]+m[1][1])/(m[0][0]+m[1][1]+m[0][1]+m[1][0])
    cnt = 0
    for i in range(len(y_trues)):
        if np.array_equal(y_trues[i], y_preds[i]):
            cnt+=1
    return totAcc/len(ms), cnt/len(y_trues), totAccW/totSupp, accs
def multi_label_auc(y_true, y_pred):
    total_auc = 0.
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = 0.5
        #print(auc)
        total_auc += auc
    multi_auc = total_auc / y_true.shape[1]
    return multi_auc

def print_metrics(y_trues, y_preds, M):
    label_names = list(M.keys())
    res_pd = pd.DataFrame()
    res_pd["label"] = label_names
    #M: dict for labels
    AUC = multi_label_auc(y_trues, y_preds)
    y_preds = (y_preds>.5).astype(int)
    ms = multilabel_confusion_matrix(y_trues, y_preds)
    ms_p = []
    for i,name in enumerate(label_names):
        ms_p.append(str(ms[i]))
    res_pd["confusion_matrix"] = ms_p
    #from sklearn.metrics import classification_report
    #print(classification_report(y_trues, y_preds, target_names=label_names))
    #Note that for “micro”-averaging in a multiclass setting with all labels included will produce equal precision, recall and F, while “weighted” averaging may produce an F-score that is not between precision and recall.

    macro_spec, weighted_spec, specs = multilabel_specificity(ms)
    macro_acc, strict_acc, weighted_acc, accs = multilabel_accuracy(y_trues, y_preds,ms)
    res_pd["Specificity"] = specs
    res_pd["Accuracy"] = accs
    micro_precision = metrics.precision_score(y_trues, y_preds, average='micro')
    macro_precision = metrics.precision_score(y_trues, y_preds, average='macro')
    weighted_precision = metrics.precision_score(y_trues, y_preds, average='weighted')
    precisions = metrics.precision_score(y_trues, y_preds, average=None)

    micro_recall = metrics.recall_score(y_trues, y_preds, average='micro')
    macro_recall = metrics.recall_score(y_trues, y_preds, average='macro')
    weighted_recall = metrics.recall_score(y_trues, y_preds, average='weighted')
    recalls = metrics.recall_score(y_trues, y_preds, average=None)

    micro_f1 = metrics.f1_score(y_trues, y_preds, average='micro')
    macro_f1 = metrics.f1_score(y_trues, y_preds, average='macro')
    weighted_f1 = metrics.f1_score(y_trues, y_preds, average='weighted')
    f1s = metrics.f1_score(y_trues, y_preds, average=None)
    #

    res_pd["Precision"] = precisions
    res_pd["Sensitivity"] = recalls
    res_pd["F1 score"] = f1s 
    #OP, OR, OF1, CP, CR, CF1 = metric(preds, labels, )
    # print(aps)
    # map=np.mean(aps)
    # print("mAP: {:4f}".format(map))
    print("AUC: {:4f}".format(AUC))
    print("macro_spec: {:4f}, weighted_spec: {:4f}".format(macro_spec, weighted_spec))
    # print("macro_acc: {:4f}, strict_acc: {:4f}".format(macro_acc, strict_acc))
    print("macro_acc: {:4f}, strict_acc: {:4f}, weighted_acc: {:4f}".format(macro_acc, strict_acc, weighted_acc))
    print("macro_precision: {:4f}, micro_precision: {:4f}, weighted_precision: {:4f}".format(macro_precision, micro_precision, weighted_precision))
    print("macro_recall: {:4f}, micro_recall: {:4f}, weighted_recall: {:4f}".format(macro_recall, micro_recall, weighted_recall))
    print("macro_f1: {:4f}, micro_f1: {:4f}, weighted_f1: {:4f}".format(macro_f1, micro_f1, weighted_f1))
    #res_pd.to_csv("./each_label.csv")
    # res={'macro_spec':macro_spec,'weighted_spec':weighted_spec,'macro_acc':macro_acc,'strict_acc':strict_acc,'macro_precision':macro_precision,'micro_precision':micro_precision,'weighted_precision':weighted_precision,'macro_recall':macro_recall,'micro_recall':micro_recall,'weighted_recall':weighted_recall,'macro_f1':macro_f1,'micro_f1':micro_f1,'weighted_f1':weighted_f1}
    res={'AUC':AUC, 'macro_spec':macro_spec,'weighted_spec':weighted_spec,'macro_acc':macro_acc,'strict_acc':strict_acc,'weighted_acc':weighted_acc, 'macro_precision':macro_precision,'micro_precision':micro_precision,'weighted_precision':weighted_precision,'macro_recall':macro_recall,'micro_recall':micro_recall,'weighted_recall':weighted_recall,'macro_f1':macro_f1,'micro_f1':micro_f1,'weighted_f1':weighted_f1}
    res_pd=res_pd.sort_values('F1 score',ascending=0)
    return res_pd, res

def evaldfcap(df,cap,pcap,include=None):
    t_all = df[cap].to_list()
    dic = set()
    for s in t_all:
        ss = s.split(",")
        for i in ss:
            if include:
                if i in include:
                    dic.add(i)
            else:
                dic.add(i)
    # print(dic)
    M = {}
    for i, l in enumerate(dic):
        M[l]=i
    print(M)

    y_trues= np.zeros((len(t_all), len(M)))
    #print(y_trues)
    for row,s in enumerate(t_all):
        ss = s.split(",")
        for i in ss:
            if include:
                if i in include:
                    y_trues[row, M[i]]=1
            else:
                y_trues[row, M[i]]=1
        #print(ss)
        #print(y_trues[row])
        #break
    p_all = df[pcap].to_list()
    y_preds = np.zeros((len(t_all), len(M)))
    for row,s in enumerate(p_all):
        if str(s)=='nan':
            continue
        ss = s.split(",")
        #print(ss)
        for i in ss:
            if include:
                if i in include:
                    if i in M:
                        y_preds[row, M[i]]=1
            else:
                if i in M:
                    y_preds[row, M[i]]=1
                # print(i)
                # pass
                
    res_pd, res = print_metrics(y_trues, y_preds, M)
    return res_pd, res

# def rm_small_dup(ls):
#     c=[]
#     for w in ls:
#         if ',' in w:
#             c.extend(w.split(','))
#         else:
#             c.append(w)
#     c = sorted(set(c), key=c.index)
#     C=c.copy()
#     c.sort(key= lambda ss:len(ss),reverse=False)

#     for i,small in enumerate(c):
#         for big in c[i+1:]:
#             if re.search(small.replace(' ','.*'), big):
#                 if small in C:
#                     C.remove(small)
#     return C
def rm_small_dup(ls):
    if not isinstance(ls,list):
        ls=ls.split(',')
    c=[]
    for w in ls:
        if ',' in w:
            c.extend(w.split(','))
        else:
            c.append(w)

    c = sorted(set(c), key=c.index)
    C=c.copy()
    c.sort(key= lambda ss:len(ss),reverse=False)

    for i,small in enumerate(c):
        for big in c[i+1:]:
            if re.search(small.replace(' ','.*'), big):
                if small in C:
                    C.remove(small)
    return ','.join(C)

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



# def clean_fa_cn(df,lb):
#     W = '|'.join(lb['w'].values.tolist())
#     W = '|'.join(sorted(W.split('|'), key=lambda x: len(x), reverse=True))
#     df['des']=np.nan
#     df['des']=df['des'].astype(object)
#     pos = '|'.join(['quality', 'phase', 'ves', 'shape', 'fluoh', 'fluo', 'block',
#         'defect', 'size', 'sign', 'dis', 'class', 'further'])
#     uniterm=set()
#     # t=4
#     # for i in df.index[t:t+1]:
#     for i in tqdm(df.index[:]):
#         cap=df.loc[i,'Findings']
#         cap = re.sub(r'([A-Za-z]+)', r' \1 ',cap) #英文前后加space
#         caps= re.split(r"[;；]", cap)
#         LS={}
#         for n,cap in enumerate(caps):
#             Ls=[]
#             cs = re.split(r"[,，]", cap)
#             for c in cs:
#                 ls={}
#                 matches = re.finditer('(?<!未见)(?<!未见明显)'+'('+W+')', c)
#                 matches = sorted(matches, key=lambda match: match.start())
#                 res = [match.group() for match in matches]

#                 if len(res)==0:
#                     continue
#                 ls['full'] = res
#                 for k,w in lb[['k','w']].values:
#                     res=re.findall('(?<!未见)(?<!未见明显)'+'('+w+')',c)
#                     if len(res)>0:
#                         # print(res)
#                         ls[k]=res
#                         # uniterm[k]=uniterm[k] | set(res)
#                         uniterm=uniterm | set(res)
#                 # print(c)
#                 print(ls)
#                 if len(ls)>0:
#                     if re.search(pos,','.join(ls.keys())):
#                         Ls.append(ls)
#             if len(Ls)>0:
#                 LS[n]=Ls
#         cap=df.loc[i,'Impression']
#         if isinstance(cap,str):
#             caps= re.split(r"[;；]", cap)
#             ds=[]
#             for cap in caps:
#                 ls={}
#                 matches = re.finditer('(?<!未见)(?<!未见明显)'+'('+W+')', cap)
#                 matches = sorted(matches, key=lambda match: match.start())
#                 res = [match.group() for match in matches]
#                 # print(res)
#                 if len(res)>0:
#                     ls['full']=res
#                 for k,w in lb[['k','w']].values:
#                     res=re.findall('(?<!未见)(?<!未见明显)'+'('+w+')',cap)

#                     if len(res)==0:
#                         continue
#                     ls[k]=res
#                     # uniterm[k]=uniterm[k] | set(res)
#                     uniterm=uniterm| set(res)
#                 print('dis---------------------------------------------')
#                 print(cap)
#                 # print(ls)
#                 if len(ls)>0:
#                     ds.append(ls)
#             if len(ds)>0:        
#                 LS['dis']=ds
#                 dis = []
#                 for d in ds:
#                     for kk in ['sign','dis']:
#                         if kk in d.keys():
#                             dis.extend(d[kk])
#                 dis=','.join(dis)
#                 df.loc[i,'dis']=dis
        
#         df.at[i,'des']=LS
#     return df

def exclude(kw,ex,cap):
    matched=1
    bf = re.findall('(.*)'+kw,cap)
    if len(bf)>0:
        bf=bf[0]
        if len(bf)>15:
            bf=bf[-15:]
        # print(bf)
        if isinstance(bf,str):
            if re.search('\\b('+ex+')\\b',bf,re.IGNORECASE):
                matched = 0
    return matched
def addkw(cap,kw):
    if kw not in cap:
        cap = cap+','+kw
    return cap

def rm_dup(cap):
    cap=cap.split(',')
    cap=','.join(set(cap))
    return cap
            
def clean(kw):
    ls=kw.split('|')
    ls=[x.strip() for x in ls if len(x)>0]
    kw='|'.join(ls)
    return kw

import warnings
warnings.filterwarnings('ignore')
# kwcol='name'
# s=0
# t=-1
# def formatco(df,kwcol,sheet,addfinding=1,f='/home/danli/caption/multilabel/condict0729.xlsx'):
#     dis=pd.read_excel(f,sheet_name=sheet,usecols=['ABB', 'IMP', 'KW', 'FINDING', 'Abb', 'Imp','Kw', 'Finding','EX','Ex'])
#     dis.dropna(inplace=True, axis=0, how='all')
#     dis['Abb']=dis['Abb'].fillna(dis['Imp'])
#     dis['ABB']=dis['ABB'].fillna(dis['IMP'])
#     dis['ABB']=dis['ABB'].str.strip()
#     dis['Abb']=dis['Abb'].str.strip()
#     dis['Finding']=dis['Finding'].fillna('')
#     dis['FINDING']=dis['FINDING'].fillna('')
#     dis['KW']=dis['KW'].fillna('')
#     dis['KW']=dis['IMP']+'|'+dis['KW']
#     dis['KW'] = dis['KW'].apply(clean)

#     dis['Kw']=dis['Imp']+'|'+dis['Kw']
#     dis['Kw']=dis['Kw'].fillna('')
#     dis['Kw'] = dis['Kw'].apply(clean)

#     dis['EX']=dis['EX'].fillna('')
#     dis['EX'] = 'without|no|non|'+dis['EX']
#     dis['Ex']=dis['Ex'].fillna('')
#     dis['Ex'] = 'without|no|non|'+dis['Ex']

#     for c in ['ABB', 'IMP', 'KW', 'FINDING', 'Abb', 'Imp','Kw', 'Finding','EX','Ex']:
#         # dis[c]=dis[c].str.replace('||','|')
#         dis[c]=dis[c].str.strip('|')
    
#     d = dis.dropna(subset=['ABB','IMP']).set_index('ABB')['IMP'].to_dict()
#     d.update(dis.dropna(subset=['Abb','Imp']).set_index('Abb')['Imp'].to_dict())
#     # print(dis.ABB_.values,dis.Abb_.values)
#     # orgcols=list(df.columns)
#     temp=df.copy()
#     for i in df.index[:]:
#         cap=df.loc[i,kwcol]
#         # print(cap)
#         for j in dis.index:
#             # try:
#             ABB=dis.loc[j,'ABB']
#             Abb=dis.loc[j,'Abb']
#             # except: print('error',dis.loc[j])
#             if not pd.isna(Abb):
#             # if len(Abb)>0:
#                 kwstring=dis.loc[j,'Kw']
#                 if len(kwstring)>0:
#                     matched=0
#                     Kw_ = '\\b('+kwstring+')'
#                     Abb_ = '\\b('+Abb+')\\b'
#                     for kw,flag in zip([Kw_,Abb_],[re.IGNORECASE,0]):
                        
#                         if re.search(kw,cap, flag):
#                             matched+=exclude(kw,dis.loc[j,'Ex'],cap)
#                     if matched:
#                         temp.loc[i,Abb]=1
#                         temp.loc[i,ABB]=1
#                         if addfinding:
#                             cap=addkw(cap,dis.loc[j,'Finding'])
#                             cap=addkw(cap,dis.loc[j,'FINDING'])
#                     # print(kwstring,Abb,ABB)
#             kwstring=dis.loc[j,'KW']
#             if len(kwstring)>1:
#                 matched=0
#                 KW_ = '\\b('+kwstring+')'
#                 ABB_ = '\\b('+ABB+')\\b'               
#                 for kw,flag in zip([KW_,ABB_],[re.IGNORECASE,0]):
#                     # print(kw,cap)
#                     if re.search(kw,cap, flag):
#                         matched+=exclude(kw,dis.loc[j,'EX'],cap)
#                 if matched:
#                     temp.loc[i,ABB]=1
#                     if addfinding:
#                         cap=addkw(cap,dis.loc[j,'FINDING'])
#         if addfinding:
#             df.loc[i,'rkw']=cap

#     cols=set(df.columns) & set(d.keys())
#     for i in df.index[:]:
#         cap=[]

#         capfull=[]
#         for c in cols:
#             v = temp.loc[i,c]
#             if v==1:
#                 cap.append(c)
#                 capfull.append(d[c])
#         # cap=','.join(cap)
#         df.loc[i,'capabb']=','.join(cap)
#         df.loc[i,'capfull']=','.join(set(capfull))
#     # print(orgcols)
#     # df=df[orgcols+['rkw','capabb','capfull']]
#     return df,cols,d

def formatco(df,kwcol,sheet,addfinding=1,f='/home/danli/caption/FFA/condict08.xlsx'):
    dis=pd.read_excel(f,sheet_name=sheet,usecols=['ABB', 'IMP', 'KW', 'FINDING', 'Abb', 'Imp','Kw', 'Finding','EX','Ex'])
    dis.dropna(inplace=True, axis=0, how='all')
    dis['Abb']=dis['Abb'].fillna(dis['Imp'])
    dis['ABB']=dis['ABB'].fillna(dis['IMP'])
    dis['ABB']=dis['ABB'].str.strip()
    dis['Abb']=dis['Abb'].str.strip()
    dis['Finding']=dis['Finding'].fillna('')
    dis['FINDING']=dis['FINDING'].fillna('')
    dis['KW']=dis['KW'].fillna('')
    dis['KW']=dis['IMP']+'|'+dis['KW']
    dis['KW'] = dis['KW'].apply(clean)

    dis['Kw']=dis['Imp']+'|'+dis['Kw']
    dis['Kw']=dis['Kw'].fillna('')
    dis['Kw'] = dis['Kw'].apply(clean)

    dis['EX']=dis['EX'].fillna('')
    dis['EX'] = 'without|no|non|'+dis['EX']
    dis['Ex']=dis['Ex'].fillna('')
    dis['Ex'] = 'without|no|non|'+dis['Ex']

    for c in ['ABB', 'IMP', 'KW', 'FINDING', 'Abb', 'Imp','Kw', 'Finding','EX','Ex']:
        # dis[c]=dis[c].str.replace('||','|')
        dis[c]=dis[c].str.strip('|')
    
    d = dis.dropna(subset=['ABB','IMP']).set_index('ABB')['IMP'].to_dict()
    d.update(dis.dropna(subset=['Abb','Imp']).set_index('Abb')['Imp'].to_dict())
    # print(dis.ABB_.values,dis.Abb_.values)
    # orgcols=list(df.columns)
    temp=df.copy()
    for i in df.index[:]:
        cap=df.loc[i,kwcol]
        # print(cap)
        for j in dis.index:
            # try:
            ABB=dis.loc[j,'ABB']
            Abb=dis.loc[j,'Abb']
            # except: print('error',dis.loc[j])
            if not pd.isna(Abb):
            # if len(Abb)>0:
                kwstring=dis.loc[j,'Kw']
                if len(kwstring)>0:
                    matched=0
                    Kw_ = '\\b('+kwstring+')'
                    Abb_ = '\\b('+Abb+')\\b'
                    for kw,flag in zip([Kw_,Abb_],[re.IGNORECASE,0]):
                        
                        if re.search(kw,cap, flag):
                            matched+=exclude(kw,dis.loc[j,'Ex'],cap)
                    if matched:
                        temp.loc[i,Abb]=1
                        temp.loc[i,ABB]=1
                        if addfinding:
                            cap=addkw(cap,dis.loc[j,'Finding'])
                            cap=addkw(cap,dis.loc[j,'FINDING'])
                    # print(kwstring,Abb,ABB)
            kwstring=dis.loc[j,'KW']
            if len(kwstring)>1:
                matched=0
                KW_ = '\\b('+kwstring+')'
                ABB_ = '\\b('+ABB+')\\b'               
                for kw,flag in zip([KW_,ABB_],[re.IGNORECASE,0]):
                    # print(kw,cap)
                    if re.search(kw,cap, flag):
                        matched+=exclude(kw,dis.loc[j,'EX'],cap)
                if matched:
                    temp.loc[i,ABB]=1
                    if addfinding:
                        cap=addkw(cap,dis.loc[j,'FINDING'])
        if addfinding:
            df.loc[i,'rkw']=cap

    cols=set(temp.columns) & set(d.keys())
    for i in df.index[:]:
        cap=[]

        capfull=[]
        for c in cols:
            v = temp.loc[i,c]
            if v==1:
                cap.append(c)
                capfull.append(d[c])
        # cap=','.join(cap)
        df.loc[i,'capabb']=','.join(cap)
        df.loc[i,'capfull']=','.join(set(capfull))
    # print(orgcols)
    return df,cols,d


