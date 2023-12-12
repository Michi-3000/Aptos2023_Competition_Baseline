from .bleu.bleu import Bleu
from .cider.cider import Cider
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .spice.spice import Spice
from .wmd.wmd import WMD
import numpy as np
import json
def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def eval(gts,res):
    #print(res)
    scorer = Bleu(n=4)
    s1, s1s = scorer.compute_score(gts, res)
    #print(s1s)
    
    scorer = Cider()
    s2, s2s = scorer.compute_score(gts, res)

    #scorer = Meteor()
    #s3, _ = scorer.compute_score(gts, res)

    scorer = Rouge()
    s4, s4s = scorer.compute_score(gts, res)

    
    #scorer = Spice()
    #s5, _ = scorer.compute_score(gts, res)
    ''' 
    out = {}
    spice_data={}
    
    for k in _[0].keys():
        if k != 'All':
            out['SPICE_'+k] = np.array([v[k]['f'] for v in _])
            spice_data['SPICE_'+k] = list(out['SPICE_'+k][out['SPICE_'+k]==out['SPICE_'+k]])
            out['SPICE_'+k] = (out['SPICE_'+k][out['SPICE_'+k]==out['SPICE_'+k]]).mean()
    
    if(args.save_spice):
        filename=args.save_path
        with open(filename,'w') as file_obj:
            json.dump(spice_data,file_obj)
        print('save to',filename)
    '''

    # scorer = WMD()
    # s6, _ = scorer.compute_score(gts, res)

    return {'bleu':s1,'cider':s2,'rouge':s4, 'bleus':s1s,'ciders':s2s, 'rouges':s4s}
    #return {'bleu':s1,'cider':s2,'meteor':s3,'rouge':s4,'spice':s5,'spice_detail':out}

def test_eval(gts,res,spice=1):
    #print(res)
    scorer = Bleu(n=4)
    s1, s1s = scorer.compute_score(gts, res)
    #print(s1s)
    
    scorer = Cider()
    s2, s2s = scorer.compute_score(gts, res)

    #scorer = Meteor()
    #s3, _ = scorer.compute_score(gts, res)

    scorer = Rouge()
    s4, s4s = scorer.compute_score(gts, res)

    #scorer = Spice()
    #s5, _ = scorer.compute_score(gts, res)
    #print(gts)
    if spice:
        scorer = Spice()
        maxl = 507
        gts_={}
        res_={}
        for k,v in gts.items():
            #print(v)
            gts_[k] = []
            for i,vv in enumerate(v):
                if is_contains_chinese(vv):
                    vv = vv.replace(" ","")
                #print(vv)
                #print(len(vv))
                gts_[k].append(vv[:maxl])
            #break
        for k,v in res.items():
            res_[k] = []
            for i,vv in enumerate(v):
                if is_contains_chinese(vv):
                    vv = vv.replace(" ","")
                #print(vv)
                #print(len(vv))
                res_[k].append(vv[:maxl])
            #break
        try: 
            s5, _ = scorer.compute_score(gts_, res_)
        except Exception as e:
            print(e)
            s5 = np.nan
    else:s5=np.nan
    
    
    ''' 
    out = {}
    spice_data={}
    
    for k in _[0].keys():
        if k != 'All':
            out['SPICE_'+k] = np.array([v[k]['f'] for v in _])
            spice_data['SPICE_'+k] = list(out['SPICE_'+k][out['SPICE_'+k]==out['SPICE_'+k]])
            out['SPICE_'+k] = (out['SPICE_'+k][out['SPICE_'+k]==out['SPICE_'+k]]).mean()
    
    if(args.save_spice):
        filename=args.save_path
        with open(filename,'w') as file_obj:
            json.dump(spice_data,file_obj)
        print('save to',filename)
    '''

    # scorer = WMD()
    # s6, _ = scorer.compute_score(gts, res)

    return {'bleu':s1,'cider':s2,'rouge':s4, 'spice':s5, 'bleus':s1s,'ciders':s2s, 'rouges':s4s}

def get_bleu(gts,res):
    scorer = Bleu(n=4)
    s, _ = scorer.compute_score(gts, res)
    return s

def get_meteor(gts, res):
    scorer = Meteor()
    s, _ = scorer.compute_score(gts, res)
    return s

def get_cider(gts, res):
    scorer = Cider()
    s, _ = scorer.compute_score(gts, res)
    return s

def get_rouge(gts, res):
    scorer = Rouge()
    s, _ = scorer.compute_score(gts, res)
    return s


def get_spice(gts, res):
    scorer = Spice()
    s, _ = scorer.compute_score(gts, res)
    out = {}
    for k in list(_.values())[0]['SPICE'].keys():
        if k != 'All':
            out['SPICE_'+k] = np.array([v['SPICE'][k]['f'] for v in _.values()])
            out['SPICE_'+k] = (out['SPICE_'+k][out['SPICE_'+k]==out['SPICE_'+k]]).mean()
    return s


def get_wmd(gts, res):
    scorer = WMD()
    s, _ = scorer.compute_score(gts, res)
    return s
