import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
"""
   Copyright 2020 paper:Ophthalmic Disease Diagnosis Based On Visual Language Generation
   File name:metrics.py
   Data Created: 01/11/2020
   Data last modified: 04/01/2021
   address: Artificial Intelligence Application Research Center, Huawei Technologies Co Ltd, Shenzhen, China
"""
def bleu(gts, res):
    scorer = Bleu(n=4)
    score = scorer.compute_score(gts, res)
    return score
def cider(gts, res):
    scorer = Cider()
    score = scorer.compute_score(gts, res)
    return score / 10
def rouge(gts, res):
    scorer = Rouge()
    score = scorer.compute_score(gts, res)
    return score
def evaluation_indicator(score, target, vocab):
    x, y, z = score.shape
    res = {}
    gts = {}
    for idx in range(0,x):
        score_idx = np.argmax(score[idx, :].cpu().detach().numpy(), axis=1)
        zero_index = np.argwhere(score_idx==0)
        score_idx = np.delete(score_idx, zero_index)
        two_index = np.argwhere(score_idx==2)
        score_idx = np.delete(score_idx, two_index)

        scores_word = [vocab.idx2word[ind] for ind in score_idx]
        scores_str = ' '
        scores_str = scores_str.join(scores_word)
        res[str(idx)] = [scores_str]

        target_idx = target[idx, :].cpu().numpy()
        zero_index = np.argwhere(target_idx==0)
        target_idx = np.delete(target_idx, zero_index)
        two_index = np.argwhere(target_idx==2)
        target_idx = np.delete(target_idx, two_index)
        target_word = [vocab.idx2word[ind] for ind in target_idx]
        target_str = ' '
        target_str = target_str.join(target_word)
        gts[str(idx)] = [target_str]
    blue_score = bleu(gts=gts, res=res)
    cider_score = cider(gts=gts, res=res)
    rouge_score = rouge(gts=gts, res=res)
    return blue_score, cider_score, rouge_score

def evaluation_indicator(score, target, vocab):
    x, y, z = score.shape
    res = {}
    gts = {}
    for idx in range(0,x):
        score_idx = np.argmax(score[idx, :].cpu().detach().numpy(), axis=1)
        zero_index = np.argwhere(score_idx==0)
        score_idx = np.delete(score_idx, zero_index)
        two_index = np.argwhere(score_idx==2)
        score_idx = np.delete(score_idx, two_index)

        scores_word = [vocab.idx2word[ind] for ind in score_idx]
        scores_str = ' '
        scores_str = scores_str.join(scores_word)
        res[str(idx)] = [scores_str]

        target_idx = target[idx, :].cpu().numpy()
        zero_index = np.argwhere(target_idx==0)
        target_idx = np.delete(target_idx, zero_index)
        two_index = np.argwhere(target_idx==2)
        target_idx = np.delete(target_idx, two_index)
        target_word = [vocab.idx2word[ind] for ind in target_idx]
        target_str = ' '
        target_str = target_str.join(target_word)
        gts[str(idx)] = [target_str]
    blue_score = bleu(gts=gts, res=res)
    cider_score = cider(gts=gts, res=res)
    rouge_score = rouge(gts=gts, res=res)
    return blue_score, cider_score, rouge_score

