import re
import json
import os

import torch
import torch.distributed as dist

import utils

def pre_caption(caption,max_words=200):
    # caption = re.sub(
    #     r"([.!\"()*#:;~])",       
    #     ' ',
    #     caption.lower(),
    # )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def pre_question(question,max_ques_words=150):
    # question = re.sub(
    #     r"([.!\"()*#:;~])",
    #     '',
    #     question.lower(),
    # ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    print(final_result_file)
    
    json.dump(result,open(result_file,'w'),ensure_ascii=False)

    # dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'), ensure_ascii=False)
        print('result file saved to %s'%final_result_file)

    return final_result_file

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import time
def validate(configs, val_loader, test_loader, encoder, decoder, criterion, device, vocab, split = 'val'):
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5acc = AverageMeter()
    top1acc = AverageMeter()
    blue1_acc = AverageMeter()
    blue2_acc = AverageMeter()
    blue3_acc = AverageMeter()
    blue4_acc = AverageMeter()
    cider_acc = AverageMeter()
    rouge_acc = AverageMeter()
    dev_loader = val_loader if split is 'val' else test_loader

    start = time.time()
    last_idx = len(dev_loader) - 1

    # navalue=vocab('<unk>')
    # rep = torch.tensor(0.0).cuda()
    # print(navalue)
    with torch.no_grad():
        for i, (img, caps, cap_len) in enumerate(dev_loader):
            last_batch = i == last_idx
            img = img.to(device)
            caps = caps.to(device)
            cap_len = cap_len.to(device)
            if encoder is not None:
                # img = encoder(img, caps)
                img = encoder(img)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(img, caps, cap_len)

            targets = caps_sorted[:, 1:]
            N = sum(decode_lengths)
            # print(N)
            # N = N-sum(targets==navalue)
            # print(sum(targets==navalue))
            # mscores = scores.clone()
            # scores[targets==navalue]=0
            # mtargets = targets.clone()
            # targets[targets==navalue]=0

            metrics_blue, metrics_cider, metrics_rouge_l = \
                evaluation_indicator(score=scores, target=targets, vocab=vocab)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            loss = criterion(scores, targets)
            loss += configs['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()
            losses.update(loss.item(), N)
            top5 = accuracy(scores, targets, 5)
            top5acc.update(top5, N)
            top1 = accuracy(scores, targets, 1)
            top1acc.update(top1, N)
            batch_time.update(time.time() - start)

            blue1_acc.update(metrics_blue[0], N)
            blue2_acc.update(metrics_blue[1], N)
            blue3_acc.update(metrics_blue[2], N)
            blue4_acc.update(metrics_blue[3], N)
            cider_acc.update(metrics_cider, N)
            rouge_acc.update(metrics_rouge_l, N)

            start = time.time()
            if last_batch:#i % configs['print_freq'] == 0:                
                print('-' * 50)
                print(f'{split}: [{0}/{1}] || '
                      'Batch Time val(avg): {batch_time.val:.3f} ({batch_time.avg:.3f}) || '
                      'Loss val(avg): {loss.val:.4f} ({loss.avg:.4f})'.format(i, len(dev_loader), batch_time=batch_time,
                                                               loss=losses))
                print(f'Blue: {blue1_acc.val} {blue2_acc.val} {blue3_acc.val} {blue4_acc.val}\n')
                print(f'Cider: {cider_acc.val}\n')
                print(f'Rouge-L: {rouge_acc.val}\n')
                print('-' * 50)
        # return losses, top1acc
        return OrderedDict([('Loss', losses.avg),('Top1', top1acc.avg),('Blue1',blue1_acc.avg),
    ('Blue2',blue1_acc.avg),('Blue3',blue1_acc.avg),('Blue4',blue1_acc.avg),
    ('Cider',cider_acc.avg),('Rouge-L',rouge_acc.val)])        
# from pycocotools.coco import COCO
#from pycocoevalcap.eval import COCOEvalCap
# from torchvision.datasets.utils import download_url

# def coco_caption_eval(coco_gt_root, results_file, split):
#     urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
#             'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
#     filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
#     download_url(urls[split],coco_gt_root)
#     annotation_file = os.path.join(coco_gt_root,filenames[split])
    
#     # create coco object and coco_result object
#     coco = COCO(annotation_file)
#     coco_result = coco.loadRes(results_file)

#     # create coco_eval object by taking coco and coco_result
#     coco_eval = COCOEvalCap(coco, coco_result)

#     # evaluate on a subset of images by setting
#     # coco_eval.params['image_id'] = coco_result.getImgIds()
#     # please remove this line when evaluating the full validation set
#     # coco_eval.params['image_id'] = coco_result.getImgIds()

#     # evaluate results
#     # SPICE will take a few minutes the first time, but speeds up due to caching
#     coco_eval.evaluate()

#     # print output evaluation scores
#     for metric, score in coco_eval.eval.items():
#         print(f'{metric}: {score:.3f}')
    
#     return coco_eval
