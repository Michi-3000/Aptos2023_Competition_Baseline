import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
# from .cal_mAP import json_map
from .cal_PR import json_metric, metric, json_metric_top3
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_auc_score

def print_metrics(y_trues, y_preds, M):
    label_names = list(M.keys())
    res_pd = pd.DataFrame()
    res_pd["label"] = label_names
    #M: dict for labels
    from sklearn.metrics import multilabel_confusion_matrix
    ms = multilabel_confusion_matrix(y_trues, y_preds)
    ms_p = []
    for i,name in enumerate(label_names):
        ms_p.append(str(ms[i]))
    res_pd["confusion_matrix"] = ms_p
    #from sklearn.metrics import classification_report
    #print(classification_report(y_trues, y_preds, target_names=label_names))
    #Note that for “micro”-averaging in a multiclass setting with all labels included will produce equal precision, recall and F, while “weighted” averaging may produce an F-score that is not between precision and recall.

    macro_spec, weighted_spec, specs = multilabel_specificity(ms)
    macro_acc, strict_acc, weighted_acc, accs = multilabel_accuracy(labels, preds,ms)
    res_pd["specificity"] = specs
    res_pd["accuracy"] = accs
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

    res_pd["precision"] = precisions
    res_pd["recall"] = recalls
    res_pd["f1"] = f1s 
    #OP, OR, OF1, CP, CR, CF1 = metric(preds, labels, )
    # print(aps)
    print("macro_spec: {:4f}, weighted_spec: {:4f}".format(macro_spec, weighted_spec))
    print("macro_acc: {:4f}, strict_acc: {:4f}, weighted_acc: {:4f}".format(macro_acc, strict_acc, weighted_acc))
    print("macro_precision: {:4f}, micro_precision: {:4f}, weighted_precision: {:4f}".format(macro_precision, micro_precision, weighted_precision))
    print("macro_recall: {:4f}, micro_recall: {:4f}, weighted_recall: {:4f}".format(macro_recall, micro_recall, weighted_recall))
    print("macro_f1: {:4f}, micro_f1: {:4f}, weighted_f1: {:4f}".format(macro_f1, micro_f1, weighted_f1))
    #res_pd.to_csv("./each_label.csv")
    res={'macro_spec':macro_spec,'weighted_spec':weighted_spec,'macro_acc':macro_acc,'strict_acc':strict_acc,'weighted_acc':weighted_acc,'macro_precision':macro_precision,'micro_precision':micro_precision,'weighted_precision':weighted_precision,'macro_recall':macro_recall,'micro_recall':micro_recall,'weighted_recall':weighted_recall,'macro_f1':macro_f1,'micro_f1':micro_f1,'weighted_f1':weighted_f1}
    return res_pd, res
    
def multilabel_specificity(ms):
    totSpec = 0
    totSupp = 0
    totSpecW = 0
    spec = []
    for m in ms:
        # print(m)
        #TN/(FP+TN)
        spec.append(m[0][0]/(m[1][0]+m[0][0]))
        totSpec+=m[0][0]/(m[1][0]+m[0][0])
        totSupp += m[1][0]+m[1][1]
        totSpecW += m[0][0]/(m[1][0]+m[0][0])*(m[1][0]+m[1][1])
    #print("Specificity: ########################")
    #print("Specificity: ########################", file=f)
    #for i, name in enumerate(label_names):
        #print("For {} specificity: {:.2f}".format(name, spec[i]))
        #print("For {} specificity: {:.2f}".format(name, spec[i]), file= f)
    #print("Macro Specificity: {:.2f}".format(totSpec/len(label_names)))
    #print("Weighted Sepcificity: {:.2f}".format(totSpecW/totSupp))
    return totSpec/len(ms), totSpecW/totSupp, spec

def multilabel_accuracy(y_trues, y_preds, ms):
    totAcc = 0
    totSupp = 0
    totAccW = 0
    accs = []
    #print("Accuracy: ########################")
    #print("Accuracy: ########################", file=f)
    for i,m in enumerate(ms):
        #print(sum(sum(m)))
        #print("For {} accuracy: {:.2f}".format(label_names[i], (m[0][0]+m[1][1])/sum(sum(m))))
        #print("For {} accuracy: {:.2f}".format(label_names[i], (m[0][0]+m[1][1])/sum(sum(m))), file=f)
        accs.append((m[0][0]+m[1][1])/(m[0][0]+m[1][1]+m[0][1]+m[1][0]))
        totSupp += m[1][0]+m[1][1]
        totAccW +=(m[0][0]+m[1][1])/(m[0][0]+m[1][1]+m[0][1]+m[1][0])*(m[1][0]+m[1][1])
        totAcc+=(m[0][0]+m[1][1])/(m[0][0]+m[1][1]+m[0][1]+m[1][0])
    cnt = 0
    for i in range(len(y_trues)):
        if np.array_equal(y_trues[i], y_preds[i]):
            cnt+=1
    #print("Strict Accuracy: {:.2f}".format(cnt/len(y_trues)))
    #print("Strict Accuracy: {:.2f}".format(cnt/len(y_trues)), file=f)
    #print("Macro Accuracy {:.2f}".format(totAcc/len(label_names)))
    #print("Macro Accuracy {:.2f}".format(totAcc/len(label_names)), file=f)
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

def mAP(cls_id, scores, labels):
    # assert len(ann_json) == len(pred_json)
    num = len(scores)
    predict = np.zeros((num), dtype=np.float64)
    target = np.zeros((num), dtype=np.float64)
    
    for i in range(num):
        predict[i]= (scores[i][cls_id])#pred[i]["scores"][cls_id]
        target[i] = (labels[i][cls_id])#pred[i]["target"][cls_id]
        #print(type(anns.loc[i,'target']),cls_id)
        # target[i] = eval('['+anns.loc[i,'target']+']')[cls_id]
    tmp = np.argsort(-predict)
    target = target[tmp]
    predict = predict[tmp]
    #print(target)
    #print(predict)
    pre, obj = 0, 0
    for i in range(num):
        if target[i]>0.99999:
            obj += 1.0
            pre += obj / (i+1)
    if obj:
        pre /= obj
        return pre
    else:
        return np.nan

# def evaluation(pred, classes, data):
def evaluation(labels, scores,classes):
    print("Evaluation")
    #print(len(pred))
    #print(pred[0]["target"].shape)
    #print(classes[0])
    #print(data.columns)
    #print(len(data))
    # scores = []
    # labels = []
    # for i in pred:
    #     scores.append(i["scores"])
    #     labels.append(i["target"].numpy())
        #print(i["scores"])
        #print(i["target"])
    aps = []#np.zeros(len(classes), dtype=np.float64)
    
    labels = np.array(labels)
    scores = np.array(scores)
    # print(labels,scores)
    for i, _ in enumerate(tqdm(classes)):
        ap = mAP(i, scores, labels)
        #print(ap)
        if str(ap) != "nan":          
            aps.append(ap)
    #labels = labels.reshape(len(data), num_classes)
    preds = []
    for sample in  scores:
        preds.append([1 if i>=0.5 else 0 for i in sample ] )
    preds = np.array(preds)

    ms = multilabel_confusion_matrix(labels, preds)
    # for i,m in enumerate(ms):
        # print(classes[i], m)
    macro_spec, weighted_spec, _ = multilabel_specificity(ms)
    macro_acc, strict_acc, weighted_acc, _ = multilabel_accuracy(labels, preds,ms)
    
    micro_precision = metrics.precision_score(labels, preds, average='micro')
    macro_precision = metrics.precision_score(labels, preds, average='macro')
    weighted_precision = metrics.precision_score(labels, preds, average='weighted')
    
    micro_recall = metrics.recall_score(labels, preds, average='micro')
    macro_recall = metrics.recall_score(labels, preds, average='macro')
    weighted_recall = metrics.recall_score(labels, preds, average='weighted')
    
    micro_f1 = metrics.f1_score(labels, preds, average='micro')
    macro_f1 = metrics.f1_score(labels, preds, average='macro')
    weighted_f1 = metrics.f1_score(labels, preds, average='weighted')    
    AUC = multi_label_auc(labels, preds)
    #OP, OR, OF1, CP, CR, CF1 = metric(preds, labels, )
    # print(aps)
    meanAP=np.mean(aps)
    # print("mAP: {:4f}".format(map))
    # print("AUC: {:4f}".format(AUC))
    # print("macro_spec: {:4f}, weighted_spec: {:4f}".format(macro_spec, weighted_spec))
    # print("macro_acc: {:4f}, strict_acc: {:4f}, weighted_acc: {:4f}".format(macro_acc, strict_acc, weighted_acc))
    # print("macro_precision: {:4f}, micro_precision: {:4f}, weighted_precision: {:4f}".format(macro_precision, micro_precision, weighted_precision))
    # print("macro_recall: {:4f}, micro_recall: {:4f}, weighted_recall: {:4f}".format(macro_recall, micro_recall, weighted_recall))
    # print("macro_f1: {:4f}, micro_f1: {:4f}, weighted_f1: {:4f}".format(macro_f1, micro_f1, weighted_f1))
    
    res={'mAP':meanAP, 'AUC':AUC, 'macro_spec':macro_spec,'weighted_spec':weighted_spec,'macro_acc':macro_acc,'strict_acc':strict_acc,'weighted_acc':weighted_acc, 'macro_precision':macro_precision,'micro_precision':micro_precision,'weighted_precision':weighted_precision,'macro_recall':macro_recall,'micro_recall':micro_recall,'weighted_recall':weighted_recall,'macro_f1':macro_f1,'micro_f1':micro_f1,'weighted_f1':weighted_f1}
    print(res)
    return res



