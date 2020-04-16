from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
import os

def sigmoid(x, factor=1.0):
    return 1 / (1 + np.exp(-x))

# calc classifier accuracy
def calcAccuracy(params, out_dir, pred, l, threshold=False, prefix="val"):
    total = len(pred)
    correct = 0

    pred = sigmoid(pred)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    condG = 0
    condB = 0
    predG = 0
    predB = 0

    fpr, tpr, thresholds = metrics.roc_curve(l.astype(int), pred, pos_label=1)
    AUC = metrics.auc(fpr, tpr)

    mindist = 1.0
    mindistIdx = -1
    for i in range(len(thresholds)):
        dist = np.sqrt((1-tpr[i])*(1-tpr[i])+fpr[i]*fpr[i])
        if dist < mindist:
            mindist = dist
            mindistIdx = i
    if not threshold:
        threshold = thresholds[mindistIdx]
        print("\nselected threshold: {}".format(threshold))
    else:
        print("\nfixed threshold: {}".format(threshold))
    for i in range(len(pred)):
        if l[i] == 0.0 and pred[i] <= threshold:
            correct += 1
            tn += 1
            condB += 1
            predB += 1
        elif l[i] == 1.0 and pred[i] > threshold:
            correct += 1
            tp += 1
            condG += 1
            predG += 1
        elif l[i] == 1.0 and pred[i] <= threshold:
            fn += 1
            condG += 1
            predB += 1
        elif l[i] == 0.0 and pred[i] > threshold:
            fp += 1
            condB += 1
            predG += 1
    TPR = float(tp)/float(max(1, condG))
    TNR = float(tn)/float(max(1, condB))
    FPR = float(fp)/float(max(1, condB))
    FNR = float(fn)/float(max(1, condG))
    PPV = float(tp)/float(max(1, predG))
    NPV = float(tn)/float(max(1, predB))
    F1 =  2.0/((1.0/max(0.00001, TPR)) + (1.0/max(0.00001, PPV)))
    acc = float(correct)/float(total)
    if "val" in prefix:
        print("tp: {}, tn: {}, fp: {}, fn: {}".format(tp, tn, fp, fn))

        print("TPR: {}".format(TPR))
        print("TNR: {}".format(TNR))
        print("FPR: {}".format(FPR))
        print("FNR: {}".format(FNR))
        print("PPV: {}".format(PPV))
        print("NPV: {}".format(NPV))
        print("F1Score: {}".format(F1))
        print("AUC: {}".format(AUC))

        print(prefix+"-Accuracy: {} (correct: {}, total: {})".format(acc, correct, total))
    return acc, TPR, TNR, FPR, FNR, PPV, NPV, F1, AUC
