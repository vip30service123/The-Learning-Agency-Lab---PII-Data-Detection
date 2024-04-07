# Calculate predicted labels from pred
# 1. args max and filter by using threshold
# 2. from tokenized labels to token labels
# 3. remove all label phone, email, id, url
# 4. add rule base phone, email, id, url id
# 5. calculate f1 score
# step 1 -> 4 is crucial for predicting test ds so iit must be written in other file




from typing import Dict, List

from src.const import all_labels
from src.rule_base import *


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn)


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp)


def f1_score(true_labels: List[str], pred_labels: List[str], beta: int = 5, strategy: str = "micro") -> Dict:
    if strategy not in ["micro", "macro", "average"]:
        raise ValueError("strategy must be 'micro', 'macro', 'average'")
    
    all_labels = []
    for label in list(set(pred_labels + true_labels)):
        if label != "0" and label[2:] not in all_labels: # ignore label "O" and only remove prefix from label for simplicity
            all_labels.append(label)

    tp_fp_fn = []
    for label in all_labels:
        tp = sum([int(i == j and i == label) for i, j in zip(true_labels, pred_labels)])
        fp = sum([int(i != label and j == label) for i, j in zip(true_labels, pred_labels)])
        fn = sum([int(i == label and j != label) for i, j in zip(true_labels, pred_labels)])

        tp_fp_fn.append((tp, fp, fn))

    if strategy == "micro":
        avg_tp = sum([i[0] for i in tp_fp_fn]) / len(all_labels)
        avg_fp = sum([i[1] for i in tp_fp_fn]) / len(all_labels)
        avg_fn = sum([i[2] for i in tp_fp_fn]) / len(all_labels)

        rec = recall(avg_tp, avg_fn)
        pre = precision(avg_tp, avg_fp)

        try:
            f1 = (1 + beta ** 2) * rec * pre / (beta ** 2 * pre + rec)
        except:
            f1 = 0
    
    if strategy == "macro":
        pre = sum([precision(i[0], i[1]) for i in tp_fp_fn])
        rec = sum([recall(i[0], i[2]) for i in tp_fp_fn])

        try:
            f1 = (1 + beta ** 2) * rec * pre / (beta ** 2 * pre + rec)
        except:
            f1 = 0

    if strategy == "average":
        all_pre = [precision(i[0], i[1]) for i in tp_fp_fn]
        all_rec = [precision(i[0], i[2]) for i in tp_fp_fn]

        all_f1 = []
        for pre, rec in zip(all_pre, all_rec):
            f1 = (1 + beta ** 2) * rec * pre / (beta ** 2 * pre + rec)
            all_f1.append(f1)
        
        f1 = sum(all) / len(all_labels)

    return {
        "recall": rec,
        "precision": pre,
        "f1": f1
    }       


def flatten(matrix: List[List]) -> List:
    ls = []
    for l in matrix:
        ls += l
    
    return ls


class MetricForPII:
    def __init__(self, threshold: float, strategy: str = "micro"):
        # self.metric = metric
        self.threshold = threshold
        self.strategy = strategy

    def __call__(self, p):
        predictions, labels = p
        predictions = predictions.argmax(axis=2)

        true_predictions = [
            [all_labels[p] for (p, l) in zip(prediction, label) if l != 0 and p != 0]
            for prediction, label in zip(predictions, labels)
        ]
        true_predictions = flatten(true_predictions)

        true_labels = [
            [all_labels[l] for (p, l) in zip(prediction, label) if l != 0 and p != 0]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = flatten(true_labels)

        return f1_score(true_labels, true_predictions, 5, self.strategy)



