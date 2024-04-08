# Calculate predicted labels from pred
# 1. args max and filter by using threshold
# 2. from tokenized labels to token labels
# 3. remove all label phone, email, id, url
# 4. add rule base phone, email, id, url id
# 5. calculate f1 score
# step 1 -> 4 is crucial for predicting test ds so iit must be written in other file




from typing import Dict, List

import numpy as np
import torch

from src.const import *
from src.rule_base import *
from src.utils import (
    tokenized_labels_to_token_labels, 
    replace_with_rule_base, 
    int_labels_to_str_labels,
    logits_to_pred_tokenized_tokens
)

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

    # Accuracy
    acc = sum([1 for i, j in zip(true_labels, pred_labels) if i == j]) / len(true_labels)

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
        "f1": f1,
        "accuracy": acc
    }       


def flatten(matrix: List[List]) -> List:
    ls = []
    for l in matrix:
        ls += l
    
    return ls


class MetricForPII:
    def __init__(self, eval_ds, threshold: float, strategy: str = "micro"):
        self.eval_ds = eval_ds
        self.threshold = threshold
        self.strategy = strategy

    def __call__(self, p):

        predictions, labels = p
        final_prediction = logits_to_pred_tokenized_tokens(predictions, self.threshold)


        all_pred_labels = []
        all_true_labels = []

        for pred, instance in zip(final_prediction, self.eval_ds):
            true_labels, offset_mapping, tokens, trailing_whitespace, token_maps = (instance['true_labels'], 
                                                                                    instance['offset_mapping'], 
                                                                                    instance['tokens'], 
                                                                                    instance['trailing_whitespace'],
                                                                                    instance['token_maps'])
            
            # from tokenized predicted labels to predicted labels
            token_pred = tokenized_labels_to_token_labels(pred, offset_mapping, token_maps)

            # from id to label
            token_pred = int_labels_to_str_labels(token_pred)

            token_pred = replace_with_rule_base(token_pred, tokens, trailing_whitespace)

            # modify pred len or true len so that they are equaled
            min_len = min(len(token_pred), len(true_labels))

            all_pred_labels.append(token_pred[:min_len])
            all_true_labels.append(true_labels[:min_len])
        
        # flatten then because my f1_score function need that
        all_pred_labels = flatten(all_pred_labels)
        all_true_labels = flatten(all_true_labels)

        f1 = f1_score(
            all_true_labels,
            all_pred_labels,
            5,
            "micro"
        )

        return f1


