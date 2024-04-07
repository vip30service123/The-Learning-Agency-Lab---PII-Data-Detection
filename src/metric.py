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
from src.utils import tokenized_labels_to_token_labels


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
    def __init__(self, eval_ds, threshold: float, strategy: str = "micro"):
        self.eval_ds = eval_ds
        self.threshold = threshold
        self.strategy = strategy

    def __call__(self, p):
        # predictions, labels = p
        # predictions = predictions.argmax(axis=2)

        # true_predictions = [
        #     [all_labels[p] for (p, l) in zip(prediction, label) if l != 0 and p != 0]
        #     for prediction, label in zip(predictions, labels)
        # ]
        # true_predictions = flatten(true_predictions)

        # true_labels = [
        #     [all_labels[l] for (p, l) in zip(prediction, label) if l != 0 and p != 0]
        #     for prediction, label in zip(predictions, labels)
        # ]
        # true_labels = flatten(true_labels)

        # return f1_score(true_labels, true_predictions, 5, self.strategy)

        predictions, labels = p

        pred = predictions.argmax(-1)

        predictions_O = predictions[:, :, 0]

        prediction_without_O = predictions.copy()
        prediction_without_O[:, :, 0] = 0
        prediction_without_O = prediction_without_O.argmax(-1)

        # final_prediction = torch.where(predictions_O < self.threshold, prediction_without_O, pred).tolist()
        final_prediction = np.where(predictions_O < self.threshold, prediction_without_O, pred)

        # o_index = label2id["O"]
        # preds = predictions.argmax(-1)
        # preds_without_o = predictions.copy()
        # preds_without_o[:,:,o_index] = 0
        # preds_without_o = preds_without_o.argmax(-1)
        # o_preds = predictions[:,:,o_index]
        # preds_final = np.where(o_preds < self.threshold, preds_without_o , preds)


        all_pred_labels = []
        all_true_labels = []

        # for pred, true_labels, offset_mapping, tokens, trailing_whitespace, token_maps in zip(final_prediction, 
        #                                                                           self.eval_ds['true_labels'], 
        #                                                                           self.eval_ds['offset_mapping'], 
        #                                                                           self.eval_ds['tokens'], 
        #                                                                           self.eval_ds['trailing_whitespace'],
        #                                                                           self.eval_ds['token_maps']):

        for pred, instance in zip(final_prediction, self.eval_ds):
            true_labels, offset_mapping, tokens, trailing_whitespace, token_maps = (instance['true_labels'], 
                                                                                    instance['offset_mapping'], 
                                                                                    instance['tokens'], 
                                                                                    instance['trailing_whitespace'],
                                                                                    instance['token_maps'])

            
            # from tokenized predicted labels to predicted labels
            token_pred = tokenized_labels_to_token_labels(pred, offset_mapping, token_maps)

            # from id to label
            token_pred = [id2label[i] for i in token_pred]

            # remove email, phone, url and id labels since I update it with rule base functions
            for id in range(len(token_pred)):
                if token_pred in rule_base_labels:
                    token_pred[id] = "O"

            # change labels using rule base functions
            email_ids = find_email_ids(tokens)
            phone_ids = find_phone_number_ids(tokens, trailing_whitespace)
            url_ids = find_urls_ids(tokens)
            id_ids = find_id_ids(tokens)

            for id in email_ids:
                if id < len(token_pred):
                    token_pred[id] = "B-EMAIL"
            
            for id in phone_ids:
                if id < len(token_pred):
                    token_pred[id] = "B-PHONE_NUM'"

            for id in url_ids:
                if id < len(token_pred):
                    token_pred[id] = "B-URL_PERSONAL"

            for id in id_ids:
                if id < len(token_pred):
                    token_pred[id] = "B-ID_NUM"

            # modify pred len or true len so that they are equaled
            min_len = min(len(token_pred), len(true_labels))

            all_pred_labels.append(token_pred[:min_len])
            all_true_labels.append(true_labels[:min_len])
        
        # flatten then because my f1_score function need that
        all_pred_labels = flatten(all_pred_labels)
        all_true_labels = flatten(all_true_labels)

        return f1_score(
            all_true_labels,
            all_pred_labels,
            5,
            "micro"
        )


