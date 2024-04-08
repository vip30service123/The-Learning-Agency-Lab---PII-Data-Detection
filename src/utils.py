from typing import Dict, List, Tuple
import json
import random

import numpy as np

from src.const import id2label, label2id, rule_base_labels
from src.rule_base import *


def load_json(data_path: str) -> Dict:
	with open(data_path, 'r') as f:
		data = json.load(f)
	return data	


def down_sample_non_labeled_data(data: List[Dict], sample_percentage: float = 0.5, is_equaled: bool = False) -> List[Dict]:
	if sample_percentage and is_equaled:
		raise Exception("Only set either sample_percentage or is_equaled, not both.")

	label_data = []
	non_label_data = []
	for instance in data:
		if any([label != "O" for label in instance["labels"]]):
			label_data.append(instance)
		else:
			non_label_data.append(instance)
	
	if len(label_data) > len(non_label_data):
		raise Exception("Non labeled data are supposed to have more data than labeled data")

	if sample_percentage:
		down_sample_len = int(len(data) * sample_percentage)
	
	if is_equaled:
		down_sample_len = len(label_data)

	non_label_data = random.sample(non_label_data, down_sample_len)
	
	return non_label_data + label_data



def logits_to_pred_tokenized_tokens(predictions: np.array, threshold: float) -> List[int]:
	id_label_O = label2id["O"]

	pred = predictions.argmax(-1)

	predictions_O = predictions[:, :, id_label_O]

	prediction_without_O = predictions.copy()
	prediction_without_O[:, :, id_label_O] = 0
	prediction_without_O = prediction_without_O.argmax(-1)

	final_prediction = np.where(predictions_O < threshold, prediction_without_O, pred)

	return final_prediction


def tokenized_labels_to_token_labels(
		tokenized_labels: List[int], 
		offset_mapping: List[Tuple[int]], 
		token_maps: List[int]
	) -> List[int]:

	token_labels = {}
	for id, (start, end) in enumerate(offset_mapping):
		if start == 0 and end == 0:
			continue

		if token_maps[start] == - 1:
			start += 1
		
		token_labels[token_maps[start]] = tokenized_labels[id]
	
	return list(token_labels.values())


def int_labels_to_str_labels(
	labels: List[int]
) -> List[str]:
	return [id2label[i] for i in labels]


def replace_with_rule_base(
	labels: List[str],
	tokens: List[str],
	trailing_whitespace: List[bool]
) -> List[str]:
	# remove email, phone, url and id labels since I update it with rule base functions
	for id in range(len(labels)):
		if labels in rule_base_labels:
			labels[id] = "O"

	# change labels using rule base functions
	email_ids = find_email_ids(tokens)
	phone_ids = find_phone_number_ids(tokens, trailing_whitespace)
	url_ids = find_urls_ids(tokens)
	id_ids = find_id_ids(tokens)

	for id in email_ids:
		if id < len(labels):
			labels[id] = "B-EMAIL"
	
	for id in phone_ids:
		if id < len(labels):
			labels[id] = "B-PHONE_NUM'"

	for id in url_ids:
		if id < len(labels):
			labels[id] = "B-URL_PERSONAL"

	for id in id_ids:
		if id < len(labels):
			labels[id] = "B-ID_NUM"

	return labels
	
	
