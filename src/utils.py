from typing import Dict, List, Tuple
import json


def load_json(data_path: str) -> Dict:
	with open(data_path, 'r') as f:
		data = json.load(f)
	return data	


def tokenized_labels_to_token_labels(
		offset_mapping: List[Tuple[int]], 
		tokenized_labels: List[int], 
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