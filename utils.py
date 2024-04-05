from typing import Dict, List
import json



def load_json(data_path: str) -> Dict:
	with open(data_path, 'r') as f:
		data = json.load(f)
	return data	


def tokenized_labels_to_token_labels() -> List[str]:
	pass