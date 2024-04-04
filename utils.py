from typing import Dict
import json



def load_json(data_path: str) -> Dict:
	with open(data_path, 'r') as f:
		data = json.load(f)
	return data	
