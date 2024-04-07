from typing import Dict

from transformers import AutoTokenizer


def tokenize(tokenizer: AutoTokenizer, instance: Dict, max_length: int, label2id: Dict) -> Dict: 
	if "trailing_whitespace" not in instance:
		raise Exception("trainling_whitespace is missing.")

	if "tokens" not in instance:
		raise Exception("tokens is missing.")	

	if "labels" not in instance:
		raise Exception("lebels is missing.")

	text = []
	token_maps = []
	token_labels = []
	
	for i, tw in enumerate(instance["trailing_whitespace"]):
		text.append(instance["tokens"][i])
		token_maps += [i] * len(instance["tokens"][i])

		if tw:
			text.append(' ')
			token_maps.append(-1)
	
	tokenized_text = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length, truncation=True, padding="max_length")

	for start, end in tokenized_text['offset_mapping']:
		if start == 0 and end == 0:
			token_labels.append(0)
			continue
			
		if token_maps[start] == -1:
			start += 1

		token_labels.append(label2id[instance['labels'][token_maps[start]]])

	return {
		**tokenized_text,
		'token_labels': token_labels,
		'token_maps': token_maps		
	}
