from typing import Dict, Self

from transformers import AutoTokenizer


def tokenize(tokenizer: AutoTokenizer, instance: Dict, max_length: int) -> Dict: 
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
	
	tokenized_text = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length, truncation=True)

	for start, end in tokenized_text['offset_mapping']:
		if start == 0 and end == 0:
			token_labels.append('O')
			continue
			
		if token_maps[start] == -1:
			start += 1

		# print(start)
		# print(token_maps)
		token_labels.append(instance['labels'][token_maps[start]])

	return {
		**tokenized_text,
		'token_labels': token_labels,
		'token_maps': token_maps
	}
