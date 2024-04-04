from typing import Dict, Self

from transformers import AutoTokenizer


class CustomTokenizer:
	def __init__(
		self,
		tokenizer_name_or_path: str,
	) -> Self:
		tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

	def __call__(self, instance: Dict) -> Dict: 
		if "trailing_whitespace" not in instance:
			raise Exception("trainling_whitespace is missing.")

		if "tokens" not in instance:
			raise Exception("tokens is missing.")	

		if "labels" not in instance:
			raise Exception("lebels is missing.")

		text = []
		cha_num = []
		token_labels = []
		
		for i, tw in enumerate(instance["trailing_whitespace"]):
			text.append(instance["tokens"][i])
			cha_num.append([i] * len(instance["tokens"][i]))			

			if tw:
				text.append(' ')
				cha_num.append(-1)
		
		tokenized_text = self.tokenizer("".join(text), return_offsets_mapping=True)
	
		for start, end in tokenized_text['offset_mapping']:
			if start == 0 and end == 0:
				continue
				
			if cha_num[start] == -1:
				start += 1

			token_labels.append(instance['labels'][cha[start]])

		return {
			**tokenized_text,
			'token_labels': token_labels,
			'cha_num': cha_num
		}
