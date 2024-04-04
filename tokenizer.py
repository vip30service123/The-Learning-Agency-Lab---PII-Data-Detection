from typing import Dict, Self

from transformers import AutoTokenizer


class CustomTokenizer:
	def __init__(
		self,
		tokenizer_name_or_path: str,
	) -> Self:
		tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

	def __call__(self, data: Dict) -> Dict: 
		pass
