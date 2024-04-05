import re
from typing import List


def find_phone_number_ids(tokens: List[str], trailing_whitespace: List[bool]) -> List[int]:
	""" Find phone number whose form is (ddd)ddd-dddd."""
	pattern = "\(\d{3}\)\d{3}-\d{4}"	

	text = ""
	token_maps = []
	for id, (token, tw) in enumerate(zip(tokens, trailing_whitespace)):
		text += token
		token_maps += [id] * len(token)
		if tw:
			text += " "
			token_maps.append(-1)

	phone_number_token_ids = []
	step = 0 # to know token id in token maps after cutting text
	while True:
		searched_phone_number = re.search(pattern, text)
		if not searched_phone_number:
			break

		start, end = searched_phone_number.span()
		start += step
		end += step

		for i in range(start, end):
			if token_maps[i] not in phone_number_token_ids:
				phone_number_token_ids.append(token_maps[i])
		
		text = text[end:]
		step = end
	
	return phone_number_token_ids


def find_email_ids(tokens: List[str]) -> List[int]:
	""" Find token which contain @."""
	email_pattern = "\w+@\w+\.\w+"
	
	email_token_ids = []

	for id, token in enumerate(tokens):
		if re.search(email_pattern, token):
			email_token_ids.append(id)
	
	return email_token_ids	


def find_id_ids(tokens: List[str]) -> List[int]:
	""" Find token id which has the last 12 characters are integer."""
	id_token_ids = []

	for id, token in enumerate(tokens):
		if re.search("\d{12}", token[-12:]):
			id_token_ids.append(id)

	return id_token_ids


def find_urls_ids(tokens: List[str]) -> List[int]:
	""" Find token id which start with http"""
	url_token_ids = []

	for id, token in enumerate(tokens):
		if token[:4] == 'http':
			url_token_ids.append(id)
	
	return url_token_ids
