import re
from typing import List


def find_phone_number_tokens(tokens: List[str], trailing_whitespace: List[bool]) -> List[int]:
	""" Find phone number whose form is (ddd)ddd-dddd."""
	pattern = "\(\d{3}\)\d{3}-\d{4}"	


def find_email(tokens: List[str]) -> List[int]:
	""" Find token which contain @."""
	email_pattern = "\w+@\w+\.\w+"
	
	email_tokens = []

	for id, token in enumerate(tokens):
		if re.search(email_pattern, token):
			email_tokens.append(token)
	
	return email_tokens
			


def find_id(tokens: List[str]) -> List[int]:
	""" Find token id which has the last 12 characters are integer."""
	id_tokens = []

	for id, token in enumerate(tokens):
		if re.search("\d{12}", token[-12:]):
			id_tokens.append(id)

	return id_tokens


def find_urls(tokens: List[str]) -> List[int]:
	""" Find token id which start with http"""
	url_tokens = []

	for id, token in enumerate(tokens):
		if tokens[:4] == 'http':
			url_tokens.append(id)
	
	return url_tokens
