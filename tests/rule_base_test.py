import os
import sys
parent_directory = os.path.abspath('..')
sys.path.append(f"{parent_directory}/The-Learning-Agency-Lab---PII-Data-Detection")


import unittest
from src.rule_base import *


class TestRuleBase(unittest.TestCase):
    def test_find_email(self):
        tokens1 = ["I", "have", "an", "email", "which", "is", "johann@gmail.com", ".", "Do", "you", "have", "any", "?"]
        tokens2 = ["What", "is", "that", "?"]

        self.assertEqual(find_email_ids(tokens1), [6], "Wrong email.")
        self.assertEqual(find_email_ids(tokens2), [], "Wrong email.")

    
    def test_find_id(self):
        tokens = ["My", "id", "is", "123456789012", "."]

        self.assertEqual(find_id_ids(tokens), [3], "Wrong id.")

    
    def test_find_urls(self):
        tokens = ["My", "Facebook", "is", "https://facebook.com/johann", ".", "Please", "add", "friend", "me", "."]

        self.assertEqual(find_urls_ids(tokens), [3], "Wrong url.")

    
    def test_find_phone_number_ids(self):
        tokens1 = ["I", "saw", "a", "phone", "number", "on", "the", "wall", ",", "which", "is", "(123", ")432", "-", "2134", ".", "Another", "phone", "number", "is", "(321", ")123", "-", "3213", "."]
        trailing_whitespace1 = [True, True, True, True, True, True, True, False, True, True, True, False, False, False, False, False, True, True, True, True, False, False, False, False, False]

        tokens2 = ["Nothing", "here", "."]
        trailing_whitespace2 = [True, False, False]

        self.assertEqual(find_phone_number_ids(tokens1, trailing_whitespace1), [11, 12, 13, 14, 20, 21, 22, 23], "Wrong number.")
        self.assertEqual(find_phone_number_ids(tokens2, trailing_whitespace2), [], "Wrong numbers.")


if __name__=="__main__":
    unittest.main()