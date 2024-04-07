from src.utils import *



data_path = "data/train.json"


data = load_json(data_path)

print(len(data))


print(data[0].keys())


print(data[0]["full_text"][:50])
print(data[0]["tokens"][:7])
print(data[0]["trailing_whitespace"][:7])
print(data[0]["labels"][:7])


all_labels = []
for instance in data:
	for label in instance['labels']:
		if label not in all_labels:
			all_labels.append(label)


print(all_labels)


# print("######## Phone number")
# for instance in data:
# 	flag = False
# 	for label, token in zip(instance['labels'], instance['tokens']):
# 		if label == "B-PHONE_NUM" or label == "I-PHONE_NUM":
# 			print(label, token)
# 			flag = True
# 	if flag:
# 		print()


# print("######## Personal URL")
# for instance in data:
# 	flag = False
# 	for label, token in zip(instance['labels'], instance['tokens']):
# 		if label == "B-URL_PERSONAL" or label == "I-URL_PERSONAL":
# 			print(label, token)
# 			flag = True
# 	if flag:
# 		print()


# print("######### Street address")
# for instance in data:
# 	flag = False
# 	for label, token in zip(instance['labels'], instance['tokens']):
# 		if label == "B-STREET_ADDRESS" or label == "I-STREET_ADDRESS":
# 			print(label, token)
# 			flag = True
# 	if flag:
# 		print()



# print("######### Id num")
# for instance in data:
# 	flag = False
# 	for label, token in zip(instance['labels'], instance['tokens']):
# 		if label == "B-ID_NUM" or label == "I-ID_NUM":
# 			print(label, token)
# 			flag = True
# 	if flag:
# 		print()


# print("######### Email")
# for instance in data:
# 	flag = False
# 	for label, token in zip(instance['labels'], instance['tokens']):
# 		if label == "B-EMAIL":
# 			print(label, token)
# 			flag = True
# 	if flag:
# 		print()











