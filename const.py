all_labels = [
    'O', 'B-NAME_STUDENT', 'I-NAME_STUDENT', 'B-URL_PERSONAL', 'B-EMAIL', 'B-ID_NUM', 'I-URL_PERSONAL', 'B-USERNAME', 'B-PHONE_NUM', 'I-PHONE_NUM', 'B-STREET_ADDRESS', 'I-STREET_ADDRESS', 'I-ID_NUM'
]

label2id = {l: i for i, l in enumerate(all_labels)}

id2label = {v: k for k, v in label2id.items()}