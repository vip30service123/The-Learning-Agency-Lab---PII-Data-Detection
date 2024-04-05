from omegaconf import OmegaConf
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from const import label2id
from tokenizer import tokenize
from utils import load_json


def process(config):
    print("#### Start processing.")
    data = load_json(config.dataset.data_path)

    data_dict = {k: [] for k in data[0].keys()}
    for instance in data:
        for k, v in instance.items():
            data_dict[k].append(v)
    
    df = pd.DataFrame(data_dict)

    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_name_or_path)

    print("#### Start tokenizing.")
    df['input_ids'], df['token_type_ids'], df['attention_mask'], df['offset_mapping'], df['token_labels'], df['token_maps'] = \
        zip(*df.apply(lambda x: list(tokenize(tokenizer, 
                                              {'tokens': x['tokens'], 
                                               'trailing_whitespace': x['trailing_whitespace'], 
                                               'labels': x['labels']}, 
                                              50, 
                                              label2id).values()), axis=1))

    train_df, test_df = train_test_split(df, train_size=config.dataset.train_percentage, random_state=config.dataset.random_seed)
    
    print("#### Save processed data.")
    train_df.to_csv(config.dataset.processed_train_path)
    test_df.to_csv(config.dataset.processed_test_path)

    print("#### Done processing.")

if __name__=="__main__":
    config = OmegaConf.load('config.yaml')
    process(config)