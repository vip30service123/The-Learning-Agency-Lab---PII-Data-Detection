import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class DatasetForProcessedData(Dataset):
    def __init__(self, df_path):
        self.ds = pd.read_csv(df_path)
        
    def __len__(self):
        return self.ds.shape[0]

    def __getitem__(self, id: int):
        instance = self.ds.iloc[id]
        
        input_ids = torch.Tensor(eval(instance['input_ids'])).type(torch.int64)
        token_type_ids = torch.Tensor(eval(instance['token_type_ids'])).type(torch.int64)
        attention_mask = torch.Tensor(eval(instance['attention_mask'])).type(torch.int64)
        token_labels = torch.Tensor(eval(instance['token_labels'])).type(torch.int64)

        return {
            "input_ids": input_ids, 
            "token_type_ids": token_type_ids, 
            "attention_mask": attention_mask, 
            "labels": token_labels
        }

