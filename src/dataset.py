import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.const import label2id, id2label


class DatasetForProcessedData(Dataset):
    def __init__(self, df_path):
        self.ds = pd.read_csv(df_path)
        
    def __len__(self):
        return self.ds.shape[0]

    def __getitem__(self, id: int):
        instance = self.ds.iloc[id]

        out = {}
        for col in self.ds.columns:
            if col == "labels":
                out['true_labels'] = eval(instance[col])
            
            elif col == 'token_labels':
                out['labels'] = torch.Tensor(eval(instance[col])).type(torch.int64)

            else:
                try:
                    out[col] = torch.Tensor(eval(instance[col])).type(torch.int64)
                except:
                    out[col] = instance[col]

        return out

