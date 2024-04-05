from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, Trainer


from dataset import DatasetForProcessedData


def train(config):
	train_ds = DatasetForProcessedData(config.dataset.processed_train_path)
	test_ds = DatasetForProcessedData(config.dataset.processed_test_path)

	train_dl = DataLoader(train_ds, batch_size=config.train.bsz, shuffle=config.train.is_shuffle)
	test_dl = DataLoader(test_ds, batch_size=config.train.bsz, shuffle=config.train.is_shuffle)



if __name__=="__main__":
	config = OmegaConf.load('config.yaml')
	train(config)