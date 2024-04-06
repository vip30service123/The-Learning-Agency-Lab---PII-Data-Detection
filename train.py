from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments

from src.const import label2id, id2label 
from src.dataset import DatasetForProcessedData


def train(config):
	print("#### Prepare dataset.")

	train_ds = DatasetForProcessedData(config.dataset.processed_train_path)
	test_ds = DatasetForProcessedData(config.dataset.processed_test_path)

	train_dl = DataLoader(train_ds, batch_size=config.dataset.bsz, shuffle=config.dataset.is_shuffle)
	test_dl = DataLoader(test_ds, batch_size=config.dataset.bsz, shuffle=config.dataset.is_shuffle)

	model = AutoModelForTokenClassification.from_pretrained(config.model.base_model_name_or_path,
															id2label=id2label,
															label2id=label2id,
															finetuning_task="ner")
	
	print("#### Prepare config.")
	training_args = TrainingArguments(
		output_dir=config.trainer.output_dir,          # output directory
		num_train_epochs=config.trainer.num_train_epochs,              # total number of training epochs
		per_device_train_batch_size=config.trainer.per_device_train_batch_size,  # batch size per device during training
		per_device_eval_batch_size=config.trainer.per_device_eval_batch_size,   # batch size for evaluation
		# warmup_steps=500,                # number of warmup steps for learning rate scheduler
		weight_decay=config.trainer.weight_decay,               # strength of weight decay
		logging_dir=config.trainer.logging_dir,            # directory for storing logs

		evaluation_strategy=config.trainer.evaluation_strategy,
		logging_steps=config.trainer.logging_steps,
		eval_steps=config.trainer.eval_steps,
	)

	print("#### Prepare trainer.")
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_ds,
		eval_dataset=test_ds,
		# data_collator=data_collator,
		# tokenizer=tokenizer,
		# compute_metrics=compute_metrics,
	)

	trainer.train()


if __name__=="__main__":
	config = OmegaConf.load('params.yaml')
	train(config)