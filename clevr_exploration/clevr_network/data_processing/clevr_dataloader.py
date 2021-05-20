"""
This is a DataModule used for training with a PyTorch Lightning Trainer.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from clevr_dataset import *

# DataModule makes data reusable and easy to share.
class CLEVRDataModule(pl.LightningDataModule):
	# ALl the code in Lightning makes sure that this method is called from ONLY one GPU.
	def __init__(self, data_dir, batch_size, specific_attributes_flag):
		super().__init__()
		self.data_dir = data_dir
		self.batch_size = batch_size
		
		# Flag is true for 2n embedding, else n embedding.
		self.specific_attributes_flag = specific_attributes_flag

	def setup(self, stage):
		# We use the data augmentations for the training data, but val / test do not use them.
		self.clevr_dataset_train = CLEVRDataset(folder_path=self.data_dir + 'train/', specific_attributes_flag=self.specific_attributes_flag, train_flag=True)
		self.clevr_dataset_val = CLEVRDataset(folder_path=self.data_dir + 'val/', specific_attributes_flag=self.specific_attributes_flag, train_flag=False)
		self.clevr_dataset_test = CLEVRDataset(folder_path=self.data_dir + 'test/', specific_attributes_flag=self.specific_attributes_flag, train_flag=False)

	# These are responsible for returning the appropriate data split.
	def train_dataloader(self):
		return DataLoader(dataset=self.clevr_dataset_train, batch_size=self.batch_size, shuffle=True, collate_fn=self.my_collate)

	def val_dataloader(self):
		return DataLoader(dataset=self.clevr_dataset_val, shuffle=True, batch_size=self.batch_size)

	def test_dataloader(self):
		return DataLoader(dataset=self.clevr_dataset_test, shuffle=True, batch_size=self.batch_size)

	# Custom collate function to be passed into DataLoader.
	# To be only used with train dataset.
	def my_collate(self, batch):
		len_batch = len(batch)

		# Filter out the None values from the batch.
		batch = list(filter(lambda x: x is not None, batch))
		if len_batch > len(batch):
			len_diff = len_batch - len(batch)
			while len_diff != 0:
				poss_val = self.clevr_dataset_train[np.random.randint(0, len(self.clevr_dataset_train))]
				if poss_val is None:
					continue
				batch.append(poss_val)
				len_diff -= 1

		return torch.utils.data.dataloader.default_collate(batch)
