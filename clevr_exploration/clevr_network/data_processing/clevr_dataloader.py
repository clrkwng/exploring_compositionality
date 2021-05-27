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
	def __init__(self, data_dir, em_number, batch_size, train_transforms, train_disallowed_combos_json):
		super().__init__()
		self.data_dir = data_dir
		self.em_number = em_number
		self.batch_size = batch_size
		self.train_transforms = train_transforms
		self.train_disallowed_combos_json = train_disallowed_combos_json

	def setup(self, stage):
		# We use the data augmentations for the training data, but val / test do not use them.
		self.clevr_dataset_train = CLEVRDataset(folder_path=self.data_dir + f'train{self.em_number}/', train_flag=True, \
																						train_disallowed_combos_json=self.train_disallowed_combos_json, train_transforms=self.train_transforms)
		self.clevr_dataset_val = CLEVRDataset(folder_path=self.data_dir + f'val{self.em_number}/', train_flag=False)
		self.clevr_dataset_test = CLEVRDataset(folder_path=self.data_dir + f'test{self.em_number}/', train_flag=False)

	# These are responsible for returning the appropriate data split.
	def train_dataloader(self):
		return DataLoader(dataset=self.clevr_dataset_train, batch_size=self.batch_size, shuffle=True, collate_fn=self.my_collate, num_workers=8)

	def val_dataloader(self):
		return DataLoader(dataset=self.clevr_dataset_val, shuffle=False, batch_size=self.batch_size, num_workers=8)

	def test_dataloader(self):
		return DataLoader(dataset=self.clevr_dataset_test, shuffle=False, batch_size=self.batch_size, num_workers=8)

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
