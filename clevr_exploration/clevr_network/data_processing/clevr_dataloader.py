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
	def __init__(self, data_dir, em_number, batch_size, train_transforms, train_disallowed_combos_json, lvl1_flag=False, concat_disallowed_combos = []):
		super().__init__()
		self.data_dir = data_dir
		self.em_number = em_number
		self.batch_size = batch_size
		self.train_transforms = train_transforms
		self.train_disallowed_combos_json = train_disallowed_combos_json
		self.lvl1_flag = lvl1_flag
		self.concat_disallowed_combos = concat_disallowed_combos

	def setup(self, stage):
		# We use the data augmentations for the training data, but val / test do not use them.
		self.clevr_dataset_train = CLEVRDataset(folder_path=f'{self.data_dir}base-clevr-data{self.em_number}/train/', train_flag=True, \
																						train_disallowed_combos_json=self.train_disallowed_combos_json, train_transforms=self.train_transforms)
		
		# The train_disallowed_combos_json flag is also passed in here, so that in level 1+ experiments, we can get in distribution val/test.
		self.clevr_dataset_val = CLEVRDataset(folder_path=f'{self.data_dir}base-clevr-data{self.em_number}/val/', train_flag=False, \
																						train_disallowed_combos_json=self.train_disallowed_combos_json)
																						
		if self.lvl1_flag:
			# This list comprehension gets list of the datasets to use for out of distribution validation.
			self.out_dist_val_lst = [CLEVRDataset(folder_path=f'{self.data_dir}clevr_datasets/{concat_combo}/', train_flag=False)
														for concat_combo in self.concat_disallowed_combos]

	# These are responsible for returning the appropriate data split.
	def train_dataloader(self):
		return DataLoader(dataset=self.clevr_dataset_train, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_collate, num_workers=4)

	def val_dataloader(self):
		in_dist_loader = DataLoader(dataset=self.clevr_dataset_val, shuffle=False, batch_size=self.batch_size, collate_fn=self.val_collate, num_workers=4)
		if not self.lvl1_flag:
			return in_dist_loader
		else:
			out_dist_loaders = [DataLoader(dataset=out_dist_val, shuffle=False, batch_size=self.batch_size, num_workers=4)
												for out_dist_val in self.out_dist_val_lst]
			out_dist_loaders.insert(0, in_dist_loader)

			# Note: The index of out_dist loaders is offset by +1.
			return out_dist_loaders

	# Custom collate function to be passed into DataLoader.
	# To be only used with train dataset.
	def train_collate(self, batch):
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

	def val_collate(self, batch):
		len_batch = len(batch)

		# Filter out the None values from the batch.
		batch = list(filter(lambda x: x is not None, batch))
		if len_batch > len(batch):
			len_diff = len_batch - len(batch)
			while len_diff != 0:
				poss_val = self.clevr_dataset_val[np.random.randint(0, len(self.clevr_dataset_val))]
				if poss_val is None:
					continue
				batch.append(poss_val)
				len_diff -= 1

		return torch.utils.data.dataloader.default_collate(batch)