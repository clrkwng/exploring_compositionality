"""
This is a DataModule used for training with a PyTorch Lightning Trainer.
"""

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
		self.clevr_dataset_train = CLEVRDataset(self.data_dir + 'train/', self.specific_attributes_flag)
		self.clevr_dataset_val = CLEVRDataset(self.data_dir + 'val/', self.specific_attributes_flag)
		self.clevr_dataset_test = CLEVRDataset(self.data_dir + 'test/', self.specific_attributes_flag)

	# These are responsible for returning the appropriate data split.
	def train_dataloader(self):
		return DataLoader(self.clevr_dataset_train, batch_size=self.batch_size)

	def val_dataloader(self):
		return DataLoader(self.clevr_dataset_val, batch_size=self.batch_size)

	def test_dataloader(self):
		return DataLoader(self.clevr_dataset_test, batch_size=self.batch_size)