"""
This is a DataModule used for training with a PyTorch Lightning Trainer.
"""

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from clevr_dataset import *

BATCH_SIZE = 256
# DataModule makes data reusable and easy to share.
class CLEVRDataModule(pl.LightningDataModule):
	# ALl the code in Lightning makes sure that this method is called from ONLY one GPU.
	def setup(self, stage):
		self.clevr_dataset_train = CLEVRDataset('../clevr-dataset-gen/output/train/')
		self.clevr_dataset_val = CLEVRDataset('../clevr-dataset-gen/output/val/')
		self.clevr_dataset_test = CLEVRDataset('../clevr-dataset-gen/output/test/')

	# These are responsible for returning the appropriate data split.
	def train_dataloader(self):
		return DataLoader(self.clevr_dataset_train, batch_size=BATCH_SIZE)

	def val_dataloader(self):
		return DataLoader(self.clevr_dataset_val, batch_size=BATCH_SIZE)

	def test_dataloader(self):
		return DataLoader(self.clevr_dataset_test, batch_size=BATCH_SIZE)