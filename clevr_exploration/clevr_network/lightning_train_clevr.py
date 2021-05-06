"""
Code for fitting LightningCLEVRClassifier to CLEVRDataModule.
"""
from comet_ml import Experiment
import os, os.path

import sys
sys.path.insert(0, 'data_processing/')
from clevr_dataloader import *
sys.path.pop(0)
sys.path.insert(0, 'model/')
from lightning_model import *
sys.path.pop(0)

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

def main():
	comet_logger = CometLogger(
		api_key='5zqkkwKFbkhDgnFn7Alsby6py',
		workspace='clrkwng',
		project_name='clevr-network',
		experiment_name='lightning',
	)

	# Grabs the number of images used in train, val, test.
	train_size = len([n for n in os.listdir('../clevr-dataset-gen/output/train/images/')])
	val_size = len([n for n in os.listdir('../clevr-dataset-gen/output/val/images/')])
	test_size = len([n for n in os.listdir('../clevr-dataset-gen/output/test/images/')])

	data_module = CLEVRDataModule()
	model = LightningCLEVRClassifier(layers=[1, 1, 1, 1], 
																	 image_channels=3, 
																	 batch_size=BATCH_SIZE,
																	 train_size=train_size,
																	 val_size=val_size,
																	 test_size=test_size)
	trainer = pl.Trainer(
		gpus=1,
		profiler=True,
		logger=comet_logger,
		check_val_every_n_epoch=1,
		max_epochs=100,
	)
	trainer.fit(model, data_module)

if __name__ == "__main__":
	main()