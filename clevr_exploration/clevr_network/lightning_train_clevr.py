"""
Code for fitting LightningCLEVRClassifier to CLEVRDataModule.
"""
from comet_ml import Experiment

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

	data_module = CLEVRDataModule()
	model = LightningCLEVRClassifier([1, 1, 1, 1], 3)
	trainer = pl.Trainer(
		gpus=1,
		profiler=True,
		logger=comet_logger,
		check_val_every_n_epoch=5,
		max_epochs=100,
	)
	trainer.fit(model, data_module)

if __name__ == "__main__":
	main()