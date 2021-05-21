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
from lightning_comp_2_task_model import *
sys.path.pop(0)

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

def main():
	comet_logger = CometLogger(
		api_key='5zqkkwKFbkhDgnFn7Alsby6py',
		workspace='clrkwng',
		project_name='clevr-properties',
		experiment_name='data_aug',
	)

	# Grabs the number of images used in train, val, test.
	# Even with data augmentation, since the dataset is "dynamically" augmented, it's fine to
	# supply the sizes of each dataset manually.
	train_size = len([n for n in os.listdir('../clevr-dataset-gen/output/train/images/')])
	val_size = len([n for n in os.listdir('../clevr-dataset-gen/output/val/images/')])
	test_size = len([n for n in os.listdir('../clevr-dataset-gen/output/test/images/')])
	BATCH_SIZE = 256
	LR = 1e-3
	MOMENTUM = 0.9
	specific_attributes_flag = False
	data_module = CLEVRDataModule('../clevr-dataset-gen/output/', BATCH_SIZE, specific_attributes_flag)
	model = LightningCLEVRClassifier(layers=[1, 1, 1, 1], 
																	 image_channels=3, 
																	 batch_size=BATCH_SIZE,
																	 train_size=train_size,
																	 val_size=val_size,
																	 test_size=test_size,
																	 lr=LR,
																	 momentum=MOMENTUM)
	trainer = pl.Trainer(
		gpus=1,
		profiler=True,
		logger=comet_logger,
		num_sanity_val_steps=0,
		check_val_every_n_epoch=1,
		max_epochs=200,
	)
	trainer.fit(model, data_module)
	trainer.test()

if __name__ == "__main__":
	main()