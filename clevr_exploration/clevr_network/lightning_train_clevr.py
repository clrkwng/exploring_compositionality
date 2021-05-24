"""
Code for fitting LightningCLEVRClassifier to CLEVRDataModule.
"""
from comet_ml import Experiment
import os, os.path
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

import sys
sys.path.insert(0, 'data_processing/')
from clevr_dataloader import *
from clevr_data_utils import *
sys.path.pop(0)
sys.path.insert(0, 'model/')
from lightning_comp_task_model import *
sys.path.pop(0)

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
	LR = 1e-1
	MOMENTUM = 0.9
	NUM_EPOCHS = 100
	OPTIMIZER = "SGD" # Pass in SGD or Adam.

	RGB_MEAN = load_pickle("data/rgb_mean.pickle")
	RGB_STD = load_pickle("data/rgb_std.pickle")
	TRAIN_TRANSFORMS = transforms.Compose([
						transforms.ToPILImage(),
						transforms.RandomHorizontalFlip(p=0.5), 
						transforms.RandomVerticalFlip(p=0.5), 
						transforms.RandomRotation(degrees=30),
						transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
						transforms.ToTensor(),
						transforms.Normalize(
							mean=RGB_MEAN,
							std=RGB_STD,
						),
					])
	join_labels_flag = True
	data_module = CLEVRDataModule('../clevr-dataset-gen/output/', BATCH_SIZE, join_labels_flag, TRAIN_TRANSFORMS)
	model = LightningCLEVRClassifier(layers=[1, 1, 1, 1], 
																	 image_channels=3, 
																	 batch_size=BATCH_SIZE,
																	 num_epochs=NUM_EPOCHS,
																	 train_size=train_size,
																	 val_size=val_size,
																	 test_size=test_size,
																	 optimizer=OPTIMIZER,
																	 lr=LR,
																	 momentum=MOMENTUM,
																	 join_labels_flag=join_labels_flag)
	trainer = pl.Trainer(
		gpus=1,
		profiler="simple",
		logger=comet_logger,
		num_sanity_val_steps=0,
		check_val_every_n_epoch=1,
		max_epochs=NUM_EPOCHS,
	)
	trainer.fit(model, data_module)
	trainer.test()

if __name__ == "__main__":
	main()