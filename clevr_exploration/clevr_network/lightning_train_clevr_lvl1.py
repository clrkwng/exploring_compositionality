"""
Code for fitting LightningCLEVRClassifier to CLEVRDataModule.
"""
from comet_ml import Experiment
import os, os.path, argparse
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

import sys
sys.path.insert(0, 'data_processing/')
from clevr_dataloader import *
from clevr_data_utils import *
sys.path.pop(0)
sys.path.insert(0, 'model/')
from lightning_clevr_lvl1_model import *
sys.path.pop(0)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--batch_size', default=256, type=int,
    help="Batch size that is passed into the dataloader.")
parser.add_argument('--num_epochs', default=100, type=int,
    help="Number of epochs to train the model.")
parser.add_argument('--resnet18_flag', default='False',
		help="True if using resnet18, else False for miniresnet18.")
parser.add_argument('--scheduler', default=None,
		help="Input options are StepLR/CosineAnnealingLR/None for the optimizer scheduler.")
parser.add_argument('--train_disallowed_combos_json', default='data/lvl1_disallowed_combos.json',
		help="Optional path to a JSON file containing combos that are not allowed in the \
		      train data set. This is used for each level of experimentation.")
		
parser.add_argument('--lr', type=float, required=True,
		help="Learning rate used for the optimizer.")
parser.add_argument('--momentum', type=float, required=True,
    help="Momentum used for SGD. If Adam is used, then this argument is ignored.")
parser.add_argument('--optimizer', required=True,
		help="Optimizer used, must choose one of SGD/Adam.")
parser.add_argument('--em_number', required=True, type=int,
		help="Currently supports em2/em3 for train/val/test data.")

def main(args):
	comet_api_key = '5zqkkwKFbkhDgnFn7Alsby6py'
	comet_workspace = 'clrkwng'
	project_name = 'clevr-lvl1-experiments'
	experiment_name = 'exp_lvl1'

	comet_logger = CometLogger(
		api_key=comet_api_key,
		workspace=comet_workspace,
		project_name=project_name,
		experiment_name=experiment_name,
	)

	BATCH_SIZE = args.batch_size
	LR = args.lr
	MOMENTUM = args.momentum
	NUM_EPOCHS = args.num_epochs
	OPTIMIZER = args.optimizer
	SCHEDULER = args.scheduler
	TRAIN_DISALLOWED_COMBOS_JSON = None if args.train_disallowed_combos_json=="None" else args.train_disallowed_combos_json
	RESNET18_FLAG = True if args.resnet18_flag=="True" else False
	em_number = args.em_number if args.em_number in [2,3] else None

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
					
	# Grabs the number of images used in train, val, test.
	# Even with data augmentation, since the dataset is "dynamically" augmented, it's fine to
	# supply the sizes of each dataset manually.
	train_size = len([n for n in os.listdir(f'../clevr-dataset-gen/output/train{em_number}/images/')])
	val_size = len([n for n in os.listdir(f'../clevr-dataset-gen/output/val{em_number}/images/')])
	out_dist_val_size = len([n for n in os.listdir(f'../clevr-dataset-gen/output/lvl1_data/lvl1_val/images/')])
	test_size = len([n for n in os.listdir(f'../clevr-dataset-gen/output/test{em_number}/images/')])
	out_dist_test_size = len([n for n in os.listdir(f'../clevr-dataset-gen/output/lvl1_data/lvl1_test/images/')])

	if TRAIN_DISALLOWED_COMBOS_JSON is not None:
		disallowed_combos_lst = get_disallowed_combos_lst(TRAIN_DISALLOWED_COMBOS_JSON)
	else:
		disallowed_combos_lst = []
	print(f"Train disallowed combos: {disallowed_combos_lst}")

	# Log these params into comet.ml for easier view.
	params = {
		"em_number": em_number,
		"batch_size": BATCH_SIZE,
		"lr": LR,
		"momentum": MOMENTUM,
		"num_epochs": NUM_EPOCHS,
		"optimizer": OPTIMIZER,
		"scheduler": SCHEDULER,
		"resnet18_flag": RESNET18_FLAG,
		"train_transforms": TRAIN_TRANSFORMS,
		"train_disallowed_combos_lst": [list(c) for c in disallowed_combos_lst] # Since set is not JSON serializable.
	}

	comet_logger.log_hyperparams(params)

	data_module = CLEVRDataModule(data_dir='../clevr-dataset-gen/output/',
																em_number=em_number,
																batch_size=BATCH_SIZE, 
																train_transforms=TRAIN_TRANSFORMS, 
																train_disallowed_combos_json=TRAIN_DISALLOWED_COMBOS_JSON,
																lvl1_flag=True)
	model = LightningCLEVRClassifier(resnet18_flag=RESNET18_FLAG,
																	 layers=[1, 1, 1, 1], 
																	 image_channels=3, 
																	 batch_size=BATCH_SIZE,
																	 num_epochs=NUM_EPOCHS,
																	 train_size=train_size,
																	 val_size=val_size,
																	 out_dist_val_size=out_dist_val_size,
																	 test_size=test_size,
																	 out_dist_test_size=out_dist_test_size,
																	 optimizer=OPTIMIZER,
																	 lr=LR,
																	 momentum=MOMENTUM,
																	 scheduler=SCHEDULER)
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
	if '--help' in sys.argv or '-h' in sys.argv:
		parser.print_help()
	else:
		argv = extract_args()
		args = parser.parse_args(argv)
		main(args)