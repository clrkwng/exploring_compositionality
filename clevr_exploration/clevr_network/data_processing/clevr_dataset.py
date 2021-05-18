"""
This is a custom dataset for the CLEVR dataset. Each data point is
a CLEVR array with label of (num_cubes, num_cylinders, num_spheres).
Each CLEVR array is (256x256x3), where 3 channels are RGB.
"""

import glob

import torch
# According to documentation, need to set seed for torch random generator.
torch.manual_seed(17)

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from clevr_data_utils import *

# These were determined from 10,000 CLEVR images.
RGB_MEAN = [0.47035027, 0.46635654, 0.45921228]
RGB_STD = [0.09705831, 0.09378762, 0.09461603]

class CLEVRDataset(Dataset):
	def __init__(self, folder_path, specific_attributes_flag, train_flag):
		self.image_path = f"{folder_path}images/"
		self.data_len = len(glob.glob(self.image_path + "*"))

		self.label_path = f"{folder_path}scenes/"

		# If in training mode, use a different set of data augmentation.
		if train_flag:
			self.transform = transforms.Compose([
				transforms.ToPILImage(),
				transforms.RandomHorizontalFlip(p=0.5),
				transforms.RandomVerticalFlip(p=0.5),
				transforms.RandomRotation(degrees=30),
				# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=RGB_MEAN,
					std=RGB_STD,
				),
			])
		else:
			self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(
					mean=RGB_MEAN,
					std=RGB_STD,
				),
			])

		# This flag denotes whether to use 2, or 2n embedding setup for the label.
		# If True, will be using 2n embedding, else using 2 embedding.
		self.specific_attributes_flag = specific_attributes_flag

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		single_image_path = self.image_path + f"CLEVR_new_{str(index).zfill(6)}.png"
		im = Image.open(single_image_path).convert('RGB') # Convert RGBA -> RGB.
		im = np.asarray(im).copy() # This makes it available to be modified.
		im = self.transform(im)

		# The label is based on what is specified in task_properties.json.
		# Can look at LABEL_FORMAT_LST to see what each label means.
		label_path = self.label_path + f"CLEVR_new_{str(index).zfill(6)}.json"
		label = get_image_labels(label_path, self.specific_attributes_flag)
		return (im, label)

	def __len__(self):
		# Note: can hardcode the length that is returned here, to restrict which data is returned.
		return self.data_len