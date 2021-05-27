"""
This is a custom dataset for the CLEVR dataset. Each data point is
a CLEVR array with label of (num_cubes, num_cylinders, num_spheres).
Each CLEVR array is (256x256x3), where 3 channels are RGB.
"""

import glob, torch
# According to documentation, need to set seed for torch random generator.
torch.manual_seed(17)

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from clevr_data_utils import *

class CLEVRDataset(Dataset):
	def __init__(self, folder_path, train_flag, train_disallowed_combos_json=None, train_transforms=None):
		self.image_path = f"{folder_path}images/"
		self.data_len = len(glob.glob(self.image_path + "*"))

		self.label_path = f"{folder_path}scenes/"

		# These were determined from 10,000 CLEVR images.
		RGB_MEAN = load_pickle("data/rgb_mean.pickle")
		RGB_STD = load_pickle("data/rgb_std.pickle")

		# If in training mode, use a different set of data augmentation.
		self.train_flag = train_flag
		if self.train_flag:
			assert train_transforms is not None, "train_transforms shouldn't be empty."
			self.transform = train_transforms
		else:
			self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(
					mean=RGB_MEAN,
					std=RGB_STD,
				),
			])

		self.train_disallowed_combos_json = train_disallowed_combos_json

	def __len__(self):
		# Note: can hardcode the length that is returned here, to restrict which data is returned.
		return self.data_len
		
	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		# The label is based on what is specified in task_properties.json.
		# Can look at LABEL_FORMAT_LST to see what each label means.
		label_path = self.label_path + f"CLEVR_new_{str(index).zfill(6)}.json"

		# If the image contains a disallowed combo, then we reject it.
		# This will be cleaned up in the collate_fn, defined in clevr_dataloader.py.
		# Only do this for train dataset.
		if self.train_flag and scene_has_disallowed_combo(label_path, self.train_disallowed_combos_json):
			return None

		single_image_path = self.image_path + f"CLEVR_new_{str(index).zfill(6)}.png"
		im = Image.open(single_image_path).convert('RGB') # Convert RGBA -> RGB.
		im = np.asarray(im).copy() # This makes it available to be modified.
		im = self.transform(im)

		concat_label = get_concat_labels(label_path)
		return (im, concat_label)