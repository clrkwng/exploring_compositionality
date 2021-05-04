"""
This is a custom dataset for the CLEVR dataset. Each data point is
a CLEVR array with label of (num_cubes, num_cylinders, num_spheres).
Each CLEVR array is (256x256x3), where 3 channels are RGB.
"""

import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from clevr_data_utils import *

class CLEVRDataset(Dataset):
	def __init__(self, folder_path):
		self.image_path = f"{folder_path}images/"
		self.data_len = len(glob.glob(self.image_path + "*"))

		self.label_path = f"{folder_path}scenes/"

		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.47035027, 0.46635654, 0.45921228],
				std=[0.09705831, 0.09378762, 0.09461603],
			),
		])

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		single_image_path = self.image_path + f"CLEVR_new_{str(index).zfill(6)}.png"
		im = Image.open(single_image_path).convert('RGB') # Convert RGBA -> RGB.
		im = np.asarray(im).copy() # This makes it available to be modified.
		im = self.transform(im)

		label_path = self.label_path + f"CLEVR_new_{str(index).zfill(6)}.json"
		label = parse_objects_from_json(label_path)
		return (im, label)

	def __len__(self):
		# Note: can hardcode the length that is returned here, to restrict which data is returned.
		return self.data_len