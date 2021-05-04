"""
This file provides different methods that are used across the CLEVR dataset exploration.
"""

import glob
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from pickle_logic import *


NUM_CHANNELS = 3

# Returns np.array of [num_cubes, num_cylinders, num_spheres] from json_path.
# This serves as the label for each image in CLEVRDataset.
def parse_objects_from_json(json_path):
	with open(json_path) as f:
		data = json.load(f)

	num_cubes = 0
	num_cylinders = 0
	num_spheres = 0

	for o in data["objects"]:
		shape_name = o["shape"]
		num_cubes += 1 if shape_name == "cube" else 0
		num_cylinders += 1 if shape_name == "cylinder" else 0
		num_spheres += 1 if shape_name == "sphere" else 0

	return np.array([num_cubes, num_cylinders, num_spheres])

# Takes a CUDA tensor, returns a numpy.
def tensor_to_numpy(tnsr):
	return tnsr.detach().cpu().numpy()

# Calculates the per channel mean and steddev in the train set.
# Was used once, now the mean and std are written in clevr_dataset.py.
def calc_trainset_mean_std(train_path):
	image_list = glob.glob(train_path + "*")

	pixel_num = 0
	channel_sum = np.zeros(NUM_CHANNELS)
	channel_sum_squared = np.zeros(NUM_CHANNELS)

	for img_path in tqdm(image_list):
		im = Image.open(img_path).convert('RGB') # Convert RGBA -> RGB.
		im = np.asarray(im).copy() # This makes it available to be modified.
		im = im / 255.0 # This converts [0,255] -> [0,1].
		pixel_num += (im.size/NUM_CHANNELS) # Increment number of pixels per channel.
		channel_sum += np.sum(im, axis=(0,1))
		channel_sum_squared += np.sum(np.square(im), axis=(0,1))
	
	rgb_mean = channel_sum / pixel_num
	rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))

	save_pickle(rgb_mean, '../pickle_files/rgb_mean.pickle')
	save_pickle(rgb_std, '../pickle_files/rgb_std.pickle')

	return (rgb_mean, rgb_std)

def get_num_correct(preds, labels):
	pred = preds.max(1, keepdim=True)[1]
	num_correct = pred.eq(labels.view_as(pred)).sum().item()
	return num_correct