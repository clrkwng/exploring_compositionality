"""
This file provides different methods that are used across the CLEVR dataset exploration.
"""

import glob, json, torch, itertools
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import *

NUM_CHANNELS = 3
LABEL_FORMAT_LST = []

# Returns np.array of [num_cubes, num_cylinders, num_spheres] from json_path.
# This serves as the label for each image in CLEVRDataset.
def parse_num_objects_from_json(json_path):
  with open(json_path, 'r') as f:
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

# Return a hashmap of the objects in json_path, where elements are tuple in order of task_properties.
# The key is the tuple, and the value is the count.
def parse_obj_properties_from_json(json_path, task_properties):
  # Using a hashmap will allow for duplicate objects, in lieu of a set.
  obj_map = {}
  with open(json_path, 'r') as f:
    data = json.load(f)

  for o in data["objects"]:
    # Get the key, which is a tuple of the attributes.
    tuple_props = []
    for prop in task_properties:
      prop = prop[:-1]
      tuple_props.append(o[prop])
    tuple_props = tuple(tuple_props)

    if tuple_props not in obj_map:
      obj_map[tuple_props] = 0
    obj_map[tuple_props] += 1

  return obj_map

# Get the labels, based on what is in task_properties.json. 
def get_image_labels(json_path):
  global LABEL_FORMAT_LST

  # Opening task_properties.json and properties.json files.
  with open('../data/task_properties.json', 'r') as f:
    task_properties = json.load(f)  
  with open('../../clevr-dataset-gen/image_generation/data/properties.json', 'r') as f:
    properties = json.load(f)

  # If LABEL_FORMAT_LST hasn't been instantiated yet, then populate it.
  if len(LABEL_FORMAT_LST) == 0:
    # Generate the list of label formats. Currently supporting the "2|B|" embedding label.
    attribute_lst = [[k for k, _ in properties[prop].items()] for prop in task_properties]
    LABEL_FORMAT_LST = list(itertools.product(*attribute_lst))
  
  label_vec = [0] * len(LABEL_FORMAT_LST)
  obj_map = parse_obj_properties_from_json(json_path, task_properties)
  # For now, label_vec is just a boolean vector (denoting presence of a certain tuple or not).
  # TODO: When looking at presence of N > 1 objects, change the label_vec value to be val of obj_map.
  for i in range(len(LABEL_FORMAT_LST)):
    if LABEL_FORMAT_LST[i] in obj_map:
      label_vec[i] = 1

  return label_vec

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

  save_pickle(rgb_mean, '../data/rgb_mean.pickle')
  save_pickle(rgb_std, '../data/rgb_std.pickle')

  return (rgb_mean, rgb_std)

def get_num_correct(preds, labels):
  pred = preds.max(1, keepdim=True)[1]
  num_correct = pred.eq(labels.view_as(pred)).sum().item()
  return num_correct