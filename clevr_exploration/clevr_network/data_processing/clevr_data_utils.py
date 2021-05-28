"""
This file provides different methods that are used across the CLEVR dataset exploration.
"""

import glob, json, itertools
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import *

NUM_CHANNELS = 3
DISALLOWED_LIST = []
CONCAT_LABEL_FORMAT_LST = []

# Returns np.array of [num_cubes, num_cylinders, num_spheres] from json_path.
# Used to serve as the label for each image in CLEVRDataset.
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
def parse_obj_properties_from_json(json_path, task_properties, join_labels_flag):
  # Using a hashmap will allow for duplicate objects, in lieu of a set.
  obj_map = {}
  with open(json_path, 'r') as f:
    data = json.load(f)

  for o in data["objects"]:
    # For the 2|B| embedding case.
    if join_labels_flag:
      # Get the key, which is a tuple of the attributes.
      tuple_props = []
      for prop in task_properties:
        # Exclude the last letter, which is just 's'.
        prop = prop[:-1]
        tuple_props.append(o[prop])
      tuple_props = tuple(tuple_props)

      if tuple_props not in obj_map:
        obj_map[tuple_props] = 0
      obj_map[tuple_props] += 1

    else:
      for prop in task_properties:
        prop = prop[:-1]
        if o[prop] not in obj_map:
          obj_map[o[prop]] = 0
        obj_map[o[prop]] += 1

  return obj_map

def instantiate_concat_label_format_lst():
  global CONCAT_LABEL_FORMAT_LST

  # Opening task_properties.json and properties.json files.
  with open('data/task_properties.json', 'r') as f:
    task_properties = json.load(f)
  with open('../clevr-dataset-gen/image_generation/data/properties.json', 'r') as f:
    properties = json.load(f)

  attribute_lst = [[k for k, _ in properties[prop].items()] for prop in task_properties]

  # Generate list of label formats. This label is for the concat label.
  CONCAT_LABEL_FORMAT_LST = [attribute for prop_list in attribute_lst for attribute in prop_list]  

# Get the labels, based on what is in task_properties.json.
# This will return (concat_label, join_label).
def get_concat_labels(json_path):
  global CONCAT_LABEL_FORMAT_LST

  # Opening task_properties.json and properties.json files.
  with open('data/task_properties.json', 'r') as f:
    task_properties = json.load(f)  
  
  # If the CONCAT_LABEL_FORMAT_LST has not been intialized yet, then initialize it.
  if len(CONCAT_LABEL_FORMAT_LST) == 0:
    instantiate_concat_label_format_lst()

  assert len(CONCAT_LABEL_FORMAT_LST) != 0, "Should be instantiated"
  
  concat_label_vec = [0] * len(CONCAT_LABEL_FORMAT_LST)
  obj_map = parse_obj_properties_from_json(json_path, task_properties, False)
  # For now, label_vec is just a boolean vector (denoting presence of a certain tuple or not).
  # TODO: When looking at presence of N > 1 objects, change the label_vec value to be val of obj_map.
  for i in range(len(CONCAT_LABEL_FORMAT_LST)):
    if CONCAT_LABEL_FORMAT_LST[i] in obj_map:
      concat_label_vec[i] = 1

  return np.array(concat_label_vec)

# Process and return the disallowed combos list.
def get_disallowed_combos_lst(train_disallowed_combos_json):
  with open(train_disallowed_combos_json, 'r') as f:
    combos = json.load(f)
  return [set(c) for c in combos]

# Returns True if the scene contains a disallowed combo, else False.
def scene_has_disallowed_combo(json_path, train_disallowed_combos_json):
  # Initialize DISALLOWED_LIST if it is empty.
  global DISALLOWED_LIST
  if train_disallowed_combos_json and len(DISALLOWED_LIST) == 0:
    DISALLOWED_LIST = get_disallowed_combos_lst(train_disallowed_combos_json)

  # Data is the map contained in json_path.
  with open(json_path, 'r') as f:
    data = json.load(f)

  # Now, return True if any of the objects are disallowed.
  for o in data["objects"]:
    obj_set = set([o["shape"], o["color"], o["material"], o["size"]])
    for combo in DISALLOWED_LIST:
      if combo.issubset(obj_set):
        return True
  return False

# Takes a CUDA tensor, returns it in numpy.
def tensor_to_numpy(tnsr):
  return tnsr.detach().cpu().numpy()

# Calculates the per channel mean and stddev in the train set.
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

# This works when label is a number, and preds are logits.
def single_label_get_num_correct(preds, labels):
  # pred will have the index that corresponds to the largest logit.
  pred = preds.max(1, keepdim=True)[1]
  num_correct = pred.eq(labels.view_as(pred)).sum().item()
  return num_correct

# This works when label is a vector, and the prediction is logits.
def vector_label_get_num_correct(preds, labels):
  # Converts logits into 0 or 1.
  preds = (preds > 0).float()

  # This will sum up the number of correct positions, in each vector.
  # This results in a batch_size length tensor.
  count_correct = (labels == preds).float().sum(1)

  # This gets the number of preds that are completely equal to labels.
  return int((count_correct == labels.shape[1]).float().sum().item())

# Updates the accuracy map passed in, given the list of attributes.
# Each key value pair in accuracy_map should be: attribute: [attribute_correct, attribute_total].
def update_accuracy_map(attribute_set, preds, labels, accuracy_map):
  # Instantiate the CONCAT_LABEL_FORMAT_LST if it is empty.
  if len(CONCAT_LABEL_FORMAT_LST) == 0:
    instantiate_concat_label_format_lst()

  for attribute in attribute_set:
    att_index = CONCAT_LABEL_FORMAT_LST.index(attribute)
    attribute_correct = vector_label_get_num_correct(preds[:,att_index].reshape(-1,1),
                                                     labels[:,att_index].reshape(-1,1))
    attribute_total = len(preds)

    if attribute not in accuracy_map:
      accuracy_map[attribute] = [0, 0, 0]
    accuracy_map[attribute][0] += attribute_correct
    accuracy_map[attribute][1] += attribute_total
    accuracy_map[attribute][2] = get_fair_counts(preds, labels)

  return accuracy_map

# Returns a list in the form: (zero_correct, zero_total, one_correct, one_total).
def get_fair_counts(preds, labels):
  np_lbls = tensor_to_numpy(labels)

  zero_indices = [i for i, x in enumerate(np_lbls) if np.all(x == 0)]
  zero_total = len(zero_indices)
  zero_correct = vector_label_get_num_correct(preds[zero_indices], labels[zero_indices])

  one_indices = [i for i, x in enumerate(np_lbls) if np.all(x == 1)]
  one_total = len(one_indices)
  one_correct = vector_label_get_num_correct(preds[one_indices], labels[one_indices])
  
  assert zero_total + one_total == len(preds) and len(preds) == len(labels), "Missed some labels/predictions."
  return (zero_correct, zero_total, one_correct, one_total)

  