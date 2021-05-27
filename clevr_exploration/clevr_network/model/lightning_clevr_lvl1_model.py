"""
Implementing the mini ResNet18 using PyTorch Lightning.
This model is used to handle compositionality tasks.
"""
from comet_ml import Experiment

import math, json, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models

import sys
sys.path.insert(0, '../data_processing/')
from clevr_data_utils import *
sys.path.pop(0)

# This code creates a residual block used in ResNets.
class ResBlock(pl.LightningModule):
  def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(intermediate_channels)
    self.conv2 = nn.Conv2d(intermediate_channels,
                            intermediate_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)
    self.bn2 = nn.BatchNorm2d(intermediate_channels)
    self.relu = nn.ReLU(inplace=True)
    self.identity_downsample = identity_downsample
    self.stride = stride

  def forward(self, x):
    identity = x.clone()

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)

    if self.identity_downsample is not None:
      identity = self.identity_downsample(identity)

    x += identity
    x = self.relu(x)
    return x

class LightningCLEVRClassifier(pl.LightningModule):
  def __init__(self, resnet18_flag, layers, image_channels, batch_size, num_epochs, train_size, val_size, out_dist_val_size, test_size, out_dist_test_size, optimizer, lr, momentum, scheduler):
    super().__init__()

    self.optimizer = optimizer
    self.lr = lr
    self.momentum = momentum
    self.num_epochs = num_epochs
    self.scheduler = scheduler

    # Grab the properties we want to output from the model.
    # It is up to the user to input the properties in order, as denoted in properties.json.
    with open('data/task_properties.json', 'r') as f:
      task_properties = set(json.load(f))
      
    self.shape_flag = "shapes" in task_properties
    self.color_flag = "colors" in task_properties
    self.material_flag = "materials" in task_properties
    self.size_flag = "sizes" in task_properties

    self.resnet18_flag = resnet18_flag

    if resnet18_flag:
      print(f"Using ResNet18.\n")
      self.resnet18 = models.resnet18()
      self.resnet18_lin_layer = nn.Linear(1000, 512)
    else:
      print(f"Using MiniResNet18.\n")
      self.in_channels = 64
      self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.bn1 = nn.BatchNorm2d(64)
      self.relu = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

      # Essentially the entire ResNet architecture are in these 4 lines below.
      self.layer1 = self._make_layer(layers[0], intermediate_channels=64, stride=1)
      self.layer2 = self._make_layer(layers[1], intermediate_channels=128, stride=2)
      self.layer3 = self._make_layer(layers[2], intermediate_channels=256, stride=2)
      self.layer4 = self._make_layer(layers[3], intermediate_channels=512, stride=2)

      self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    # These MLPs are used for shape, color, material, size guesses.
    # The output sizes of these layers are based on num_attributes for each property.
    # TODO: Need to change this hard-coding if we decide to use continuous properties.
    # Also, initialize the num_correct parameters for each property.
    if self.shape_flag:
      self.shape_train_correct, self.shape_val_correct, self.shape_test_correct = 0, 0, 0
      self.shape_layers = nn.Sequential(nn.Linear(512, 256),\
                                      nn.ReLU(),\
                                      nn.Linear(256, 32),\
                                      nn.ReLU(),\
                                      nn.Linear(32, 8),\
                                      nn.ReLU(),\
                                      nn.Linear(8, 3))
    if self.color_flag:
      self.color_train_correct, self.color_val_correct, self.color_test_correct = 0, 0, 0
      self.color_layers = nn.Sequential(nn.Linear(512, 256),\
                                      nn.ReLU(),\
                                      nn.Linear(256, 32),\
                                      nn.ReLU(),\
                                      nn.Linear(32, 8))
    if self.material_flag:
      self.material_train_correct, self.material_val_correct, self.material_test_correct = 0, 0, 0
      self.material_layers = nn.Sequential(nn.Linear(512, 256),\
                                      nn.ReLU(),\
                                      nn.Linear(256, 32),\
                                      nn.ReLU(),\
                                      nn.Linear(32, 8),\
                                      nn.ReLU(),\
                                      nn.Linear(8, 2))

      # These hashmaps are used for calculating the "fair" metric accuracy.
      # The faircorrect_maps are used to increment the number of correct predictions for train/val/test.
      # The count_maps are used to increment the number of times that label appears in train/val/test.
      self.material_train_faircorrect_map = {}
      self.material_val_faircorrect_map = {}
      self.material_test_faircorrect_map = {}

      self.material_train_count_map = {}
      self.material_val_count_map = {}
      self.material_test_count_map = {}

    if self.size_flag:
      self.size_train_correct, self.size_val_correct, self.size_test_correct = 0, 0, 0
      self.size_layers = nn.Sequential(nn.Linear(512, 256),\
                                      nn.ReLU(),\
                                      nn.Linear(256, 32),\
                                      nn.ReLU(),\
                                      nn.Linear(32, 8),\
                                      nn.ReLU(),\
                                      nn.Linear(8, 2))

      # These hashmaps are used for calculating the "fair" metric accuracy.
      self.size_train_faircorrect_map = {}
      self.size_val_faircorrect_map = {}
      self.size_test_faircorrect_map = {}

      self.size_train_count_map = {}
      self.size_val_count_map = {}
      self.size_test_count_map = {}

    # These are used to get the accuracy on the concatenated label.
    # That is, the concatenated correctness of [shape, color, material, size].
    self.concat_label_train_correct, self.concat_label_val_correct, self.concat_label_test_correct = 0, 0, 0

    self.batch_size = batch_size
    self.train_size, self.val_size, self.test_size = train_size, val_size, test_size
    self.num_train_batches = math.ceil(1.0 * self.train_size / self.batch_size)
    self.num_val_batches = math.ceil(1.0 * self.val_size / self.batch_size)
    self.num_test_batches = math.ceil(1.0 * self.test_size / self.batch_size)

    # Initialize some variables used for reporting training and validation accuracies.
    self.best_val_loss = 1e6
    self.save_model_path = 'data/clevr_model_state_dict.pt'
    self.step = 0

    # Initialize these variables for tracking attribute accuracy in level 1.
    # NOTE: These are hard-coded, since in level 1 we exclude ["small", "rubber"].
    self.out_dist_val_size, self.out_dist_test_size = out_dist_val_size, out_dist_test_size
    self.num_out_dist_val_batches = math.ceil(1.0 * self.out_dist_val_size / self.batch_size)
    self.num_out_dist_train_batches = math.ceil(1.0 * self.out_dist_test_size / self.batch_size)
    self.attribute_set = set(["cube", "sphere", "cylinder", "rubber", "metal"])
    self.val_attribute_accuracy_map = {}
    self.test_attribute_accuracy_map = {}

  def _make_layer(self, num_residual_blocks, intermediate_channels, stride):
    identity_downsample = None
    layers = []

    # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
    # we need to adapt the Identity (skip connection) so it will be able to be added
    # to the layer that's ahead
    if stride != 1 or self.in_channels != intermediate_channels:
      identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,
                                                    intermediate_channels,
                                                    kernel_size=1,
                                                    stride=stride,
                                                    bias=False),
                                          nn.BatchNorm2d(intermediate_channels),)
    layers.append(ResBlock(self.in_channels, intermediate_channels, identity_downsample, stride))
    self.in_channels = intermediate_channels

    # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
    # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
    # and also same amount of channels.
    for i in range(num_residual_blocks - 1):
      layers.append(ResBlock(self.in_channels, intermediate_channels))

    return nn.Sequential(*layers)
  
  def configure_optimizers(self):
    # Pass in self.parameters(), since the LightningModule IS the model.
    if self.optimizer == "SGD":
      optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
    elif self.optimizer == "Adam":
      optimizer = optim.Adam(self.parameters(), lr=self.lr)
    else:
      print("Optimizer must be either SGD or Adam.")
      sys.exit(-1)

    if self.scheduler == None:
      return optimizer
    elif self.scheduler == "StepLR":
      scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.9)
    elif self.scheduler == "CosineAnnealingLR":
      scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.num_epochs)
    else:
      print("Scheduler must be StepLR/CosineAnnealingLR/None.")
      sys.exit(-1)
    return [optimizer], [scheduler]

  # Loss function used for this model is BCEWithLogitsLoss().
  def bc_entropy_loss(self, logits, labels):
    logits = logits.float()
    labels = labels.float()
    criterion = torch.nn.BCEWithLogitsLoss()
    return criterion(logits, labels)

  # Based on task_properties, split labels into an appropriate tuple.
  # NOTE: lbls is 2-dimensional, based on batch size.
  def split_lbl_by_properties(self, lbls):
    return_lst = []
    # Now, get the parts of lbls according to which properties are present.
    if self.shape_flag:
      return_lst.append(lbls[:, :3])
      lbls = lbls[:, 3:]
    if self.color_flag:
      return_lst.append(lbls[:, :8])
      lbls = lbls[:, 8:]
    if self.material_flag:
      return_lst.append(lbls[:, :2])
      lbls = lbls[:, 2:]
    if self.size_flag:
      return_lst.append(lbls[:, :2])
      lbls = lbls[:, 2:]

    # Here, should assert that lbls is now empty.
    assert lbls.shape[1] == 0, "lbls is not empty yet."
    return return_lst

  # This method returns a loss tensor, based on which properties are being tested.
  # NOTE: This method is to be called after using split_lbl_by_properties.
  def get_loss_by_properties(self, preds, labels):
    assert len(labels) == len(preds), "Number of labels doesn't match preds."

    # idx is used in place of popping, so that preds and labels are not empty after this method.
    losses = []
    idx = 0
    if self.shape_flag:
      losses.append(self.bc_entropy_loss(preds[idx], labels[idx]))
      idx += 1
    if self.color_flag:
      losses.append(self.bc_entropy_loss(preds[idx], labels[idx]))
      idx += 1
    if self.material_flag:
      losses.append(self.bc_entropy_loss(preds[idx], labels[idx]))
      idx += 1
    if self.size_flag:
      losses.append(self.bc_entropy_loss(preds[idx], labels[idx]))
      idx += 1
    
    assert idx == len(labels), "Not all the properties were accounted for."

    # This summing will still keep the gradient fn.
    return sum(losses)

  # This method will update the number of predictions that were correct.
  # Used to calculate the "fair" metric accuracy.
  def update_fair_count(self, preds, lbls, prop, run_flag):
    np_lbls = tensor_to_numpy(lbls)
    # Grab all the unique labels present in lbls.
    unique_lbls = np.unique(np_lbls, axis=0)
    total_lbls_count = 0
    for u_lbl in unique_lbls:
      u_lbl_indices = [i for i, x in enumerate(np_lbls) if np.all(x == u_lbl)]

      # Use this to increment the count_maps.
      num_lbls = len(u_lbl_indices)
      total_lbls_count += num_lbls

      num_correct = vector_label_get_num_correct(preds[u_lbl_indices], lbls[u_lbl_indices])

      # Now, update the appropriate map with the number of correct predictions.
      map_key = tuple(u_lbl)
      if run_flag == "train":
        if prop == "material":
          if map_key not in self.material_train_faircorrect_map: self.material_train_faircorrect_map[map_key] = 0
          self.material_train_faircorrect_map[map_key] += num_correct

          if map_key not in self.material_train_count_map: self.material_train_count_map[map_key] = 0
          self.material_train_count_map[map_key] += num_lbls

        elif prop == "size":
          if map_key not in self.size_train_faircorrect_map: self.size_train_faircorrect_map[map_key] = 0
          self.size_train_faircorrect_map[map_key] += num_correct

          if map_key not in self.size_train_count_map: self.size_train_count_map[map_key] = 0
          self.size_train_count_map[map_key] += num_lbls

        else:
          print("prop must be size or material.")
          sys.exit(-1)

      elif run_flag == "val":
        if prop == "material":
          if map_key not in self.material_val_faircorrect_map: self.material_val_faircorrect_map[map_key] = 0
          self.material_val_faircorrect_map[map_key] += num_correct

          if map_key not in self.material_val_count_map: self.material_val_count_map[map_key] = 0
          self.material_val_count_map[map_key] += num_lbls

        elif prop == "size":
          if map_key not in self.size_val_faircorrect_map: self.size_val_faircorrect_map[map_key] = 0
          self.size_val_faircorrect_map[map_key] += num_correct

          if map_key not in self.size_val_count_map: self.size_val_count_map[map_key] = 0
          self.size_val_count_map[map_key] += num_lbls

        else:
          print("prop must be size or material.")
          sys.exit(-1)

      elif run_flag == "test":
        if prop == "material":
          if map_key not in self.material_test_faircorrect_map: self.material_test_faircorrect_map[map_key] = 0
          self.material_test_faircorrect_map[map_key] += num_correct

          if map_key not in self.material_test_count_map: self.material_test_count_map[map_key] = 0
          self.material_test_count_map[map_key] += num_lbls

        elif prop == "size":
          if map_key not in self.size_test_faircorrect_map: self.size_test_faircorrect_map[map_key] = 0
          self.size_test_faircorrect_map[map_key] += num_correct

          if map_key not in self.size_test_count_map: self.size_test_count_map[map_key] = 0
          self.size_test_count_map[map_key] += num_lbls

        else:
          print("prop must be size or material.")
          sys.exit(-1)

      else:
        print("run_flag must be train/val/test.")
        sys.exit(-1)

    assert total_lbls_count == len(np_lbls), "Overcounted some labels."

  # Update train/val accuracies, based on train_flag's value.
  # NOTE: This method is to be called after using split_lbl_by_properties.
  def update_acc_by_properties(self, preds, labels, eval_mode):
    assert len(labels) == len(preds), "Number of labels doesn't match preds."

    idx = 0
    if self.shape_flag:
      shape_correct = vector_label_get_num_correct(preds[idx], labels[idx])
      if eval_mode == "train": self.shape_train_correct += shape_correct
      elif eval_mode == "val": self.shape_val_correct += shape_correct
      elif eval_mode == "test": self.shape_test_correct += shape_correct
      else:
        print("Entered incorrect eval_mode.")
        sys.exit(-1)
      idx += 1
    if self.color_flag:
      color_correct = vector_label_get_num_correct(preds[idx], labels[idx])
      if eval_mode == "train": self.color_train_correct += color_correct
      elif eval_mode == "val": self.color_val_correct += color_correct
      elif eval_mode == "test": self.color_test_correct += color_correct
      else:
        print("Entered incorrect eval_mode.")
        sys.exit(-1)
      idx += 1
    if self.material_flag:
      material_correct = vector_label_get_num_correct(preds[idx], labels[idx])
      self.update_fair_count(preds=preds[idx], lbls=labels[idx], prop="material", run_flag=eval_mode)
      if eval_mode == "train": self.material_train_correct += material_correct
      elif eval_mode == "val": self.material_val_correct += material_correct
      elif eval_mode == "test": self.material_test_correct += material_correct
      else:
        print("Entered incorrect eval_mode.")
        sys.exit(-1)
      idx += 1
    if self.size_flag:
      size_correct = vector_label_get_num_correct(preds[idx], labels[idx])
      self.update_fair_count(preds=preds[idx], lbls=labels[idx], prop="size", run_flag=eval_mode)
      if eval_mode == "train": self.size_train_correct += size_correct
      elif eval_mode == "val": self.size_val_correct += size_correct
      elif eval_mode == "test": self.size_test_correct += size_correct
      else:
        print("Entered incorrect eval_mode.")
        sys.exit(-1)
      idx += 1

    assert idx == len(labels), "Not all the properties were accounted for."

  def forward(self, x):
    if self.resnet18_flag:
      x = self.resnet18(x)
      x = self.resnet18_lin_layer(x)

    else:
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)

      x = self.avgpool(x)
      x = x.reshape(x.shape[0], -1)
    
    # Get the return tuple of the properties wanted.
    pred_lst = []
    if self.shape_flag:
      pred_lst.append(self.shape_layers(x))
    if self.color_flag:
      pred_lst.append(self.color_layers(x))
    if self.material_flag:
      pred_lst.append(self.material_layers(x))
    if self.size_flag:
      pred_lst.append(self.size_layers(x))

    return pred_lst

  def training_step(self, train_batch, batch_idx):
    inputs, concat_labels = train_batch
    concat_preds = self.forward(inputs)

    marginal_labels = self.split_lbl_by_properties(concat_labels)    
    # Call this loss helper method on the split labels and preds.
    loss = self.get_loss_by_properties(preds=concat_preds, labels=marginal_labels)
  
    # Update the number of training correct.
    self.update_acc_by_properties(preds=concat_preds, labels=marginal_labels, eval_mode="train")
    # Need to use torch.cat to get concat_preds and concat_labels to the same dimension.
    self.concat_label_train_correct += vector_label_get_num_correct(torch.cat(concat_preds, dim=1), concat_labels)

    self.log('train_loss', loss)
    self.step += 1

    # If we reach the end of one epoch, log accuracies and zero out the number of correct.
    if batch_idx == self.num_train_batches - 1:
      print(f"Logging training accuracy at train_batch {batch_idx}")

      if self.shape_flag:
        shape_train_acc = round(self.shape_train_correct/self.train_size, 6)
        self.logger.experiment.log_metric("shape_train_acc", shape_train_acc, step=self.step)
        self.shape_train_correct = 0

      if self.color_flag:
        color_train_acc = round(self.color_train_correct/self.train_size, 6)
        self.logger.experiment.log_metric("color_train_acc", color_train_acc, step=self.step)
        self.color_train_correct = 0

      if self.material_flag:
        material_train_acc = round(self.material_train_correct/self.train_size, 6)
        self.logger.experiment.log_metric("material_train_acc", material_train_acc, step=self.step)
        self.material_train_correct = 0

        material_train_fair_acc = 0
        for lbl, correct_count in self.material_train_faircorrect_map.items():
          total_count = self.material_train_count_map[lbl]
          assert total_count >= correct_count, "Incorrectly incremented maps."
          material_train_fair_acc += correct_count / total_count
        material_train_fair_acc /= len(self.material_train_count_map)

        self.material_train_faircorrect_map.clear()
        self.material_train_count_map.clear()

        self.logger.experiment.log_metric("material_train_fair_acc", material_train_fair_acc, step=self.step)

      if self.size_flag:
        size_train_acc = round(self.size_train_correct/self.train_size, 6)
        self.logger.experiment.log_metric("size_train_acc", size_train_acc, step=self.step)
        self.size_train_correct = 0

        size_train_fair_acc = 0
        for lbl, correct_count in self.size_train_faircorrect_map.items():
          total_count = self.size_train_count_map[lbl]
          assert total_count >= correct_count, "Incorrectly incremented maps."
          size_train_fair_acc += correct_count / total_count
        size_train_fair_acc /= len(self.size_train_count_map)

        self.size_train_faircorrect_map.clear()
        self.size_train_count_map.clear()

        self.logger.experiment.log_metric("size_train_fair_acc", size_train_fair_acc, step=self.step)

      concat_label_train_acc = round(self.concat_label_train_correct/self.train_size, 6)
      self.logger.experiment.log_metric("concat_label_train_acc", concat_label_train_acc, step=self.step)
      self.concat_label_train_correct = 0

    return loss

  def validation_step(self, val_batch, batch_idx, dataset_idx):
    with self.logger.experiment.validate():
      # First, handle the in distribution accuracies.
      if dataset_idx == 0:
        inputs, concat_labels = val_batch
        concat_preds = self.forward(inputs)

        marginal_labels = self.split_lbl_by_properties(concat_labels)
        # Call this loss helper method on the split labels and preds.
        loss = self.get_loss_by_properties(preds=concat_preds, labels=marginal_labels)

        # Update the number of validation correct.
        self.update_acc_by_properties(preds=concat_preds, labels=marginal_labels, eval_mode="val")
        self.concat_label_val_correct += vector_label_get_num_correct(torch.cat(concat_preds, dim=1), concat_labels)

        self.log('val_loss', loss)
        if loss < self.best_val_loss:
          self.best_val_loss = loss
          torch.save(self.state_dict(), self.save_model_path)

        if batch_idx == self.num_val_batches - 1:
          print(f"Logging in dist validation accuracy at val_batch {batch_idx}")
          
          if self.shape_flag:
            shape_val_acc = round(self.shape_val_correct/self.val_size, 6)
            self.logger.experiment.log_metric("shape_acc", shape_val_acc, step=self.step)
            self.shape_val_correct = 0

          if self.color_flag:
            color_val_acc = round(self.color_val_correct/self.val_size, 6)
            self.logger.experiment.log_metric("color_acc", color_val_acc, step=self.step)
            self.color_val_correct = 0

          if self.material_flag:
            material_val_acc = round(self.material_val_correct/self.val_size, 6)
            self.logger.experiment.log_metric("material_acc", material_val_acc, step=self.step)
            self.material_val_correct = 0

            material_val_fair_acc = 0
            for lbl, correct_count in self.material_val_faircorrect_map.items():
              total_count = self.material_val_count_map[lbl]
              assert total_count >= correct_count, "Incorrectly incremented maps."
              material_val_fair_acc += correct_count / total_count
            material_val_fair_acc /= len(self.material_val_count_map)

            self.material_val_faircorrect_map.clear()
            self.material_val_count_map.clear()

            self.logger.experiment.log_metric("material_val_fair_acc", material_val_fair_acc, step=self.step)

          if self.size_flag:
            size_val_acc = round(self.size_val_correct/self.val_size, 6)
            self.logger.experiment.log_metric("size_acc", size_val_acc, step=self.step)
            self.size_val_correct = 0

            size_val_fair_acc = 0
            for lbl, correct_count in self.size_val_faircorrect_map.items():
              total_count = self.size_val_count_map[lbl]
              assert total_count >= correct_count, "Incorrectly incremented maps."
              size_val_fair_acc += correct_count / total_count
            size_val_fair_acc /= len(self.size_val_count_map)

            self.size_val_faircorrect_map.clear()
            self.size_val_count_map.clear()

            self.logger.experiment.log_metric("size_val_fair_acc", size_val_fair_acc, step=self.step)

          concat_label_val_acc = round(self.concat_label_val_correct/self.val_size, 6)
          self.logger.experiment.log_metric("concat_label_acc", concat_label_val_acc, step=self.step)
          self.concat_label_val_correct = 0
      elif dataset_idx == 1:
        # Now, handle the out of distribution batch of data.
        inputs, concat_labels = val_batch
        concat_preds = self.forward(inputs)

        marginal_labels = self.split_lbl_by_properties(concat_labels)
        # Call this loss helper method on the split labels and preds.

        self.val_attribute_accuracy_map = update_accuracy_map(attribute_set=self.attribute_set, 
                                                          preds=torch.cat(concat_preds, dim=1), 
                                                          labels=concat_labels, accuracy_map=self.val_attribute_accuracy_map)
        
        if batch_idx == self.num_out_dist_val_batches - 1:
          print(f"Logging out dist validation accuracy at val_batch {batch_idx}")
          for attribute, counts in self.val_attribute_accuracy_map.items():
            att_acc = round(counts[0]/counts[1], 6)
            self.logger.experiment.log_metric(f"out_dist_{attribute}_acc", att_acc, step=self.step)
          self.val_attribute_accuracy_map.clear()

  def test_step(self, test_batch, batch_idx, dataset_idx):
    with self.logger.experiment.test():
      # First, handle the in distribution accuracies.
      if dataset_idx == 0:
        inputs, concat_labels = test_batch
        concat_preds = self.forward(inputs)

        marginal_labels = self.split_lbl_by_properties(concat_labels)

        # Update the number of test correct.
        self.update_acc_by_properties(preds=concat_preds, labels=marginal_labels, eval_mode="test")        
        self.concat_label_test_correct += vector_label_get_num_correct(torch.cat(concat_preds, dim=1), concat_labels)

        if batch_idx == self.num_test_batches - 1:
          print(f"Logging test accuracy at test_batch {batch_idx}")

          if self.shape_flag:
            shape_test_acc = round(self.shape_test_correct/self.test_size, 6)
            self.logger.experiment.log_metric("test_shape_acc", shape_test_acc, step=self.step)
            self.shape_test_correct = 0

          if self.color_flag:
            color_test_acc = round(self.color_test_correct/self.test_size, 6)
            self.logger.experiment.log_metric("test_color_acc", color_test_acc, step=self.step)
            self.color_test_correct = 0

          if self.material_flag:
            material_test_acc = round(self.material_test_correct/self.test_size, 6)
            self.logger.experiment.log_metric("test_material_acc", material_test_acc, step=self.step)
            self.material_test_correct = 0

            material_test_fair_acc = 0
            for lbl, correct_count in self.material_test_faircorrect_map.items():
              total_count = self.material_test_count_map[lbl]
              assert total_count >= correct_count, "Incorrectly incremented maps."
              material_test_fair_acc += correct_count / total_count
            material_test_fair_acc /= len(self.material_test_count_map)

            self.material_test_faircorrect_map.clear()
            self.material_test_count_map.clear()

            self.logger.experiment.log_metric("material_test_fair_acc", material_test_fair_acc, step=self.step)

          if self.size_flag:
            size_test_acc = round(self.size_test_correct/self.test_size, 6)
            self.logger.experiment.log_metric("test_size_acc", size_test_acc, step=self.step)
            self.size_test_correct = 0

            size_test_fair_acc = 0
            for lbl, correct_count in self.size_test_faircorrect_map.items():
              total_count = self.size_test_count_map[lbl]
              assert total_count >= correct_count, "Incorrectly incremented maps."
              size_test_fair_acc += correct_count / total_count
            size_test_fair_acc /= len(self.size_test_count_map)

            self.size_test_faircorrect_map.clear()
            self.size_test_count_map.clear()
            
            self.logger.experiment.log_metric("size_test_fair_acc", size_test_fair_acc, step=self.step)

          concat_label_test_acc = round(self.concat_label_test_correct/self.test_size, 6)
          self.logger.experiment.log_metric("test_concat_label_acc", concat_label_test_acc, step=self.step)
          self.concat_label_test_correct = 0

      elif dataset_idx == 1:
        # Now, handle the out of distribution batch of data.
        inputs, concat_labels = test_batch
        concat_preds = self.forward(inputs)

        marginal_labels = self.split_lbl_by_properties(concat_labels)
        # Call this loss helper method on the split labels and preds.

        self.test_attribute_accuracy_map = update_accuracy_map(attribute_set=self.attribute_set, 
                                                          preds=torch.cat(concat_preds, dim=1), 
                                                          labels=concat_labels, accuracy_map=self.test_attribute_accuracy_map)
        if batch_idx == self.num_out_dist_test_batches - 1:
          print(f"Logging out dist validation accuracy at val_batch {batch_idx}")
          for attribute, counts in self.test_attribute_accuracy_map.items():
            att_acc = round(counts[0]/counts[1], 6)
            self.logger.experiment.log_metric(f"out_dist_{attribute}_acc", att_acc, step=self.step)
          self.test_attribute_accuracy_map.clear()