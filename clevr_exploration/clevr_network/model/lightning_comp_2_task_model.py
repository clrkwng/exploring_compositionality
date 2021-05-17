"""
Implementing the mini ResNet18 using PyTorch Lightning.
This model is used to handle compositionality tasks.
"""
from comet_ml import Experiment

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

import sys
sys.path.insert(0, '../data_processing/')
from clevr_data_utils import *
sys.path.pop(0)

LR = 1e-2
MOMENTUM = 0.9

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
  def __init__(self, layers, image_channels, batch_size, train_size, val_size, test_size):
    super().__init__()

    # Grab the properties we want to output from the model.
    # It is up to the user to input the properties in order, as denoted in properties.json.
    with open('data/task_properties.json', 'r') as f:
      task_properties = json.load(f)
      
    self.shape_flag = "shapes" in task_properties
    self.color_flag = "colors" in task_properties
    self.material_flag = "materials" in task_properties
    self.size_flag = "sizes" in task_properties

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
      self.shape_train_correct, self.shape_val_correct = 0, 0
      self.shape_layers = nn.Sequential(nn.Linear(512, 256),\
                                      nn.ReLU(),\
                                      nn.Linear(256, 32),\
                                      nn.ReLU(),\
                                      nn.Linear(32, 8),\
                                      nn.ReLU(),\
                                      nn.Linear(8, 3))
    if self.color_flag:
      self.color_train_correct, self.color_val_correct = 0, 0
      self.color_layers = nn.Sequential(nn.Linear(512, 256),\
                                      nn.ReLU(),\
                                      nn.Linear(256, 32),\
                                      nn.ReLU(),\
                                      nn.Linear(32, 8))
    if self.material_flag:
      self.material_train_correct, self.material_val_correct = 0, 0
      self.material_layers = nn.Sequential(nn.Linear(512, 256),\
                                      nn.ReLU(),\
                                      nn.Linear(256, 32),\
                                      nn.ReLU(),\
                                      nn.Linear(32, 8),\
                                      nn.ReLU(),\
                                      nn.Linear(8, 2))
    if self.size_flag:
      self.size_train_correct, self.size_val_correct = 0, 0
      self.size_layers = nn.Sequential(nn.Linear(512, 256),\
                                      nn.ReLU(),\
                                      nn.Linear(256, 32),\
                                      nn.ReLU(),\
                                      nn.Linear(32, 8),\
                                      nn.ReLU(),\
                                      nn.Linear(8, 2))

    self.batch_size = batch_size
    self.train_size, self.val_size, self.test_size = train_size, val_size, test_size
    self.num_train_batches = math.ceil(1.0 * self.train_size / self.batch_size)
    self.num_val_batches = math.ceil(1.0 * self.val_size / self.batch_size)
    self.num_test_batches = math.ceil(1.0 * self.test_size / self.batch_size)

    # Initialize some variables used for reporting training and validation accuracies.
    self.best_val_loss = 1e6
    self.save_model_path = 'data/clevr_model_state_dict.pt'
    self.step = 0

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
  
  def forward(self, x):
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
  
  def configure_optimizers(self):
    # Pass in self.parameters(), since the LightningModule IS the model.
    optimizer = optim.SGD(self.parameters(), lr=LR, momentum=MOMENTUM)
    return optimizer

  # Loss function used for this model is BCEWithLogitsLoss().
  def bc_entropy_loss(self, logits, labels):
    logits = logits.float()
    labels = labels.float()
    criterion = torch.nn.BCEWithLogitsLoss()
    val = criterion(logits, labels)
    return val

  # Based on task_properties, split labels/preds into an appropriate tuple.
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
    return losses

  # Update train/val accuracies, based on train_flag's value.
  # NOTE: This method is to be called after using split_lbl_by_properties.
  def update_acc_by_properties(self, preds, labels, train_flag):
    assert len(labels) == len(preds), "Number of labels doesn't match preds."

    idx = 0
    if self.shape_flag:
      shape_correct = vector_label_get_num_correct(preds[idx], labels[idx])
      idx += 1
      if train_flag: self.shape_train_correct += shape_correct
      else: self.shape_val_correct += shape_correct
    if self.color_flag:
      color_correct = vector_label_get_num_correct(preds[idx], labels[idx])
      idx += 1
      if train_flag: self.color_train_correct += color_correct
      else: self.color_val_correct += color_correct
    if self.material_flag:
      material_correct = vector_label_get_num_correct(preds[idx], labels[idx])
      idx += 1
      if train_flag: self.material_train_correct += material_correct
      else: self.material_val_correct += material_correct
    if self.size_flag:
      size_correct = vector_label_get_num_correct(preds[idx], labels[idx])
      idx += 1
      if train_flag: self.size_train_correct += size_correct
      else: self.size_val_correct += size_correct

    assert idx == len(labels), "Not all the properties were accounted for."

  def training_step(self, train_batch, batch_idx):
    inputs, labels = train_batch
    labels = self.split_lbl_by_properties(labels)

    preds = self.forward(inputs)
    
    # Call this loss helper method on the split labels and preds.
    losses = self.get_loss_by_properties(preds=preds, labels=labels)

    # This summing still keeps the gradient fn of the tensor.
    loss = sum(losses)

    self.log('train_loss', loss)
    self.step += 1
    
    # Update the number of training correct.
    self.update_acc_by_properties(preds=preds, labels=labels, train_flag=True)

    # If we reach the end of one epoch, log accuracies and zero out the number of correct.
    if batch_idx == self.num_train_batches - 1:
      print(f"Logging Training Accuracy at train_batch {batch_idx}")

      # Log and reset the number of training correct.
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
      if self.size_flag:
        size_train_acc = round(self.size_train_correct/self.train_size, 6)
        self.logger.experiment.log_metric("size_train_acc", size_train_acc, step=self.step)
        self.size_train_correct = 0

    return loss

  def validation_step(self, val_batch, batch_idx):
    with self.logger.experiment.validate():
      inputs, labels = val_batch
      labels = self.split_lbl_by_properties(labels)

      preds = self.forward(inputs)

      # Call this loss helper method on the split labels and preds.
      losses = self.get_loss_by_properties(preds=preds, labels=labels)
      loss = sum(losses)
      self.log('val_loss', loss)
      if loss < self.best_val_loss:
        self.best_val_loss = loss
        torch.save(self.state_dict(), self.save_model_path)

      # Update the number of validation correct.
      self.update_acc_by_properties(preds=preds, labels=labels, train_flag=False)

      if batch_idx == self.num_val_batches - 1:
        print(f"Logging Validation Accuracy at val_batch {batch_idx}")

        # Log and reset the number of validation correct.
        if self.shape_flag:
          shape_val_acc = round(self.shape_val_correct/self.val_size, 6)
          self.logger.experiment.log_metric("shape_val_acc", shape_val_acc, step=self.step)
          self.shape_val_correct = 0
        if self.color_flag:
          color_val_acc = round(self.color_val_correct/self.val_size, 6)
          self.logger.experiment.log_metric("color_val_acc", color_val_acc, step=self.step)
          self.color_val_correct = 0
        if self.material_flag:
          material_val_acc = round(self.material_val_correct/self.val_size, 6)
          self.logger.experiment.log_metric("material_val_acc", material_val_acc, step=self.step)
          self.material_val_correct = 0
        if self.size_flag:
          size_val_acc = round(self.size_val_correct/self.val_size, 6)
          self.logger.experiment.log_metric("size_val_acc", size_val_acc, step=self.step)
          self.size_val_correct = 0