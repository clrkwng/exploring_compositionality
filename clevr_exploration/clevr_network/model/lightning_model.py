"""
Implementing the mini ResNet18 using PyTorch Lightning.
"""
from comet_ml import Experiment

import math
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

    # These MLPs are used for cube, cylinder, sphere guesses.
    self.cube_layers = nn.Sequential(nn.Linear(512, 256),\
                                     nn.ReLU(),\
                                     nn.Linear(256, 32),\
                                     nn.ReLU(),\
                                     nn.Linear(32, 11))
    self.cylinder_layers = nn.Sequential(nn.Linear(512, 256),\
                                         nn.ReLU(),\
                                         nn.Linear(256, 32),\
                                         nn.ReLU(),\
                                         nn.Linear(32, 11))
    self.sphere_layers = nn.Sequential(nn.Linear(512, 256),\
                                       nn.ReLU(),\
                                       nn.Linear(256, 32),\
                                       nn.ReLU(),\
                                       nn.Linear(32, 11))

    self.batch_size = batch_size
    self.train_size, self.val_size, self.test_size = train_size, val_size, test_size
    self.num_train_batches = math.ceil(1.0 * self.train_size / self.batch_size)
    self.num_val_batches = math.ceil(1.0 * self.val_size / self.batch_size)
    self.num_test_batches = math.ceil(1.0 * self.test_size / self.batch_size)

    # Initialize some variables used for reporting training and validation accuracies.
    self.best_val_loss = 1e6
    self.save_model_path = 'pickle_files/clevr_model_state_dict.pt'
    self.step = 0

    self.train_cube_correct, self.train_cylinder_correct, self.train_sphere_correct = 0, 0, 0
    self.val_cube_correct, self.val_cylinder_correct, self.val_sphere_correct = 0, 0, 0

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

    return self.cube_layers(x), self.cylinder_layers(x), self.sphere_layers(x)
  
  def configure_optimizers(self):
    # Pass in self.parameters(), since the LightningModule IS the model.
    optimizer = optim.SGD(self.parameters(), lr=LR, momentum=MOMENTUM)
    return optimizer

  def cross_entropy_loss(self, logits, labels):
    return F.cross_entropy(logits, labels)

  def training_step(self, train_batch, batch_idx):
    inputs, labels = train_batch
    cube_labels, cylinder_labels, sphere_labels = labels[:,0], labels[:,1], labels[:,2]
    cube_preds, cylinder_preds, sphere_preds = self.forward(inputs)
    loss = self.cross_entropy_loss(cube_preds, cube_labels) + \
           self.cross_entropy_loss(cylinder_preds, cylinder_labels) + \
           self.cross_entropy_loss(sphere_preds, sphere_labels)
    self.log('train_loss', loss)
    self.step += 1
    
    # Update the number of training correct.
    self.train_cube_correct += get_num_correct(cube_preds, cube_labels)
    self.train_cylinder_correct += get_num_correct(cylinder_preds, cylinder_labels)
    self.train_sphere_correct += get_num_correct(sphere_preds, sphere_labels)

    # If we reach the end of one epoch, log accuracies and zero out the number of correct.
    if batch_idx == self.num_train_batches - 1:
      print(f"Logging Training Accuracy at train_batch {batch_idx}")
      cube_acc = round(self.train_cube_correct/self.train_size, 6)
      cylinder_acc = round(self.train_cylinder_correct/self.train_size, 6)
      sphere_acc = round(self.train_sphere_correct/self.train_size, 6)
      self.logger.experiment.log_metric("train_cube_acc", cube_acc, step=self.step)
      self.logger.experiment.log_metric("train_cylinder_acc", cylinder_acc, step=self.step)
      self.logger.experiment.log_metric("train_sphere_acc", sphere_acc, step=self.step)

      # Reset the number of training correct.
      self.train_cube_correct, self.train_cylinder_correct, self.train_sphere_correct = 0, 0, 0

    return loss

  def validation_step(self, val_batch, batch_idx):
    with self.logger.experiment.validate():
      inputs, labels = val_batch
      cube_labels, cylinder_labels, sphere_labels = labels[:,0], labels[:,1], labels[:,2]
      cube_preds, cylinder_preds, sphere_preds = self.forward(inputs)
      loss = self.cross_entropy_loss(cube_preds, cube_labels) + \
            self.cross_entropy_loss(cylinder_preds, cylinder_labels) + \
            self.cross_entropy_loss(sphere_preds, sphere_labels)
      self.log('val_loss', loss)
      if loss < self.best_val_loss:
        self.best_val_loss = loss
        torch.save(self.state_dict(), self.save_model_path)

      # Update the number of validation correct.
      self.val_cube_correct += get_num_correct(cube_preds, cube_labels)
      self.val_cylinder_correct += get_num_correct(cylinder_preds, cylinder_labels)
      self.val_sphere_correct += get_num_correct(sphere_preds, sphere_labels)

      if batch_idx == self.num_val_batches - 1:
        print(f"Logging Validation Accuracy at val_batch {batch_idx}")
        cube_acc = round(self.val_cube_correct/self.val_size, 6)
        cylinder_acc = round(self.val_cylinder_correct/self.val_size, 6)
        sphere_acc = round(self.val_sphere_correct/self.val_size, 6)
        self.logger.experiment.log_metric("cube_acc", cube_acc, step=self.step)
        self.logger.experiment.log_metric("cylinder_acc", cylinder_acc, step=self.step)
        self.logger.experiment.log_metric("sphere_acc", sphere_acc, step=self.step)

        # Reset the number of validation correct.
        self.val_cube_correct, self.val_cylinder_correct, self.val_sphere_correct = 0, 0, 0


# Testing code below.
# def MiniResNet18(img_channel=3):
#   return LightningCLEVRClassifier([1, 1, 1, 1], img_channel).cuda()

# def test():
#   net = MiniResNet18(img_channel=3)
#   y = net(torch.randn(4, 3, 256, 256).cuda())