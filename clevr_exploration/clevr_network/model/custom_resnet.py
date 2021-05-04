import torch
import torch.nn as nn

# This model takes modified parts of Residual blocks and composes together.
# It faces the problem of squeezing down too fast at the end during adaptive pool.
class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
		super(ResBlock, self).__init__()
		self.expansion = 4 # Number of channels after a block is always 4 times the number entering the block.
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
		self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
		self.relu = nn.ReLU()
		self.identity_downsample = identity_downsample # Conv layer applied to identity mapping to match shapes.

	def forward(self, x):
		identity = x

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

# ResNet50 has layers = [3, 4, 6, 3], says how many times to use each ResBlock.
# In CLEVR dataset, image_channels = 3.
class WangNetV3(nn.Module):
	def __init__(self, block, layers, image_channels):
		assert len(layers) == 1, "Implementation requires len(layers) == 1."

		super(WangNetV3, self).__init__()
		self.in_channels = 64
		self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		# Residual Block.
		self.layer1 = self._make_layer(ResBlock, layers[0], out_channels=64, stride=1)

		self.avgpool = nn.AdaptiveAvgPool2d((1,1))

		# The three heads to learn num_cubes, num_cylinders, num_spheres respectively.
		# Number of each shape can be 0-11.
		self.cube_layers = nn.Sequential(nn.Linear(64*4, 32),\
																		 nn.ReLU(),
																		 nn.Linear(32, 11))
		self.cylinder_layers = nn.Sequential(nn.Linear(64*4, 32),\
																				 nn.ReLU(),
																				 nn.Linear(32, 11))
		self.sphere_layers = nn.Sequential(nn.Linear(64*4, 32),\
																			 nn.ReLU(),
																			 nn.Linear(32, 11))

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.avgpool(x)
		x = x.reshape(x.shape[0], -1) # Make sure it can go into the fc layer.

		return self.cube_layers(x), self.cylinder_layers(x), self.sphere_layers(x)
	
	def _make_layer(self, block, num_residual_blocks, out_channels, stride):
		identity_downsample = None
		layers = []

		if stride != 1 or self.in_channels != out_channels * 4:
			identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),\
																					nn.BatchNorm2d(out_channels * 4))
		layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
		self.in_channels = out_channels * 4

		for i in range(num_residual_blocks - 1):
			layers.append(block(self.in_channels, out_channels))

		return nn.Sequential(*layers)

def test():
	model = WangNetV3(ResBlock, [1], 3)
	x = torch.randn(2, 3, 256, 256)
	y1, y2, y3 = model(x)
	print(y1, y2, y3)
	print(sum(p.numel() for p in model.parameters() if p.requires_grad))