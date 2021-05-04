import torch
import torch.nn as nn
import torchvision.models as models

# This model is just using ResNet18, along with the MLPs for cubes, cylinders, spheres.
class WangNetV4(nn.Module):
	def __init__(self):
		super(WangNetV4, self).__init__()
		self.resnet18 = models.resnet18()
		self.relu = nn.ReLU()

		# The three heads to learn num_cubes, num_cylinders, num_spheres respectively.
		# Number of each shape can be 0-11.
		self.cube_layers = nn.Sequential(nn.Linear(1000, 256),\
																		 nn.ReLU(),\
																		 nn.Linear(256, 32),\
																		 nn.ReLU(),\
																		 nn.Linear(32, 11))
		self.cylinder_layers = nn.Sequential(nn.Linear(1000, 256),\
																		 		 nn.ReLU(),\
																		 		 nn.Linear(256, 32),\
																		 		 nn.ReLU(),\
																		 		 nn.Linear(32, 11))
		self.sphere_layers = nn.Sequential(nn.Linear(1000, 256),\
																		 	 nn.ReLU(),\
																		 	 nn.Linear(256, 32),\
																		 	 nn.ReLU(),\
																		   nn.Linear(32, 11))

	def forward(self, x):
		x = self.resnet18(x)
		return self.cube_layers(x), self.cylinder_layers(x), self.sphere_layers(x)

def test():
	model = WangNetV4()
	print(model)

test()