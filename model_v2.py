import torch
import torch.nn as nn
import torch.nn.functional as F

"""
LBDLayer: Linear+ReLU -> Batchnorm -> Dropout layer
Takes in layer input_size and output_size
batch_flag is True if using batchnorm, else False
drop_p denotes dropout probability
"""
class LBDLayer(nn.Module):
	def __init__(self, input_size, output_size, batch_flag, drop_p):
		super().__init__()

		self.linear = nn.Linear(input_size, output_size)
		nn.init.kaiming_normal_(self.linear.weight.data)
		self.relu = nn.LeakyReLU()

		self.batch_flag = batch_flag
		self.batchnorm = nn.BatchNorm1d(output_size)

		self.dropout = nn.Dropout(drop_p)

	def forward(self, x):
		x = self.relu(self.linear(x))

		if self.batch_flag:
			x = self.batchnorm(x)

		x = self.dropout(x)
		return x

class WangNet(nn.Module):
	def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, output_size, hidden_drop_p, batch_flag):
		super().__init__()
		
		# Parameter is telling PyTorch to learn this tensor
		self.embed_dim = emb_dims[1]
		# self.emb = nn.Parameter(torch.zeros(emb_dims[0], self.embed_dim))
		# nn.init.normal_(self.emb , std=0.02)
		self.embedder = nn.Embedding(num_embeddings=emb_dims[0], embedding_dim=self.embed_dim)
		self.normalize_input = nn.BatchNorm1d(no_of_cont, affine=False)

		# Can implement varying dropout_p for each layer (maybe input and hidden layers will differ) later on.
		input_size = self.embed_dim + no_of_cont
		self.lbd_layers = nn.ModuleList([LBDLayer(input_size, lin_layer_sizes[0], batch_flag, hidden_drop_p)] + \
																	[LBDLayer(lin_layer_sizes[i], lin_layer_sizes[i+1], batch_flag, hidden_drop_p) for i in range(len(lin_layer_sizes) - 1)])

		self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
		nn.init.kaiming_normal_(self.output_layer.weight.data)

	def forward(self, cont_data, cat_data):
		cont_data = self.normalize_input(cont_data)

		if self.embed_dim != 0:
			# x = [self.emb[xi] for xi in cat_data]
			# x = torch.stack(x).squeeze()
			x = self.embedder(cat_data.long())
			x = torch.cat([cont_data, x], dim=1)
		else:
			x = cont_data

		for lbd_layer in self.lbd_layers:
			x = lbd_layer(x)

		x = self.output_layer(x)
		return x