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
		if self.batch_flag:
			self.batchnorm = nn.BatchNorm1d(output_size)

		self.dropout = nn.Dropout(drop_p)

	def forward(self, x):
		x = self.linear(x)
		x = self.relu(x)

		if self.batch_flag:
			x = self.batchnorm(x)

		x = self.dropout(x)
		return x

class WangNet(nn.Module):
	def __init__(self, boolvec_dim, emb_dims, num_cont, lin_layer_sizes, output_size, hidden_drop_p, batch_flag):
		super().__init__()
		
		self.embed_dim = emb_dims[1]
		if self.embed_dim != 0:
			self.embedder = nn.Embedding(num_embeddings=emb_dims[0], embedding_dim=self.embed_dim)

		# Layers that will scale the continuous data up to embedded categorical data size.
		self.scale_cont_layers = nn.ModuleList([LBDLayer(num_cont, 8, batch_flag, 0), LBDLayer(8, self.embed_dim * boolvec_dim, batch_flag, 0)]) #, LBDLayer(32, 64, batch_flag, 0), LBDLayer(64, self.embed_dim * boolvec_dim, batch_flag, 0)])
		# Shouldn't use dropout on the input layer.
		# The flattened embedded size for each boolean vector is now embed_dim * boolvec_dim
		# Also, the continuous data has been scaled up to match categorical input size.
		input_size = (self.embed_dim * boolvec_dim) * 2
		self.lbd_layers = nn.ModuleList([LBDLayer(input_size, lin_layer_sizes[0], batch_flag, 0)] + \
		[LBDLayer(lin_layer_sizes[i], lin_layer_sizes[i+1], batch_flag, hidden_drop_p) for i in range(len(lin_layer_sizes) - 1)])

		self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
		nn.init.kaiming_normal_(self.output_layer.weight.data)

	def forward(self, cont_data, cat_data):
		for l in self.scale_cont_layers:
			cont_data = l(cont_data)

		if self.embed_dim != 0:
			x = self.embedder(cat_data.long())
			x = x.reshape(cat_data.shape[0], -1)
			x = torch.cat([cont_data, x], dim=1)
		else:
			x = cont_data

		for lbd_layer in self.lbd_layers:
			x = lbd_layer(x)

		x = self.output_layer(x)
		return x