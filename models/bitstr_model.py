import torch
import torch.nn as nn
import torch.nn.functional as F

"""
LBDLayer: Linear+LeakyReLU -> Batchnorm -> Dropout layer
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

class BitNet(nn.Module):
	def __init__(self, boolvec_dim, emb_dims, lin_layer_sizes, output_size, hidden_drop_p, batch_flag):
		super().__init__()

		self.embed_dim = emb_dims[1]
		self.embedder = nn.Embedding(num_embeddings=emb_dims[0], embedding_dim=self.embed_dim)

		input_size = (self.embed_dim * boolvec_dim)
		self.lbd_layers = nn.ModuleList([LBDLayer(input_size, lin_layer_sizes[0], batch_flag, 0)] + \
		[LBDLayer(lin_layer_sizes[i], lin_layer_sizes[i+1], batch_flag, hidden_drop_p) for i in range(len(lin_layer_sizes) - 1)])

		self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
		nn.init.kaiming_normal_(self.output_layer.weight.data)

	def forward(self, cat_data):
		x = self.embedder(cat_data.long())
		x = x.reshape(cat_data.shape[0], -1)
		
		for lbd_layer in self.lbd_layers:
			x = lbd_layer(x)

		x = self.output_layer(x)
		return x