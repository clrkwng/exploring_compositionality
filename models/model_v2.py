import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import *

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

class WangNet(nn.Module):
	def __init__(self, boolvec_dim, emb_dims, num_cont, lin_layer_sizes, output_size, hidden_drop_p, batch_flag, \
							permute_emb_flag=False, use_param_flag=False, use_rand_emb_flag=False, use_trans_emb_flag=False):
		super().__init__()

		self.boolvec_dim = boolvec_dim
		self.permute_emb_flag = permute_emb_flag
		self.use_param_flag = use_param_flag
		self.use_rand_emb_flag = use_rand_emb_flag
		self.use_trans_emb_flag = use_trans_emb_flag
		
		self.embed_dim = emb_dims[1]
		if self.embed_dim != 0:
			self.embedder = nn.Embedding(num_embeddings=emb_dims[0], embedding_dim=self.embed_dim)

		if self.use_param_flag:
			self.zero_embed = nn.Parameter(torch.rand(self.embed_dim))
			self.one_embed = nn.Parameter(torch.rand(self.embed_dim))
			self.emb_dict = {}

			# Add a random tensor to each of the embedded vectors (from nn.Parameter).
			if self.use_rand_emb_flag:
				self.rand_emb = get_random_embedding(boolvec_dim, self.embed_dim)

			# Add on the Transformer embedding to each of the embedded vectors (from nn.Parameter).
			elif self.use_trans_emb_flag:
				self.trans_emb = get_transformer_embedding(self.embed_dim, boolvec_dim)

		# The flattened embedded size for each boolean vector is now embed_dim * boolvec_dim
		# Also, the continuous data has been scaled up to match categorical input size.
		input_size = self.embed_dim * 2 if emb_dims[1] // emb_dims[0] != 2 else ((self.embed_dim * boolvec_dim) * 2)
		# Layers that will scale the continuous data up to embedded categorical data size.
		self.scale_cont_layers = nn.ModuleList([LBDLayer(num_cont, 8, batch_flag, 0), LBDLayer(8, 32, batch_flag, 0), LBDLayer(32, 64, batch_flag, 0), LBDLayer(64, input_size // 2, batch_flag, 0)])

		# Shouldn't use dropout on the input layer.
		self.lbd_layers = nn.ModuleList([LBDLayer(input_size, lin_layer_sizes[0], batch_flag, 0)] + \
		[LBDLayer(lin_layer_sizes[i], lin_layer_sizes[i+1], batch_flag, hidden_drop_p) for i in range(len(lin_layer_sizes) - 1)])

		self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
		nn.init.kaiming_normal_(self.output_layer.weight.data)

	def forward(self, cont_data, cat_data):
		for l in self.scale_cont_layers:
			cont_data = l(cont_data)

		# Using nn.Parameter flag will override using nn.Embedding conditional check.
		# Note: This only works with boolean vectors that have not been translated to account for (position, value).
		if self.use_param_flag:
			# The following code will generate a dictionary, mapping tuple(boolean) -> the embedded version, using nn.Parameter.
			self.emb_dict = {}
			npy_bools = tensor_to_numpy(cat_data.long())
			unique_bools = np.unique(npy_bools, axis=0)
			# This assert makes sure that the vectors are only bits.
			assert set(np.unique(unique_bools).tolist()) == set([0, 1]), "Boolean vectors were not used."

			for b in unique_bools:
				emb_b_tup = tuple([self.zero_embed if i==0 else self.one_embed for i in b])
				emb_b = torch.cat(emb_b_tup)
				emb_b = torch.unsqueeze(emb_b, 0)

				# If using a random embedding for the parameter experiment, add on the random, pre-intialized embedding.
				if self.use_rand_emb_flag:
					emb_b = emb_b + self.rand_emb.view(emb_b.shape)

				elif self.use_trans_emb_flag:
					emb_b = emb_b + self.trans_emb.view(emb_b.shape)
				
				self.emb_dict[tuple(b)] = emb_b

			# Now, need to translate cat_x into its "embedded" form.
			# If we need to permute each embedding, then handle this separately.
			if self.permute_emb_flag:
				x_tup = []
				for b in npy_bools:
					# Shuffle each parameter around.
					emb_b = self.emb_dict[tuple(b)]
					emb_b = emb_b.reshape((self.boolvec_dim, -1))
					emb_b = emb_b[torch.randperm(emb_b.size()[0])]
					emb_b = emb_b.reshape(1, -1)
					x_tup.append(emb_b)
				x_tup = tuple(x_tup)
			else:
				x_tup = tuple([self.emb_dict[tuple(b)] for b in npy_bools])

			x = torch.cat(x_tup)
			x = torch.cat([cont_data, x], dim=1)

		elif self.embed_dim != 0:
			x = self.embedder(cat_data.long())

			# This will randomly permute the embeddings.
			# Note: this method is trying permutation after the embedding.
			if self.permute_emb_flag:
				for i in range(x.shape[0]):
					tnsr = x[i]
					idx = torch.randperm(len(tnsr))
					x[i] = tnsr[idx]

			x = x.reshape(cat_data.shape[0], -1)
			x = torch.cat([cont_data, x], dim=1)

		else:
			x = cont_data

		for lbd_layer in self.lbd_layers:
			x = lbd_layer(x)

		x = self.output_layer(x)
		return x

	def test_embed(self, cat_data):
		return self.embedder(cat_data.long())