import numpy as np
import torch

# These values were determined by generating a random nn.Embedding 1000 times, then embedding a boolean vector into it.
# Then, these are approximately the max and min values from the embedding, from all runs.
max_val_emb = 4.5
min_val_emb = -4.5

# Returns the positional embedding calculated using the formula from the Transformer model.
def trans_emb_val(i, j, d_model):
	if j % 2 == 0:
		return np.sin(i / (10000 ** (j / d_model)))
	else:
		return np.cos(i / (10000 ** ((j - 1)/d_model)))

# Here, d_model = embed_dim, and N = boolvec_dim.
def get_transformer_embedding(d_model, N):
	trans_tup = []
	for j in range(N):
		emb_tnsr = torch.tensor([trans_emb_val(i, j, d_model) for i in range(d_model)]).cuda()
		trans_tup.append(emb_tnsr)
	return torch.cat(tuple(trans_tup))

# Returns random embedding, based on min and max values of nn.Embedding.
# Returns in size of (num_vecs, vec_size)
def get_random_embedding(num_vecs, vec_size):
	return (max_val_emb - min_val_emb) * torch.rand(num_vecs, vec_size).cuda() + min_val_emb

# Takes a CUDA tensor, returns a numpy.
def tensor_to_numpy(tnsr):
	return tnsr.detach().cpu().numpy()