import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from tqdm import tqdm

import sys
from train_vecbool_model import *
from vecbool_data_gen import *
sys.path[0] = '../models/'
from model_v2 import *
sys.path[0] = '../pickled_files/'
from pickle_logic import *

def get_testing_data(test_size, test_split_fn, test_dist):
	X_test, true_labels, y_test = test_split_fn(test_size, test_dist)
	X_test = torch.tensor(X_test).cuda()
	true_labels = torch.tensor(true_labels).cuda()
	y_test = torch.tensor(y_test).cuda()

	cont_x = X_test[:, :2].float()
	cat_x = X_test[:, 2:].long()

	return cont_x, cat_x, true_labels, y_test

def main():
	assert torch.cuda.is_available(), "GPU isn't available."

	model = WangNet(boolvec_dim=boolvec_dim, emb_dims=emb_dims, num_cont=num_cont, lin_layer_sizes=lin_layer_sizes, \
               		output_size=num_classes, hidden_drop_p=hidden_drop_p, batch_flag=batch_flag).cuda()
	save_path = "../saved_model_params/vecbool_model_state_dict.pt"
	model.load_state_dict(torch.load(save_path))

	test_size = 10000
	
	# Getting the max number of flips possible
	max_num_flips = 1
	while True:
		if len(get_neighbor_bools(rep_bools, boolvec_dim, max_num_flips)) == 0:
			break
		max_num_flips += 1
		print(max_num_flips)
	
	print(rep_bools)
	x_values = []
	y_values = []
	for _ in range(10):
		for test_dist in range(1, max_num_flips):
			cont_x, cat_x, true_labels, y_test = get_testing_data(test_size, get_test_splitB, test_dist)

			# This code only keeps testing data where the ground truth label matches rotated label.
			matching_indices = ((y_test == true_labels).nonzero(as_tuple=True)[0])
			if len(matching_indices) != 0:
				cont_x = cont_x[matching_indices]
				cat_x = cat_x[matching_indices]
				y_test = y_test[matching_indices]

				model.eval()
				with torch.no_grad():
					preds = model(cont_x, cat_x)
					print(f"Predictions: {preds.max(1, keepdim=True)[1]}")
					print(f"Labels: {y_test}")
					test_acc = get_num_correct(preds, y_test) / test_size
				x_values.append(int(test_dist))
				y_values.append(test_acc)

				print(f"Num Flips: {test_dist}, Test Accuracy: {test_acc}")

	save_plot(xvalues=x_values, yvalues=[y_values], xlabel='Num Flips',\
						ylabel='Test Accuracy', title='Test Accuracy vs Num Flips', file_name='test_ac.png', fn=plt.scatter)

if __name__ == "__main__":
	main()