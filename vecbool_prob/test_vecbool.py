from comet_ml import Experiment

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
	X_test, y_true, y_test = test_split_fn(test_size, test_dist)

	X_test = torch.tensor(X_test).cuda()
	y_true = torch.tensor(y_true).cuda()
	y_test = torch.tensor(y_test).cuda()

	cont_x = X_test[:, :hyper_params["num_cont"]].float()
	cat_x = X_test[:, hyper_params["num_cont"]:].long()

	return cont_x, cat_x, y_true, y_test

def test_model(model, cont_X_test, cat_X_test, y_true, y_test, num):
	model.eval()
	with torch.no_grad():
		print(cont_X_test)
		print(cat_X_test)
		print("\n")

		preds = model(cont_X_test, cat_X_test)
		test_acc = get_num_correct(preds, y_test, k=0) / len(y_test)
		preds = preds.max(1, keepdim=True)[1]

		y_p = tensor_to_numpy(y_true).reshape((-1,))
		y = tensor_to_numpy(y_test).reshape((-1,))
		y_hat = tensor_to_numpy(preds).reshape((-1,))
		save_3D_plot(y, y_p, y_hat, 'y_plot' + num + '.png', "y", "y_p", "y_hat")

	return test_acc, (y_p, y, y_hat)

def main():
	assert torch.cuda.is_available(), "GPU isn't available."

	model = WangNet(boolvec_dim=hyper_params["boolvec_dim"], emb_dims=hyper_params["emb_dims"], num_cont=hyper_params["num_cont"], \
		lin_layer_sizes=hyper_params["lin_layer_sizes"], output_size=hyper_params["num_classes"], hidden_drop_p=hyper_params["hidden_drop_p"], \
		batch_flag=hyper_params["batch_flag"]).cuda()
	save_path = "../saved_model_params/vecbool_model_state_dict.pt"
	model.load_state_dict(torch.load(save_path))

	test_size = (5 if balanceGTLabelFlag else 1) * 10000
	
	# Getting the max number of flips possible
	max_num_flips = 1
	neighbor_bools_lst = []
	while True:
		nbr_bools = get_neighbor_bools(rep_bools, boolvec_dim, max_num_flips)
		if len(nbr_bools) == 0:
			break
		neighbor_bools_lst.append(nbr_bools)
		max_num_flips += 1
		print(max_num_flips)
	
	x_values = []
	y_values = []
	label_tuples = []
	for test_dist in range(1, max_num_flips):
		for i in range(10):
			cont_x, cat_x, y_true, y_test = get_testing_data(test_size, get_test_splitA, test_dist)
			# This code only keeps testing data where the ground truth label matches rotated label.
			# matching_indices = ((y_test == true_labels).nonzero(as_tuple=True)[0])
			# if len(matching_indices) != 0:
			# 	cont_x = cont_x[matching_indices]
			# 	cat_x = cat_x[matching_indices]
			# 	y_test = y_test[matching_indices]

			test_acc, y_tuple = test_model(model, cont_x, cat_x, y_true, y_test, str(test_dist) + ":" + str(i))
			print(f"Neighbor bools with test_dist {test_dist}: {neighbor_bools_lst[test_dist - 1]}")
			x_values.append(int(test_dist))
			y_values.append(test_acc)

			label_tuples.append(y_tuple)

			print(f"Num Flips: {test_dist}, Test Accuracy: {test_acc}")

	save_plot(xvalues=x_values, yvalues=[y_values], xlabel='Num Flips',\
						ylabel='Test Accuracy', title='Test Accuracy vs Num Flips', file_name='test_ac.png', fn=plt.scatter)

if __name__ == "__main__":
	main()