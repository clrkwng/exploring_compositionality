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

def get_testing_data(test_size):
	X_test, y_test = get_test_splitA(test_size)
	X_test = torch.tensor(X_test).cuda()
	y_test = torch.tensor(y_test).cuda()

	cont_x = X_test[:, :2].float()
	cat_x = X_test[:, 2:].long()

	return cont_x, cat_x, y_test

def main():
	assert torch.cuda.is_available(), "GPU isn't available."
	model = WangNet(boolvec_dim=boolvec_dim, emb_dims=emb_dims, num_cont=num_cont, lin_layer_sizes=lin_layer_sizes, \
               		output_size=num_classes, hidden_drop_p=hidden_drop_p, batch_flag=batch_flag).cuda()
	save_path = "../saved_model_params/vecbool_model_state_dict.pt"
	model.load_state_dict(torch.load(save_path))

	test_size = 1000
	cont_x, cat_x, y_test = get_testing_data(test_size)

	model.eval()
	with torch.no_grad():
		preds = model(cont_x, cat_x)

	print(f"Test Accuracy: {get_num_correct(preds, y_test)/test_size}")

if __name__ == "__main__":
	main()