import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from tqdm import tqdm

import sys
from vecbool_data_gen import *
from train_vecbool_model import *
sys.path.insert(0, '../models/')
from model_v2 import *
sys.path[0] = '../pickled_files/'
from pickle_logic import *

# This test distribution is the same as the training distribution.
def get_test_splitA(test_size):
	X_test, X_test_true_labels, y_test = get_train_data(test_size)
	X_test = torch.tensor(X_test).float().cuda()
	y_test = torch.tensor(y_test).long().cuda()
	X_test_true_labels = torch.tensor(X_test_true_labels).long().cuda()

	return X_test, X_test_true_labels, y_test

def main():
	assert torch.cuda.is_available(), "GPU isn't available."

	emb_dims = [2*boolvec_dim, 8*boolvec_dim]
	model = WangNet(emb_dims=emb_dims, no_of_cont=2, lin_layer_sizes=[64, 256, 512, 320, 256, 128, 64, 32, 20], \
               		output_size=num_classes, hidden_drop_p=0, batch_flag=False).cuda()
	save_path = "../saved_model_params/vecbool_model_state_dict.pt"
	model.load_state_dict(torch.load(save_path))

	test_size = 1000
	X_test, X_test_true_labels, y_test = get_test_splitA(test_size)
	cont_x = X_test[:, :2].float()
	cat_x = X_test[:, 2].long()

	model.eval()
	print(len(cont_x))
	with torch.no_grad():
		preds = model(cont_x, cat_x)

	print(get_num_correct(preds, X_test_true_labels))

if __name__ == "__main__":
	main()