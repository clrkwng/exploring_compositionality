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
from vecbool_data_gen import *
sys.path.insert(0, '../models/')
from model_v2 import *
sys.path[0] = '../pickled_files/'
from pickle_logic import *
sys.path.pop(0)

# Takes in train_size, and val_size. Returns train, validation split.
def get_train_valid_data(train_size, val_size):
	if switchDataSetsFlag:
		X_train, ygt_train, y_train = get_test_splitB(train_size, hyper_params["test_dist"])
		X_valid, ygt_valid, y_valid = get_test_splitB(val_size, hyper_params["test_dist"])
	else:
		X_train, ygt_train, y_train = get_train_data(train_size)
		X_valid, ygt_valid, y_valid = get_test_splitA(val_size)

	print(f"y_train distribution: {np.unique(y_train, return_counts=True)}\n")
	X_train = torch.tensor(X_train).float().cuda()
	y_train = torch.tensor(y_train).long().cuda()

	train_data = []
	for i in range(len(X_train)):
		train_data.append([X_train[i], y_train[i]])

	print(f"y_valid distribution: {np.unique(y_valid, return_counts=True)}\n")
	X_valid = torch.tensor(X_valid).float().cuda()
	y_valid = torch.tensor(y_valid).long().cuda()

	return train_data, (X_valid, y_valid)

def train_model(model, trainloader, valid_data, num_batches, train_size, save_path):
	# Testing: Remove these lines.
	test_size = (5 if balanceGTLabelFlag else 1) * 10000
	if switchDataSetsFlag:
		X_test, ygt_test, y_test = get_train_data(test_size)
	else:
		X_test, ygt_test, y_test = get_test_splitB(test_size, hyper_params["test_dist"])
	print(f"y_test distribution: {np.unique(y_test, return_counts=True)}\n")

	t = tqdm(range(1, hyper_params["num_epochs"]+1), miniters=100)
	best_val_acc = 0

	X_test = torch.tensor(X_test).float().cuda()
	y_test = torch.tensor(y_test).long().cuda()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=hyper_params["lr"])
	count = 0
	with experiment.train():
		step = 0
		for epoch in t:
			total_loss = 0
			train_correct = 0
			train_total = 0
			for i, (inputs, labels) in enumerate(trainloader, 0):
				model.train()
				cont_x = inputs[:, :hyper_params["num_cont"]].float()
				cat_x = inputs[:, hyper_params["num_cont"]:].long()

				# Forward + Backward + Optimize
				optimizer.zero_grad()
				preds = model(cont_x, cat_x)
				loss = criterion(preds, labels)
				loss.backward()
				optimizer.step()

				batch_correct = get_num_correct(preds, labels)
				train_correct += batch_correct
				batch_total = labels.size(0)
				train_total += batch_total
				total_loss += loss.item()

				step += 1
				if step % 5 == 0:
					count += 1
					train_acc = round(train_correct/train_total, 6)
					experiment.log_metric("accuracy", train_acc, step=step)
					# Getting the validation accuracy now.
					with experiment.validate():
						model.eval()
						with torch.no_grad():
							val_inputs, val_labels = valid_data
							cont_val = val_inputs[:, :hyper_params["num_cont"]].float()
							cat_val = val_inputs[:, hyper_params["num_cont"]:].long()
							val_preds = model(cont_val, cat_val)
							val_acc = round(get_num_correct(val_preds, val_labels)/val_inputs.shape[0], 6)
							experiment.log_metric("accuracy", val_acc, step=step)

							if val_acc > best_val_acc:
								best_val_acc = val_acc
								torch.save(model.state_dict(), save_path)

					# Getting the test accuracy now.
					with experiment.test():
						model.eval()
						with torch.no_grad():
							cont_test = X_test[:, :hyper_params["num_cont"]].float()
							cat_test = X_test[:, hyper_params["num_cont"]:].long()	
							test_preds = model(cont_test, cat_test)

							# Looking at the distribution of what the model is predicting.
							# preds_numpy = tensor_to_numpy(test_preds)
							# print(f"Model prediction distribution {np.unique(preds_numpy, return_counts=True)}")

							test_acc = round(get_num_correct(test_preds, y_test, print_preds=False)/X_test.shape[0], 6)
							experiment.log_metric("accuracy_during_training", test_acc, step=step)

			epoch_loss = round(total_loss/num_batches, 6)

			n_epochs = hyper_params["num_epochs"]
			t.set_description(f"Epoch: {epoch}/{n_epochs}, Loss: {epoch_loss}, Train Acc: {train_acc}, Val Acc: {val_acc}, Test Acc: {test_acc}")

	print(count)
			
def main():
	assert torch.cuda.is_available(), "GPU isn't available."

	train_size, valid_size = (5 if balanceGTLabelFlag else 1) * 80000, (5 if balanceGTLabelFlag else 1) * 10000
	train_data, valid_data = get_train_valid_data(train_size, valid_size)

	trainloader = torch.utils.data.DataLoader(dataset=train_data, shuffle=True, batch_size=hyper_params["batch_size"])
	num_batches = math.ceil(1.0 * train_size / hyper_params["batch_size"])

	model = WangNet(boolvec_dim=hyper_params["boolvec_dim"], emb_dims=hyper_params["emb_dims"], num_cont=hyper_params["num_cont"], \
		lin_layer_sizes=hyper_params["lin_layer_sizes"], output_size=hyper_params["num_classes"], hidden_drop_p=hyper_params["hidden_drop_p"], \
		batch_flag=hyper_params["batch_flag"], permute_emb_flag=permute_emb_flag, use_param_flag=use_param_flag, use_rand_emb_flag=use_rand_emb_flag, use_trans_emb_flag=use_trans_emb_flag).cuda()

	# print(model.zero_embed)
	# print(model.one_embed)
	# print(model.trans_emb)
	save_path = "../saved_model_params/vecbool_model_state_dict.pt"
	train_model(model, trainloader, valid_data, num_batches, train_size, save_path)
	# print(model.zero_embed)
	# print(model.one_embed)
	# print(model.trans_emb)


if __name__ == "__main__":
	main()