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

# Takes in train_size, and val_size. Returns train, validation split.
def get_train_valid_data(train_size, val_size):
	X_train, y_train = get_train_data(train_size)

	# Testing: Remove this line.
	for i in range(len(X_train)):
		X_train[i, num_cont:] = convert_boolvec_to_position_vec(X_train[i, num_cont:])
	print(f"X_train: {X_train}\n")

	X_train = torch.tensor(X_train).float().cuda()
	y_train = torch.tensor(y_train).long().cuda()

	train_data = []
	for i in range(len(X_train)):
		train_data.append([X_train[i], y_train[i]])

	X_valid, y_valid = get_train_data(val_size)
	
	# Testing: Remove this line.
	for i in range(len(X_valid)):
		X_valid[i, num_cont:] = convert_boolvec_to_position_vec(X_valid[i, num_cont:])
	print(f"X_valid: {X_valid}\n")

	X_valid = torch.tensor(X_valid).float().cuda()
	y_valid = torch.tensor(y_valid).long().cuda()

	return train_data, (X_valid, y_valid)

def train_model(model, n_epochs, trainloader, valid_data, num_batches, train_size, save_path):
	loss_values = []
	train_acc_values = []
	val_acc_values = []

	# Testing: Remove this line.
	test_acc_values = []

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	t = tqdm(range(1, n_epochs+1), miniters=100)
	best_val_acc = 0

	# Testing: Remove these lines.
	X_test, y_test = get_test_splitB(10000, 1)
	for i in range(len(X_test)):
		X_test[i, num_cont:] = convert_boolvec_to_position_vec(X_test[i, num_cont:])
	print(f"X_test: {X_test}\n")

	X_test = torch.tensor(X_test).float().cuda()
	y_test = torch.tensor(y_test).long().cuda()

	for epoch in t:
		total_loss = 0
		train_correct = 0
		model.train()
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			cont_x = inputs[:, :num_cont].float()
			cat_x = inputs[:, num_cont:].long()

			optimizer.zero_grad()

			preds = model(cont_x, cat_x)
			loss = criterion(preds, labels)
			total_loss += loss.item()
			train_correct += get_num_correct(preds, labels)

			loss.backward()
			optimizer.step()
		epoch_loss = round(total_loss/num_batches, 6)
		train_acc = round(train_correct/train_size, 6)
		
		# Getting the validation accuracy now.
		val_inputs, val_labels = valid_data
		cont_val = val_inputs[:, :num_cont].float()
		cat_val = val_inputs[:, num_cont:].long()
		model.eval()
		with torch.no_grad():
			val_preds = model(cont_val, cat_val)
			val_acc = round(get_num_correct(val_preds, val_labels)/val_inputs.shape[0], 6)

		# Testing: Remove these lines.
		cont_test = X_test[:, :num_cont].float()
		cat_test = X_test[:, num_cont:].long()
		model.eval()
		with torch.no_grad():
			test_preds = model(cont_test, cat_test)
			test_acc = round(get_num_correct(test_preds, y_test)/X_test.shape[0], 6)

		t.set_description(f"Epoch: {epoch}/{n_epochs}, Loss: {epoch_loss}, Train Acc: {train_acc}, Val Acc: {val_acc}, Test Acc: {test_acc}")

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), save_path)

		loss_values.append(epoch_loss)
		train_acc_values.append(train_acc)
		val_acc_values.append(val_acc)
		test_acc_values.append(test_acc)

	save_plot(xvalues=np.arange(0, n_epochs), yvalues=[loss_values], xlabel='Epochs',\
		 ylabel='Training Loss', title='Training Loss vs Epochs', file_name='train_loss.png', fn=plt.plot)
	save_plot(xvalues=np.arange(0, n_epochs), yvalues=[[train_acc_values, "Training Accuracy"], [val_acc_values, "Validation Accuracy"]], xlabel='Epochs',\
		 ylabel='Accuracy', title='Accuracy vs Epochs', file_name='accuracy.png', fn=plt.plot)

	# Testing: Remove this line.
	save_plot(xvalues=np.arange(0, n_epochs), yvalues=[test_acc_values], xlabel='Epochs',\
			ylabel='Test Accuracy', title='Test Accuracy During Training', file_name='test_acc.png', fn=plt.plot)

	save_pickle(loss_values, "../pickled_files/loss_values.pickle")
	save_pickle(train_acc_values, "../pickled_files/acc_values.pickle")

def main():
	assert torch.cuda.is_available(), "GPU isn't available."

	train_size = 40000
	valid_size = 20000
	train_data, valid_data = get_train_valid_data(train_size, valid_size)

	batch_size = 256
	trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
	num_batches = math.ceil(1.0 * len(train_data) / batch_size)

	n_epochs = 100
	model = WangNet(boolvec_dim=boolvec_dim, emb_dims=emb_dims, num_cont=num_cont, lin_layer_sizes=lin_layer_sizes, \
               		output_size=num_classes, hidden_drop_p=hidden_drop_p, batch_flag=batch_flag).cuda()
	save_path = "../saved_model_params/vecbool_model_state_dict.pt"
	train_model(model, n_epochs, trainloader, valid_data, num_batches, len(train_data), save_path)

	save_pickle(cache, "../pickled_files/stats_cache.pickle")

if __name__ == "__main__":
	main()