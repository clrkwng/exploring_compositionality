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

# Takes in total data size, and train percentage. Returns train, validation split.
def get_train_valid_data(data_size, train_pct):
	X_data, y_data = get_train_data(data_size)
	X_data = torch.tensor(X_data).float().cuda()
	y_data = torch.tensor(y_data).long().cuda()

	split_idx = (int)(np.floor(data_size * train_pct))
	X_train, y_train = X_data[:split_idx], y_data[:split_idx]
	X_valid, y_valid = X_data[split_idx:], y_data[split_idx:]

	train_data = []
	for i in range(len(X_train)):
		train_data.append([X_train[i], y_train[i]])

	return train_data, (X_valid, y_valid)

def train_model(model, n_epochs, trainloader, valid_data, num_batches, train_size, save_path):
	loss_values = []
	train_acc_values = []
	val_acc_values = []

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	t = tqdm(range(1, n_epochs+1), miniters=100)
	best_val_acc = 0

	for epoch in t:
		total_loss = 0
		train_correct = 0
		model.train()
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			cont_x = inputs[:, :2].float()
			cat_x = inputs[:, 2:].long()

			optimizer.zero_grad()

			preds = model(cont_x, cat_x)

			loss = criterion(preds, labels)
			total_loss += loss.item()
			train_correct += get_num_correct(preds, labels)

			loss.backward()
			optimizer.step()
		epoch_loss = total_loss/num_batches
		train_acc = train_correct/train_size
		
		# Getting the validation accuracy now.
		val_inputs, val_labels = valid_data
		cont_val = val_inputs[:, :2].float()
		cat_val = val_inputs[:, 2:].long()
		model.eval()
		with torch.no_grad():
			val_preds = model(cont_val, cat_val)
			val_acc = get_num_correct(val_preds, val_labels)/val_inputs.shape[0]

		t.set_description(f"Epoch: {epoch}/{n_epochs}, Loss: {total_loss/num_batches}, Train Acc: {train_acc}, Val Acc: {val_acc}")

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), save_path)

		loss_values.append(epoch_loss)
		train_acc_values.append(train_acc)
		val_acc_values.append(val_acc)

	save_plot(xvalues=np.arange(0, n_epochs), yvalues=[loss_values], xlabel='Epochs',\
		 ylabel='Training Loss', title='Training Loss vs Epochs', file_name='train_loss.png')
	save_plot(xvalues=np.arange(0, n_epochs), yvalues=[train_acc_values, val_acc_values], xlabel='Epochs',\
		 ylabel='Accuracy', title='Accuracy vs Epochs', file_name='accuracy.png')

	save_pickle(loss_values, "../pickled_files/loss_values.pickle")
	save_pickle(train_acc_values, "../pickled_files/acc_values.pickle")

def main():
	assert torch.cuda.is_available(), "GPU isn't available."

	data_size = 40000
	train_data, valid_data = get_train_valid_data(data_size, 0.85)

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