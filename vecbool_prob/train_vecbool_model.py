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


def get_training_data(train_size):
	X_train, X_train_true_labels, y_train = get_train_data(train_size)
	X_train = torch.tensor(X_train).float().cuda()
	y_train = torch.tensor(y_train).long().cuda()
	X_train_true_labels = torch.tensor(X_train_true_labels).long().cuda()

	train_data = []
	for i in range(len(X_train)):
		train_data.append([X_train[i], X_train_true_labels[i]])

	return train_data

def get_num_correct(preds, labels):
	pred = preds.max(1, keepdim=True)[1]
	correct = pred.eq(labels.view_as(pred)).sum().item()
	return correct

def train_model(model, n_epochs, trainloader, num_batches, train_size, save_path):
	model = model.cuda()
	model.train()
	loss_values = []
	acc_values = []

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	t = tqdm(range(1, n_epochs+1), miniters=100)
	best_acc = 0

	for epoch in t:
		total_loss = 0
		correct = 0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			cont_x = inputs[:, :2].float()
			cat_x = inputs[:, 2].long()

			optimizer.zero_grad()

			preds = model(cont_x, cat_x)

			loss = criterion(preds, labels)
			total_loss += loss.item()
			correct += get_num_correct(preds, labels)

			loss.backward()
			optimizer.step()
		epoch_loss = total_loss/num_batches
		acc = correct/train_size
		t.set_description(f"-----Epoch: {epoch}/{n_epochs}, Loss: {total_loss/num_batches}, Accuracy: {acc}-----")
		loss_values.append(epoch_loss)
		acc_values.append(acc)
		if acc > best_acc:
			best_acc = acc
			torch.save(model.state_dict(), save_path)
	save_pickle(loss_values, "../pickled_files/loss_values.pickle")
	save_pickle(acc_values, "../pickled_files/acc_values.pickle")

def main():
	assert torch.cuda.is_available(), "GPU isn't available."

	train_size = 3000
	train_data = get_training_data(train_size)

	emb_dims = [2*boolvec_dim, 8*boolvec_dim]
	batch_size = 256
	trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
	num_batches = math.ceil(1.0 * len(train_data) / batch_size)

	n_epochs = 100
	model = WangNet(emb_dims=emb_dims, no_of_cont=2, lin_layer_sizes=[64, 256, 512, 320, 256, 128, 64, 32, 20], \
               		output_size=num_classes, hidden_drop_p=0.1, batch_flag=False)	
	save_path = "../saved_model_params/vecbool_model_state_dict.pt"
	train_model(model, n_epochs, trainloader, num_batches, train_size, save_path)


if __name__ == "__main__":
	main()