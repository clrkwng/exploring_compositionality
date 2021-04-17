import numpy as np
import math
import torch
import torch.optim as optim
from tqdm import tqdm

import sys
sys.path.insert(0, '../models/')
from model_v1 import *
sys.path.pop(0)
from simple_heatmap import *
from simplebool_data_gen import *

stepsize = 0.1
num_decimals = 1

def main():
	assert torch.cuda.is_available(), "GPU isn't available."
	emb_dims = [2,4]
	X_train, y_train = get_train_data([5000, 5000])

	X_train = torch.tensor(X_train).float()
	y_train = torch.tensor(y_train).view(-1, 1).float()

	train_data = []
	for i in range(len(X_train)):
		train_data.append([X_train[i], y_train[i]])

	batch_size = 256
	trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
	num_batches = math.ceil(1.0 * len(train_data) / batch_size)
	lr=1e-3

	model = WangNet(emb_dims=emb_dims, no_of_cont=2, lin_layer_sizes=[8, 16, 8, 4], \
                output_size=1, hidden_drop_p=0, batch_flag=False).cuda()

	n_epochs = 100
	loss_values = []
	acc_values = []
	norm_values = []
	test_splitA_values = []
	test_splitB_values = []

	pos_weight = torch.FloatTensor([np.count_nonzero(y_train==0), np.count_nonzero(y_train==1)])/len(y_train)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

	# Let's look at how the model does on test split A, during training.
	X_testA, X_testA_orig, y_testA = get_test_splitA(1000)
	X_testA = torch.tensor(X_testA).float().cuda()
	y_testA = torch.tensor(y_testA).view(-1, 1).float().cuda()

	# Let's look at how the model does on test split B, during training.
	X_testB, X_testB_orig, y_testB = get_test_splitB([500,500])
	X_testB = torch.tensor(X_testB).float().cuda()
	y_testB = torch.tensor(y_testB).view(-1, 1).float().cuda()

	# Let's plot test split D, to see how the model guesses during training phase.
	X_testD, X_testD_orig, y_testD = get_test_splitD(1000)
	X_testD = torch.tensor(X_testD).float().cuda()
	y_testD = torch.tensor(y_testD).view(-1, 1).float().cuda()

	t = tqdm(range(1, n_epochs+1), miniters=100)
	steps = 0

	for epoch in t:
		total_loss = 0
		correct = 0
			
		with torch.no_grad():
			predsA = model(X_testA[:,:2].float(), X_testA[:,2].long())
			testA_acc = get_num_correct(y_testA, predsA) / len(X_testA)
			test_splitA_values.append(testA_acc)
									
			predsB = model(X_testB[:,:2].float(), X_testB[:,2].long())
			testB_acc = get_num_correct(y_testB, predsB) / len(X_testB)
			test_splitB_values.append(testB_acc)
					
			predsD = model(X_testD[:,:2].float(), X_testD[:,2].long())
			testD_acc = get_num_correct(y_testD, predsD) / len(X_testD)
	#         save_plt(X_testD_orig, y_testD, predsD, epoch, testD_acc)
					
			get_heatmap(model, epoch, stepsize, num_decimals, lr, "Adam")
				
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			cont_x = inputs[:,:2].float().cuda()
			cat_x = inputs[:,2].long().cuda()
			labels = labels.cuda()

			optimizer.zero_grad()
					
			preds = model(cont_x, cat_x)

			loss = criterion(preds, labels)
			total_loss += loss.item()
					
			loss.backward()
			optimizer.step()
			# scheduler.step()
			correct += get_num_correct(labels, preds)
					
		for name, param in model.named_parameters():
			grad_norm_sum = 0
			if param.requires_grad and param.grad is not None:
				grad = param.grad.view(-1)
				grad_norm_sum += torch.norm(grad).item()
							
		norm_values.append(grad_norm_sum)

		acc = correct/len(train_data)
		t.set_description(f"-----Epoch: {epoch}/{n_epochs}, Loss: {total_loss/num_batches}, Accuracy: {acc}-----")
		loss_values.append(total_loss/num_batches)
		acc_values.append(acc)

	get_heatmap(model, 1, stepsize, num_decimals, lr, "Adam", baseline=True)

if __name__ == "__main__":
	main()