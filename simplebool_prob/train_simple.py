import numpy as np
import math
import torch
import torch.optim as optim
from tqdm import tqdm
import os

import sys
sys.path.insert(0, '../models/')
from model_v1 import *
sys.path.pop(0)
from simple_heatmap import *
from simplebool_data_gen import *

stepsize = 0.1
num_decimals = 1
save_path = "../saved_model_params/simple_model_state_dict.pt"
lin_layers = [8, 16, 8, 4]

def main():
	assert len(sys.argv[1:]) == 2, "Need to pass in lr and optim."
	lr, optimizer = float(sys.argv[1]), sys.argv[2]

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

	model = WangNet(emb_dims=emb_dims, no_of_cont=2, lin_layer_sizes=lin_layers, \
                output_size=1, hidden_drop_p=0, batch_flag=False).cuda()

	n_epochs = 100
	loss_values = []
	acc_values = []
	test_splitA_values = []
	test_splitB_values = []

	criterion = nn.BCEWithLogitsLoss()
	if optimizer == "Adam":
		optimizer = optim.Adam(model.parameters(), lr=lr)
	elif optimizer == "SGD":
		optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
	else:
		print("Optimizer incorrectly passed in!")
		return

	# Let's look at how the model does on test split A, during training.
	X_testA, X_testA_orig, y_testA = get_test_splitA(1000)
	X_testA = torch.tensor(X_testA).float().cuda()
	y_testA = torch.tensor(y_testA).view(-1, 1).float().cuda()

	# Let's look at how the model does on test split B, during training.
	X_testB, X_testB_orig, y_testB = get_test_splitB([500,500])
	X_testB = torch.tensor(X_testB).float().cuda()
	y_testB = torch.tensor(y_testB).view(-1, 1).float().cuda()

	t = tqdm(range(1, n_epochs+1), miniters=100)
	steps = 0

	for epoch in t:
		total_loss = 0
		correct = 0
		t_p = 0
		f_p = 0
		f_n = 0
			
		with torch.no_grad():
			predsA = model(X_testA[:,:2].float(), X_testA[:,2].long())
			testA_acc = get_num_correct(y_testA, predsA) / len(X_testA)
			test_splitA_values.append(testA_acc)
									
			predsB = model(X_testB[:,:2].float(), X_testB[:,2].long())
			testB_acc = get_num_correct(y_testB, predsB) / len(X_testB)
			test_splitB_values.append(testB_acc)
					
			get_heatmap(model, epoch, stepsize, num_decimals, lr)
			print(f"Out of distribution acc: {testA_acc}\n")
			print(f"In distribution acc: {testB_acc}\n")
				
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
			correct += get_num_correct(labels, preds)
			preds_class = get_pred_class(preds)
			t_p += get_t_p(labels, preds_class)
			f_p += get_f_p(labels, preds_class)
			f_n += get_f_n(labels, preds_class)

		acc = round(correct/len(train_data), 6)
		
		# Calculating f_score.
		f_denom = (t_p + .5 * (f_p + f_n))
		if f_denom == 0:
			f_score = "undefined"
		else:
			f_score = round(t_p / f_denom, 6)

		t.set_description(f"--Epoch: {epoch}/{n_epochs}, Loss: {total_loss/num_batches}, Accuracy: {acc}, f-score: {f_score}--")
		loss_values.append(total_loss/num_batches)
		acc_values.append(acc)

	torch.save(model.state_dict(), save_path)
	os.system(f'ffmpeg -r 2 -f image2 -s 1920x1080 -i model_guesses_over_epoch/heatmap0/plot0_%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p model_guesses_over_epoch/videos/bool0.mp4')
	os.system(f'ffmpeg -r 2 -f image2 -s 1920x1080 -i model_guesses_over_epoch/heatmap1/plot1_%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p model_guesses_over_epoch/videos/bool1.mp4')
	os.system(f'ffmpeg -i model_guesses_over_epoch/videos/bool0.mp4 -i model_guesses_over_epoch/videos/bool1.mp4 -filter_complex hstack model_guesses_over_epoch/videos/combined_bool.mp4')

if __name__ == "__main__":
	main()