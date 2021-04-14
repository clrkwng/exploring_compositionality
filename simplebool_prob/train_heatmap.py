import numpy as np
import torch
import pandas as pd
import seaborn as sns

from simplebool_data_gen import *

zeropts_map = {}
onepts_map = {}

# Takes a CUDA tensor, returns a numpy.
def tensor_to_numpy(tnsr):
	return tnsr.detach().cpu().numpy()

# Saves two heatmaps, per epoch.
def get_heatmap(model, epoch, num_samples, lr, optimizer, baseline=False):
	global zeropts_map
	global onepts_map

	# Here, the key is tuple of quadrant and the value is number of 0's guessed by model in that quadrant.
	zero_map = {}
	one_map = {}

	for low_x in range(-10, 10):
		for low_y in range(-10, 10):
			if (low_x, low_y) not in zeropts_map or (low_x, low_y) not in onepts_map:
				assert (low_x, low_y) not in zeropts_map and (low_x, low_y) not in onepts_map, "Messed up."
				# Sample num_samples points in this current quadrant.
				x_pts = np.random.uniform(low=low_x, high=low_x + 1, size=num_samples).reshape(-1,1)
				y_pts = np.random.uniform(low=low_y, high=low_y + 1, size=num_samples).reshape(-1,1)
				pts = np.concatenate((x_pts, y_pts), axis=1)
				
				# Now, create (x,y,0) and (x,y,1) data points.
				zero_pts = np.concatenate((np.copy(pts), np.zeros((num_samples, 1))), axis=1)
				one_pts = np.concatenate((np.copy(pts), np.ones((num_samples, 1))), axis=1)

				# Create the tensors that can be fed into the model.
				zero_tnsr = torch.tensor(zero_pts).float()
				one_tnsr = torch.tensor(one_pts).float()

				zeropts_map[(low_x, low_y)] = zero_tnsr
				onepts_map[(low_x, low_y)] = one_tnsr
			else:
				zero_tnsr = zeropts_map[(low_x, low_y)]
				one_tnsr = onepts_map[(low_x, low_y)]

			# Get the predictions from the model.
			with torch.no_grad():
				if not baseline:
					zero_preds = model(zero_tnsr[:,:2].float(), zero_tnsr[:,2].long())
					zero_preds = get_pred_class(zero_preds)

					one_preds = model(one_tnsr[:,:2].float(), one_tnsr[:,2].long())
					one_preds = get_pred_class(one_preds)
				else:
					X_zero = tensor_to_numpy(zero_tnsr)
					zero_preds = torch.tensor(true_f(true_g, X_zero))

					X_one = tensor_to_numpy(one_tnsr)
					one_preds = torch.tensor(true_f(true_g, X_one))

					epoch = "Baseline"

			# Put into the maps how many times the model guessed 0.
			zero_map[(low_x, low_y)] = torch.sum(zero_preds == 0).item()
			one_map[(low_x, low_y)] = torch.sum(one_preds == 0).item()

	save_heatmap(zero_map, epoch, 0, lr, optimizer)
	save_heatmap(one_map, epoch, 1, lr, optimizer)

# Gets the x, y, z values.
def get_coords_from_map(pt_map):
	x_vals = []
	y_vals = []
	z_vals = []

	for key, value in pt_map.items():
		x_vals.append(key[0])
		y_vals.append(key[1])
		z_vals.append(value)

	return x_vals, y_vals, z_vals

# Saves the heatmap.
def save_heatmap(pt_map, epoch, boolean, lr, optimizer):
	sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
	x_vals, y_vals, z_vals = get_coords_from_map(pt_map)
	data = pd.DataFrame(data={'x':x_vals, 'y':y_vals, 'z':z_vals})
	data = data.pivot(index='x', columns='y', values='z')
	data = data.transpose()

	ticks = np.linspace(-10, 10, 21, dtype=np.int)

	ax = sns.heatmap(data, vmin=0, vmax=100, square=True)
	# ax.set_xticks(ticks)
	ax.invert_yaxis()
	ax.set_title(f"Boolean: {boolean}, Epoch: {epoch}, lr: {lr}, optim: {optimizer}", fontsize=12)

	figure = ax.get_figure()
	figure.savefig(f"model_guesses_over_epoch/heatmap{boolean}/plot{epoch}_{boolean}.png", dpi=400)
	plt.close()