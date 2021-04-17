import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from scipy.ndimage.filters import gaussian_filter

from simplebool_data_gen import *
import sys
sys.path.insert(0, '../pickled_files/')
from pickle_logic import *
sys.path.pop(0)
from pathlib import Path

# zeropts_map = {}
# onepts_map = {}

x_pts_path = "../pickled_files/x_pts.pickle"
x_pts_file = Path(x_pts_path)
x_pts = load_pickle(x_pts_file) if x_pts_file.is_file() else []

y_pts_path = "../pickled_files/y_pts.pickle"
y_pts_file = Path(y_pts_path)
y_pts = load_pickle(y_pts_file) if y_pts_file.is_file() else []

# These values are used for the region that we are sampling from.
x_low = -1
x_high = 6
y_low = -1
y_high = 6

x_gt_pts = []
y_gt_pts = []

def get_gt_pts(boolean):
	global x_gt_pts
	global y_gt_pts

	# If the points haven't been sampled yet, then initialize them and save them globally.
	if (len(x_gt_pts) == 0):
		pts = np.random.uniform(0, 5, (400, 2))
		x_gt_pts = pts[:, 0]
		y_gt_pts = pts[:, 1]

	pts = np.concatenate((x_gt_pts.reshape(-1,1), y_gt_pts.reshape(-1,1)), axis=1)

	# Now, pts is in a form that is exactly like in get_heatmap.
	if boolean == 0:
		pts = np.concatenate((np.copy(pts), np.zeros((pts.shape[0], 1))), axis=1)
	else:
		pts = np.concatenate((np.copy(pts), np.ones((pts.shape[0], 1))), axis=1)

	labels = true_f(true_g, pts)
	colors = []
	for i, lbl in enumerate(labels):
		c = ""
		c += "b" if lbl == 1 else "y"

		x_val = x_gt_pts[i]
		y_val = y_gt_pts[i]
		if boolean == 0:
			if is_in(x_val, 0, 5) and is_in(y_val, 0, 5):
				c += "x"
			else:
				c += "."
		elif boolean == 1:
			if (is_in(x_val, 0, 0.3) and is_in(y_val, 0, 0.3)) or \
					(is_in(x_val, 4.7, 5) and is_in(y_val, 4.7, 5)):
				c += "x"
			else:
				c += "."
		
		colors.append(c)
	# colors = ['bo' if x == 1 else 'yo' for x in labels]

	return x_gt_pts, y_gt_pts, colors

	# This is outdated: was used for gridlike, uniformly spaced points.
	# spacing = 4
	# xvalues = np.arange(x_low + spacing, x_high, spacing)
	# yvalues = np.arange(y_low + spacing, y_high, spacing)
	# orig_xx, orig_yy = np.meshgrid(xvalues, yvalues)
	# xx, yy = orig_xx.reshape(-1), orig_yy.reshape(-1) # This reshaping is so that it is now (nxn,) shape.
	# pts = np.concatenate((xx.reshape(-1,1), yy.reshape(-1,1)), axis=1)

# The given logits are the logits for model guessing 1. We want to return P[model guesses 0].
def get_prob_zero(logits):
	return 1 - torch.sigmoid(logits)

# Converts the val into the stretched coordinate used for plotting.
def convert_coord_val(val):
	return 10 * (val - x_low)

# Returns all the x_pts, y_pts in a grid fashion.
def get_grid_pts(stepsize, round_decimals):
	global x_pts
	global y_pts

	rng_pts = np.round(np.arange(x_low, x_high + stepsize, stepsize), round_decimals).tolist()
	len_range = len(rng_pts)

	if len(x_pts) != len_range**2 or len(y_pts) != len_range**2:
		print("Creating new grid points.")

		x_pts = []
		y_pts = []

		for i in range(len_range):
			x_pts.extend(rng_pts)
			y_pts.extend([rng_pts[i]] * len_range)

		save_pickle(x_pts, x_pts_path)
		save_pickle(y_pts, y_pts_path)

	assert len(x_pts) == len_range**2 and len(y_pts) == len_range**2, "Grid points initialized wrong."
	return np.array(x_pts), np.array(y_pts)

# Returns x and y points of sample_size.
def get_pts(sample_size):
	global x_pts
	global y_pts
	
	if len(x_pts) != sample_size or len(y_pts) != sample_size:
		# Sampling from [-10,10]^2.
		x_pts = np.random.uniform(low=x_low, high=x_high, size=sample_size)
		y_pts = np.random.uniform(low=y_low, high=y_high, size=sample_size)

		# Save the generated points into their respective pickles.
		save_pickle(x_pts, x_pts_path)
		save_pickle(y_pts, y_pts_path)

	assert len(x_pts) == sample_size and len(y_pts) == sample_size, "Points were not initialized properly."
	return x_pts, y_pts

# Saves two heatmaps, per epoch.
def get_heatmap(model, epoch, stepsize, round_decimals, lr, optimizer, baseline=False):
	x_pts_orig, y_pts_orig = get_grid_pts(stepsize, round_decimals)
	# x_pts_zero, y_pts_zero = np.copy(x_pts_orig), np.copy(y_pts_orig)
	# x_pts_one, y_pts_one = np.copy(x_pts_orig), np.copy(y_pts_orig)
	# resolution = 250
	# nbrs = 64

	pts = np.concatenate((x_pts_orig.reshape(-1,1), y_pts_orig.reshape(-1,1)), axis=1)
	
	# Now, create (x,y,0) and (x,y,1) data points.
	zero_pts = np.concatenate((np.copy(pts), np.zeros((len(x_pts_orig), 1))), axis=1)
	one_pts = np.concatenate((np.copy(pts), np.ones((len(x_pts_orig), 1))), axis=1)

	# Create the tensors that can be fed into the model.
	zero_tnsr = torch.tensor(zero_pts).float().cuda()
	one_tnsr = torch.tensor(one_pts).float().cuda()

	# Get the predictions from the model.
	with torch.no_grad():
		if not baseline:
			# The model returns the logit for 1.
			zero_preds = model(zero_tnsr[:,:2].float(), zero_tnsr[:,2].long())

			# To get the probabilities that the model guesses 0, take the sigmoid and subtract that from 1.
			zero_preds = get_prob_zero(zero_preds)

			one_preds = model(one_tnsr[:,:2].float(), one_tnsr[:,2].long())
			one_preds = get_prob_zero(one_preds)
		else:
			X_zero = tensor_to_numpy(zero_tnsr)

			# Subtract from 1 to get the "prob" that model outputs 0, i.e. 1 - 0 = 1.
			zero_preds = 1 - torch.tensor(true_f(true_g, X_zero))

			X_one = tensor_to_numpy(one_tnsr)
			one_preds = 1 - torch.tensor(true_f(true_g, X_one))

			epoch = "Baseline"

	zero_vals = tensor_to_numpy(zero_preds).reshape(-1)
	one_vals = tensor_to_numpy(one_preds).reshape(-1)

	heatmap2d(zero_vals, x_pts_orig, y_pts_orig, 0, epoch, lr, "Adam")
	heatmap2d(one_vals, x_pts_orig, y_pts_orig, 1, epoch, lr, "Adam")

	# # Get the indices that correspond to the model guessing 0.
	# zero_indices = tensor_to_numpy((zero_preds==0).view(-1))
	# one_indices = tensor_to_numpy((one_preds==0).view(-1))

	# plotHeatmap(x_pts_zero[zero_indices], y_pts_zero[zero_indices], epoch, 0, lr, optimizer)
	# plotHeatmap(x_pts_one[one_indices], y_pts_one[one_indices], epoch, 1, lr, optimizer)

# Takes a CUDA tensor, returns a numpy.
def tensor_to_numpy(tnsr):
	return tnsr.detach().cpu().numpy()

def heatmap2d(vals, x_pts, y_pts, boolean, epoch, lr, optimizer):
	n = int(np.sqrt(len(x_pts)))
	arr = vals.reshape(n,n)
	arr = gaussian_filter(arr, sigma=6)

	fig = plt.figure()
	fig.patch.set_facecolor('white')
	plt.imshow(arr, cmap='plasma', vmin=0, vmax=1.0)

	# Another way to do this, using meshgrid and pcolormesh.
	# x, y = np.meshgrid(np.arange(-10, 10.1, 0.1), np.arange(-10, 10.1, 0.1))
	# plt.pcolormesh(x, y, arr, cmap='plasma', vmin=0, vmax=1.0)

	fig.suptitle(f"Boolean: {boolean}, Epoch: {epoch}, lr: {lr}, optim: {optimizer}", fontsize=12)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.colorbar()

	# plt.annotate(f"{arr[0][10]}", (10*(x_pts[0] + 10), 10*(y_pts[10] + 10)), )
	if epoch != "Baseline":
		epoch -= 1

	plt.gca().invert_yaxis()

	# Now, plotting some ground truth points onto the heatmap.

	xx, yy, colors = get_gt_pts(boolean)

	b_count, y_count = 0, 0
	for x, y, c in zip(xx, yy, colors):
		# This fix is to send [-10,10]^2 -> [0,200]^2
		x, y = convert_coord_val(x), convert_coord_val(y)

		# Get the markersizes.
		if 'x' in c:
			markersize = 7
		else:
			markersize = 3

		if c == 'bo' and b_count == 0:
			lbl = 'Label = 1'
			b_count += 1
		elif c == 'yo' and y_count == 0:
			lbl = 'Label = 0'
			y_count += 1
		else:
			lbl = '_nolegend_'

		plt.plot(x, y, c, label=lbl, markersize=markersize)

	# Make sure the legends are in the same order, for both setups.
	ax = plt.gca()
	handles, labels = ax.get_legend_handles_labels()
	if boolean == 1:
		handles = handles[::-1]
		labels = labels[::-1]
	ax.legend(handles, labels, title='GT Points', bbox_to_anchor=(0.05, 0.95))

	# Drawing the white bboxes for the training points.
	if boolean == 0:
		rect1 = patches.Rectangle((convert_coord_val(0), convert_coord_val(0)), \
			convert_coord_val(5) - convert_coord_val(0), convert_coord_val(5) - convert_coord_val(0),\
				edgecolor='green', fill=False)
		ax.add_patch(rect1)
	elif boolean == 1:
		rect1 = patches.Rectangle((convert_coord_val(0), convert_coord_val(0)), \
			convert_coord_val(0.3) - convert_coord_val(0), convert_coord_val(0.3) - convert_coord_val(0),\
				edgecolor='green', fill=False)
		rect2 = patches.Rectangle((convert_coord_val(4.7), convert_coord_val(4.7)), \
			convert_coord_val(5) - convert_coord_val(4.7), convert_coord_val(5) - convert_coord_val(4.7),\
				edgecolor='green', fill=False)
		ax.add_patch(rect1)
		ax.add_patch(rect2)


	plt.yticks([0, n], [x_low, x_high])
	plt.xticks([0, n], [y_low, y_high])

	fig.savefig(f"model_guesses_over_epoch/heatmap{boolean}/plot{boolean}_{epoch}.png", dpi=400)
	plt.close()

	
# # Code is from: https://stackoverflow.com/questions/2369492/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set/59920744#59920744
# def data_coord2view_coord(p, resolution, pmin, pmax):
# 	dp = pmax - pmin
# 	dv = (p - pmin) / dp * resolution
# 	return dv

# def kNN2DDens(xv, yv, resolution, neighbors, dim=2):
# 	tree = cKDTree(np.array([xv, yv]).T)
# 	grid = np.mgrid[0:resolution, 0:resolution].T.reshape(resolution**2, dim)
# 	dists = tree.query(grid, neighbors)
# 	inv_sum_dists = 1. / dists[0].sum(1)

# 	im = inv_sum_dists.reshape(resolution, resolution)
# 	return im

# def plotHeatmap(xs, ys, epoch, boolean, lr, optimizer, resolution=250, neighbors=256):
# 	fig = plt.figure()
# 	fig.patch.set_facecolor('white')
# 	xs = np.append(xs, [10,10,-10,-10])
# 	ys = np.append(ys, [10,-10,10,-10])

# 	if len(xs) == 0 or len(ys) == 0:
# 		print("Blank figure.")
# 	else:
# 		extent = [np.min(xs), np.max(xs), np.min(ys), np.max(ys)]
# 		xv = data_coord2view_coord(xs, resolution, extent[0], extent[1])
# 		yv = data_coord2view_coord(ys, resolution, extent[2], extent[3])

# 		im = kNN2DDens(xv, yv, resolution, neighbors)
# 		plt.imshow(im, origin='lower', extent=extent, cmap=cm.plasma)

# 	fig.suptitle(f"Boolean: {boolean}, Epoch: {epoch}, lr: {lr}, optim: {optimizer}", fontsize=12)

# 	plt.xlim([x_low, x_high])
# 	plt.ylim([y_low, y_high])

# 	# ax = plt.axes()
# 	# ax.set_facecolor('#0A0880')
	
# 	fig.savefig(f"model_guesses_over_epoch/heatmap_{boolean}/plot{boolean}_{epoch-1}.png", dpi=1000)
# 	plt.close()

# 	fig = plt.figure()
# 	plt.plot(xs, ys, 'k.', markersize=1)
# 	fig.suptitle(f"Ground Truth Boolean: {boolean}, Epoch: {epoch}")
# 	fig.savefig(f"model_guesses_over_epoch/gt_heatmap_{boolean}/plot{epoch}_{boolean}.png")
# 	plt.close()

# # Gets the x, y, z values.
# def get_coords_from_map(pt_map):
# 	x_vals = []
# 	y_vals = []
# 	z_vals = []

# 	for key, value in pt_map.items():
# 		x_vals.append(key[0])
# 		y_vals.append(key[1])
# 		z_vals.append(value)

# 	return x_vals, y_vals, z_vals

# # Saves the heatmap.
# def save_heatmap(pt_map, epoch, boolean, lr, optimizer):
# 	sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
# 	x_vals, y_vals, z_vals = get_coords_from_map(pt_map)
# 	data = pd.DataFrame(data={'x':x_vals, 'y':y_vals, 'z':z_vals})
# 	data = data.pivot(index='x', columns='y', values='z')
# 	data = data.transpose()

# 	# ticks = [x for x in range(-7, 8)]
# 	ax = sns.heatmap(data, vmin=0, vmax=1, square=True, cmap="flare", xticklabels=False, yticklabels=False)
# 	# ax.set_xticks(ticks)
# 	ax.invert_yaxis()
# 	ax.set_title(f"Boolean: {boolean}, Epoch: {epoch}, lr: {lr}, optim: {optimizer}", fontsize=12)

# 	figure = ax.get_figure()
# 	figure.savefig(f"model_guesses_over_epoch/heatmap{boolean}/plot{epoch}_{boolean}.png", dpi=400)
# 	plt.close()

# # Saves two heatmaps, per epoch.
# def get_heatmap(model, epoch, stepsize, num_samples, lr, optimizer, baseline=False):
# 	global zeropts_map
# 	global onepts_map

# 	# Here, the key is tuple of quadrant and the value is number of 0's guessed by model in that quadrant.
# 	zero_map = {}
# 	one_map = {}

# 	for low_x in np.arange(-6, 6, stepsize):
# 		for low_y in np.arange(-6, 6, stepsize):
# 			if (low_x, low_y) not in zeropts_map or (low_x, low_y) not in onepts_map:
# 				assert (low_x, low_y) not in zeropts_map and (low_x, low_y) not in onepts_map, "Messed up."
# 				# Sample num_samples points in this current quadrant.
# 				x_pts = np.random.uniform(low=low_x, high=low_x + stepsize, size=num_samples).reshape(-1,1)
# 				y_pts = np.random.uniform(low=low_y, high=low_y + stepsize, size=num_samples).reshape(-1,1)
# 				pts = np.concatenate((x_pts, y_pts), axis=1)
				
# 				# Now, create (x,y,0) and (x,y,1) data points.
# 				zero_pts = np.concatenate((np.copy(pts), np.zeros((num_samples, 1))), axis=1)
# 				one_pts = np.concatenate((np.copy(pts), np.ones((num_samples, 1))), axis=1)

# 				# Create the tensors that can be fed into the model.
# 				zero_tnsr = torch.tensor(zero_pts).float().cuda()
# 				one_tnsr = torch.tensor(one_pts).float().cuda()

# 				zeropts_map[(low_x, low_y)] = zero_tnsr
# 				onepts_map[(low_x, low_y)] = one_tnsr
# 			else:
# 				zero_tnsr = zeropts_map[(low_x, low_y)]
# 				one_tnsr = onepts_map[(low_x, low_y)]

# 			# Get the predictions from the model.
# 			with torch.no_grad():
# 				if not baseline:
# 					zero_preds = model(zero_tnsr[:,:2].float(), zero_tnsr[:,2].long())
# 					zero_preds = get_pred_class(zero_preds)

# 					one_preds = model(one_tnsr[:,:2].float(), one_tnsr[:,2].long())
# 					one_preds = get_pred_class(one_preds)
# 				else:
# 					X_zero = tensor_to_numpy(zero_tnsr)
# 					zero_preds = torch.tensor(true_f(true_g, X_zero))

# 					X_one = tensor_to_numpy(one_tnsr)
# 					one_preds = torch.tensor(true_f(true_g, X_one))

# 					epoch = "Baseline"

# 			# Put into the maps prob of the the model guessing 0.
# 			zero_map[(low_x, low_y)] = torch.sum(zero_preds == 0).item() / num_samples
# 			one_map[(low_x, low_y)] = torch.sum(one_preds == 0).item() /num_samples

# 	save_heatmap(zero_map, epoch, 0, lr, optimizer)
# 	save_heatmap(one_map, epoch, 1, lr, optimizer)