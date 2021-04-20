import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage.filters import gaussian_filter

from simplebool_data_gen import *
import sys
sys.path.insert(0, '../pickled_files/')
from pickle_logic import *
sys.path.pop(0)
from pathlib import Path

# These values are used for the region that we are plotting on.
x_low = -1
x_high = 6
y_low = -1
y_high = 6

# Multfactor is stored to use for convert_coord_val.
# It is inverse of stepsize.
multfactor = 0

# These are the ground truth points that are plotted on top of the heatmap.
x_gt_pts = []
y_gt_pts = []

# Method gets the ground truth points to plot on top of heatmap.
def get_gt_pts(boolean):
	global x_gt_pts
	global y_gt_pts

	# If the points haven't been sampled yet, then initialize them and save them globally.
	if (len(x_gt_pts) == 0):
		pts = np.random.uniform(0, 5, (296, 2))
		# Artificially introduce points, so that the heatmaps have all samples of points from the training distribution.
		chosen_pts = [[0.01,0.03], [0.29, 0.24], [0.18,0.31], [4.9,4.9]]
		pts = np.concatenate((pts, chosen_pts), axis=0)

		x_gt_pts = pts[:, 0]
		y_gt_pts = pts[:, 1]

	pts = np.concatenate((x_gt_pts.reshape(-1,1), y_gt_pts.reshape(-1,1)), axis=1)

	# Now, put pts into a form that is exactly like in get_heatmap.
	if boolean == 0:
		pts = np.concatenate((np.copy(pts), np.zeros((pts.shape[0], 1))), axis=1)
	else:
		pts = np.concatenate((np.copy(pts), np.ones((pts.shape[0], 1))), axis=1)

	labels = true_f(true_g, pts)
	colors = []

	# Give the labels, according to 1/0 value and whether it is in training or not.
	for i, lbl in enumerate(labels):
		c = "b" if lbl == 1 else "y"

		x_val = x_gt_pts[i]
		y_val = y_gt_pts[i]
		if boolean == 0:
			if is_in(x_val, 0, 5) and is_in(y_val, 0, 5):
				c += "x"
			else:
				c += "."
		elif boolean == 1:
			if (is_in(x_val, 0, 0.24) and is_in(y_val, 0, 0.24)) or \
					(is_in(x_val, 4.7, 5) and is_in(y_val, 4.7, 5)):
				c += "x"
			else:
				c += "."
		
		colors.append(c)

	return x_gt_pts, y_gt_pts, colors

# Converts the val into the stretched coordinate used for plotting.
def convert_coord_val(val):
	assert multfactor != 0, "Multfactor not initialized properly."
	return multfactor * (val - x_low)

# Returns all the x_pts, y_pts in a grid fashion.
def get_grid_pts(stepsize, round_decimals):
	global x_pts
	global y_pts

	xvalues = np.arange(x_low, x_high + stepsize, stepsize)
	yvalues = np.arange(y_low, y_high + stepsize, stepsize)
	x_pts, y_pts = np.meshgrid(xvalues, yvalues)
	x_pts, y_pts = x_pts.reshape(-1), y_pts.reshape(-1)
	return x_pts, y_pts

# Saves two heatmaps, per epoch.
def get_heatmap(model, epoch, stepsize, round_decimals, lr, baseline=False):
	# Save the multfactor for convert_coord_val.
	global multfactor
	multfactor = 1 / stepsize

	x_pts_orig, y_pts_orig = get_grid_pts(stepsize, round_decimals)

	pts = np.concatenate((x_pts_orig.reshape(-1,1), y_pts_orig.reshape(-1,1)), axis=1)
	
	# Now, create (x,y,0) and (x,y,1) data points.
	zero_pts = np.concatenate((np.copy(pts), np.zeros((len(x_pts_orig), 1))), axis=1)
	one_pts = np.concatenate((np.copy(pts), np.ones((len(x_pts_orig), 1))), axis=1)

	# IMPORTANT: Standardize the data, before feeding into model.
	zero_pts, _, _ = standardize_data(zero_pts)
	one_pts, _, _ = standardize_data(one_pts)

	# Create the tensors that can be fed into the model.
	zero_tnsr = torch.tensor(zero_pts).float().cuda()
	one_tnsr = torch.tensor(one_pts).float().cuda()

	# Get the predictions from the model.
	with torch.no_grad():
		if not baseline:
			# The model returns the logit for 1.
			zero_preds = model(zero_tnsr[:,:2].float(), zero_tnsr[:,2].long())
			# To get the probabilities that the model guesses 0, take the sigmoid and subtract that from 1.
			zero_probs = get_prob_zero(zero_preds)

			one_preds = model(one_tnsr[:,:2].float(), one_tnsr[:,2].long())
			one_probs = get_prob_zero(one_preds)
		else:
			X_zero = tensor_to_numpy(zero_tnsr)

			# Subtract from 1 to get the "prob" that model outputs 0, i.e. 1 - 0 = 1.
			zero_preds = torch.tensor(true_f(true_g, X_zero))
			zero_probs = 1 - zero_preds

			X_one = tensor_to_numpy(one_tnsr)
			one_preds = torch.tensor(true_f(true_g, X_one))
			one_probs = 1 - one_preds

			epoch = "Baseline"

	zero_probs = tensor_to_numpy(zero_probs).reshape(-1)
	one_probs = tensor_to_numpy(one_probs).reshape(-1)

	heatmap2d(zero_probs, x_pts_orig, y_pts_orig, 0, epoch, lr, model)
	heatmap2d(one_probs, x_pts_orig, y_pts_orig, 1, epoch, lr, model)

def heatmap2d(val_probs, x_pts, y_pts, boolean, epoch, lr, model):
	n = int(np.sqrt(len(x_pts)))
	arr1 = val_probs.reshape(n,n)
	arr1 = gaussian_filter(arr1, sigma=6)

	fig = plt.figure(figsize=(4,5))
	fig.patch.set_facecolor('white')
	plt.imshow(arr1, cmap='plasma', vmin=0, vmax=1.0)

	fig.suptitle(f"Boolean: {boolean}, Epoch: {epoch}, lr: {lr}", fontsize=12)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.colorbar()

	if epoch != "Baseline":
		epoch -= 1

	plt.gca().invert_yaxis()

	# Now, plotting some ground truth points onto the heatmap.

	xx, yy, colors = get_gt_pts(boolean)

	bx_count, yx_count = 0, 0
	bdot_count, ydot_count = 0, 0
	for x, y, c in zip(xx, yy, colors):
		# This fix is to send [-10,10]^2 -> [0,200]^2
		x, y = convert_coord_val(x), convert_coord_val(y)

		# Get the markersizes.
		if 'x' in c:
			markersize = 4
		else:
			markersize = 2

		if c == 'bx' and bx_count == 0:
			lbl = 'Train Label = 1'
			bx_count += 1
		elif c == 'b.' and bdot_count == 0:
			lbl = 'Test Label = 1'
			bdot_count += 1
		elif c == 'yx' and yx_count == 0:
			lbl = 'Train Label = 0'
			yx_count += 1
		elif c == 'y.' and ydot_count == 0:
			lbl = 'Test Label = 0'
			ydot_count += 1
		else:
			lbl = '_nolegend_'

		plt.plot(x, y, c, label=lbl, markersize=markersize)

	# Make sure the legends are in the same order, for both setups.
	ax = plt.gca()
	if boolean == 1:
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles, labels, title='GT Points', bbox_to_anchor=(.98, .03), prop={'size':6})

	# Drawing the green bboxes for the training points.
	if boolean == 0:
		rect1 = patches.Rectangle((convert_coord_val(0), convert_coord_val(0)), \
			convert_coord_val(5) - convert_coord_val(0), convert_coord_val(5) - convert_coord_val(0),\
				edgecolor='green', fill=False)
		ax.add_patch(rect1)
	elif boolean == 1:
		rect1 = patches.Rectangle((convert_coord_val(0), convert_coord_val(0)), \
			convert_coord_val(0.24) - convert_coord_val(0), convert_coord_val(0.24) - convert_coord_val(0),\
				edgecolor='green', fill=False)
		rect2 = patches.Rectangle((convert_coord_val(4.7), convert_coord_val(4.7)), \
			convert_coord_val(5) - convert_coord_val(4.7), convert_coord_val(5) - convert_coord_val(4.7),\
				edgecolor='green', fill=False)
		ax.add_patch(rect1)
		ax.add_patch(rect2)

	plt.yticks([0, n], [x_low, x_high])
	plt.xticks([0, n], [y_low, y_high])
	# plt.tight_layout()

	# Now, let's get the training and testing accuracy on this plot.
	if boolean == 0:
		train_pts = np.random.uniform(0, 5, (10000, 2))
		train_pts = np.concatenate((train_pts, np.zeros((10000, 1))), axis=1)
		X_train, _, _ = standardize_data(train_pts)
		y_train = true_f(true_g, train_pts)
		train_acc = model_eval_acc(model, X_train, y_train)

		test_acc = "n/a"
	else:
		train_pts1 = np.random.uniform(0, 0.24, (5000, 2))
		train_pts2 = np.random.uniform(4.7, 5, (5000, 2))
		train_pts = np.concatenate((train_pts1, train_pts2), axis=0)
		train_pts = np.concatenate((train_pts, np.ones((10000, 1))), axis=1)
		X_train, _, _ = standardize_data(train_pts)
		y_train = true_f(true_g, train_pts)
		train_acc = model_eval_acc(model, X_train, y_train)

		# Here, calculations are done for equally weighted test points.
		denom = (4.7**2 + 0.3*4.7 + 0.3*4.46 + 0.06*0.24)
		size1 = int((4.7**2)/denom * 10000)
		test_pts1 = np.random.uniform(0.3, 4.7, (size1, 2))
		test_pts1[:,1] -= 0.3

		size2 = int((0.3*4.7)/denom * 10000)
		x_2 = np.random.uniform(0, 4.7, (size2, 1))
		y_2 = np.random.uniform(4.7, 5.0, (size2, 1))
		test_pts2 = np.concatenate((x_2, y_2), axis=1)

		size3 = int((0.3*4.46)/denom * 10000)
		x_3 = np.random.uniform(0, 0.3, (size3, 1))
		y_3 = np.random.uniform(0.24, 4.7, (size3, 1))
		test_pts3 = np.concatenate((x_3, y_3), axis=1)

		size4 = int((0.06*0.24)/denom * 10000)
		x_4 = np.random.uniform(0.24, 0.3, (size4, 1))
		y_4 = np.random.uniform(0, 0.24, (size4, 1))
		test_pts4 = np.concatenate((x_4, y_4), axis=1)

		test_pts = np.concatenate((test_pts1, test_pts2, test_pts3, test_pts4), axis=0)
		test_pts = np.concatenate((test_pts, np.ones((len(test_pts),1))), axis=1)
		X_test, _, _ = standardize_data(test_pts)
		y_test = true_f(true_g, test_pts)
		test_acc = round(model_eval_acc(model, X_test, y_test), 3)
	
	descr = f"In distribution test acc: {train_acc}\nOut of distribution test acc: {test_acc}"
	plt.text(x=1, y=(y_high + 1.3) * multfactor, s=descr)

	fig.savefig(f"model_guesses_over_epoch/heatmap{boolean}/plot{boolean}_{epoch}.png", dpi=400)
	plt.close()

# Returns the accuracy the model gets on these points.
def model_eval_acc(model, X_pts, y_pts):
	X_pts = torch.tensor(X_pts).float().cuda()
	y_pts = torch.tensor(y_pts).view(-1,1).float().cuda()
	with torch.no_grad():
		preds = model(X_pts[:,:2].float(), X_pts[:,2].long())
		acc = round(get_num_correct(y_pts, preds) / len(X_pts), 3)
	return acc