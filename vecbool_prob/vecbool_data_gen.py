from comet_ml import Experiment

import torch
import torch.nn as nn

import numpy as np
import sys
from bool_utils import *
from gen_rep_bools import *
sys.path[0] = '../pickled_files/'
from pickle_logic import *
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path

# Global cache that stores the statistics of train and test sets.
cache_path = "../pickled_files/stats_cache.pickle"
cache_file = Path(cache_path)
cache = load_pickle(cache_path) if cache_file.is_file() else {}

if rep_bools_len != (num_symbols - 1) * boolvec_dim + 1:
	file_path = "../pickled_files/" + str(rep_bools_len) + "_rep_bools.pickle"
else:
	file_path = "../pickled_files/rep_bools.pickle"
rep_bools = load_pickle(file_path) if Path(file_path).is_file() else gen_bools()

# This test_dist is used for the testing accuracy, which is looked at during training.
test_dist = 1
neighbor_bools = get_neighbor_bools(rep_bools, boolvec_dim, test_dist)

hyper_params = {
	"cont_range": [0, 5],
	"boolvec_dim": boolvec_dim, # boolvec_dim is defined in bool_utils.py.
	"emb_dims": [num_symbols * boolvec_dim, 2 * num_symbols * boolvec_dim],
	"num_cont": 2,
	"lin_layer_sizes": [128, 512, 128, 32],
	"num_classes": 10,
	"hidden_drop_p": 0.1,
	"batch_flag": True,
	"rep_bools": rep_bools,
	"test_dist": test_dist,
	"neighbor_bools": neighbor_bools,
	"lr": 1e-3,
	"num_epochs": 100,
	"batch_size": 256,
	"num_symbols": num_symbols # num_symbols is defined in bool_utils.py
}

# Flag denotes whether to use true labels, or rotated labels.
useRealLabels = False

# Toggle this flag, if running the unrotation experiment or not.
unrotationExperimentFlag = False

# Toggle this flag, if balancing ground truth labels.
balanceGTLabelFlag = True

# Toggle this flag, if switching training and test data sets.
switchDataSetsFlag = False

# Toggle this flag, if converting the boolean vector to take into account (position, value).
convertBooleanFlag = True

# Toggle this flag if shuffling the data in true_g method.
shuffleFlag = True

# Toggle this flag if using random_f for the test data.
random_flag = False

# Toggle this flag if using bitstring interpretation of boolvec.
bitstring_flag = False

# Toggle this flag if using an arbitrary nn as the underlying g fn.
arbitrary_fn_flag = False

test_params = {
  "useRealLabels": useRealLabels,
  "unrotationExperimentFlag": unrotationExperimentFlag,
  "balanceGTLabelFlag": balanceGTLabelFlag,
	"switchDataSetsFlag": switchDataSetsFlag,
	"convertBooleanFlag": convertBooleanFlag,
	"arbitrary_fn_flag": arbitrary_fn_flag,
	"bitstring_flag": bitstring_flag,
	"shuffleFlag": shuffleFlag
}

# Toggle this flag if logging the experiment information in comet.ml, or not.
log_experiment_flag = True

if log_experiment_flag:
	with open('../ssh_keys/comet_api_key.txt', 'r') as file:
		comet_key = file.read().replace('\n', '')

	experiment = Experiment(api_key=comet_key, project_name="vecbool_report", workspace="clrkwng")
	experiment.log_parameters(hyper_params)
	experiment.log_parameters(test_params)

# Save a plot, with the following parameters. yvalues needs to be passed in as a list.
def save_plot(xvalues, yvalues, xlabel, ylabel, title, file_name, fn):
	if len(yvalues) == 1:
		fn(xvalues, yvalues[0])
	else:
		fn(xvalues, yvalues[0][0], label=yvalues[0][1])
		fn(xvalues, yvalues[1][0], label=yvalues[1][1])
		plt.legend(loc="upper left")

	plt.suptitle(title, fontsize=18)
	plt.xlabel(xlabel, fontsize=14)
	plt.ylabel(ylabel, fontsize=14)
	plt.savefig("images/" + file_name)
	plt.clf()

# Save 3D plot.
def save_3D_plot(x_vals, y_vals, z_vals, file_name, x_label, y_label, z_label):
	ax = plt.axes(projection='3d')
	ax.scatter3D(x_vals, y_vals, z_vals, c=y_vals)
	for x, y, z in zip(x_vals, y_vals, z_vals):
		label = "  " + str(int(x)) + ", " + str(int(z))
		ax.text(x, y, z, '%s' % label, size=7, zorder=1, color='k')
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	ax.set_zlabel(z_label)
	xx, yy = np.meshgrid(range(10), range(10))
	ax.plot_surface(xx, yy, xx, alpha=0.2)
	plt.savefig("images/" + file_name)
	plt.clf()

# Standardize the data (subtract mean, divide by std).
# Will use training data's mean and std for both train and test data.
def standardize_data(X_orig, mode="test", save_stats=True):
	global cache

	X = np.copy(X_orig)

	if mode == "train":
		X_mean = np.mean(X[:, :hyper_params["num_cont"]], axis=0)
		X_std = np.std(X[:, :hyper_params["num_cont"]], axis=0)
		X[:, :hyper_params["num_cont"]] -= X_mean
		X[:, :hyper_params["num_cont"]] /= X_std
		cache["X_train_mean"] = X_mean
		cache["X_train_std"] = X_std
		if save_stats:
			save_pickle(cache, "../pickled_files/stats_cache.pickle")

	else:
		assert "X_train_mean" in cache and "X_train_std" in cache,\
			"Train data statistics have not been cached yet."
			
		X[:, :hyper_params["num_cont"]] -= cache["X_train_mean"]
		X[:, :hyper_params["num_cont"]] /= cache["X_train_std"]

	return X

# Return unstandardized data, used for plotting.
def unstandardize_data(X_standardized):
	X = np.copy(X_standardized)
	X[:, :hyper_params["num_cont"]] *= cache["X_train_std"]
	X[:, :hyper_params["num_cont"]] += cache["X_train_mean"]

	return X

# g : R^2 -> {0, 1, ..., num_classes - 1}.
# Here, each x in X is in R^2.
def true_g(X):
	X_sum = np.sum(X, axis=1)

	# Percentage gets what percentage the sum of x1, x2 is of the the possible continuous sum.
	percentage = (X_sum - (2 * hyper_params["cont_range"][0])) / (2 * (hyper_params["cont_range"][1] - hyper_params["cont_range"][0]))
	true_labels = np.floor(percentage * hyper_params["num_classes"])

	if balanceGTLabelFlag:
		# This code will balance the number of samples, based on the ground truth labels.
		X_rebalanced, true_labels_rebalanced = [], []
		# min_count is the minimum number of data points present, out of all classes.
		min_count = min(np.unique(true_labels, return_counts=True)[1])
		for i in range(hyper_params["num_classes"]):
			class_indices = (true_labels == i)
			X_rebalanced.extend(X[class_indices][:min_count])
			true_labels_rebalanced.extend(true_labels[class_indices][:min_count])

		X, true_labels = X_rebalanced, true_labels_rebalanced

	# Shuffle X, true_labels together so that the classes aren't in the same order.
	if shuffleFlag:
		tmp = list(zip(X, true_labels))
		np.random.shuffle(tmp)
		X, true_labels = zip(*tmp)		

	return list(X), list(true_labels)

# Artificially choosing class in a balanced manner.
def arb_class_helper(val):
	val = val * 1000
	if 31 <= val <= 34:
		return 0
	elif 34 <= val <= 36.5:
		return 1
	elif 36.5 <= val <= 38:
		return 2
	elif 38 <= val <= 39.1:
		return 3
	elif 39.1 <= val <= 39.9:
		return 4
	elif 39.9 <= val <= 41.5:
		return 5
	elif 41.5 <= val <= 43:
		return 6
	elif 43 <= val <= 44:
		return 7
	elif 44 <= val <= 47:
		return 8
	else:
		return 9

def return_arb_classes(preds):
	lsts = tensor_to_numpy(preds).tolist()
	diffs = []
	for lst in lsts:
		diffs.append(max(lst) - min(lst))
	return [arb_class_helper(d) for d in diffs]

# g : R^2 -> {0, 1, ..., num_classes - 1}.
# Here, each x in X is in R^2.
# The fn here is just an arbitrary MLP.
def arbitrary_g(X):
	print("Using arbitrary fn, beware!")
	
	# Either loading the random MLP, or initializing it.
	model = nn.Sequential(
					nn.Linear(2, 16),
					nn.ReLU(),
					nn.Linear(16, 32),
					nn.ReLU(),
					nn.Linear(32, 10),
					nn.Softmax()
				).cuda()

	model_path = "../saved_model_params/rand_model_state_dict.pt"
	model_file = Path(model_path)
	if model_file.is_file():
		model.load_state_dict(torch.load(model_path))
	else:
		torch.save(model.state_dict(), model_path)

	X_standardized = standardize_data(X, mode="train", save_stats=False)
	model.eval()
	with torch.no_grad():
		X_tensor = torch.tensor(X_standardized).float().cuda()
		preds = model(X_tensor)
		# preds = tensor_to_numpy(preds.max(1)[1]).tolist()

	X_arr = [np.asarray(x_lst) for x_lst in X]

	return X_arr, return_arb_classes(preds)


# Given true_labels, and each x in X has continuous and categorical data.
def true_f(true_labels, X):
	if not bitstring_flag:
		rot_amts = get_rotation_amount(X[:, hyper_params["num_cont"]:].T)
	else:
		rot_amts = bool_to_dec(X[:, hyper_params["num_cont"]:].T)
	rotated_labels = rotate_class(true_labels, rot_amts, hyper_params["num_classes"])
	return rotated_labels

# Takes a CUDA tensor, returns a numpy.
def tensor_to_numpy(tnsr):
	return tnsr.detach().cpu().numpy()

# Returns the number of correct predictions.
# Parameter k is used for weaker accuracy, i.e. if k = 1, we also count classes +- 1 as correct.
def get_num_correct(preds, labels, k=0, print_preds=False):
	pred = preds.max(1, keepdim=True)[1]
	if print_preds:
		print(f"{np.unique(tensor_to_numpy(pred), return_counts=True)}\n")
	correct = pred.eq(labels.view_as(pred)).sum().item()
	for i in range(1, k + 1):
		correct += pred.eq((labels.view_as(pred) + 1) % 10).sum().item()
		correct += pred.eq((labels.view_as(pred) - 1) % 10).sum().item()
	return correct

# Instead of rotating labels by adding, use multiplication.
def random_f(true_labels, X):
	if not bitstring_flag:
		mult_amts = get_rotation_amount(X[:, hyper_params["num_cont"]:].T)
	else:
		mult_amts = bool_to_dec(X[:, hyper_params["num_cont"]:].T)
	rotated_labels = mod_mult(true_labels, mult_amts, hyper_params["num_classes"])
	return rotated_labels

def get_train_data(train_size):
	global cache

	X_01 = np.random.uniform(hyper_params["cont_range"][0], hyper_params["cont_range"][1], size=(train_size, hyper_params["num_cont"]))
	if arbitrary_fn_flag:
		X_01, true_labels = arbitrary_g(X_01)
	else:
		X_01, true_labels = true_g(X_01)

	X_02 = get_rep_bool_vecs(len(X_01), hyper_params["boolvec_dim"], hyper_params["rep_bools"])
	X_train = np.concatenate((X_01, X_02), axis=1)
	rotated_labels = true_f(true_labels, X_train)

	X_train = standardize_data(X_train, mode="train")

	# Testing: Remove this line.
	if convertBooleanFlag:
		for i in range(len(X_train)):
			X_train[i, hyper_params["num_cont"]:] = convert_boolvec_to_position_vec(X_train[i, hyper_params["num_cont"]:])
	
	if balanceGTLabelFlag and not arbitrary_fn_flag:
		assert min(np.unique(true_labels, return_counts=True)[1]) == max(np.unique(true_labels, return_counts=True)[1]), "Ground truth labels not balanced."

	return X_train, true_labels, rotated_labels

# This test distribution is the same as the training distribution.
# I use this method to get the validation data.
def get_test_splitA(test_size, *unused):
	global cache

	X_01 = np.random.uniform(hyper_params["cont_range"][0], hyper_params["cont_range"][1], size=(test_size, hyper_params["num_cont"]))
	if arbitrary_fn_flag:
		X_01, true_labels = arbitrary_g(X_01)
	else:
		X_01, true_labels = true_g(X_01)

	X_02 = get_rep_bool_vecs(len(X_01), hyper_params["boolvec_dim"], hyper_params["rep_bools"])
	X_test = np.concatenate((X_01, X_02), axis=1)
	rotated_labels = true_f(true_labels, X_test)

	X_test = standardize_data(X_test)

	# Testing: Remove this line.
	if convertBooleanFlag:
		for i in range(len(X_test)):
			X_test[i, hyper_params["num_cont"]:] = convert_boolvec_to_position_vec(X_test[i, hyper_params["num_cont"]:])
	
	if balanceGTLabelFlag and not arbitrary_fn_flag:
		assert min(np.unique(true_labels, return_counts=True)[1]) == max(np.unique(true_labels, return_counts=True)[1]), "Ground truth labels not balanced."

	return X_test, true_labels, rotated_labels

# This test distribution tests compositionality.
def get_test_splitB(test_size, test_dist):
	global cache

	X_01 = np.random.uniform(hyper_params["cont_range"][0], hyper_params["cont_range"][1], size=(test_size, hyper_params["num_cont"]))
	if arbitrary_fn_flag:
		X_01, true_labels = arbitrary_g(X_01)
	else:
		X_01, true_labels = true_g(X_01)

	X_02 = get_dist_bool_vecs(len(X_01), hyper_params["boolvec_dim"], hyper_params["rep_bools"], test_dist)
	X_test = np.concatenate((X_01, X_02), axis=1)

	# Change the fn call here, depending on if using random_f or true_f.
	if random_flag:
		rotated_labels = random_f(true_labels, X_test)
	else:
		rotated_labels = true_f(true_labels, X_test)

	if unrotationExperimentFlag:
		matching_indices = (true_labels == rotated_labels)
		true_labels = true_labels[matching_indices]
		rotated_labels = rotated_labels[matching_indices]
		X_test = X_test[matching_indices]
	
	X_test = standardize_data(X_test)

	# Testing: Remove this line.
	if convertBooleanFlag:
		for i in range(len(X_test)):
			X_test[i, hyper_params["num_cont"]:] = convert_boolvec_to_position_vec(X_test[i, hyper_params["num_cont"]:])
	
	if balanceGTLabelFlag and not arbitrary_fn_flag:
		assert min(np.unique(true_labels, return_counts=True)[1]) == max(np.unique(true_labels, return_counts=True)[1]), "Ground truth labels not balanced."

	return X_test, true_labels, rotated_labels