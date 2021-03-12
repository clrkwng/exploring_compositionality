from comet_ml import Experiment

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

if rep_bools_len != boolvec_dim + 1:
	file_path = "../pickled_files/" + str(rep_bools_len) + "_rep_bools.pickle"
else:
	file_path = "../pickled_files/rep_bools.pickle"
rep_bools = load_pickle(file_path) if Path(file_path).is_file() else gen_bools()

test_dist = 1
neighbor_bools = get_neighbor_bools(rep_bools, boolvec_dim, test_dist)

hyper_params = {
	"cont_range": [0, 5],
	"boolvec_dim": boolvec_dim, # boolvec_dim is defined in gen_rep_bools.py.
	"emb_dims": [2 * boolvec_dim, 4 * boolvec_dim],
	"num_cont": 2,
	"lin_layer_sizes": [128, 512, 1024, 800, 512, 128, 32, 20],
	"num_classes": 10,
	"hidden_drop_p": 0.1,
	"batch_flag": True,
	"rep_bools": rep_bools,
	"test_dist": test_dist,
	"neighbor_bools": neighbor_bools,
	"lr": 1e-3,
	"num_epochs": 100,
	"batch_size": 256
}

# Flag denotes whether to use true labels, or rotated labels.
useRealLabels = False

# Toggle this flag, if running the unrotation experiment or not.
unrotationExperimentFlag = False

# Toggle this flag, if balancing ground truth labels.
balanceGTLabelFlag = True

# Toggle this flag, if switching training and test data sets.
switchDataSetsFlag = False

test_params = {
  "useRealLabels": useRealLabels,
  "unrotationExperimentFlag": unrotationExperimentFlag,
  "balanceGTLabelFlag": balanceGTLabelFlag,
	"switchDataSetsFlag": switchDataSetsFlag
}

with open('../ssh_keys/comet_api_key.txt', 'r') as file:
	comet_key = file.read().replace('\n', '')

experiment = Experiment(api_key=comet_key, project_name="vecbool", workspace="clrkwng")
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

# Standardize the data (subtract mean, divide by std).
# Will use training data's mean and std for both train and test data.
def standardize_data(X_orig, mode="test"):
	global cache

	X = np.copy(X_orig)

	if mode == "train":
		X_mean = np.mean(X[:, :hyper_params["num_cont"]], axis=0)
		X_std = np.std(X[:, :hyper_params["num_cont"]], axis=0)
		cache["X_train_mean"] = X_mean
		cache["X_train_std"] = X_std
		print(cache)
		save_pickle(cache, "../pickled_files/stats_cache.pickle")

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

	return X, true_labels

# Given true_labels, and each x in X has continuous and categorical data.
def true_f(true_labels, X):
	rot_amts = get_rotation_amount(X[:, hyper_params["num_cont"]:].T)
	rotated_labels = rotate_class(true_labels, rot_amts, hyper_params["num_classes"])
	return rotated_labels

# Takes a CUDA tensor, returns a numpy.
def tensor_to_numpy(tnsr):
	return tnsr.detach().cpu().numpy()

# Returns the number of correct predictions.
def get_num_correct(preds, labels, print_preds=False):
	pred = preds.max(1, keepdim=True)[1]
	if print_preds:
		print(f"{np.unique(tensor_to_numpy(pred), return_counts=True)}\n")
	correct = pred.eq(labels.view_as(pred)).sum().item()
	return correct

def get_train_data(train_size):
	global cache

	X_01 = np.random.uniform(hyper_params["cont_range"][0], hyper_params["cont_range"][1], size=(train_size, hyper_params["num_cont"]))
	X_01, true_labels = true_g(X_01)

	X_02 = get_rep_bool_vecs(len(X_01), hyper_params["boolvec_dim"], hyper_params["rep_bools"])
	X_train = np.concatenate((X_01, X_02), axis=1)
	rotated_labels = true_f(true_labels, X_train)

	X_train = standardize_data(X_train, mode="train")

	# Testing: Remove this line.
	for i in range(len(X_train)):
		X_train[i, hyper_params["num_cont"]:] = convert_boolvec_to_position_vec(X_train[i, hyper_params["num_cont"]:])
	
	if balanceGTLabelFlag:
		assert min(np.unique(true_labels, return_counts=True)[1]) == max(np.unique(true_labels, return_counts=True)[1]), "Ground truth labels not balanced."

	return X_train, true_labels if useRealLabels else rotated_labels

# This test distribution is the same as the training distribution.
def get_test_splitA(test_size, *unused):
	global cache

	X_01 = np.random.uniform(hyper_params["cont_range"][0], hyper_params["cont_range"][1], size=(test_size, hyper_params["num_cont"]))
	X_01, true_labels = true_g(X_01)

	X_02 = get_rep_bool_vecs(len(X_01), hyper_params["boolvec_dim"], hyper_params["rep_bools"])
	X_test = np.concatenate((X_01, X_02), axis=1)
	rotated_labels = true_f(true_labels, X_test)

	X_test = standardize_data(X_test)

	# Testing: Remove this line.
	for i in range(len(X_test)):
		X_test[i, hyper_params["num_cont"]:] = convert_boolvec_to_position_vec(X_test[i, hyper_params["num_cont"]:])
	
	if balanceGTLabelFlag:
		assert min(np.unique(true_labels, return_counts=True)[1]) == max(np.unique(true_labels, return_counts=True)[1]), "Ground truth labels not balanced."

	return X_test, true_labels if useRealLabels else rotated_labels

# This test distribution tests compositionality.
def get_test_splitB(test_size, test_dist):

	global cache

	X_01 = np.random.uniform(hyper_params["cont_range"][0], hyper_params["cont_range"][1], size=(test_size, hyper_params["num_cont"]))
	X_01, true_labels = true_g(X_01)

	X_02 = get_dist_bool_vecs(len(X_01), hyper_params["boolvec_dim"], hyper_params["rep_bools"], test_dist)
	X_test = np.concatenate((X_01, X_02), axis=1)
	rotated_labels = true_f(true_labels, X_test)

	if unrotationExperimentFlag:
		matching_indices = (true_labels == rotated_labels)
		true_labels = true_labels[matching_indices]
		rotated_labels = rotated_labels[matching_indices]
		X_test = X_test[matching_indices]
	
	X_test = standardize_data(X_test)

	# Testing: Remove this line.
	for i in range(len(X_test)):
		X_test[i, hyper_params["num_cont"]:] = convert_boolvec_to_position_vec(X_test[i, hyper_params["num_cont"]:])
	
	if balanceGTLabelFlag:
		assert min(np.unique(true_labels, return_counts=True)[1]) == max(np.unique(true_labels, return_counts=True)[1]), "Ground truth labels not balanced."

	return X_test, true_labels if useRealLabels else rotated_labels