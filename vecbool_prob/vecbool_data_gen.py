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

bools_file = Path("../pickled_files/rep_bools.pickle")
rep_bools = load_pickle("../pickled_files/rep_bools.pickle") if bools_file.is_file() else gen_bools()

hyper_params = {
	"cont_range": [0, 5],
	"boolvec_dim": boolvec_dim, # boolvec_dim is defined in gen_rep_bools.py.
	"emb_dims": [2 * boolvec_dim, 4 * boolvec_dim],
	"num_cont": 2,
	"lin_layer_sizes": [680, 2048, 800, 512, 128, 32, 20],
	"num_classes": 10,
	"hidden_drop_p": 0.1,
	"batch_flag": True,
	"rep_bools": rep_bools,
	"lr": 1e-3,
	"num_epochs": 100,
	"batch_size": 256
}

with open('../ssh_keys/comet_api_key.txt', 'r') as file:
	comet_key = file.read().replace('\n', '')

experiment = Experiment(api_key=comet_key,
												project_name="vecbool",
												workspace="clrkwng")
experiment.log_parameters(hyper_params)

# Flag denotes whether to use true labels, or labels after mod rotation.
useRealLabels = False

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
def standardize_data(X_orig, train_mode=False):
	global cache

	X = np.copy(X_orig)
	X_mean = np.mean(X[:, :hyper_params["num_cont"]], axis=0)
	X_std = np.std(X[:, :hyper_params["num_cont"]], axis=0)

	if train_mode:
		X[:, :hyper_params["num_cont"]] -= X_mean
		X[:, :hyper_params["num_cont"]] /= X_std

	else:
		assert "X_train_mean" in cache and "X_train_std" in cache,\
			"Train data statistics have not been cached yet."
			
		X[:, :hyper_params["num_cont"]] -= cache["X_train_mean"]
		X[:, :hyper_params["num_cont"]] /= cache["X_train_std"]

	return X, X_mean, X_std

# Return unstandardized data, used for plotting.
def unstandardize_data(X_train, X_mean, X_std):
	X = np.copy(X_train)
	X[:, :hyper_params["num_cont"]] *= X_std
	X[:, :hyper_params["num_cont"]] += X_mean

	return X

# g : R^2 -> {0, 1, ..., num_classes - 1}.
# Here, each x in X is in R^2.
def true_g(X):
	X_sum = np.sum(X, axis=1)

	# Percentage gets what percentage the sum of x1, x2 is of the the possible continuous sum.
	percentage = (X_sum - (2 * hyper_params["cont_range"][0])) / (2 * (hyper_params["cont_range"][1] - hyper_params["cont_range"][0]))
	return np.floor(percentage * hyper_params["num_classes"])

# Given g, and each x in X has continuous and categorical data.
def true_f(g, X):
	true_labels = g(X[:, :hyper_params["num_cont"]])
	rot_amts = get_rotation_amount(X[:, hyper_params["num_cont"]:].T)
	rotated_labels = rotate_class(true_labels, rot_amts, hyper_params["num_classes"])
	return true_labels, rotated_labels

# Returns the number of correct predictions.
def get_num_correct(preds, labels):
	pred = preds.max(1, keepdim=True)[1]
	correct = pred.eq(labels.view_as(pred)).sum().item()
	return correct

def get_train_data(train_size):
	global cache
	global rep_bools

	X_01 = np.random.uniform(hyper_params["cont_range"][0], hyper_params["cont_range"][1], size=(train_size, hyper_params["num_cont"]))
	X_02 = get_rep_bool_vecs(train_size, hyper_params["boolvec_dim"], hyper_params["rep_bools"])
	X = np.concatenate((X_01, X_02), axis=1)
	true_labels, rotated_labels = true_f(true_g, X)

	X_train, X_train_mean, X_train_std = standardize_data(X, train_mode=True)
	cache["X_train_mean"] = X_train_mean
	cache["X_train_std"] = X_train_std

	# Testing: Remove this line.
	for i in range(len(X_train)):
		X_train[i, hyper_params["num_cont"]:] = convert_boolvec_to_position_vec(X_train[i, hyper_params["num_cont"]:])

	return X_train, true_labels if useRealLabels else rotated_labels

# This test distribution is the same as the training distribution.
def get_test_splitA(test_size, *unused):
	global cache

	X_01 = np.random.uniform(hyper_params["cont_range"][0], hyper_params["cont_range"][1], size=(test_size, hyper_params["num_cont"]))
	X_02 = get_rep_bool_vecs(test_size, hyper_params["boolvec_dim"], hyper_params["rep_bools"])
	X = np.concatenate((X_01, X_02), axis=1)
	true_labels, rotated_labels = true_f(true_g, X)

	X_test, X_test_mean, X_test_std = standardize_data(X)
	cache["X_testA_mean"] = X_test_mean
	cache["X_testA_std"] = X_test_std

	# Testing: Remove this line.
	for i in range(len(X_test)):
		X_test[i, hyper_params["num_cont"]:] = convert_boolvec_to_position_vec(X_test[i, hyper_params["num_cont"]:])

	return X_test, true_labels if useRealLabels else rotated_labels

# This test distribution tests compositionality.
def get_test_splitB(test_size, test_dist):
	global cache

	X_01 = np.random.uniform(hyper_params["cont_range"][0], hyper_params["cont_range"][1], size=(test_size, hyper_params["num_cont"]))
	X_02 = get_dist_bool_vecs(test_size, hyper_params["boolvec_dim"], hyper_params["rep_bools"], test_dist)
	X = np.concatenate((X_01, X_02), axis=1)
	true_labels, rotated_labels = true_f(true_g, X)

	X_test, X_test_mean, X_test_std = standardize_data(X)
	cache["X_testB_mean"] = X_test_mean
	cache["X_testB_std"] = X_test_std

	# Testing: Remove this line.
	for i in range(len(X_test)):
		X_test[i, hyper_params["num_cont"]:] = convert_boolvec_to_position_vec(X_test[i, hyper_params["num_cont"]:])

	return X_test, true_labels if useRealLabels else rotated_labels
