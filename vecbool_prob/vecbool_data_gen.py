import numpy as np
import sys
from bool_utils import *
sys.path[0] = '../pickled_files/'
from pickle_logic import *
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path

my_file = Path("../pickled_files/stats_cache.pickle")
# Global cache that stores the statistics of train set and each test set.
cache = load_pickle("../pickled_files/stats_cache.pickle") if my_file.is_file() else {}

# Define the model parameters here.
cont_range = [0, 5]

boolvec_dim = 6
emb_dims = [2 * boolvec_dim, 4 * boolvec_dim]
num_cont = 2
lin_layer_sizes = [512, 256, 128, 64, 32, 20]
num_classes = 10
hidden_drop_p = 0.1
batch_flag = True

rep_bools = get_representative_bools(boolvec_dim)

# Flag denotes whether to use true labels, or labels after mod rotation.
useRealLabels = False

def save_plot(xvalues, yvalues, xlabel, ylabel, title, file_name):
	if len(yvalues) == 1:
		plt.plot(xvalues, yvalues[0])
	else:
		plt.plot(xvalues, yvalues[0], label="Train Accuracy")
		plt.plot(xvalues, yvalues[1], label="Validation Accuracy")
		plt.legend(loc="upper left")

	plt.suptitle(title, fontsize=18)
	plt.xlabel(xlabel, fontsize=14)
	plt.ylabel(ylabel, fontsize=14)
	plt.savefig(file_name)
	plt.clf()

# Standardize the data (subtract mean, divide by std).
# Will use training data's mean and std for both train and test data.
def standardize_data(X_orig, train_mode=False):
	global cache

	X = np.copy(X_orig)
	X_mean = np.mean(X[:, :2], axis=0)
	X_std = np.std(X[:, :2], axis=0)

	if train_mode:
		X[:, :2] -= X_mean
		X[:, :2] /= X_std

	else:
		assert "X_train_mean" in cache and "X_train_std" in cache,\
			"Train data statistics have not been cached yet."
			
		X[:, :2] -= cache["X_train_mean"]
		X[:, :2] /= cache["X_train_std"]

	return X, X_mean, X_std

# Return unstandardized data, used for plotting.
def unstandardize_data(X_orig, X_mean, X_std):
	global cache

	X = np.copy(X_orig)
	X[:, :2] *= X_std
	X[:, :2] += X_mean

	return X

# g : R^2 -> {0, 1, ..., num_classes - 1}.
# Here, each x in X is in R^2.
def true_g(X):
	X_sum = np.sum(X, axis=1)

	# Percentage gets what percentage the sum of x1, x2 is of the the possible continuous sum.
	percentage = (X_sum - (2 * cont_range[0])) / (2 * (cont_range[1] - cont_range[0]))
	return np.floor(percentage * num_classes)

# Given g, and each x in X is in R^3.
def true_f(g, X):
	true_labels = g(X[:, :2])
	rot_amts = get_rotation_amount(X[:, 2:].T)
	rotated_labels = rotate_class(true_labels, rot_amts, num_classes)
	return true_labels, rotated_labels

# Returns the number of correct predictions.
def get_num_correct(preds, labels):
	pred = preds.max(1, keepdim=True)[1]
	correct = pred.eq(labels.view_as(pred)).sum().item()
	return correct

def get_train_data(train_size):
	global cache

	X_01 = np.random.uniform(cont_range[0], cont_range[1], size=(train_size, 2))
	X_02 = get_rep_bool_vecs(train_size, boolvec_dim, rep_bools)
	X = np.concatenate((X_01, X_02), axis=1)
	true_labels, rotated_labels = true_f(true_g, X)

	X_train, X_train_mean, X_train_std = standardize_data(X, train_mode=True)
	cache["X_train_mean"] = X_train_mean
	cache["X_train_std"] = X_train_std

	return X_train, true_labels if useRealLabels else rotated_labels

# This test distribution is the same as the training distribution.
def get_test_splitA(test_size):
	global cache

	X_01 = np.random.uniform(cont_range[0], cont_range[1], size=(test_size, 2))
	X_02 = get_rand_bool_vecs(test_size, boolvec_dim)
	X = np.concatenate((X_01, X_02), axis=1)
	true_labels, rotated_labels = true_f(true_g, X)

	X_test, X_test_mean, X_test_std = standardize_data(X)
	cache["X_testA_mean"] = X_test_mean
	cache["X_testA_std"] = X_test_std

	return X_test, true_labels if useRealLabels else rotated_labels

test_dist = 5
# This test distribution tests compositionality.
def get_test_splitB(test_size):
	global cache

	X_01 = np.random.uniform(cont_range[0], cont_range[1], size=(test_size, 2))
	X_02 = get_dist_bool_vecs(test_size, boolvec_dim, rep_bools, test_dist)
	X = np.concatenate((X_01, X_02), axis=1)
	true_labels, rotated_labels = true_f(true_g, X)

	X_test, X_test_mean, X_test_std = standardize_data(X)
	cache["X_testB_mean"] = X_test_mean
	cache["X_testB_std"] = X_test_std

	return X_test, true_labels if useRealLabels else rotated_labels