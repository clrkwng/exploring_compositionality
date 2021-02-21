import numpy as np
import torch

# Global Cache that stores the statistics of train set and each test set.
cache = {}

# g : R^2 -> {0,1}.
# Here, each x in X is in R^2.
def true_g(X):
	return np.where(np.sum(X, axis=1) < 1, 0, 1)

# Given g, and each x in X is in R^3.
def true_f(g, X):
	g_result = g(X[:,:2])
	return (g_result) * (1 - X[:, 2]) + (1 - g_result) * X[:, 2]

# Standardize the data (subtract mean, divide by std).
# Will use training data's mean and std for both train and test data.
def standardize_data(X_orig, train_mode=False):
	global cache

	X = np.copy(X_orig)

	if train_mode:
		X_mean = np.mean(X[:, :2], axis=0)
		X_std = np.std(X[:, :2], axis=0)
		X[:, :2] -= X_mean
		X[:, :2] /= X_std

	else:
		assert "X_train_mean" in cache and "X_train_std" in cache,\
			"Train data statistics have not been cached yet."

		X_mean = np.mean(X[:, :2], axis=0)
		X_std = np.std(X[:, :2], axis=0)
		X[:, :2] -= cache["X_train_mean"]
		X[:, :2] /= cache["X_train_std"]

	return X, X_mean, X_std

# Return unstandardized data, used for plotting.
def unstandardize_data(X_orig, X_mean, X_std):
	X = np.copy(X_orig)

	X[:, :2] *= X_std
	X[:, :2] += X_mean

	return X

# Number of training points will be split_sizes[0] + split_sizes[1].
def get_train_data(split_sizes):
	global cache

	n0 = split_sizes[0]
	n1 = split_sizes[1]

	X1_01 = np.random.uniform(0, 5, (n0, 2))
	X1_2 = np.zeros((n0, 1))
	X1 = np.concatenate((X1_01, X1_2), axis=1)
	y1 = true_f(true_g, X1)

	X2_01 = np.random.uniform(0, 1, (n1, 2))
	X2_2 = np.ones((n1, 1))
	X2 = np.concatenate((X2_01, X2_2), axis=1)
	y2 = true_f(true_g, X2)

	X_train = np.concatenate((X1, X2), axis=0)
	y_train = np.concatenate((y1, y2), axis=0)

	X, X_mean, X_std = standardize_data(X_train, train_mode=True)
	cache["X_train_mean"] = X_mean
	cache["X_train_std"] = X_std

	return X, y_train
	
# This is a test of compositionality (on data that model hasn't been trained on).
def get_test_splitA(test_size):
	global cache

	X_01 = np.random.uniform(1, 5, (test_size, 2))
	X_02 = np.zeros((test_size, 1))
	X_test = np.concatenate((X_01, X_02), axis=1)

	y_test = true_f(true_g, X_test)

	X, X_mean, X_std = standardize_data(X_test)
	cache["X_testA_mean"] = X_mean
	cache["X_testA_std"] = X_std

	return X, y_test

# This test matches training data distribution.
def get_test_splitB(split_sizes):
	global cache

	n0 = split_sizes[0]
	n1 = split_sizes[1]

	X1_01 = np.random.uniform(0, 5, (n0, 2))
	X1_02 = np.zeros((n0, 1))
	X1 = np.concatenate((X1_01, X1_02), axis=1)

	X2_01 = np.random.uniform(0, 1, (n1, 2))
	X2_02 = np.ones((n1, 1))
	X2 = np.concatenate((X2_01, X2_02), axis=1)

	X_test = np.concatenate((X1, X2), axis=0)
	y_test = true_f(true_g, X_test)

	X, X_mean, X_std = standardize_data(X_test)
	cache["X_testB_mean"] = X_mean
	cache["X_testB_std"] = X_std

	return X, y_test

# This test looks at x in (0,1), which is overlapped region for boolean = 1 and boolean = 0.
def get_test_splitC(test_size):
	global cache

	X_01 = np.random.uniform(0, 1, (test_size, 2))
	X_02 = np.zeros((test_size, 1))
	X_test = np.concatenate((X_01, X_02), axis=1)

	y_test = true_f(true_g, X_test)

	X, X_mean, X_std = standardize_data(X_test)
	cache["X_testC_mean"] = X_mean
	cache["X_testC_std"] = X_std

	return X, y_test