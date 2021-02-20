import numpy as np
import torch

cache = {}
normalize_training_sep = False

# g : R^2 -> {0,1}
# Here, each x \in X are \in R^2
def true_g(X):
	return np.where(np.linalg.norm(X, axis=1) < 1, 0, 1)

# Given g, each x \in X are \in R^3
def true_f(g, X):
	g_result = g(X[:,:2])
	return (g_result) * (1 - X[:, 2]) + (1 - g_result) * X[:, 2]

# Normalize the data (subtract mean, divide by std)
def normalize_data(X_orig):
	X = np.copy(X_orig)

	X_mean = np.mean(X[:, :2], axis=0)
	X_std = np.std(X[:, :2], axis=0)

	X[:, :2] -= X_mean
	X[:, :2] /= X_std

	return X, X_mean, X_std

def unnormalize_data(X_orig, X_mean, X_std):
	X = np.copy(X_orig)

	X[:, :2] *= X_std
	X[:, :2] += X_mean

	return X

# Number of training points will be split_sizes[0] + split_sizes[1]
def get_train_data(split_sizes):
	n0 = split_sizes[0]
	n1 = split_sizes[1]

	X1_01 = np.random.uniform(0, 5, (n0, 2))
	X1_2 = np.zeros((n0, 1))
	X1 = np.concatenate((X1_01, X1_2), axis=1)
	y1 = true_f(true_g, X1)

	if normalize_training_sep:
		X1, X1_mean, X1_std, y1, y1_mean, y1_std = normalize_data(X1, y1)

	X2_01 = np.random.uniform(0, 1, (n1, 2))
	X2_2 = np.ones((n1, 1))
	X2 = np.concatenate((X2_01, X2_2), axis=1)
	y2 = true_f(true_g, X2)

	if normalize_training_sep:
		X2, X2_mean, X2_std, y2, y2_mean, y2_std = normalize_data(X2, y2)

	X_train = np.concatenate((X1, X2), axis=0)
	y_train = np.concatenate((y1, y2), axis=0)
	
	if normalize_training_sep:
		return X_train, y_train

	X, X_mean, X_std = normalize_data(X_train)
	cache["X_train_mean"] = X_mean
	cache["X_train_std"] = X_std

	return X, y_train

def get_test_splitA(test_size):
	X_01 = np.random.uniform(1, 5, (test_size, 2))
	X_02 = np.ones((test_size, 1))
	X_test = np.concatenate((X_01, X_02), axis=1)

	y_test = true_f(true_g, X_test)

	X, X_mean, X_std = normalize_data(X_test)
	cache["X_testA_mean"] = X_mean
	cache["X_testA_std"] = X_std
	# print(cache)

	return X, y_test

def get_test_splitB(split_sizes):
	n0 = split_sizes[0]
	n1 = split_sizes[1]

	X1_01 = np.random.uniform(0, 5, (n0, 2))
	X1_2 = np.zeros((n0, 1))
	X1 = np.concatenate((X1_01, X1_2), axis=1)

	X2_01 = np.random.uniform(0, 1, (n1, 2))
	X2_2 = np.ones((n1, 1))
	X2 = np.concatenate((X2_01, X2_2), axis=1)

	X_test = np.concatenate((X1, X2), axis=0)
	y_test = true_f(true_g, X_test)

	X, X_mean, X_std = normalize_data(X_test)
	cache["X_testB_mean"] = X_mean
	cache["X_testB_std"] = X_std

	return X, y_test

def get_test_splitC(test_size):
	X_01 = np.random.uniform(6, 9, (test_size, 2))
	X_02 = np.ones((test_size, 1))
	X_test = np.concatenate((X_01, X_02), axis=1)

	y_test = true_f(true_g, X_test)

	X, X_mean, X_std = normalize_data(X_test)
	cache["X_testC_mean"] = X_mean
	cache["X_testC_std"] = X_std

	return X, y_test