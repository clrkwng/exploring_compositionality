import numpy as np
import torch

cache = {}

# g : R^2 -> R
# Here, each x \in X are \in R^2
def true_g(X):
	return np.sum(np.power(X, 2), axis=1)
	# return np.where(np.linalg.norm(X, axis=1) < 4, 0, 1)

# Given g, each x \in X are \in R^3
def true_f(g, X):
	g_result = g(X[:,:2])
	return (0 + g_result) * (1 - X[:,2]) + -g_result * X[:,2]

# Normalize the data (subtract mean, divide by std)
def normalize_data(X_orig, y_orig):
	X = np.copy(X_orig)
	y = np.copy(y_orig)

	X_mean = np.mean(X[:, :2], axis=0)
	X_std = np.std(X[:, :2], axis=0)
	y_mean = np.mean(y)
	y_std = np.std(y)

	X[:, :2] -= X_mean
	X[:, :2] /= X_std

	y -= y_mean
	y /= y_std

	return X, X_mean, X_std, y, y_mean, y_std

def unnormalize_data(X_orig, y_orig, X_mean, X_std, y_mean, y_std):
	X = np.copy(X_orig)
	y = np.copy(y_orig)

	X[:, :2] *= X_std
	X[:, :2] += X_mean

	y *= y_std
	y += y_mean

	return X, y

# Number of training points will be split_sizes[0] + split_sizes[1]
def get_train_data(split_sizes):
	n0 = split_sizes[0]
	n1 = split_sizes[1]

	X1_01 = np.random.uniform(0, 5, (n0, 2))
	X1_2 = np.zeros((n0, 1))
	X1 = np.concatenate((X1_01, X1_2), axis=1)

	X2_01 = np.random.uniform(0, 1, (n1, 2))
	X2_2 = np.ones((n1, 1))
	X2 = np.concatenate((X2_01, X2_2), axis=1)

	X_train = np.concatenate((X1, X2), axis=0)
	y_train = true_f(true_g, X_train)

	X, X_mean, X_std, y, y_mean, y_std = normalize_data(X_train, y_train)
	cache["X_train_mean"] = X_mean
	cache["X_train_std"] = X_std
	cache["y_train_mean"] = y_mean
	cache["y_train_std"] = y_std

	return X, y

def get_test_splitA(test_size):
	X_01 = np.random.uniform(1, 5, (test_size, 2))
	X_02 = np.ones((test_size, 1))
	X_test = np.concatenate((X_01, X_02), axis=1)

	y_test = true_f(true_g, X_test)

	X, X_mean, X_std, y, y_mean, y_std = normalize_data(X_test, y_test)
	cache["X_testA_mean"] = X_mean
	cache["X_testA_std"] = X_std
	cache["y_testA_mean"] = y_mean
	cache["y_testA_std"] = y_std

	return X, y

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

	X, X_mean, X_std, y, y_mean, y_std = normalize_data(X_test, y_test)
	cache["X_testB_mean"] = X_mean
	cache["X_testB_std"] = X_std
	cache["y_testB_mean"] = y_mean
	cache["y_testB_std"] = y_std

	return X, y

def get_test_splitC(test_size):
	X_01 = np.random.uniform(50, 100, (test_size, 2))
	X_02 = np.zeros((test_size, 1))
	X_test = np.concatenate((X_01, X_02), axis=1)

	y_test = true_f(true_g, X_test)

	X, X_mean, X_std, y, y_mean, y_std = normalize_data(X_test, y_test)
	cache["X_testC_mean"] = X_mean
	cache["X_testC_std"] = X_std
	cache["y_testC_mean"] = y_mean
	cache["y_testC_std"] = y_std

	return X, y