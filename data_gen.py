import numpy as np
import torch

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
def normalize_data(X, y):
	X_mean = np.mean(X[:, :2], axis=0)
	X_std = np.std(X[:, :2], axis=0)
	y_mean = np.mean(y)
	y_std = np.std(y)

	X[:, :2] -= X_mean
	X[:, :2] /= X_std

	y -= y_mean
	y /= y_std

	return X, y

# Number of training points will be split_sizes[0] + split_sizes[1]
def get_train_data(split_sizes):
	n0 = split_sizes[0]
	n1 = split_sizes[1]

	X1_01 = np.random.uniform(0, 5, (n0, 2))
	X1_2 = np.zeros((n0, 1))
	X1 = np.concatenate((X1_01, X1_2), axis=1)

	X2_01 = np.random.uniform(0, 5, (n1, 2))
	X2_2 = np.ones((n1, 1))
	X2 = np.concatenate((X2_01, X2_2), axis=1)

	X_train = np.concatenate((X1, X2), axis=0)
	y_train = true_f(true_g, X_train)
	
	return normalize_data(X_train, y_train)