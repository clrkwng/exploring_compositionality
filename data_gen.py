import numpy as np
import torch

# g : R^2 -> R
# Here, each x \in X are \in R^2
def true_g(X):
	return torch.sum(torch.pow(X, 2), dim=1)
	# return np.where(np.linalg.norm(X, axis=1) < 4, 0, 1)

# Given g, each x \in X are \in R^3
def true_f(g, X):
	g_result = g(X[:,:2])
	return (500 + g_result) * (1 - X[:,2]) + -g_result * X[:,2]

# Number of training points will be split_sizes[0] + split_sizes[1]
def get_train_data(split_sizes):
	n0 = split_sizes[0]
	n1 = split_sizes[1]

	X1_01 = torch.rand((n0, 2)) * 5
	X1_2 = torch.zeros((n0, 1))
	X1 = torch.cat((X1_01, X1_2), dim=1)

	X2_01 = torch.rand((n1, 2)) * 2
	X2_2 = torch.ones((n1, 1))
	X2 = torch.cat((X2_01, X2_2), dim=1)

	X_train = torch.cat((X1, X2), dim=0)
	y_train = true_f(true_g, X_train)
	return X_train, y_train