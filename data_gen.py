import numpy as np

# g : R^2 -> R
# Here, each x \in X are \in R^2
def true_g(X):
	return np.sum(X, axis=1)
	# return np.where(np.linalg.norm(X, axis=1) < 4, 0, 1)

# Given g, each x \in X are \in R^3
def true_f(g, X):
	g_result = g(X[:,:2])
	return g_result * (1 - X[:,2]) + (-1 * g_result) * X[:,2]

# Number of training points will be split_sizes[0] + split_sizes[1]
def get_train_data(split_sizes):
	n0 = split_sizes[0]
	n1 = split_sizes[1]

	X1_01 = np.around(np.random.uniform(0, 5, size=(n0,2)), 6)
	X1_2 = np.zeros((n0, 1))
	X1 = np.hstack((X1_01, X1_2))

	X2_01 = np.around(np.random.uniform(0, 5, size=(n1,2)), 6)
	X2_2 = np.ones((n1, 1))
	X2 = np.hstack((X2_01, X2_2))

	X_train = np.vstack((X1, X2))
	y_train = true_f(true_g, X_train)
	return X_train, y_train