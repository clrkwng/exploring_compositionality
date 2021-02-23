import numpy as np

# Global cache that stores the statistics of train set and each test set.
cache = {}

num_classes = 10
boolvec_dim = 4
cont_range = [0, 5]

# g : R^2 -> {0, 1, ..., num_classes - 1}.
# Here, each x in X is in R^2.
def true_g(X):
	X_sum = np.sum(X, axis=1)
	percentage = (X_sum - (2 * cont_range[0]))/(2 * cont_range[1])
	return np.floor(percentage * num_classes)

# Given g, and each x in X is in R^3.
def true_f(g, X):
	g_result = g(X[:, :2])
	rot_amts = get_rotation_amount(X[:, 2:].T)
	g_result = rotate_class(g_result, rot_amts)
	return g_result

# Returns a weighted sum of value times one-indexed index val.
def get_rotation_amount(bool_vec):
	amt = 0
	for i, val in enumerate(bool_vec, start=1):
		amt += i * val
	return amt

# Given classes and rotation amount, rotate class under mod.
def rotate_class(classes, rot_amts):
	return (classes + rot_amts) % num_classes

def get_train_data(train_size):
	X_01 = np.random.uniform(cont_range[0], cont_range[1], size=(train_size, 2))
	X_02 = np.random.randint(2, size=(train_size, boolvec_dim))
	X = np.concatenate((X_01, X_02), axis=1)
	y = true_f(true_g, X)

	return X, y
